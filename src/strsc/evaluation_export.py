"""Export completed evaluations for the browser."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

SCHEMA = "strsc-evaluation-bundle-v1"


def finite_json(value):
    if isinstance(value, dict):
        return {str(key): finite_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [finite_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _scalar(value):
    if value is None or value == "":
        return None
    if value in ("True", "False", "true", "false"):
        return str(value).lower() == "true"
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return value


def _table(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        columns = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    if not columns or not rows:
        raise ValueError(f"No result rows in {path.name}")
    return columns, rows


def _number(value):
    if value is None or value == "":
        return None
    result = float(value)
    return result if math.isfinite(result) else None


class Bundle:
    def __init__(self, workflow, title, synthetic):
        self.data = dict(schema=SCHEMA, title=title, workflow=workflow,
                         synthetic=synthetic, metrics=[], rows=[], sources=[], notes=[])
        self.identities = set()
        self.metric_ids = set()

    def metric(self, key, label, description, unit="score", object="prediction"):
        if key not in self.metric_ids:
            self.metric_ids.add(key)
            self.data["metrics"].append(dict(id=key, label=label, description=description,
                                              unit=unit, higher_is_better=True, object=object))

    def row(self, model, task, unit, unit_type, metric, value, support,
            status=None, details=None, role="prediction"):
        identity = (str(model), str(task), str(unit), metric)
        if identity in self.identities:
            raise ValueError(f"Duplicate result identity: {' / '.join(identity)}")
        self.identities.add(identity)
        value = _number(value)
        self.data["rows"].append(dict(model=str(model), task=str(task), unit_id=str(unit),
                                     unit_type=unit_type, metric=metric, value=value,
                                     status=status or ("ESTIMABLE" if value is not None else "NOT_ESTIMABLE"),
                                     support=support, details=finite_json(details or {}), model_role=role))

    def table(self, path, index):
        columns, rows = _table(path)
        self.data["sources"].append(dict(name=f"run{index}/{path.name}", columns=columns,
                                         rows=[{key: _scalar(value) for key, value in row.items()} for row in rows]))
        return rows


def detect_workflow(path):
    if path.is_dir():
        candidates = []
        if (path / "analysis.json").is_file():
            record = json.loads((path / "analysis.json").read_text(encoding="utf-8-sig"))
            if record.get("protocol") == "HEST_G0047B":
                candidates.append("hest")
        if (path / "patient_scores.tsv").is_file():
            candidates.append("her2")
        if list(path.glob("*_SPECIMEN_BAND_RSC.tsv")) or list(path.glob("*_RNA_SUPPORT_SPECIMEN_BAND.tsv")):
            candidates.append("cell")
        if not candidates:
            for table in path.glob("*.tsv"):
                with table.open(encoding="utf-8-sig") as handle:
                    columns = handle.readline().strip().split("\t")
                if {"wavelength_band", "v_rna", "rsc"}.issubset(columns):
                    candidates.append("fourier")
                    break
        if len(candidates) != 1:
            raise ValueError(f"Could not identify one workflow in {path.name}. Use a completed result directory.")
        return candidates[0]
    columns, _ = _table(path)
    if {"wavelength_band", "v_rna", "rsc"}.issubset(columns):
        return "fourier"
    if {"model_id", "patient_id", "normalized_pairwise_accuracy"}.issubset(columns):
        return "her2"
    if {"specimen", "view", "band", "rsc"}.issubset(columns) or {"specimen", "view", "band", "rna_split_agreement"}.issubset(columns):
        return "cell"
    raise ValueError(f"Unrecognized score table: {path.name}")


def _hest(bundle, path, index, unit):
    record = json.loads((path / "analysis.json").read_text(encoding="utf-8-sig"))
    if record.get("status") != "COMPLETE" or record.get("protocol") != "HEST_G0047B":
        raise ValueError("HEST requires a completed HEST_G0047B analysis.json")
    if unit and unit != record["specimen"]:
        raise ValueError("--unit must match the saved HEST specimen")
    details = {key: record[key] for key in ("protocol", "graph_retained_spots", "eligible_genes", "thinning_seeds") if key in record}
    bundle.metric("pearson_graph_support", "Whole-section Pearson r", "Mean gene-wise Pearson on the connected spots.", "correlation")
    for metric, label, description in (
        ("rsc_broad", "Broad-scale RSC", "RSC calculated from combined B1+B2 covariance and variance components."),
        ("rsc_fine", "Fine-scale RSC", "RSC calculated from combined B4+B5 covariance and variance components."),
    ):
        bundle.metric(metric, label, description)
    for row in record["models"]:
        for field, metric in (("ordinary_PCC_graph_support", "pearson_graph_support"), ("coarse", "rsc_broad"), ("fine", "rsc_fine")):
            bundle.row(row["model"], record["task"], record["specimen"], "specimen", metric,
                       row[field], "connected spots", row.get(field + "_status"), details)
    for row in record["band_scores"]:
        key = f"rsc_{row['band']}"
        bundle.metric(key, f"RSC, {row['band']}", "Relative scale of the neighbor graph, from B1 (broadest) to B5 (finest).")
        bundle.row(row["model"], record["task"], record["specimen"], "specimen", key,
                   row["rsc"], "connected spots", row.get("rsc_status"), {**details, **row})
    for name in ("specimen_scores.tsv", "band_scores.tsv", "gene_scores.tsv"):
        if (path / name).is_file():
            bundle.table(path / name, index)
    bundle.data["notes"] = ["Saved HEST scores are reproducible-signal correlation (RSC).",
                            "Model identifiers are the prediction names in the HEST manifest."]


def _her2(bundle, path, index, unit):
    file = path / "patient_scores.tsv" if path.is_dir() else path
    rows = bundle.table(file, index)
    specs = (
        ("model_accuracy", "Pairwise accuracy of models", "Fraction of spot pairs for which the prediction identifies which spot expresses a gene more highly in one RNA half. The pairs are chosen in the other half.", "accuracy", "prediction"),
        ("normalized_pairwise_accuracy", "Normalized pairwise accuracy", "(Model accuracy minus majority-order baseline accuracy) divided by (split-half accuracy minus majority-order baseline accuracy).", "score", "prediction"),
        ("lopo_gene_majority_baseline", "Majority-order baseline", "Accuracy of a baseline that gives each spot pair of a gene the direction observed most often across the training patients.", "accuracy", "baseline"),
        ("rna_split_half_accuracy", "Split-half agreement", "Accuracy with which one RNA half predicts the order in the other half, on the same spot pairs.", "accuracy", "split_half"),
    )
    for key, label, definition, measure, role in specs:
        bundle.metric(key, label, definition, measure, role)
    references = set()
    for row in rows:
        if unit and unit != row["patient_id"]:
            raise ValueError("--unit must match every patient in the HER2 table. Usually omit it.")
        task = row.get("task") or row["arena"]
        details = {key: _scalar(value) for key, value in row.items()}
        for key, _, _, _, role in specs:
            model = row["model_id"] if role == "prediction" else "Split-half agreement" if role == "split_half" else "Majority-order baseline"
            identity = (model, task, row["patient_id"], key)
            if role != "prediction" and identity in references:
                continue
            references.add(identity)
            status = row.get("status") if key == "normalized_pairwise_accuracy" else None
            bundle.row(model, task, row["patient_id"], "patient", key, row[key],
                       "spot pairs chosen in one RNA half, where every model has a prediction", status, details, role)
    if path.is_dir() and (path / "question_support.tsv").is_file():
        bundle.table(path / "question_support.tsv", index)
    bundle.data["notes"] = ["Patient scores are means over count splits.",
                            "Split-half agreement and the majority-order baseline are references."]


def _fourier(bundle, path, index, unit):
    files = [path] if path.is_file() else [file for file in path.glob("*.tsv") if detect_file_fourier(file)]
    if len(files) != 1:
        raise ValueError("Select one Fourier TSV per --run and give each an explicit --unit specimen ID")
    rows = bundle.table(files[0], index)
    if not unit:
        raise ValueError("Fourier results do not store specimen IDs. Provide --unit SPECIMEN for each --run.")
    bundle.metric("physical_rsc", "RSC by physical scale", "Saved covariance divided by the square root of the split-half covariance times the prediction variance, for this wavelength window.")
    for row in rows:
        spacing = row["scale_um"]
        task = f"{row['wavelength_band']} | grid {spacing} um"
        bundle.row(row["model"], task, unit, "specimen", "physical_rsc", row["rsc"],
                   f"occupied regular-grid sites; {row['nyquist_status']}", row.get("status"),
                   {key: _scalar(value) for key, value in row.items()})
    bundle.data["notes"] = ["Wavelengths are full spatial cycles. Windows that the grid resolves in part are labeled.",
                            "ALL_NON_DC is RSC calculated from the combined covariance and variance components of all windows."]


def detect_file_fourier(path):
    with path.open(encoding="utf-8-sig") as handle:
        return {"wavelength_band", "v_rna", "rsc"}.issubset(handle.readline().strip().split("\t"))


def _cell(bundle, path, index, unit):
    files = [path] if path.is_file() else sorted([*path.glob("*_SPECIMEN_BAND_RSC.tsv"), *path.glob("*_RNA_SUPPORT_SPECIMEN_BAND.tsv"), *path.glob("*_PROLIFERATION_HOTSPOTS.tsv")])
    if not files:
        raise ValueError("No final specimen-level cell result tables found")
    bundle.metric("cell_rsc", "Cell-level RSC", "Saved RSC for the specified graph view and lazy-diffusion band.")
    bundle.metric("rna_split_agreement", "Split-half reliability", "Saved correlation between the two RNA halves on the specified cell graph view.", "correlation", "split_half")
    hotspot_specs = (
        ("E_model_normalized_log_score", "Prediction-defined hotspot enrichment", "RNA normalized-log score difference between prediction-defined hotspots and remaining eligible neoplastic neighborhoods.", "prediction"),
        ("E_RNA_normalized_log_score", "RNA-defined hotspot enrichment", "RNA normalized-log score difference between hotspots defined in one RNA half and remaining neighborhoods, measured in the other half.", "split_half"),
    )
    for file in files:
        rows = bundle.table(file, index)
        for row in rows:
            if unit and unit != row["specimen"]:
                raise ValueError("--unit must match saved cell specimen IDs")
            details = {key: _scalar(value) for key, value in row.items()}
            track = row.get("track") or "xenium_centred"
            if "view" in row and "band" in row:
                task = f"{track} | {row['view']} | {row['band']}"
                model = row.get("model") or "Split-half reliability"
                if "rsc" in row:
                    bundle.row(model, task, row["specimen"], "specimen", "cell_rsc", row["rsc"],
                               "cells with a prediction from this model", details=details)
                else:
                    bundle.row("Split-half reliability", task, row["specimen"], "specimen", "rna_split_agreement", row["rna_split_agreement"],
                               "all cells of the view", details=details, role="split_half")
            elif "E_model_normalized_log_score" in row:
                for key, label, definition, role in hotspot_specs:
                    bundle.metric(key, label, definition, "score", role)
                    # The split-half reference can depend on the model's retained support; retain that context.
                    task = f"{track} | within-Neoplastic proliferation | support: {row['model']}"
                    model = row["model"] if role == "prediction" else "RNA-defined hotspots"
                    bundle.row(model, task, row["specimen"], "specimen", key, row[key],
                               "saved eligible neoplastic neighborhoods", details=details, role=role)
    used = {row["metric"] for row in bundle.data["rows"]}
    bundle.data["metrics"] = [metric for metric in bundle.data["metrics"] if metric["id"] in used]
    bundle.data["notes"] = ["Units are specimens, and patient IDs are in the row details.",
                            "Each model is scored on the cells that it predicts, which can differ between models."]


def export_results(runs, output, workflow="auto", units=None, title=None, synthetic=False):
    runs = [Path(path) for path in runs]
    output = Path(output)
    if output.exists():
        raise ValueError("Output exists. Choose a new JSON filename.")
    if not runs:
        raise ValueError("At least one --run is required")
    if units and len(units) != len(runs):
        raise ValueError("Supply one --unit per --run, in the same order")
    units = units or [None] * len(runs)
    identified = [detect_workflow(path) for path in runs]
    if len(set(identified)) != 1 or (workflow != "auto" and any(item != workflow for item in identified)):
        raise ValueError("All inputs must belong to the same selected workflow")
    workflow = identified[0]
    bundle = Bundle(workflow, title or f"{'Synthetic ' if synthetic else ''}{workflow.upper()} evaluation", synthetic)
    exporter = {"hest": _hest, "her2": _her2, "fourier": _fourier, "cell": _cell}[workflow]
    for index, (path, unit) in enumerate(zip(runs, units), 1):
        exporter(bundle, path, index, unit)
    if not bundle.data["rows"]:
        raise ValueError("No exportable result rows")
    if synthetic:
        bundle.data["notes"].insert(0, "The data are synthetic.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(finite_json(bundle.data), handle, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        handle.write("\n")
    return bundle.data


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow", choices=["auto", "hest", "her2", "fourier", "cell"], default="auto")
    parser.add_argument("--run", type=Path, action="append", required=True, help="Completed result directory or supported TSV. Repeat for multiple units.")
    parser.add_argument("--unit", action="append", help="One specimen ID per --run, required for Fourier tables")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title")
    parser.add_argument("--synthetic", action="store_true", help="Label generated inputs as synthetic. Never label real data this way.")
    args = parser.parse_args(argv)
    try:
        bundle = export_results(args.run, args.output, args.workflow, args.unit, args.title, args.synthetic)
    except (ValueError, KeyError, OSError) as error:
        parser.exit(2, f"Export stopped: {error}\n")
    print(json.dumps(dict(output=str(args.output), schema=SCHEMA, workflow=bundle["workflow"], rows=len(bundle["rows"]), spatial_scoring_rerun=False)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
