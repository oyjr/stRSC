"""Export completed evaluations for the browser without rerunning their scores."""
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
            raise ValueError(f"Could not identify one workflow in {path.name}; use a completed result directory.")
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
    bundle.metric("pearson_graph_support", "Overall Pearson", "Mean gene-wise Pearson on the saved graph-retained locations.", "correlation")
    for metric, label, description in (
        ("rsc_broad", "Broad-scale RSC", "Combined B1+B2 covariance and variance components; not the mean of band scores."),
        ("rsc_fine", "Fine-scale RSC", "Combined B4+B5 covariance and variance components; not the training endpoint's mean of B4 and B5."),
    ):
        bundle.metric(metric, label, description)
    for row in record["models"]:
        for field, metric in (("ordinary_PCC_graph_support", "pearson_graph_support"), ("coarse", "rsc_broad"), ("fine", "rsc_fine")):
            bundle.row(row["model"], record["task"], record["specimen"], "specimen", metric,
                       row[field], "graph-retained locations", row.get(field + "_status"), details)
    for row in record["band_scores"]:
        key = f"rsc_{row['band']}"
        bundle.metric(key, f"RSC, {row['band']}", "Saved relative graph-frequency band; B1 broadest, B5 finest. These are not micrometre intervals.")
        bundle.row(row["model"], record["task"], record["specimen"], "specimen", key,
                   row["rsc"], "graph-retained locations", row.get("rsc_status"), {**details, **row})
    for name in ("specimen_scores.tsv", "band_scores.tsv", "gene_scores.tsv"):
        if (path / name).is_file():
            bundle.table(path / name, index)
    bundle.data["notes"] = ["Saved HEST scores are the reproducible-signal correlation (RSC). Exporting does not rerun them.",
                            "Model identifiers are the prediction names in the HEST manifest. A single prediction is valid."]


def _her2(bundle, path, index, unit):
    file = path / "patient_scores.tsv" if path.is_dir() else path
    rows = bundle.table(file, index)
    specs = (
        ("model_accuracy", "Model ordering accuracy", "Saved answer accuracy on source-defined, judge-half RNA questions.", "accuracy", "prediction"),
        ("normalized_pairwise_accuracy", "Normalized pairwise accuracy", "(Model accuracy minus held-out gene-majority accuracy) divided by (split-half accuracy minus held-out gene-majority accuracy).", "score", "prediction"),
        ("lopo_gene_majority_baseline", "Majority-order baseline", "Gene-majority answer accuracy fitted using other patients.", "accuracy", "baseline"),
        ("rna_split_half_accuracy", "Split-half agreement", "Accuracy with which one RNA half answers the same source-defined questions on the other half.", "accuracy", "split_half"),
    )
    for key, label, definition, measure, role in specs:
        bundle.metric(key, label, definition, measure, role)
    references = set()
    for row in rows:
        if unit and unit != row["patient_id"]:
            raise ValueError("--unit must match every patient in the HER2 table; usually omit it")
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
                       "source-defined, judge-half questions on common model support", status, details, role)
    if path.is_dir() and (path / "question_support.tsv").is_file():
        bundle.table(path / "question_support.tsv", index)
    bundle.data["notes"] = ["Patient scores average the saved molecule-split seeds. The normalized increment is not ordinary correlation or all-pairs concordance.",
                            "Split-half and held-out baseline metrics describe references, not competing prediction models."]


def _fourier(bundle, path, index, unit):
    files = [path] if path.is_file() else [file for file in path.glob("*.tsv") if detect_file_fourier(file)]
    if len(files) != 1:
        raise ValueError("Select one Fourier TSV per --run and give each an explicit --unit specimen ID")
    rows = bundle.table(files[0], index)
    if not unit:
        raise ValueError("Fourier results do not store specimen IDs; provide --unit SPECIMEN for each --run")
    bundle.metric("physical_rsc", "RSC by physical scale", "Saved covariance divided by the square root of the split-half covariance times the prediction variance, for this wavelength window.")
    for row in rows:
        spacing = row["scale_um"]
        task = f"{row['wavelength_band']} | grid {spacing} um"
        bundle.row(row["model"], task, unit, "specimen", "physical_rsc", row["rsc"],
                   f"occupied regular-grid sites; {row['nyquist_status']}", row.get("status"),
                   {key: _scalar(value) for key, value in row.items()})
    bundle.data["notes"] = ["Physical wavelength is a full spatial cycle, not grid spacing. Partial high-frequency windows remain explicitly labeled.",
                            "ALL_NON_DC combines component statistics rather than averaging band RSC values."]


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
        ("E_RNA_normalized_log_score", "RNA-defined hotspot enrichment", "Judge-half RNA normalized-log score difference between source-half RNA hotspots and remaining neighborhoods.", "split_half"),
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
                model = row.get("model") or "Two RNA halves"
                if "rsc" in row:
                    bundle.row(model, task, row["specimen"], "specimen", "cell_rsc", row["rsc"],
                               "saved cell model support; inspect eligible genes and view", details=details)
                else:
                    bundle.row("Split-half reliability", task, row["specimen"], "specimen", "rna_split_agreement", row["rna_split_agreement"],
                               "saved RNA graph support", details=details, role="split_half")
            elif "E_model_normalized_log_score" in row:
                for key, label, definition, role in hotspot_specs:
                    bundle.metric(key, label, definition, "score", role)
                    # The split-half reference can depend on the model's retained support; retain that context.
                    task = f"{track} | within-Neoplastic proliferation | support: {row['model']}"
                    model = row["model"] if role == "prediction" else "Two RNA halves"
                    bundle.row(model, task, row["specimen"], "specimen", key, row[key],
                               "saved eligible neoplastic neighborhoods", details=details, role=role)
    used = {row["metric"] for row in bundle.data["rows"]}
    bundle.data["metrics"] = [metric for metric in bundle.data["metrics"] if metric["id"] in used]
    bundle.data["notes"] = ["Units are specimens; patient identifiers remain in row details. Do not count sections as independent patients.",
                            "Model-specific cell supports may differ. Export preserves the saved support; it does not manufacture a common-support comparison.",
                            "Hotspot enrichment is a descriptive task result, not a survival estimate or a clinical prediction."]


def export_results(runs, output, workflow="auto", units=None, title=None, synthetic=False):
    runs = [Path(path) for path in runs]
    output = Path(output)
    if output.exists():
        raise ValueError("Output exists; choose a new JSON filename.")
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
        bundle.data["notes"].insert(0, "Synthetic software example. No patient data or manuscript evidence.")
    bundle.data["notes"].append("This bundle exports saved results without rerunning, clipping, averaging or recalibrating the scientific scores.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(finite_json(bundle.data), handle, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        handle.write("\n")
    return bundle.data


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow", choices=["auto", "hest", "her2", "fourier", "cell"], default="auto")
    parser.add_argument("--run", type=Path, action="append", required=True, help="Completed result directory or supported TSV; repeat for multiple units")
    parser.add_argument("--unit", action="append", help="One specimen ID per --run, required for Fourier tables")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title")
    parser.add_argument("--synthetic", action="store_true", help="Label generated inputs as synthetic; never label real data this way")
    args = parser.parse_args(argv)
    try:
        bundle = export_results(args.run, args.output, args.workflow, args.unit, args.title, args.synthetic)
    except (ValueError, KeyError, OSError) as error:
        parser.exit(2, f"Export stopped: {error}\n")
    print(json.dumps(dict(output=str(args.output), schema=SCHEMA, workflow=bundle["workflow"], rows=len(bundle["rows"]), spatial_scoring_rerun=False)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
