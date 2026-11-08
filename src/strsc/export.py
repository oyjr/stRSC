#!/usr/bin/env python3
"""Export a completed HEST run and its matched predictions for browser plots."""
from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path

import numpy as np
from .hest import InputError, read_counts, read_prediction
from .hest_core import _corr_cols


def export_analysis(manifest_path: Path, run_path: Path, output: Path,
                    baseline: str = "baseline", intervention: str = "intervention",
                    baseline_label: str = "Baseline", intervention_label: str = "Intervention") -> dict:
    if output.exists():
        raise InputError("Output exists; choose a new JSON filename.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    record = json.loads((run_path / "analysis.json").read_text(encoding="utf-8-sig"))
    if record.get("status") != "COMPLETE" or record.get("protocol") != "HEST_G0047B":
        raise InputError("Select a completed HEST run.")
    task, unit = record["task"], record["specimen"]
    if (task, unit) != (manifest["task"], manifest["specimen"]) or baseline == intervention:
        raise InputError("Run and manifest identities must agree; select two different predictions.")
    raw = read_counts(manifest_path.parent / manifest["cohort"][unit])
    snapshot_path = run_path / "inputs" / task / f"{unit}.npz"
    if snapshot_path.exists():
        snapshot = read_counts(snapshot_path)
        if any(not np.array_equal(raw[key], snapshot[key]) for key in raw):
            raise InputError("Counts, coordinates or IDs differ from the saved scoring input.")
    keep_path = run_path / "basis" / task / unit / "keep.npy"
    if keep_path.exists():
        keep = np.load(keep_path, allow_pickle=False)
        if keep.dtype != bool or keep.shape != (len(raw["counts"]),):
            raise InputError("Invalid saved graph support.")
    elif record["patch_common_spots"] == record["graph_retained_spots"] == len(raw["counts"]):
        keep = np.ones(len(raw["counts"]), dtype=bool)
    else:
        raise InputError("Saved basis/.../keep.npy is required when graph locations were excluded.")
    if int(keep.sum()) != record["graph_retained_spots"]:
        raise InputError("Saved graph support does not match the run.")
    truth = np.log1p(raw["counts"][keep])
    with (run_path / "gene_scores.tsv").open(encoding="utf-8-sig", newline="") as handle:
        gene_rows = list(csv.DictReader(handle, delimiter="\t"))
    summaries = {row["model"]: row for row in record["models"]}
    arms = {}
    for key, source_name in (("baseline", baseline), ("intervention", intervention)):
        if source_name not in summaries or source_name not in manifest["predictions"]:
            raise InputError(f"Missing prediction or saved score: {source_name}")
        predicted = read_prediction(manifest_path.parent / manifest["predictions"][source_name], raw, manifest["prediction_transform"])[keep]
        by_gene = {row["gene"]: row for row in gene_rows if row["model"] == source_name and row["specimen"] == unit}
        if set(by_gene) != set(raw["genes"]):
            raise InputError("Saved gene scores do not match the input panel.")
        scores = []
        # A descriptive alignment check catches changed prediction files without rerunning spatial scoring.
        corr = _corr_cols(predicted, truth)
        for index, gene in enumerate(raw["genes"]):
            row = by_gene[gene]
            saved = float(row["pearson_graph_support"])
            if np.isfinite(saved) != np.isfinite(corr[index]) or (np.isfinite(saved) and abs(saved - corr[index]) > 1e-9):
                raise InputError(f"Prediction and saved Pearson disagree: {source_name}, {gene}")
            scores.append({"pearson": saved if np.isfinite(saved) else None,
                           "spatial_eligible": row["spatial_eligible"].lower() == "true",
                           "rna_weight": float(row["rna_weight"])})
        bands = sorted((row for row in record["band_scores"] if row["model"] == source_name), key=lambda row: row["band"])
        if [row["band"] for row in bands] != [f"B{k}" for k in range(1, 6)]:
            raise InputError("Five saved spatial bands are required.")
        arms[key] = {"source_name": source_name, "values": predicted.tolist(), "gene_scores": scores,
                     "bands": [{"accuracy": row["rsc"], "status": row["rsc_status"],
                                "rna_variance": row["V_RNA"], "prediction_variance": row["V_model"],
                                "covariance": row["C_model"]} for row in bands]}
    model = manifest.get("model", "Model")
    pair = {"model": model, "task": task, "unit_id": unit, "unit_type": "specimen",
            "score_definition": "combined_components", "baseline_condition": baseline_label,
            "intervention_condition": intervention_label, "source_table": "Completed HEST analysis"}
    for arm, source_name in (("baseline", baseline), ("intervention", intervention)):
        for metric, field in (("fine", "fine"), ("coarse", "coarse"), ("pcc", "ordinary_PCC_graph_support")):
            value = summaries[source_name][field]
            if value is None or not np.isfinite(value):
                raise InputError("Paired comparison requires finite fine, broad-scale and Pearson scores.")
            pair[f"{metric}_{arm}"] = value
    table = io.StringIO(newline="")
    writer = csv.DictWriter(table, fieldnames=list(pair), delimiter="\t")
    writer.writeheader()
    writer.writerow(pair)
    bundle = {"schema": "strsc-analysis-bundle-v1", "score_table": table.getvalue(),
              "specimens": [{"model": model, "task": task, "unit_id": unit,
                             "expression_scale": "log1p(raw count)", "coordinate_unit": "input coordinates",
                             "support": "graph-retained locations", "score_interpretation": "Reproducible-signal correlation (RSC); saved HEST scores",
                             "genes": raw["genes"].tolist(), "spot_ids": raw["barcodes"][keep].tolist(),
                             "xy": raw["xy"][keep].tolist(), "truth": truth.tolist(), **arms}]}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(bundle, handle, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        handle.write("\n")
    return {"output": str(output), "specimens": 1, "locations": len(truth), "genes": truth.shape[1], "spatial_scoring_rerun": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", default="baseline")
    parser.add_argument("--intervention", default="intervention")
    parser.add_argument("--baseline-label", default="Baseline")
    parser.add_argument("--intervention-label", default="Intervention")
    args = parser.parse_args()
    try:
        print(json.dumps(export_analysis(args.manifest, args.run, args.output, args.baseline, args.intervention, args.baseline_label, args.intervention_label)))
    except (InputError, OSError, ValueError, KeyError) as error:
        parser.exit(2, f"Export stopped: {error}\n")


if __name__ == "__main__":
    main()
