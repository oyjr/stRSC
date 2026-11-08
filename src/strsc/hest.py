#!/usr/bin/env python3
"""Evaluate held-out HEST predictions using counts, coordinates and a cohort manifest."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from . import hest_core as cal


class InputError(ValueError):
    pass


def read_counts(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as source:
        required = {"counts", "xy", "barcodes", "genes"}
        if not required.issubset(source.files):
            raise InputError(f"{path.name}: requires counts, xy, barcodes and genes")
        data = {key: source[key] for key in required}
    counts, xy = data["counts"], data["xy"]
    if counts.ndim != 2 or counts.shape[1] != 50 or counts.shape[0] < 20:
        raise InputError(f"{path.name}: requires at least 20 locations and the 50-gene HEST panel")
    if not np.issubdtype(counts.dtype, np.number) or not np.isfinite(counts).all():
        raise InputError(f"{path.name}: counts must be finite numbers")
    if np.any(counts < 0) or np.any(counts != np.rint(counts)) or np.any(counts > np.iinfo(np.uint32).max):
        raise InputError(f"{path.name}: raw nonnegative integer molecule counts are required")
    if xy.shape != (len(counts), 2) or not np.isfinite(xy).all() or np.ptp(xy, axis=0).min() <= 0:
        raise InputError(f"{path.name}: xy must contain two varying, finite spatial coordinates")
    for key, size in (("barcodes", len(counts)), ("genes", 50)):
        ids = data[key].astype(str)
        if ids.shape != (size,) or len(set(ids)) != size or any(not x.strip() for x in ids):
            raise InputError(f"{path.name}: {key} must contain unique, nonempty IDs")
        data[key] = ids
    data["counts"] = counts.astype(np.int64)
    data["xy"] = xy.astype(np.float64)
    return data


def read_prediction(path: Path, raw: dict, transform: str) -> np.ndarray:
    with h5py.File(path, "r") as source:
        if not {"prediction", "barcodes", "genes"}.issubset(source.keys()):
            raise InputError(f"{path.name}: prediction, barcodes and genes are required")
        barcodes, genes = cal._decode(source["barcodes"][...]), cal._decode(source["genes"][...])
        if len(set(barcodes)) != len(barcodes) or set(barcodes) != set(raw["barcodes"]):
            raise InputError(f"{path.name}: prediction locations must match the complete specimen")
        if not np.array_equal(genes.astype(str), raw["genes"]):
            raise InputError(f"{path.name}: prediction gene order must match counts")
        lookup = {barcode: i for i, barcode in enumerate(barcodes)}
        order = np.asarray([lookup[x] for x in raw["barcodes"]])
        prediction = np.asarray(source["prediction"], np.float64)
        if prediction.shape != raw["counts"].shape or not np.isfinite(prediction).all():
            raise InputError(f"{path.name}: invalid prediction shape or nonfinite values")
        if "truth_transform_id" in source.attrs and str(source.attrs["truth_transform_id"]) != transform:
            raise InputError(f"{path.name}: prediction truth transform differs from the declared HEST transform")
        if "raw_counts" in source and not np.array_equal(source["raw_counts"][...][order], raw["counts"]):
            raise InputError(f"{path.name}: embedded raw counts differ from the specimen input")
        if "transformed_truth" in source and not np.allclose(source["transformed_truth"][...][order], np.log1p(raw["counts"]), atol=1e-12, rtol=0):
            raise InputError(f"{path.name}: embedded truth is not log1p(raw counts)")
        if "spatial_coords" in source and not np.allclose(source["spatial_coords"][...][order], raw["xy"], atol=1e-9, rtol=0):
            raise InputError(f"{path.name}: embedded coordinates differ from the specimen input")
    return prediction[order]


def molecule_moments(counts: np.ndarray, task: str, specimen: str) -> dict:
    # Exact RNA-only scalar moments from cal.actual_moments. Its phase-rotation
    # simulations audit calibration and are not inputs to the scoring formula.
    d2 = np.zeros(counts.shape[1])
    covariance = np.zeros_like(d2)
    ell2 = np.zeros_like(d2)
    for seed in cal.THINNING_SEEDS:
        a, b = cal._halves(counts, task, specimen, seed)
        d2 += 0.5 * (np.mean(a * a, axis=0) + np.mean(b * b, axis=0))
        covariance += np.mean(a * b, axis=0)
        ell2 += np.sum(a * b, axis=0)
    scale = 1.0 / len(cal.THINNING_SEEDS)
    return {"d2": d2 * scale, "covariance": covariance * scale, "ell2": ell2 * scale}


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def write_table(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def accuracy(covariance: float, vrna: float, vpred: float) -> tuple[float, str]:
    if vpred <= cal.NUMERIC_TOL:
        return math.nan, "NO_PREDICTED_VARIANCE"
    if vrna <= 0:
        return math.nan, "RNA_UNRESOLVED"
    return float(covariance / math.sqrt(vrna * vpred)), "ESTIMABLE"


def run(manifest_path: Path, output: Path, threads: int = 8) -> dict:
    manifest_path, output = manifest_path.resolve(), output.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    task, specimen = manifest.get("task"), manifest.get("specimen")
    if manifest.get("protocol") != "HEST_G0047B" or task not in cal.TASKS or specimen not in cal.TASKS[task]:
        raise InputError("Use the HEST_G0047B protocol and a frozen HEST task/specimen")
    if set(manifest.get("cohort", {})) != set(cal.TASKS[task]):
        raise InputError("The complete task cohort is required to recompute leave-one-specimen-out RNA weights")
    transform = "log1p_raw_integer_counts_no_library_normalization"
    if manifest.get("prediction_transform") != transform:
        raise InputError(f"prediction_transform must be {transform}")
    paths = manifest.get("predictions", {})
    if not paths or any(not isinstance(x, str) or not x.strip() for x in paths):
        raise InputError("Supply at least one named prediction file")
    if output.exists():
        raise InputError("Output directory already exists; choose a new directory to preserve previous results")
    if not 1 <= threads <= 32:
        raise InputError("threads must be between 1 and 32")
    torch.set_num_threads(threads)
    started = time.time()
    cohort = {name: read_counts(manifest_path.parent / path) for name, path in manifest["cohort"].items()}
    raw = cohort[specimen]
    for name, data in cohort.items():
        if not np.array_equal(data["genes"], raw["genes"]):
            raise InputError(f"{name}: all cohort members must use the same ordered gene panel")
    predictions = {name: read_prediction(manifest_path.parent / path, raw, transform) for name, path in paths.items()}
    output.mkdir(parents=True)
    print("Computing molecule moments and held-out RNA weights...", flush=True)
    moments, retained = {}, {}
    for name, data in cohort.items():
        _, keep, _, _, _ = cal._graph(data["xy"])
        if not keep.any():
            raise InputError(f"{name}: no connected component has at least 20 locations")
        retained[name] = keep
        moments[name] = molecule_moments(data["counts"][keep], task, name)
    others = [name for name in cal.TASKS[task] if name != specimen]
    d2 = np.mean(np.stack([moments[name]["d2"] for name in others]), axis=0)
    cov = np.mean(np.stack([moments[name]["covariance"] for name in others]), axis=0)
    h = np.divide(cov, d2, out=np.zeros_like(d2), where=d2 > cal.NUMERIC_TOL)
    w, ell2 = np.maximum(h, 0.0), moments[specimen]["ell2"]
    eligible = (d2 > cal.NUMERIC_TOL) & (w > 0) & (ell2 > 0)
    d = np.sqrt(d2)
    rna_scale = np.divide(np.sqrt(w), d, out=np.zeros_like(w), where=d > cal.NUMERIC_TOL)
    rna_scale[~eligible] = 0.0
    p_scale = np.divide(np.sqrt(w) * np.sqrt(np.maximum(ell2, 0)), d, out=np.zeros_like(w), where=d > cal.NUMERIC_TOL)
    p_scale[~eligible] = 0.0
    np.savez_compressed(output / "recomputed_weights.npz", d2=d2, h=h, w=w, ell2=ell2, eligible=eligible)

    keep = retained[specimen]
    if int(keep.sum()) > 6000:
        raise InputError("This local CPU demo supports up to 6,000 graph-retained locations; do not crop the specimen to bypass the limit")
    cal.INPUT, cal.BASIS = output / "inputs", output / "basis"
    (cal.INPUT / task).mkdir(parents=True)
    np.savez_compressed(cal.INPUT / task / f"{specimen}.npz", **raw)
    print("Building the full coordinate graph and five spectral bands...", flush=True)
    cal.build_basis(task, specimen, "cpu")
    basis = cal._basis_dir(task, specimen)
    if not np.array_equal(np.load(basis / "keep.npy"), keep):
        raise InputError("Graph support changed during computation")
    bands = cal._device_bands([np.load(basis / f"U5_B{k}.npy") for k in range(1, 6)], "cpu")

    print("Recomputing RNA band coefficients from 20 molecule splits...", flush=True)
    vrna = np.zeros(5, dtype=np.float64)
    mean_observed = [None] * 5
    for seed in cal.THINNING_SEEDS:
        a, b = cal._halves(raw["counts"][keep], task, specimen, seed)
        a *= rna_scale[None, :]
        b *= rna_scale[None, :]
        ac, bc = cal._coefficients(bands, a, "cpu"), cal._coefficients(bands, b, "cpu")
        for k in range(5):
            vrna[k] += np.sum(ac[k] * bc[k])
            current = 0.5 * (ac[k] + bc[k])
            mean_observed[k] = current if mean_observed[k] is None else mean_observed[k] + current
    vrna /= len(cal.THINNING_SEEDS)
    mean_observed = [x / len(cal.THINNING_SEEDS) for x in mean_observed]
    band_rows, gene_rows, summaries = [], [], []
    truth = np.log1p(raw["counts"])
    for name, prediction_full in predictions.items():
        print(f"Scoring {name}...", flush=True)
        prediction = prediction_full[keep]
        centered = prediction - prediction.mean(axis=0, keepdims=True)
        pnorm = np.sqrt(np.sum(centered * centered, axis=0))
        p = np.divide(centered, pnorm[None, :], out=np.zeros_like(centered), where=pnorm[None, :] > cal.NUMERIC_TOL) * p_scale[None, :]
        pc = cal._coefficients(bands, p, "cpu")
        vpred = np.asarray([np.sum(x * x) for x in pc], dtype=np.float64)
        covariance = np.asarray([np.sum(pc[k] * mean_observed[k]) for k in range(5)], dtype=np.float64)
        gene_pcc = cal._corr_cols(prediction, truth[keep])
        full_pcc = cal._corr_cols(prediction_full, truth)
        for k in range(5):
            rsc, status = accuracy(covariance[k], vrna[k], vpred[k])
            band_rows.append({"task": task, "specimen": specimen, "model": name, "band": f"B{k+1}", "V_RNA": float(vrna[k]), "C_model": float(covariance[k]), "V_model": float(vpred[k]), "rsc": rsc, "rsc_status": status})
        summary = {"model": name, "task": task, "specimen": specimen, "ordinary_PCC_full_support": float(np.nanmean(full_pcc)) if np.isfinite(full_pcc).any() else math.nan, "ordinary_PCC_graph_support": float(np.nanmean(gene_pcc)) if np.isfinite(gene_pcc).any() else math.nan}
        for label, indices in (("coarse", [0, 1]), ("fine", [3, 4])):
            rsc, status = accuracy(float(covariance[indices].sum()), float(vrna[indices].sum()), float(vpred[indices].sum()))
            summary[label], summary[f"{label}_status"] = rsc, status
        summaries.append(summary)
        for k, gene in enumerate(raw["genes"]):
            gene_rows.append({"model": name, "specimen": specimen, "gene": gene, "pearson_graph_support": float(gene_pcc[k]), "spatial_eligible": bool(eligible[k]), "rna_weight": float(w[k]), "lopo_d2": float(d2[k]), "ell2": float(ell2[k])})

    write_table(output / "band_scores.tsv", band_rows)
    write_table(output / "specimen_scores.tsv", summaries)
    write_table(output / "gene_scores.tsv", gene_rows)
    if {"baseline", "intervention"}.issubset(predictions):
        by_name = {row["model"]: row for row in summaries}
        comparison = {"model": manifest.get("model", "uploaded_model"), "task": task, "unit_id": specimen, "unit_type": "specimen"}
        for arm in ("baseline", "intervention"):
            comparison.update({f"fine_{arm}": by_name[arm]["fine"], f"coarse_{arm}": by_name[arm]["coarse"], f"pcc_{arm}": by_name[arm]["ordinary_PCC_graph_support"]})
        if all(np.isfinite(value) for key, value in comparison.items() if key.startswith(("fine_", "coarse_", "pcc_"))):
            write_table(output / "result_comparison.tsv", [comparison])
    record = json_safe({"status": "COMPLETE", "protocol": manifest["protocol"], "task": task, "specimen": specimen, "patch_common_spots": len(keep), "graph_retained_spots": int(keep.sum()), "eligible_genes": int(eligible.sum()), "reference_specimens": others, "thinning_seeds": list(cal.THINNING_SEEDS), "truth_transform": cal.TRUTH_TRANSFORM_ID, "graph_edge_multiplier": cal.EDGE_MULTIPLIER, "minimum_component": cal.MIN_COMPONENT, "bands": "five equal-rank nonconstant eigenmode bands per component", "score_interpretation": "RSC; the tests with synthetic predictions are not rerun by this computation", "raw_counts_recomputed": True, "saved_graph_or_rna_cache_read": False, "saved_weights_read": False, "reference_scores_read": False, "confidence_interval": None, "device": "cpu", "threads": threads, "elapsed_seconds": time.time() - started, "models": summaries, "band_scores": band_rows})
    (output / "analysis.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    try:
        result = run(args.manifest, args.output, args.threads)
    except (InputError, OSError, ValueError, KeyError, cal.CalibrationError) as error:
        print(f"Analysis stopped: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"status": result["status"], "specimen": result["specimen"], "spots": result["graph_retained_spots"], "elapsed_seconds": result["elapsed_seconds"], "output": str(args.output.resolve())}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
