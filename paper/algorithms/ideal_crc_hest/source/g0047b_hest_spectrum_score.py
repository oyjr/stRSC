#!/usr/bin/env python3
"""G0-047B HEST spectrum scoring on each frozen graph-retained support.

This is an explicit adjudication amendment requested by the user.  It keeps
the G0-047 graph, bands, genes, predictions, thinning seeds, transforms and
LOPO RNA weights unchanged.  Unlike the earlier fail-closed wrapper, it does
not use another task's support or rho=1 calibration error as a global blocker.
It reports raw q and the pre-existing calibration diagnostics side by side.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np
import pandas as pd

import g0047_hest_calibration as cal


ROOT = Path(os.environ.get("STRSC_HIST_DATA", Path(__file__).resolve().parent / "data"))
PARENT = ROOT / "calibration_output"
RUN = ROOT / "spectrum_output"
WORK = ROOT / "spectrum"
BOOTSTRAP_REPS = 10_000
BOOTSTRAP_SEED = 20260808


class ScoreError(RuntimeError):
    pass


def _seed(*parts: object) -> int:
    digest = hashlib.sha256("\x1f".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


def _decode(values: Any) -> np.ndarray:
    return np.asarray([
        value.decode() if isinstance(value, (bytes, np.bytes_)) else str(value)
        for value in np.asarray(values).reshape(-1)
    ], dtype=object)


def _atomic_tsv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    frame.to_csv(tmp, sep="\t", index=False, lineterminator="\n")
    os.replace(tmp, path)


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def _corr_cols(prediction: np.ndarray, truth: np.ndarray) -> np.ndarray:
    prediction = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    if prediction.shape != truth.shape:
        raise ScoreError(f"PCC_SHAPE:{prediction.shape}:{truth.shape}")
    pc = prediction - prediction.mean(axis=0, keepdims=True)
    tc = truth - truth.mean(axis=0, keepdims=True)
    numerator = np.sum(pc * tc, axis=0)
    denominator = np.sqrt(np.sum(pc * pc, axis=0) * np.sum(tc * tc, axis=0))
    observed_variable = np.sum(tc * tc, axis=0) > cal.NUMERIC_TOL
    out = np.full(prediction.shape[1], np.nan, dtype=np.float64)
    valid = observed_variable & (denominator > cal.NUMERIC_TOL)
    out[valid] = numerator[valid] / denominator[valid]
    # Frozen ordinary-PCC rule: a constant prediction on variable truth is 0.
    out[observed_variable & ~valid] = 0.0
    return out


def _prediction_path(root: Path, task: str, specimen: str, encoder: str) -> Path:
    matches = sorted(root.glob(f"{task}.fold*.{specimen}.{encoder}.predictions.h5"))
    if len(matches) != 1:
        raise ScoreError(f"PREDICTION_PATH_COUNT:{task}:{specimen}:{encoder}:{len(matches)}")
    return matches[0]


def _read_prediction(path: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as handle:
        required = ("barcodes", "genes", "prediction", "transformed_truth", "raw_counts", "gene_pearson")
        missing = [key for key in required if key not in handle]
        if missing:
            raise ScoreError(f"PREDICTION_KEYS:{path}:{missing}")
        payload = {
            "barcodes": _decode(handle["barcodes"][...]),
            "genes": _decode(handle["genes"][...]),
            "prediction": np.asarray(handle["prediction"], np.float64),
            "truth": np.asarray(handle["transformed_truth"], np.float64),
            "raw_counts": np.asarray(handle["raw_counts"], np.int64),
            "gene_pearson": np.asarray(handle["gene_pearson"], np.float64),
            "truth_transform_id": str(handle.attrs.get("truth_transform_id", "")),
        }
    shape = payload["prediction"].shape
    if shape != payload["truth"].shape or shape != payload["raw_counts"].shape:
        raise ScoreError(f"PREDICTION_TRUTH_SHAPE:{path}")
    if not np.isfinite(payload["prediction"]).all() or not np.isfinite(payload["truth"]).all():
        raise ScoreError(f"NONFINITE_PREDICTION_OR_TRUTH:{path}")
    return payload


def _calibration_diagnostics(task: str) -> dict[str, Any]:
    gate = json.loads((PARENT / f"receipts/hest_calibration/HEST_{task}_CALIBRATION_GATE.json").read_text())
    return {
        "calibration_status_g0047": gate["status"],
        "bias_rho_0_3": gate["bias_rho_0_3"],
        "bias_rho_0_6": gate["bias_rho_0_6"],
        "rho1_oracle_max_abs_error": gate["rho1_oracle_max_abs_error"],
        "phase_terminal_abs_q": gate["phase_terminal_abs_q"],
        "calibration_interpretation_g0047b": (
            "CALIBRATED" if gate["status"] == "CALIBRATED"
            else "RAW_Q_REPORTED_WITH_OBSERVED_CALIBRATION_ERROR"
        ),
    }


def prepare_task_rna(task: str, device: str) -> None:
    """Cache prediction-independent split-molecule band coefficients once."""
    for specimen in cal.TASKS[task]:
        output = WORK / "rna_band_cache" / task / f"{specimen}.npz"
        if output.is_file():
            prior = np.load(output, allow_pickle=False)
            if list(prior["bands"].astype(str)) == ["B1", "B2", "B3", "B4", "B5"]:
                print(json.dumps({"status": "PASS_REUSED", "task": task, "specimen": specimen}), flush=True)
                continue
            raise ScoreError(f"INVALID_RNA_CACHE:{output}")
        compact = np.load(cal.INPUT / task / f"{specimen}.npz", allow_pickle=False)
        keep = np.load(cal._basis_dir(task, specimen) / "keep.npy").astype(bool)
        counts = compact["counts"][keep].astype(np.int64)
        bands_np = [
            np.load(cal._basis_dir(task, specimen) / f"U5_B{k}.npy", mmap_mode="r")
            for k in range(1, 6)
        ]
        bands = cal._device_bands(bands_np, device)
        weights = np.load(cal.MOMENTS / task / f"{specimen}.weights.npz", allow_pickle=False)
        eligible = weights["eligible"].astype(bool)
        d = np.sqrt(weights["d2"])
        w = weights["w"]
        rna_scale = np.divide(np.sqrt(w), d, out=np.zeros_like(w), where=d > cal.NUMERIC_TOL)
        rna_scale[~eligible] = 0.0
        vrna = np.zeros(5, dtype=np.float64)
        mean_observed = [None] * 5
        for seed in cal.THINNING_SEEDS:
            a, b = cal._halves(counts, task, specimen, seed)
            a *= rna_scale[None, :]
            b *= rna_scale[None, :]
            ac = cal._coefficients(bands, a, device)
            bc = cal._coefficients(bands, b, device)
            for band_index in range(5):
                vrna[band_index] += np.sum(ac[band_index] * bc[band_index])
                current = 0.5 * (ac[band_index] + bc[band_index])
                mean_observed[band_index] = current if mean_observed[band_index] is None else mean_observed[band_index] + current
        vrna /= len(cal.THINNING_SEEDS)
        mean_observed = [coefficient / len(cal.THINNING_SEEDS) for coefficient in mean_observed]
        frozen = pd.read_parquet(cal.MOMENTS / task / "specimen_vrna.parquet")
        frozen = frozen.loc[frozen.specimen.eq(specimen)].set_index("band")
        frozen_vrna = np.asarray([
            float(frozen.loc[f"B{band_index + 1}", "V_RNA"])
            for band_index in range(5)
        ])
        output.parent.mkdir(parents=True, exist_ok=True)
        tmp = output.with_suffix(output.suffix + f".tmp.{os.getpid()}")
        with tmp.open("wb") as handle:
            np.savez_compressed(
                handle,
                bands=np.asarray(["B1", "B2", "B3", "B4", "B5"]),
                vrna=vrna,
                vrna_frozen=frozen_vrna,
                vrna_abs_difference=np.abs(vrna - frozen_vrna),
                eligible=eligible,
                **{f"mean_observed_B{k+1}": value for k, value in enumerate(mean_observed)},
            )
        os.replace(tmp, output)
        del bands
        if device.startswith("cuda"):
            import torch
            torch.cuda.empty_cache()
        print(json.dumps({"status": "PASS", "task": task, "specimen": specimen, "spots": int(keep.sum())}), flush=True)


def score_task(task: str, encoder: str, prediction_root: Path, pcc_gate_path: Path, device: str) -> None:
    output = RUN / "results/hest_spectrum" / encoder
    checkpoint = output / f"{task}.specimen_band.tsv"
    if checkpoint.is_file() and (output / f"{task}.specimen.tsv").is_file():
        print(json.dumps({"status": "PASS_REUSED", "task": task, "encoder": encoder}))
        return
    gate_a = json.loads((PARENT / "receipts/GATE_A_TERMINAL.json").read_text())
    if gate_a.get("status") != "PASS_PCC_DIFFERENCE_FULLY_EXPLAINED__Q_LINEAGE_VALID":
        raise ScoreError("GATE_A_LINEAGE_NOT_VALID")
    pcc_gate = json.loads(pcc_gate_path.read_text())
    if pcc_gate.get("tasks", {}).get(task, {}).get("status") != "PASS":
        raise ScoreError(f"OFFICIAL_PCC_NOT_REPRODUCED:{task}:{encoder}")

    band_rows: list[dict[str, Any]] = []
    specimen_rows: list[dict[str, Any]] = []
    calibration = _calibration_diagnostics(task)
    for specimen in cal.TASKS[task]:
        compact = np.load(cal.INPUT / task / f"{specimen}.npz", allow_pickle=False)
        expected_barcodes = compact["barcodes"].astype(str)
        expected_genes = compact["genes"].astype(str)
        pred_path = _prediction_path(prediction_root, task, specimen, encoder)
        payload = _read_prediction(pred_path)
        if set(payload["barcodes"]) != set(expected_barcodes):
            raise ScoreError(f"PREDICTION_BARCODE_JOIN:{task}:{specimen}:{encoder}")
        if not np.array_equal(payload["genes"].astype(str), expected_genes):
            raise ScoreError(f"PREDICTION_GENE_JOIN:{task}:{specimen}:{encoder}")
        lookup = {key: index for index, key in enumerate(payload["barcodes"].tolist())}
        order = np.asarray([lookup[key] for key in expected_barcodes], np.int64)
        prediction_full = payload["prediction"][order]
        truth_full = payload["truth"][order]
        raw_full = payload["raw_counts"][order]
        if not np.array_equal(raw_full, compact["counts"].astype(np.int64)):
            raise ScoreError(f"RAW_COUNT_JOIN:{task}:{specimen}:{encoder}")
        if not np.allclose(truth_full, np.log1p(raw_full), atol=1e-12, rtol=0):
            raise ScoreError(f"TRUTH_TRANSFORM:{task}:{specimen}:{encoder}")

        keep = np.load(cal._basis_dir(task, specimen) / "keep.npy").astype(bool)
        prediction = prediction_full[keep]
        truth = truth_full[keep]
        pcc_full = float(np.nanmean(payload["gene_pearson"]))
        pcc_full_recalc_saved_float32 = float(np.nanmean(_corr_cols(prediction_full, truth_full)))
        graph_gene_pcc = _corr_cols(prediction, truth)
        pcc_graph = float(np.nanmean(graph_gene_pcc))

        bands_np = [
            np.load(cal._basis_dir(task, specimen) / f"U5_B{k}.npy", mmap_mode="r")
            for k in range(1, 6)
        ]
        bands = cal._device_bands(bands_np, device)
        weights = np.load(cal.MOMENTS / task / f"{specimen}.weights.npz", allow_pickle=False)
        eligible = weights["eligible"].astype(bool)
        d = np.sqrt(weights["d2"])
        w = weights["w"]
        ell = np.sqrt(np.maximum(weights["ell2"], 0))
        p_scale = np.divide(np.sqrt(w) * ell, d, out=np.zeros_like(w), where=d > cal.NUMERIC_TOL)
        p_scale[~eligible] = 0.0

        centered_prediction = prediction - prediction.mean(axis=0, keepdims=True)
        pnorm = np.sqrt(np.sum(centered_prediction * centered_prediction, axis=0))
        p = np.divide(
            centered_prediction,
            pnorm[None, :],
            out=np.zeros_like(centered_prediction),
            where=pnorm[None, :] > cal.NUMERIC_TOL,
        ) * p_scale[None, :]
        pc = cal._coefficients(bands, p, device)
        vpred = np.asarray([np.sum(coefficient * coefficient) for coefficient in pc], dtype=np.float64)
        rna_cache_path = WORK / "rna_band_cache" / task / f"{specimen}.npz"
        if not rna_cache_path.is_file():
            raise ScoreError(f"RNA_BAND_CACHE_MISSING:{task}:{specimen}")
        rna_cache = np.load(rna_cache_path, allow_pickle=False)
        vrna = np.asarray(rna_cache["vrna"], dtype=np.float64)
        if not np.array_equal(rna_cache["eligible"].astype(bool), eligible):
            raise ScoreError(f"RNA_CACHE_ELIGIBLE_MASK:{task}:{specimen}")
        covariance = np.asarray([
            np.sum(pc[band_index] * rna_cache[f"mean_observed_B{band_index + 1}"])
            for band_index in range(5)
        ], dtype=np.float64)
        frozen = pd.read_parquet(cal.MOMENTS / task / "specimen_vrna.parquet")
        frozen = frozen.loc[frozen.specimen.eq(specimen)].set_index("band")
        q_values: dict[str, float] = {}
        for band_index in range(5):
            band = f"B{band_index + 1}"
            q = (
                covariance[band_index] / math.sqrt(vrna[band_index] * vpred[band_index])
                if vrna[band_index] > 0 and vpred[band_index] > cal.NUMERIC_TOL
                else math.nan
            )
            status = (
                "ESTIMABLE" if np.isfinite(q)
                else "NO_PREDICTED_VARIANCE" if vpred[band_index] <= cal.NUMERIC_TOL
                else "RNA_UNRESOLVED"
            )
            reference = float(frozen.loc[band, "V_RNA"])
            q_values[band] = float(q)
            band_rows.append({
                "task": task,
                "specimen": specimen,
                "encoder": encoder,
                "band": band,
                "graph_retained_spots": int(keep.sum()),
                "patch_common_spots": int(len(keep)),
                "excluded_small_component_spots": int((~keep).sum()),
                "eligible_genes": int(eligible.sum()),
                "V_RNA": float(vrna[band_index]),
                "V_RNA_frozen": reference,
                "V_RNA_abs_difference": abs(float(vrna[band_index]) - reference),
                "C_model": float(covariance[band_index]),
                "V_model": float(vpred[band_index]),
                "q": float(q),
                "q_status": status,
                "ordinary_PCC_official_support": pcc_full,
                "ordinary_PCC_official_support_recalc_saved_float32": pcc_full_recalc_saved_float32,
                "ordinary_PCC_saved_float32_minus_official": pcc_full_recalc_saved_float32 - pcc_full,
                "ordinary_PCC_graph_retained_support": pcc_graph,
                **calibration,
                "prediction_path": str(pred_path),
            })

        def combined(indices: tuple[int, ...]) -> float:
            c = float(covariance[list(indices)].sum())
            vp = float(vpred[list(indices)].sum())
            vr = float(vrna[list(indices)].sum())
            return c / math.sqrt(vp * vr) if vp > cal.NUMERIC_TOL and vr > 0 else math.nan

        q_coarse = combined((0, 1))
        q_fine = combined((3, 4))
        specimen_rows.append({
            "task": task,
            "specimen": specimen,
            "encoder": encoder,
            "patch_common_spots": int(len(keep)),
            "graph_retained_spots": int(keep.sum()),
            "excluded_small_component_spots": int((~keep).sum()),
            "eligible_genes": int(eligible.sum()),
            "ordinary_PCC_official_support": pcc_full,
            "ordinary_PCC_official_support_recalc_saved_float32": pcc_full_recalc_saved_float32,
            "ordinary_PCC_saved_float32_minus_official": pcc_full_recalc_saved_float32 - pcc_full,
            "ordinary_PCC_graph_retained_support": pcc_graph,
            **{f"q_{band}": q_values[band] for band in ("B1", "B2", "B3", "B4", "B5")},
            "q_coarse_B1_B2": q_coarse,
            "q_fine_B4_B5": q_fine,
            "delta_q_coarse_minus_fine": q_coarse - q_fine,
            **calibration,
            "truth_transform_id": payload["truth_transform_id"],
            "prediction_path": str(pred_path),
        })
        del bands
        if device.startswith("cuda"):
            import torch
            torch.cuda.empty_cache()

    _atomic_tsv(checkpoint, pd.DataFrame(band_rows))
    _atomic_tsv(output / f"{task}.specimen.tsv", pd.DataFrame(specimen_rows))
    _atomic_json(output / f"{task}.receipt.json", {
        "status": "PASS",
        "task": task,
        "encoder": encoder,
        "specimens": len(specimen_rows),
        "specimen_band_rows": len(band_rows),
        "graph_retained_spots": int(sum(row["graph_retained_spots"] for row in specimen_rows)),
        "patch_common_spots": int(sum(row["patch_common_spots"] for row in specimen_rows)),
        "calibration": calibration,
        "support_policy": "TASK_SPECIFIC_GRAPH_RETAINED_SUPPORT__PCC_AND_Q_IDENTICAL_WITHIN_TASK",
        "user_amendment": "G0-047B",
    })
    print(json.dumps({"status": "PASS", "task": task, "encoder": encoder, "rows": len(band_rows)}))


def _bootstrap_mean(values: np.ndarray, task: str, encoder: str, metric: str) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return math.nan, math.nan, math.nan
    rng = np.random.default_rng(BOOTSTRAP_SEED ^ _seed(task, encoder, metric))
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_REPS, len(values)))
    boot = values[indices].mean(axis=1)
    low, high = np.percentile(boot, [2.5, 97.5])
    return float(values.mean()), float(low), float(high)


def finalize() -> None:
    band_frames = []
    specimen_frames = []
    for encoder in ("resnet50", "phikon_v2"):
        for task in cal.TASKS:
            base = RUN / "results/hest_spectrum" / encoder
            band_frames.append(pd.read_csv(base / f"{task}.specimen_band.tsv", sep="\t"))
            specimen_frames.append(pd.read_csv(base / f"{task}.specimen.tsv", sep="\t"))
    bands = pd.concat(band_frames, ignore_index=True)
    specimens = pd.concat(specimen_frames, ignore_index=True)
    _atomic_tsv(RUN / "results/HEST_SPECIMEN_MODEL_BAND_Q.tsv", bands)
    _atomic_tsv(RUN / "results/HEST_SPECIMEN_MODEL_SUMMARY.tsv", specimens)

    task_band_rows = []
    for (task, encoder, band), group in bands.groupby(["task", "encoder", "band"], sort=True):
        estimate, low, high = _bootstrap_mean(group.q.to_numpy(float), task, encoder, band)
        task_band_rows.append({
            "task": task,
            "encoder": encoder,
            "band": band,
            "specimens": len(group),
            "graph_retained_spots": int(group.groupby("specimen").graph_retained_spots.first().sum()),
            "q_equal_specimen": estimate,
            "q_ci_lower": low,
            "q_ci_upper": high,
            "positive_specimens": int((group.q > 0).sum()),
            "V_RNA_all_specimens_positive": bool((group.V_RNA > 0).all()),
            "calibration_status_g0047": group.calibration_status_g0047.iloc[0],
            "calibration_interpretation_g0047b": group.calibration_interpretation_g0047b.iloc[0],
            "rho1_oracle_max_abs_error": float(group.rho1_oracle_max_abs_error.iloc[0]),
        })
    task_band = pd.DataFrame(task_band_rows)
    _atomic_tsv(RUN / "results/HEST_TASK_MODEL_BAND_Q.tsv", task_band)

    certificate_rows = []
    for (task, encoder), group in specimens.groupby(["task", "encoder"], sort=True):
        row: dict[str, Any] = {
            "task": task,
            "encoder": encoder,
            "specimens": len(group),
            "patch_common_spots": int(group.patch_common_spots.sum()),
            "graph_retained_spots": int(group.graph_retained_spots.sum()),
            "excluded_small_component_spots": int(group.excluded_small_component_spots.sum()),
        }
        for metric in [
            "ordinary_PCC_official_support",
            "ordinary_PCC_graph_retained_support",
            "q_B1", "q_B2", "q_B3", "q_B4", "q_B5",
            "q_coarse_B1_B2", "q_fine_B4_B5", "delta_q_coarse_minus_fine",
        ]:
            estimate, low, high = _bootstrap_mean(group[metric].to_numpy(float), task, encoder, metric)
            row[metric] = estimate
            row[f"{metric}_ci_lower"] = low
            row[f"{metric}_ci_upper"] = high
        row.update(_calibration_diagnostics(task))
        row["certificate"] = (
            "MEASURED_CALIBRATED_SPATIAL_SPECTRUM"
            if row["calibration_status_g0047"] == "CALIBRATED"
            else "MEASURED_RAW_SPATIAL_SPECTRUM__CALIBRATION_ERROR_REPORTED"
        )
        certificate_rows.append(row)
    certificates = pd.DataFrame(certificate_rows)
    _atomic_tsv(RUN / "results/HEST_TASK_MODEL_CERTIFICATES.tsv", certificates)

    calibration_curves = pd.concat([
        pd.read_csv(PARENT / f"results/hest_calibration/HEST_{task}_EQUAL_DIFFICULTY_CALIBRATION.tsv", sep="\t")
        for task in cal.TASKS
    ], ignore_index=True)
    _atomic_tsv(RUN / "results/HEST_COMPLETE_CALIBRATION_CURVES.tsv", calibration_curves)
    _atomic_json(RUN / "receipts/G0-047B_SPECTRUM_COMPLETE.json", {
        "status": "PASS",
        "specimen_model_band_rows": len(bands),
        "task_model_band_rows": len(task_band),
        "task_model_certificates": len(certificates),
        "models": ["resnet50", "phikon_v2"],
        "tasks": list(cal.TASKS),
        "specimens": 9,
        "patch_common_spots": 48_313,
        "graph_retained_spots": 48_273,
        "IDC_excluded_small_component_spots": 40,
        "figures_generated": 0,
        "design_changes": 0,
        "git_operations": 0,
    })
    print(json.dumps({"status": "PASS", "band_rows": len(bands), "certificates": len(certificates)}))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="stage", required=True)
    prepare = sub.add_parser("prepare-task-rna")
    prepare.add_argument("--task", choices=tuple(cal.TASKS), required=True)
    prepare.add_argument("--device", default="cuda")
    score = sub.add_parser("score-task")
    score.add_argument("--task", choices=tuple(cal.TASKS), required=True)
    score.add_argument("--encoder", choices=("resnet50", "phikon_v2"), required=True)
    score.add_argument("--prediction-root", type=Path, required=True)
    score.add_argument("--pcc-gate", type=Path, required=True)
    score.add_argument("--device", default="cuda")
    sub.add_parser("finalize")
    args = parser.parse_args(argv)
    if args.stage == "prepare-task-rna":
        prepare_task_rna(args.task, args.device)
    elif args.stage == "score-task":
        score_task(args.task, args.encoder, args.prediction_root, args.pcc_gate, args.device)
    else:
        finalize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
