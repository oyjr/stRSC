#!/usr/bin/env python3
"""G0-048: place CRC virtual-ST recovery on a physical micrometre axis.

The script replays the already-fitted G0-031 patient-held-out predictors, splits
integer RNA molecules into independent source/judge halves, and measures signed
model/RNA concordance in fixed physical wavelength bands on the regular CRC
pseudo-spot grids. No model is retrained.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import sparse
from scipy import fft as spfft
import torch
from torch.nn import functional as F

# G0-031 checkpoints were serialized with NumPy 2.x module names. The current
# frozen analysis environment uses NumPy 1.24; these aliases only restore the
# original array objects and do not change their values.
sys.modules.setdefault("numpy._core", np.core)
sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
sys.modules.setdefault("numpy._core.numeric", np.core.numeric)


SAMPLES = ("P1CRC", "P2CRC", "P3NAT")
SCALES = (32, 64, 96, 128, 192, 256, 384, 512)
SPLIT_SEEDS = tuple(range(2026073101, 2026073106))
GENE_PANEL_SIZE = 1000
GENE_BLOCK = 32

# Fixed physical wavelength windows. The first and last bins are open-ended so
# every non-constant Fourier mode is represented exactly once.
WAVELENGTH_BANDS = (
    ("LT128", 0.0, 128.0),
    ("128_256", 128.0, 256.0),
    ("256_512", 256.0, 512.0),
    ("512_1024", 512.0, 1024.0),
    ("1024_2048", 1024.0, 2048.0),
    ("2048_4096", 2048.0, 4096.0),
    ("GE4096", 4096.0, math.inf),
)

FROZEN_MODEL_BY_SCALE = {
    32: "LOCAL_CONTEXT_NETWORK",
    64: "LOCAL_CONTEXT_NETWORK",
    96: "MORPHOLOGY_RIDGE",
    128: "MORPHOLOGY_RIDGE",
    192: "MORPHOLOGY_RIDGE",
    256: "MORPHOLOGY_RIDGE",
    384: "COMPOSITION_RIDGE",
    512: "COMPOSITION_RIDGE",
}

SCALE_CONTEXT = {
    32: (0.5216, 0.2730, 0.0250, 0.1805),
    64: (0.3267, 0.4382, 0.0309, 0.2041),
    96: (0.2267, 0.5244, 0.0354, 0.2135),
    128: (0.1671, 0.5732, 0.0286, 0.2311),
    192: (0.1010, 0.6222, 0.0038, 0.2730),
    256: (0.0685, 0.6411, 0.0243, 0.2662),
    384: (0.0372, 0.6488, 0.0007, 0.3133),
    512: (0.0235, 0.6650, -0.0745, 0.3861),
}


def gelu_numpy(x: np.ndarray) -> np.ndarray:
    return F.gelu(torch.from_numpy(x)).numpy()


def layer_norm_numpy(
    x: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: float = 1e-5
) -> np.ndarray:
    mean = x.mean(axis=1, keepdims=True)
    variance = x.var(axis=1, keepdims=True)
    return ((x - mean) / np.sqrt(variance + eps)) * weight + bias


def apply_transform(x: np.ndarray, spec: dict[str, Any]) -> np.ndarray:
    return (
        (np.clip(x, spec["low"], spec["high"]) - spec["mean"])
        / spec["scale"]
    ).astype(np.float32)


def pca_transform(x: np.ndarray, spec: dict[str, Any]) -> np.ndarray:
    projected = (x - spec["mean"]) @ spec["components"].T
    projected /= np.sqrt(spec["explained_variance"])
    return projected.astype(np.float32)


def composition_features(path: Path, windows: pd.DataFrame) -> np.ndarray:
    frame = pq.read_table(path).to_pandas()
    columns = sorted(c for c in frame.columns if c.startswith("composition__"))
    density = np.log1p(
        frame["total_umi"].to_numpy(np.float64)
        / np.maximum(frame["effective_area_um2"].to_numpy(np.float64), 1.0)
    )
    return np.column_stack(
        [
            frame[columns].to_numpy(np.float32),
            density.astype(np.float32),
            frame["tissue_fraction"].to_numpy(np.float32),
        ]
    ).astype(np.float32)


class RidgeFold:
    def __init__(
        self,
        checkpoint: Path,
        composition: np.ndarray,
        morphology: np.ndarray,
        n_genes: int,
    ) -> None:
        payload = np.load(checkpoint, allow_pickle=False)
        selected = payload["selected_gene_indices"].astype(np.int64)
        self.position = np.full(n_genes, -1, dtype=np.int64)
        self.position[selected] = np.arange(len(selected), dtype=np.int64)
        raw_x = {
            "COMPOSITION_RIDGE": composition,
            "MORPHOLOGY_RIDGE": np.concatenate(
                [composition, morphology], axis=1
            ).astype(np.float32),
        }
        self.x: dict[str, np.ndarray] = {}
        self.params: dict[str, tuple[np.ndarray, ...]] = {}
        self.alpha_morphology = float(payload["alpha_morphology"][0])
        for model, prefix in (
            ("COMPOSITION_RIDGE", "composition"),
            ("MORPHOLOGY_RIDGE", "morphology"),
        ):
            self.params[model] = tuple(
                payload[f"{prefix}_{name}"]
                for name in (
                    "mean_x",
                    "scale_x",
                    "mean_y",
                    "weight",
                    "low_x",
                    "high_x",
                )
            )
            mean_x, scale_x, _, _, low_x, high_x = self.params[model]
            self.x[model] = (
                (np.clip(raw_x[model], low_x, high_x) - mean_x) / scale_x
            ).astype(np.float32)

    def predict(self, model: str, genes: np.ndarray) -> np.ndarray:
        local = self.position[genes]
        keep = local >= 0
        output = np.zeros((len(self.x[model]), len(genes)), dtype=np.float32)
        if not keep.any():
            return output
        _, _, mean_y, weight, _, _ = self.params[model]
        x = self.x[model]
        output[:, keep] = (
            x @ weight[:, local[keep]] + mean_y[local[keep]]
        ).astype(np.float32)
        return output


class LocalContextFold:
    def __init__(
        self,
        checkpoint: Path,
        composition_raw: np.ndarray,
        phikon_local_raw: np.ndarray,
        phikon_context_raw: np.ndarray,
        n_genes: int,
    ) -> None:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        selected = np.asarray(payload["selected_gene_indices"], dtype=np.int64)
        self.position = np.full(n_genes, -1, dtype=np.int64)
        self.position[selected] = np.arange(len(selected), dtype=np.int64)
        state = {
            key: value.detach().cpu().numpy().astype(np.float32)
            for key, value in payload["model_state_dict"].items()
        }
        comp = apply_transform(composition_raw, payload["composition_transform"])
        local = apply_transform(
            pca_transform(phikon_local_raw, payload["pca_parameters"]["local"]),
            payload["local_transform"],
        )
        context = apply_transform(
            pca_transform(phikon_context_raw, payload["pca_parameters"]["context"]),
            payload["context_transform"],
        )
        comp_h = gelu_numpy(comp @ state["composition.0.weight"].T + state["composition.0.bias"])
        local_h = gelu_numpy(local @ state["local.0.weight"].T + state["local.0.bias"])
        context_h = gelu_numpy(context @ state["context.0.weight"].T + state["context.0.bias"])
        hidden = np.concatenate([comp_h, local_h, context_h], axis=1)
        hidden = gelu_numpy(hidden @ state["head.0.weight"].T + state["head.0.bias"])
        self.hidden = layer_norm_numpy(
            hidden, state["head.2.weight"], state["head.2.bias"]
        ).astype(np.float32)
        self.final_weight = state["head.3.weight"]
        self.final_bias = state["head.3.bias"]
        self.target_mean = np.asarray(payload["target_mean"], dtype=np.float32)
        self.target_scale = np.asarray(payload["target_scale"], dtype=np.float32)

    def predict(self, genes: np.ndarray) -> np.ndarray:
        local = self.position[genes]
        keep = local >= 0
        output = np.zeros((len(self.hidden), len(genes)), dtype=np.float32)
        if keep.any():
            values = (
                self.hidden @ self.final_weight[local[keep]].T
                + self.final_bias[local[keep]]
            )
            values = (
                values * self.target_scale[local[keep]]
                + self.target_mean[local[keep]]
            )
            output[:, keep] = values.astype(np.float32)
        return output


def prepare_predictors(root: Path, sample: str, scale: int, windows: pd.DataFrame, n_genes: int):
    feature_root = root / "cache" / "crc_features" / sample / "PHIKON"
    local_raw = np.load(feature_root / f"{scale}um__dynamic_local.npy", mmap_mode="r")
    context_raw = np.load(feature_root / f"{scale}um__dynamic_context.npy", mmap_mode="r")
    task_root = root / "results" / "model_fastlane_checkpoints" / f"{sample}_{scale}um_discovery3_v1"
    pca = np.load(task_root / "phikon_pca.npz", allow_pickle=False)
    raw_joined = np.concatenate(
        [np.asarray(local_raw, dtype=np.float32), np.asarray(context_raw, dtype=np.float32)],
        axis=1,
    )
    morphology = (
        (raw_joined - pca["mean"]) @ pca["components"].T
        / np.sqrt(pca["explained_variance"])
    ).astype(np.float32)
    ridge: list[RidgeFold] = []
    local_context: list[LocalContextFold] = []
    for fold in range(3):
        comp = composition_features(
            root / "cache" / "composition" / sample / f"{scale}um" / f"fold{fold}.parquet",
            windows,
        )
        ridge.append(RidgeFold(task_root / f"fold{fold}.npz", comp, morphology, n_genes))
        if scale in (32, 64):
            local_context.append(
                LocalContextFold(
                    root
                    / "results"
                    / "neural_ladder_checkpoints"
                    / f"{sample}_{scale}um_fold{fold}"
                    / "LOCAL_CONTEXT_NETWORK.pt",
                    comp,
                    np.asarray(local_raw, dtype=np.float32),
                    np.asarray(context_raw, dtype=np.float32),
                    n_genes,
                )
            )
    return ridge, local_context, pca, morphology


def source_half_transform(
    counts: sparse.csr_matrix, area: np.ndarray, seed: int
) -> sparse.csr_matrix:
    rng = np.random.default_rng(seed)
    source = counts.copy().astype(np.float64)
    source.data = rng.binomial(
        source.data.astype(np.int64), 0.5
    ).astype(np.float64)
    row_index = np.repeat(
        np.arange(source.shape[0]), np.diff(source.indptr)
    )
    source.data = np.sqrt(
        source.data / np.maximum(0.5 * area[row_index], 1e-12)
    )
    source.eliminate_zeros()
    return source.tocsr()


def morphology_from_pca(
    root: Path, sample: str, scale: int, pca: Any
) -> np.ndarray:
    feature_root = root / "cache" / "crc_features" / sample / "PHIKON"
    local = np.load(
        feature_root / f"{scale}um__dynamic_local.npy", mmap_mode="r"
    )
    context = np.load(
        feature_root / f"{scale}um__dynamic_context.npy", mmap_mode="r"
    )
    joined = np.concatenate(
        [np.asarray(local, dtype=np.float32), np.asarray(context, dtype=np.float32)],
        axis=1,
    )
    return (
        (joined - pca["mean"]) @ pca["components"].T
        / np.sqrt(pca["explained_variance"])
    ).astype(np.float32)


def fit_image_only_prediction(
    root: Path,
    heldout: str,
    scale: int,
    panel: np.ndarray,
    genes: pd.DataFrame,
    ridge: list[RidgeFold],
    pca: Any,
    test_morphology: np.ndarray,
) -> np.ndarray:
    train_samples = [sample for sample in SAMPLES if sample != heldout]
    train_x_parts: list[np.ndarray] = []
    train_y_parts: list[sparse.csr_matrix] = []
    for sample in train_samples:
        grid_root = root / "cache" / "crc_grids" / sample
        windows = pq.read_table(
            grid_root / f"{scale}um" / "window_axis.parquet"
        ).to_pandas()
        counts = sparse.load_npz(
            grid_root / f"{scale}um" / "counts_windows_by_genes.npz"
        ).tocsr()[:, panel]
        train_y_parts.append(
            source_half_transform(
                counts,
                windows["effective_area_um2"].to_numpy(np.float64),
                SPLIT_SEEDS[0],
            )
        )
        train_x_parts.append(morphology_from_pca(root, sample, scale, pca))
    train_x_raw = np.concatenate(train_x_parts, axis=0)
    train_y = sparse.vstack(train_y_parts, format="csr")
    output = np.zeros((len(test_morphology), len(panel)), dtype=np.float32)
    panel_folds = genes.iloc[panel]["evaluation_fold"].to_numpy(np.int64)
    for fold in range(3):
        keep = np.flatnonzero(panel_folds == fold)
        if not len(keep):
            continue
        mean_x, scale_x, _, _, low_x, high_x = ridge[fold].params[
            "MORPHOLOGY_RIDGE"
        ]
        comp_dim = ridge[fold].x["COMPOSITION_RIDGE"].shape[1]
        mean_m = mean_x[comp_dim:]
        scale_m = scale_x[comp_dim:]
        low_m = low_x[comp_dim:]
        high_m = high_x[comp_dim:]
        x_train = (
            (np.clip(train_x_raw, low_m, high_m) - mean_m) / scale_m
        ).astype(np.float64)
        x_test = (
            (np.clip(test_morphology, low_m, high_m) - mean_m) / scale_m
        ).astype(np.float64)
        target = train_y[:, keep].tocsr()
        mean_y = np.asarray(target.mean(axis=0)).ravel().astype(np.float64)
        gram = x_train.T @ x_train
        gram.flat[:: len(gram) + 1] += ridge[fold].alpha_morphology
        rhs = np.asarray(target.T @ x_train, dtype=np.float64).T
        weight = np.linalg.solve(gram, rhs)
        output[:, keep] = (x_test @ weight + mean_y).astype(np.float32)
    return output


def predict_block(
    ridge: list[RidgeFold],
    local_context: list[LocalContextFold],
    model: str,
    genes: np.ndarray,
) -> np.ndarray:
    rows = len(ridge[0].x["COMPOSITION_RIDGE"])
    output = np.zeros((rows, len(genes)), dtype=np.float32)
    if model == "LOCAL_CONTEXT_NETWORK":
        for predictor in local_context:
            values = predictor.predict(genes)
            keep = predictor.position[genes] >= 0
            output[:, keep] = values[:, keep]
    else:
        for predictor in ridge:
            values = predictor.predict(model, genes)
            keep = predictor.position[genes] >= 0
            output[:, keep] = values[:, keep]
    return output


def normalize_prediction(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float64, copy=False)
    values = values - values.mean(axis=0, keepdims=True)
    norm = np.sqrt(np.square(values).sum(axis=0))
    keep = norm > 1e-12
    values[:, keep] /= norm[keep]
    values[:, ~keep] = 0.0
    return values.astype(np.float32)


def split_transform(
    block: sparse.csr_matrix,
    area: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    a = block.copy().astype(np.float64)
    a.data = rng.binomial(a.data.astype(np.int64), 0.5).astype(np.float64)
    b = block.astype(np.float64) - a
    for matrix in (a, b):
        row_index = np.repeat(np.arange(matrix.shape[0]), np.diff(matrix.indptr))
        matrix.data = np.sqrt(
            matrix.data / np.maximum(0.5 * area[row_index], 1e-12)
        )
        matrix.eliminate_zeros()
    aa = a.toarray()
    bb = b.toarray()
    aa -= aa.mean(axis=0, keepdims=True)
    bb -= bb.mean(axis=0, keepdims=True)
    scale = np.sqrt(0.5 * (np.square(aa).sum(axis=0) + np.square(bb).sum(axis=0)))
    keep = scale > 1e-12
    aa[:, keep] /= scale[keep]
    bb[:, keep] /= scale[keep]
    aa[:, ~keep] = 0.0
    bb[:, ~keep] = 0.0
    return aa.astype(np.float32), bb.astype(np.float32)


def grid_contract(windows: pd.DataFrame, scale: int):
    x = windows["center_x_um"].to_numpy(np.float64)
    y = windows["center_y_um"].to_numpy(np.float64)
    ix = np.rint((x - x.min()) / scale).astype(np.int64)
    iy = np.rint((y - y.min()) / scale).astype(np.int64)
    nx, ny = int(ix.max() + 1), int(iy.max() + 1)
    if len(np.unique(iy * nx + ix)) != len(windows):
        raise RuntimeError("CRC grid coordinates are not unique")
    fx = np.fft.fftfreq(nx, d=float(scale))
    fy = np.fft.fftfreq(ny, d=float(scale))
    radial = np.sqrt(np.square(fy[:, None]) + np.square(fx[None, :]))
    wavelength = np.divide(
        1.0,
        radial,
        out=np.full_like(radial, np.inf),
        where=radial > 0,
    )
    masks: dict[str, np.ndarray] = {}
    for name, low, high in WAVELENGTH_BANDS:
        masks[name] = (radial > 0) & (wavelength >= low) & (wavelength < high)
    if np.any(np.sum(np.stack(list(masks.values())), axis=0) != (radial > 0)):
        raise RuntimeError("Physical wavelength windows do not partition non-DC modes")
    return iy, ix, ny, nx, masks


def fft_fields(values: np.ndarray, iy: np.ndarray, ix: np.ndarray, ny: int, nx: int):
    grid = np.zeros((ny, nx, values.shape[1]), dtype=np.float32)
    grid[iy, ix, :] = values
    return spfft.fft2(grid, axes=(0, 1), norm="ortho", workers=2)


def task(sample: str, scale: int, root_text: str, panel: list[int]) -> list[dict[str, Any]]:
    root = Path(root_text)
    grid_root = root / "cache" / "crc_grids" / sample
    genes = pq.read_table(grid_root / "gene_axis.parquet").to_pandas()
    windows = pq.read_table(grid_root / f"{scale}um" / "window_axis.parquet").to_pandas()
    counts = sparse.load_npz(grid_root / f"{scale}um" / "counts_windows_by_genes.npz").tocsr()
    panel_array = np.asarray(panel, dtype=np.int64)
    ridge, lcn, pca, test_morphology = prepare_predictors(
        root, sample, scale, windows, len(genes)
    )
    iy, ix, ny, nx, masks = grid_contract(windows, scale)
    models = (
        "IMAGE_ONLY_PHIKON_RIDGE",
        "COMPOSITION_RIDGE",
        "MORPHOLOGY_RIDGE",
        "FROZEN_ENVELOPE",
    )
    terms: dict[tuple[str, str], dict[str, float]] = {
        (model, band): {"v_rna": 0.0, "c_model": 0.0, "v_model": 0.0}
        for model in models
        for band in masks
    }
    mode_counts = {name: int(mask.sum()) for name, mask in masks.items()}
    area = windows["effective_area_um2"].to_numpy(np.float64)
    selected_model = FROZEN_MODEL_BY_SCALE[scale]
    image_only_full = fit_image_only_prediction(
        root,
        sample,
        scale,
        panel_array,
        genes,
        ridge,
        pca,
        test_morphology,
    )

    for offset in range(0, len(panel_array), GENE_BLOCK):
        block_genes = panel_array[offset : offset + GENE_BLOCK]
        count_block = counts[:, block_genes].tocsr()
        prediction_values = {
            "IMAGE_ONLY_PHIKON_RIDGE": image_only_full[:, offset : offset + len(block_genes)],
            "COMPOSITION_RIDGE": predict_block(ridge, lcn, "COMPOSITION_RIDGE", block_genes),
            "MORPHOLOGY_RIDGE": predict_block(ridge, lcn, "MORPHOLOGY_RIDGE", block_genes),
        }
        if selected_model in prediction_values:
            prediction_values["FROZEN_ENVELOPE"] = prediction_values[selected_model]
        else:
            prediction_values["FROZEN_ENVELOPE"] = predict_block(
                ridge, lcn, selected_model, block_genes
            )
        fp = {}
        for model, values in prediction_values.items():
            if model == "FROZEN_ENVELOPE" and selected_model in fp:
                fp[model] = fp[selected_model]
            else:
                fp[model] = fft_fields(
                    normalize_prediction(values), iy, ix, ny, nx
                )
        for model in models:
            for band, mask in masks.items():
                terms[(model, band)]["v_model"] += float(
                    np.square(np.abs(fp[model][mask, :])).sum()
                )

        seed_vrna = {band: 0.0 for band in masks}
        seed_c = {(model, band): 0.0 for model in models for band in masks}
        for seed in SPLIT_SEEDS:
            aa, bb = split_transform(count_block, area, seed + offset * 1009)
            fa = fft_fields(aa, iy, ix, ny, nx)
            fb = fft_fields(bb, iy, ix, ny, nx)
            for band, mask in masks.items():
                seed_vrna[band] += float(
                    np.real(np.conj(fa[mask, :]) * fb[mask, :]).sum()
                )
                judge_sum = fa[mask, :] + fb[mask, :]
                for model in models:
                    seed_c[(model, band)] += float(
                        0.5
                        * np.real(np.conj(fp[model][mask, :]) * judge_sum).sum()
                    )
        for band in masks:
            vrna = seed_vrna[band] / len(SPLIT_SEEDS)
            for model in models:
                terms[(model, band)]["v_rna"] += vrna
                terms[(model, band)]["c_model"] += (
                    seed_c[(model, band)] / len(SPLIT_SEEDS)
                )

    rows: list[dict[str, Any]] = []
    for model in models:
        total = {key: sum(terms[(model, band)][key] for band in masks) for key in ("v_rna", "c_model", "v_model")}
        for band, low, high in WAVELENGTH_BANDS:
            values = terms[(model, band)]
            denominator = math.sqrt(max(values["v_rna"], 0.0) * max(values["v_model"], 0.0))
            q = values["c_model"] / denominator if denominator > 0 else math.nan
            rows.append(
                {
                    "sample": sample,
                    "scale_um": scale,
                    "model": model,
                    "frozen_model_at_scale": selected_model,
                    "wavelength_band": band,
                    "wavelength_low_um": low,
                    "wavelength_high_um": high,
                    "fourier_modes": mode_counts[band],
                    "nyquist_status": (
                        "FULL_PHYSICAL_WINDOW"
                        if low >= 2.0 * scale
                        else "PARTIAL_HIGH_FREQUENCY_WINDOW"
                    ),
                    "genes": len(panel_array),
                    "molecule_split_seeds": len(SPLIT_SEEDS),
                    "v_rna": values["v_rna"],
                    "c_model": values["c_model"],
                    "v_model": values["v_model"],
                    "q": q,
                    "rna_signal_fraction": values["v_rna"] / total["v_rna"] if total["v_rna"] > 0 else math.nan,
                }
            )
        denominator = math.sqrt(max(total["v_rna"], 0.0) * max(total["v_model"], 0.0))
        rows.append(
            {
                "sample": sample,
                "scale_um": scale,
                "model": model,
                "frozen_model_at_scale": selected_model,
                "wavelength_band": "ALL_NON_DC",
                "wavelength_low_um": 0.0,
                "wavelength_high_um": math.inf,
                "fourier_modes": sum(mode_counts.values()),
                "nyquist_status": "ALL_OBSERVED_NON_DC_MODES",
                "genes": len(panel_array),
                "molecule_split_seeds": len(SPLIT_SEEDS),
                "v_rna": total["v_rna"],
                "c_model": total["c_model"],
                "v_model": total["v_model"],
                "q": total["c_model"] / denominator if denominator > 0 else math.nan,
                "rna_signal_fraction": 1.0,
            }
        )
    return rows


def select_panel(root: Path) -> tuple[list[int], pd.DataFrame]:
    axes = [
        pq.read_table(root / "cache" / "crc_grids" / sample / "gene_axis.parquet").to_pandas()
        for sample in SAMPLES
    ]
    gene_ids = axes[0]["gene_id"].astype(str).to_numpy()
    if not all(np.array_equal(gene_ids, axis["gene_id"].astype(str).to_numpy()) for axis in axes[1:]):
        raise RuntimeError("CRC gene axes differ across samples")
    abundance = np.zeros(len(gene_ids), dtype=np.float64)
    for sample in SAMPLES:
        counts = sparse.load_npz(
            root / "cache" / "crc_grids" / sample / "32um" / "counts_windows_by_genes.npz"
        )
        abundance += np.asarray(counts.sum(axis=0)).ravel()
    eligible = np.flatnonzero(axes[0]["protein_coding"].to_numpy(bool) & (abundance > 0))
    selected = eligible[np.argsort(-abundance[eligible], kind="stable")[:GENE_PANEL_SIZE]]
    panel = axes[0].iloc[selected][["gene_id", "gene_symbol", "protein_coding", "evaluation_fold"]].copy()
    panel.insert(0, "gene_index", selected)
    panel["total_umi_at_32um_across_samples"] = abundance[selected]
    panel["selection_rank"] = np.arange(1, len(panel) + 1)
    return selected.tolist(), panel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    root = args.asset_root / "payload"
    args.out.mkdir(parents=True, exist_ok=True)
    panel, panel_frame = select_panel(root)
    panel_frame.to_csv(args.out / "CRC_PHYSICAL_GENE_PANEL.tsv", sep="\t", index=False)

    all_rows: list[dict[str, Any]] = []
    task_root = args.out / "tasks"
    task_root.mkdir(parents=True, exist_ok=True)
    futures = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for sample in SAMPLES:
            for scale in reversed(SCALES):
                task_path = task_root / f"{sample}_{scale}um.tsv"
                if task_path.is_file():
                    all_rows.extend(pd.read_csv(task_path, sep="\t").to_dict(orient="records"))
                    print(f"G0_048_TASK=REUSED sample={sample} scale_um={scale}", flush=True)
                else:
                    future = pool.submit(task, sample, scale, str(root), panel)
                    futures[future] = (sample, scale, task_path)
        for future in as_completed(futures):
            sample, scale, task_path = futures[future]
            rows = future.result()
            pd.DataFrame(rows).to_csv(task_path, sep="\t", index=False)
            all_rows.extend(rows)
            print(f"G0_048_TASK=PASS sample={sample} scale_um={scale}", flush=True)

    frame = pd.DataFrame(all_rows).sort_values(
        ["sample", "scale_um", "model", "wavelength_low_um"], kind="stable"
    )
    frame.to_csv(args.out / "CRC_PHYSICAL_WAVELENGTH_Q.tsv", sep="\t", index=False)

    summary = (
        frame.groupby(["scale_um", "model", "wavelength_band"], as_index=False)
        .agg(
            q_mean=("q", "mean"),
            q_min=("q", "min"),
            q_max=("q", "max"),
            rna_signal_fraction_mean=("rna_signal_fraction", "mean"),
            samples=("sample", "nunique"),
        )
    )
    summary.to_csv(args.out / "CRC_PHYSICAL_WAVELENGTH_Q_SUMMARY.tsv", sep="\t", index=False)

    total = frame[frame["wavelength_band"] == "ALL_NON_DC"].copy()
    total[["sampling_fraction", "composition_reference_fraction", "signed_model_increment", "unattributed_fraction"]] = total["scale_um"].map(SCALE_CONTEXT).apply(pd.Series)
    total.to_csv(args.out / "CRC_PHYSICAL_SCALE_BRIDGE.tsv", sep="\t", index=False)

    frozen = total[total["model"] == "FROZEN_ENVELOPE"]
    headline = (
        frozen.groupby("scale_um", as_index=False)
        .agg(q_mean=("q", "mean"), q_min=("q", "min"), q_max=("q", "max"))
        .sort_values("scale_um")
    )
    headline.to_csv(args.out / "CRC_FROZEN_ENVELOPE_Q_BY_SCALE.tsv", sep="\t", index=False)
    report = {
        "run_id": "G0-048",
        "status": "PASS",
        "samples": list(SAMPLES),
        "scales_um": list(SCALES),
        "gene_panel_size": len(panel),
        "molecule_split_seeds": list(SPLIT_SEEDS),
        "models": [
            "IMAGE_ONLY_PHIKON_RIDGE",
            "COMPOSITION_RIDGE",
            "MORPHOLOGY_RIDGE",
            "FROZEN_ENVELOPE",
        ],
        "wavelength_bands": [list(x) for x in WAVELENGTH_BANDS],
        "tasks_complete": int(frame[["sample", "scale_um"]].drop_duplicates().shape[0]),
        "rows": int(len(frame)),
        "headline_frozen_q": headline.to_dict(orient="records"),
    }
    (args.out / "CRC_PHYSICAL_SPECTRUM_RUN.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("G0_048_CRC_PHYSICAL_SPECTRUM=PASS tasks=24", flush=True)


if __name__ == "__main__":
    main()
