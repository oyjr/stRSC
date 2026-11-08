"""RNA-referenced Fourier scores on regular spatial grids.

Physical-band transforms are retained from the original CRC scorer.
"""
from __future__ import annotations
import argparse
import math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from scipy import fft as spfft


SPLIT_SEEDS = tuple(range(2026073101, 2026073106))

GENE_BLOCK = 32

WAVELENGTH_BANDS = (
    ("LT128", 0.0, 128.0),
    ("128_256", 128.0, 256.0),
    ("256_512", 256.0, 512.0),
    ("512_1024", 512.0, 1024.0),
    ("1024_2048", 1024.0, 2048.0),
    ("2048_4096", 2048.0, 4096.0),
    ("GE4096", 4096.0, math.inf),
)


def normalize_prediction(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float64, copy=False)
    values = values - values.mean(axis=0, keepdims=True)
    norm = np.sqrt(np.square(values).sum(axis=0))
    keep = norm > 1e-12
    values[:, keep] /= norm[keep]
    values[:, ~keep] = 0.0
    return values.astype(np.float32)


def split_transform(
    block: sparse.csr_matrix, area: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    a = block.copy().astype(np.float64)
    a.data = rng.binomial(a.data.astype(np.int64), 0.5).astype(np.float64)
    b = block.astype(np.float64) - a
    for matrix in (a, b):
        row_index = np.repeat(np.arange(matrix.shape[0]), np.diff(matrix.indptr))
        matrix.data = np.sqrt(matrix.data / np.maximum(0.5 * area[row_index], 1e-12))
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
    return (aa.astype(np.float32), bb.astype(np.float32))


def grid_contract(windows: pd.DataFrame, scale: int):
    x = windows["center_x_um"].to_numpy(np.float64)
    y = windows["center_y_um"].to_numpy(np.float64)
    ix = np.rint((x - x.min()) / scale).astype(np.int64)
    iy = np.rint((y - y.min()) / scale).astype(np.int64)
    nx, ny = (int(ix.max() + 1), int(iy.max() + 1))
    if len(np.unique(iy * nx + ix)) != len(windows):
        raise RuntimeError("CRC grid coordinates are not unique")
    fx = np.fft.fftfreq(nx, d=float(scale))
    fy = np.fft.fftfreq(ny, d=float(scale))
    radial = np.sqrt(np.square(fy[:, None]) + np.square(fx[None, :]))
    wavelength = np.divide(
        1.0, radial, out=np.full_like(radial, np.inf), where=radial > 0
    )
    masks: dict[str, np.ndarray] = {}
    for name, low, high in WAVELENGTH_BANDS:
        masks[name] = (radial > 0) & (wavelength >= low) & (wavelength < high)
    if np.any(np.sum(np.stack(list(masks.values())), axis=0) != (radial > 0)):
        raise RuntimeError("Physical wavelength windows do not partition non-DC modes")
    return (iy, ix, ny, nx, masks)


def fft_fields(values: np.ndarray, iy: np.ndarray, ix: np.ndarray, ny: int, nx: int):
    grid = np.zeros((ny, nx, values.shape[1]), dtype=np.float32)
    grid[iy, ix, :] = values
    return spfft.fft2(grid, axes=(0, 1), norm="ortho", workers=2)


def score(
    counts, prediction, coords, area, *, spacing_um, methods=None, seeds=SPLIT_SEEDS
):
    """Score fixed predictions from raw counts on one specimen's regular grid."""
    counts = sparse.csr_matrix(counts)
    if counts.ndim != 2 or min(counts.shape) < 1:
        raise ValueError("Counts must have shape (occupied sites, genes)")
    if not np.issubdtype(counts.dtype, np.integer) or np.any(counts.data < 0):
        raise ValueError("Counts must be nonnegative integers before normalization")
    counts.sum_duplicates()
    counts.eliminate_zeros()
    counts.sort_indices()
    n, genes = counts.shape
    prediction = np.asarray(prediction)
    if prediction.ndim == 2:
        prediction = prediction[:, :, None]
    if (
        prediction.ndim != 3
        or prediction.shape[:2] != (n, genes)
        or prediction.shape[2] < 1
    ):
        raise ValueError("Predictions must have shape (sites, genes[, models])")
    if not np.isfinite(prediction).all():
        raise ValueError(
            "Provide finite predictions on the common occupied-site and gene axes"
        )
    methods = (
        tuple(map(str, methods))
        if methods is not None
        else tuple(f"model_{i + 1}" for i in range(prediction.shape[2]))
    )
    if len(methods) != prediction.shape[2] or len(set(methods)) != len(methods):
        raise ValueError("Provide one unique name per model")
    coords, area = np.asarray(coords, dtype=float), np.asarray(area, dtype=float)
    if coords.shape != (n, 2) or not np.isfinite(coords).all():
        raise ValueError("Coordinates must be finite with shape (sites, 2)")
    if area.shape != (n,) or not np.isfinite(area).all() or np.any(area <= 0):
        raise ValueError(
            "Each occupied site needs a positive effective tissue area in square micrometres"
        )
    if not math.isfinite(float(spacing_um)) or spacing_um <= 0:
        raise ValueError("Grid spacing must be positive, in micrometres")
    scaled = (coords - coords.min(axis=0)) / spacing_um
    if not np.allclose(scaled, np.rint(scaled), rtol=0, atol=1e-6):
        raise ValueError("Coordinates do not lie on the specified regular grid")
    seeds = tuple(seeds)
    if (
        not seeds
        or any(not isinstance(s, (int, np.integer)) or s < 0 for s in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("Provide distinct nonnegative integer molecule-split seeds")

    windows = pd.DataFrame({"center_x_um": coords[:, 0], "center_y_um": coords[:, 1]})
    iy, ix, ny, nx, masks = grid_contract(windows, spacing_um)
    terms = {
        (model, band): {"v_rna": 0.0, "c_model": 0.0, "v_model": 0.0}
        for model in methods
        for band in masks
    }

    # Preserve the original gene blocks and their split-seed offsets.
    for offset in range(0, genes, GENE_BLOCK):
        stop = min(offset + GENE_BLOCK, genes)
        block = counts[:, offset:stop].tocsr()
        fp = {
            model: fft_fields(
                normalize_prediction(prediction[:, offset:stop, j]), iy, ix, ny, nx
            )
            for j, model in enumerate(methods)
        }
        for model in methods:
            for band, mask in masks.items():
                terms[model, band]["v_model"] += float(
                    np.square(np.abs(fp[model][mask, :])).sum()
                )
        seed_vrna = {band: 0.0 for band in masks}
        seed_c = {(model, band): 0.0 for model in methods for band in masks}
        for seed in seeds:
            aa, bb = split_transform(block, area, int(seed) + offset * 1009)
            fa, fb = (fft_fields(values, iy, ix, ny, nx) for values in (aa, bb))
            for band, mask in masks.items():
                seed_vrna[band] += float(
                    np.real(np.conj(fa[mask, :]) * fb[mask, :]).sum()
                )
                judge_sum = fa[mask, :] + fb[mask, :]
                for model in methods:
                    seed_c[model, band] += float(
                        0.5 * np.real(np.conj(fp[model][mask, :]) * judge_sum).sum()
                    )
        for band in masks:
            for model in methods:
                terms[model, band]["v_rna"] += seed_vrna[band] / len(seeds)
                terms[model, band]["c_model"] += seed_c[model, band] / len(seeds)

    rows = []
    for model in methods:
        total = {
            key: sum(terms[model, band][key] for band in masks)
            for key in ("v_rna", "c_model", "v_model")
        }
        for band, low, high in (*WAVELENGTH_BANDS, ("ALL_NON_DC", 0.0, math.inf)):
            values = total if band == "ALL_NON_DC" else terms[model, band]
            modes = (
                sum(int(mask.sum()) for mask in masks.values())
                if band == "ALL_NON_DC"
                else int(masks[band].sum())
            )
            denominator = math.sqrt(
                max(values["v_rna"], 0.0) * max(values["v_model"], 0.0)
            )
            status = (
                "NO_FOURIER_MODES"
                if modes == 0
                else "NONPOSITIVE_RNA_SIGNAL"
                if values["v_rna"] <= 0
                else "ZERO_MODEL_VARIANCE"
                if values["v_model"] <= 0
                else "ESTIMABLE"
            )
            rows.append(
                {
                    "model": model,
                    "scale_um": spacing_um,
                    "wavelength_band": band,
                    "wavelength_low_um": low,
                    "wavelength_high_um": high,
                    "fourier_modes": modes,
                    "genes": genes,
                    "molecule_split_seeds": len(seeds),
                    "occupied_sites": n,
                    "grid_rows": ny,
                    "grid_columns": nx,
                    "nyquist_status": "ALL_OBSERVED_NON_DC_MODES"
                    if band == "ALL_NON_DC"
                    else "FULL_PHYSICAL_WINDOW"
                    if low >= 2.0 * spacing_um
                    else "PARTIAL_HIGH_FREQUENCY_WINDOW",
                    **values,
                    "rsc": values["c_model"] / denominator
                    if denominator > 0
                    else math.nan,
                    "rna_signal_fraction": 1.0
                    if band == "ALL_NON_DC"
                    else values["v_rna"] / total["v_rna"]
                    if total["v_rna"] > 0
                    else math.nan,
                    "status": status,
                }
            )
    return pd.DataFrame(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="NPZ with counts, prediction, coords and area",
    )
    parser.add_argument("--spacing-um", type=float, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=SPLIT_SEEDS)
    parser.add_argument("--out", type=Path, required=True, help="Output TSV file")
    args = parser.parse_args(argv)
    with np.load(args.input, allow_pickle=False) as data:
        table = score(
            **{key: data[key] for key in ("counts", "prediction", "coords", "area")},
            methods=data["methods"] if "methods" in data else None,
            spacing_um=args.spacing_um,
            seeds=args.seeds,
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, sep="\t", index=False)
    print(f"Wrote {len(table)} band scores to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
