"""Coordinate-graph spectra and molecule splitting for the HEST protocol."""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

TASKS = {
    "IDC": ("TENX99", "TENX95", "NCBI785", "NCBI783"),
    "PAAD": ("TENX116", "TENX140", "TENX126"),
    "LUNG": ("TENX141", "TENX118"),
}
EDGE_MULTIPLIER = 1.25
MIN_COMPONENT = 20
BANDS = 5
THINNING_SEEDS = tuple(range(2026080701, 2026080721))
NUMERIC_TOL = 1e-12
TRUTH_TRANSFORM_ID = "HEST_OFFICIAL_LOG1P_RAW_INTEGER_COUNTS__HALVES_LOG1P_2X"
INPUT = Path("inputs")
BASIS = Path("basis")


class CalibrationError(RuntimeError):
    """A frozen calibration input or identity failed."""


def _seed(*parts: object) -> int:
    digest = hashlib.sha256("\x1f".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


def _sha256(path: Path, chunk: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def _atomic_tsv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    frame.to_csv(tmp, sep="\t", index=False, lineterminator="\n")
    os.replace(tmp, path)


def _save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with tmp.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _graph(xy: np.ndarray) -> tuple[Any, np.ndarray, np.ndarray, float, float]:
    tree = cKDTree(xy)
    nearest = tree.query(xy, k=2)[0][:, 1]
    positive = nearest[np.isfinite(nearest) & (nearest > 0)]
    if not len(positive):
        raise CalibrationError("NO_POSITIVE_NEAREST_DISTANCE")
    d0 = float(np.median(positive))
    radius = EDGE_MULTIPLIER * d0
    pairs = tree.query_pairs(radius, output_type="ndarray")
    if not len(pairs):
        raise CalibrationError("GRAPH_NO_EDGES")
    adjacency = coo_matrix(
        (np.ones(2 * len(pairs), np.float64),
         (np.r_[pairs[:, 0], pairs[:, 1]], np.r_[pairs[:, 1], pairs[:, 0]])),
        shape=(len(xy), len(xy)),
    ).tocsr()
    n_raw, labels_raw = connected_components(adjacency, directed=False)
    sizes = np.bincount(labels_raw, minlength=n_raw)
    keep = sizes[labels_raw] >= MIN_COMPONENT
    kept = np.flatnonzero(keep)
    adjacency = adjacency[kept][:, kept]
    _, labels = connected_components(adjacency, directed=False)
    return adjacency, keep, labels, d0, radius


def _basis_dir(task: str, specimen: str) -> Path:
    return BASIS / task / specimen


def build_basis(task: str, specimen: str, device: str) -> None:
    """Build the exact full coordinate graph spectrum on a GPU eigensolver."""
    if specimen not in TASKS[task]:
        raise CalibrationError(f"UNFROZEN_SPECIMEN:{task}:{specimen}")
    import torch

    payload = np.load(INPUT / task / f"{specimen}.npz", allow_pickle=False)
    xy = payload["xy"]
    adjacency, keep, labels, d0, radius = _graph(xy)
    kept_n = int(keep.sum())
    component_ids = sorted(np.unique(labels).tolist())
    component_sizes = [int(np.sum(labels == component)) for component in component_ids]
    totals = np.zeros(BANDS, dtype=np.int64)
    chunks_by_component: dict[int, list[np.ndarray]] = {}
    for component, size in zip(component_ids, component_sizes):
        chunks = np.array_split(np.arange(max(size - 1, 0)), BANDS)
        chunks_by_component[component] = chunks
        totals += np.asarray([len(chunk) for chunk in chunks])
    out = _basis_dir(task, specimen)
    if (out / "receipt.json").is_file():
        prior = json.loads((out / "receipt.json").read_text())
        if prior.get("status") == "PASS" and prior.get("spots_retained") == kept_n:
            print(json.dumps({"status": "PASS_REUSED", "task": task, "specimen": specimen}))
            return
        raise CalibrationError(f"INVALID_EXISTING_BASIS:{out}")
    out.mkdir(parents=True, exist_ok=True)
    # A pre-effect worker interruption may leave only these owned temporary
    # memmaps. They are never a checkpoint and are replaced atomically.
    for band_index in range(1, BANDS + 1):
        (out / f"U5_B{band_index}.npy.tmp").unlink(missing_ok=True)
    _save_npy(out / "keep.npy", keep.astype(bool))
    _save_npy(out / "component_labels.npy", labels.astype(np.int32))
    band_maps = [
        np.lib.format.open_memmap(out / f"U5_B{k+1}.npy.tmp", mode="w+", dtype=np.float64, shape=(kept_n, int(totals[k])))
        for k in range(BANDS)
    ]
    offsets = np.zeros(BANDS, dtype=np.int64)
    metadata: list[dict[str, Any]] = []
    reconstruction_error = 0.0
    eigen_residual = 0.0
    orthogonality_probe = 0.0
    started = time.time()
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and torch_device.index is None:
        # Each scheduler subprocess owns exactly one CUDA_VISIBLE_DEVICES entry.
        # Torch requires the explicit process-local index for set_device.
        torch_device = torch.device("cuda:0")
    if torch_device.type == "cuda":
        torch.cuda.set_device(torch_device)
        torch.cuda.reset_peak_memory_stats(torch_device)
    for component in component_ids:
        local = np.flatnonzero(labels == component)
        block = adjacency[local][:, local]
        pairs = np.column_stack(block.nonzero())
        laplacian = torch.zeros((len(local), len(local)), dtype=torch.float64, device=torch_device)
        if len(pairs):
            index = torch.as_tensor(pairs.T, dtype=torch.long, device=torch_device)
            laplacian[index[0], index[1]] = -1.0
        degree = torch.as_tensor(np.asarray(block.sum(axis=1)).reshape(-1), dtype=torch.float64, device=torch_device)
        laplacian.diagonal().copy_(degree)
        values, vectors = torch.linalg.eigh(laplacian, UPLO="L")
        order = torch.argsort(values, stable=True)
        values = values[order]
        vectors = vectors[:, order]
        # The component constant is the sole null mode under the frozen graph.
        nonzero_values = values[1:]
        nonzero_vectors = vectors[:, 1:]
        probe = torch.randn((len(local), 4), dtype=torch.float64, device=torch_device,
                            generator=torch.Generator(device=torch_device).manual_seed(_seed("basis", task, specimen, component)))
        centered = probe - probe.mean(dim=0, keepdim=True)
        rebuilt = nonzero_vectors @ (nonzero_vectors.T @ centered)
        reconstruction_error = max(reconstruction_error, float(torch.max(torch.abs(rebuilt - centered)).cpu()))
        sample_modes = torch.linspace(0, max(len(local) - 2, 0), min(16, max(len(local) - 1, 1)), device=torch_device).long()
        if len(local) > 1:
            selected = nonzero_vectors[:, sample_modes]
            residual = laplacian @ selected - selected * nonzero_values[sample_modes][None, :]
            eigen_residual = max(eigen_residual, float(torch.max(torch.abs(residual)).cpu()))
            gram = selected.T @ selected
            orthogonality_probe = max(orthogonality_probe, float(torch.max(torch.abs(gram - torch.eye(len(sample_modes), dtype=torch.float64, device=torch_device))).cpu()))
        chunks = chunks_by_component[component]
        for band_index, indices in enumerate(chunks):
            start = int(offsets[band_index]); end = start + len(indices)
            if len(indices):
                band_maps[band_index][local, start:end] = nonzero_vectors[:, indices].cpu().numpy()
                band_values = nonzero_values[indices].cpu().numpy()
                mean_degree = float(degree.mean().cpu())
                relative = (indices + 1.0) / max(len(nonzero_values), 1)
                theta = np.arccos(np.clip(1.0 - band_values / max(mean_degree, NUMERIC_TOL), -1.0, 1.0))
                wavelength = np.divide(2 * np.pi, theta, out=np.full_like(theta, np.inf), where=theta > 1e-12)
                metadata.append({
                    "task": task, "specimen": specimen, "component": component,
                    "component_spots": len(local), "band": f"B{band_index+1}", "modes": len(indices),
                    "eigen_min": float(band_values.min()), "eigen_max": float(band_values.max()),
                    "relative_frequency_min": float(relative.min()), "relative_frequency_max": float(relative.max()),
                    "approx_hop_wavelength_median": float(np.median(wavelength)), "mean_degree": mean_degree,
                })
            offsets[band_index] = end
        del laplacian, values, vectors, nonzero_values, nonzero_vectors
        if torch_device.type == "cuda":
            torch.cuda.empty_cache()
    for band_index, matrix in enumerate(band_maps, 1):
        matrix.flush()
        # Local portability only: Windows cannot rename an open memory map.
        matrix._mmap.close()
        del matrix
        os.replace(out / f"U5_B{band_index}.npy.tmp", out / f"U5_B{band_index}.npy")
    if reconstruction_error >= 1e-8 or orthogonality_probe >= 1e-8:
        raise CalibrationError(f"GRAPH_IDENTITY:{task}:{specimen}:{reconstruction_error}:{orthogonality_probe}")
    _atomic_tsv(out / "graph_bands.tsv", pd.DataFrame(metadata))
    files = [out / "keep.npy", out / "component_labels.npy", out / "graph_bands.tsv"] + [out / f"U5_B{k}.npy" for k in range(1, 6)]
    receipt = {
        "status": "PASS", "task": task, "specimen": specimen,
        "spots_input": len(xy), "spots_retained": kept_n, "coverage": kept_n / len(xy),
        "components": len(component_ids), "component_sizes": component_sizes,
        "d0": d0, "edge_radius": radius, "edge_multiplier": EDGE_MULTIPLIER,
        "laplacian": "COMBINATORIAL_L_EQUALS_D_MINUS_A", "band_modes": totals.tolist(),
        "reconstruction_probe_max_abs": reconstruction_error,
        "eigen_residual_probe_max_abs": eigen_residual,
        "orthogonality_probe_max_abs": orthogonality_probe,
        "device": device, "elapsed_seconds": time.time() - started,
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(torch_device)) if torch_device.type == "cuda" else 0,
        "files": [{"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)} for path in files],
        "prediction_files_opened": 0,
    }
    _atomic_json(out / "receipt.json", receipt)
    print(json.dumps({"status": "PASS", "task": task, "specimen": specimen, "spots": kept_n, "seconds": receipt["elapsed_seconds"]}))


def _halves(counts: np.ndarray, task: str, specimen: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(_seed("G0-047", task, specimen, seed))
    a_raw = rng.binomial(counts, 0.5).astype(np.float64)
    b_raw = counts.astype(np.float64) - a_raw
    if not np.array_equal(a_raw + b_raw, counts):
        raise CalibrationError(f"MOLECULE_CONSERVATION:{task}:{specimen}:{seed}")
    a = np.log1p(2.0 * a_raw)
    b = np.log1p(2.0 * b_raw)
    a -= a.mean(axis=0, keepdims=True)
    b -= b.mean(axis=0, keepdims=True)
    return a, b


def _device_bands(bands: list[np.ndarray], device: str) -> list[Any]:
    """Materialize one specimen basis once per GPU process."""
    import torch

    return [torch.as_tensor(np.asarray(matrix), dtype=torch.float32, device=device) for matrix in bands]


def _coefficients(bands: list[Any], x: np.ndarray, device: str) -> list[np.ndarray]:
    import torch

    output: list[np.ndarray] = []
    tx = torch.as_tensor(x, dtype=torch.float32, device=device)
    for matrix in bands:
        u = matrix if isinstance(matrix, torch.Tensor) else torch.as_tensor(np.asarray(matrix), dtype=torch.float32, device=device)
        output.append((u.T @ tx).cpu().numpy().astype(np.float64))
    del tx
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return output

def _decode(values: Any) -> np.ndarray:
    return np.asarray([
        value.decode() if isinstance(value, (bytes, np.bytes_)) else str(value)
        for value in np.asarray(values).reshape(-1)
    ], dtype=object)


class ScoreError(RuntimeError):
    pass


def _corr_cols(prediction: np.ndarray, truth: np.ndarray) -> np.ndarray:
    prediction = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    if prediction.shape != truth.shape:
        raise ScoreError(f"PCC_SHAPE:{prediction.shape}:{truth.shape}")
    pc = prediction - prediction.mean(axis=0, keepdims=True)
    tc = truth - truth.mean(axis=0, keepdims=True)
    numerator = np.sum(pc * tc, axis=0)
    denominator = np.sqrt(np.sum(pc * pc, axis=0) * np.sum(tc * tc, axis=0))
    observed_variable = np.sum(tc * tc, axis=0) > NUMERIC_TOL
    out = np.full(prediction.shape[1], np.nan, dtype=np.float64)
    valid = observed_variable & (denominator > NUMERIC_TOL)
    out[valid] = numerator[valid] / denominator[valid]
    # Frozen ordinary-PCC rule: a constant prediction on variable truth is 0.
    out[observed_variable & ~valid] = 0.0
    return out
