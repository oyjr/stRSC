#!/usr/bin/env python3
"""G0-047 model-blind HEST spectrum calibration.

This module never opens an encoder embedding or a prediction.  It binds the
official patch-common support by barcode, builds the frozen coordinate-only
combinatorial-Laplacian bands, and performs task-specific raw-molecule
calibration for IDC, PAAD and LUNG.

The formal HEST transform is ``log1p(raw integer count)``.  A molecular half is
therefore mapped to full exposure as ``log1p(2 * half_count)``.  No library-size
normalisation is introduced into this task-specific benchmark transform.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import h5py
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

import g0046_hest as hest


ROOT = Path(os.environ.get("STRSC_HIST_DATA", Path(__file__).resolve().parent / "data"))
RUN = ROOT / "calibration_output"
WORK = ROOT / "calibration"
DATA = ROOT / "hest"
INPUT = WORK / "input"
BASIS = WORK / "basis"
MOMENTS = WORK / "moments"
SHARDS = WORK / "calibration_shards_v3_replicate_latent_oracle"
RESULT = RUN / "results/hest_calibration"
RECEIPT = RUN / "receipts/hest_calibration"

TASKS = hest.TASK_SAMPLES
EDGE_MULTIPLIER = 1.25
MIN_COMPONENT = 20
BANDS = 5
THINNING_SEEDS = tuple(range(2026080701, 2026080721))
CALIBRATION_REPLICATES = 200
RHO = (0.0, 0.3, 0.6, 1.0)
THETA = (0.0, math.pi / 8, math.pi / 4, 3 * math.pi / 8, math.pi / 2)
CALIBRATION_BIAS_LIMIT = 0.03
NUMERIC_TOL = 1e-12
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260807
TRUTH_TRANSFORM_ID = "HEST_OFFICIAL_LOG1P_RAW_INTEGER_COUNTS__HALVES_LOG1P_2X"


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


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    frame.to_parquet(tmp, index=False, compression="zstd")
    os.replace(tmp, path)


def _save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with tmp.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _decode(values: Any) -> np.ndarray:
    return np.asarray([
        value.decode() if isinstance(value, (bytes, np.bytes_)) else str(value)
        for value in np.asarray(values).reshape(-1)
    ], dtype=object)


def _sparse_or_dense(group: Any) -> Any:
    if isinstance(group, h5py.Dataset):
        return np.asarray(group[...])
    encoding = group.attrs.get("encoding-type", "")
    if isinstance(encoding, bytes):
        encoding = encoding.decode()
    if encoding not in {"csr_matrix", "csc_matrix"}:
        raise CalibrationError(f"UNSUPPORTED_MATRIX_ENCODING:{encoding}")
    from scipy import sparse

    shape = tuple(int(x) for x in group.attrs["shape"])
    constructor = sparse.csr_matrix if encoding == "csr_matrix" else sparse.csc_matrix
    return constructor(
        (np.asarray(group["data"]), np.asarray(group["indices"]), np.asarray(group["indptr"])),
        shape=shape,
    )


def prepare_inputs() -> None:
    """Bind exact patch barcodes to integer counts and h5ad coordinates."""
    rows: list[dict[str, Any]] = []
    for task, specimens in TASKS.items():
        genes = hest.read_genes(DATA / task)
        for specimen in specimens:
            patch_path = DATA / task / "patches" / f"{specimen}.h5"
            adata_path = DATA / task / "adata" / f"{specimen}.h5ad"
            patch_barcodes, patch_xy, _, _ = hest.read_patch_keys(patch_path)
            with h5py.File(adata_path, "r") as handle:
                matrix = _sparse_or_dense(handle["X"])
                all_barcodes = _decode(handle["obs/_index"][...])
                all_genes = _decode(handle["var/_index"][...])
                spatial = np.asarray(handle["obsm/spatial"][...], dtype=np.float64)
            barcode_lookup = {key: index for index, key in enumerate(all_barcodes.tolist())}
            gene_lookup = {key: index for index, key in enumerate(all_genes.tolist())}
            if len(barcode_lookup) != len(all_barcodes) or len(set(patch_barcodes.tolist())) != len(patch_barcodes):
                raise CalibrationError(f"NONUNIQUE_BARCODE:{task}:{specimen}")
            if any(x not in barcode_lookup for x in patch_barcodes) or any(x not in gene_lookup for x in genes):
                raise CalibrationError(f"EXPLICIT_KEY_JOIN_FAILED:{task}:{specimen}")
            ridx = np.asarray([barcode_lookup[x] for x in patch_barcodes], dtype=np.int64)
            gidx = np.asarray([gene_lookup[x] for x in genes], dtype=np.int64)
            raw = matrix[np.ix_(ridx, gidx)]
            total = np.asarray(matrix[ridx].sum(axis=1)).reshape(-1)
            try:
                from scipy import sparse
                if sparse.issparse(raw):
                    raw = raw.toarray()
            except ImportError:  # pragma: no cover
                pass
            raw = np.asarray(raw)
            if raw.shape != (len(patch_barcodes), 50):
                raise CalibrationError(f"COUNT_SHAPE:{task}:{specimen}:{raw.shape}")
            if raw.min(initial=0) < 0 or not np.allclose(raw, np.rint(raw)):
                raise CalibrationError(f"NONINTEGER_COUNT:{task}:{specimen}")
            xy = spatial[ridx]
            if xy.shape != (len(raw), 2) or not np.isfinite(xy).all() or np.ptp(xy, axis=0).min() <= 0:
                raise CalibrationError(f"COORDINATE_CONTRACT:{task}:{specimen}")
            output = INPUT / task / f"{specimen}.npz"
            output.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                output,
                counts=np.rint(raw).astype(np.uint32),
                total_depth=np.rint(total).astype(np.uint64),
                xy=xy.astype(np.float64),
                patch_xy=np.asarray(patch_xy, dtype=np.float64),
                barcodes=np.asarray(patch_barcodes, dtype=str),
                genes=np.asarray(genes, dtype=str),
            )
            ordered_key = hashlib.sha256()
            for barcode in patch_barcodes:
                ordered_key.update(task.encode() + b"\0" + specimen.encode() + b"\0" + str(barcode).encode() + b"\n")
            rows.append({
                "task": task,
                "specimen": specimen,
                "h5ad_rows": len(all_barcodes),
                "patch_common_rows": len(patch_barcodes),
                "genes": len(genes),
                "count_dtype": str(raw.dtype),
                "counts_nonnegative_integer": "PASS",
                "barcode_one_to_one": "PASS",
                "coordinate_join": "EXACT_BARCODE",
                "ordered_support_sha256": ordered_key.hexdigest(),
                "compact_input": str(output),
                "compact_bytes": output.stat().st_size,
                "compact_sha256": _sha256(output),
                "prediction_files_opened": 0,
            })
    frame = pd.DataFrame(rows)
    expected = {"IDC": 35536, "PAAD": 7571, "LUNG": 5206}
    observed = frame.groupby("task").patch_common_rows.sum().to_dict()
    status = "PASS" if observed == expected and int(frame.patch_common_rows.sum()) == 48313 else "FAIL"
    _atomic_tsv(RESULT / "HEST_MODEL_BLIND_INPUT_BINDING.tsv", frame)
    _atomic_json(RECEIPT / "HEST_MODEL_BLIND_INPUT_BINDING.json", {
        "status": status,
        "whole_expression_rows": 62272,
        "patch_common_rows": int(frame.patch_common_rows.sum()),
        "task_rows": observed,
        "expected_task_rows": expected,
        "truth_transform_id": TRUTH_TRANSFORM_ID,
        "prediction_files_opened": 0,
    })
    if status != "PASS":
        raise CalibrationError(f"PATCH_COMMON_TOTAL:{observed}")
    print(json.dumps({"status": status, "specimens": len(frame), "rows": int(frame.patch_common_rows.sum())}))


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


def _load_specimen(task: str, specimen: str) -> tuple[np.ndarray, list[np.ndarray]]:
    input_payload = np.load(INPUT / task / f"{specimen}.npz", allow_pickle=False)
    keep = np.load(_basis_dir(task, specimen) / "keep.npy", mmap_mode="r").astype(bool)
    counts = input_payload["counts"][keep].astype(np.int64)
    bands = [np.load(_basis_dir(task, specimen) / f"U5_B{k}.npy", mmap_mode="r") for k in range(1, 6)]
    if sum(matrix.shape[1] for matrix in bands) != len(counts) - len(np.unique(np.load(_basis_dir(task, specimen) / "component_labels.npy"))):
        raise CalibrationError(f"BASIS_MODE_COUNT:{task}:{specimen}")
    return counts, bands


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


def actual_moments(task: str, device: str) -> None:
    """Freeze task-specific LOPO RNA scales and V_RNA judgeability moments."""
    specimens = TASKS[task]
    completion = RECEIPT / f"HEST_{task}_ACTUAL_MOMENTS.json"
    required = [MOMENTS / task / "specimen_vrna.parquet"]
    required += [MOMENTS / task / f"{specimen}.weights.npz" for specimen in specimens]
    required += [MOMENTS / task / f"{specimen}.source_phase.parquet" for specimen in specimens]
    if completion.is_file() and json.loads(completion.read_text()).get("status") == "PASS" and all(path.is_file() for path in required):
        print(json.dumps({"status": "PASS_REUSED", "task": task, "stage": "actual-moments"}))
        return
    per_specimen: dict[str, dict[str, Any]] = {}
    for specimen in specimens:
        counts, bands = _load_specimen(task, specimen)
        bands_device = _device_bands(bands, device)
        d2 = np.zeros(counts.shape[1]); covariance = np.zeros(counts.shape[1]); ell2 = np.zeros(counts.shape[1])
        band_ab = np.zeros((BANDS, counts.shape[1]))
        phase_rows: list[dict[str, Any]] = []
        for split_index, seed in enumerate(THINNING_SEEDS):
            a, b = _halves(counts, task, specimen, seed)
            d2 += 0.5 * (np.mean(a * a, axis=0) + np.mean(b * b, axis=0))
            covariance += np.mean(a * b, axis=0)
            ell2 += np.sum(a * b, axis=0)
            ac = _coefficients(bands_device, a, device)
            bc = _coefficients(bands_device, b, device)
            for band_index in range(BANDS):
                band_ab[band_index] += np.sum(ac[band_index] * bc[band_index], axis=0)
                c = ac[band_index]
                v_rna = float(np.sum(c * bc[band_index]))
                v_pred = float(np.sum(c * c))
                for perturbation in range(32):
                    rng = np.random.default_rng(_seed("phase-source", task, specimen, seed, band_index, perturbation))
                    z = rng.normal(size=c.shape)
                    numerator = np.sum(c * z, axis=0, keepdims=True)
                    denominator = np.sum(c * c, axis=0, keepdims=True)
                    z -= c * np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > NUMERIC_TOL)
                    znorm = np.sqrt(np.sum(z * z, axis=0)); cnorm = np.sqrt(np.sum(c * c, axis=0))
                    z *= np.divide(cnorm, znorm, out=np.zeros_like(cnorm), where=znorm > NUMERIC_TOL)[None, :]
                    energy_error = float(np.max(np.abs(np.sum(z*z, axis=0)-np.sum(c*c,axis=0)) / np.maximum(np.sum(c*c,axis=0), NUMERIC_TOL)))
                    for theta in THETA:
                        rotated = math.cos(theta) * c + math.sin(theta) * z
                        c_judge = float(np.sum(rotated * bc[band_index]))
                        vp = float(np.sum(rotated * rotated))
                        q = c_judge / math.sqrt(vp * v_rna) if vp > 0 and v_rna > 0 else math.nan
                        phase_rows.append({
                            "task": task, "specimen": specimen, "split_seed": seed,
                            "perturbation_seed": perturbation, "band": f"B{band_index+1}",
                            "theta": theta, "cos_theta": math.cos(theta), "q": q,
                            "V_RNA": v_rna, "V_source": v_pred,
                            "energy_relative_error": energy_error,
                            "judge_read_for_direction": 0,
                        })
        scale = 1.0 / len(THINNING_SEEDS)
        per_specimen[specimen] = {
            "d2": d2 * scale, "covariance": covariance * scale,
            "ell2": ell2 * scale, "band_ab": band_ab * scale,
            "spots": len(counts),
        }
        phase = pd.DataFrame(phase_rows)
        _atomic_parquet(MOMENTS / task / f"{specimen}.source_phase.parquet", phase)
        np.savez_compressed(
            MOMENTS / task / f"{specimen}.npz",
            d2=d2*scale, covariance=covariance*scale, ell2=ell2*scale,
            band_ab=band_ab*scale, spots=np.asarray([len(counts)]),
        )
        del bands_device
        if device.startswith("cuda"):
            import torch
            torch.cuda.empty_cache()
    rows: list[dict[str, Any]] = []
    for specimen in specimens:
        others = [x for x in specimens if x != specimen]
        d2 = np.mean(np.stack([per_specimen[x]["d2"] for x in others]), axis=0)
        covariance = np.mean(np.stack([per_specimen[x]["covariance"] for x in others]), axis=0)
        h = np.divide(covariance, d2, out=np.zeros_like(d2), where=d2 > NUMERIC_TOL)
        w = np.maximum(h, 0.0)
        ell2 = per_specimen[specimen]["ell2"]
        eligible = (d2 > NUMERIC_TOL) & (w > 0) & (ell2 > 0)
        rna_scale2 = np.divide(w, d2, out=np.zeros_like(w), where=d2 > NUMERIC_TOL)
        rna_scale2[~eligible] = 0.0
        weight_path = MOMENTS / task / f"{specimen}.weights.npz"
        np.savez_compressed(weight_path, d2=d2, h=h, w=w, ell2=ell2, eligible=eligible, rna_scale2=rna_scale2)
        values = np.sum(per_specimen[specimen]["band_ab"] * rna_scale2[None, :], axis=1)
        for band_index, value in enumerate(values, 1):
            rows.append({
                "task": task, "specimen": specimen, "band": f"B{band_index}",
                "V_RNA": float(value), "V_RNA_per_retained_spot": float(value / per_specimen[specimen]["spots"]),
                "retained_spots": per_specimen[specimen]["spots"], "eligible_genes": int(eligible.sum()),
                "status": "ESTIMABLE" if value > 0 else "RNA_UNRESOLVED",
            })
    _atomic_parquet(MOMENTS / task / "specimen_vrna.parquet", pd.DataFrame(rows))
    _atomic_json(completion, {
        "status": "PASS", "task": task, "specimens": len(specimens),
        "thinning_seeds": list(THINNING_SEEDS), "truth_transform_id": TRUTH_TRANSFORM_ID,
        "molecule_conservation": "PASS", "prediction_files_opened": 0,
    })
    print(json.dumps({"status": "PASS", "task": task, "stage": "actual-moments"}))


def _orthogonal_equal_energy(
    c: np.ndarray, rng: np.random.Generator, target: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    norms = np.sqrt(np.sum(c * c, axis=0))
    if target is None:
        positive = norms > NUMERIC_TOL
        scalar = float(np.median(norms[positive])) if positive.any() else 0.0
        target = np.full_like(norms, scalar)
    target = np.asarray(target, dtype=np.float64)
    canonical = np.divide(c, norms[None, :], out=np.zeros_like(c), where=norms[None, :] > NUMERIC_TOL) * target[None, :]
    z = rng.normal(size=c.shape)
    numerator = np.sum(canonical * z, axis=0, keepdims=True)
    denominator = np.sum(canonical * canonical, axis=0, keepdims=True)
    z -= canonical * np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > NUMERIC_TOL)
    z_norm = np.sqrt(np.sum(z * z, axis=0))
    z = np.divide(z, z_norm[None, :], out=np.zeros_like(z), where=z_norm[None, :] > NUMERIC_TOL) * target[None, :]
    energy = np.sum(canonical * canonical, axis=0)
    error = float(np.max(np.abs(np.sum(z*z,axis=0)-energy) / np.maximum(energy, NUMERIC_TOL)))
    return canonical, z, error


def _reconstruct(bands: list[Any], coefficients: list[np.ndarray], device: str) -> np.ndarray:
    """Reconstruct non-null graph signal from coefficient blocks."""
    import torch

    result = None
    for matrix, coefficient in zip(bands, coefficients):
        u = matrix if isinstance(matrix, torch.Tensor) else torch.as_tensor(np.asarray(matrix), dtype=torch.float32, device=device)
        block = u @ torch.as_tensor(coefficient, dtype=torch.float32, device=device)
        result = block if result is None else result + block
    if result is None:
        raise CalibrationError("EMPTY_BAND_RECONSTRUCTION")
    output = result.cpu().numpy().astype(np.float64)
    del result
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return output


def _beta_binomial_dispersion(counts: np.ndarray, depth: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    """Freeze gene means and beta-binomial overdispersion from real depth/counts."""
    n = depth.astype(np.float64)[:,None]
    p = np.divide(counts.sum(axis=0), float(depth.sum()), out=np.zeros(counts.shape[1]), where=depth.sum()>0)
    p = np.clip(p, 1e-12, 1-1e-12)
    expected = n*p[None,:]
    binomial = n*p[None,:]*(1-p[None,:])
    denominator = n*np.maximum(n-1,0)*p[None,:]*(1-p[None,:])
    numerator = np.sum((counts-expected)**2-binomial,axis=0)
    rho = np.divide(numerator,np.sum(denominator,axis=0),out=np.zeros_like(numerator),where=np.sum(denominator,axis=0)>0)
    # This is a frozen count-model safety bound, not tuned on q. Values above
    # it make the beta-binomial nearly all-or-none and do not resemble the
    # observed HEST target-gene marginals.
    rho = np.clip(rho,0.0,0.20)
    return p,rho


def _latent_count_model(
    counts: np.ndarray, depth: np.ndarray, bands: list[Any], device: str,
    task: str, specimen: str,
) -> dict[str,Any]:
    """Build equal-band latent log1p means on real graph/depth/gene marginals."""
    full = np.log1p(counts.astype(np.float64)); full -= full.mean(axis=0,keepdims=True)
    observed = _coefficients(bands,full,device)
    norms = np.stack([np.sqrt(np.sum(coefficient*coefficient,axis=0)) for coefficient in observed])
    positive = np.all(norms>NUMERIC_TOL,axis=0)
    common = np.where(positive,np.median(norms,axis=0),0.0)
    equalized=[]
    for coefficient,norm in zip(observed,norms):
        equalized.append(np.divide(coefficient,norm[None,:],out=np.zeros_like(coefficient),where=norm[None,:]>NUMERIC_TOL)*common[None,:])
    spatial = _reconstruct(bands,equalized,device)
    mean_count = counts.mean(axis=0).astype(np.float64)
    scaled = np.zeros_like(spatial); latent_mu=np.zeros_like(spatial); amplitudes=np.zeros(counts.shape[1])
    depth64=np.maximum(depth.astype(np.float64),1.0)
    for gene in range(counts.shape[1]):
        x=spatial[:,gene]
        # Maximum feasible common multiplier preserves equal energy across all
        # bands while ensuring a valid mean and beta-binomial probability.
        low,high=0.0,1.0
        for _ in range(48):
            alpha=(low+high)/2
            ax=alpha*x
            xmax=float(np.max(ax))
            log_mean_exp=xmax+math.log(float(np.mean(np.exp(ax-xmax))))
            constant=math.log(mean_count[gene]+1.0)-log_mean_exp
            transformed=constant+ax
            mu=np.expm1(transformed)
            feasible=bool(np.min(transformed)>=-1e-12 and np.max(mu/depth64)<=0.80)
            if feasible: low=alpha
            else: high=alpha
        alpha=low; amplitudes[gene]=alpha; scaled[:,gene]=alpha*x
        ax=scaled[:,gene]; xmax=float(np.max(ax)); log_mean_exp=xmax+math.log(float(np.mean(np.exp(ax-xmax))))
        constant=math.log(mean_count[gene]+1.0)-log_mean_exp
        latent_mu[:,gene]=np.maximum(np.expm1(constant+ax),0.0)
    probability=np.clip(latent_mu/depth64[:,None],0.0,1.0-1e-12)
    latent_transform=np.log1p(latent_mu); latent_transform-=latent_transform.mean(axis=0,keepdims=True)
    canonical=_coefficients(bands,latent_transform,device)
    latent_norms=np.stack([np.sqrt(np.sum(c*c,axis=0)) for c in canonical])
    target=np.median(latent_norms,axis=0)
    orthogonal=[]; errors=[]
    for band_index,c in enumerate(canonical):
        cc,z,error=_orthogonal_equal_energy(c,np.random.default_rng(_seed("calibration-new-y-direction",task,specimen,band_index)),target)
        canonical[band_index]=cc; orthogonal.append(z); errors.append(error)
    _,dispersion=_beta_binomial_dispersion(counts,depth64)
    return {
        "probability":probability,"depth":depth.astype(np.int64),"dispersion":dispersion,
        "canonical":canonical,"orthogonal":orthogonal,
        "energy_error":max(errors),"amplitude_min":float(amplitudes.min()),
        "amplitude_median":float(np.median(amplitudes)),"amplitude_max":float(amplitudes.max()),
        "dispersion_min":float(dispersion.min()),"dispersion_median":float(np.median(dispersion)),
        "dispersion_max":float(dispersion.max()),
    }


def _sample_count_model(model: dict[str,Any], rng: np.random.Generator) -> tuple[np.ndarray,np.ndarray]:
    """Draw replicate latent rates and a fresh full Y from the frozen model."""
    probability=model["probability"]; depth=model["depth"]; dispersion=model["dispersion"]
    output=np.empty(probability.shape,dtype=np.int64); latent_mu=np.empty(probability.shape,dtype=np.float64)
    for gene in range(probability.shape[1]):
        p=probability[:,gene]; rho=float(dispersion[gene])
        if rho>1e-10:
            concentration=1.0/rho-1.0
            alpha=np.maximum(p*concentration,1e-10); beta=np.maximum((1-p)*concentration,1e-10)
            pdraw=rng.beta(alpha,beta)
        else:
            pdraw=p
        latent_mu[:,gene]=depth*pdraw
        output[:,gene]=rng.binomial(depth,pdraw)
    return output,latent_mu


def calibration_shard(task: str, shard_index: int, shard_count: int, device: str) -> None:
    """Run independent split replicates; no prediction/embedding is opened."""
    specimens = TASKS[task]
    frozen: dict[str, dict[str, Any]] = {}
    for specimen in specimens:
        counts, bands = _load_specimen(task, specimen)
        bands_device = _device_bands(bands, device)
        weight = np.load(MOMENTS / task / f"{specimen}.weights.npz", allow_pickle=False)
        compact=np.load(INPUT/task/f"{specimen}.npz",allow_pickle=False)
        keep=np.load(_basis_dir(task,specimen)/"keep.npy").astype(bool)
        count_model=_latent_count_model(counts,compact["total_depth"][keep],bands_device,device,task,specimen)
        frozen[specimen] = {
            "counts": counts, "bands": bands_device, "weight": weight,
            **count_model,
        }
    for replicate in range(shard_index, CALIBRATION_REPLICATES, shard_count):
        output = SHARDS / task / f"replicate_{replicate:03d}.parquet"
        if output.is_file():
            prior = pd.read_parquet(output, columns=["task", "replicate"])
            if len(prior) and prior.task.eq(task).all() and prior.replicate.eq(replicate).all():
                print(json.dumps({"status": "PASS_REUSED", "task": task, "replicate": replicate}), flush=True)
                continue
            raise CalibrationError(f"INVALID_REPLICATE_CHECKPOINT:{output}")
        rows: list[dict[str, Any]] = []
        for specimen in specimens:
            item = frozen[specimen]
            synthetic_y,latent_mu=_sample_count_model(item,np.random.default_rng(_seed("semi-synthetic-new-y",task,specimen,replicate)))
            a, b = _halves(synthetic_y, task, specimen, 3000000000 + replicate)
            ac = _coefficients(item["bands"], a, device)
            bc = _coefficients(item["bands"], b, device)
            latent=np.log1p(latent_mu); latent-=latent.mean(axis=0,keepdims=True)
            canonical=_coefficients(item["bands"],latent,device); orthogonal=[]; energy_errors=[]
            for band_index,c in enumerate(canonical):
                target=np.sqrt(np.sum(c*c,axis=0))
                cc,z,error=_orthogonal_equal_energy(c,np.random.default_rng(_seed("replicate-latent-direction",task,specimen,replicate,band_index)),target)
                canonical[band_index]=cc; orthogonal.append(z); energy_errors.append(error)
            weight = item["weight"]
            eligible = weight["eligible"].astype(bool)
            d = np.sqrt(weight["d2"]); w = weight["w"]; ell = np.sqrt(np.maximum(weight["ell2"], 0))
            rna_scale = np.divide(np.sqrt(w), d, out=np.zeros_like(w), where=d > NUMERIC_TOL)
            p_scale = np.divide(np.sqrt(w) * ell, d, out=np.zeros_like(w), where=d > NUMERIC_TOL)
            rna_scale[~eligible] = 0.0; p_scale[~eligible] = 0.0
            pnorm = np.sqrt(sum(np.sum(c*c, axis=0) for c in canonical)).clip(min=NUMERIC_TOL)
            for rho in RHO:
                for band_index in range(BANDS):
                    pc = rho * canonical[band_index] + math.sqrt(max(0.0, 1-rho*rho)) * orthogonal[band_index]
                    aa = ac[band_index] * rna_scale[None, :]
                    bb = bc[band_index] * rna_scale[None, :]
                    pp = pc * (p_scale / pnorm)[None, :]
                    vrna = float(np.sum(aa * bb)); cmodel = float(0.5 * np.sum(pp * (aa + bb))); vmodel = float(np.sum(pp * pp))
                    q = cmodel / math.sqrt(vrna * vmodel) if vrna > 0 and vmodel > 0 else math.nan
                    rows.append({
                        "task": task, "specimen": specimen, "replicate": replicate,
                        "rho": rho, "theta": math.nan, "band": f"B{band_index+1}",
                        "V_RNA": vrna, "C_model": cmodel, "V_model": vmodel, "q": q,
                        "eligible_genes": int(eligible.sum()), "status": "ESTIMABLE" if np.isfinite(q) else "NOT_ESTIMABLE",
                        "equal_energy_relative_error": max(energy_errors), "prediction_files_opened": 0,
                        "count_model":"FROZEN_REAL_DEPTH_GENE_MEAN_DISPERSION_BETA_BINOMIAL_NEW_Y_AND_REPLICATE_LATENT_ORACLE",
                    })
            for theta in THETA:
                rho = math.cos(theta)
                for band_index in range(BANDS):
                    pc = rho * canonical[band_index] + math.sin(theta) * orthogonal[band_index]
                    aa = ac[band_index] * rna_scale[None, :]; bb = bc[band_index] * rna_scale[None, :]
                    pp = pc * (p_scale / pnorm)[None, :]
                    vrna = float(np.sum(aa * bb)); cmodel = float(0.5*np.sum(pp*(aa+bb))); vmodel = float(np.sum(pp*pp))
                    q = cmodel / math.sqrt(vrna*vmodel) if vrna > 0 and vmodel > 0 else math.nan
                    rows.append({
                        "task": task, "specimen": specimen, "replicate": replicate,
                        "rho": math.nan, "theta": theta, "band": f"B{band_index+1}",
                        "V_RNA": vrna, "C_model": cmodel, "V_model": vmodel, "q": q,
                        "eligible_genes": int(eligible.sum()), "status": "ESTIMABLE" if np.isfinite(q) else "NOT_ESTIMABLE",
                        "equal_energy_relative_error": max(energy_errors), "prediction_files_opened": 0,
                        "count_model":"FROZEN_REAL_DEPTH_GENE_MEAN_DISPERSION_BETA_BINOMIAL_NEW_Y_AND_REPLICATE_LATENT_ORACLE",
                    })
        _atomic_parquet(output, pd.DataFrame(rows))
        print(json.dumps({"status": "PASS", "task": task, "replicate": replicate, "rows": len(rows)}), flush=True)


def summarize_task(task: str) -> None:
    paths = [SHARDS / task / f"replicate_{index:03d}.parquet" for index in range(CALIBRATION_REPLICATES)]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise CalibrationError(f"CALIBRATION_REPLICATES_MISSING:{task}:{len(missing)}")
    frame = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    rho = frame.loc[frame.rho.notna()].copy()
    task_rep = rho.groupby(["task", "replicate", "rho", "band"], as_index=False).q.mean()
    calibration = task_rep.groupby(["task", "rho", "band"], as_index=False).agg(
        q_mean=("q", "mean"), q_abs_mean=("q", lambda x: float(np.mean(np.abs(x)))),
        q_sd=("q", "std"), replicates=("q", "count"),
    )
    pivot = calibration.pivot(index="band", columns="rho", values="q_mean")
    q0 = calibration.loc[calibration.rho.eq(0)]
    range03 = float(calibration.loc[calibration.rho.eq(.3), "q_mean"].max() - calibration.loc[calibration.rho.eq(.3), "q_mean"].min())
    range06 = float(calibration.loc[calibration.rho.eq(.6), "q_mean"].max() - calibration.loc[calibration.rho.eq(.6), "q_mean"].min())
    oracle_error = float(np.max(np.abs(calibration.loc[calibration.rho.eq(1.0), "q_mean"]-1.0)))
    checks = {
        "rho0_mean_abs_q_le_0_02": bool(q0.q_abs_mean.max() <= .02),
        "rho03_band_range_le_0_03": bool(range03 <= CALIBRATION_BIAS_LIMIT),
        "rho06_band_range_le_0_03": bool(range06 <= CALIBRATION_BIAS_LIMIT),
        "monotone_each_band": bool(np.all(np.diff(pivot[[0.0,.3,.6,1.0]].to_numpy(), axis=1) >= -.01)),
        "equal_energy_error_le_1e_8": bool(frame.equal_energy_relative_error.max() <= 1e-8),
        "rho1_oracle_max_abs_error_le_0_03": oracle_error <= .03,
        "fresh_Y_each_replicate": bool(frame.count_model.eq("FROZEN_REAL_DEPTH_GENE_MEAN_DISPERSION_BETA_BINOMIAL_NEW_Y_AND_REPLICATE_LATENT_ORACLE").all()),
    }
    phase = frame.loc[frame.theta.notna()].copy()
    phase_task = phase.groupby(["task", "replicate", "theta", "band"], as_index=False).q.mean()
    phase_summary = phase_task.groupby(["task", "theta", "band"], as_index=False).agg(q_mean=("q","mean"),q_sd=("q","std"),replicates=("q","count"))
    phase_pivot = phase_summary.pivot(index="band",columns="theta",values="q_mean")
    phase_null = float(np.max(np.abs(phase_pivot[math.pi/2].to_numpy())))
    phase_monotone = bool(np.all(np.diff(phase_pivot[sorted(phase_pivot.columns)].to_numpy(),axis=1) <= .01))
    checks["phase_terminal_abs_q_le_0_02"] = phase_null <= .02
    checks["phase_monotone"] = phase_monotone
    status = "CALIBRATED" if all(checks.values()) else "TRANSFORM_CALIBRATION_FAIL"
    vrna_specimen = pd.read_parquet(MOMENTS / task / "specimen_vrna.parquet")
    rng = np.random.default_rng(BOOTSTRAP_SEED ^ _seed(task))
    vrna_summary: list[dict[str, Any]] = []
    for band, group in vrna_specimen.groupby("band", sort=True):
        values = group.V_RNA_per_retained_spot.to_numpy(float)
        boot = values[rng.integers(0, len(values), size=(BOOTSTRAP_REPLICATES, len(values)))].mean(axis=1)
        estimate = float(values.mean()); lower, upper = np.percentile(boot, [2.5,97.5])
        vrna_summary.append({
            "task": task, "band": band, "specimens": len(values), "V_RNA_per_spot": estimate,
            "ci_lower": float(lower), "ci_upper": float(upper), "positive_specimens": int((values>0).sum()),
            "judgeability": "ESTIMABLE_CI_ABOVE_ZERO" if lower > 0 else "V_RNA_NOT_ESTIMABLE",
        })
    vrna_table = pd.DataFrame(vrna_summary)
    if not vrna_table.judgeability.eq("ESTIMABLE_CI_ABOVE_ZERO").all() and status == "CALIBRATED":
        # Calibration remains a transform property, while individual bands are
        # independently marked unresolved for downstream q authorization.
        pass
    source_phase = pd.concat([
        pd.read_parquet(MOMENTS / task / f"{specimen}.source_phase.parquet") for specimen in TASKS[task]
    ], ignore_index=True)
    source_phase_spec = source_phase.groupby(["task","specimen","band","theta","cos_theta"],as_index=False).q.mean()
    source_phase_summary = source_phase_spec.groupby(["task","band","theta","cos_theta"],as_index=False).agg(
        q_mean=("q","mean"), q_sd=("q","std"), specimens=("specimen","nunique"),
    )
    _atomic_tsv(RESULT / f"HEST_{task}_EQUAL_DIFFICULTY_CALIBRATION.tsv", calibration)
    _atomic_tsv(RESULT / f"HEST_{task}_PHASE_KNOWN_ANSWER.tsv", phase_summary)
    _atomic_tsv(RESULT / f"HEST_{task}_SOURCE_PHASE.tsv", source_phase_summary)
    _atomic_tsv(RESULT / f"HEST_{task}_V_RNA_SPECIMEN.tsv", vrna_specimen)
    _atomic_tsv(RESULT / f"HEST_{task}_V_RNA_JUDGEABILITY.tsv", vrna_table)
    _atomic_json(RECEIPT / f"HEST_{task}_CALIBRATION_GATE.json", {
        "status": status, "task": task, "threshold_max_band_bias": CALIBRATION_BIAS_LIMIT,
        "calibration_replicates": CALIBRATION_REPLICATES, "checks": checks,
        "bias_rho_0_3": range03, "bias_rho_0_6": range06,
        "rho1_oracle_max_abs_error": oracle_error,
        "count_model": "FROZEN_REAL_DEPTH_GENE_MEAN_DISPERSION_BETA_BINOMIAL_NEW_Y_AND_REPLICATE_LATENT_ORACLE",
        "implementation_attempt": 3,
        "max_band_bias": max(range03, range06), "phase_terminal_abs_q": phase_null,
        "truth_transform_id": TRUTH_TRANSFORM_ID,
        "V_RNA_band_adjudication": vrna_summary,
        "prediction_files_opened": 0,
    })
    print(json.dumps({"status": status, "task": task, "max_band_bias": max(range03,range06), "phase_null": phase_null}))


def finalize() -> None:
    gates = []
    for task in TASKS:
        path = RECEIPT / f"HEST_{task}_CALIBRATION_GATE.json"
        if not path.is_file():
            raise CalibrationError(f"MISSING_TASK_GATE:{task}")
        payload = json.loads(path.read_text())
        gates.append({
            "task": task, "calibration_status": payload["status"],
            "max_band_bias": payload["max_band_bias"],
            "phase_terminal_abs_q": payload["phase_terminal_abs_q"],
            "estimable_bands": sum(x["judgeability"] == "ESTIMABLE_CI_ABOVE_ZERO" for x in payload["V_RNA_band_adjudication"]),
            "prediction_files_opened": payload["prediction_files_opened"],
        })
    _atomic_tsv(RESULT / "HEST_TASK_CALIBRATION_ADJUDICATION.tsv", pd.DataFrame(gates))
    status = "PASS" if all(x["calibration_status"] == "CALIBRATED" for x in gates) else "PARTIAL_OR_FAIL"
    _atomic_json(RECEIPT / "HEST_MODEL_BLIND_CALIBRATION_COMPLETE.json", {
        "status": status, "tasks": gates, "formal_q_authorization": "SUBJECT_TO_GATE_A_AND_MODEL_PCC",
        "prediction_files_opened": 0,
    })
    print(json.dumps({"status": status, "tasks": gates}, sort_keys=True))


def synthetic() -> None:
    rng = np.random.default_rng(20260807)
    q, _ = np.linalg.qr(rng.normal(size=(80,80)))
    chunks = np.array_split(np.arange(1,80),5)
    c = rng.normal(size=(len(chunks[0]),12))
    canonical, orthogonal, energy_error = _orthogonal_equal_energy(c, rng)
    shift_scale = []
    for theta in THETA:
        rotated = math.cos(theta)*canonical + math.sin(theta)*orthogonal
        shift_scale.append(float(np.max(np.abs(np.sum(rotated*rotated,axis=0)-np.sum(canonical*canonical,axis=0)) / np.maximum(np.sum(canonical*canonical,axis=0),NUMERIC_TOL))))
    checks = {
        "five_disjoint_mode_groups": len(chunks)==5 and sum(map(len,chunks))==79,
        "equal_energy_orthogonal_direction": energy_error < 1e-10,
        "phase_rotation_energy_conserved": max(shift_scale) < 1e-10,
        "raw_half_transform_bound": TRUTH_TRANSFORM_ID.endswith("HALVES_LOG1P_2X"),
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    _atomic_json(RECEIPT / "HEST_CALIBRATION_SYNTHETIC.json", {"status":status,"checks":checks,"max_energy_error":max([energy_error,*shift_scale]),"prediction_files_opened":0})
    if status != "PASS": raise CalibrationError("SYNTHETIC_FAIL")
    print(json.dumps({"status":status,"checks":checks}))


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    sub = command.add_subparsers(dest="stage", required=True)
    sub.add_parser("prepare-inputs")
    sub.add_parser("synthetic")
    basis = sub.add_parser("build-basis"); basis.add_argument("--task",choices=tuple(TASKS),required=True); basis.add_argument("--specimen",required=True); basis.add_argument("--device",default="cuda")
    moments = sub.add_parser("actual-moments"); moments.add_argument("--task",choices=tuple(TASKS),required=True); moments.add_argument("--device",default="cuda")
    shard = sub.add_parser("calibration-shard"); shard.add_argument("--task",choices=tuple(TASKS),required=True); shard.add_argument("--shard-index",type=int,required=True); shard.add_argument("--shard-count",type=int,required=True); shard.add_argument("--device",default="cuda")
    summary = sub.add_parser("summarize-task"); summary.add_argument("--task",choices=tuple(TASKS),required=True)
    sub.add_parser("finalize")
    return command


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.stage == "prepare-inputs": prepare_inputs()
    elif args.stage == "synthetic": synthetic()
    elif args.stage == "build-basis": build_basis(args.task,args.specimen,args.device)
    elif args.stage == "actual-moments": actual_moments(args.task,args.device)
    elif args.stage == "calibration-shard": calibration_shard(args.task,args.shard_index,args.shard_count,args.device)
    elif args.stage == "summarize-task": summarize_task(args.task)
    elif args.stage == "finalize": finalize()
    else: raise AssertionError(args.stage)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
