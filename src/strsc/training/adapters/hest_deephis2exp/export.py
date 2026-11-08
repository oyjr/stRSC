"""Prediction inference/export without invoking any STAudit endpoint."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from .batching import partition_context_rows
from .data import (
    AlignedHestSpecimen,
    HestAdapterError,
    HestSpotDataset,
    TRUTH_TRANSFORM_ID,
    load_specimen_batch,
)
from .models import canonical_model_name
from .objective import predict_expression


def _device_batch(
    batch: Mapping[str, Any],
    device: torch.device,
    *,
    keep_images_cpu: bool = False,
) -> dict[str, Any]:
    return {
        key: (
            value
            if keep_images_cpu and key == "images"
            else value.to(device)
            if isinstance(value, torch.Tensor)
            else value
        )
        for key, value in batch.items()
    }


@torch.no_grad()
def predict_aligned_specimen(
    model_name: str,
    model: torch.nn.Module,
    specimen: AlignedHestSpecimen,
    *,
    device: str | torch.device,
    patch_batch_size: int = 64,
    context_num_subsets: int = 1,
    num_workers: int = 0,
) -> np.ndarray:
    canonical = canonical_model_name(model_name)
    target_device = torch.device(device)
    model = model.to(target_device)
    model.eval()
    if canonical in {"ST-Net", "DeepSpaCE"}:
        dataset = HestSpotDataset((specimen,))
        loader = DataLoader(
            dataset,
            batch_size=int(patch_batch_size),
            shuffle=False,
            num_workers=int(num_workers),
            pin_memory=target_device.type == "cuda",
            persistent_workers=bool(num_workers),
        )
        blocks = [
            predict_expression(canonical, model, _device_batch(batch, target_device))
            .detach()
            .cpu()
            .numpy()
            for batch in loader
        ]
        prediction = np.concatenate(blocks, axis=0)
    elif canonical == "Hist2ST" and int(context_num_subsets) > 1:
        prediction = np.empty_like(specimen.truth, dtype=np.float32)
        chunks = partition_context_rows(
            np.arange(len(specimen.barcodes), dtype=np.int64),
            num_subsets=int(context_num_subsets),
            shuffle=False,
            seed=0,
        )
        for rows in chunks:
            batch = load_specimen_batch(specimen, rows=rows)
            block = (
                predict_expression(canonical, model, _device_batch(batch, target_device))
                .detach()
                .cpu()
                .numpy()
            )
            prediction[rows] = np.asarray(block, dtype=np.float32)
    else:
        batch = load_specimen_batch(specimen)
        prediction = (
            predict_expression(
                canonical,
                model,
                _device_batch(
                    batch,
                    target_device,
                    keep_images_cpu=canonical == "HisToGene",
                ),
            )
            .detach()
            .cpu()
            .numpy()
        )
    prediction = np.asarray(prediction, dtype=np.float32)
    if prediction.shape != specimen.truth.shape or not np.isfinite(prediction).all():
        raise HestAdapterError(
            f"PREDICTION_ALIGNMENT:{specimen.specimen}:{prediction.shape}:{specimen.truth.shape}"
        )
    return prediction


def export_hest_prediction(
    path: Path,
    *,
    specimen: AlignedHestSpecimen,
    prediction: np.ndarray,
    model_name: str,
    fold: int,
    seed: int,
    rho: float,
    provenance: Mapping[str, Any] | None = None,
) -> Path:
    """Write predictions aligned with HEST barcodes, genes and coordinates."""

    canonical = canonical_model_name(model_name)
    values = np.asarray(prediction, dtype=np.float32)
    if values.shape != specimen.truth.shape or not np.isfinite(values).all():
        raise HestAdapterError(
            f"EXPORT_PREDICTION:{values.shape}:{specimen.truth.shape}"
        )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + f".tmp.{os.getpid()}")
    with h5py.File(temporary, "w") as handle:
        strings = h5py.string_dtype("utf-8")
        handle.create_dataset(
            "barcodes", data=specimen.barcodes.astype(object), dtype=strings
        )
        handle.create_dataset("genes", data=specimen.genes.astype(object), dtype=strings)
        handle.create_dataset("spatial_coords", data=specimen.spatial_coords)
        handle.create_dataset(
            "raw_counts", data=np.rint(specimen.raw_counts).astype(np.uint32), compression="lzf"
        )
        handle.create_dataset("transformed_truth", data=specimen.truth, compression="lzf")
        handle.create_dataset("prediction", data=values, compression="lzf")
        handle.attrs.update(
            {
                "status": "PASS",
                "experiment": "G0-055",
                "task": specimen.task,
                "fold": int(fold),
                "sample_id": specimen.specimen,
                "model": canonical,
                "seed": int(seed),
                "rho": float(rho),
                "truth_transform_id": TRUTH_TRANSFORM_ID,
                "input_transform": "raw_rgb_float32_0_255",
                "scientific_endpoint_opened": False,
            }
        )
        for key, value in (provenance or {}).items():
            if key in handle.attrs:
                raise HestAdapterError(f"PROVENANCE_ATTRIBUTE_COLLISION:{key}")
            handle.attrs[str(key)] = value
        handle.flush()
    os.replace(temporary, output)
    return output
