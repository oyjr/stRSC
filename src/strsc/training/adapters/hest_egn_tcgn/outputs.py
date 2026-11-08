"""Prediction-only artifacts with explicit specimen, barcode and gene alignment."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np
import pandas as pd

from .contracts import canonical_model_name


class PredictionManifestError(RuntimeError):
    pass


@dataclass(frozen=True)
class PredictionRecord:
    model: str
    task: str
    fold: int
    rho: float
    seed: int
    specimen: str
    prediction_h5: str
    prediction_key: str
    barcode_key: str
    gene_key: str
    n_spots: int
    n_genes: int
    expression_scale: str


def _atomic_replace(temporary: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temporary, destination)


def write_prediction_h5(
    path: Path,
    *,
    model: str,
    task: str,
    fold: int,
    rho: float,
    seed: int,
    specimen: str,
    predictions: np.ndarray,
    barcodes: np.ndarray,
    genes: np.ndarray,
    expression_scale: str,
) -> PredictionRecord:
    name = canonical_model_name(model)
    values = np.asarray(predictions, dtype=np.float32)
    barcode_values = np.asarray(barcodes, dtype=object)
    gene_values = np.asarray(genes, dtype=object)
    if values.shape != (len(barcode_values), len(gene_values)):
        raise PredictionManifestError(
            f"PREDICTION_ALIGNMENT:{values.shape}:{len(barcode_values)}:{len(gene_values)}"
        )
    if len(gene_values) != 50 or len(set(gene_values.tolist())) != 50:
        raise PredictionManifestError(f"GENE_CONTRACT:{len(gene_values)}")
    if len(set(barcode_values.tolist())) != len(barcode_values):
        raise PredictionManifestError("DUPLICATE_PREDICTION_BARCODE")
    if not np.isfinite(values).all():
        raise PredictionManifestError("PREDICTION_NONFINITE")
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    string_type = h5py.string_dtype(encoding="utf-8")
    with h5py.File(temporary, "w") as handle:
        handle.create_dataset("predictions", data=values, compression="gzip")
        handle.create_dataset("barcodes", data=barcode_values.astype(string_type))
        handle.create_dataset("genes", data=gene_values.astype(string_type))
        handle.attrs["model"] = name
        handle.attrs["task"] = str(task)
        handle.attrs["fold"] = int(fold)
        handle.attrs["rho"] = float(rho)
        handle.attrs["seed"] = int(seed)
        handle.attrs["specimen"] = str(specimen)
        handle.attrs["expression_scale"] = str(expression_scale)
        handle.attrs["endpoint_scores_present"] = False
    _atomic_replace(temporary, path)
    return PredictionRecord(
        model=name,
        task=str(task),
        fold=int(fold),
        rho=float(rho),
        seed=int(seed),
        specimen=str(specimen),
        prediction_h5=str(path),
        prediction_key="predictions",
        barcode_key="barcodes",
        gene_key="genes",
        n_spots=len(barcode_values),
        n_genes=len(gene_values),
        expression_scale=str(expression_scale),
    )


def write_prediction_manifest(
    path: Path,
    records: Sequence[PredictionRecord],
    *,
    provenance: dict[str, object],
) -> tuple[Path, Path]:
    if not records:
        raise PredictionManifestError("EMPTY_PREDICTION_MANIFEST")
    frame = pd.DataFrame([asdict(record) for record in records])
    if frame.duplicated(["model", "task", "fold", "rho", "seed", "specimen"]).any():
        raise PredictionManifestError("DUPLICATE_PREDICTION_RECORD")
    tsv = path.resolve()
    json_path = tsv.with_suffix(".json")
    tsv.parent.mkdir(parents=True, exist_ok=True)
    temporary_tsv = tsv.with_suffix(tsv.suffix + f".tmp.{os.getpid()}")
    frame.to_csv(temporary_tsv, sep="\t", index=False)
    _atomic_replace(temporary_tsv, tsv)
    payload = {
        "schema": "G0-055_prediction_manifest_v1",
        "records": [asdict(record) for record in records],
        "provenance": provenance,
        "contains_predictions": True,
        "contains_measured_held_out_rna": False,
        "contains_held_out_pcc": False,
        "contains_staudit_endpoint_scores": False,
    }
    temporary_json = json_path.with_suffix(json_path.suffix + f".tmp.{os.getpid()}")
    temporary_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _atomic_replace(temporary_json, json_path)
    return tsv, json_path
