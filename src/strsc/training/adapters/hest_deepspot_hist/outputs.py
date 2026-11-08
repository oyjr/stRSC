"""Endpoint-blind, barcode/gene-exact formal prediction artifacts."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np

from .contract import canonical_model_name, validate_rho


@dataclass(frozen=True)
class PredictionRecord:
    model: str
    task: str
    fold: int
    specimen: str
    seed: int
    rho: float
    prediction_h5: str
    prediction_key: str
    barcode_key: str
    gene_key: str
    n_spots: int
    n_genes: int
    expression_scale: str


def write_prediction_h5(
    path: str | Path,
    *,
    model: str,
    task: str,
    fold: int,
    specimen: str,
    seed: int,
    rho: float,
    predictions: np.ndarray,
    barcodes: Sequence[str],
    genes: Sequence[str],
    expression_scale: str,
) -> PredictionRecord:
    name = canonical_model_name(model)
    arm = validate_rho(rho)
    values = np.asarray(predictions, dtype=np.float32)
    barcode_values = np.asarray([str(value) for value in barcodes], dtype=object)
    gene_values = np.asarray([str(value) for value in genes], dtype=object)
    if values.shape != (len(barcode_values), len(gene_values)):
        raise ValueError(
            f"PREDICTION_ALIGNMENT:{values.shape}:{len(barcode_values)}:{len(gene_values)}"
        )
    if len(gene_values) != 50 or len(set(gene_values.tolist())) != 50:
        raise ValueError(f"PREDICTION_GENES:{len(gene_values)}")
    if len(set(barcode_values.tolist())) != len(barcode_values):
        raise ValueError("PREDICTION_DUPLICATE_BARCODES")
    if not np.isfinite(values).all():
        raise ValueError("PREDICTION_NONFINITE")

    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + f".tmp.{os.getpid()}")
    string_type = h5py.string_dtype(encoding="utf-8")
    with h5py.File(temporary, "w") as handle:
        handle.create_dataset("predictions", data=values, compression="gzip")
        handle.create_dataset(
            "barcodes", data=barcode_values.astype(string_type), dtype=string_type
        )
        handle.create_dataset(
            "genes", data=gene_values.astype(string_type), dtype=string_type
        )
        handle.attrs["model"] = name
        handle.attrs["task"] = str(task)
        handle.attrs["fold"] = int(fold)
        handle.attrs["specimen"] = str(specimen)
        handle.attrs["seed"] = int(seed)
        handle.attrs["rho"] = arm
        handle.attrs["expression_scale"] = str(expression_scale)
        handle.attrs["measured_held_out_rna_present"] = False
        handle.attrs["endpoint_scores_present"] = False
    os.replace(temporary, output)
    return PredictionRecord(
        model=name,
        task=str(task),
        fold=int(fold),
        specimen=str(specimen),
        seed=int(seed),
        rho=arm,
        prediction_h5=str(output),
        prediction_key="predictions",
        barcode_key="barcodes",
        gene_key="genes",
        n_spots=len(barcode_values),
        n_genes=len(gene_values),
        expression_scale=str(expression_scale),
    )


def write_pair_manifest(
    stem: str | Path,
    records: Sequence[PredictionRecord],
    *,
    provenance: dict[str, Any],
) -> tuple[Path, Path]:
    rows = [asdict(record) for record in records]
    if len(rows) != 2 or {float(row["rho"]) for row in rows} != {0.0, 0.5}:
        raise ValueError("PAIR_MANIFEST_ARMS")
    identities = {
        (
            row["model"],
            row["task"],
            int(row["fold"]),
            row["specimen"],
            int(row["seed"]),
        )
        for row in rows
    }
    if len(identities) != 1:
        raise ValueError("PAIR_MANIFEST_IDENTITY")

    tsv = Path(stem).resolve()
    if tsv.suffix.lower() != ".tsv":
        tsv = tsv.with_suffix(".tsv")
    tsv.parent.mkdir(parents=True, exist_ok=True)
    temporary_tsv = tsv.with_suffix(tsv.suffix + f".tmp.{os.getpid()}")
    with temporary_tsv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary_tsv, tsv)

    json_path = tsv.with_suffix(".json")
    payload = {
        "schema": "g0055_hest_deepspot_hist_pair_predictions_v1",
        "records": rows,
        "provenance": provenance,
        "contains_predictions": True,
        "contains_measured_held_out_rna": False,
        "contains_held_out_pcc": False,
        "contains_staudit_endpoint_scores": False,
    }
    temporary_json = json_path.with_suffix(
        json_path.suffix + f".tmp.{os.getpid()}"
    )
    temporary_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary_json, json_path)
    return tsv, json_path
