#!/usr/bin/env python3
"""Create aligned cell counts and metadata from one HEST Xenium slide."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import sparse
from scipy.spatial import cKDTree

from .cell_core import ContractError, atomic_tsv, decode_text, load_protocol, save_canonical


def read_gene_list(path: Path | None) -> list[str] | None:
    if path is None:
        return None
    frame = pd.read_csv(path)
    if "gene" in frame:
        values = frame["gene"]
    else:
        values = frame.iloc[:, -1]
    return sorted(set(values.dropna().astype(str)))


def metadata_pixel_size(path: Path) -> float:
    payload = json.loads(path.read_text(encoding="utf-8"))
    for key in ("pixel_size_um_estimated", "pixel_size_um_embedded", "pixel_size"):
        value = payload.get(key)
        if value is not None and float(value) > 0:
            return float(value)
    raise ContractError(f"PIXEL_SIZE_MISSING:{path}")


def segmentation_axis(path: Path, pixel_size_um: float) -> pd.DataFrame:
    frame = gpd.read_parquet(path)
    if frame.index.has_duplicates:
        raise ContractError("DUPLICATE_XENIUM_CELL_ID")
    cell_id = decode_text(frame.index.to_numpy())
    center = frame.geometry.centroid
    output = pd.DataFrame(
        {
            "cell_id": cell_id,
            "x_px": center.x.to_numpy(np.float64),
            "y_px": center.y.to_numpy(np.float64),
        }
    )
    output["x_um"] = output.x_px * pixel_size_um
    output["y_um"] = output.y_px * pixel_size_um
    return output


def cellvit_labels(path: Path, cells: pd.DataFrame, pixel_size_um: float, max_distance_um: float) -> pd.DataFrame:
    frame = gpd.read_parquet(path)
    center = frame.geometry.centroid
    reference_xy = np.column_stack([center.x, center.y]).astype(np.float64)
    query_xy = cells[["x_px", "y_px"]].to_numpy(np.float64)
    distance_px, index = cKDTree(reference_xy).query(query_xy, k=1, workers=-1)
    _, reverse_index = cKDTree(query_xy).query(reference_xy, k=1, workers=-1)
    distance_um = distance_px * pixel_size_um
    classes = frame["class"].astype(str).to_numpy()[index]
    if "cell_id" in frame:
        reference_id = decode_text(frame["cell_id"].to_numpy())
    else:
        reference_id = decode_text(frame.index.to_numpy())
    matched = distance_um <= max_distance_um
    mutual = matched & (reverse_index[index] == np.arange(len(cells)))
    return pd.DataFrame(
        {
            "cell_class": np.where(matched, classes, "Nolabel"),
            "cellvit_cell_id": np.where(matched, reference_id[index], ""),
            "cellvit_match_distance_um": distance_um,
            "cellvit_match": matched,
            "cellvit_mutual_nearest": mutual,
            "cell_class_source": np.where(matched, "CellViT_H&E_nearest", "unmatched"),
        }
    )


def count_transcripts(
    path: Path,
    cells: pd.DataFrame,
    target_genes: list[str] | None,
    qv_min: float,
    control_prefixes: tuple[str, ...],
) -> tuple[sparse.csr_matrix, list[str], dict[str, int]]:
    cell_lookup = {key: index for index, key in enumerate(cells.cell_id.astype(str))}
    if target_genes is None:
        gene_set: set[str] = set()
        for batch in pq.ParquetFile(path).iter_batches(
            batch_size=1_000_000, columns=["cell_id", "feature_name", "qv"]
        ):
            frame = batch.to_pandas()
            gene = decode_text(frame.feature_name.to_numpy())
            qv = frame.qv.to_numpy(np.float64)
            good = qv > qv_min
            gene_set.update(x for x in gene[good] if not x.startswith(control_prefixes))
        genes = sorted(gene_set)
    else:
        genes = [x for x in target_genes if not x.startswith(control_prefixes)]
    gene_lookup = {key: index for index, key in enumerate(genes)}
    row_parts: list[np.ndarray] = []
    col_parts: list[np.ndarray] = []
    assigned_rows = retained_rows = 0
    for batch in pq.ParquetFile(path).iter_batches(
        batch_size=1_000_000, columns=["cell_id", "feature_name", "qv"]
    ):
        frame = batch.to_pandas()
        cell = decode_text(frame.cell_id.to_numpy())
        gene = decode_text(frame.feature_name.to_numpy())
        qv = frame.qv.to_numpy(np.float64)
        assigned = cell != "UNASSIGNED"
        assigned_rows += int(assigned.sum())
        good = assigned & (qv > qv_min)
        row = np.fromiter((cell_lookup.get(x, -1) for x in cell[good]), np.int64)
        col = np.fromiter((gene_lookup.get(x, -1) for x in gene[good]), np.int64)
        keep = (row >= 0) & (col >= 0)
        row_parts.append(row[keep])
        col_parts.append(col[keep])
        retained_rows += int(keep.sum())
    row = np.concatenate(row_parts)
    col = np.concatenate(col_parts)
    counts = sparse.coo_matrix(
        (np.ones(len(row), np.uint32), (row, col)),
        shape=(len(cells), len(genes)),
        dtype=np.uint32,
    ).tocsr()
    counts.sum_duplicates()
    receipt = {
        "transcripts_assigned_before_qv_and_gene_filter": assigned_rows,
        "transcripts_retained": retained_rows,
        "cells_with_retained_transcripts": int(np.asarray(counts.sum(axis=1)).reshape(-1).astype(bool).sum()),
        "genes": len(genes),
    }
    return counts, genes, receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--specimen", required=True)
    parser.add_argument("--patient", required=True)
    parser.add_argument("--transcripts", type=Path, required=True)
    parser.add_argument("--xenium-nuclei", type=Path, required=True)
    parser.add_argument("--cellvit", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--gene-list", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--cellvit-max-distance-um", type=float)
    args = parser.parse_args()

    protocol = load_protocol(args.protocol)
    if args.specimen not in protocol["specimens"]:
        raise ContractError(f"UNFROZEN_SPECIMEN:{args.specimen}")
    pixel_size = metadata_pixel_size(args.metadata)
    cellvit_max_distance_um = (
        float(args.cellvit_max_distance_um)
        if args.cellvit_max_distance_um is not None
        else float(protocol["cellvit_mapping"]["maximum_distance_um"])
    )
    cells = segmentation_axis(args.xenium_nuclei, pixel_size)
    labels = cellvit_labels(args.cellvit, cells, pixel_size, cellvit_max_distance_um)
    cells = pd.concat([cells.reset_index(drop=True), labels], axis=1)
    counts, genes, transcript_receipt = count_transcripts(
        args.transcripts,
        cells,
        read_gene_list(args.gene_list),
        float(protocol["quality_value_min_exclusive"]),
        tuple(protocol["control_prefixes"]),
    )
    has_rna = np.asarray(counts.sum(axis=1)).reshape(-1) > 0
    cells = cells.loc[has_rna].reset_index(drop=True)
    counts = counts[has_rna].tocsr()
    cells.insert(0, "cell_index", np.arange(len(cells), dtype=np.int64))
    cells["total_molecules"] = np.asarray(counts.sum(axis=1)).reshape(-1).astype(np.uint64)
    proliferation = set(protocol["proliferation"]["genes"])
    gene_axis = pd.DataFrame(
        {
            "gene_index": np.arange(len(genes), dtype=np.int64),
            "gene_symbol": genes,
            "total_molecules": np.asarray(counts.sum(axis=0)).reshape(-1).astype(np.uint64),
            "proliferation_gene": [gene in proliferation for gene in genes],
        }
    )
    if counts.min() < 0 or not np.issubdtype(counts.dtype, np.integer):
        raise ContractError("COUNTS_NOT_NONNEGATIVE_INTEGER")
    metadata = {
        "status": "PASS",
        "protocol_id": protocol["protocol_id"],
        "specimen": args.specimen,
        "patient": args.patient,
        "pixel_size_um": pixel_size,
        "cells": len(cells),
        "genes": len(genes),
        "molecules": int(counts.sum()),
        "neoplastic_cells": int(cells.cell_class.eq("Neoplastic").sum()),
        "cellvit_matched_cells": int(cells.cellvit_match.sum()),
        "cellvit_mutual_nearest_cells": int(cells.cellvit_mutual_nearest.sum()),
        "cellvit_match_fraction": float(cells.cellvit_match.mean()),
        "cellvit_match_distance_um_median": float(cells.cellvit_match_distance_um.median()),
        "cellvit_match_distance_um_p95": float(cells.cellvit_match_distance_um.quantile(0.95)),
        **transcript_receipt,
        "source": {
            "transcripts": str(args.transcripts),
            "xenium_nuclei": str(args.xenium_nuclei),
            "cellvit": str(args.cellvit),
            "metadata": str(args.metadata),
            "gene_list": str(args.gene_list) if args.gene_list else None,
        },
    }
    output = args.output_root / args.specimen
    save_canonical(output, counts, cells, gene_axis, metadata)
    atomic_tsv(args.output_root.parent / "results" / f"{args.specimen}_DATA_PREFLIGHT.tsv", pd.DataFrame([metadata | {"source": "HEST"}]))
    print(json.dumps(metadata, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
