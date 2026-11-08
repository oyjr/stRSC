#!/usr/bin/env python3
"""RNA split-half repeatability of the four-gene proliferation programme.

The script never reads model predictions.  It measures whether the four-gene
proliferation signal and its 64-um Neoplastic-cell hotspots reproduce in two
non-overlapping molecule halves.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree

from .cell_core import ContractError, atomic_json, atomic_tsv, load_canonical, load_protocol, split_counts


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, np.float64)
    right = np.asarray(right, np.float64)
    good = np.isfinite(left) & np.isfinite(right)
    left, right = left[good], right[good]
    if len(left) < 3:
        return math.nan
    left = left - left.mean()
    right = right - right.mean()
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > 0 else math.nan


def neighborhood_operator(coords_um: np.ndarray, radius_um: float) -> sparse.csr_matrix:
    tree = cKDTree(np.asarray(coords_um, np.float64))
    pairs = tree.query_pairs(radius_um, output_type="ndarray")
    n = len(coords_um)
    if len(pairs):
        rows = np.concatenate([pairs[:, 0], pairs[:, 1], np.arange(n)])
        cols = np.concatenate([pairs[:, 1], pairs[:, 0], np.arange(n)])
    else:
        rows = cols = np.arange(n)
    operator = sparse.coo_matrix(
        (np.ones(len(rows), np.float32), (rows, cols)), shape=(n, n)
    ).tocsr()
    operator.sum_duplicates()
    operator.data.fill(1.0)
    return operator


def half_cell_score(module: np.ndarray, other: np.ndarray) -> np.ndarray:
    module = np.asarray(module, np.float64)
    all_gene = module + np.asarray(other, np.float64)
    return np.log1p(1000.0 * module / np.maximum(all_gene, 1.0))


def top_fraction(values: np.ndarray, fraction: float) -> np.ndarray:
    count = max(1, int(math.ceil(len(values) * float(fraction))))
    selected = np.argpartition(values, len(values) - count)[-count:]
    mask = np.zeros(len(values), dtype=bool)
    mask[selected] = True
    return mask


def summarize(frame: pd.DataFrame, group: list[str], values: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for key, local in frame.groupby(group, sort=False, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(group, key))
        row["split_seeds"] = int(local.seed.nunique())
        for value in values:
            data = local[value].to_numpy(np.float64)
            row[f"{value}_mean"] = float(np.nanmean(data))
            row[f"{value}_min"] = float(np.nanmin(data))
            row[f"{value}_max"] = float(np.nanmax(data))
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--specimen", required=True)
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()

    protocol = load_protocol(args.protocol)
    counts, cells, genes, metadata = load_canonical(args.canonical_root / args.specimen)
    proliferation = protocol["proliferation"]
    requested_genes = list(proliferation["genes"])
    gene_lookup = dict(zip(genes.gene_symbol.astype(str), genes.gene_index.astype(int)))
    missing = [gene for gene in requested_genes if gene not in gene_lookup]
    if missing:
        raise ContractError(f"PROLIFERATION_GENES_MISSING:{missing}")
    module_index = np.asarray([gene_lookup[gene] for gene in requested_genes], np.int64)

    neoplastic = cells.cell_class.astype(str).eq(proliferation["population"]).to_numpy()
    if not neoplastic.any():
        raise ContractError("NO_NEOPLASTIC_CELLS")
    neoplastic_counts = counts[neoplastic].tocsr()
    neoplastic_coords = cells.loc[neoplastic, ["x_um", "y_um"]].to_numpy(np.float64)
    neighbor = neighborhood_operator(neoplastic_coords, float(proliferation["radius_um"]))
    neighbor_count = np.asarray(neighbor.sum(axis=1)).reshape(-1)
    valid_anchor = neighbor_count >= int(proliferation["minimum_neoplastic_cells"])
    if valid_anchor.sum() < 20:
        raise ContractError(f"TOO_FEW_HOTSPOT_ANCHORS:{valid_anchor.sum()}")

    module_rows: list[dict[str, object]] = []
    hotspot_rows: list[dict[str, object]] = []
    seeds = [int(seed) for seed in protocol["preflight_split_seeds"]]
    full_module = np.asarray(counts[:, module_index].sum(axis=1)).reshape(-1)
    neoplastic_module = np.asarray(neoplastic_counts[:, module_index].sum(axis=1)).reshape(-1).astype(np.int64)
    neoplastic_all_gene = np.asarray(neoplastic_counts.sum(axis=1)).reshape(-1).astype(np.int64)
    neoplastic_other = neoplastic_all_gene - neoplastic_module

    for seed in seeds:
        a, b = split_counts(counts, args.specimen, seed)
        for gene, gene_index in zip(requested_genes, module_index):
            ga_count = a[:, gene_index].toarray().reshape(-1)
            gb_count = b[:, gene_index].toarray().reshape(-1)
            module_rows.append(
                {
                    "specimen": args.specimen,
                    "patient": metadata["patient"],
                    "seed": seed,
                    "feature": gene,
                    "support_scale": "raw_molecule_count",
                    "cells": len(cells),
                    "cells_nonzero_full": int(counts[:, gene_index].getnnz(axis=0)[0]),
                    "transcripts_full": int(counts[:, gene_index].sum()),
                    "split_half_r": correlation(ga_count, gb_count),
                }
            )
        ma_count = np.asarray(a[:, module_index].sum(axis=1)).reshape(-1)
        mb_count = np.asarray(b[:, module_index].sum(axis=1)).reshape(-1)
        module_rows.append(
            {
                "specimen": args.specimen,
                "patient": metadata["patient"],
                "seed": seed,
                "feature": "four_gene_module",
                "support_scale": "raw_molecule_count",
                "cells": len(cells),
                "cells_nonzero_full": int(np.count_nonzero(full_module)),
                "transcripts_full": int(full_module.sum()),
                "split_half_r": correlation(ma_count, mb_count),
            }
        )

        # Reproduce the registered hotspot preflight exactly: a fresh RNG for
        # each seed, module counts split first and other panel counts second.
        rng = np.random.default_rng(seed)
        module_a = rng.binomial(neoplastic_module, 0.5)
        module_b = neoplastic_module - module_a
        other_a = rng.binomial(neoplastic_other, 0.5)
        other_b = neoplastic_other - other_a
        cell_a = half_cell_score(module_a, other_a)
        cell_b = half_cell_score(module_b, other_b)
        neighborhood_a = np.asarray(neighbor @ cell_a).reshape(-1) / neighbor_count
        neighborhood_b = np.asarray(neighbor @ cell_b).reshape(-1) / neighbor_count
        score_a = neighborhood_a[valid_anchor]
        score_b = neighborhood_b[valid_anchor]
        hotspot_a = top_fraction(score_a, float(proliferation["hotspot_fraction"]))
        hotspot_b = top_fraction(score_b, float(proliferation["hotspot_fraction"]))
        intersection = int(np.count_nonzero(hotspot_a & hotspot_b))
        union = int(np.count_nonzero(hotspot_a | hotspot_b))
        b_hot = float(score_b[hotspot_a].mean())
        b_other = float(score_b[~hotspot_a].mean())
        hotspot_rows.append(
            {
                "specimen": args.specimen,
                "patient": metadata["patient"],
                "seed": seed,
                "population": proliferation["population"],
                "radius_um": float(proliferation["radius_um"]),
                "neoplastic_cells": int(neoplastic.sum()),
                "valid_neighborhoods": int(valid_anchor.sum()),
                "hotspot_neighborhoods": int(hotspot_a.sum()),
                "rna_hotspot_split_half_r": correlation(score_a, score_b),
                "top10_jaccard": float(intersection / union) if union else math.nan,
                "B_score_mean_A_hotspot": b_hot,
                "B_score_mean_other": b_other,
                "normalized_log_score_effect": b_hot - b_other,
                "normalized_log_score_mean_ratio": b_hot / b_other if b_other > 0 else math.nan,
            }
        )

    module_frame = pd.DataFrame(module_rows)
    hotspot_frame = pd.DataFrame(hotspot_rows)
    module_summary = summarize(
        module_frame, ["specimen", "patient", "feature", "support_scale"], ["split_half_r"]
    )
    hotspot_summary = summarize(
        hotspot_frame,
        ["specimen", "patient"],
        [
            "rna_hotspot_split_half_r",
            "top10_jaccard",
            "normalized_log_score_effect",
            "normalized_log_score_mean_ratio",
        ],
    )
    for column in ("neoplastic_cells", "valid_neighborhoods", "hotspot_neighborhoods"):
        hotspot_summary[column] = int(hotspot_frame[column].iloc[0])

    args.results.mkdir(parents=True, exist_ok=True)
    atomic_tsv(args.results / f"{args.specimen}_RNA_MODULE_SPLIT_SEEDS.tsv", module_frame)
    atomic_tsv(args.results / f"{args.specimen}_RNA_MODULE_SUPPORT.tsv", module_summary)
    atomic_tsv(args.results / f"{args.specimen}_PROLIFERATION_RNA_HOTSPOT_SEEDS.tsv", hotspot_frame)
    atomic_tsv(args.results / f"{args.specimen}_PROLIFERATION_RNA_HOTSPOTS.tsv", hotspot_summary)
    receipt = {
        "status": "PASS",
        "protocol_id": protocol["protocol_id"],
        "specimen": args.specimen,
        "patient": metadata["patient"],
        "cells": int(len(cells)),
        "neoplastic_cells": int(neoplastic.sum()),
        "valid_neighborhoods": int(valid_anchor.sum()),
        "split_seeds": len(seeds),
        "score": proliferation["cell_score"],
        "module_split_half_r_mean": float(
            module_summary.loc[module_summary.feature.eq("four_gene_module"), "split_half_r_mean"].iloc[0]
        ),
        "rna_hotspot_split_half_r_mean": float(hotspot_summary.rna_hotspot_split_half_r_mean.iloc[0]),
        "top10_jaccard_mean": float(hotspot_summary.top10_jaccard_mean.iloc[0]),
        "normalized_log_score_effect_mean": float(hotspot_summary.normalized_log_score_effect_mean.iloc[0]),
        "normalized_log_score_mean_ratio_mean": float(
            hotspot_summary.normalized_log_score_mean_ratio_mean.iloc[0]
        ),
    }
    atomic_json(args.results / f"{args.specimen}_PROLIFERATION_RNA_RECEIPT.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
