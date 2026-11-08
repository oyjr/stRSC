#!/usr/bin/env python3
"""Score aligned cell predictions with the paper cell-scale and hotspot protocol."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from .cell_core import (
    BANDS,
    ContractError,
    add_terms,
    atomic_json,
    atomic_tsv,
    combine_bands,
    empty_terms,
    lazy_band_grams,
    load_canonical,
    load_protocol,
    model_rsc_from_terms,
    split_counts,
    transform_prediction,
    transform_rna_pair,
)
from .cell_biology import correlation, half_cell_score, neighborhood_operator, top_fraction
from .cell_rna import cache_view, create_view, make_membership


PAIRS = (("A", "A"), ("A", "B"), ("A", "P"), ("B", "B"), ("B", "P"), ("P", "P"))


def read_predictions(
    path: Path,
    specimen: str,
    cell_ids: np.ndarray,
    allow_missing: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    data = ad.read_h5ad(path)
    observed = np.asarray(data.obs_names.astype(str), dtype=object)
    suffix = f"_{specimen}"
    observed = np.asarray([x[: -len(suffix)] if x.endswith(suffix) else x for x in observed], dtype=object)
    lookup = {key: index for index, key in enumerate(observed)}
    order = np.fromiter((lookup.get(str(key), -1) for key in cell_ids), dtype=np.int64)
    retained = order >= 0
    missing = int(np.count_nonzero(~retained))
    if missing and not allow_missing:
        raise ContractError(f"CELL_CANONICAL_CELLS_MISSING:{missing}:{len(cell_ids)}")
    if not np.any(retained):
        raise ContractError("CELL_NO_ALIGNED_PREDICTIONS")
    matrix = data.X[order[retained]]
    if sparse.issparse(matrix):
        matrix = matrix.toarray()
    matrix = np.asarray(matrix, np.float32)
    if not np.isfinite(matrix).all():
        raise ContractError("CELL_NONFINITE_PREDICTION")
    receipt = {
        "prediction_file": str(path),
        "prediction_cells": int(data.n_obs),
        "alignment_cells_requested": int(len(cell_ids)),
        "alignment_cells_missing_prediction": missing,
        "canonical_cells_scored": int(retained.sum()),
        "prediction_genes": int(data.n_vars),
        "prediction_scale": "model_native_log1p_normalized_output",
    }
    return matrix, np.asarray(data.var_names.astype(str), dtype=object), retained, receipt


def load_view(path: Path) -> dict[str, object]:
    return {
        "membership": sparse.load_npz(path / "membership.npz").tocsr(),
        "diffusion": sparse.load_npz(path / "lazy_diffusion.npz").tocsr(),
        "exposure": np.load(path / "exposure.npy"),
        "retained": np.load(path / "retained.npy"),
        "labels": np.load(path / "component_labels.npy"),
        "receipt": json.loads((path / "receipt.json").read_text(encoding="utf-8")),
    }


def ensure_views(root: Path, specimen: str, counts_cells: pd.DataFrame, protocol: dict[str, object]) -> Path:
    cache = root / specimen
    expected = ["native_all", "native_neoplastic"] + [
        f"grid_{int(value)}um" for value in protocol["aggregation_um"]
    ]
    if all((cache / name / "receipt.json").is_file() for name in expected):
        return cache
    coords = counts_cells[["x_um", "y_um"]].to_numpy(np.float64)
    identity = sparse.eye(len(counts_cells), format="csr", dtype=np.float32)
    views = [create_view("native_all", "all", 0.0, identity, coords, np.ones(len(counts_cells)), protocol)]
    neoplastic = counts_cells.cell_class.astype(str).eq(protocol["proliferation"]["population"]).to_numpy()
    source = np.flatnonzero(neoplastic)
    membership = sparse.coo_matrix(
        (np.ones(len(source), np.float32), (np.arange(len(source)), source)),
        shape=(len(source), len(counts_cells)),
    ).tocsr()
    views.append(
        create_view(
            "native_neoplastic", "Neoplastic", 0.0, membership,
            coords[source], np.ones(len(source)), protocol,
        )
    )
    for bin_um in protocol["aggregation_um"]:
        membership, grid_coords, exposure = make_membership(coords, int(bin_um))
        views.append(
            create_view(
                f"grid_{int(bin_um)}um", "all", float(bin_um), membership,
                grid_coords, exposure, protocol,
            )
        )
    for view in views:
        cache_view(root, specimen, view)
    return cache


def rsc_rows(terms: dict[str, dict[tuple[str, str], float]], context: dict[str, object]) -> list[dict[str, object]]:
    frame = pd.DataFrame(model_rsc_from_terms(terms))
    frame = pd.concat([frame, pd.DataFrame([combine_bands(frame, ("B4", "B5"), "model")])])
    return [{**context, **row} for row in frame.to_dict("records")]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["sCellST", "DeepSpot2Cell"], default="sCellST")
    parser.add_argument("--specimen", required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument("--view-cache-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--gene-block", type=int, default=64)
    parser.add_argument("--track", choices=["xenium_centred", "cellvit_native"], default="xenium_centred")
    args = parser.parse_args()

    protocol = load_protocol(args.protocol)
    counts, cells, genes, metadata = load_canonical(args.canonical_root / args.specimen)
    canonical_counts = counts
    analysis_rows = np.arange(len(cells), dtype=np.int64)
    if args.track == "cellvit_native":
        required = {"cellvit_cell_id", "cellvit_mutual_nearest", "cellvit_match_distance_um"}
        missing_columns = sorted(required.difference(cells.columns))
        if missing_columns:
            raise ContractError(f"CELLVIT_ALIGNMENT_COLUMNS_MISSING:{missing_columns}")
        aligned = cells.cellvit_mutual_nearest.astype(bool).to_numpy() & (
            cells.cellvit_match_distance_um.to_numpy(np.float64)
            <= float(protocol["cellvit_mapping"]["maximum_distance_um"])
        )
        aligned_rows = np.flatnonzero(aligned)
        aligned_cells = cells.loc[aligned].reset_index(drop=True)
        prediction_ids = aligned_cells.cellvit_cell_id.astype(str).to_numpy()
        prediction, prediction_genes, prediction_retained, prediction_receipt = read_predictions(
            args.prediction, args.specimen, prediction_ids, allow_missing=True
        )
        analysis_rows = aligned_rows[prediction_retained]
        counts = canonical_counts[analysis_rows].tocsr()
        cells = aligned_cells.loc[prediction_retained].reset_index(drop=True)
        cache_specimen = f"{args.specimen}__cellvit_native"
    else:
        prediction_ids = cells.cell_id.astype(str).to_numpy()
        prediction, prediction_genes, prediction_retained, prediction_receipt = read_predictions(
            args.prediction, args.specimen, prediction_ids, allow_missing=True
        )
        if not np.all(prediction_retained):
            analysis_rows = analysis_rows[prediction_retained]
            counts = canonical_counts[analysis_rows].tocsr()
            cells = cells.loc[prediction_retained].reset_index(drop=True)
            cache_specimen = f"{args.specimen}__xenium_centred"
        else:
            cache_specimen = args.specimen
    full_counts, full_genes = counts, genes
    pred_lookup = {gene: i for i, gene in enumerate(prediction_genes)}
    count_lookup = dict(zip(genes.gene_symbol.astype(str), genes.gene_index.astype(int)))
    common = sorted(set(pred_lookup).intersection(count_lookup))
    if not common:
        raise ContractError("CELL_NO_COMMON_GENES")
    count_index = np.asarray([count_lookup[x] for x in common], np.int64)
    pred_index = np.asarray([pred_lookup[x] for x in common], np.int64)
    prediction = prediction[:, pred_index]
    counts = counts[:, count_index].tocsr()

    seeds = [int(x) for x in protocol["split_seeds"]]
    split = []
    for seed in seeds:
        full_a, full_b = split_counts(canonical_counts, args.specimen, seed)
        split.append(
            (
                full_a[analysis_rows][:, count_index].tocsr(),
                full_b[analysis_rows][:, count_index].tocsr(),
            )
        )
    seed_rows: list[dict[str, object]] = []
    final_rows: list[dict[str, object]] = []
    cache = ensure_views(args.view_cache_root, cache_specimen, cells, protocol)
    for view_path in sorted(x for x in cache.iterdir() if x.is_dir()):
        view = load_view(view_path)
        membership = view["membership"]
        retained = view["retained"]
        exposure = view["exposure"]
        labels = view["labels"]
        diffusion = view["diffusion"]
        receipt = view["receipt"]
        pred_view = np.asarray(membership @ prediction, np.float32)
        pred_view /= np.asarray(exposure, np.float32)[:, None]
        pred_view = pred_view[retained]
        split_view = [
            ((membership @ a).tocsr()[retained], (membership @ b).tocsr()[retained])
            for a, b in split
        ]
        seed_terms = {seed: empty_terms(PAIRS) for seed in seeds}
        eligible_by_seed = {seed: 0 for seed in seeds}
        for start in range(0, len(common), args.gene_block):
            block = slice(start, min(start + args.gene_block, len(common)))
            pp, pred_eligible = transform_prediction(pred_view[:, block], labels)
            for seed, (a_view, b_view) in zip(seeds, split_view):
                aa, bb, rna_eligible = transform_rna_pair(
                    a_view[:, block], b_view[:, block], exposure[retained], labels
                )
                eligible = pred_eligible & rna_eligible
                aa[:, ~eligible] = 0
                bb[:, ~eligible] = 0
                local_p = pp.copy()
                local_p[:, ~eligible] = 0
                eligible_by_seed[seed] += int(eligible.sum())
                add_terms(
                    seed_terms[seed],
                    lazy_band_grams(diffusion, {"A": aa, "B": bb, "P": local_p}, args.device),
                )
        total = empty_terms(PAIRS)
        for seed in seeds:
            add_terms(total, seed_terms[seed])
            seed_rows.extend(
                rsc_rows(
                    seed_terms[seed],
                    {
                        "specimen": args.specimen,
                        "patient": metadata["patient"],
                        "model": args.model,
                        "track": args.track,
                        "view": view_path.name,
                        "population": receipt["population"],
                        "output_grid_um": receipt["scale_um"],
                        "seed": seed,
                        "eligible_genes": eligible_by_seed[seed],
                    },
                )
            )
        for band in BANDS:
            for pair in PAIRS:
                total[band][pair] /= len(seeds)
        final_rows.extend(
            rsc_rows(
                total,
                {
                    "specimen": args.specimen,
                    "patient": metadata["patient"],
                    "model": args.model,
                    "track": args.track,
                    "view": view_path.name,
                    "population": receipt["population"],
                    "output_grid_um": receipt["scale_um"],
                    "seed": "cross_product_mean",
                    "eligible_genes": min(eligible_by_seed.values()),
                },
            )
        )
        print(json.dumps({"stage": "model_q_view_ready", "view": view_path.name}), flush=True)

    # Fixed within-Neoplastic proliferation hotspot evaluation.
    module_genes = list(protocol["proliferation"]["genes"])
    missing_module = [gene for gene in module_genes if gene not in common]
    if missing_module:
        raise ContractError(f"CELL_PROLIFERATION_GENES_MISSING:{missing_module}")
    module_index = np.asarray([common.index(gene) for gene in module_genes], np.int64)
    neoplastic = cells.cell_class.astype(str).eq(protocol["proliferation"]["population"]).to_numpy()
    coords = cells.loc[neoplastic, ["x_um", "y_um"]].to_numpy(np.float64)
    neighbor = neighborhood_operator(coords, float(protocol["proliferation"]["radius_um"]))
    neighbor_count = np.asarray(neighbor.sum(axis=1)).reshape(-1)
    valid = neighbor_count >= int(protocol["proliferation"]["minimum_neoplastic_cells"])
    pred_linear = np.expm1(np.maximum(prediction[neoplastic], 0.0))
    model_cell = half_cell_score(pred_linear[:, module_index].sum(axis=1), pred_linear.sum(axis=1) - pred_linear[:, module_index].sum(axis=1))
    model_neighborhood = np.asarray(neighbor @ model_cell).reshape(-1) / neighbor_count
    model_hotspot = top_fraction(model_neighborhood[valid], float(protocol["proliferation"]["hotspot_fraction"]))
    full_lookup = dict(zip(full_genes.gene_symbol.astype(str), full_genes.gene_index.astype(int)))
    full_module_index = np.asarray([full_lookup[x] for x in module_genes], np.int64)
    neoplastic_full_counts = full_counts[neoplastic].tocsr()
    module_full = np.asarray(neoplastic_full_counts[:, full_module_index].sum(axis=1)).reshape(-1).astype(np.int64)
    other_full = np.asarray(neoplastic_full_counts.sum(axis=1)).reshape(-1).astype(np.int64) - module_full
    hotspot_rows: list[dict[str, object]] = []
    for seed in protocol["preflight_split_seeds"]:
        rng = np.random.default_rng(int(seed))
        module_a = rng.binomial(module_full, 0.5)
        module_b = module_full - module_a
        other_a = rng.binomial(other_full, 0.5)
        other_b = other_full - other_a
        score_a = np.asarray(neighbor @ half_cell_score(module_a, other_a)).reshape(-1) / neighbor_count
        score_b = np.asarray(neighbor @ half_cell_score(module_b, other_b)).reshape(-1) / neighbor_count
        score_a, score_b = score_a[valid], score_b[valid]
        rna_hotspot = top_fraction(score_a, float(protocol["proliferation"]["hotspot_fraction"]))
        b_rna_hot, b_rna_other = float(score_b[rna_hotspot].mean()), float(score_b[~rna_hotspot].mean())
        b_model_hot, b_model_other = float(score_b[model_hotspot].mean()), float(score_b[~model_hotspot].mean())
        hotspot_rows.append(
            {
                "specimen": args.specimen,
                "patient": metadata["patient"],
                "model": args.model,
                "track": args.track,
                "seed": int(seed),
                "neoplastic_cells": int(neoplastic.sum()),
                "valid_neighborhoods": int(valid.sum()),
                "rna_hotspot_split_half_r": correlation(score_a, score_b),
                "E_RNA_normalized_log_score": b_rna_hot - b_rna_other,
                "E_model_normalized_log_score": b_model_hot - b_model_other,
                "RNA_hotspot_mean_ratio": b_rna_hot / b_rna_other if b_rna_other > 0 else math.nan,
                "model_hotspot_mean_ratio": b_model_hot / b_model_other if b_model_other > 0 else math.nan,
            }
        )

    args.results.mkdir(parents=True, exist_ok=True)
    seed_frame, final_frame, hotspot_frame = pd.DataFrame(seed_rows), pd.DataFrame(final_rows), pd.DataFrame(hotspot_rows)
    output_tag = args.model.upper() + "_" + args.track.upper()
    atomic_tsv(args.results / f"{args.specimen}_{output_tag}_SEED_BAND_RSC.tsv", seed_frame)
    atomic_tsv(args.results / f"{args.specimen}_{output_tag}_SPECIMEN_BAND_RSC.tsv", final_frame)
    atomic_tsv(args.results / f"{args.specimen}_{output_tag}_PROLIFERATION_HOTSPOT_SEEDS.tsv", hotspot_frame)
    hotspot_summary = hotspot_frame.groupby(["specimen", "patient", "model"], as_index=False).agg(
        split_seeds=("seed", "nunique"),
        neoplastic_cells=("neoplastic_cells", "first"),
        valid_neighborhoods=("valid_neighborhoods", "first"),
        E_RNA_normalized_log_score=("E_RNA_normalized_log_score", "mean"),
        E_model_normalized_log_score=("E_model_normalized_log_score", "mean"),
        RNA_hotspot_mean_ratio=("RNA_hotspot_mean_ratio", "mean"),
        model_hotspot_mean_ratio=("model_hotspot_mean_ratio", "mean"),
    )
    hotspot_summary["track"] = args.track
    atomic_tsv(args.results / f"{args.specimen}_{output_tag}_PROLIFERATION_HOTSPOTS.tsv", hotspot_summary)
    primary = final_frame.loc[
        final_frame.view.eq("native_neoplastic") & final_frame.band.eq("B4+B5"), "rsc"
    ]
    receipt = {
        "status": "PASS",
        "protocol_id": protocol["protocol_id"],
        "specimen": args.specimen,
        "patient": metadata["patient"],
        "model": args.model,
        "track": args.track,
        "common_genes": len(common),
        "primary_rsc_B4_B5": float(primary.iloc[0]),
        "E_model_normalized_log_score": float(hotspot_summary.E_model_normalized_log_score.iloc[0]),
        **prediction_receipt,
    }
    atomic_json(args.results / f"{args.specimen}_{output_tag}_RECEIPT.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
