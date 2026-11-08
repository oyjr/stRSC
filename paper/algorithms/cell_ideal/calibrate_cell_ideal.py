#!/usr/bin/env python3
"""Noisy ideal-cell prediction calibration on frozen G0-050 views.

Writes only the new run output. Counts, graphs and model outputs are read-only.
Prediction H5AD, when supplied, is opened only for cell/gene axes, never X.
Each invocation is one specimen/view/scorer, suitable for independent ACP tasks.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import poisson

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "source"))
from g0050_core import (BANDS, NUMERIC_TOL, atomic_json, atomic_tsv,
                        center_components, lazy_band_grams, load_canonical,
                        stable_seed, transform_prediction, transform_rna_pair)
from amplitude_match import match_prediction_amplitude

ORACLES = ("expected_half_log", "log1p_lambda")
SCORER_VERSION = "cell_ideal_fixed_mean_amplitude_v1"
PAIRS = (("A", "A"), ("A", "B"), ("A", "P"),
         ("B", "B"), ("B", "P"), ("P", "P"))


def expected_half_log(integer_rates, tail_probability=1e-13):
    """E log(1+2Y), Y~Pois(lambda/2), by deterministic probability summation.

    Cell exposure is one. Pooling later averages these ideal cell predictions,
    exactly as the saved-prediction scorer; no grid-level oracle substitutes it.
    """
    rates = np.asarray(integer_rates, dtype=np.float64)
    distinct, inverse = np.unique(rates, return_inverse=True)
    expectation = np.zeros(len(distinct), dtype=np.float64)
    for i, lam in enumerate(distinct):
        if lam == 0:
            continue
        mu = lam / 2.0
        lower = max(0, int(poisson.ppf(tail_probability / 2, mu)))
        upper = int(poisson.ppf(1 - tail_probability / 2, mu))
        k = np.arange(lower, upper + 1, dtype=np.float64)
        expectation[i] = np.sum(poisson.pmf(k, mu) * np.log1p(2 * k))
    return expectation[inverse].reshape(rates.shape)


def h5_axis(path, key):
    """Read an H5AD dataframe's index without loading any prediction values."""
    import h5py
    with h5py.File(path, "r") as handle:
        group = handle[key]
        index = group.attrs.get("_index", "_index")
        if isinstance(index, bytes):
            index = index.decode()
        values = group[index][:]
    return np.asarray([v.decode() if isinstance(v, bytes) else str(v) for v in values])


def load_support(args):
    counts, cells, genes, metadata = load_canonical(args.canonical)
    if counts.nnz and (np.any(counts.data < 0) or not np.equal(counts.data, np.floor(counts.data)).all()):
        raise ValueError("LAMBDA_MUST_BE_ORIGINAL_NONNEGATIVE_INTEGER_COUNTS")
    counts = counts.astype(np.int64)
    all_genes = genes.gene_symbol.astype(str).to_numpy()
    if len(set(all_genes)) != len(all_genes):
        raise ValueError("DUPLICATE_CANONICAL_GENE_NAMES")
    canonical_rows = np.arange(len(cells))
    requested_genes = None
    if args.prediction_axis:
        pred_cells = h5_axis(args.prediction_axis, "obs")
        suffix = f"_{args.specimen}"
        pred_cells = np.asarray([s[:-len(suffix)] if s.endswith(suffix) else s for s in pred_cells])
        if len(set(pred_cells)) != len(pred_cells):
            raise ValueError("DUPLICATE_PREDICTION_AXIS_IDS")
        # The original read_predictions retains canonical order, not prediction order.
        keep = pd.Index(pred_cells).get_indexer(cells.cell_id.astype(str)) >= 0
        canonical_rows = np.flatnonzero(keep)
        requested_genes = set(h5_axis(args.prediction_axis, "var"))
    if args.cell_indices:
        canonical_rows = np.load(args.cell_indices, allow_pickle=False).astype(np.int64)
        if len(set(canonical_rows.tolist())) != len(canonical_rows):
            raise ValueError("DUPLICATE_CELL_SUBSET_INDICES")
    if args.gene_list:
        frame = pd.read_csv(args.gene_list, sep=None, engine="python")
        gene_column = next((s for s in ("gene", "gene_symbol", "gene_canonical") if s in frame.columns), frame.columns[-1])
        requested_genes = set(frame[gene_column].astype(str))
    if requested_genes is None:
        raise ValueError("FROZEN_GENE_SUPPORT_REQUIRED:provide --gene-list or --prediction-axis")
    indices = np.flatnonzero(np.isin(all_genes, list(requested_genes)))
    if not len(indices):
        raise ValueError("EMPTY_GENE_INTERSECTION")
    counts = counts[canonical_rows][:, indices].tocsc()
    return counts, cells.iloc[canonical_rows], all_genes[indices], indices, metadata, canonical_rows


def load_view(path, canonical_cells):
    membership = sparse.load_npz(path / "membership.npz").tocsr()
    if membership.shape[1] != canonical_cells:
        raise ValueError(f"CACHE_CELL_SUPPORT_MISMATCH:{membership.shape[1]}!={canonical_cells}; provide exact prediction-axis/cell-indices")
    retained = np.load(path / "retained.npy", allow_pickle=False)
    labels = np.load(path / "component_labels.npy", allow_pickle=False)
    exposure = np.load(path / "exposure.npy", allow_pickle=False)[retained]
    diffusion = sparse.load_npz(path / "lazy_diffusion.npz").tocsr()
    receipt = json.loads((path / "receipt.json").read_text(encoding="utf-8"))
    membership = membership[retained]
    if diffusion.shape != (len(labels), len(labels)) or membership.shape[0] != len(labels):
        raise ValueError("CACHE_NODE_ALIGNMENT")
    return membership, exposure, diffusion, labels, receipt


def simulated_halves(block, absolute_gene_indices, specimen, seed, replicate):
    """Gene-specific streams preserve draws when task/block partitioning changes."""
    a, b = block.copy(), block.copy()
    for col, gene in enumerate(absolute_gene_indices):
        lo, hi = block.indptr[col:col+2]
        rng = np.random.default_rng(stable_seed(seed, specimen, replicate, int(gene)))
        x = rng.poisson(block.data[lo:hi])
        half = rng.binomial(x, 0.5)
        a.data[lo:hi], b.data[lo:hi] = half, x - half
    a.eliminate_zeros()
    b.eliminate_zeros()
    return a, b


def summarize_terms(terms):
    aa, vr, ap, bb, bp, vp = (terms[pair] for pair in PAIRS)
    cross = 0.5 * (ap + bp)
    # Match the original scorer's denominator threshold, not separate per-term thresholds.
    denominator = math.sqrt(max(vp, 0.0) * max(vr, 0.0))
    rna_denominator = math.sqrt(max(aa, 0.0) * max(bb, 0.0))
    q = cross / denominator if denominator > NUMERIC_TOL else math.nan
    repeatability = vr / rna_denominator if rna_denominator > NUMERIC_TOL else math.nan
    return dict(A_energy=aa, B_energy=bb, V_RNA=vr, C_model=cross, V_model=vp,
                q_ideal=q, rna_repeatability=repeatability,
                q_status="defined" if math.isfinite(q) else "nonpositive_or_nearzero_reference_or_prediction_energy")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--specimen", required=True)
    parser.add_argument("--canonical", required=True, type=Path)
    parser.add_argument("--view-cache", required=True, type=Path)
    parser.add_argument("--prediction-axis", type=Path)
    parser.add_argument("--gene-list", type=Path)
    parser.add_argument("--cell-indices", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mode", choices=["original", "amplitude_matched"], default="original")
    parser.add_argument("--amplitude-reference", type=Path,
                        help="Original-mode RNA_FULLBAND_MOMENTS.npz; required for amplitude_matched")
    parser.add_argument("--skip-complete", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--gene-block", type=int, default=32)
    parser.add_argument("--replicates", type=int, default=20)
    parser.add_argument("--base-seed", type=int, default=20261001)
    parser.add_argument("--scope", choices=["manuscript", "additional_malignant_grid"], default="manuscript")
    args = parser.parse_args()
    if args.replicates != 20:
        raise ValueError("PROTOCOL_REQUIRES_20_REPLICATES")
    if args.mode == "amplitude_matched" and args.amplitude_reference is None:
        raise ValueError("MATCHED_SCORER_REQUIRES_ORIGINAL_20_SPLIT_AMPLITUDE_REFERENCE")
    if args.skip_complete and (args.output / "COMPLETE.json").is_file():
        old = json.loads((args.output / "COMPLETE.json").read_text(encoding='utf8'))
        expected = dict(scorer_version=SCORER_VERSION, specimen=args.specimen, mode=args.mode,
                        base_seed=args.base_seed, replicates=args.replicates,
                        canonical_path=str(args.canonical), cache_path=str(args.view_cache),
                        prediction_axis_path=str(args.prediction_axis) if args.prediction_axis else None,
                        gene_list_path=str(args.gene_list) if args.gene_list else None,
                        cell_indices_path=str(args.cell_indices) if args.cell_indices else None,
                        scope=args.scope,
                        amplitude_reference=str(args.amplitude_reference) if args.amplitude_reference else None)
        if old.get("status") != "COMPLETE" or any(old.get(k) != v for k, v in expected.items()):
            raise ValueError("EXISTING_COMPLETE_RECEIPT_SETTINGS_MISMATCH")
        print(json.dumps({"status": "COMPLETE_REUSED", "output": str(args.output)}), flush=True)
        return
    started = time.monotonic()
    counts, cells, genes, gene_indices, metadata, canonical_rows = load_support(args)
    member, exposure, diffusion, labels, receipt = load_view(args.view_cache, len(cells))
    args.output.mkdir(parents=True, exist_ok=True)
    binding = dict(scorer_version=SCORER_VERSION, specimen=args.specimen, patient=metadata["patient"], mode=args.mode,
                   scope=args.scope, canonical_path=str(args.canonical), cache_path=str(args.view_cache),
                   prediction_axis_path=str(args.prediction_axis) if args.prediction_axis else None,
                   gene_list_path=str(args.gene_list) if args.gene_list else None,
                   cell_indices_path=str(args.cell_indices) if args.cell_indices else None,
                   cells=len(cells), nodes=len(labels), genes=len(genes), replicates=args.replicates,
                   base_seed=args.base_seed, view=args.view_cache.name, graph=receipt,
                   oracle="E log(1+2 Poi(lambda/2)) per cell, then mean cell predictions per pooled node",
                   secondary_oracle="log1p(lambda) per cell, then mean cell predictions per pooled node",
                   expectation_tail_probability=1e-13, original_counts_as_fixed_rates=True,
                   half_generation="X~Poisson(lambda); A~Binomial(X,.5); B=X-A",
                   normalization="original transform_rna_pair and transform_prediction",
                   amplitude_reference=str(args.amplitude_reference) if args.amplitude_reference else None,
                   amplitude_rule="sqrt(max(mean_20_splits(sum_nodes(A_scaled*B_scaled)),0)) per gene; fixed across splits; full-band only" if args.mode == "amplitude_matched" else None)
    fixed_cross_mean = None
    if args.amplitude_reference:
        with np.load(args.amplitude_reference, allow_pickle=False) as reference:
            if (str(reference["specimen"].item()) != args.specimen or
                str(reference["view"].item()) != args.view_cache.name or
                int(reference["base_seed"].item()) != args.base_seed or
                int(reference["replicates"].item()) != args.replicates or
                not np.array_equal(reference["genes"].astype(str), genes) or
                not np.array_equal(reference["canonical_gene_indices"], gene_indices) or
                not np.array_equal(reference["canonical_cell_indices"], canonical_rows)):
                raise ValueError("AMPLITUDE_REFERENCE_SUPPORT_OR_SEED_MISMATCH")
            fixed_cross_mean = reference["mean_fullband_cross"].astype(np.float64)
        if fixed_cross_mean.shape != (len(genes),) or not np.isfinite(fixed_cross_mean).all():
            raise ValueError("AMPLITUDE_REFERENCE_INVALID")
    atomic_json(args.output / "INPUT_BINDING.json", binding)
    atomic_tsv(args.output / "GENE_SUPPORT.tsv", pd.DataFrame({"canonical_gene_index": gene_indices, "gene": genes}))
    np.save(args.output / "CANONICAL_CELL_INDICES.npy", canonical_rows)
    terms = {oracle: [{band: {pair: 0.0 for pair in PAIRS} for band in BANDS}
                      for _ in range(args.replicates)] for oracle in ORACLES}
    eligible_counts = {oracle: np.zeros(args.replicates, dtype=int) for oracle in ORACLES}
    negative_cross_counts = {oracle: np.zeros(args.replicates, dtype=int) for oracle in ORACLES}
    rna_cross_sum = np.zeros(len(genes), dtype=np.float64)
    for start in range(0, len(genes), args.gene_block):
        stop = min(start + args.gene_block, len(genes))
        block = counts[:, start:stop].tocsc()
        ideal_expected, ideal_log = block.astype(np.float64), block.astype(np.float64)
        ideal_expected.data = expected_half_log(ideal_expected.data)
        ideal_log.data = np.log1p(ideal_log.data)
        predictions = {}
        for oracle, ideal in zip(ORACLES, (ideal_expected, ideal_log)):
            values = (member @ ideal).toarray() / exposure[:, None]
            predictions[oracle] = transform_prediction(values, labels)
        for replicate in range(args.replicates):
            a, b = simulated_halves(block, gene_indices[start:stop], args.specimen, args.base_seed, replicate)
            a_view, b_view = (member @ a).tocsr(), (member @ b).tocsr()
            aa, bb, rna_eligible = transform_rna_pair(a_view, b_view, exposure, labels)
            # Prediction-independent amplitude estimate in the unchanged RNA coordinates.
            rna_cross_sum[start:stop] += np.sum(aa * bb, axis=0, dtype=np.float64)
            # Each oracle has its original prediction/RNA eligibility intersection.
            for oracle in ORACLES:
                pp, pred_eligible = predictions[oracle]
                eligible = pred_eligible & rna_eligible
                local_a, local_b, local_p = aa.copy(), bb.copy(), pp.copy()
                local_a[:, ~eligible] = 0
                local_b[:, ~eligible] = 0
                local_p[:, ~eligible] = 0
                eligible_counts[oracle][replicate] += int(eligible.sum())
                if args.mode == "amplitude_matched":
                    cross = fixed_cross_mean[start:stop]
                    local_p = match_prediction_amplitude(local_p, cross)
                    negative_cross_counts[oracle][replicate] += int(np.sum((cross < 0) & eligible))
                add = lazy_band_grams(diffusion, {"A": local_a, "B": local_b, "P": local_p}, args.device)
                for band in BANDS:
                    for pair in PAIRS:
                        terms[oracle][replicate][band][pair] += add[band][pair]
        print(json.dumps({"stage": "gene_block_complete", "specimen": args.specimen,
                          "view": args.view_cache.name, "mode": args.mode,
                          "genes_done": stop, "genes_total": len(genes),
                          "elapsed_seconds": time.monotonic()-started}), flush=True)
    rows, summaries = [], []
    for oracle in ORACLES:
        for replicate in range(args.replicates):
            split = terms[oracle][replicate]
            split["B4+B5"] = {pair: split["B4"][pair]+split["B5"][pair] for pair in PAIRS}
            for band, moments in split.items():
                rows.append(dict(specimen=args.specimen, patient=metadata["patient"],
                                 view=args.view_cache.name, mode=args.mode, oracle=oracle, scope=args.scope,
                                 band=band, replicate=replicate, nodes=len(labels), genes=len(genes),
                                 eligible_genes=int(eligible_counts[oracle][replicate]),
                                 negative_fullband_cross_genes=int(negative_cross_counts[oracle][replicate]),
                                 **summarize_terms(moments)))
        for band in list(BANDS) + ["B4+B5"]:
            mean = {pair: np.mean([t[band][pair] for t in terms[oracle]]) for pair in PAIRS}
            summaries.append(dict(specimen=args.specimen, patient=metadata["patient"],
                                  view=args.view_cache.name, mode=args.mode, oracle=oracle, scope=args.scope,
                                  band=band, nodes=len(labels), genes=len(genes),
                                  eligible_genes_min=int(eligible_counts[oracle].min()),
                                  eligible_genes_max=int(eligible_counts[oracle].max()),
                                  **summarize_terms(mean)))
    raw, summary = pd.DataFrame(rows), pd.DataFrame(summaries)
    distribution = raw.groupby(["oracle", "band"]).q_ideal.agg(
        replica_q_mean="mean", replica_q_min="min", replica_q_max="max", defined_replicas="count").reset_index()
    summary = summary.merge(distribution, on=["oracle", "band"], validate="one_to_one")
    atomic_tsv(args.output / "REPLICATE_METRICS.tsv", raw)
    atomic_tsv(args.output / "SECTION_METRICS.tsv", summary)
    np.savez_compressed(args.output / "RNA_FULLBAND_MOMENTS.npz",
                        mean_fullband_cross=rna_cross_sum / args.replicates,
                        genes=genes.astype(str), canonical_gene_indices=gene_indices,
                        canonical_cell_indices=canonical_rows, specimen=np.array(args.specimen),
                        view=np.array(args.view_cache.name), base_seed=np.array(args.base_seed),
                        replicates=np.array(args.replicates))
    primary = summary[summary.oracle.eq("expected_half_log")]
    failed = primary[~np.isfinite(primary.q_ideal) | primary.q_ideal.lt(.95)]
    atomic_json(args.output / "COMPLETE.json", dict(status="COMPLETE", **binding,
        elapsed_seconds=time.monotonic()-started, ideal_gate_pass=failed.empty,
        gate_threshold=.95, failed_primary_bands=failed.band.tolist(),
        correction_needed=(args.mode == "original" and not failed.empty),
        no_model_fitting=True, no_original_artifacts_modified=True,
        aggregation_caveat="mean of ideal cell-level log predictions differs in general from log of pooled RNA counts per cell; this original operator order is retained"))
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
