#!/usr/bin/env python3
"""Measure split-half RNA support on native cells and fixed physical grids."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from .cell_core import (
    BANDS,
    ContractError,
    add_terms,
    atomic_json,
    atomic_tsv,
    build_lazy_graph,
    combine_bands,
    empty_terms,
    lazy_band_grams,
    load_canonical,
    load_protocol,
    split_counts,
    support_from_terms,
    transform_rna_pair,
)


PAIRS = (("A", "A"), ("A", "B"), ("B", "B"))


@dataclass
class View:
    name: str
    population: str
    scale_um: float
    membership: sparse.csr_matrix
    coords_um: np.ndarray
    exposure: np.ndarray
    diffusion: sparse.csr_matrix
    retained: np.ndarray
    labels: np.ndarray
    receipt: dict[str, object]


def make_membership(coords_um: np.ndarray, bin_um: int) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
    xy = np.asarray(coords_um, np.float64)
    origin = xy.min(axis=0)
    grid = np.floor((xy - origin[None, :]) / float(bin_um)).astype(np.int64)
    _, inverse = np.unique(grid, axis=0, return_inverse=True)
    membership = sparse.coo_matrix(
        (np.ones(len(inverse), np.float32), (inverse, np.arange(len(inverse)))),
        shape=(int(inverse.max()) + 1, len(inverse)),
    ).tocsr()
    exposure = np.asarray(membership.sum(axis=1)).reshape(-1)
    coords = np.column_stack(
        [np.asarray(membership @ xy[:, axis]).reshape(-1) / exposure for axis in (0, 1)]
    )
    return membership, coords, exposure


def create_view(
    name: str,
    population: str,
    scale_um: float,
    membership: sparse.csr_matrix,
    coords_um: np.ndarray,
    exposure: np.ndarray,
    protocol: dict[str, object],
) -> View:
    graph = protocol["graph"]
    max_edge = float(graph["native_max_edge_um"]) if scale_um == 0 else 1.5 * scale_um
    diffusion, retained, labels, receipt = build_lazy_graph(
        coords_um,
        int(graph["neighbors"]),
        max_edge,
        int(graph["minimum_component_cells"]),
    )
    receipt.update({"view": name, "population": population, "scale_um": scale_um})
    return View(
        name=name,
        population=population,
        scale_um=scale_um,
        membership=membership,
        coords_um=coords_um,
        exposure=np.asarray(exposure, np.float64),
        diffusion=diffusion,
        retained=retained,
        labels=labels,
        receipt=receipt,
    )


def cache_view(root: Path, specimen: str, view: View) -> None:
    output = root / specimen / view.name
    output.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(output / "membership.npz", view.membership, compressed=True)
    sparse.save_npz(output / "lazy_diffusion.npz", view.diffusion, compressed=True)
    np.save(output / "coords_um.npy", view.coords_um)
    np.save(output / "exposure.npy", view.exposure)
    np.save(output / "retained.npy", view.retained)
    np.save(output / "component_labels.npy", view.labels)
    atomic_json(output / "receipt.json", view.receipt)


def terms_to_rows(
    terms: dict[str, dict[tuple[str, str], float]], specimen: str, patient: str,
    view: View, seed: int | str,
) -> list[dict[str, object]]:
    support = pd.DataFrame(support_from_terms(terms))
    support = pd.concat([support, pd.DataFrame([combine_bands(support, ("B4", "B5"), "support")])])
    rows: list[dict[str, object]] = []
    for row in support.to_dict("records"):
        rows.append(
            {
                "specimen": specimen,
                "patient": patient,
                "view": view.name,
                "population": view.population,
                "output_grid_um": view.scale_um,
                "seed": seed,
                "nodes_retained": int(view.receipt["nodes_retained"]),
                **row,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--specimen", required=True)
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--scoring-gene-list", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--gene-block", type=int, default=32)
    args = parser.parse_args()

    protocol = load_protocol(args.protocol)
    counts, cells, genes, metadata = load_canonical(args.canonical_root / args.specimen)
    coords = cells[["x_um", "y_um"]].to_numpy(np.float64)
    identity = sparse.eye(len(cells), format="csr", dtype=np.float32)
    views: list[View] = [
        create_view("native_all", "all", 0.0, identity, coords, np.ones(len(cells)), protocol)
    ]
    neoplastic = cells.cell_class.astype(str).eq(protocol["proliferation"]["population"]).to_numpy()
    if neoplastic.sum() >= int(protocol["graph"]["minimum_component_cells"]):
        source = np.flatnonzero(neoplastic)
        membership = sparse.coo_matrix(
            (np.ones(len(source), np.float32), (np.arange(len(source)), source)),
            shape=(len(source), len(cells)),
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
        cache_view(args.cache_root, args.specimen, view)
        print(
            json.dumps(
                {"stage": "graph_ready", "view": view.name, **view.receipt},
                ensure_ascii=False,
            ),
            flush=True,
        )

    positive_gene = np.asarray(counts.sum(axis=0)).reshape(-1) > 0
    if args.scoring_gene_list is not None:
        requested = pd.read_csv(args.scoring_gene_list)
        requested_values = set(requested["gene"] if "gene" in requested else requested.iloc[:, -1])
        positive_gene &= genes.gene_symbol.astype(str).isin(requested_values).to_numpy()
    gene_indices = np.flatnonzero(positive_gene)
    if not len(gene_indices):
        raise ContractError("NO_POSITIVE_GENES")
    seeds = [int(seed) for seed in protocol["split_seeds"]]
    seed_rows: list[dict[str, object]] = []
    final_rows: list[dict[str, object]] = []
    receipt_rows: list[dict[str, object]] = []

    for view in views:
        total_over_seeds = empty_terms(PAIRS)
        for seed in seeds:
            a, b = split_counts(counts, args.specimen, seed)
            a_view = (view.membership @ a).tocsr()[view.retained]
            b_view = (view.membership @ b).tocsr()[view.retained]
            exposure = view.exposure[view.retained]
            seed_terms = empty_terms(PAIRS)
            eligible_genes = 0
            for start in range(0, len(gene_indices), args.gene_block):
                block = gene_indices[start : start + args.gene_block]
                aa, bb, eligible = transform_rna_pair(
                    a_view[:, block], b_view[:, block], exposure, view.labels
                )
                eligible_genes += int(eligible.sum())
                add_terms(
                    seed_terms,
                    lazy_band_grams(view.diffusion, {"A": aa, "B": bb}, args.device),
                )
            add_terms(total_over_seeds, seed_terms)
            local_rows = terms_to_rows(seed_terms, args.specimen, metadata["patient"], view, seed)
            for row in local_rows:
                row["eligible_genes"] = eligible_genes
            seed_rows.extend(local_rows)
            print(
                json.dumps(
                    {
                        "stage": "rna_support_seed_ready",
                        "view": view.name,
                        "seed": seed,
                        "eligible_genes": eligible_genes,
                    }
                ),
                flush=True,
            )

        for band in BANDS:
            for pair in PAIRS:
                total_over_seeds[band][pair] /= len(seeds)
        local_final = terms_to_rows(
            total_over_seeds, args.specimen, metadata["patient"], view, "cross_product_mean"
        )
        for row in local_final:
            row["eligible_genes"] = int(
                pd.DataFrame(seed_rows).loc[
                    lambda x: (x.view == view.name) & (x.band == row["band"]), "eligible_genes"
                ].min()
            )
        final_rows.extend(local_final)
        receipt_rows.append({"specimen": args.specimen, "patient": metadata["patient"], **view.receipt})

    args.results.mkdir(parents=True, exist_ok=True)
    seed_frame = pd.DataFrame(seed_rows)
    final_frame = pd.DataFrame(final_rows)
    receipt_frame = pd.DataFrame(receipt_rows)
    atomic_tsv(args.results / f"{args.specimen}_RNA_SUPPORT_SEED_BAND.tsv", seed_frame)
    atomic_tsv(args.results / f"{args.specimen}_RNA_SUPPORT_SPECIMEN_BAND.tsv", final_frame)
    view_summary = final_frame.loc[final_frame.band.eq("B4+B5")].copy()
    atomic_tsv(args.results / f"{args.specimen}_RNA_SUPPORT_SPECIMEN_VIEW.tsv", view_summary)
    atomic_tsv(args.results / f"{args.specimen}_GRAPH_RECEIPTS.tsv", receipt_frame)
    receipt = {
        "status": "PASS",
        "protocol_id": protocol["protocol_id"],
        "specimen": args.specimen,
        "patient": metadata["patient"],
        "split_seeds": len(seeds),
        "genes_positive": int(len(gene_indices)),
        "views": [view.name for view in views],
        "primary_rna_support_B4_B5": float(
            view_summary.loc[view_summary.view.eq("native_neoplastic"), "rna_split_agreement"].iloc[0]
        ),
    }
    atomic_json(args.results / f"{args.specimen}_RNA_SUPPORT_RECEIPT.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
