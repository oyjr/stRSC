"""Training-only non-neighbour loss graphs; no fitting or evaluation on import.

Input node order and local edges must come from the unchanged model adapter.
This module never constructs the graph used for model message passing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


GRAPH_SEED = 20260905


def _canonical(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


def _validated_edges(edge_index: np.ndarray, n_nodes: int) -> np.ndarray:
    edges = np.asarray(edge_index)
    if edges.ndim != 2 or edges.shape[0] != 2 or edges.shape[1] < 2:
        raise ValueError("Expected at least two undirected edges in a (2, m) array")
    if not np.issubdtype(edges.dtype, np.integer):
        raise ValueError("Node indices must have integer dtype")
    edges = np.sort(edges.astype(np.int64, copy=True), axis=0)
    if edges.min() < 0 or edges.max() >= n_nodes or np.any(edges[0] == edges[1]):
        raise ValueError("Invalid node index or self-loop")
    if len(set(map(tuple, edges.T))) != edges.shape[1]:
        raise ValueError("Duplicate undirected local edges")
    return edges


def rewire_non_neighbours(
    coords: np.ndarray,
    local_edge_index: np.ndarray,
    radius: float,
    *,
    seed: int = GRAPH_SEED,
    restarts: int = 8,
    partner_attempts: int = 256,
    mixing_swaps_per_edge: int = 10,
) -> tuple[np.ndarray, dict]:
    """Keep m and every node's degree; require all new distances > radius.

    A two-edge switch (a,b),(c,d) -> (a,c),(b,d) or (a,d),(b,c)
    preserves node incidence exactly. First remove all original local edges,
    then perform additional valid switches. This is a reproducible rewiring,
    NOT a claimed uniform sample from all graphs with this degree sequence.
    Failure raises an error, never relaxes the spatial rule or degree sequence.
    """
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 2 or not np.isfinite(coords).all():
        raise ValueError("Finite coordinates with shape (n, 2) required")
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("Positive finite original local radius required")
    if restarts < 1 or partner_attempts < 1 or mixing_swaps_per_edge < 0:
        raise ValueError("Invalid fixed construction budget")
    original = _validated_edges(local_edge_index, len(coords))
    n_edges = original.shape[1]
    original_set = set(map(tuple, original.T))
    degree = np.bincount(original.ravel(), minlength=len(coords))
    if np.any(degree > len(coords) - 1 - degree):
        raise ValueError("Complement capacity insufficient for the original degree sequence")
    radius_squared = float(radius) ** 2
    original_distances = np.sum((coords[original[0]] - coords[original[1]]) ** 2, axis=1)
    if np.any(original_distances > radius_squared * (1 + 1e-12)):
        raise ValueError("Input local graph contains an edge beyond its stated radius")

    def nonlocal_pair(pair: tuple[int, int]) -> bool:
        delta = coords[pair[0]] - coords[pair[1]]
        return float(delta @ delta) > radius_squared

    for restart in range(restarts):
        rng = np.random.default_rng(np.random.SeedSequence([seed, restart]))
        edges = original.T.copy()
        current = set(original_set)
        bad = set(range(n_edges))
        switches = 0

        def switch(i: int, j: int) -> bool:
            nonlocal switches
            if i == j:
                return False
            a, b = map(int, edges[i])
            c, d = map(int, edges[j])
            if len({a, b, c, d}) != 4:
                return False
            proposals = [(_canonical(a, c), _canonical(b, d)),
                         (_canonical(a, d), _canonical(b, c))]
            if rng.integers(2):
                proposals.reverse()
            for first, second in proposals:
                if first in current or second in current:
                    continue
                if not nonlocal_pair(first) or not nonlocal_pair(second):
                    continue
                current.remove((a, b))
                current.remove((c, d))
                current.add(first)
                current.add(second)
                edges[i], edges[j] = first, second
                bad.discard(i)
                bad.discard(j)
                switches += 1
                return True
            return False

        for i in rng.permutation(n_edges):
            i = int(i)
            if i not in bad:
                continue
            for _ in range(partner_attempts):
                if switch(i, int(rng.integers(n_edges))):
                    break
            else:
                break
        if bad:
            continue
        desired_mix = mixing_swaps_per_edge * n_edges
        before_mix = switches
        for _ in range(max(1, desired_mix * 20)):
            if switches - before_mix >= desired_mix:
                break
            switch(int(rng.integers(n_edges)), int(rng.integers(n_edges)))
        if switches - before_mix < desired_mix:
            continue
        # Keep edge slots: adapters reuse the same edge-index schedule. Sorting
        # by the new endpoints would change this schedule unnecessarily.
        result = np.ascontiguousarray(edges.T)
        result_degree = np.bincount(result.ravel(), minlength=len(coords))
        if not np.array_equal(degree, result_degree) or len(current) != n_edges:
            raise AssertionError("Two-edge switching violated incidence invariants")
        if original_set & current:
            raise AssertionError("A local edge survived")
        distances = np.linalg.norm(coords[result[0]] - coords[result[1]], axis=1)
        if not np.all(distances > radius):
            raise AssertionError("Non-neighbour distance rule violated")
        return result, {
            "status": "GRAPH_ONLY_NO_TRAINING_NO_EVALUATION",
            "algorithm": "degree_preserving_two_edge_switch_v1",
            "seed": int(seed), "restart": restart,
            "nodes": len(coords), "edges": n_edges,
            "local_radius": float(radius), "degree_preserved": True,
            "original_edge_overlap": 0,
            "min_new_distance": float(distances.min()),
            "median_new_distance": float(np.median(distances)),
            "accepted_switches": switches,
            "post_removal_mixing_switches": switches - before_mix,
            "uniform_graph_sampling_claimed": False,
        }
    raise RuntimeError(
        "NONNEIGHBOUR_GRAPH_CONSTRUCTION_FAILED_WITH_FIXED_BUDGET; "
        "do not relax constraints or claim mathematical infeasibility"
    )


def training_diagnostics(target: np.ndarray, local: np.ndarray, nonlocal_edges: np.ndarray,
                         initial_prediction: np.ndarray | None = None) -> dict:
    """Optional train-only descriptions; never used to tune rho or select edges.

    Residual-gradient norms refer to derivatives with respect to predictions,
    not parameter-gradient norms. Only this module's caller supplies arrays.
    """
    target = np.asarray(target, dtype=np.float64)
    if target.ndim != 2 or not np.isfinite(target).all():
        raise ValueError("Finite training target matrix required")
    out = {}
    for name, edges in (("local", local), ("nonlocal", nonlocal_edges)):
        edges = _validated_edges(edges, len(target))
        delta = target[edges[0]] - target[edges[1]]
        values = {
            "target_difference_mean_square": float(np.mean(delta ** 2)),
            "absolute_target_difference_quantiles_10_50_90": np.quantile(
                np.abs(delta), [0.1, 0.5, 0.9]).tolist(),
        }
        if initial_prediction is not None:
            initial = np.asarray(initial_prediction, dtype=np.float64)
            if initial.shape != target.shape or not np.isfinite(initial).all():
                raise ValueError("Aligned finite initial training predictions required")
            residual = initial - target
            difference = residual[edges[0]] - residual[edges[1]]
            gradient = np.zeros_like(residual)
            term = 2 * difference / difference.size
            np.add.at(gradient, edges[0], term)
            np.add.at(gradient, edges[1], -term)
            values.update({
                "initial_pair_loss": float(np.mean(difference ** 2)),
                "initial_prediction_gradient_l2": float(np.linalg.norm(gradient)),
            })
        out[name] = values
    return out


def matched_context_chunks(local: np.ndarray, nonlocal_edges: np.ndarray,
                           *, max_nodes: int) -> list[dict[str, np.ndarray]]:
    """Hist2ST integration primitive: identical forward rows for the two arms.

    Call on corresponding edge slots AFTER the unchanged pack permutation.
    Each returned context must be loaded once per arm with identical row order
    and spatial context adjacency; only loss_edge_index is arm-specific.
    This changes old Hist2ST context packing and requires a NEW local arm.
    """
    local = np.asarray(local, dtype=np.int64)
    remote = np.asarray(nonlocal_edges, dtype=np.int64)
    if local.ndim != 2 or local.shape[0] != 2 or local.shape != remote.shape:
        raise ValueError("Aligned edge slots required")
    if local.shape[1] < 1 or max_nodes < 4 or local.min() < 0 or remote.min() < 0:
        raise ValueError("Nonempty edges and max_nodes >= 4 required")
    result = []
    start = 0
    nodes: set[int] = set()

    def append_chunk(stop: int) -> None:
        rows = np.array(sorted(nodes), dtype=np.int64)
        result.append({
            "forward_rows": rows,
            "edge_slots": np.arange(start, stop, dtype=np.int64),
            "local_loss_edge_index": np.searchsorted(rows, local[:, start:stop]),
            "nonlocal_loss_edge_index": np.searchsorted(rows, remote[:, start:stop]),
        })

    for slot in range(local.shape[1]):
        endpoints = set(map(int, np.concatenate((local[:, slot], remote[:, slot]))))
        if nodes and len(nodes | endpoints) > max_nodes:
            append_chunk(slot)
            start, nodes = slot, set()
        nodes.update(endpoints)
    append_chunk(local.shape[1])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path,
                        help="Training-only NPZ: coords, local_edge_index, radius")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    receipt = args.output.with_suffix(".json")
    if args.output.exists() or receipt.exists():
        raise FileExistsError("Preserve existing graph: choose an unoccupied output path")
    with np.load(args.input, allow_pickle=False) as data:
        edges, metadata = rewire_non_neighbours(
            data["coords"], data["local_edge_index"], float(data["radius"].item()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        np.savez_compressed(handle, nonlocal_edge_index=edges)
    receipt.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata))


if __name__ == "__main__":
    main()
