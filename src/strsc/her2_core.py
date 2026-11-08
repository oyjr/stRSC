"""Source-half question construction and ordering scores for HER2ST.

Adapted from the study's source-isolated evaluator. Random-seed identifiers and
selection thresholds are retained to preserve its numerical behavior.
"""
from __future__ import annotations
import hashlib
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


DIRECTIONS = ("A_TO_B", "B_TO_A")

GENE_FOLDS = 10

PCA_COMPONENTS = 20

NEIGHBORS = 8

COMPOSITION_DISTANCE_QUANTILE = 0.25

DEPTH_RATIO_MAX = 1.25

SOURCE_DELTA_QUANTILE = 0.75

MIN_ELIGIBLE_POSITIONS = 20

MIN_DETECTION_FRACTION = 0.05

MIN_TOTAL_MOLECULES = 20

MIN_JUDGED_QUESTIONS = 10

QUESTION_KEY = [
    "patient_id",
    "section_id",
    "seed",
    "direction",
    "gene",
    "left_position",
    "right_position",
]

ARENA_KEY = ["arena", *QUESTION_KEY]

DISTANCE_LABELS = ("Q1", "Q2", "Q3", "Q4")


def stable_int(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big")


def gene_fold(gene: str) -> int:
    return stable_int(f"G0-038-FIG01-GENE-FOLD\x00{gene}") % GENE_FOLDS


def split_seed(section: str, seed: int) -> int:
    return stable_int(f"G0-038-FIG01-SPLIT\x00{section}\x00{seed}")


def split_molecules(
    counts: np.ndarray, section: str, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(counts)
    if values.ndim != 2 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError("F1_INTEGER_COUNT_MATRIX_REQUIRED")
    if np.any(values < 0):
        raise ValueError("F1_NEGATIVE_INTEGER_COUNT")
    rng = np.random.default_rng(split_seed(section, seed))
    first = rng.binomial(values, 0.5).astype(np.int32)
    second = values.astype(np.int32) - first
    if not np.array_equal(first.astype(np.int64) + second.astype(np.int64), values):
        raise RuntimeError("F1_MOLECULE_SPLIT_CLOSURE_FAILED")
    return (first, second)


def _splitmix64(values: np.ndarray | np.uint64) -> np.ndarray:
    """Vectorised SplitMix64 finalizer with deliberate uint64 wraparound."""
    x = np.asarray(values, dtype=np.uint64).copy()
    with np.errstate(over="ignore"):
        x = x + np.uint64(11400714819323198485)
        x = (x ^ x >> np.uint64(30)) * np.uint64(13787848793156543929)
        x = (x ^ x >> np.uint64(27)) * np.uint64(10723151780598845931)
        x = x ^ x >> np.uint64(31)
    return x


def stable_candidate_hashes(
    *,
    patient: str,
    section: str,
    gene: str,
    seed: int,
    quartile: str,
    directions: np.ndarray,
    left_positions: Sequence[str],
    right_positions: Sequence[str],
) -> np.ndarray:
    """Return deterministic uint64 priorities without per-candidate Python SHA.

    The stratum fields seed the mixer.  The required payload fields
    ``direction,left_position,right_position`` are represented by stable SHA
    token hashes and mixed in their ordered slots.  Canonical string fields are
    retained as the deterministic collision tie-break in ``stable_cap_indices``.
    """
    n = len(directions)
    if len(left_positions) != n or len(right_positions) != n:
        raise ValueError("F1_HASH_AXIS_MISMATCH")
    base = np.uint64(
        stable_int(
            f"G0-039-F1-BROAD-HASH-V1\x00{patient}\x00{section}\x00{gene}\x00{seed}\x00{quartile}"
        )
    )
    left = np.fromiter(
        (stable_int(f"LEFT\x00{x}") for x in left_positions), dtype=np.uint64, count=n
    )
    right = np.fromiter(
        (stable_int(f"RIGHT\x00{x}") for x in right_positions), dtype=np.uint64, count=n
    )
    direction = np.fromiter(
        (stable_int(f"DIRECTION\x00{x}") for x in directions), dtype=np.uint64, count=n
    )
    with np.errstate(over="ignore"):
        combined = (
            base
            ^ _splitmix64(direction)
            ^ _splitmix64(left) * np.uint64(15485907386658061715)
            ^ _splitmix64(right) * np.uint64(11935478331139982891)
        )
    return _splitmix64(combined)


def stable_cap_indices(
    hashes: np.ndarray,
    directions: Sequence[str],
    left_positions: Sequence[str],
    right_positions: Sequence[str],
    cap: int,
) -> np.ndarray:
    """Select the lowest hashes with exact canonical handling at a hash tie."""
    h = np.asarray(hashes, dtype=np.uint64)
    n = len(h)
    if not len(directions) == len(left_positions) == len(right_positions) == n:
        raise ValueError("F1_CAP_AXIS_MISMATCH")
    if cap <= 0 or n == 0:
        return np.empty(0, dtype=np.int64)
    if cap >= n:
        selected = np.arange(n, dtype=np.int64)
    else:
        boundary = np.partition(h, cap - 1)[cap - 1]
        below = np.flatnonzero(h < boundary)
        tied = np.flatnonzero(h == boundary)
        remaining = cap - len(below)
        if remaining < 0:
            raise RuntimeError("F1_HASH_PARTITION_INVALID")
        if remaining:
            tie_order = sorted(
                tied.tolist(),
                key=lambda i: (
                    str(directions[i]),
                    str(left_positions[i]),
                    str(right_positions[i]),
                ),
            )
            selected = np.concatenate(
                [below, np.asarray(tie_order[:remaining], dtype=np.int64)]
            )
        else:
            selected = below
    order = sorted(
        selected.tolist(),
        key=lambda i: (
            int(h[i]),
            str(directions[i]),
            str(left_positions[i]),
            str(right_positions[i]),
        ),
    )
    return np.asarray(order, dtype=np.int64)


def physical_distance_geometry(
    coordinates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Freeze all section-local unordered pairs and right-closed quartiles."""
    xy = np.asarray(coordinates, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 2:
        raise ValueError("F1_REGISTERED_PIXEL_COORDINATES_INVALID")
    if not np.isfinite(xy).all():
        raise ValueError("F1_REGISTERED_PIXEL_COORDINATES_NONFINITE")
    left, right = np.triu_indices(len(xy), k=1)
    distance = np.sqrt(np.square(xy[left] - xy[right]).sum(axis=1))
    cutpoints = np.quantile(distance, [0.25, 0.5, 0.75], method="linear")
    quartile = np.searchsorted(cutpoints, distance, side="left").astype(np.int8) + 1
    return (
        left.astype(np.int32),
        right.astype(np.int32),
        distance.astype(np.float64),
        quartile,
        cutpoints.astype(np.float64),
    )


def assign_distance_quartile(distance: np.ndarray, cutpoints: np.ndarray) -> np.ndarray:
    values = np.asarray(distance, dtype=np.float64)
    cuts = np.asarray(cutpoints, dtype=np.float64)
    if cuts.shape != (3,) or np.any(np.diff(cuts) < 0):
        raise ValueError("F1_DISTANCE_CUTPOINTS_INVALID")
    return np.searchsorted(cuts, values, side="left").astype(np.int8) + 1


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    truth = np.asarray(y_true)
    pred = np.asarray(y_pred)
    if truth.shape != pred.shape or truth.ndim != 1 or (not len(truth)):
        return math.nan
    recalls: list[float] = []
    for label in (-1, 1):
        selected = truth == label
        if selected.any():
            recalls.append(float(np.mean(pred[selected] == label)))
    return float(np.mean(recalls)) if recalls else math.nan


@dataclass(frozen=True)
class SectionData:
    patient: str
    section: str
    positions: tuple[str, ...]
    genes: tuple[str, ...]
    methods: tuple[str, ...]
    counts: np.ndarray
    prediction: np.ndarray
    common_support: np.ndarray
    regions: np.ndarray
    pixel_xy: np.ndarray
    fold_axis: np.ndarray
    pair_left: np.ndarray
    pair_right: np.ndarray
    pair_distance: np.ndarray
    pair_quartile: np.ndarray
    distance_cutpoints: np.ndarray


def _direction_arrays(
    first: np.ndarray, second: np.ndarray, direction: str
) -> tuple[np.ndarray, np.ndarray]:
    if direction == "A_TO_B":
        return (first, second)
    if direction == "B_TO_A":
        return (second, first)
    raise ValueError(f"F1_DIRECTION_INVALID:{direction}")


def _position_indices(
    values: Sequence[Any], position_index: Mapping[str, int]
) -> np.ndarray:
    try:
        return np.fromiter(
            (position_index[str(value)] for value in values),
            dtype=np.int32,
            count=len(values),
        )
    except KeyError as error:
        raise RuntimeError(
            f"F1_QUESTION_POSITION_NOT_ON_COMMON_AXIS:{error.args[0]}"
        ) from error


def _gene_indices(values: Sequence[Any], gene_index: Mapping[str, int]) -> np.ndarray:
    try:
        return np.fromiter(
            (gene_index[str(value)] for value in values),
            dtype=np.int32,
            count=len(values),
        )
    except KeyError as error:
        raise RuntimeError(
            f"F1_QUESTION_GENE_NOT_ON_COMMON_AXIS:{error.args[0]}"
        ) from error


def source_gene_eligibility(
    section_data: SectionData, source: np.ndarray
) -> np.ndarray:
    """Apply the frozen numeric eligibility thresholds to one source half only."""
    values = np.asarray(source)
    if values.shape != section_data.counts.shape or not np.issubdtype(
        values.dtype, np.integer
    ):
        raise ValueError("F1_SOURCE_ELIGIBILITY_AXIS_OR_DTYPE_INVALID")
    if np.any(values < 0):
        raise ValueError("F1_SOURCE_ELIGIBILITY_NEGATIVE_COUNT")
    return (
        (section_data.common_support.sum(axis=0) >= MIN_ELIGIBLE_POSITIONS)
        & (np.mean(values > 0, axis=0) >= MIN_DETECTION_FRACTION)
        & (values.sum(axis=0) >= MIN_TOTAL_MOLECULES)
    )


def source_composition_scores(
    source: np.ndarray, feature_mask: np.ndarray, random_state: int
) -> np.ndarray:
    """Encode source-half composition using genes outside the target fold."""
    feature = np.asarray(source[:, feature_mask], dtype=np.float64)
    depth = feature.sum(axis=1)
    feature = np.log1p(
        np.divide(
            feature * 10000.0,
            depth[:, None],
            out=np.zeros_like(feature),
            where=depth[:, None] > 0,
        )
    )
    variance = feature.var(axis=0)
    feature = feature[:, variance > 0]
    components = min(PCA_COMPONENTS, len(feature) - 1, feature.shape[1])
    if components < 2:
        return np.empty((len(feature), 0), dtype=np.float64)
    scores = PCA(
        n_components=components,
        svd_solver="randomized",
        iterated_power=3,
        random_state=random_state % (2**32 - 1),
    ).fit_transform(feature)
    scale = scores.std(axis=0, ddof=1)
    return np.divide(
        scores, scale[None, :], out=np.zeros_like(scores), where=scale[None, :] > 0
    )


def source_composition_matched_pairs(
    source: np.ndarray, feature_mask: np.ndarray, random_state: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rebuild the frozen matcher using source-half RNA only."""
    scores = source_composition_scores(source, feature_mask, random_state)
    source_depth = source[:, feature_mask].sum(axis=1).astype(np.float64)
    valid = (source_depth > 0) & np.isfinite(scores).all(axis=1)
    valid_index = np.flatnonzero(valid)
    if len(valid_index) < 2 or scores.shape[1] == 0:
        return (
            np.empty((0, 2), dtype=np.int32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.float32),
        )
    n_neighbors = min(NEIGHBORS + 1, len(valid_index))
    distance, neighbor = (
        NearestNeighbors(n_neighbors=n_neighbors, algorithm="auto", metric="euclidean")
        .fit(scores[valid])
        .kneighbors(scores[valid])
    )
    candidate: dict[tuple[int, int], float] = {}
    for row in range(len(valid_index)):
        left = int(valid_index[row])
        for column in range(1, n_neighbors):
            right = int(valid_index[int(neighbor[row, column])])
            pair = (left, right) if left < right else (right, left)
            value = float(distance[row, column])
            old = candidate.get(pair)
            if old is None or value < old:
                candidate[pair] = value
    if not candidate:
        return (
            np.empty((0, 2), dtype=np.int32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.float32),
        )
    pairs = np.asarray(sorted(candidate), dtype=np.int32)
    distances = np.asarray([candidate[tuple(pair)] for pair in pairs], dtype=np.float64)
    left_depth = source_depth[pairs[:, 0]]
    right_depth = source_depth[pairs[:, 1]]
    depth_ratio = np.maximum(left_depth / right_depth, right_depth / left_depth)
    finite = (
        np.isfinite(distances)
        & np.isfinite(depth_ratio)
        & (depth_ratio <= DEPTH_RATIO_MAX)
    )
    threshold = (
        float(np.quantile(distances[finite], COMPOSITION_DISTANCE_QUANTILE))
        if finite.any()
        else math.nan
    )
    selected = finite & (distances <= threshold)
    pairs = pairs[selected]
    distances = distances[selected]
    depth_ratio = depth_ratio[selected]
    order = (
        np.lexsort((pairs[:, 1], pairs[:, 0], distances))
        if len(pairs)
        else np.empty(0, dtype=int)
    )
    used = np.zeros(len(source), dtype=bool)
    keep: list[int] = []
    for index in order:
        left, right = pairs[index]
        if not used[left] and (not used[right]):
            used[left] = True
            used[right] = True
            keep.append(int(index))
    keep_array = np.asarray(keep, dtype=np.int64)
    return (
        pairs[keep_array],
        distances[keep_array].astype(np.float32),
        depth_ratio[keep_array].astype(np.float32),
    )


def _region_fields(
    section_data: SectionData, left: np.ndarray, right: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    left_region = section_data.regions[left].astype(object)
    right_region = section_data.regions[right].astype(object)
    left_nonempty = np.fromiter(
        (bool(str(value).strip()) for value in left_region), bool, len(left_region)
    )
    right_nonempty = np.fromiter(
        (bool(str(value).strip()) for value in right_region), bool, len(right_region)
    )
    return (
        left_region,
        right_region,
        left_nonempty & right_nonempty & (left_region == right_region),
    )


def build_corrected_arena2_membership(
    *, section_data: SectionData, seed: int, first: np.ndarray, second: np.ndarray
) -> pd.DataFrame:
    """Build the inferential composition-matched arena without judge access."""
    rows: list[pd.DataFrame] = []
    for direction in DIRECTIONS:
        source, _ = _direction_arrays(first, second, direction)
        eligible = source_gene_eligibility(section_data, source)
        pair_by_fold: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for fold in range(GENE_FOLDS):
            pair_by_fold[fold] = source_composition_matched_pairs(
                source,
                section_data.fold_axis != fold,
                stable_int(
                    f"G0-038-FIG01-PCA\x00{section_data.section}\x00{seed}\x00{direction}\x00{fold}"
                ),
            )
        for gene_index, gene in enumerate(section_data.genes):
            if not eligible[gene_index]:
                continue
            pair, composition_distance, depth_ratio = pair_by_fold[
                int(section_data.fold_axis[gene_index])
            ]
            if not len(pair):
                continue
            valid_pair = (
                section_data.common_support[pair[:, 0], gene_index]
                & section_data.common_support[pair[:, 1], gene_index]
            )
            pair = pair[valid_pair]
            composition_distance = composition_distance[valid_pair]
            depth_ratio = depth_ratio[valid_pair]
            if not len(pair):
                continue
            source_delta = 2 * (
                source[pair[:, 0], gene_index].astype(np.int64)
                - source[pair[:, 1], gene_index].astype(np.int64)
            )
            nonzero = np.abs(source_delta) > 0
            if int(nonzero.sum()) < MIN_JUDGED_QUESTIONS:
                continue
            delta_threshold = float(
                np.quantile(np.abs(source_delta[nonzero]), SOURCE_DELTA_QUANTILE)
            )
            selected = nonzero & (np.abs(source_delta) >= delta_threshold)
            selected_pair = pair[selected]
            if not len(selected_pair):
                continue
            source_delta = source_delta[selected]
            composition_distance = composition_distance[selected]
            depth_ratio = depth_ratio[selected]
            left = selected_pair[:, 0]
            right = selected_pair[:, 1]
            distance = np.sqrt(
                np.square(
                    section_data.pixel_xy[left] - section_data.pixel_xy[right]
                ).sum(axis=1)
            )
            quartile = assign_distance_quartile(
                distance, section_data.distance_cutpoints
            )
            left_region, right_region, same_region = _region_fields(
                section_data, left, right
            )
            rows.append(
                pd.DataFrame(
                    {
                        "arena": "COMPOSITION_MATCHED_ORIGINAL",
                        "patient_id": section_data.patient,
                        "section_id": section_data.section,
                        "seed": seed,
                        "direction": direction,
                        "gene": gene,
                        "gene_fold": int(section_data.fold_axis[gene_index]),
                        "left_position": [section_data.positions[int(x)] for x in left],
                        "right_position": [
                            section_data.positions[int(x)] for x in right
                        ],
                        "source_delta_full_exposure": source_delta,
                        "composition_distance": composition_distance,
                        "source_depth_ratio": depth_ratio,
                        "source_delta_threshold": delta_threshold,
                        "physical_distance_registered_pixel": distance,
                        "physical_distance_quartile": [
                            DISTANCE_LABELS[x - 1] for x in quartile
                        ],
                        "left_pathology_region": left_region,
                        "right_pathology_region": right_region,
                        "same_pathology_region": same_region,
                        "question_membership_provenance": "SOURCE_HALF_ONLY_G0038_MATCHER_REBUILD",
                        "eligibility_provenance": "SOURCE_HALF_ONLY_UNSCALED_FROZEN_THRESHOLDS",
                        "sampling_hash_u64": np.zeros(len(left), dtype=np.uint64),
                    }
                )
            )
    if not rows:
        return pd.DataFrame()
    output = pd.concat(rows, ignore_index=True)
    if output.duplicated(ARENA_KEY).any():
        raise RuntimeError(
            f"F1_CORRECTED_ARENA2_KEY_DUPLICATE:{section_data.section}:{seed}"
        )
    return output


def strict_arena3_subset(arena2: pd.DataFrame) -> pd.DataFrame:
    required = {
        "arena",
        "left_pathology_region",
        "right_pathology_region",
        "same_pathology_region",
    }
    if not required.issubset(arena2.columns):
        raise ValueError(
            f"F1_ARENA3_ANNOTATION_MISSING:{sorted(required - set(arena2.columns))}"
        )
    left = arena2.left_pathology_region.astype("string").fillna("").str.strip()
    right = arena2.right_pathology_region.astype("string").fillna("").str.strip()
    keep = left.ne("") & right.ne("") & left.eq(right)
    if not arena2.same_pathology_region.astype(bool).eq(keep).all():
        raise RuntimeError("F1_ARENA3_REGION_FLAG_DRIFT")
    output = arena2.loc[keep].copy()
    output["arena"] = "JOINT_STRUCTURE_MATCHED"
    output[
        "question_membership_provenance"
    ] = "STRICT_SOURCE_ISOLATED_ARENA2_SAME_PATHOLOGY_REGION_SUBSET"
    if len(output) > len(arena2):
        raise RuntimeError("F1_ARENA3_NOT_SUBSET")
    return output


def _broad_candidates_one_direction(
    *, section_data: SectionData, source: np.ndarray, direction: str, gene_index: int
) -> dict[str, np.ndarray | float]:
    valid = (
        section_data.common_support[section_data.pair_left, gene_index]
        & section_data.common_support[section_data.pair_right, gene_index]
    )
    left = section_data.pair_left[valid]
    right = section_data.pair_right[valid]
    distance = section_data.pair_distance[valid]
    quartile = section_data.pair_quartile[valid]
    delta = 2 * (
        source[left, gene_index].astype(np.int64)
        - source[right, gene_index].astype(np.int64)
    )
    nonzero = np.abs(delta) > 0
    if not nonzero.any():
        return {
            "left": np.empty(0, np.int32),
            "right": np.empty(0, np.int32),
            "distance": np.empty(0, np.float64),
            "quartile": np.empty(0, np.int8),
            "delta": np.empty(0, np.int64),
            "direction": np.empty(0, object),
            "threshold": math.nan,
        }
    threshold = float(np.quantile(np.abs(delta[nonzero]), SOURCE_DELTA_QUANTILE))
    selected = nonzero & (np.abs(delta) >= threshold)
    count = int(selected.sum())
    return {
        "left": left[selected],
        "right": right[selected],
        "distance": distance[selected],
        "quartile": quartile[selected],
        "delta": delta[selected],
        "direction": np.repeat(direction, count),
        "threshold": threshold,
    }


def build_arena1(
    *,
    section_data: SectionData,
    seed: int,
    first: np.ndarray,
    second: np.ndarray,
    arena2: pd.DataFrame,
) -> pd.DataFrame:
    """Build broad source-only membership with direction-isolated caps."""
    cap = (
        arena2.groupby(
            ["direction", "gene", "physical_distance_quartile"], observed=True
        )
        .size()
        .to_dict()
    )
    rows: list[pd.DataFrame] = []
    for direction in DIRECTIONS:
        source, _ = _direction_arrays(first, second, direction)
        eligible = source_gene_eligibility(section_data, source)
        for gene_index, gene in enumerate(section_data.genes):
            if not eligible[gene_index]:
                continue
            if not any(
                (
                    int(cap.get((direction, gene, label), 0)) > 0
                    for label in DISTANCE_LABELS
                )
            ):
                continue
            candidate = _broad_candidates_one_direction(
                section_data=section_data,
                source=source,
                direction=direction,
                gene_index=gene_index,
            )
            left = np.asarray(candidate["left"], np.int32)
            right = np.asarray(candidate["right"], np.int32)
            distance = np.asarray(candidate["distance"], np.float64)
            quartile = np.asarray(candidate["quartile"], np.int8)
            source_delta = np.asarray(candidate["delta"], np.int64)
            directions = np.asarray(candidate["direction"], object)
            if not len(left):
                continue
            for quartile_number, quartile_label in enumerate(DISTANCE_LABELS, start=1):
                support_cap = int(cap.get((direction, gene, quartile_label), 0))
                if support_cap <= 0:
                    continue
                in_stratum = np.flatnonzero(quartile == quartile_number)
                if not len(in_stratum):
                    continue
                left_positions = [
                    section_data.positions[int(x)] for x in left[in_stratum]
                ]
                right_positions = [
                    section_data.positions[int(x)] for x in right[in_stratum]
                ]
                hashes = stable_candidate_hashes(
                    patient=section_data.patient,
                    section=section_data.section,
                    gene=gene,
                    seed=seed,
                    quartile=quartile_label,
                    directions=directions[in_stratum],
                    left_positions=left_positions,
                    right_positions=right_positions,
                )
                local_keep = stable_cap_indices(
                    hashes,
                    directions[in_stratum],
                    left_positions,
                    right_positions,
                    support_cap,
                )
                keep = in_stratum[local_keep]
                left_region, right_region, same_region = _region_fields(
                    section_data, left[keep], right[keep]
                )
                rows.append(
                    pd.DataFrame(
                        {
                            "arena": "BROAD_STRUCTURE_ALLOWED",
                            "patient_id": section_data.patient,
                            "section_id": section_data.section,
                            "seed": seed,
                            "direction": direction,
                            "gene": gene,
                            "gene_fold": int(section_data.fold_axis[gene_index]),
                            "left_position": [
                                section_data.positions[int(x)] for x in left[keep]
                            ],
                            "right_position": [
                                section_data.positions[int(x)] for x in right[keep]
                            ],
                            "source_delta_full_exposure": source_delta[keep],
                            "composition_distance": np.nan,
                            "source_depth_ratio": np.nan,
                            "source_delta_threshold": float(candidate["threshold"]),
                            "physical_distance_registered_pixel": distance[keep],
                            "physical_distance_quartile": quartile_label,
                            "left_pathology_region": left_region,
                            "right_pathology_region": right_region,
                            "same_pathology_region": same_region,
                            "question_membership_provenance": "SOURCE_HALF_ONLY_HASH_CAPPED_BY_SOURCE_ISOLATED_ARENA2",
                            "eligibility_provenance": "SOURCE_HALF_ONLY_UNSCALED_FROZEN_THRESHOLDS",
                            "sampling_hash_u64": hashes[local_keep],
                        }
                    )
                )
    if not rows:
        return pd.DataFrame()
    output = pd.concat(rows, ignore_index=True)
    if output.duplicated(ARENA_KEY).any():
        raise RuntimeError(f"F1_ARENA1_KEY_DUPLICATE:{section_data.section}:{seed}")
    observed = (
        output.groupby(
            ["direction", "gene", "physical_distance_quartile"], observed=True
        )
        .size()
        .to_dict()
    )
    violations = {
        key: (value, int(cap.get(key, 0)))
        for key, value in observed.items()
        if value > int(cap.get(key, 0))
    }
    if violations:
        raise RuntimeError(f"F1_ARENA1_CAP_VIOLATION:{violations}")
    return output


def attach_judge_halves(
    membership: pd.DataFrame,
    *,
    section_data: SectionData,
    first: np.ndarray,
    second: np.ndarray,
) -> pd.DataFrame:
    """Attach judge values only after membership has an immutable receipt."""
    if {"judge_delta_full_exposure", "judge_status"}.intersection(membership.columns):
        raise RuntimeError("F1_MEMBERSHIP_PREMATURELY_CONTAINS_JUDGE")
    output = membership.copy()
    position_index = {
        value: index for index, value in enumerate(section_data.positions)
    }
    gene_index = {value: index for index, value in enumerate(section_data.genes)}
    left = _position_indices(output.left_position.astype(str).tolist(), position_index)
    right = _position_indices(
        output.right_position.astype(str).tolist(), position_index
    )
    genes = _gene_indices(output.gene.astype(str).tolist(), gene_index)
    judge_delta = np.empty(len(output), dtype=np.int64)
    direction_values = output.direction.astype(str).to_numpy()
    for direction in DIRECTIONS:
        selected = direction_values == direction
        _, judge = _direction_arrays(first, second, direction)
        judge_delta[selected] = 2 * (
            judge[left[selected], genes[selected]].astype(np.int64)
            - judge[right[selected], genes[selected]].astype(np.int64)
        )
    output["judge_delta_full_exposure"] = judge_delta
    output["judge_status"] = np.where(
        judge_delta != 0, "DEFINED_DIRECTION", "TIE_UNJUDGEABLE"
    )
    return output


def score_methods(section_data: SectionData, questions: pd.DataFrame) -> pd.DataFrame:
    """Score model answers only after all arena memberships are frozen."""
    if questions.empty:
        return pd.DataFrame()
    position_index = {
        value: index for index, value in enumerate(section_data.positions)
    }
    gene_index = {value: index for index, value in enumerate(section_data.genes)}
    rows: list[dict[str, Any]] = []
    group_columns = [
        "arena",
        "patient_id",
        "section_id",
        "seed",
        "direction",
        "gene",
        "gene_fold",
        "physical_distance_quartile",
    ]
    for key, group in questions.groupby(group_columns, observed=True, sort=False):
        left = _position_indices(
            group.left_position.astype(str).tolist(), position_index
        )
        right = _position_indices(
            group.right_position.astype(str).tolist(), position_index
        )
        gene_id = gene_index[str(key[5])]
        judge = group.judge_delta_full_exposure.to_numpy(np.int64)
        judged = judge != 0
        if judged.any():
            prediction_delta = (
                section_data.prediction[left[judged], gene_id, :]
                - section_data.prediction[right[judged], gene_id, :]
            )
            answer = np.sign(prediction_delta).astype(np.int8)
            truth = np.sign(judge[judged]).astype(np.int8)
        else:
            answer = np.empty((0, len(section_data.methods)), dtype=np.int8)
            truth = np.empty(0, dtype=np.int8)
        for method_index, method in enumerate(section_data.methods):
            pred = answer[:, method_index] if len(answer) else np.empty(0, np.int8)
            rows.append(
                {
                    **dict(zip(group_columns, key)),
                    "model_id": method,
                    "accuracy": float(np.mean(pred == truth))
                    if len(truth)
                    else math.nan,
                    "balanced_accuracy": balanced_accuracy(truth, pred),
                    "questions_selected": int(len(group)),
                    "questions_judged": int(judged.sum()),
                    "question_ties": int((~judged).sum()),
                    "prediction_ties": int(np.sum(pred == 0)),
                    "status": "PASS"
                    if len(truth)
                    else "NOT_ESTIMABLE__NO_JUDGEABLE_QUESTIONS",
                }
            )
    return pd.DataFrame(rows)


def question_tokens(frame: pd.DataFrame) -> list[str]:
    return [
        "\x00".join(map(str, row))
        for row in frame[ARENA_KEY].itertuples(index=False, name=None)
    ]


def training_majority_predictions(
    train: pd.DataFrame, test: pd.DataFrame
) -> np.ndarray:
    majority: dict[str, int] = {}
    for gene, group in train.groupby("gene", observed=True, sort=False):
        total = int(np.sign(group.judge_delta_full_exposure.to_numpy(np.int64)).sum())
        if total:
            majority[str(gene)] = 1 if total > 0 else -1
    prediction = np.empty(len(test), dtype=np.int8)
    tokens = question_tokens(test)
    for index, (gene, token) in enumerate(zip(test.gene.astype(str), tokens)):
        value = majority.get(gene)
        if value is None:
            value = 1 if stable_int(f"G0-039-F1-MAJORITY-TIE\x00{token}") & 1 else -1
        prediction[index] = value
    return prediction


def nested_question_score(
    frame: pd.DataFrame, prediction: np.ndarray
) -> dict[str, Any]:
    if len(frame) != len(prediction):
        raise ValueError("F1_SCORE_PREDICTION_AXIS_MISMATCH")
    if not len(frame):
        return {
            "status": "NOT_ESTIMABLE__NO_HELDOUT_JUDGEABLE_QUESTIONS",
            "accuracy": math.nan,
            "balanced_accuracy": math.nan,
            "questions_judged": 0,
            "genes": 0,
            "directions": 0,
            "distance_quartiles": 0,
        }
    truth = np.sign(frame.judge_delta_full_exposure.to_numpy(np.int64)).astype(np.int8)
    work = frame[
        ["section_id", "direction", "gene", "physical_distance_quartile"]
    ].copy()
    work["truth"] = truth
    work["prediction"] = np.asarray(prediction, dtype=np.int8)
    cells: list[dict[str, Any]] = []
    for key, group in work.groupby(
        ["section_id", "direction", "gene", "physical_distance_quartile"],
        observed=True,
        sort=False,
    ):
        y = group.truth.to_numpy(np.int8)
        p = group.prediction.to_numpy(np.int8)
        cells.append(
            {
                "section_id": key[0],
                "direction": key[1],
                "gene": key[2],
                "quartile": key[3],
                "accuracy": float(np.mean(y == p)),
                "balanced_accuracy": balanced_accuracy(y, p),
            }
        )
    cell = pd.DataFrame(cells)
    by_gene_direction = (
        cell.groupby(["section_id", "direction", "gene"], observed=True)[
            ["accuracy", "balanced_accuracy"]
        ]
        .mean()
        .reset_index()
    )
    by_gene = (
        by_gene_direction.groupby(["section_id", "gene"], observed=True)[
            ["accuracy", "balanced_accuracy"]
        ]
        .mean()
        .reset_index()
    )
    by_section = by_gene.groupby("section_id", observed=True)[
        ["accuracy", "balanced_accuracy"]
    ].mean()
    return {
        "status": "PASS",
        "accuracy": float(by_section.accuracy.mean()),
        "balanced_accuracy": float(by_section.balanced_accuracy.mean()),
        "questions_judged": int(len(frame)),
        "genes": int(frame.gene.nunique()),
        "directions": int(frame.direction.nunique()),
        "distance_quartiles": int(frame.physical_distance_quartile.nunique()),
    }


def aggregate_method_patient_seed(method_cells: pd.DataFrame) -> pd.DataFrame:
    keys = ["arena", "patient_id", "model_id", "seed"]
    by_gene_direction = (
        method_cells.groupby([*keys, "section_id", "direction", "gene"], observed=True)[
            ["accuracy", "balanced_accuracy"]
        ]
        .mean()
        .reset_index()
    )
    by_gene = (
        by_gene_direction.groupby([*keys, "section_id", "gene"], observed=True)[
            ["accuracy", "balanced_accuracy"]
        ]
        .mean()
        .reset_index()
    )
    score = (
        by_gene.groupby(keys, observed=True)[["accuracy", "balanced_accuracy"]]
        .mean()
        .reset_index()
    )
    support = (
        method_cells.groupby(keys, observed=True)
        .agg(
            questions_selected=("questions_selected", "sum"),
            questions_judged=("questions_judged", "sum"),
            question_ties=("question_ties", "sum"),
            prediction_ties=("prediction_ties", "sum"),
            genes=("gene", "nunique"),
            sections=("section_id", "nunique"),
            directions=("direction", "nunique"),
            distance_quartiles=("physical_distance_quartile", "nunique"),
        )
        .reset_index()
    )
    return score.merge(support, on=keys, validate="one_to_one")
