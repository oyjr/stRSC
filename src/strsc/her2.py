"""Evaluate expression ordering on source-half-defined spatial questions."""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from . import her2_core as core

DEFAULT_SEEDS = tuple(range(2026080501, 2026080511))
TASKS = {
    "BROAD_STRUCTURE_ALLOWED": "Across regions",
    "COMPOSITION_MATCHED_ORIGINAL": "Context and depth matched",
    "JOINT_STRUCTURE_MATCHED": "Within regions",
}


def make_section(
    *, patient, section, counts, prediction, coords, regions, positions, genes, methods
) -> core.SectionData:
    """Align raw counts, fixed predictions and metadata on named axes."""
    counts = np.asarray(counts)
    prediction = np.asarray(prediction, dtype=np.float32)
    coords = np.asarray(coords, dtype=np.float64)
    positions, genes, methods = (
        tuple(map(str, axis)) for axis in (positions, genes, methods)
    )
    n, g, m = len(positions), len(genes), len(methods)
    if not str(patient).strip() or not str(section).strip():
        raise ValueError("Patient and section identifiers must be nonempty")
    if n < 2 or g < 3 or m < 1:
        raise ValueError(
            "At least two positions, three genes and one model are required"
        )
    if any(len(set(axis)) != len(axis) for axis in (positions, genes, methods)):
        raise ValueError("Position, gene and model identifiers must be unique")
    if counts.shape != (n, g) or prediction.shape != (n, g, m):
        raise ValueError("Counts and prediction shapes do not match their named axes")
    if not np.issubdtype(counts.dtype, np.integer) or np.any(counts < 0):
        raise ValueError("Counts must be nonnegative integers, before normalization")
    if coords.shape != (n, 2) or not np.isfinite(coords).all():
        raise ValueError("Coordinates must be finite with shape (positions, 2)")
    regions = np.asarray(regions, dtype=str)
    if regions.shape != (n,):
        raise ValueError("One pathology-region label is required per position")

    # Match the original loader's lexical axes before sampling molecules.
    pi, gi, mi = (np.argsort(axis) for axis in (positions, genes, methods))
    positions, genes, methods = (
        tuple(np.asarray(axis)[order])
        for axis, order in ((positions, pi), (genes, gi), (methods, mi))
    )
    counts = counts[np.ix_(pi, gi)].astype(np.int64)
    prediction = prediction[np.ix_(pi, gi, mi)]
    coords, regions = coords[pi], regions[pi]
    geometry = core.physical_distance_geometry(coords)
    return core.SectionData(
        patient=str(patient),
        section=str(section),
        positions=positions,
        genes=genes,
        methods=methods,
        counts=counts,
        prediction=prediction,
        common_support=np.isfinite(prediction).all(axis=2),
        regions=regions,
        pixel_xy=coords,
        fold_axis=np.asarray([core.gene_fold(gene) for gene in genes]),
        pair_left=geometry[0],
        pair_right=geometry[1],
        pair_distance=geometry[2],
        pair_quartile=geometry[3],
        distance_cutpoints=geometry[4],
    )


def section_questions(section: core.SectionData, seed: int) -> pd.DataFrame:
    """Select questions using source molecules, then attach independent answers."""
    first, second = core.split_molecules(section.counts, section.section, seed)
    matched = core.build_corrected_arena2_membership(
        section_data=section, seed=seed, first=first, second=second
    )
    if matched.empty:
        return pd.DataFrame()
    across = core.build_arena1(
        section_data=section, seed=seed, first=first, second=second, arena2=matched
    )
    within = core.strict_arena3_subset(matched)
    membership = pd.concat([across, matched, within], ignore_index=True, sort=False)
    membership = membership.sort_values(core.ARENA_KEY, kind="stable").reset_index(
        drop=True
    )
    if membership.duplicated(core.ARENA_KEY).any():
        raise ValueError("Duplicate spatial questions")
    return core.attach_judge_halves(
        membership, section_data=section, first=first, second=second
    )


def evaluate(
    sections: Sequence[core.SectionData], seeds=DEFAULT_SEEDS
) -> dict[str, pd.DataFrame]:
    """Return patient scores, seed scores, baselines and question counts."""
    sections, seeds = list(sections), tuple(seeds)
    if not sections or len({x.section for x in sections}) != len(sections):
        raise ValueError("Provide sections with unique identifiers")
    patients = sorted({x.patient for x in sections})
    if len(patients) < 2:
        raise ValueError(
            "The leave-one-patient-out baseline requires at least two patients"
        )
    if len(patients) != len(sections):
        raise ValueError("This HER2ST protocol uses one section per patient")
    if (
        not seeds
        or any(not isinstance(s, (int, np.integer)) or s < 0 for s in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("Split seeds must be distinct nonnegative integers")
    if len({x.genes for x in sections}) != 1 or len({x.methods for x in sections}) != 1:
        raise ValueError("All sections must use the same gene and model axes")

    model_scores, baselines, support = [], [], []
    with threadpool_limits(limits=1):
        for seed in seeds:
            questions = []
            for section in sections:
                frame = section_questions(section, int(seed))
                for arena in TASKS:
                    selected = (
                        frame.loc[frame.arena.eq(arena)] if not frame.empty else frame
                    )
                    support.append(
                        {
                            "patient_id": section.patient,
                            "section_id": section.section,
                            "seed": seed,
                            "arena": arena,
                            "questions_selected": len(selected),
                            "questions_judged": int(
                                selected.judge_delta_full_exposure.ne(0).sum()
                            )
                            if not selected.empty
                            else 0,
                        }
                    )
                if frame.empty:
                    continue
                questions.append(frame)
                model_scores.append(
                    core.aggregate_method_patient_seed(
                        core.score_methods(section, frame)
                    )
                )
            if not questions:
                continue
            questions = pd.concat(questions, ignore_index=True)
            judgeable = questions.loc[questions.judge_delta_full_exposure.ne(0)]
            for arena, patient in product(TASKS, patients):
                selected = judgeable.loc[judgeable.arena.eq(arena)]
                test = selected.loc[selected.patient_id.eq(patient)].reset_index(
                    drop=True
                )
                train = selected.loc[selected.patient_id.ne(patient)].reset_index(
                    drop=True
                )
                row = {
                    "arena": arena,
                    "patient_id": patient,
                    "seed": seed,
                    "training_questions": len(train),
                    "questions_judged": len(test),
                    "lopo_gene_majority_baseline": np.nan,
                    "rna_split_half_accuracy": np.nan,
                }
                if not train.empty and not test.empty:
                    majority = core.training_majority_predictions(train, test)
                    rna = np.sign(
                        test.source_delta_full_exposure.to_numpy(np.int64)
                    ).astype(np.int8)
                    row["lopo_gene_majority_baseline"] = core.nested_question_score(
                        test, majority
                    )["accuracy"]
                    row["rna_split_half_accuracy"] = core.nested_question_score(
                        test, rna
                    )["accuracy"]
                baselines.append(row)

    keys = ["arena", "patient_id", "model_id", "seed"]
    seed_scores = (
        pd.concat(model_scores, ignore_index=True)
        if model_scores
        else pd.DataFrame(columns=keys + ["accuracy"])
    )
    baseline_columns = ["lopo_gene_majority_baseline", "rna_split_half_accuracy"]
    baseline_seed = pd.DataFrame(
        baselines,
        columns=[
            "arena",
            "patient_id",
            "seed",
            "training_questions",
            "questions_judged",
        ]
        + baseline_columns,
    )
    patient_scores = pd.DataFrame(
        product(TASKS, patients, sections[0].methods), columns=keys[:3]
    )
    if not seed_scores.empty:
        means = seed_scores.groupby(keys[:3], observed=True, as_index=False).agg(
            model_accuracy=("accuracy", "mean"), seeds_evaluable=("seed", "nunique")
        )
        patient_scores = patient_scores.merge(
            means, how="left", on=keys[:3], validate="one_to_one"
        )
    else:
        patient_scores["model_accuracy"], patient_scores["seeds_evaluable"] = np.nan, 0
    baseline_patient = baseline_seed.groupby(keys[:2], observed=True, as_index=False)[
        baseline_columns
    ].mean()
    patient_scores = patient_scores.merge(
        baseline_patient, how="left", on=keys[:2], validate="many_to_one"
    )
    patient_scores["model_increment"] = (
        patient_scores.model_accuracy - patient_scores.lopo_gene_majority_baseline
    )
    patient_scores["split_half_increment"] = (
        patient_scores.rna_split_half_accuracy
        - patient_scores.lopo_gene_majority_baseline
    )
    denominator = patient_scores.split_half_increment.to_numpy(float)
    numerator = patient_scores.model_increment.to_numpy(float)
    estimable = (
        np.isfinite(numerator) & np.isfinite(denominator) & (denominator > 1e-12)
    )
    patient_scores["normalized_pairwise_accuracy"] = np.divide(
        numerator, denominator, out=np.full(len(numerator), np.nan), where=estimable
    )
    patient_scores["status"] = np.where(estimable, "ESTIMABLE", "NOT_ESTIMABLE")
    patient_scores["seeds_requested"] = len(seeds)
    patient_scores["seeds_evaluable"] = patient_scores.seeds_evaluable.fillna(0).astype(
        int
    )
    patient_scores["task"] = patient_scores.arena.map(TASKS)
    return {
        "patient_scores": patient_scores,
        "patient_seed_scores": seed_scores,
        "baseline_seed_scores": baseline_seed,
        "question_support": pd.DataFrame(support),
    }


def load_manifest(path: str | Path):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    sections = []
    for item in manifest["sections"]:
        with np.load(path.parent / item["path"], allow_pickle=False) as data:
            sections.append(
                make_section(
                    patient=item["patient"],
                    section=item["section"],
                    **{
                        key: data[key]
                        for key in (
                            "counts",
                            "prediction",
                            "coords",
                            "regions",
                            "positions",
                            "genes",
                            "methods",
                        )
                    },
                )
            )
    return sections, manifest.get("seeds", DEFAULT_SEEDS)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    sections, seeds = load_manifest(args.manifest)
    tables = evaluate(sections, seeds)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(args.out / f"{name}.tsv", sep="\t", index=False)
    print(f"Wrote {len(tables)} tables to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
