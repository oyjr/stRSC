import json

import numpy as np
import pandas as pd
import pytest

from strsc import her2, her2_core as core


def synthetic_section(patient="P1", seed=7):
    rng = np.random.default_rng(seed)
    n, g = 196, 40
    coords = np.column_stack(np.unravel_index(np.arange(n), (14, 14))).astype(float)
    rates = np.exp(rng.normal(3.5, 0.5, size=(n, g)))
    counts = rng.poisson(rates).astype(np.int64)
    prediction = np.stack([rates, rates[rng.permutation(n)]], axis=2)
    return her2.make_section(
        patient=patient,
        section=f"{patient}_section",
        counts=counts,
        prediction=prediction,
        coords=coords,
        regions=np.where(coords[:, 0] < 7, "R1", "R2"),
        positions=[f"s{i:03d}" for i in range(n)],
        genes=[f"g{i:02d}" for i in range(g)],
        methods=["aligned_example", "permuted_example"],
    )


def test_questions_do_not_depend_on_prediction_values():
    original = synthetic_section()
    from dataclasses import replace

    changed = replace(original, prediction=-original.prediction)
    first = her2.section_questions(original, 2026080501)
    second = her2.section_questions(changed, 2026080501)
    assert not first.empty
    pd.testing.assert_frame_equal(first, second)
    within = first.loc[first.arena.eq("JOINT_STRUCTURE_MATCHED")]
    assert len(within) > 0
    assert within.left_pathology_region.eq(within.right_pathology_region).all()


def test_judge_half_is_not_used_to_select_questions():
    section = synthetic_section()
    first, second = core.split_molecules(section.counts, section.section, 2026080501)
    before = core.build_corrected_arena2_membership(
        section_data=section, seed=2026080501, first=first, second=second
    )
    after = core.build_corrected_arena2_membership(
        section_data=section, seed=2026080501, first=first, second=second[::-1].copy()
    )
    # In A_TO_B the changed second half is only the judge; B_TO_A uses it as source.
    a = before.loc[before.direction.eq("A_TO_B")].reset_index(drop=True)
    b = after.loc[after.direction.eq("A_TO_B")].reset_index(drop=True)
    assert not a.empty
    pd.testing.assert_frame_equal(a, b)


def test_patient_baseline_and_reported_ratio():
    sections = [synthetic_section("P1", 7), synthetic_section("P2", 11)]
    seed = 2026080501
    tables = her2.evaluate(sections, [seed])
    scores = tables["patient_scores"]
    assert len(scores) == 12
    assert set(scores.task) == set(her2.TASKS.values())
    assert scores.model_accuracy.notna().all()
    questions = pd.concat(
        [her2.section_questions(section, seed) for section in sections]
    )
    questions = questions.loc[questions.judge_delta_full_exposure.ne(0)]
    for row in tables["baseline_seed_scores"].itertuples():
        task = questions.loc[questions.arena.eq(row.arena)]
        train = task.loc[task.patient_id.ne(row.patient_id)].reset_index(drop=True)
        test = task.loc[task.patient_id.eq(row.patient_id)].reset_index(drop=True)
        expected = core.nested_question_score(
            test, core.training_majority_predictions(train, test)
        )["accuracy"]
        assert row.lopo_gene_majority_baseline == pytest.approx(expected)
    estimable = scores.status.eq("ESTIMABLE")
    assert estimable.any()
    expected = (scores.model_accuracy - scores.lopo_gene_majority_baseline) / (
        scores.rna_split_half_accuracy - scores.lopo_gene_majority_baseline
    )
    np.testing.assert_allclose(
        scores.loc[estimable, "normalized_pairwise_accuracy"],
        expected[estimable],
    )


def test_manifest_and_cli(tmp_path):
    sections = [synthetic_section("P1", 7), synthetic_section("P2", 11)]
    manifest = {"seeds": [2026080501], "sections": []}
    for section in sections:
        filename = f"{section.section}.npz"
        np.savez_compressed(
            tmp_path / filename,
            counts=section.counts,
            prediction=section.prediction,
            coords=section.pixel_xy,
            regions=section.regions.astype(str),
            positions=section.positions,
            genes=section.genes,
            methods=section.methods,
        )
        manifest["sections"].append(
            {"patient": section.patient, "section": section.section, "path": filename}
        )
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert her2.main(["--manifest", str(path), "--out", str(tmp_path / "results")]) == 0
    assert pd.read_csv(tmp_path / "results/patient_scores.tsv", sep="\t").shape[0] == 12
    with pytest.raises(ValueError, match="at least two patients"):
        her2.evaluate(sections[:1], [2026080501])
