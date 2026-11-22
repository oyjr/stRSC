# Pairwise accuracy across and within tissue regions

This workflow computes pairwise accuracy on spot pairs that are selected in
one RNA half and scored with the other. Inputs are raw integer molecule counts,
registered spatial coordinates, pathology-region labels and predictions aligned
to the same spots and genes.

```bash
strsc her2 --manifest inputs/her2.json --out outputs/her2
```

The JSON manifest identifies one section per patient. At least two patients are
needed to fit the leave-one-patient-out majority-order baseline.

```json
{
  "sections": [
    {"patient": "P1", "section": "P1_section", "path": "P1.npz"},
    {"patient": "P2", "section": "P2_section", "path": "P2.npz"}
  ]
}
```

Each NPZ file contains the following arrays. File paths are relative to the
manifest. String arrays must use a string dtype, not pickled Python objects.

| Array | Shape | Content |
|---|---|---|
| `counts` | spots × genes | Nonnegative integer molecule counts, before normalization |
| `prediction` | spots × genes × models | Fixed expression predictions |
| `coords` | spots × 2 | Registered spatial coordinates in consistent units |
| `regions` | spots | Pathology-region labels, with an empty string for an unknown label |
| `positions` | spots | Unique spot identifiers |
| `genes` | genes | Unique gene identifiers |
| `methods` | models | Unique model identifiers |

All sections use the same genes and models. Named axes are sorted internally. Comparisons use the positions with finite
predictions in every model for each gene. Missing values therefore change the
shared evaluable set. Inputs should contain held-out predictions if the goal is
to compare model generalization.

The default uses ten count-split seeds, 2026080501 through 2026080510.
Use these seeds to reproduce the paper analysis, or provide a `seeds` list
in the manifest for a different analysis.

RNA molecules are split into two halves. A question asks which spot of a pair
expresses a gene more highly. Questions are selected from one half, the source
half, and answered with the other, the evaluation half, then the halves are
exchanged. Context matching
uses source-half expression from genes outside the target gene's fold, and
matches sequencing depth. The within-region task retains matching pairs with
the same nonempty pathology-region label. Across-region questions are sampled
with matched question counts in physical-distance strata. The task name
“Across regions” includes pairs with the same or different region labels.

Question construction uses the following eligibility and matching settings:
20 common positions, at least 5% source-half detection, at least 20 source
molecules, eight composition neighbors, a maximum depth ratio of 1.25, and
the top quartile of nonzero source expression differences. Genes or tasks with
insufficient support have no score.

Four TSV files are written:

- `patient_scores.tsv`: model accuracy, the patient-held-out majority-order
  baseline, RNA split-half accuracy, their differences, and the normalized
  pairwise accuracy (`normalized_pairwise_accuracy`).
- `patient_seed_scores.tsv`: model scores before averaging count-split seeds.
- `baseline_seed_scores.tsv`: baseline scores and training/test question counts
  for each held-out patient and seed.
- `question_support.tsv`: numbers of selected questions and of questions
  without a tie in the evaluation half, per section, seed and task, including
  empty tasks.

The normalized pairwise accuracy G is calculated after averaging seeds within each patient:

```text
(model accuracy − majority-order baseline accuracy)
───────────────────────────────────────────────
(RNA split-half accuracy − majority-order baseline accuracy)
```

G expresses the gain of the model over the baseline relative to the gain of
the two RNA halves, and can exceed 1 or fall below 0. A nonpositive or near-zero denominator
produces `NOT_ESTIMABLE` and a missing normalized score. The accuracies are still
written. The majority-order baseline is fitted on the other patients.

Counts are aggregated over distance strata, directions and genes before
patient-level reporting. The workflow scores the models, the two RNA halves
and the majority-order baseline with deterministic sampling.

The Python API provides `strsc.her2.make_section(...)`,
`section_questions(section, seed)` and `evaluate(sections, seeds)`.
Toolkit tests and examples use synthetic inputs. Paper summary tables are
provided separately under `paper/`.
