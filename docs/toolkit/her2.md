# Expression ordering across and within tissue regions

This workflow evaluates expression ordering using questions constructed from
one RNA half and scored with the other. Inputs are raw integer molecule counts,
registered spatial coordinates, pathology-region labels and predictions aligned
to the same spots and genes.

```bash
strsc her2 --manifest inputs/her2.json --out outputs/her2
```

The JSON manifest identifies one section per patient. At least two patients are
needed to fit the leave-one-patient-out gene-majority baseline.

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
| `regions` | spots | Pathology-region labels; an empty string denotes an unknown label |
| `positions` | spots | Unique spot identifiers |
| `genes` | genes | Unique gene identifiers |
| `methods` | models | Unique model identifiers |

All sections use the same genes and models. Named axes are sorted internally. Comparisons use the positions with finite
predictions in every model for each gene. Missing values therefore change the
shared evaluable set. Inputs should contain held-out predictions if the goal is
to compare model generalization.

The default uses ten molecule-split seeds, 2026080501 through 2026080510.
Use these seeds to reproduce the paper analysis, or provide a `seeds` list
in the manifest for a different analysis.

RNA molecules are split into two halves. Questions are selected from one half
and answered with the other, then the halves are exchanged. Context matching
uses source-half expression from genes outside the target gene's fold, and
matches sequencing depth. The within-region task retains matching pairs with
the same nonempty pathology-region label. Across-region questions are sampled
with matched question counts in physical-distance strata. The task name
“Across regions” includes pairs with the same or different region labels.

Question construction uses the following eligibility and matching settings:
20 common positions, at least 5% source-half detection, at least 20 source
molecules, eight composition neighbours, a maximum depth ratio of 1.25, and
the top quartile of nonzero source expression differences. Genes or tasks with
insufficient support receive a missing score.

Four TSV files are written:

- `patient_scores.tsv`: model accuracy, the patient-held-out gene-majority
  baseline, RNA split-half accuracy, their differences, and the normalized
  pairwise accuracy (`normalized_pairwise_accuracy`).
- `patient_seed_scores.tsv`: model scores before averaging molecule-split seeds.
- `baseline_seed_scores.tsv`: baseline scores and training/test question counts
  for each held-out patient and seed.
- `question_support.tsv`: selected and judgeable question counts per section,
  seed and task, including empty tasks.

The normalized pairwise accuracy G is calculated after averaging seeds within each patient:

```text
(model accuracy − held-out gene-majority accuracy)
───────────────────────────────────────────────
(RNA split-half accuracy − held-out gene-majority accuracy)
```

G expresses the gain of the model over the baseline relative to the gain of
the two RNA halves, and retains its full numerical range. A nonpositive or near-zero denominator
produces `NOT_ESTIMABLE` and a missing normalized score. Raw components remain
in the output. The gene-majority baseline is fitted on the other patients.

Counts are aggregated over distance strata, directions and genes before
patient-level reporting. The workflow includes the ordering comparison,
two RNA halves and the gene-majority baseline with deterministic sampling.

The Python API provides `strsc.her2.make_section(...)`,
`section_questions(section, seed)` and `evaluate(sections, seeds)`.
Toolkit tests and examples use synthetic inputs. Paper summary tables are
provided separately under `paper/`.
