# stRSC evaluation bundle v1

`schema: "strsc-evaluation-bundle-v1"` stores evaluation scores for one or more predictions and biological units. Each workflow retains its own metrics and support definitions. Use `strsc-analysis-bundle-v1` for paired expression maps.

```json
{
  "schema": "strsc-evaluation-bundle-v1",
  "title": "My HEST evaluation",
  "workflow": "hest",
  "synthetic": false,
  "metrics": [{
    "id": "rsc_fine",
    "label": "Fine-scale RSC",
    "description": "RSC calculated from combined B4+B5 covariance and variance components.",
    "unit": "score",
    "higher_is_better": true,
    "object": "prediction"
  }],
  "rows": [{
    "model": "prediction_A",
    "task": "LUNG",
    "unit_id": "specimen_1",
    "unit_type": "specimen",
    "metric": "rsc_fine",
    "value": null,
    "status": "ZERO_MODEL_VARIANCE",
    "support": "graph-retained locations",
    "details": {},
    "model_role": "prediction"
  }],
  "sources": [{"name": "run1/specimen_scores.tsv", "columns": [], "rows": []}],
  "notes": ["Scores use graph-retained locations."]
}
```

This schema example illustrates an undefined score. Computed examples are available in [`examples/bundles/`](../../examples/bundles/).

## Fields and metric definitions

- `workflow` is `hest`, `her2`, `fourier`, or `cell`. Each bundle contains results from one workflow.
- Row identity is `(model, task, unit_id, metric)` and must be unique. Model and unit names come from the saved results or explicit command arguments.
- `value` is a finite number or `null`. Undefined scores retain a status; negative values and scores above one are preserved. Nonfinite values in details and source tables become `null`.
- `metrics[].object` and `rows[].model_role` are `prediction`, `split_half`, or `baseline`. Model comparisons use prediction rows with the same metric, task and unit support; Split-half references and baselines are displayed separately.
- HEST metrics are `pearson_graph_support`, `rsc_broad`, `rsc_fine`, `rsc_B1` through `rsc_B5`. Broad and fine scores use combined covariance and variance components from bands 1–2 and 4–5, respectively. The five bands describe relative graph frequencies.
- HER2 task labels identify across-region, matched-context or within-region questions. Prediction metrics are `model_accuracy` and `normalized_pairwise_accuracy`; reference metrics are `lopo_gene_majority_baseline` and `rna_split_half_accuracy`. Baseline/reference rows are emitted once per task and patient.
- Fourier metric is `physical_rsc`; task includes the physical wavelength band and the grid spacing. `details.nyquist_status` explicitly identifies partial high-frequency support. `ALL_NON_DC` is a combined-components score. The original table lacks specimen IDs, so `--unit` is mandatory and repeated in the same order as `--run`.
- Cell metric `cell_rsc` preserves track/view/band. RNA-only tables yield `rna_split_agreement`. Hotspot tables yield `E_model_normalized_log_score` and `E_RNA_normalized_log_score`, with model-specific support in the task. Specimens remain specimens; patient identity stays in details. Each model retains the support recorded in its source tables.
- `sources` stores the scored table contents. For spatial expression maps, use the paired HEST export, which includes matched expression matrices and coordinates.
- `synthetic` labels generated examples. Set it explicitly with `--synthetic` when exporting generated inputs.

## Commands

```bash
strsc export-results --run outputs/hest --output outputs/hest-results.json
strsc export-results --workflow her2 --run outputs/her2 --output outputs/ordering-results.json
strsc export-results --run outputs/fourier_P1.tsv --unit P1 --run outputs/fourier_P2.tsv --unit P2 --output outputs/physical-results.json
strsc export-results --workflow cell --run outputs/cell_scores --output outputs/cell-results.json
```

`--workflow auto` is the default. An explicit workflow must match the saved results. Repeat `--run` to merge compatible results, including multiple HEST specimens or cell output directories. Choose a new output filename for each export.

## Computed examples

Files in [`examples/bundles/`](../../examples/bundles/):

| File | Numerical path used | Structure |
|---|---|---|
| `hest.json` | HEST count/graph scorer; 20 molecule splits | One prediction, one specimen, 8 metrics |
| `her2.json` | Ordering scorer; one specified molecule split | Two predictions, two patients, 3 tasks, separate reference rows |
| `fourier.json` | Fourier scorer; one specified molecule split | Two predictions, two specimens, eight windows including ALL_NON_DC |
| `cell.json` | Lazy graph band and component functions | One prediction, one specimen, five bands and split-half reliability |

The cell example evaluates graph-band components. The HER2 and Fourier examples use one molecule split for a short demonstration; the paper analyses use the full split sets specified in their workflow guides. Generate the examples with `python examples/export_results_example.py --output outputs/export-examples`.
