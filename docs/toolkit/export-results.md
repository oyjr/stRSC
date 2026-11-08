# Export one or multiple model evaluations to the browser

Export a completed evaluation and upload the JSON in **Analyze data → Computed results** on the website ([stRSC.org](https://stRSC.org/analyze?input=results)). The command reads saved scores using Python's standard library.

```bash
strsc export-results --run outputs/hest --output outputs/hest-results.json
strsc export-results --workflow her2 --run outputs/her2 --output outputs/ordering-results.json
strsc export-results --run outputs/fourier_P1.tsv --unit P1 --run outputs/fourier_P2.tsv --unit P2 --output outputs/physical-results.json
strsc export-results --workflow cell --run outputs/cell_scores --output outputs/cell-results.json
```

`--run` can be repeated to combine results from the same workflow. Each model/task/unit/metric identity must be unique. HEST supports one or more named predictions. For Fourier tables, supply one `--unit` for each `--run` in the same order; other workflows read unit identifiers from the saved results.

The bundle includes metric definitions, score status and source tables. Scores retain their full numerical range, and each model retains its gene and cell support. Patient and specimen identifiers are preserved. Split-half references and held-out baselines are labeled separately from prediction metrics.

For matched expression maps, use the paired HEST export:

```bash
strsc export --manifest data/manifest.json --run outputs/hest --output outputs/paired-expression.json
```

## Generate small examples

```bash
python examples/export_results_example.py --output outputs/export-examples
```

This command writes `hest.json`, `her2.json`, `fourier.json` and `cell.json` from generated inputs. It demonstrates a single-prediction HEST evaluation, a two-specimen Fourier merge and cell graph-band calculations. The HER2 and Fourier examples use short molecule-split lists for demonstration. All four bundles are labelled synthetic.

On Windows, activate the Python or Conda environment before running numerical workflows so its compiled dependencies are available.

See the [evaluation bundle format](evaluation-bundle.md) for fields, metric IDs and support semantics.
