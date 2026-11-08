# HEST scoring

The HEST entry point computes RSC for held-out predictions on the 50-gene benchmark panel. It uses the paper's cohort definitions, log1p count transform, graph, 20 molecule splits and leave-one-specimen-out RNA weights.

Each count input is an NPZ file with:

| Key | Shape | Meaning |
| --- | --- | --- |
| `counts` | locations × 50 | Nonnegative integer molecule counts |
| `xy` | locations × 2 | Spatial coordinates |
| `barcodes` | locations | Unique string IDs |
| `genes` | 50 | Ordered unique gene IDs |

Each prediction HDF5 file contains `prediction`, `barcodes` and `genes`. Prediction columns must use the count file's gene order; rows are aligned by barcode. Predictions must be on the `log1p(raw count)` scale. Supply held-out predictions from a separately trained model.

The manifest identifies the complete reference cohort:

```json
{
  "protocol": "HEST_G0047B",
  "task": "LUNG",
  "specimen": "TENX118",
  "model": "my_model",
  "prediction_transform": "log1p_raw_integer_counts_no_library_normalization",
  "cohort": {"TENX118": "TENX118.npz", "TENX141": "TENX141.npz"},
  "predictions": {"baseline": "baseline.h5", "intervention": "intervention.h5"}
}
```

Paths are relative to the manifest. The supported cohorts are IDC (`TENX99`, `TENX95`, `NCBI785`, `NCBI783`), PAAD (`TENX116`, `TENX140`, `TENX126`) and LUNG (`TENX141`, `TENX118`). All members of the evaluated task are needed to estimate held-out RNA weights. The CPU entry point supports up to 6,000 retained locations per specimen.

```bash
strsc hest --manifest data/manifest.json --output outputs/hest
strsc export --manifest data/manifest.json --run outputs/hest --output outputs/hest.json
```

The scorer writes broad- and fine-scale RSC for each specimen, RSC in five bands, gene-wise Pearson correlations, RNA weight components, graph support and an analysis record. The exported JSON adds matched expression, coordinates and predictions for the browser.

Broad- and fine-scale RSC combine covariance and variance components from bands 1–2 and 4–5, respectively. The training comparison uses the arithmetic two-band mean. Scores retain their full numerical range; undefined quantities remain missing. The five bands represent relative graph frequencies. Validation on an additional cohort requires a separate assessment.
