# Inputs for count-level recalculation

The `analyses/` commands use the supplied result tables. The `algorithms/` commands recalculate count-level quantities and require the source data and supports described below.

## Cell ideal-prediction calibration

```bash
python paper/algorithms/cell_ideal/calibrate_cell_ideal.py --help
```

Supply `--specimen`, `--canonical`, `--view-cache` and `--output`. The canonical directory contains:

| File | Schema |
|---|---|
| `counts_cells_by_genes.npz` | SciPy CSR matrix of nonnegative integer counts, cells by genes. |
| `cell_axis.parquet` | Cell metadata in count-matrix row order, including `cell_id`. |
| `gene_axis.parquet` | Gene metadata in matrix-column order, including `gene_symbol`. |
| `metadata.json` | Specimen metadata including the `patient` identifier. |

The view cache contains `membership.npz`, `lazy_diffusion.npz`, `retained.npy`, `component_labels.npy`, `exposure.npy` and `receipt.json`. The membership columns must match the canonical cells after optional `--prediction-axis` or `--cell-indices` restriction. Either `--gene-list` or `--prediction-axis` must supply the exact gene support. The script validates the matching dimensions and bindings.

The paper analysis uses 20 Poisson-then-binomial replicates, base seed 20261001 and the prediction support used by the scorer. The default ideal is the expected half-depth logarithmic transform; the simplified log transform is reported separately. Amplitude matching is an optional method selected explicitly.

## CRC and HEST calibration

```bash
python paper/algorithms/ideal_crc_hest/ideal_calibration.py --help
```

The HEST and CRC scorer implementations are included in `algorithms/ideal_crc_hest/source/` and loaded by default. HEST also includes its `g0046_hest.py` helper. Install the dependencies in that algorithm directory's `requirements.txt`. HEST needs a frozen data directory containing `input/`, `basis/` and `moments/`; CRC needs the exact gene panel TSV with `gene_index` and the specimen's count/geometry payload under `cache/crc_grids/<specimen>/`. Supply these data inputs at the paths passed to the command.

```bash
python paper/algorithms/ideal_crc_hest/ideal_calibration.py hest --frozen-root /path/to/hest-calibration --task IDC --specimen TENX99 --output results/hest
python paper/algorithms/ideal_crc_hest/ideal_calibration.py crc --crc-payload /path/to/crc-payload --crc-panel /path/to/panel.tsv --specimen P1CRC --scale 32 --output results/crc
```

The optional `--source-dir` and `--crc-source` arguments select alternative scorer source directories. HEST helper scripts use `STRSC_HIST_DATA` for any standalone preparation/scoring workspace; the ideal-calibration command binds the input, basis and moment directories directly from `--frozen-root`.

## External TLS

```bash
python paper/algorithms/tls/run_tls_repeatability.py --data-root /path/to/tls-inputs --workers 4
python paper/algorithms/tls/calibrate_tls.py --data /path/to/tls-inputs --workers 4
```

Before running, place the source-provider `coef-full.tsv.gz` in `paper/algorithms/tls/source/`. It is a gene-indexed TSV with one coefficient column, optionally including `INTERCEPT`. Obtain the coefficient file from its source provider. Both commands use the included transformation and gene-binding functions.

The data root must preserve these relative paths for KC1–KC3 and LC1–LC5:

- `inputs/derived/<section>/COUNTS_COMMON.npz`: `counts` (integer spots × 748 genes), `barcodes`, and `gene_canonical`.
- `predicted_tls/folds/<section>/SPOT_READOUTS.tsv.gz`: unique `barcode`, `measured_tls`, `ordinary_tls`, and `neighbor_tls` columns on the prediction-valid spot support.
- `predicted_tls/folds/<section>/METRICS.tsv`: `arm` and `tls_score_spearman` for the two prediction arms.
- `measured_tls/GENE_COEFFICIENT_BINDING.tsv`: `sample`, `gene_canonical`, and `coefficient`.

There are 747 matching coefficients within the 748-gene normalization panel. The function divides each spot by its panel total, then each gene by its section-wide standard deviation, before applying coefficients and intercept. Repeatability uses 20 binomial splits with base seed 20260930, the same spot support and independent transformations of each half. The commands evaluate supplied predictions and write outputs under the algorithm directory. Completed repeatability outputs are retained.

## Full prediction and training reproduction

Recalculating the common-gene comparison from predictions requires the exact GHIST, sCellST and DeepSpot2Cell prediction matrices and their gene/cell axes. Recalculating normalized ordering requires the original HER2ST/HEST pair banks, molecule splits, full-library depths and prediction values. Model retraining additionally requires cohort access, histology, checkpoints or reference models, and the corresponding upstream software licences and environments.
