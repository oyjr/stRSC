# RSC on your own data

```bash
pip install strsc
```

`pip install "strsc[io]"` adds pyarrow, for Parquet position files of Space
Ranger and Xenium. Python 3.10 or later is needed.

```python
import strsc

counts = strsc.read_counts("sample.h5ad")        # a table, .h5ad, Space Ranger or Xenium folder
result = strsc.rsc(counts, {"my_model": "predictions.csv"}, layout="spots")
print(result.scales)                             # RSC, Pearson r and split-half reliability per scale
result.save("strsc-output")
```

The same analysis from the command line:

```bash
strsc rsc sample.h5ad -p my_model=predictions.csv --layout spots --out strsc-output
```

This is the calculation that the Calculate RSC tab of
[stRSC.org](https://stRSC.org/analyze) runs in the browser, with the same steps
and random seeds. Each result page there shows the Python call and the command
for its own files.

## Counts

`strsc.read_counts(path)` reads raw integer molecule counts with the position
of every spot, bin or cell:

- a CSV or TSV table with `sample_id`, `unit_id`, `x` and `y`, optional
  `patient_id`, `region`, `cell_class` and `area` columns, then one column per
  gene (the test files of the website have this format);
- an AnnData `.h5ad` file, such as a HEST-1k sample. Counts come from
  `layers['counts']` when present, otherwise from `X`; coordinates from
  `obsm['spatial']`, otherwise from `obs` columns `x` and `y`;
- a 10x Genomics `.h5` matrix or `filtered_feature_bc_matrix` folder with the
  `tissue_positions` file of Space Ranger (Visium spots or Visium HD bins), or
  `cell_feature_matrix.h5` with the `cells` file of Xenium;
- a folder that holds one of these, such as a Space Ranger `outs` folder, a
  Visium HD `square_016um` folder or a Xenium output folder. A folder that holds
  several such folders gives one section per folder.

A list of paths gives one section per path. Options: `sample` names the
section, `positions` points to the positions file of a 10x matrix,
`layer`, `spatial_key`, `sample_column` and `columns` select parts of an
`.h5ad` file, and `um_per_unit` converts coordinates to micrometers when the
file does not say how. `counts.with_labels("labels.csv")` adds columns such as
`region` or `cell_class` from a table with `unit_id` (and `sample_id` when
there are several sections).

Counts must be whole numbers from before any normalization or log transform,
because RSC splits molecules. Xenium control probes are left out.

## Predictions

A prediction is a CSV or TSV table with `unit_id` (and `sample_id` when there
are several sections) and one column per gene, or an `.h5ad` file whose
observation names are the unit IDs. Give one or more as a dict of model name to
file, `Prediction` or array of the shape of the counts. RSC is computed on the
genes that the counts and every prediction share, and on the locations that
have a value for each of them.

## Layouts

| `layout` | Data | Scales | Half-count transform | Count splits | Gene weighting |
|---|---|---|---|---|---|
| `"spots"` | Visium, HEST | five relative scales, Broadest to Finest: equal-rank groups of the nonconstant eigenvectors of the neighbor-graph Laplacian | log(1 + 2 N_half) | 20 | two or more sections: agreement of the halves in the other sections, as for the HEST samples; one section: a common factor for each gene's halves |
| `"grid"` | Visium HD bins and other regular grids | seven physical scales of a two-dimensional Fourier basis, from below 128 µm to 4,096 µm and above | sqrt(N_half / (0.5 a)), with a the bin area | 5 | a common factor for each gene's halves |
| `"cells"` | Xenium and other single-cell data | five relative scales from lazy diffusion on the graph of the eight nearest neighbors within 40 µm | log(1 + 2 N_half) | 5 | a common factor for each gene's halves |

These are the steps of the paper (Methods, "Reproducible-signal correlation"
and "Scales and count transforms"). The `"grid"` layout gives the same RSC as
`strsc.fourier.score`. One spot section and several spot sections are weighted
differently, so their values are not directly comparable.

Further arguments of `strsc.rsc`:

| Argument | Meaning |
|---|---|
| `spacing_um` | Grid spacing in micrometers (`"grid"`); read from Visium HD files when not given |
| `within` | One or more values of the `cell_class` column to score on their own graphs (`"cells"`), as for the malignant cells of the paper |
| `all_cells` | Whether to score all cells as well (default `True`) |
| `cell_class` | Column that holds the cell classes (default `"cell_class"`) |
| `genes` | Restrict the genes |
| `splits` | Use only the first count splits of the paper; fewer splits give noisier values |
| `progress` | A function that receives progress events |

## Results

`strsc.rsc` returns a `Result` with three tables:

- `scales`: one row per model, section and scale, with `rsc`, `pearson_r` (the
  Pearson r of the prediction with the mean of the two halves at that scale),
  `split_half_reliability` (the correlation between the two halves), the terms
  `C`, `V_model`, `V_RNA`, `V_A` and `V_B`, the lowest and highest RSC over the
  count splits (`rsc_split_min`, `rsc_split_max`) and a `status`;
- `summary`: one row per model and section. For spots and cells, fine-scale RSC
  (Fine and Finest scales) and broad-scale RSC (Broadest and Broad), the
  difference between the broadest and finest scales, the fine-scale split-half
  reliability and the whole-section Pearson r. For a grid, the broadest and
  finest scales instead;
- `genes`: the same scores gene by gene.

With two or more sections, rows named `All sections` give the mean over
sections, as the means over samples in the paper. For cells, `C`, `V_model` and
`V_RNA` are averaged over sections before the ratio, as for the patients of the
paper.

`status` is `ESTIMABLE`, `NO_PREDICTED_VARIANCE` (the prediction does not vary
at that scale) or `RNA_UNRESOLVED` (the covariance between the two halves is
not positive, so RSC is not defined). `All sections` rows have the status
`MEAN_OVER_SECTIONS`. `result.save(folder)` writes `scales.tsv`,
`summary.tsv`, `genes.tsv`, `settings.json` and `strsc-evaluation.json`. The
last file opens under [Computed results](https://stRSC.org/analyze?input=results)
on the website, and `result.to_bundle()` returns the same content as a dict.

## Between and within regions

```python
counts = strsc.read_counts("sample.h5ad").with_labels("regions.csv")
result = strsc.regions(counts, {"my_model": "predictions.csv"}, focus="TLS")
print(result.summary)
```

`strsc.regions` scores agreement between and within annotated regions, as for
the TLS sections of the paper (Methods, "Decomposition by region" and
"TLS-associated genes"). Each prediction and each RNA half is split into the
mean of its region and the residual from that mean. RSC is computed for the
whole section, between regions and within regions. When `focus` names a
region, RSC is also computed within it, between its structures (connected
groups of its spots) and within one structure. There is no spatial decomposition.

Measured expression is log(1 + 10,000 N / total), so predictions should be on
the same log scale. Every gene is divided by the standard deviation of its
unsplit expression over the spots of the term. Intervals are 95% percentile
intervals from 2,000 resamples of focus structures (terms within the focus
region) and of blocks of 10 × 10 spots (the other terms). A term is scored in a
section only when the lower limit of the interval of its split-half
reliability is above zero. The other terms have no RSC and the status
`NOT_TESTABLE`. The region column is `region` unless `region=` names another.

From the command line, add `--regions region` and, optionally,
`--focus TLS` and `--labels regions.csv`.

## Expected expression as a prediction

```python
table = strsc.ideal_check(counts, layout="spots", draws=3)
```

`strsc.ideal_check` repeats the paper's test of RSC (Methods, "Expected
expression as a prediction"). The measured counts are taken as Poisson rates,
count matrices are drawn from them, and on each the expected transformed
half-count is scored as the prediction. Its RSC should be near 1 at every
scale. In the paper it was 0.956 to 1.000. The table gives the mean, lowest and
highest RSC over the draws at each scale.

## Command line

```text
strsc rsc COUNTS [COUNTS ...] -p NAME=FILE [-p NAME=FILE ...] [options]
```

| Option | Meaning |
|---|---|
| `-p`, `--prediction` | A prediction file, as `name=file` or a file alone; repeat for each model |
| `--layout` | `spots` (default), `grid` or `cells` |
| `--spacing-um` | Grid spacing in micrometers |
| `--within CLASS` | Score the cells of this class on their own graph; repeat for several classes |
| `--only-within` | With `--within`, do not score all cells as well |
| `--cell-class` | Column that holds the cell classes |
| `--regions COLUMN` | Score between and within the regions in this column |
| `--focus REGION` | With `--regions`, also score within this region and its structures |
| `--labels FILE` | A table with `unit_id` and columns such as `region` or `cell_class` |
| `--um-per-unit` | Micrometers per coordinate unit, when the files do not say |
| `--genes` | Comma-separated genes to score |
| `--splits` | Use only the first count splits |
| `--out` | Output folder (default `strsc-output`) |

## Reproducibility

The toolkit and the website use the same steps and seeds. The numbers agree
except at locations with many molecules of a gene, where numpy draws the count
split with floating-point steps that can differ between computers or numpy
versions. The results then differ within the scatter between count splits
(`rsc_split_min` to `rsc_split_max`). For spots, the projections onto the scales
run in single precision, so the last digits can differ.
