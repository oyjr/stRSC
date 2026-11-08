# stRSC

Reproducible-signal correlation (RSC) for spatial transcriptomics predicted from
histology.

[Website](https://stRSC.org/) · [Interactive examples](https://stRSC.org/examples) · [User guide](https://stRSC.org/docs)

This repository accompanies **A noise-corrected metric for evaluating spatial
transcriptomics predicted from histology**. It contains the code for RSC, the
evaluations in the study and the pairwise difference loss, together with the
scripts that produce the study's result tables and figures.

## What RSC measures

Correlation with measured RNA counts mixes the expression signal with sampling
noise, which dominates sparse counts at fine spatial scales. RSC splits the
counts of each gene in each bin, spot or cell binomially into two halves. The
two halves share the expected expression and carry independent sampling noise.
At each spatial scale `b`, with halves `A` and `B` and prediction `P`:

```text
C       = (<P, A>_b + <P, B>_b) / 2      mean covariance of the prediction with the halves
V_model = <P, P>_b                       variance of the prediction
V_RNA   = <A, B>_b                       covariance between the halves
RSC     = C / sqrt(V_model * V_RNA)
```

The split-half reliability `V_RNA / sqrt(<A, A>_b <B, B>_b)` is the correlation
between the two halves. Sums run over all genes and all bins, spots or cells of
a section, and scores are averaged over repeated count splits. Counts must be
raw integer molecule counts, split before normalization, and predictions must
not use the counts they are evaluated against.

## Installation

Use Python 3.10 or later:

```bash
git clone https://github.com/oyjr/stRSC.git
cd stRSC
python -m venv .venv
```

Activate the environment:

```bash
# macOS / Linux
source .venv/bin/activate
```

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install the package:

```bash
python -m pip install -e .
```

## Run an example

Generate counts and two predictions, then compute RSC in five spatial bands:

```bash
strsc example --output outputs/example
strsc export-results --run outputs/example/scores --output outputs/results.json --synthetic
```

The example writes:

- `outputs/example/scores/`: RSC by band, gene-wise Pearson correlations and the RNA components;
- `outputs/example/analysis-bundle.json`: paired spatial maps for the website;
- `outputs/results.json`: saved evaluation results for the website.

The example runs on a CPU with generated data. Choose a new output directory for
each run.

To inspect the output online, open
[Computed results](https://stRSC.org/analyze?input=results), upload
`outputs/example/analysis-bundle.json` or `outputs/results.json`, then select
**View results**. Files are processed in the browser. The same workflow
applies to results computed from your own data.

## Evaluate your predictions

| Analysis | Command or API | Inputs and instructions |
|---|---|---|
| Pairwise accuracy across and within tissue regions | `strsc her2` | [Spot pairs](docs/toolkit/her2.md) |
| RSC in graph-frequency bands of spot data | `strsc hest` | [HEST workflow](docs/toolkit/hest.md) |
| RSC by physical wavelength on regular grids | `strsc fourier` | [Fourier workflow](docs/toolkit/fourier.md) |
| RSC for cells and aggregated grids, proliferation hotspots | `python -m strsc.cell_score` | [Cell workflow](docs/toolkit/cell.md) |
| Pairwise difference loss and model adapters | `strsc.training` | [Loss and model adapters](docs/toolkit/training.md) |
| Export evaluation results | `strsc export-results` | [Result format](docs/toolkit/export-results.md) |

Each guide specifies the expression transform, cell or site identifiers, gene
identifiers and spatial coordinates required by its workflow. Install optional
dependencies for cell preparation or model training:

```bash
python -m pip install -e ".[cell]"
python -m pip install -e ".[training]"
```

## Paper analyses and figures

The [paper directory](paper/README.md) contains result tables, aggregation
scripts and figure sources. The documented aggregation commands run from bundled
tables. [Count-level recalculation](paper/RAW_INPUTS.md) uses source RNA counts
and saved predictions supplied in the specified formats.

## Tests

```bash
python -m pip install -e ".[test]"
python -m pytest -q
```

## Citation and license

Author information and the software version are in [CITATION.cff](CITATION.cff).
The accompanying manuscript is *A noise-corrected metric for evaluating spatial
transcriptomics predicted from histology* by Ouyang, Gao and colleagues.

The code is released under the [MIT license](LICENSE). Third-party
dependencies, models and datasets retain their own terms; see [NOTICE.md](NOTICE.md).
