# Paper analyses and editable figures

This directory contains the paper's result tables, reanalysis code and editable figure sources, including the common-gene GHIST comparison, scorer checks and external TLS repeatability analysis.

Run the following commands from the repository root with Python 3.10 or later:

```bash
python -m pip install -r paper/requirements.txt
python paper/analyses/common_gene_support/aggregate.py
python paper/analyses/cell_ideal/aggregate.py
python paper/figures/extended/ED05/figure.py
```

The two aggregators use bundled tables and write an `output/` directory beside their scripts. Their input and output locations can be changed with `--help`. The figure command writes PDF, SVG and PNG beside the source.

| Directory | Contents |
|---|---|
| `source_data/` | Manuscript tables, common-gene results, ordering sensitivity, ideal-prediction checks and TLS results. |
| `analyses/common_gene_support/` | Recomputes patient and cohort summaries, changes and method ranks from 45 section/seed result sets. The `baseline/` tables supply the preceding comparison values. |
| `analyses/cell_ideal/` | Recomputes the calibration gate from 54 specimen/view result sets and their Gram moments. |
| `figures/` | Editable code and required display inputs for 6 main, 9 Extended Data and 12 supplementary figures. |
| `algorithms/` | Count-level cell calibration, CRC/HEST ideal calibration and TLS transformation/repeatability implementations. These require the inputs described in [RAW_INPUTS.md](RAW_INPUTS.md). |

[FIGURES.csv](FIGURES.csv) maps all 27 figure numbers to entrypoints. Run Python entries with `python <entrypoint>` from this directory, or prefix the path with `paper/` from the repository root. Supplementary Figures 6–7 and 9–11 share a renderer. Supplementary Figure 2 writes two PNG panels and an HTML result view. Each renderer uses its adjacent data and writes beside the source.

Figure 1 exports its editable SVG with Node.js and Playwright:

```bash
npm install --prefix paper/figures/main/Figure1 playwright
node paper/figures/main/Figure1/node_modules/playwright/cli.js install chromium
node paper/figures/main/Figure1/render_overview.cjs
```

Set `STRSC_BROWSER` to a compatible installed browser executable to use it instead. Arial is preferred; matplotlib sources supply fallback fonts where supported.

The renderers produce editable figures from the supplied input tables. Small tissue-image crops are included in Figure 1's SVG and Figure 5's input PNG; their source-provider terms apply.

[DATA_SCHEMA.tsv](DATA_SCHEMA.tsv) lists the exact columns and row counts of the supplied analysis tables. Columns named `q` in the supplied tables hold RSC values. Panel summaries aggregate section-level Gram moments within each patient before calculating RSC, then average across patients. The common-gene analysis retains the five GHIST seeds; the two comparator predictions are shared across those seed-specific gene supports. Empty numerical fields indicate undefined quantities.

The supplied tables support result reanalysis and figure generation. GHIST summaries use ten training epochs, five held-out patients and five seeds. Retraining models or regenerating predictions requires the corresponding datasets, named cell/gene axes, splits, graphs and upstream model environments described in [RAW_INPUTS.md](RAW_INPUTS.md).
