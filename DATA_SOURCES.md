# Data in this repository

| Location | Content | How to use it |
|---|---|---|
| `examples/` | Generated arrays and example result bundles | Run the toolkit examples |
| `paper/` | Analysis tables, figure inputs and selected result records | Follow the paper reproduction instructions |

Figure inputs include tissue illustrations embedded in the Figure 1 SVG and a
KC2 tissue crop for Figure 5. These assets retain the source data providers'
terms.

Generated examples are labeled synthetic. The CLI uses several HEST specimen
identifiers as protocol keys in its synthetic workflow; all expression arrays
in that workflow are generated.

## Inputs for scoring research data

The study uses HER2ST, HEST, Visium HD colorectal tissue, Xenium breast tissue
and external kidney/lung TLS sections. Scoring starts from integer RNA counts,
spatial coordinates and saved predictions, with the cohort, cells, genes and
labels required by each analysis. The toolkit guides describe the file layouts;
the [paper instructions](paper/README.md) describe the inputs for its analyses.

Source assay data, histology, annotations and model weights are obtained from
their providers under the applicable access and redistribution terms. Bundled
data comprise the example assets, result tables and figure inputs listed above.
