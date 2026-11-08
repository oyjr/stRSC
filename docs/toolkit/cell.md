# Cell-level evaluation

The cell pipeline uses the same graph and five lazy-diffusion filters for RNA halves and predictions. It evaluates native cells and aggregated grids. `configs/cell_protocol.json` defines the molecule-split seeds, graph parameters, grid sizes and proliferation gene programme.

Install the cell dependencies:

```bash
python -m pip install -e ".[cell]"
```

## Prepare counts and cell metadata

```bash
python -m strsc.cell_prepare --specimen MY_SLIDE --patient MY_PATIENT \
  --transcripts data/transcripts.parquet \
  --xenium-nuclei data/xenium_nuclei.parquet \
  --cellvit data/cellvit_cells.parquet --metadata data/metadata.json \
  --protocol configs/cell_protocol.json --output-root data/canonical
```

This entry point follows the study's HEST Xenium format: transcript Parquet columns `cell_id`, `feature_name` and `qv`; Xenium nuclei in GeoParquet with cell IDs in the index and registered pixel-space geometries; CellViT GeoParquet with `geometry` and `class` columns (and cell IDs in a column or index); and JSON metadata containing `pixel_size_um_estimated`, `pixel_size_um_embedded` or `pixel_size`. It counts retained transcripts, aligns cell identities and records the spatial units. Use `--gene-list` for an explicit input panel.

The command writes a sparse `counts_cells_by_genes.npz`, `cell_axis.parquet`, `gene_axis.parquet` and `metadata.json` to the canonical directory.

For already prepared inputs, `strsc.cell_core.save_canonical` accepts a CSR integer count matrix, a cell table, a gene table and metadata. Cell rows need `cell_id`, `x_um`, `y_um` and `cell_class`; gene rows need `gene_index` and `gene_symbol`; metadata needs `patient`. Array order must match both tables. The optional CellViT-native track additionally requires the alignment columns produced by the preparation step.

## Split-half reliability

```bash
python -m strsc.cell_rna --specimen MY_SLIDE \
  --canonical-root data/canonical --protocol configs/cell_protocol.json \
  --cache-root outputs/views --results outputs/rna --device cpu
```

The command constructs native and aggregated views, splits molecules and accumulates RNA components using the protocol's filtering, exposure normalization and graph support.

## Score predictions

```bash
python -m strsc.cell_score --model sCellST --specimen MY_SLIDE \
  --prediction data/prediction.h5ad --canonical-root data/canonical \
  --view-cache-root outputs/views --protocol configs/cell_protocol.json \
  --results outputs/cell_scores --device cpu
```

Use `--model DeepSpot2Cell` for that model. The prediction H5AD stores cells in rows, genes in columns, IDs in `obs_names` and gene symbols in `var_names`, with the model's native log1p-normalized output in `X`. The scorer aligns IDs explicitly and records missing coverage. The two model entry points share the same numerical implementation.

The five filters are `S^16`, `S^8-S^16`, `S^4-S^8`, `S^2-S^4` and `I-S^2`, where `S` is the lazy symmetric diffusion operator. The filters sum to the identity. Cell scoring also evaluates proliferation hotspots within Neoplastic cells, using the four programme genes specified in the protocol.
