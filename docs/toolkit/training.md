# Pairwise difference loss

The pairwise difference loss is

`(1 - rho) × MSE(prediction, target) + rho × MSE(predicted differences, measured differences)`.

The `rho=0` branch calls the native PyTorch mean-squared-error loss directly. The adjacent-difference term uses a coordinate-only graph constructed independently within each training specimen. Its radius is 1.5 times the specimen's median positive nearest-neighbour distance. The evaluation graph is a separate object.

```python
from strsc.training import build_spatial_edges, mixed_expression_mse

edges, spacing, radius = build_spatial_edges(training_xy)
prediction = model(training_inputs)
loss = mixed_expression_mse(prediction, training_expression, edges, rho=0.5)
loss.backward()
```

Node order must match the prediction rows. For minibatches, use the corresponding model adapter to retain and remap the relevant training edges. Preserve the model's other losses, architecture, split and initialization when comparing objectives.

Run the synthetic MLP example:

```bash
python examples/train_mlp.py --steps 100
```

It trains the two-layer MLP head on generated arrays and demonstrates both objectives.

## Model adapters

The adapters use separately installed upstream models. Obtain each model's source and weights, pin the revision declared in the adapter's contract or upstream module, and install its dependencies in a compatible environment. Supply the required image patches or embeddings.

| Adapter package under `strsc.training.adapters` | Models | Entry module |
| --- | --- | --- |
| `ridge_mlp` | Ridge, two-layer MLP | `train_pair` (MLP); `strsc.training.hest_data` (Ridge) |
| `hest_deephis2exp` | ST-Net, DeepSpaCE, HisToGene, Hist2ST | `formal_runner` |
| `hest_deepspot_hist` | DeepSpot and HiST | `formal_runner` |
| `hest_egn_tcgn` | EGNv1, EGNv2, TCGN | `train_pair` |
| `hest_thitogene` | THItoGene | `train_pair` |
| `scellst` | sCellST | `formal_runner` |
| `deepspot2cell` | DeepSpot2Cell | `formal_runner` |

For example:

```bash
python -m strsc.training.adapters.ridge_mlp.train_pair --help
python -m strsc.training.adapters.hest_egn_tcgn.train_pair --help
python -m strsc.training.adapters.scellst.formal_runner --help
```

HEST runners take explicit dataset, gene-manifest, embedding/patch and output paths. Their shared input reader is `strsc.training.hest_data`. Use `--upstream-root` or `--source-root` for separately obtained model code. sCellST and DeepSpot2Cell runners also accept the upstream configuration file.

Some upstream models, including EGN, require older NumPy/PyTorch versions than the scoring environment. Use a separate model environment and add this repository's `src` directory to its `PYTHONPATH`. Export predictions there, then evaluate them in the scoring environment.

Paired runners write fixed training/validation indices and a shared initial state before the two objectives run. When an adapter requires an `engineering-receipt`, its corresponding `engineering_gate` module creates it from the supplied model and data. See each module's `--help` for its model-specific sequence.

## Non-neighbouring control

`strsc.training.non_neighbours.rewire_non_neighbours` constructs the paper's control loss graph. It preserves every node's degree and the number of edges, while requiring all new distances to exceed the original local radius. The function reports an error if these constraints cannot be satisfied. Use the returned graph for the loss and retain the model's message-passing graph.

Reproducing a model comparison requires its data split, model revision, environment and inputs.
