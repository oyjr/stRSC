# Physical-scale scoring on a regular grid

```bash
strsc fourier --input inputs/grid.npz --spacing-um 32 --out outputs/fourier.tsv
```

This command starts from raw integer RNA counts and fixed predictions. It
performs count splitting, exposure correction, normalization, Fourier
decomposition and RSC scoring on a prepared regular grid.

The NPZ file contains:

| Array | Shape | Meaning |
|---|---|---|
| `counts` | occupied sites × genes | Nonnegative integer RNA molecule counts |
| `prediction` | occupied sites × genes, optionally × models | Fixed, finite expression predictions |
| `coords` | occupied sites × 2 | Grid-centre x and y coordinates in micrometres |
| `area` | occupied sites | Effective tissue area in square micrometres, strictly positive |
| `methods` | models | Optional unique model names, stored as strings |

All arrays must describe the same occupied sites in the same order and the same
fixed gene panel. Supply the selected genes and occupied sites explicitly.
`area` is the tissue area within each occupied bin;
it can be smaller than the squared grid spacing at tissue boundaries.

Only occupied tissue sites are supplied. They must lie on the specified regular
grid. The bounding rectangle is reconstructed and sites outside the supplied
tissue outline are zero-padded. Predictions are centred and L2-normalized for
each gene across occupied sites.

RNA counts undergo five independent binomial 50/50 splits with seeds 2026073101 through 2026073105. Each half is transformed as
`sqrt(half_counts / (0.5 * effective_area))`, centred across sites and divided
by the same gene-specific scale for both halves:
`sqrt((sum(A²) + sum(B²)) / 2)`. Constant columns become zero. Genes are
processed in blocks of 32 with block-specific seed offsets. Exact reproduction
requires the same site order, gene order and gene panel. Use `--seeds` to
specify a different molecule-split set.

The orthonormal two-dimensional Fourier transform partitions all nonconstant
modes into seven physical wavelength bands: below 128, 128–256, 256–512,
512–1,024, 1,024–2,048, 2,048–4,096 and at least 4,096 micrometres.
Wavelength is the length of one complete spatial cycle. Bounds include the
lower limit and exclude the upper limit.

For normalized model coefficients `P` and RNA-half coefficients `A` and `B`,
each band reports the summed sufficient statistics:

```text
v_rna   = mean over splits of sum Re(conj(A) * B)
c_model = mean over splits of 0.5 * sum Re(conj(P) * (A + B))
v_model = sum |P|²
rsc     = c_model / sqrt(v_rna * v_model)
```

Sums cover the band's modes and supplied genes. Negative RNA signal estimates
are retained. A band with no Fourier modes, nonpositive RNA signal or zero
model variance receives a missing `rsc` and an explicit status. Valid `rsc` values
retain their sign and full numerical range. `ALL_NON_DC` combines the sufficient
statistics of every band before calculating its score.

`nyquist_status` distinguishes fully represented physical windows from partial
high-frequency windows. A band is marked full when its lower wavelength bound
is at least twice the grid spacing. Scores from partial windows remain visible
for inspection, but the paper's physical-scale comparisons use full windows.

The command evaluates a prepared regular grid. Reproducing the full CRC
analysis also requires model fitting, gene selection, virtual-spot construction
from transcript coordinates, specimen aggregation and confidence interval
estimation.

The Python API is `strsc.fourier.score(counts, prediction, coords, area,
spacing_um=32, methods=None, seeds=...)`; `counts` may also be a SciPy CSR matrix.
