import numpy as np
from scipy import sparse

from strsc.cell_core import aggregate_cells, build_lazy_graph, lazy_band_grams, split_counts


def test_sparse_molecule_halves_conserve_counts():
    counts = sparse.csr_matrix(np.arange(40).reshape(10, 4))
    a, b = split_counts(counts, 'synthetic', 17)
    assert (a + b != counts).nnz == 0
    assert np.all(a.data >= 0) and np.all(b.data >= 0)


def test_aggregation_preserves_molecules_and_cell_exposure():
    xy = np.array([[0., 0.], [1., 0.], [5., 0.], [6., 0.]])
    counts = sparse.csr_matrix(np.arange(8).reshape(4, 2))
    output, centers, exposure = aggregate_cells([counts], xy, 4)
    np.testing.assert_array_equal(output[0].sum(axis=0), counts.sum(axis=0))
    np.testing.assert_array_equal(exposure, [2, 2])
    np.testing.assert_allclose(centers, [[0.5, 0], [5.5, 0]])


def test_five_diffusion_band_cross_products_sum_to_identity():
    x, y = np.meshgrid(np.arange(6), np.arange(5))
    xy = np.column_stack([x.ravel(), y.ravel()])
    diffusion, keep, _, _ = build_lazy_graph(xy, 8, 3, 20)
    rng = np.random.default_rng(17)
    a = rng.normal(size=(int(keep.sum()), 4)).astype(np.float32)
    b = rng.normal(size=a.shape).astype(np.float32)
    bands = lazy_band_grams(diffusion, {'A': a, 'B': b}, 'cpu')
    observed = sum(band[('A', 'B')] for band in bands.values())
    np.testing.assert_allclose(observed, np.sum(a.astype(np.float64) * b), atol=1e-5)
