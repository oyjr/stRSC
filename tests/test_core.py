import numpy as np
from scipy.stats import pearsonr

from strsc.hest_core import _corr_cols, _graph, _halves


def test_pearson_matches_scalar_reference_and_keeps_undefined():
    rng = np.random.default_rng(8)
    truth = rng.normal(size=(40, 4))
    prediction = truth + rng.normal(size=truth.shape)
    prediction[:, 3] = 1
    observed = _corr_cols(prediction, truth)
    np.testing.assert_allclose(observed[:3], [pearsonr(prediction[:, i], truth[:, i]).statistic for i in range(3)])
    assert observed[3] == 0
    truth[:, 3] = 1
    assert np.isnan(_corr_cols(prediction, truth)[3])


def test_molecule_split_is_reproducible_and_centered():
    counts = np.arange(1, 161).reshape(40, 4)
    a, b = _halves(counts, 'LUNG', 'TENX118', 2026080701)
    a2, b2 = _halves(counts, 'LUNG', 'TENX118', 2026080701)
    np.testing.assert_array_equal(a, a2)
    np.testing.assert_array_equal(b, b2)
    np.testing.assert_allclose(a.mean(axis=0), 0, atol=1e-14)
    np.testing.assert_allclose(b.mean(axis=0), 0, atol=1e-14)


def test_graph_discards_small_disconnected_components():
    x, y = np.meshgrid(np.arange(6), np.arange(5))
    xy = np.column_stack([x.ravel(), y.ravel()])
    xy = np.vstack([xy, [[100, 100], [101, 100]]])
    adjacency, keep, labels, spacing, radius = _graph(xy)
    assert keep.sum() == 30
    assert adjacency.shape == (30, 30)
    assert len(set(labels)) == 1
    assert radius == 1.25 * spacing
