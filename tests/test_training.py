import numpy as np
import torch
import torch.nn.functional as F

from strsc.training import build_within_specimen_adjacency, mixed_expression_mse
from strsc.training.non_neighbours import rewire_non_neighbours


def test_zero_weight_is_native_mse_even_without_edges():
    prediction = torch.tensor([[1., 2.], [4., 5.]], requires_grad=True)
    truth = torch.zeros_like(prediction)
    ordinary = F.mse_loss(prediction, truth)
    actual = mixed_expression_mse(prediction, truth, None, rho=0)
    assert torch.equal(actual, ordinary)
    assert torch.equal(torch.autograd.grad(actual, prediction)[0], torch.autograd.grad(ordinary, prediction)[0])


def test_adjacency_never_connects_specimens():
    xy = np.array([[0., 0.], [1., 0.], [2., 0.]])
    graph = build_within_specimen_adjacency([xy, xy], specimen_ids=['a', 'b'])
    left, right = graph.edge_index
    assert np.all((left < 3) == (right < 3))


def test_non_neighbour_control_preserves_degrees():
    xy = np.column_stack([np.arange(20), np.zeros(20)]).astype(float)
    edges = np.vstack([np.arange(19), np.arange(1, 20)])
    control, metadata = rewire_non_neighbours(xy, edges, radius=1.5)
    np.testing.assert_array_equal(np.bincount(edges.ravel(), minlength=20), np.bincount(control.ravel(), minlength=20))
    assert np.all(np.linalg.norm(xy[control[0]] - xy[control[1]], axis=1) > 1.5)
