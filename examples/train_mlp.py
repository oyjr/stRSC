"""Small synthetic demonstration of neighbouring-pair supervision."""
import argparse
import copy
import json

import numpy as np
import torch

from strsc.training import build_spatial_edges, mixed_expression_mse
from strsc.training.adapters.ridge_mlp.model import TwoLayerMLP


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--seed', type=int, default=17)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    x, y = np.meshgrid(np.arange(12), np.arange(10))
    xy = np.column_stack([x.ravel(), y.ravel()]).astype(float)
    edges, _, _ = build_spatial_edges(xy)
    features = torch.tensor(np.column_stack([xy / 10, np.sin(xy), np.cos(xy)]), dtype=torch.float32)
    target = torch.cat([features[:, :2], features[:, 2:4]], dim=1)
    initial = TwoLayerMLP(features.shape[1], 32, target.shape[1])
    results = []
    for rho in (0.0, 0.5):
        model = copy.deepcopy(initial)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        for _ in range(args.steps):
            optimizer.zero_grad()
            loss = mixed_expression_mse(model(features), target, edges.copy(), rho=rho)
            loss.backward()
            optimizer.step()
        results.append({'rho': rho, 'synthetic_training_loss': float(loss.detach())})
    print(json.dumps({'example': 'synthetic; not paper results', 'steps': args.steps, 'runs': results}, indent=2))


if __name__ == '__main__':
    main()
