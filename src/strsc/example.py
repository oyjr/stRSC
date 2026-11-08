"""Generate synthetic counts and predictions, then run the HEST scorer."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def generate(output: Path, seed: int = 17) -> Path:
    if output.exists():
        raise ValueError("Choose a new output directory.")
    output.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    x, y = np.meshgrid(np.arange(12), np.arange(10))
    xy = np.column_stack([x.ravel(), y.ravel()]).astype(float)
    genes = np.array([f"synthetic_gene_{i:02d}" for i in range(50)])
    barcodes = np.array([f"synthetic_site_{i:03d}" for i in range(len(xy))])
    phase = np.linspace(0, 2 * np.pi, 50, endpoint=False)
    broad = 1.0 + np.sin(xy[:, :1] / 4 + phase)
    local = 1.0 + np.cos(2 * xy[:, 1:2] + phase)
    rate = 40 + 30 * broad + 20 * local
    for name in ("TENX118", "TENX141"):
        counts = rng.poisson(rate).astype(np.int64)
        np.savez_compressed(output / f"{name}.npz", counts=counts, xy=xy, genes=genes, barcodes=barcodes)
    for name, prediction in (
        ("baseline", np.log1p(40 + 30 * broad) + rng.normal(0, 0.1, rate.shape)),
        ("intervention", np.log1p(rate) + rng.normal(0, 0.04, rate.shape)),
    ):
        with h5py.File(output / f"{name}.h5", "w") as handle:
            handle.create_dataset("prediction", data=prediction)
            handle.create_dataset("barcodes", data=barcodes.astype('S'))
            handle.create_dataset("genes", data=genes.astype('S'))
    manifest = {
        "protocol": "HEST_G0047B",
        "task": "LUNG",
        "specimen": "TENX118",
        "model": "Synthetic illustration",
        "prediction_transform": "log1p_raw_integer_counts_no_library_normalization",
        "cohort": {name: f"{name}.npz" for name in ("TENX118", "TENX141")},
        "predictions": {name: f"{name}.h5" for name in ("baseline", "intervention")},
        "synthetic": True,
        "note": "All arrays are generated. HEST specimen names are protocol keys only; these are not patient data or study results.",
    }
    path = output / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--generate-only', action='store_true')
    args = parser.parse_args()
    manifest = generate(args.output, args.seed)
    if not args.generate_only:
        from .hest import run
        from .export import export_analysis
        run(manifest, args.output / 'scores', threads=2)
        export_analysis(manifest, args.output / 'scores', args.output / 'analysis-bundle.json',
                        baseline_label='Synthetic prediction A', intervention_label='Synthetic prediction B')
    print(f"Synthetic example: {args.output.resolve()}")


if __name__ == '__main__':
    main()
