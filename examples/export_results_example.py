"""Generate synthetic inputs and real result bundles for four evaluation paths.

Run from the repository root after installing the package:
    python examples/export_results_example.py --output outputs/export-examples
Every value is generated; these examples are not manuscript evidence.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# Initialize the numerical runtime before NumPy/Pandas on Windows Conda builds.
import torch  # noqa: F401
import numpy as np
import pandas as pd
from strsc.evaluation_export import export_results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output directory")
    root = args.output
    root.mkdir(parents=True)
    from strsc.example import generate
    from strsc.hest import run
    manifest = generate(root / "hest")
    # Evaluate one named prediction.
    spec = json.loads(manifest.read_text())
    spec["predictions"] = {"single_prediction": spec["predictions"]["intervention"]}
    manifest.write_text(json.dumps(spec), encoding="utf-8")
    run(manifest, root / "hest" / "scores", threads=2)
    export_results([root / "hest" / "scores"], root / "hest.json", synthetic=True)

    from strsc.her2 import make_section, evaluate
    sections = []
    for patient, seed in (("synthetic_P1", 7), ("synthetic_P2", 11)):
        rng = np.random.default_rng(seed)
        coords = np.column_stack(np.unravel_index(np.arange(196), (14, 14))).astype(float)
        rates = np.exp(rng.normal(3.5, 0.5, size=(196, 40)))
        sections.append(make_section(patient=patient, section=f"{patient}_section",
            counts=rng.poisson(rates).astype(np.int64),
            prediction=np.stack([rates, rates[rng.permutation(196)]], axis=2),
            coords=coords, regions=np.where(coords[:, 0] < 7, "synthetic_R1", "synthetic_R2"),
            positions=[f"s{i:03d}" for i in range(196)], genes=[f"g{i:02d}" for i in range(40)],
            methods=["aligned_example", "permuted_example"]))
    her2 = root / "her2"
    her2.mkdir()
    for name, frame in evaluate(sections, [2026080501]).items():
        frame.to_csv(her2 / f"{name}.tsv", sep="\t", index=False)
    export_results([her2], root / "her2.json", synthetic=True)

    from strsc.fourier import score
    fourier = root / "fourier"
    fourier.mkdir()
    paths = []
    for unit, seed in (("synthetic_S1", 42), ("synthetic_S2", 43)):
        rng = np.random.default_rng(seed)
        y, x = np.indices((16, 16))
        coords = np.column_stack([x.ravel(), y.ravel()]) * 32.0
        signal = np.sin(2 * np.pi * x.ravel() / 8)
        rate = np.exp(3.5 + 0.5 * signal[:, None] + rng.normal(0, 0.2, (256, 8)))
        table = score(rng.poisson(rate).astype(np.int64), np.stack([rate, -rate], axis=2),
                      coords, np.full(256, 32.0 ** 2), spacing_um=32,
                      methods=["aligned_example", "sign_reversed_example"], seeds=[2026073101])
        path = fourier / f"{unit}.tsv"
        table.to_csv(path, sep="\t", index=False)
        paths.append(path)
    export_results(paths, root / "fourier.json", units=["synthetic_S1", "synthetic_S2"], synthetic=True)

    # Evaluate cell-band components on a small generated graph.
    from strsc.cell_core import build_lazy_graph, lazy_band_grams, model_rsc_from_terms, support_from_terms
    cell = root / "cell"
    cell.mkdir()
    x, y = np.meshgrid(np.arange(6), np.arange(5))
    graph, keep, _, _ = build_lazy_graph(np.column_stack([x.ravel(), y.ravel()]), 8, 3, 20)
    rng = np.random.default_rng(17)
    a = rng.normal(size=(int(keep.sum()), 4)).astype(np.float32)
    b = (a + rng.normal(0, 0.2, size=a.shape)).astype(np.float32)
    p = (a + rng.normal(0, 0.1, size=a.shape)).astype(np.float32)
    terms = lazy_band_grams(graph, {"A": a, "B": b, "P": p}, "cpu")
    context = dict(specimen="synthetic_cell_S1", patient="synthetic_cell_P1", view="native_neoplastic",
                   track="xenium_centred", population="Neoplastic", output_grid_um=0, eligible_genes=4)
    pd.DataFrame([{**context, "model": "synthetic_prediction", **row} for row in model_rsc_from_terms(terms)]).to_csv(
        cell / "synthetic_SPECIMEN_BAND_RSC.tsv", sep="\t", index=False)
    pd.DataFrame([{**context, **row} for row in support_from_terms(terms)]).to_csv(
        cell / "synthetic_RNA_SUPPORT_SPECIMEN_BAND.tsv", sep="\t", index=False)
    export_results([cell], root / "cell.json", synthetic=True)
    print(json.dumps({"output": str(root), "bundles": ["hest.json", "her2.json", "fourier.json", "cell.json"],
                      "synthetic": True, "hest_models": 1, "fourier_units": 2}))


if __name__ == "__main__":
    main()
