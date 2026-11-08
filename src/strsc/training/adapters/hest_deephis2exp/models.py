"""Lazy construction of unmodified DeepHis2Exp model classes."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any


_CANONICAL = {
    "st-net": "ST-Net",
    "stnet": "ST-Net",
    "deep-space": "DeepSpaCE",
    "deepspace": "DeepSpaCE",
    "deepspace": "DeepSpaCE",
    "histogene": "HisToGene",
    "his-to-gene": "HisToGene",
    "hist2st": "Hist2ST",
}


def canonical_model_name(name: str) -> str:
    value = str(name).strip()
    if value in {"ST-Net", "DeepSpaCE", "HisToGene", "Hist2ST"}:
        return value
    key = value.lower().replace("_", "-")
    if key not in _CANONICAL:
        raise ValueError(f"Unsupported DeepHis2Exp HEST model: {name!r}")
    return _CANONICAL[key]


def add_upstream_to_path(deps_root: Path) -> Path:
    root = Path(deps_root).resolve()
    if not (root / "Models").is_dir():
        raise FileNotFoundError(f"DeepHis2Exp Models directory missing: {root}")
    text = str(root)
    if text not in sys.path:
        sys.path.insert(0, text)
    return root


def build_upstream_model(
    model_name: str,
    *,
    deps_root: Path,
    n_genes: int = 50,
) -> Any:
    """Instantiate the exact class/hyperparameters used by upstream Baseline.py."""

    canonical = canonical_model_name(model_name)
    add_upstream_to_path(deps_root)
    if int(n_genes) != 50:
        raise ValueError("Frozen G0-055 HEST adapters require exactly 50 genes")

    if canonical == "ST-Net":
        from Models.STnet.stnet import STModel

        return STModel(n_genes=50, hidden_dim=1024, learning_rate=1e-6)
    if canonical == "DeepSpaCE":
        from Models.DeepSpaCE.deepspace import DeepSpaCE

        return DeepSpaCE(n_genes=50, hidden_dim=4096, learning_rate=1e-4)
    if canonical == "HisToGene":
        from Models.HisToGene.histogene import HisToGene

        return HisToGene(
            patch_size=224,
            n_layers=4,
            n_genes=50,
            dim=1024,
            learning_rate=1e-5,
            dropout=0.1,
            n_pos=256,
        )
    if canonical == "Hist2ST":
        from Models.Hist2ST.hist2st import Hist2ST

        return Hist2ST(
            depth1=2,
            depth2=8,
            n_genes=50,
            learning_rate=1e-5,
            label=None,
            kernel_size=5,
            patch_size=7,
            n_pos=256,
            heads=16,
            channel=32,
            dropout=0.2,
            zinb=0.25,
            nb=False,
            bake=5,
            lamb=0.5,
        )
    raise AssertionError(canonical)
