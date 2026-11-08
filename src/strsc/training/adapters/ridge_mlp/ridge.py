"""Ridge fitting through the shared HEST preprocessing and training module."""

from __future__ import annotations

import importlib.util
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any


class RidgeAdapterError(RuntimeError):
    pass


def default_g0054_module_path() -> Path:
    return (
        Path(__file__).resolve().parents[2] / "hest_data.py"
    )


@lru_cache(maxsize=4)
def load_g0054_module(path: str | Path | None = None) -> ModuleType:
    """Load the shared Ridge module from the package or an explicit source path."""

    source = Path(path) if path is not None else default_g0054_module_path()
    source = source.resolve()
    if not source.is_file():
        raise RidgeAdapterError(f"MISSING_G0054_MODULE:{source}")
    name = f"g0054_frozen_{abs(hash(str(source)))}"
    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None or spec.loader is None:
        raise RidgeAdapterError(f"G0054_IMPORT_SPEC:{source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    required = {
        "TASK_SAMPLES",
        "OFFICIAL_GENES",
        "OFFICIAL_PCA",
        "_read_embedding",
        "_read_gene_manifest",
        "_read_official_split",
        "_matched_truth",
        "fit_frozen_ridge",
    }
    missing = sorted(required.difference(vars(module)))
    if missing:
        raise RidgeAdapterError(f"G0054_API_MISSING:{missing}")
    return module


def fit_frozen_ridge(*args: Any, g0054_module: str | Path | None = None, **kwargs: Any):
    """Fit Ridge through the shared HEST implementation."""

    return load_g0054_module(g0054_module).fit_frozen_ridge(*args, **kwargs)
