"""Frozen HEST contract shared by the DeepSpot and HiST overlays."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Sequence

import numpy as np


MODELS = ("DeepSpot", "HiST")
SEEDS = (17, 23, 41)
RHOS = (0.0, 0.5)
GENES = 50
INNER_SPLIT_SALT = "G0055_INNER_VAL_V1"
INNER_SPLIT_SOURCE = "G0-055_GENERAL_INNER_VALIDATION_AMENDMENT_2026-08-24"
SOURCE_REVISIONS = {
    "DeepSpot": "253c334f878da2ca1c558779dfffd7476e8687c5",
    "HiST": "3e60ea39388d71a14d61063a2638db3b0b5294e6",
}
TASK_SPECIMENS: dict[str, tuple[str, ...]] = {
    "IDC": ("TENX99", "TENX95", "NCBI785", "NCBI783"),
    "PAAD": ("TENX116", "TENX140", "TENX126"),
    "LUNG": ("TENX141", "TENX118"),
}


class HestDeepSpotHiSTError(RuntimeError):
    """A frozen model, split, feature, or output contract was violated."""


def canonical_model_name(value: str) -> str:
    key = str(value).strip().lower().replace("_", "").replace("-", "")
    aliases = {
        "deepspot": "DeepSpot",
        "hist": "HiST",
    }
    if key not in aliases:
        raise HestDeepSpotHiSTError(f"MODEL_UNSUPPORTED:{value}")
    return aliases[key]


def validate_rho(value: float) -> float:
    rho = float(value)
    if rho not in RHOS:
        raise HestDeepSpotHiSTError(f"RHO_FROZEN:{value}:allowed={RHOS}")
    return rho


@dataclass(frozen=True)
class SpecimenInnerSplit:
    model: str
    task: str
    specimen: str
    barcodes: tuple[str, ...]
    train_mask: np.ndarray
    validation_mask: np.ndarray
    source: str = INNER_SPLIT_SOURCE

    def __post_init__(self) -> None:
        model = canonical_model_name(self.model)
        train = np.asarray(self.train_mask, dtype=bool)
        validation = np.asarray(self.validation_mask, dtype=bool)
        if train.ndim != 1 or validation.shape != train.shape:
            raise HestDeepSpotHiSTError(f"INNER_SPLIT_SHAPE:{self.specimen}")
        if len(self.barcodes) != len(train) or len(set(self.barcodes)) != len(self.barcodes):
            raise HestDeepSpotHiSTError(f"INNER_SPLIT_BARCODES:{self.specimen}")
        if np.any(train & validation) or not np.all(train | validation):
            raise HestDeepSpotHiSTError(f"INNER_SPLIT_PARTITION:{self.specimen}")
        if not train.any() or not validation.any():
            raise HestDeepSpotHiSTError(
                f"INNER_SPLIT_EMPTY:{model}:{self.task}:{self.specimen}:"
                f"train={int(train.sum())}:validation={int(validation.sum())}"
            )
        train.setflags(write=False)
        validation.setflags(write=False)
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "train_mask", train)
        object.__setattr__(self, "validation_mask", validation)


def is_inner_validation(
    *, model: str, task: str, specimen: str, barcode: str
) -> bool:
    canonical = canonical_model_name(model)
    token = f"{INNER_SPLIT_SALT}|{canonical}|{task}|{specimen}|{barcode}"
    digest = hashlib.blake2b(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) % 10 == 0


def build_specimen_inner_split(
    *, model: str, task: str, specimen: str, barcodes: Sequence[str]
) -> SpecimenInnerSplit:
    identity = tuple(str(value) for value in barcodes)
    validation = np.asarray(
        [
            is_inner_validation(
                model=model, task=str(task), specimen=str(specimen), barcode=barcode
            )
            for barcode in identity
        ],
        dtype=bool,
    )
    return SpecimenInnerSplit(
        model=canonical_model_name(model),
        task=str(task),
        specimen=str(specimen),
        barcodes=identity,
        train_mask=~validation,
        validation_mask=validation,
    )


def build_fold_inner_splits(fold: Any, *, model: str) -> tuple[SpecimenInnerSplit, ...]:
    if len(np.asarray(fold.genes)) != GENES:
        raise HestDeepSpotHiSTError(f"GENE_COUNT:{len(np.asarray(fold.genes))}")
    return tuple(
        build_specimen_inner_split(
            model=model,
            task=str(fold.task),
            specimen=str(specimen.specimen),
            barcodes=np.asarray(specimen.barcodes, dtype=object).tolist(),
        )
        for specimen in fold.train_specimens
    )


def expected_outer_fold(task: str, fold: int) -> str:
    task_name = str(task)
    if task_name not in TASK_SPECIMENS:
        raise HestDeepSpotHiSTError(f"TASK_UNSUPPORTED:{task_name}")
    specimens = TASK_SPECIMENS[task_name]
    if int(fold) < 0 or int(fold) >= len(specimens):
        raise HestDeepSpotHiSTError(f"FOLD_RANGE:{task_name}:{fold}")
    return specimens[int(fold)]
