"""Deterministic, model-specific inner validation splits."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Sequence

import numpy as np

from .data import AlignedHestFold, HestAdapterError
from .models import canonical_model_name


INNER_SPLIT_SALT = "G0055_INNER_VAL_V1"


@dataclass(frozen=True)
class SpecimenInnerSplit:
    model: str
    task: str
    specimen: str
    train_mask: np.ndarray
    validation_mask: np.ndarray
    source: str = "G0-055_GENERAL_INNER_VALIDATION_AMENDMENT_2026-08-24"

    def __post_init__(self) -> None:
        train = np.asarray(self.train_mask, dtype=bool)
        validation = np.asarray(self.validation_mask, dtype=bool)
        if train.ndim != 1 or validation.shape != train.shape:
            raise HestAdapterError(f"INNER_SPLIT_SHAPE:{self.specimen}")
        if np.any(train & validation) or not np.all(train | validation):
            raise HestAdapterError(f"INNER_SPLIT_PARTITION:{self.specimen}")
        if not train.any() or not validation.any():
            raise HestAdapterError(
                f"INNER_SPLIT_EMPTY:{self.model}:{self.task}:{self.specimen}:"
                f"train={int(train.sum())}:validation={int(validation.sum())}"
            )
        train.setflags(write=False)
        validation.setflags(write=False)
        object.__setattr__(self, "train_mask", train)
        object.__setattr__(self, "validation_mask", validation)


def _is_validation(model: str, task: str, specimen: str, barcode: str) -> bool:
    text = f"{INNER_SPLIT_SALT}|{model}|{task}|{specimen}|{barcode}"
    digest = hashlib.blake2b(text.encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return value % 10 == 0


def build_specimen_inner_split(
    *, model: str, task: str, specimen: str, barcodes: Sequence[str]
) -> SpecimenInnerSplit:
    canonical = canonical_model_name(model)
    validation = np.asarray(
        [_is_validation(canonical, str(task), str(specimen), str(barcode)) for barcode in barcodes],
        dtype=bool,
    )
    return SpecimenInnerSplit(
        model=canonical,
        task=str(task),
        specimen=str(specimen),
        train_mask=~validation,
        validation_mask=validation,
    )


def build_fold_inner_splits(
    fold: AlignedHestFold, *, model: str
) -> tuple[SpecimenInnerSplit, ...]:
    return tuple(
        build_specimen_inner_split(
            model=model,
            task=fold.task,
            specimen=specimen.specimen,
            barcodes=specimen.barcodes.tolist(),
        )
        for specimen in fold.train_specimens
    )
