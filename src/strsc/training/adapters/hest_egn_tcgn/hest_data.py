"""Strict HEST 9x50 alignment without loading an outer-test endpoint early."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Optional, Sequence

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from strsc.training.adapters.ridge_mlp.ridge import load_g0054_module

from .contracts import MODEL_CONFIGS, canonical_model_name


VALIDATION_PREFIX = "G0055_INNER_VAL_V1"
PATCH_VALUE_SCALE = "raw_0_255"
PHYSICAL_SUPPORT_UM = 112.0


class HestAdapterDataError(RuntimeError):
    pass


@dataclass(frozen=True)
class PatchStore:
    task: str
    specimen: str
    path: Path
    image_key: str
    barcode_key: str
    layout: str
    physical_support_um: float
    value_scale: str


@dataclass(frozen=True)
class EGNExpressionScaler:
    minimum: np.ndarray
    maximum: np.ndarray

    def transform(self, raw_counts: np.ndarray) -> np.ndarray:
        logged = np.log10(1.0 + np.asarray(raw_counts, dtype=np.float64))
        return (logged - self.minimum) / (self.maximum - self.minimum + 1e-8)

    def to_metadata(self) -> dict[str, object]:
        return {
            "source": "outer_train_supervised_rows_only",
            "minimum": self.minimum.tolist(),
            "maximum": self.maximum.tolist(),
        }


@dataclass(frozen=True)
class SectionMetadata:
    model: str
    task: str
    specimen: str
    fold: int
    is_test: bool
    barcodes: np.ndarray
    genes: np.ndarray
    spatial_coords: np.ndarray
    validation_mask: Optional[np.ndarray]
    raw_counts: Optional[np.ndarray]
    target: Optional[np.ndarray]
    patch_store: PatchStore

    @property
    def training_mask(self) -> np.ndarray:
        if self.validation_mask is None:
            raise HestAdapterDataError("TEST_SECTION_HAS_NO_TRAINING_MASK")
        return ~self.validation_mask


@dataclass(frozen=True)
class FoldData:
    model: str
    task: str
    fold: int
    training: tuple[SectionMetadata, ...]
    held_out: Optional[SectionMetadata]
    expression_scaler: Optional[EGNExpressionScaler]


def _decode(values: Any) -> np.ndarray:
    values = np.asarray(values).reshape(-1)
    return np.asarray(
        [
            value.decode("utf-8")
            if isinstance(value, (bytes, np.bytes_))
            else str(value)
            for value in values
        ],
        dtype=object,
    )


def validation_mask(
    *, model: str, task: str, specimen: str, barcodes: np.ndarray
) -> np.ndarray:
    name = canonical_model_name(model)
    result = np.zeros(len(barcodes), dtype=bool)
    for index, barcode in enumerate(barcodes):
        token = f"{VALIDATION_PREFIX}|{name}|{task}|{specimen}|{str(barcode)}"
        digest = hashlib.blake2b(token.encode("utf-8")).digest()
        result[index] = int.from_bytes(digest[:8], "big", signed=False) % 10 == 0
    if result.all() or not result.any():
        raise HestAdapterDataError(
            f"VALIDATION_EMPTY_PARTITION:{name}:{task}:{specimen}:"
            f"train={int((~result).sum())}:validation={int(result.sum())}"
        )
    return result


def tcgn_expression_transform(raw_counts: np.ndarray) -> np.ndarray:
    """Upstream scprep defaults on the ordered task-specific 50-gene panel."""

    raw = np.asarray(raw_counts, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != 50 or raw.min(initial=0) < 0:
        raise HestAdapterDataError(f"RAW_COUNT_CONTRACT:{raw.shape}")
    library = raw.sum(axis=1, keepdims=True)
    normalized = np.divide(raw, library, out=np.zeros_like(raw), where=library > 0)
    normalized *= 10000.0
    result = np.log10(1.0 + normalized)
    if not np.isfinite(result).all():
        raise HestAdapterDataError("TCGN_EXPRESSION_TRANSFORM_NONFINITE")
    return result


def fit_egn_expression_scaler(
    sections: Sequence[SectionMetadata],
) -> EGNExpressionScaler:
    blocks = []
    for section in sections:
        if section.raw_counts is None or section.validation_mask is None:
            raise HestAdapterDataError(
                f"EGN_SCALER_TRAINING_RNA_MISSING:{section.specimen}"
            )
        blocks.append(np.log10(1.0 + section.raw_counts[~section.validation_mask]))
    values = np.concatenate(blocks, axis=0)
    if values.shape[1] != 50 or not np.isfinite(values).all():
        raise HestAdapterDataError(f"EGN_SCALER_CONTENT:{values.shape}")
    return EGNExpressionScaler(values.min(axis=0), values.max(axis=0))


def load_patch_manifest(path: Path) -> dict[tuple[str, str], PatchStore]:
    frame = pd.read_csv(path, sep="\t")
    required = {
        "task",
        "specimen",
        "patch_h5",
        "image_key",
        "barcode_key",
        "layout",
        "physical_support_um",
        "value_scale",
    }
    if not required.issubset(frame.columns):
        raise HestAdapterDataError(
            f"PATCH_MANIFEST_COLUMNS:{sorted(required.difference(frame.columns))}"
        )
    if frame.duplicated(["task", "specimen"]).any():
        raise HestAdapterDataError("PATCH_MANIFEST_DUPLICATE_KEY")
    result: dict[tuple[str, str], PatchStore] = {}
    for row in frame.itertuples(index=False):
        layout = str(row.layout).upper()
        support = float(row.physical_support_um)
        value_scale = str(row.value_scale)
        if layout not in {"NCHW", "NHWC"}:
            raise HestAdapterDataError(f"PATCH_LAYOUT:{layout}")
        if not np.isclose(support, PHYSICAL_SUPPORT_UM):
            raise HestAdapterDataError(
                f"PATCH_PHYSICAL_SUPPORT:{support}:{PHYSICAL_SUPPORT_UM}"
            )
        if value_scale != PATCH_VALUE_SCALE:
            raise HestAdapterDataError(
                f"PATCH_VALUE_SCALE:{value_scale}:{PATCH_VALUE_SCALE}"
            )
        patch_path = Path(str(row.patch_h5))
        if not patch_path.is_absolute():
            patch_path = path.parent / patch_path
        item = PatchStore(
            task=str(row.task),
            specimen=str(row.specimen),
            path=patch_path.resolve(),
            image_key=str(row.image_key),
            barcode_key=str(row.barcode_key),
            layout=layout,
            physical_support_um=support,
            value_scale=value_scale,
        )
        result[(item.task, item.specimen)] = item
    return result


def _embedding_map(path: Path, encoder: str) -> dict[tuple[str, str], Path]:
    frame = pd.read_csv(path, sep="\t")
    required = {"task", "specimen", "encoder", "embedding_path"}
    if not required.issubset(frame.columns):
        raise HestAdapterDataError(
            f"EMBEDDING_MANIFEST_COLUMNS:{sorted(required.difference(frame.columns))}"
        )
    selected = frame.loc[frame["encoder"].astype(str).eq(encoder)]
    if selected.duplicated(["task", "specimen"]).any():
        raise HestAdapterDataError("EMBEDDING_MANIFEST_DUPLICATE_KEY")
    result: dict[tuple[str, str], Path] = {}
    for row in selected.itertuples(index=False):
        value = Path(str(row.embedding_path))
        if not value.is_absolute():
            value = path.parent / value
        result[(str(row.task), str(row.specimen))] = value.resolve()
    return result


def _coordinates_without_rna(
    *, dataset_root: Path, task: str, specimen: str, barcodes: np.ndarray
) -> np.ndarray:
    path = dataset_root / task / "adata" / f"{specimen}.h5ad"
    if not path.is_file():
        raise HestAdapterDataError(f"MISSING_ADATA:{path}")
    with h5py.File(path, "r") as handle:
        obs = _decode(handle["obs/_index"][...])
        coords = np.asarray(handle["obsm/spatial"], dtype=np.float64)
    lookup = {barcode: index for index, barcode in enumerate(obs)}
    if len(lookup) != len(obs) or any(barcode not in lookup for barcode in barcodes):
        raise HestAdapterDataError(
            f"HELD_OUT_BARCODE_JOIN:{task}:{specimen}"
        )
    order = np.asarray([lookup[barcode] for barcode in barcodes], dtype=np.int64)
    result = coords[order]
    if result.shape != (len(barcodes), 2) or not np.isfinite(result).all():
        raise HestAdapterDataError(
            f"HELD_OUT_COORDINATES:{task}:{specimen}:{result.shape}"
        )
    return result


def load_fold_data(
    *,
    model: str,
    dataset_root: Path,
    gene_manifest: Path,
    embedding_manifest: Path,
    patch_manifest: Path,
    task: str,
    fold: int,
    include_test: bool = False,
    include_test_rna: bool = False,
    support_encoder: str = "phikon_v2",
    g0054_module: Optional[Path] = None,
) -> FoldData:
    """Load outer-training RNA first; outer-test RNA is opt-in and off by default."""

    name = canonical_model_name(model)
    frozen = load_g0054_module(g0054_module)
    if task not in frozen.TASK_SAMPLES:
        raise HestAdapterDataError(f"TASK:{task}")
    samples = tuple(frozen.TASK_SAMPLES[task])
    if fold < 0 or fold >= len(samples):
        raise HestAdapterDataError(f"FOLD:{task}:{fold}")
    genes = frozen._read_gene_manifest(gene_manifest, task)
    split_root = dataset_root / task / "splits"
    training_specimens = frozen._read_official_split(split_root / f"train_{fold}.csv")
    test_specimens = frozen._read_official_split(split_root / f"test_{fold}.csv")
    expected_test = samples[fold]
    if test_specimens != [expected_test]:
        raise HestAdapterDataError(
            f"OFFICIAL_TEST_SPLIT:{task}:{fold}:{test_specimens}:{expected_test}"
        )
    expected_training = set(samples).difference({expected_test})
    if (
        len(training_specimens) != len(expected_training)
        or set(training_specimens) != expected_training
        or expected_test in training_specimens
    ):
        raise HestAdapterDataError(
            f"OFFICIAL_TRAIN_SPLIT:{task}:{fold}:{training_specimens}:"
            f"expected={sorted(expected_training)}"
        )
    supports = _embedding_map(embedding_manifest, support_encoder)
    patch_stores = load_patch_manifest(patch_manifest)

    def load_one(specimen: str, *, is_test: bool) -> SectionMetadata:
        key = (task, specimen)
        if key not in supports or key not in patch_stores:
            raise HestAdapterDataError(f"SECTION_MANIFEST_KEY:{task}:{specimen}")
        _, barcodes, _ = frozen._read_embedding(supports[key])
        if is_test and not include_test_rna:
            raw = None
            coords = _coordinates_without_rna(
                dataset_root=dataset_root,
                task=task,
                specimen=specimen,
                barcodes=barcodes,
            )
        else:
            raw, _, coords = frozen._matched_truth(
                dataset_root, task, specimen, barcodes, genes
            )
        mask = (
            None
            if is_test
            else validation_mask(
                model=name, task=task, specimen=specimen, barcodes=barcodes
            )
        )
        target = None
        if raw is not None and name == "TCGN":
            target = tcgn_expression_transform(raw)
        return SectionMetadata(
            model=name,
            task=task,
            specimen=specimen,
            fold=fold,
            is_test=is_test,
            barcodes=barcodes,
            genes=genes,
            spatial_coords=coords,
            validation_mask=mask,
            raw_counts=raw,
            target=target,
            patch_store=patch_stores[key],
        )

    training = tuple(load_one(specimen, is_test=False) for specimen in training_specimens)
    scaler = None
    if name.startswith("EGN"):
        scaler = fit_egn_expression_scaler(training)
        training = tuple(
            replace(section, target=scaler.transform(section.raw_counts))
            for section in training
        )
    held_out = load_one(expected_test, is_test=True) if include_test else None
    if held_out is not None and held_out.raw_counts is not None and scaler is not None:
        held_out = replace(held_out, target=scaler.transform(held_out.raw_counts))
    return FoldData(name, task, fold, training, held_out, scaler)


def read_aligned_raw_patches(metadata: SectionMetadata) -> torch.Tensor:
    store = metadata.patch_store
    if not store.path.is_file():
        raise HestAdapterDataError(f"PATCH_STORE_MISSING:{store.path}")
    with h5py.File(store.path, "r") as handle:
        if store.image_key not in handle or store.barcode_key not in handle:
            raise HestAdapterDataError(
                f"PATCH_STORE_SCHEMA:{store.path}:{store.image_key}:{store.barcode_key}"
            )
        stored_barcodes = _decode(handle[store.barcode_key][...])
        images = np.asarray(handle[store.image_key][...])
    if len(set(stored_barcodes.tolist())) != len(stored_barcodes):
        raise HestAdapterDataError(f"PATCH_BARCODE_DUPLICATE:{store.path}")
    if images.shape[0] != len(stored_barcodes):
        raise HestAdapterDataError(
            f"PATCH_IMAGE_ROWS:{images.shape[0]}:{len(stored_barcodes)}"
        )
    lookup = {barcode: index for index, barcode in enumerate(stored_barcodes)}
    if any(barcode not in lookup for barcode in metadata.barcodes):
        raise HestAdapterDataError(
            f"PATCH_BARCODE_JOIN:{metadata.task}:{metadata.specimen}"
        )
    order = np.asarray([lookup[value] for value in metadata.barcodes], dtype=np.int64)
    images = images[order]
    if store.layout == "NHWC":
        if images.ndim != 4 or images.shape[-1] != 3:
            raise HestAdapterDataError(f"PATCH_NHWC_SHAPE:{images.shape}")
        images = np.transpose(images, (0, 3, 1, 2))
    elif images.ndim != 4 or images.shape[1] != 3:
        raise HestAdapterDataError(f"PATCH_NCHW_SHAPE:{images.shape}")
    if not np.issubdtype(images.dtype, np.number) or not np.isfinite(images).all():
        raise HestAdapterDataError(f"PATCH_IMAGE_CONTENT:{images.dtype}")
    if images.min(initial=0) < 0 or images.max(initial=0) > 255:
        raise HestAdapterDataError(
            f"PATCH_VALUE_RANGE:{images.min()}:{images.max()}"
        )
    contiguous = np.ascontiguousarray(images)
    if np.issubdtype(contiguous.dtype, np.integer):
        return torch.as_tensor(contiguous, dtype=torch.uint8)
    return torch.as_tensor(contiguous, dtype=torch.float32)


def resize_patches(patches: torch.Tensor, *, pixels: int) -> torch.Tensor:
    if patches.ndim != 4 or patches.shape[1] != 3:
        raise HestAdapterDataError(f"PATCH_TENSOR_SHAPE:{tuple(patches.shape)}")
    if patches.shape[-2:] == (pixels, pixels):
        return patches
    return F.interpolate(
        patches,
        size=(pixels, pixels),
        mode="bilinear",
        align_corners=False,
    )


def normalize_model_patches(
    model: str, patches: torch.Tensor, *, training: bool = False
) -> torch.Tensor:
    """Apply the publication preprocessing in its literal operation order."""

    name = canonical_model_name(model)
    config = MODEL_CONFIGS[name]
    x = patches / 255.0
    if name == "TCGN":
        try:
            from torchvision import transforms
        except Exception as error:  # pinned gate must expose, not repair, import failure
            raise HestAdapterDataError(
                f"TCGN_TORCHVISION_PREPROCESS_IMPORT:{error}"
            ) from error
        # Literal upstream basic_transform order and antialias setting.
        x = transforms.Resize(
            (config.input_pixels, config.input_pixels), antialias=True
        )(x)
    else:
        x = resize_patches(x, pixels=config.input_pixels)
    if name.startswith("EGN"):
        mean = torch.tensor([0.5476, 0.5218, 0.6881], dtype=x.dtype, device=x.device)
        std = torch.tensor([0.2461, 0.2101, 0.1649], dtype=x.dtype, device=x.device)
    else:
        # timm.data.constants.IMAGENET_DEFAULT_MEAN/STD used upstream.
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=x.dtype, device=x.device)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=x.dtype, device=x.device)
    x = (x - mean[None, :, None, None]) / std[None, :, None, None]
    if name == "TCGN" and training:
        augmentation = transforms.Compose(
            [
                transforms.RandomRotation(180),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
            ]
        )
        # Upstream calls the transform from Dataset.__getitem__, so each image
        # samples its own angle and flip decisions rather than sharing one set
        # of random parameters across the tensor batch.
        x = torch.stack([augmentation(image) for image in x], dim=0)
    return x


def normalize_exemplar_encoder_patches(patches: torch.Tensor) -> torch.Tensor:
    """Published EGN build_exemplar.py preprocessing."""

    x = resize_patches(patches / 255.0, pixels=256)
    return (x - 0.5) / 0.5
