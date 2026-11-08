"""Published EGN exemplar encoders and outer-training-only retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from scipy.spatial import cKDTree

from .contracts import MODEL_CONFIGS, canonical_model_name
from .hest_data import SectionMetadata, normalize_exemplar_encoder_patches


class ExemplarContractError(RuntimeError):
    pass


@dataclass(frozen=True)
class ExemplarBank:
    model: str
    features: torch.Tensor
    targets: torch.Tensor
    specimen_ids: np.ndarray
    barcodes: np.ndarray
    spatial_coords: np.ndarray | None = None

    def __post_init__(self) -> None:
        expected = MODEL_CONFIGS[self.model].exemplar_dimension
        if self.features.ndim != 2 or self.features.shape[1] != expected:
            raise ExemplarContractError(
                f"BANK_FEATURE_SHAPE:{tuple(self.features.shape)}:{expected}"
            )
        if self.targets.shape != (len(self.features), 50):
            raise ExemplarContractError(
                f"BANK_TARGET_SHAPE:{tuple(self.targets.shape)}:{len(self.features)}"
            )
        if len(self.specimen_ids) != len(self.features) or len(self.barcodes) != len(
            self.features
        ):
            raise ExemplarContractError("BANK_METADATA_ROWS")
        if self.spatial_coords is not None and np.asarray(self.spatial_coords).shape != (
            len(self.features),
            2,
        ):
            raise ExemplarContractError(
                f"BANK_SPATIAL_COORDINATES:{np.asarray(self.spatial_coords).shape}"
            )


@dataclass(frozen=True)
class RetrievedExemplars:
    features: torch.Tensor
    targets: torch.Tensor
    source_specimens: np.ndarray
    source_barcodes: np.ndarray


def instantiate_published_encoder(model: str, torchvision_module) -> torch.nn.Module:
    name = canonical_model_name(model)
    if name not in {"EGNv1", "EGNv2"}:
        raise ExemplarContractError(f"MODEL_HAS_NO_EXEMPLAR_ENCODER:{name}")
    constructor = (
        torchvision_module.models.resnet50
        if name == "EGNv1"
        else torchvision_module.models.resnet18
    )
    # The pinned torchvision 0.10 publication path uses pretrained=True.
    encoder = constructor(pretrained=True)
    feature_dimension = encoder.fc.in_features
    expected = MODEL_CONFIGS[name].exemplar_dimension
    if feature_dimension != expected:
        raise ExemplarContractError(
            f"ENCODER_FEATURE_DIMENSION:{feature_dimension}:{expected}"
        )
    result = torch.nn.Sequential(*list(encoder.children())[:-1])
    for parameter in result.parameters():
        parameter.requires_grad = False
    result.eval()
    return result


@torch.no_grad()
def extract_published_features(
    encoder: torch.nn.Module,
    raw_patches: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int = 64,
) -> torch.Tensor:
    encoder = encoder.to(device)
    chunks: list[torch.Tensor] = []
    for start in range(0, len(raw_patches), batch_size):
        batch = normalize_exemplar_encoder_patches(
            raw_patches[start : start + batch_size].to(device)
        )
        chunks.append(encoder(batch).flatten(1).detach().cpu())
    if not chunks:
        raise ExemplarContractError("NO_PATCHES_FOR_FEATURE_EXTRACTION")
    result = torch.cat(chunks, dim=0).contiguous()
    if not torch.isfinite(result).all():
        raise ExemplarContractError("ENCODER_FEATURE_NONFINITE")
    return result


def build_training_bank(
    model: str,
    sections: Sequence[SectionMetadata],
    feature_blocks: Sequence[torch.Tensor],
) -> ExemplarBank:
    name = canonical_model_name(model)
    if name not in {"EGNv1", "EGNv2"}:
        raise ExemplarContractError(f"MODEL_HAS_NO_EXEMPLAR_BANK:{name}")
    if len(sections) != len(feature_blocks) or not sections:
        raise ExemplarContractError(
            f"BANK_SECTION_FEATURE_COUNT:{len(sections)}:{len(feature_blocks)}"
        )
    features: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    specimen_ids: list[str] = []
    barcodes: list[str] = []
    spatial_coords: list[np.ndarray] = []
    for section, feature in zip(sections, feature_blocks):
        if section.is_test or section.validation_mask is None or section.target is None:
            raise ExemplarContractError(
                f"BANK_REQUIRES_OUTER_TRAIN_RNA:{section.specimen}"
            )
        if len(feature) != len(section.barcodes):
            raise ExemplarContractError(
                f"BANK_FEATURE_ALIGNMENT:{section.specimen}:{len(feature)}:"
                f"{len(section.barcodes)}"
            )
        mask = ~section.validation_mask
        features.append(feature[torch.as_tensor(mask)])
        targets.append(torch.as_tensor(section.target[mask], dtype=torch.float32))
        specimen_ids.extend([section.specimen] * int(mask.sum()))
        barcodes.extend(section.barcodes[mask].astype(str).tolist())
        spatial_coords.append(np.asarray(section.spatial_coords[mask], dtype=np.float64))
    return ExemplarBank(
        model=name,
        features=torch.cat(features, dim=0).contiguous(),
        targets=torch.cat(targets, dim=0).contiguous(),
        specimen_ids=np.asarray(specimen_ids, dtype=object),
        barcodes=np.asarray(barcodes, dtype=object),
        spatial_coords=np.concatenate(spatial_coords, axis=0),
    )


def retrieve_exemplars(
    query_features: torch.Tensor,
    *,
    query_specimen: str,
    bank: ExemplarBank,
    query_barcodes: np.ndarray | None = None,
    query_coords: np.ndarray | None = None,
    allow_single_reference_fallback: bool = False,
) -> RetrievedExemplars:
    """Retrieve the exact p-norm nearest six while excluding the query slide."""

    config = MODEL_CONFIGS[bank.model]
    p = int(config.exemplar_distance_p)
    k = int(config.exemplars_per_query)
    eligible = bank.specimen_ids != str(query_specimen)
    eligible_rows = np.flatnonzero(eligible)
    if len(eligible_rows) < k:
        unique_specimens = np.unique(bank.specimen_ids.astype(str))
        single_reference = (
            allow_single_reference_fallback
            and len(unique_specimens) == 1
            and unique_specimens[0] == str(query_specimen)
        )
        if not single_reference:
            raise ExemplarContractError(
                f"INSUFFICIENT_OTHER_SPECIMEN_EXEMPLARS:{query_specimen}:"
                f"{len(eligible_rows)}:{k}"
            )
        if query_barcodes is None or query_coords is None or bank.spatial_coords is None:
            raise ExemplarContractError("SINGLE_REFERENCE_FALLBACK_METADATA_MISSING")
        barcodes = np.asarray(query_barcodes, dtype=str)
        coords = np.asarray(query_coords, dtype=np.float64)
        if len(barcodes) != len(query_features) or coords.shape != (
            len(query_features),
            2,
        ):
            raise ExemplarContractError(
                f"SINGLE_REFERENCE_QUERY_METADATA:{len(barcodes)}:{coords.shape}:"
                f"{len(query_features)}"
            )
        bank_coords = np.asarray(bank.spatial_coords, dtype=np.float64)
        bank_tree = cKDTree(bank_coords)
        nearest_distances, _ = bank_tree.query(bank_coords, k=2)
        positive = nearest_distances[:, 1][nearest_distances[:, 1] > 0]
        if not len(positive):
            raise ExemplarContractError("SINGLE_REFERENCE_SPACING_NOT_ESTIMABLE")
        radius = 1.5 * float(np.median(positive))
        eligible_rows = np.arange(len(bank.features), dtype=np.int64)
        distances = torch.cdist(
            query_features.float(), bank.features.float(), p=p
        ).detach().cpu().numpy()
        for query_row, nearby in enumerate(bank_tree.query_ball_point(coords, radius)):
            distances[query_row, np.asarray(nearby, dtype=np.int64)] = np.inf
        bank_barcode_rows: dict[str, list[int]] = {}
        for row, barcode in enumerate(bank.barcodes.astype(str)):
            bank_barcode_rows.setdefault(barcode, []).append(row)
        for query_row, barcode in enumerate(barcodes):
            exact_rows = bank_barcode_rows.get(barcode)
            if exact_rows:
                distances[query_row, exact_rows] = np.inf
        counts = np.isfinite(distances).sum(axis=1)
        if np.any(counts < k):
            raise ExemplarContractError(
                f"INSUFFICIENT_SINGLE_REFERENCE_EXEMPLARS:{query_specimen}:"
                f"{int(counts.min())}:{k}"
            )
        nearest = torch.as_tensor(
            np.argsort(distances, axis=1, kind="stable")[:, :k],
            dtype=torch.long,
        )
        global_rows = torch.as_tensor(eligible_rows, dtype=torch.long)[nearest]
        return RetrievedExemplars(
            features=bank.features[global_rows],
            targets=bank.targets[global_rows],
            source_specimens=bank.specimen_ids[global_rows.numpy()],
            source_barcodes=bank.barcodes[global_rows.numpy()],
        )
    candidates = bank.features[torch.as_tensor(eligible_rows)]
    distances = torch.cdist(query_features.float(), candidates.float(), p=p)
    # NumPy's stable sort is used because the pinned torch 1.9 API differs
    # across patch releases; ties retain the frozen candidate order.
    nearest = torch.as_tensor(
        np.argsort(distances.detach().cpu().numpy(), axis=1, kind="stable")[:, :k],
        dtype=torch.long,
    )
    global_rows = torch.as_tensor(eligible_rows, dtype=torch.long)[nearest]
    return RetrievedExemplars(
        features=bank.features[global_rows],
        targets=bank.targets[global_rows],
        source_specimens=bank.specimen_ids[global_rows.numpy()],
        source_barcodes=bank.barcodes[global_rows.numpy()],
    )
