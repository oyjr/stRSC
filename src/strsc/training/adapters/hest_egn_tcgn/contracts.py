"""Frozen upstream and training contracts for the three published models."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional


class ModelContractError(RuntimeError):
    """Raised instead of silently changing a published model."""


@dataclass(frozen=True)
class ModelConfig:
    name: str
    repository: str
    revision: str
    source_subdir: str
    input_pixels: int
    physical_support_um: float
    expression_transform: str
    epochs: int
    batch_size: int
    ordinary_shuffle: bool
    optimizer: str
    learning_rate: float
    weight_decay: float
    exemplar_encoder: Optional[str]
    exemplar_dimension: Optional[int]
    exemplar_distance_p: Optional[int]
    exemplars_per_query: Optional[int]
    correlation_auxiliary_weight: float
    full_section_graph: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


EGN_REVISION = "bae8f0574d98124d723af259116c4375ae037b24"
TCGN_REVISION = "7faf955b3fbf3bfbdb9b5c83eefa70afcfb85202"

# TCGN's source sets ``epoch=80`` and executes ``range(1, epoch)``.  Seventy-nine
# complete epochs is therefore the literal published-code budget.
MODEL_CONFIGS: dict[str, ModelConfig] = {
    "EGNv1": ModelConfig(
        name="EGNv1",
        repository="EGN",
        revision=EGN_REVISION,
        source_subdir="v1",
        input_pixels=256,
        physical_support_um=112.0,
        expression_transform="log10_1p_then_train_only_per_gene_minmax",
        epochs=50,
        batch_size=32,
        ordinary_shuffle=False,
        optimizer="AdamW(betas=(0.9,0.999))+CosineAnnealingWarmupRestarts",
        learning_rate=5e-4,
        weight_decay=1e-4,
        exemplar_encoder="torchvision_imagenet_resnet50",
        exemplar_dimension=2048,
        exemplar_distance_p=2,
        exemplars_per_query=6,
        correlation_auxiliary_weight=0.5,
        full_section_graph=False,
    ),
    "EGNv2": ModelConfig(
        name="EGNv2",
        repository="EGN",
        revision=EGN_REVISION,
        source_subdir="v2",
        input_pixels=256,
        physical_support_um=112.0,
        expression_transform="log10_1p_then_train_only_per_gene_minmax",
        epochs=300,
        batch_size=1,
        ordinary_shuffle=False,
        optimizer="AdamW(betas=(0.9,0.999))",
        learning_rate=5e-4,
        weight_decay=1e-4,
        exemplar_encoder="torchvision_imagenet_resnet18",
        exemplar_dimension=512,
        exemplar_distance_p=1,
        exemplars_per_query=6,
        correlation_auxiliary_weight=0.5,
        full_section_graph=True,
    ),
    "TCGN": ModelConfig(
        name="TCGN",
        repository="TCGN",
        revision=TCGN_REVISION,
        source_subdir=".",
        input_pixels=224,
        physical_support_um=112.0,
        expression_transform="log10_1p_library_size_10000_on_ordered_50_gene_panel",
        epochs=79,
        batch_size=32,
        ordinary_shuffle=True,
        optimizer="Adam(betas=(0.9,0.999),eps=1e-8,amsgrad=False)",
        learning_rate=1e-5,
        weight_decay=0.0,
        exemplar_encoder=None,
        exemplar_dimension=None,
        exemplar_distance_p=None,
        exemplars_per_query=None,
        correlation_auxiliary_weight=0.0,
        full_section_graph=False,
    ),
}


def canonical_model_name(value: str) -> str:
    aliases = {
        "egnv1": "EGNv1",
        "egn1": "EGNv1",
        "egn": "EGNv1",
        "egnv2": "EGNv2",
        "egn2": "EGNv2",
        "eggn": "EGNv2",
        "tcgn": "TCGN",
    }
    key = str(value).strip().lower()
    if key not in aliases:
        raise ModelContractError(f"MODEL:{value}")
    return aliases[key]
