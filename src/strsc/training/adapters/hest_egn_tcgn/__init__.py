"""HEST training adapters for EGNv1, EGNv2 and TCGN."""

from .contracts import MODEL_CONFIGS, ModelConfig, canonical_model_name
from .overlay import (
    SpotSupervision,
    build_supervision,
    correlation_loss,
    edge_pair_batch,
    full_section_loss,
    minibatch_loss,
    preserve_model_buffers,
    run_rho_zero_noop_gate,
)

__all__ = [
    "MODEL_CONFIGS",
    "ModelConfig",
    "SpotSupervision",
    "build_supervision",
    "canonical_model_name",
    "correlation_loss",
    "edge_pair_batch",
    "full_section_loss",
    "minibatch_loss",
    "preserve_model_buffers",
    "run_rho_zero_noop_gate",
]
