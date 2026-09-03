"""Proof-gated deployment codecs subordinate to canonical compiler authority."""

from .current_v4_runtime_v2 import (
    RuntimeTexturePayloadV1,
    RuntimeV2ProjectionResult,
    project_current_v4_proof_bakes_to_runtime_v2,
    materialize_current_v4_proof_bakes_runtime_v2,
)

__all__ = [
    "RuntimeTexturePayloadV1",
    "RuntimeV2ProjectionResult",
    "project_current_v4_proof_bakes_to_runtime_v2",
    "materialize_current_v4_proof_bakes_runtime_v2",
]
