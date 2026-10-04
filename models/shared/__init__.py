"""Shared learned representation research contracts for RealSaS."""
from .surface_evidence_input_v1 import (
    SCHEMA as SURFACE_EVIDENCE_INPUT_SCHEMA,
    SurfaceEvidenceEncoderInputV1,
    assert_no_downstream_authority_fields_v1,
    build_surface_evidence_encoder_input_v1,
)

__all__ = [
    "SURFACE_EVIDENCE_INPUT_SCHEMA",
    "SurfaceEvidenceEncoderInputV1",
    "assert_no_downstream_authority_fields_v1",
    "build_surface_evidence_encoder_input_v1",
]
