"""Canonical ATLAS model namespace.

ATLAS = Articulation Topology & Locus Autoregressive Synthesis.

This namespace is alias-first: it preserves legacy Geppetto checkpoint/tensor
semantics while new code migrates to canonical RealSaS naming.
"""
from models.geppetto.reference_strength_v1.geppetto_reference_strength_candidate_v1 import (
    GeppettoReferenceStrengthCandidateV1 as ATLASReferenceStrengthCandidateV1,
    GeppettoReferenceStrengthConfigV1 as ATLASReferenceStrengthConfigV1,
    GeppettoReferenceStrengthRawOutputV1 as ATLASReferenceStrengthRawOutputV1,
    DirectRiggingSurfaceEncoderV1 as ATLASSurfaceEncoderV1,
)

__all__ = [
    "ATLASReferenceStrengthCandidateV1",
    "ATLASReferenceStrengthConfigV1",
    "ATLASReferenceStrengthRawOutputV1",
    "ATLASSurfaceEncoderV1",
]
