"""Prospective R6 candidate model stack.

Learned modules emit proposal/evidence only. Compiler owns canonical qualification.
"""

from .candidate_config_v1 import (
    ARACHNE_V1,
    CANDIDATE_STACK_HASH_V1,
    GEPPETTO_V1,
    SKIN_FIELD_CODEC_V1,
    ArachneCandidateConfigV1,
    GeppettoCandidateConfigV1,
    SkinFieldCodecConfigV1,
)

__all__ = [
    "ARACHNE_V1",
    "CANDIDATE_STACK_HASH_V1",
    "GEPPETTO_V1",
    "SKIN_FIELD_CODEC_V1",
    "ArachneCandidateConfigV1",
    "GeppettoCandidateConfigV1",
    "SkinFieldCodecConfigV1",
]
