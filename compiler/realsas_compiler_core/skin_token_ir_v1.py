from __future__ import annotations

"""Discrete skin-code proposal seam for MIRA V6 / AXIS V6.

Skin tokens are learned representation, not canonical Skin authority. The Compiler
checks only exact bindings and token-domain well-formedness. It must not decode,
search, substitute or repair token values; the admitted MIRA codec owns decoding.
"""

from dataclasses import asdict, dataclass, field, replace
from typing import Any

from .hashing import content_sha256
from .types import QualificationError

Json = dict[str, Any]


@dataclass(frozen=True)
class JointSkinTokenRowIR:
    proposal_joint_id: str
    token_ids: tuple[int, ...]
    metadata: Json = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class SkinTokenProposalIR:
    rows: tuple[JointSkinTokenRowIR, ...]
    rig_program_proposal_hash: str
    carrier_evidence_hash: str
    carrier_topology_hash: str
    carrier_geometry_hash: str
    codec_architecture_id: str
    codec_checkpoint_hash: str
    codebook_size: int
    tokens_per_joint: int
    model_provenance: str
    proposal_lineage_hash: str = ""
    schema_version: str = "RealSaS.SkinTokenProposalIR.v1"
    metadata: Json = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def skin_token_proposal_lineage_hash_v1(value: SkinTokenProposalIR) -> str:
    payload = value.to_dict()
    payload.pop("proposal_lineage_hash", None)
    return content_sha256(payload)


def seal_skin_token_proposal_v1(value: SkinTokenProposalIR) -> SkinTokenProposalIR:
    return replace(value, proposal_lineage_hash=skin_token_proposal_lineage_hash_v1(value))


def validate_skin_token_proposal_v1(
    value: SkinTokenProposalIR,
    *,
    expected_proposal_joint_ids: tuple[str, ...],
    expected_rig_program_proposal_hash: str,
) -> None:
    if value.schema_version != "RealSaS.SkinTokenProposalIR.v1":
        raise QualificationError("SKIN_TOKEN_SCHEMA_DRIFT")
    if value.proposal_lineage_hash != skin_token_proposal_lineage_hash_v1(value):
        raise QualificationError("SKIN_TOKEN_LINEAGE_DRIFT")
    if value.rig_program_proposal_hash != expected_rig_program_proposal_hash:
        raise QualificationError("SKIN_TOKEN_RIG_PROGRAM_BINDING_DRIFT")
    if value.codebook_size < 2 or value.tokens_per_joint < 1:
        raise QualificationError("SKIN_TOKEN_CODEC_DOMAIN_INVALID")
    if not value.codec_architecture_id or len(value.codec_checkpoint_hash) < 8:
        raise QualificationError("SKIN_TOKEN_CODEC_IDENTITY_REQUIRED")
    if not value.model_provenance:
        raise QualificationError("SKIN_TOKEN_MODEL_PROVENANCE_REQUIRED")
    if any(not x for x in (value.carrier_evidence_hash, value.carrier_topology_hash, value.carrier_geometry_hash)):
        raise QualificationError("SKIN_TOKEN_CARRIER_BINDING_REQUIRED")

    expected = tuple(map(str, expected_proposal_joint_ids))
    row_ids = tuple(str(row.proposal_joint_id) for row in value.rows)
    if len(row_ids) != len(set(row_ids)) or set(row_ids) != set(expected):
        raise QualificationError("SKIN_TOKEN_JOINT_COVERAGE_DRIFT")
    for row in value.rows:
        if len(row.token_ids) != value.tokens_per_joint:
            raise QualificationError("SKIN_TOKEN_COUNT_PER_JOINT_DRIFT")
        for token in row.token_ids:
            token = int(token)
            if token < 0 or token >= value.codebook_size:
                raise QualificationError("SKIN_TOKEN_ID_OUT_OF_DOMAIN")


__all__ = [
    "JointSkinTokenRowIR",
    "SkinTokenProposalIR",
    "seal_skin_token_proposal_v1",
    "skin_token_proposal_lineage_hash_v1",
    "validate_skin_token_proposal_v1",
]
