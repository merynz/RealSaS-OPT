from __future__ import annotations

"""Causal evidence authorization for a discovered rigid Mechanics candidate.

The discovery layer is allowed to propose a directed canonical surface pair. It
is not allowed to turn rigid-looking skin into attachment or constraint truth.
This bridge therefore accepts only a fully addressed
``CANDIDATE_REQUIRES_CAUSAL_AUTHORIZATION`` row and delegates the decision to the
existing held-out Mechanics evidence gate.

No legacy attachment index, subject identity, semantic owner label, or bone name
is invented at this seam.
"""

from .mechanics_attachment_evidence_gate_v1 import (
    MechanicsAttachmentEvidenceGateV1,
    authorize_rigid_surface_pair_v1,
)
from .mechanics_rigid_candidate_discovery_v1 import MechanicsRigidCandidateRowV1
from .types import QualificationError


def authorize_discovered_rigid_candidate_v1(
    *,
    candidate: MechanicsRigidCandidateRowV1,
    evidence_status: str,
    heldout_confirmed: bool,
    evidence_face_a: int,
    evidence_face_b: int,
) -> MechanicsAttachmentEvidenceGateV1:
    if not isinstance(candidate, MechanicsRigidCandidateRowV1):
        raise QualificationError("MECHANICS_RIGID_CANDIDATE_TYPE_INVALID")
    if candidate.status != "CANDIDATE_REQUIRES_CAUSAL_AUTHORIZATION":
        raise QualificationError("MECHANICS_RIGID_CANDIDATE_NOT_AUTHORIZABLE")
    if candidate.held is None or candidate.holder is None:
        raise QualificationError("MECHANICS_RIGID_CANDIDATE_ADDRESS_MISSING")
    if not candidate.candidate_hash:
        raise QualificationError("MECHANICS_RIGID_CANDIDATE_HASH_MISSING")

    return authorize_rigid_surface_pair_v1(
        held_face=int(candidate.held.face_index),
        holder_face=int(candidate.holder.face_index),
        evidence_status=evidence_status,
        heldout_confirmed=heldout_confirmed,
        evidence_face_a=evidence_face_a,
        evidence_face_b=evidence_face_b,
    )


__all__ = ["authorize_discovered_rigid_candidate_v1"]
