from __future__ import annotations

"""Fail-closed authorization of a fixed rigid attachment candidate by Mechanics evidence.

This module deliberately does *not* discover an attachment, owner, held region,
bone or semantic part. A separate compiler stage may propose a canonical
``AttachmentSurfaceBindingV1`` candidate. This gate answers only whether a
held-out-confirmed relative-transport observation supports applying the existing
rigid surface-anchor consequence to that exact addressed surface pair.

Keeping candidate discovery and causal authorization separate prevents a
privileged teacher reference from silently becoming product structure.
"""

from dataclasses import dataclass
from typing import Literal

from .rigid_attachment_surface_anchor_v1 import AttachmentSurfaceBindingV1
from .types import QualificationError

SCHEMA = "RealSaS.MechanicsAttachmentEvidenceGate.v1"

MechanicsGateDecision = Literal["AUTHORIZED", "REJECTED", "ABSTAIN"]
_ALLOWED_STATUSES = {
    "RIGID_RELATIVE_TRANSPORT_SUPPORTED",
    "NONRIGID_RELATIVE_TRANSPORT_SUPPORTED",
    "ABSTAIN",
}


@dataclass(frozen=True)
class MechanicsAttachmentEvidenceGateV1:
    decision: MechanicsGateDecision
    evidence_status: str
    heldout_confirmed: bool
    evidence_face_a: int
    evidence_face_b: int
    binding_held_face: int
    binding_holder_face: int
    schema_version: str = SCHEMA


def authorize_rigid_surface_binding_v1(
    *,
    binding: AttachmentSurfaceBindingV1,
    evidence_status: str,
    heldout_confirmed: bool,
    evidence_face_a: int,
    evidence_face_b: int,
) -> MechanicsAttachmentEvidenceGateV1:
    """Authorize/reject one already-proposed binding from typed Mechanics evidence.

    Evidence face order is explicit: ``face_a`` must be the candidate held face
    and ``face_b`` the candidate holder face. A mismatch is a qualification
    error rather than an automatic remap because an unordered pair would erase
    the consequence direction.
    """
    if not isinstance(binding, AttachmentSurfaceBindingV1):
        raise QualificationError("MECHANICS_ATTACHMENT_BINDING_TYPE_INVALID")
    if evidence_status not in _ALLOWED_STATUSES:
        raise QualificationError("MECHANICS_ATTACHMENT_EVIDENCE_STATUS_INVALID")
    if type(heldout_confirmed) is not bool:
        raise QualificationError("MECHANICS_ATTACHMENT_HELDOUT_FLAG_INVALID")
    if type(evidence_face_a) is not int or type(evidence_face_b) is not int:
        raise QualificationError("MECHANICS_ATTACHMENT_EVIDENCE_FACE_INVALID")

    held_face = int(binding.held.face_index)
    holder_face = int(binding.holder.face_index)
    if evidence_face_a != held_face or evidence_face_b != holder_face:
        raise QualificationError("MECHANICS_ATTACHMENT_EVIDENCE_ADDRESS_MISMATCH")

    if not heldout_confirmed or evidence_status == "ABSTAIN":
        decision: MechanicsGateDecision = "ABSTAIN"
    elif evidence_status == "RIGID_RELATIVE_TRANSPORT_SUPPORTED":
        decision = "AUTHORIZED"
    else:
        decision = "REJECTED"

    return MechanicsAttachmentEvidenceGateV1(
        decision=decision,
        evidence_status=evidence_status,
        heldout_confirmed=heldout_confirmed,
        evidence_face_a=evidence_face_a,
        evidence_face_b=evidence_face_b,
        binding_held_face=held_face,
        binding_holder_face=holder_face,
    )
