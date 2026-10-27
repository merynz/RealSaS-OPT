from __future__ import annotations

"""Fail-closed temporal contradiction gate for a proposed static attachment policy.

The gate does not discover an attachment, semantic part, carrier or motion class.
It consumes temporal relative-transport evidence for one already-proposed exact
surface binding. A finite authored condition set may *contradict* a global static
policy, but cannot authorize static behavior for all future motions. Therefore
all-rigid measured conditions yield only `NO_CONTRADICTION_ON_MEASURED_CONDITIONS`.
"""

from dataclasses import dataclass
from typing import Iterable, Literal
import re

from .rigid_attachment_surface_anchor_v1 import AttachmentSurfaceBindingV1
from .types import QualificationError

SCHEMA = "RealSaS.MotionTemporalStaticPolicyGate.v1"

TemporalStaticDecisionV1 = Literal[
    "STATIC_POLICY_CONTRADICTED",
    "NO_CONTRADICTION_ON_MEASURED_CONDITIONS",
    "ABSTAIN",
]
_ALLOWED_STATUS = {
    "RIGID_RELATIVE_TRANSPORT_SUPPORTED",
    "NONRIGID_RELATIVE_TRANSPORT_SUPPORTED",
    "ABSTAIN",
}


@dataclass(frozen=True)
class TemporalConditionEvidenceRefV1:
    condition_binding_hash: str
    evidence_status: str
    heldout_confirmed: bool
    evidence_face_a: int
    evidence_face_b: int


@dataclass(frozen=True)
class MotionTemporalStaticPolicyGateV1:
    decision: TemporalStaticDecisionV1
    measured_condition_count: int
    rigid_confirmed_count: int
    nonrigid_confirmed_count: int
    abstained_count: int
    binding_held_face: int
    binding_holder_face: int
    product_static_authority_minted: bool
    dynamic_generation_authorized: bool
    schema_version: str = SCHEMA


def evaluate_temporal_static_policy_v1(
    *,
    binding: AttachmentSurfaceBindingV1,
    evidence: Iterable[TemporalConditionEvidenceRefV1],
) -> MotionTemporalStaticPolicyGateV1:
    if not isinstance(binding, AttachmentSurfaceBindingV1):
        raise QualificationError("MOTION_TEMPORAL_STATIC_BINDING_TYPE_INVALID")
    rows = tuple(evidence)
    if not rows:
        raise QualificationError("MOTION_TEMPORAL_STATIC_EVIDENCE_REQUIRED")

    held_face = int(binding.held.face_index)
    holder_face = int(binding.holder.face_index)
    rigid = nonrigid = abstained = 0
    seen_conditions: set[str] = set()
    for row in rows:
        if not isinstance(row, TemporalConditionEvidenceRefV1):
            raise QualificationError("MOTION_TEMPORAL_STATIC_EVIDENCE_TYPE_INVALID")
        if re.fullmatch(r"[0-9a-f]{64}", row.condition_binding_hash) is None:
            raise QualificationError("MOTION_TEMPORAL_STATIC_CONDITION_HASH_INVALID")
        if row.condition_binding_hash in seen_conditions:
            raise QualificationError("MOTION_TEMPORAL_STATIC_CONDITION_DUPLICATE")
        seen_conditions.add(row.condition_binding_hash)
        if row.evidence_status not in _ALLOWED_STATUS:
            raise QualificationError("MOTION_TEMPORAL_STATIC_EVIDENCE_STATUS_INVALID")
        if type(row.heldout_confirmed) is not bool:
            raise QualificationError("MOTION_TEMPORAL_STATIC_HELDOUT_FLAG_INVALID")
        if row.evidence_face_a != held_face or row.evidence_face_b != holder_face:
            raise QualificationError("MOTION_TEMPORAL_STATIC_EVIDENCE_ADDRESS_MISMATCH")

        if not row.heldout_confirmed or row.evidence_status == "ABSTAIN":
            abstained += 1
        elif row.evidence_status == "RIGID_RELATIVE_TRANSPORT_SUPPORTED":
            rigid += 1
        else:
            nonrigid += 1

    if nonrigid:
        decision: TemporalStaticDecisionV1 = "STATIC_POLICY_CONTRADICTED"
    elif abstained:
        decision = "ABSTAIN"
    else:
        decision = "NO_CONTRADICTION_ON_MEASURED_CONDITIONS"

    return MotionTemporalStaticPolicyGateV1(
        decision=decision,
        measured_condition_count=len(rows),
        rigid_confirmed_count=rigid,
        nonrigid_confirmed_count=nonrigid,
        abstained_count=abstained,
        binding_held_face=held_face,
        binding_holder_face=holder_face,
        product_static_authority_minted=False,
        dynamic_generation_authorized=False,
    )
