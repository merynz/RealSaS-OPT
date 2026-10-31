from __future__ import annotations

"""Typed Studio control-program patch boundary for CanonicalRigIR.v2.

Studio authoring is explicit user intent, not learned AXIS evidence and not
canonical authority.  This module converts that intent into a typed proposal,
qualifies it against an immutable base CanonicalRig, and materializes a new
CanonicalRig control program without changing deform structure.

The first shipping slice deliberately supports only adding enabled IK constraints
that target an already-existing canonical animator control.  It never synthesizes
controls, joints, frames, or deform topology.
"""

from dataclasses import asdict, dataclass, field, replace
import math
from typing import Any, Mapping

from .canonical_rig_ir_v2 import (
    CanonicalRigChainIRV2,
    CanonicalRigConstraintIRV2,
    CanonicalRigIRV2,
    _hash_control_program,
    _hash_deform_structure,
    canonical_rig_lineage_hash_v2,
)
from .hashing import content_sha256
from .rig_constraint_schema_v1 import validate_rig_constraint_operator_v1
from .types import QualificationError

Json = dict[str, Any]


@dataclass(frozen=True)
class StudioIKConstraintProposalIRV1:
    authoring_constraint_id: str
    start_canonical_joint_id: str
    end_canonical_joint_id: str
    target_canonical_control_id: str
    mix: float = 1.0
    max_stretch: float = 1.0
    metadata: Json = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class StudioRigControlPatchProposalIRV1:
    base_rig_lineage_hash: str
    base_deform_structure_hash: str
    authoring_payload_sha256: str
    ik_constraints: tuple[StudioIKConstraintProposalIRV1, ...]
    proposal_lineage_hash: str = ""
    schema_version: str = "RealSaS.StudioRigControlPatchProposalIR.v1"
    metadata: Json = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class QualifiedStudioRigControlPatchIRV1:
    proposal: StudioRigControlPatchProposalIRV1
    resolved_joint_chains: tuple[tuple[str, ...], ...]
    qualification_report: Json
    qualification_hash: str
    schema_version: str = "RealSaS.QualifiedStudioRigControlPatchIR.v1"
    metadata: Json = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def studio_rig_control_patch_proposal_hash_v1(value: StudioRigControlPatchProposalIRV1) -> str:
    payload = value.to_dict()
    payload.pop("proposal_lineage_hash", None)
    return content_sha256(payload)


def qualified_studio_rig_control_patch_hash_v1(value: QualifiedStudioRigControlPatchIRV1) -> str:
    payload = value.to_dict()
    payload.pop("qualification_hash", None)
    return content_sha256(payload)


def seal_studio_rig_control_patch_proposal_v1(
    value: StudioRigControlPatchProposalIRV1,
) -> StudioRigControlPatchProposalIRV1:
    return replace(value, proposal_lineage_hash=studio_rig_control_patch_proposal_hash_v1(value))


def _safe_id(value: Any, code: str) -> str:
    text = str(value or "").strip()
    if not text or len(text) > 256:
        raise QualificationError(code)
    allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.:/-")
    if any(ch not in allowed for ch in text):
        raise QualificationError(code)
    return text


def _finite(value: Any, code: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise QualificationError(code) from exc
    if not math.isfinite(out):
        raise QualificationError(code)
    return out


def _nonempty_rows(payload: Mapping[str, Any], field: str) -> bool:
    rows = payload.get(field)
    return isinstance(rows, list) and len(rows) > 0


def studio_rig_control_patch_from_authoring_v1(
    authoring: Mapping[str, Any],
    *,
    base_rig: CanonicalRigIRV2,
    authoring_payload_sha256: str,
) -> StudioRigControlPatchProposalIRV1:
    """Convert one saved Studio edit layer into the first IK-only proposal slice."""
    schema = str(authoring.get("schema_version") or authoring.get("schema") or "")
    if schema != "RealSaS.UserPuppetEditLayer.v3":
        raise QualificationError("STUDIO_RIG_PATCH_AUTHORING_SCHEMA_DRIFT")
    if authoring.get("authority_class") != "OPTIONAL_USER_AUTHORING_LAYER":
        raise QualificationError("STUDIO_RIG_PATCH_AUTHORING_AUTHORITY_DRIFT")
    if authoring.get("can_promote_product_directly") is not False:
        raise QualificationError("STUDIO_RIG_PATCH_DIRECT_PROMOTION_FORBIDDEN")
    if authoring.get("requires_compiler_requalification") is not True:
        raise QualificationError("STUDIO_RIG_PATCH_REQUALIFICATION_FLAG_REQUIRED")
    if len(str(authoring_payload_sha256)) != 64:
        raise QualificationError("STUDIO_RIG_PATCH_AUTHORING_HASH_REQUIRED")

    # This slice is intentionally narrow.  Mixing deform, skin, graphics, or
    # motion edits would require a different dependency cut-set.
    forbidden = (
        "rig_pose_edits",
        "mesh_vertex_edits",
        "mesh_topology_edits",
        "weight_edits",
        "binding_edits",
        "rig_topology_edits",
        "animation_edits",
    )
    for field_name in forbidden:
        if _nonempty_rows(authoring, field_name):
            raise QualificationError(f"STUDIO_RIG_PATCH_UNSUPPORTED_EDIT_DOMAIN:{field_name}")
    presentation = authoring.get("presentation_override")
    if isinstance(presentation, Mapping):
        if presentation.get("graphics_evidence_departure") is True:
            raise QualificationError("STUDIO_RIG_PATCH_UNSUPPORTED_EDIT_DOMAIN:presentation_override")
        if presentation.get("hidden_mesh_ids"):
            raise QualificationError("STUDIO_RIG_PATCH_UNSUPPORTED_EDIT_DOMAIN:presentation_override")

    rows = authoring.get("ik_constraints")
    if not isinstance(rows, list) or not rows:
        raise QualificationError("STUDIO_RIG_PATCH_IK_REQUIRED")
    if len(rows) > 256:
        raise QualificationError("STUDIO_RIG_PATCH_IK_LIMIT")

    joint_ids = {j.canonical_joint_id for j in base_rig.joints}
    control_ids = {c.canonical_control_id for c in base_rig.controls}
    seen: set[str] = set()
    constraints: list[StudioIKConstraintProposalIRV1] = []
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise QualificationError("STUDIO_RIG_PATCH_IK_OBJECT_REQUIRED")
        cid = _safe_id(raw.get("constraint_id"), "STUDIO_RIG_PATCH_IK_ID_INVALID")
        if cid in seen:
            raise QualificationError("STUDIO_RIG_PATCH_IK_ID_DUPLICATE")
        seen.add(cid)
        start = _safe_id(raw.get("start_bone_id"), "STUDIO_RIG_PATCH_IK_START_INVALID")
        end = _safe_id(raw.get("end_bone_id"), "STUDIO_RIG_PATCH_IK_END_INVALID")
        target = _safe_id(raw.get("target_control_id"), "STUDIO_RIG_PATCH_IK_TARGET_CONTROL_REQUIRED")
        if start not in joint_ids or end not in joint_ids:
            raise QualificationError("STUDIO_RIG_PATCH_IK_JOINT_UNKNOWN")
        if target not in control_ids:
            raise QualificationError("STUDIO_RIG_PATCH_IK_TARGET_CONTROL_UNKNOWN")
        if raw.get("enabled", True) is not True:
            raise QualificationError("STUDIO_RIG_PATCH_DISABLED_IK_NOT_YET_SUPPORTED")
        mix = _finite(raw.get("mix", 1.0), "STUDIO_RIG_PATCH_IK_MIX_INVALID")
        if not 0.0 <= mix <= 1.0:
            raise QualificationError("STUDIO_RIG_PATCH_IK_MIX_RANGE")
        max_stretch = _finite(raw.get("max_stretch", 1.0), "STUDIO_RIG_PATCH_IK_STRETCH_INVALID")
        if not 1.0 <= max_stretch <= 4.0:
            raise QualificationError("STUDIO_RIG_PATCH_IK_STRETCH_RANGE")
        constraints.append(
            StudioIKConstraintProposalIRV1(
                authoring_constraint_id=cid,
                start_canonical_joint_id=start,
                end_canonical_joint_id=end,
                target_canonical_control_id=target,
                mix=mix,
                max_stretch=max_stretch,
                metadata={
                    "source": "STUDIO_AUTHORING_REVISION",
                    "legacy_target_bone_semantics_used": False,
                    "control_synthesis_requested": False,
                },
            )
        )

    value = StudioRigControlPatchProposalIRV1(
        base_rig_lineage_hash=base_rig.rig_lineage_hash,
        base_deform_structure_hash=base_rig.deform_structure_hash,
        authoring_payload_sha256=str(authoring_payload_sha256),
        ik_constraints=tuple(constraints),
        metadata={
            "authoring_space": "CANONICAL_CONTROL_PROGRAM",
            "deform_structure_edit_allowed": False,
            "control_synthesis_allowed": False,
            "constraint_kind": "IK",
        },
    )
    return seal_studio_rig_control_patch_proposal_v1(value)


def _chain_from_ancestor(
    base_rig: CanonicalRigIRV2,
    start_joint_id: str,
    end_joint_id: str,
) -> tuple[str, ...]:
    parent = {j.canonical_joint_id: j.parent_canonical_joint_id for j in base_rig.joints}
    if start_joint_id not in parent or end_joint_id not in parent:
        raise QualificationError("STUDIO_RIG_PATCH_IK_JOINT_UNKNOWN")
    reverse = [end_joint_id]
    seen = {end_joint_id}
    current = end_joint_id
    while current != start_joint_id:
        next_parent = parent.get(current)
        if next_parent is None:
            raise QualificationError("STUDIO_RIG_PATCH_IK_START_NOT_ANCESTOR")
        if next_parent in seen:
            raise QualificationError("STUDIO_RIG_PATCH_BASE_PARENT_CYCLE")
        seen.add(next_parent)
        reverse.append(next_parent)
        current = next_parent
    path = tuple(reversed(reverse))
    if len(path) < 2:
        raise QualificationError("STUDIO_RIG_PATCH_IK_CHAIN_TOO_SHORT")
    return path


def qualify_studio_rig_control_patch_v1(
    proposal: StudioRigControlPatchProposalIRV1,
    *,
    base_rig: CanonicalRigIRV2,
) -> QualifiedStudioRigControlPatchIRV1:
    if proposal.schema_version != "RealSaS.StudioRigControlPatchProposalIR.v1":
        raise QualificationError("STUDIO_RIG_PATCH_SCHEMA_DRIFT")
    if proposal.proposal_lineage_hash != studio_rig_control_patch_proposal_hash_v1(proposal):
        raise QualificationError("STUDIO_RIG_PATCH_PROPOSAL_HASH_DRIFT")
    if base_rig.rig_lineage_hash != canonical_rig_lineage_hash_v2(base_rig):
        raise QualificationError("STUDIO_RIG_PATCH_BASE_RIG_LINEAGE_DRIFT")
    if base_rig.metadata.get("authority") != "COMPILER_CANONICAL_RIG":
        raise QualificationError("STUDIO_RIG_PATCH_BASE_RIG_AUTHORITY_DRIFT")
    if proposal.base_rig_lineage_hash != base_rig.rig_lineage_hash:
        raise QualificationError("STUDIO_RIG_PATCH_BASE_RIG_BINDING_DRIFT")
    if proposal.base_deform_structure_hash != base_rig.deform_structure_hash:
        raise QualificationError("STUDIO_RIG_PATCH_BASE_DEFORM_BINDING_DRIFT")

    controls = {c.canonical_control_id for c in base_rig.controls}
    resolved: list[tuple[str, ...]] = []
    for row in proposal.ik_constraints:
        if row.target_canonical_control_id not in controls:
            raise QualificationError("STUDIO_RIG_PATCH_IK_TARGET_CONTROL_UNKNOWN")
        path = _chain_from_ancestor(
            base_rig,
            row.start_canonical_joint_id,
            row.end_canonical_joint_id,
        )
        validate_rig_constraint_operator_v1(
            operator_id="IK",
            input_control_count=1,
            output_joint_count=len(path),
            has_chain=True,
            parameters={"p0": row.mix, "p2": row.max_stretch},
        )
        resolved.append(path)

    report = {
        "status": "PASS_STUDIO_RIG_CONTROL_PATCH_VALIDATED",
        "ik_constraint_count": len(proposal.ik_constraints),
        "base_rig_lineage_hash": base_rig.rig_lineage_hash,
        "base_deform_structure_hash": base_rig.deform_structure_hash,
        "deform_structure_mutation_requested": False,
        "control_synthesis_performed": False,
        "joint_synthesis_performed": False,
        "frame_repair_performed": False,
        "constraint_solver_executed": False,
        "compiler_role": "VALIDATE_AND_MATERIALIZE_EXPLICIT_USER_CONTROL_PROGRAM",
    }
    value = QualifiedStudioRigControlPatchIRV1(
        proposal=proposal,
        resolved_joint_chains=tuple(resolved),
        qualification_report=report,
        qualification_hash="",
        metadata={
            "user_authoring_is_proposal_not_authority": True,
            "base_deform_structure_must_be_preserved": True,
            "mechanics_execution_required_downstream": True,
        },
    )
    return replace(value, qualification_hash=qualified_studio_rig_control_patch_hash_v1(value))


def apply_qualified_studio_rig_control_patch_v1(
    base_rig: CanonicalRigIRV2,
    qualified: QualifiedStudioRigControlPatchIRV1,
) -> CanonicalRigIRV2:
    if qualified.qualification_hash != qualified_studio_rig_control_patch_hash_v1(qualified):
        raise QualificationError("STUDIO_RIG_PATCH_QUALIFICATION_HASH_DRIFT")
    report = dict(qualified.qualification_report or {})
    if report.get("status") != "PASS_STUDIO_RIG_CONTROL_PATCH_VALIDATED":
        raise QualificationError("STUDIO_RIG_PATCH_NOT_QUALIFIED")
    proposal = qualified.proposal
    if proposal.base_rig_lineage_hash != base_rig.rig_lineage_hash:
        raise QualificationError("STUDIO_RIG_PATCH_APPLY_BASE_RIG_DRIFT")
    if proposal.base_deform_structure_hash != base_rig.deform_structure_hash:
        raise QualificationError("STUDIO_RIG_PATCH_APPLY_BASE_DEFORM_DRIFT")
    if len(proposal.ik_constraints) != len(qualified.resolved_joint_chains):
        raise QualificationError("STUDIO_RIG_PATCH_RESOLVED_CHAIN_CARDINALITY")

    chains = list(base_rig.chains)
    constraints = list(base_rig.constraints)
    existing_chain_ids = {row.canonical_chain_id for row in chains}
    existing_constraint_ids = {row.canonical_constraint_id for row in constraints}
    for row, path in zip(proposal.ik_constraints, qualified.resolved_joint_chains):
        identity = content_sha256({
            "base_rig_lineage_hash": base_rig.rig_lineage_hash,
            "patch_qualification_hash": qualified.qualification_hash,
            "authoring_constraint_id": row.authoring_constraint_id,
        })[:20]
        chain_id = f"K:studio:{identity}"
        constraint_id = f"Q:studio:{identity}"
        if chain_id in existing_chain_ids or constraint_id in existing_constraint_ids:
            raise QualificationError("STUDIO_RIG_PATCH_CANONICAL_ID_COLLISION")
        existing_chain_ids.add(chain_id)
        existing_constraint_ids.add(constraint_id)
        chains.append(
            CanonicalRigChainIRV2(
                canonical_chain_id=chain_id,
                canonical_joint_ids=tuple(path),
                chain_kind="STUDIO_IK_CHAIN",
                source_proposal_chain_id=f"STUDIO:{row.authoring_constraint_id}:CHAIN",
                metadata={
                    "authority_source": "QUALIFIED_STUDIO_RIG_CONTROL_PATCH",
                    "studio_patch_qualification_hash": qualified.qualification_hash,
                },
            )
        )
        constraints.append(
            CanonicalRigConstraintIRV2(
                canonical_constraint_id=constraint_id,
                operator_id="IK",
                input_canonical_control_ids=(row.target_canonical_control_id,),
                output_canonical_joint_ids=tuple(path),
                canonical_chain_id=chain_id,
                parameters={"p0": float(row.mix), "p2": float(row.max_stretch)},
                source_proposal_constraint_id=f"STUDIO:{row.authoring_constraint_id}",
                metadata={
                    "authority_source": "QUALIFIED_STUDIO_RIG_CONTROL_PATCH",
                    "authoring_constraint_id": row.authoring_constraint_id,
                    "studio_patch_qualification_hash": qualified.qualification_hash,
                },
            )
        )

    combined_proposal_evidence = content_sha256({
        "base_proposal_evidence_hash": base_rig.proposal_evidence_hash,
        "studio_patch_proposal_hash": proposal.proposal_lineage_hash,
    })
    combined_qualification = content_sha256({
        "base_rig_program_qualification_hash": base_rig.rig_program_qualification_hash,
        "studio_patch_qualification_hash": qualified.qualification_hash,
    })
    metadata = dict(base_rig.metadata or {})
    metadata.update({
        "authority": "COMPILER_CANONICAL_RIG",
        "studio_control_program_patch_applied": True,
        "studio_control_program_patch_qualification_hash": qualified.qualification_hash,
        "studio_authoring_payload_sha256": proposal.authoring_payload_sha256,
        "deform_structure_mutated_by_studio_patch": False,
        "control_synthesis_performed_by_studio_patch": False,
        "constraint_execution_authority": "MECHANICS",
        "skin_binding_target": "DEFORM_STRUCTURE_HASH",
    })
    value = replace(
        base_rig,
        chains=tuple(chains),
        constraints=tuple(constraints),
        proposal_evidence_hash=combined_proposal_evidence,
        rig_program_qualification_hash=combined_qualification,
        deform_structure_hash="",
        control_program_hash="",
        rig_lineage_hash="",
        metadata=metadata,
    )
    value = replace(value, deform_structure_hash=_hash_deform_structure(value))
    if value.deform_structure_hash != base_rig.deform_structure_hash:
        raise QualificationError("STUDIO_RIG_PATCH_DEFORM_STRUCTURE_MUTATION")
    value = replace(value, control_program_hash=_hash_control_program(value))
    value = replace(value, rig_lineage_hash=canonical_rig_lineage_hash_v2(value))
    if value.rig_lineage_hash == base_rig.rig_lineage_hash:
        raise QualificationError("STUDIO_RIG_PATCH_RIG_LINEAGE_NOT_CHANGED")
    return value


__all__ = [
    "QualifiedStudioRigControlPatchIRV1",
    "StudioIKConstraintProposalIRV1",
    "StudioRigControlPatchProposalIRV1",
    "apply_qualified_studio_rig_control_patch_v1",
    "qualified_studio_rig_control_patch_hash_v1",
    "qualify_studio_rig_control_patch_v1",
    "seal_studio_rig_control_patch_proposal_v1",
    "studio_rig_control_patch_from_authoring_v1",
    "studio_rig_control_patch_proposal_hash_v1",
]
