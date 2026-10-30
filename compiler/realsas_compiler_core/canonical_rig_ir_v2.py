from __future__ import annotations

"""Canonical control-ready Rig authority for the AXIS V6 product line.

CanonicalRigIR.v2 separates deform structure identity from animator/control
program identity. Skin is expected to bind the deform-structure hash, not the
entire control program, so controller-only edits do not unnecessarily invalidate
otherwise identical skin weights.
"""

from dataclasses import asdict, dataclass, field, replace
import math
from typing import Any, Mapping

from .canonical_mesh_ir_v1 import CanonicalMeshIR, canonical_mesh_lineage_hash_v1
from .hashing import content_sha256
from .rig_program_ir_v1 import QualifiedRigProgramIR, qualified_rig_program_hash_v1
from .types import QualificationError

Json = dict[str, Any]
Vec3 = tuple[float, float, float]
Mat3 = tuple[Vec3, Vec3, Vec3]


@dataclass(frozen=True)
class CanonicalDeformJointIRV2:
    canonical_joint_id: str
    P: Vec3
    parent_canonical_joint_id: str | None
    source_proposal_joint_id: str
    support_evidence_ids: tuple[str, ...] = ()
    metadata: Json = field(default_factory=dict)


@dataclass(frozen=True)
class CanonicalJointFrameIRV2:
    canonical_joint_id: str
    basis: Mat3
    dof_labels: tuple[str, ...] = ()
    metadata: Json = field(default_factory=dict)


@dataclass(frozen=True)
class CanonicalRigControlIRV2:
    canonical_control_id: str
    control_kind: str
    target_canonical_joint_ids: tuple[str, ...] = ()
    parent_canonical_control_id: str | None = None
    rest_P: Vec3 = (0.0, 0.0, 0.0)
    source_proposal_control_id: str = ""
    metadata: Json = field(default_factory=dict)


@dataclass(frozen=True)
class CanonicalRigChainIRV2:
    canonical_chain_id: str
    canonical_joint_ids: tuple[str, ...]
    chain_kind: str
    source_proposal_chain_id: str
    metadata: Json = field(default_factory=dict)


@dataclass(frozen=True)
class CanonicalRigConstraintIRV2:
    canonical_constraint_id: str
    operator_id: str
    input_canonical_control_ids: tuple[str, ...] = ()
    output_canonical_joint_ids: tuple[str, ...] = ()
    canonical_chain_id: str | None = None
    parameters: Json = field(default_factory=dict)
    source_proposal_constraint_id: str = ""
    metadata: Json = field(default_factory=dict)


@dataclass(frozen=True)
class CanonicalAttachmentAnchorIRV2:
    canonical_anchor_id: str
    parent_canonical_joint_id: str | None = None
    parent_canonical_control_id: str | None = None
    local_P: Vec3 = (0.0, 0.0, 0.0)
    source_proposal_anchor_id: str = ""
    metadata: Json = field(default_factory=dict)


@dataclass(frozen=True)
class CanonicalRigIRV2:
    joints: tuple[CanonicalDeformJointIRV2, ...]
    local_frames: tuple[CanonicalJointFrameIRV2, ...]
    controls: tuple[CanonicalRigControlIRV2, ...]
    chains: tuple[CanonicalRigChainIRV2, ...]
    constraints: tuple[CanonicalRigConstraintIRV2, ...]
    attachment_anchors: tuple[CanonicalAttachmentAnchorIRV2, ...]
    deform_root_joint_ids: tuple[str, ...]
    mesh_lineage_hash: str
    feature_evidence_binding_hash: str
    proposal_evidence_hash: str
    rig_program_qualification_hash: str
    deform_structure_hash: str
    control_program_hash: str
    rig_lineage_hash: str
    schema_version: str = "RealSaS.CanonicalRigIR.v2"
    metadata: Json = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _hash_deform_structure(value: CanonicalRigIRV2) -> str:
    return content_sha256({
        "schema": "RealSaS.CanonicalDeformStructure.v2",
        "joints": [asdict(x) for x in value.joints],
        "local_frames": [asdict(x) for x in value.local_frames],
        "deform_root_joint_ids": value.deform_root_joint_ids,
        "mesh_lineage_hash": value.mesh_lineage_hash,
    })


def _hash_control_program(value: CanonicalRigIRV2) -> str:
    return content_sha256({
        "schema": "RealSaS.CanonicalControlProgram.v2",
        "controls": [asdict(x) for x in value.controls],
        "chains": [asdict(x) for x in value.chains],
        "constraints": [asdict(x) for x in value.constraints],
        "attachment_anchors": [asdict(x) for x in value.attachment_anchors],
        "deform_structure_hash": value.deform_structure_hash,
    })


def canonical_rig_lineage_hash_v2(value: CanonicalRigIRV2) -> str:
    payload = value.to_dict()
    payload.pop("rig_lineage_hash", None)
    return content_sha256(payload)


def _cid(prefix: str, qualification_hash: str, source_id: str) -> str:
    return prefix + ":" + content_sha256({
        "qualification_hash": qualification_hash,
        "source_id": source_id,
    })[:20]


def mint_canonical_rig_v2(
    mesh: CanonicalMeshIR,
    *,
    qualified_program: QualifiedRigProgramIR,
) -> CanonicalRigIRV2:
    if mesh.mesh_lineage_hash != canonical_mesh_lineage_hash_v1(mesh):
        raise QualificationError("CANONICAL_RIG_V2_MESH_LINEAGE_DRIFT")
    if mesh.metadata.get("authority") != "COMPILER_CANONICAL_MESH":
        raise QualificationError("CANONICAL_RIG_V2_MESH_AUTHORITY_DRIFT")
    if qualified_program.qualification_hash != qualified_rig_program_hash_v1(qualified_program):
        raise QualificationError("CANONICAL_RIG_V2_PROGRAM_QUALIFICATION_HASH_DRIFT")
    report = dict(qualified_program.qualification_report or {})
    if report.get("status") != "PASS_RIG_PROGRAM_VALIDATED":
        raise QualificationError("CANONICAL_RIG_V2_PROGRAM_NOT_QUALIFIED")
    for forbidden in (
        "parent_reselection_performed",
        "control_synthesis_performed",
        "constraint_synthesis_performed",
        "frame_repair_performed",
        "constraint_solver_executed",
    ):
        if report.get(forbidden) is not False:
            raise QualificationError(f"CANONICAL_RIG_V2_COMPILER_OVERREACH:{forbidden}")

    p = qualified_program.proposal
    if p.mesh_lineage_hash != mesh.mesh_lineage_hash:
        raise QualificationError("CANONICAL_RIG_V2_MESH_PARENT_DRIFT")

    qh = qualified_program.qualification_hash
    joint_map = {j.proposal_joint_id: _cid("J", qh, j.proposal_joint_id) for j in p.joints}
    control_map = {c.proposal_control_id: _cid("C", qh, c.proposal_control_id) for c in p.controls}
    chain_map = {c.proposal_chain_id: _cid("K", qh, c.proposal_chain_id) for c in p.chains}

    joints = tuple(
        CanonicalDeformJointIRV2(
            canonical_joint_id=joint_map[j.proposal_joint_id],
            P=tuple(map(float, j.P)),  # type: ignore[arg-type]
            parent_canonical_joint_id=(None if j.parent_proposal_joint_id is None else joint_map[j.parent_proposal_joint_id]),
            source_proposal_joint_id=j.proposal_joint_id,
            support_evidence_ids=tuple(map(str, j.support_evidence_ids)),
            metadata={"learned_proposal_identity_is_provenance_only": True},
        )
        for j in sorted(p.joints, key=lambda row: (row.sequence_index, row.proposal_joint_id))
    )
    frames_by = {f.proposal_joint_id: f for f in p.local_frames}
    frames = tuple(
        CanonicalJointFrameIRV2(
            canonical_joint_id=joint_map[j.proposal_joint_id],
            basis=frames_by[j.proposal_joint_id].basis,
            dof_labels=tuple(map(str, frames_by[j.proposal_joint_id].dof_labels)),
            metadata=dict(frames_by[j.proposal_joint_id].metadata or {}),
        )
        for j in sorted(p.joints, key=lambda row: (row.sequence_index, row.proposal_joint_id))
    )
    controls = tuple(
        CanonicalRigControlIRV2(
            canonical_control_id=control_map[c.proposal_control_id],
            control_kind=str(c.control_kind),
            target_canonical_joint_ids=tuple(joint_map[x] for x in c.target_proposal_joint_ids),
            parent_canonical_control_id=(None if c.parent_proposal_control_id is None else control_map[c.parent_proposal_control_id]),
            rest_P=tuple(map(float, c.rest_P)),  # type: ignore[arg-type]
            source_proposal_control_id=c.proposal_control_id,
            metadata=dict(c.metadata or {}),
        )
        for c in p.controls
    )
    chains = tuple(
        CanonicalRigChainIRV2(
            canonical_chain_id=chain_map[c.proposal_chain_id],
            canonical_joint_ids=tuple(joint_map[x] for x in c.proposal_joint_ids),
            chain_kind=str(c.chain_kind),
            source_proposal_chain_id=c.proposal_chain_id,
            metadata=dict(c.metadata or {}),
        )
        for c in p.chains
    )
    constraints = tuple(
        CanonicalRigConstraintIRV2(
            canonical_constraint_id=_cid("Q", qh, c.proposal_constraint_id),
            operator_id=str(c.operator_id),
            input_canonical_control_ids=tuple(control_map[x] for x in c.input_proposal_control_ids),
            output_canonical_joint_ids=tuple(joint_map[x] for x in c.output_proposal_joint_ids),
            canonical_chain_id=(None if c.proposal_chain_id is None else chain_map[c.proposal_chain_id]),
            parameters=dict(c.parameters or {}),
            source_proposal_constraint_id=c.proposal_constraint_id,
            metadata=dict(c.metadata or {}),
        )
        for c in p.constraints
    )
    anchors = tuple(
        CanonicalAttachmentAnchorIRV2(
            canonical_anchor_id=_cid("A", qh, a.proposal_anchor_id),
            parent_canonical_joint_id=(None if a.parent_proposal_joint_id is None else joint_map[a.parent_proposal_joint_id]),
            parent_canonical_control_id=(None if a.parent_proposal_control_id is None else control_map[a.parent_proposal_control_id]),
            local_P=tuple(map(float, a.local_P)),  # type: ignore[arg-type]
            source_proposal_anchor_id=a.proposal_anchor_id,
            metadata=dict(a.metadata or {}),
        )
        for a in p.attachment_anchors
    )
    roots = tuple(sorted(j.canonical_joint_id for j in joints if j.parent_canonical_joint_id is None))

    value = CanonicalRigIRV2(
        joints=joints,
        local_frames=frames,
        controls=controls,
        chains=chains,
        constraints=constraints,
        attachment_anchors=anchors,
        deform_root_joint_ids=roots,
        mesh_lineage_hash=mesh.mesh_lineage_hash,
        feature_evidence_binding_hash=p.feature_evidence_binding_hash,
        proposal_evidence_hash=p.proposal_lineage_hash,
        rig_program_qualification_hash=qh,
        deform_structure_hash="",
        control_program_hash="",
        rig_lineage_hash="",
        metadata={
            "authority": "COMPILER_CANONICAL_RIG",
            "canonical_parent": "CANONICAL_MESH",
            "learned_family_embedded_in_schema": False,
            "learned_rig_program_is_proposal_evidence": True,
            "compiler_parent_reselection_performed": False,
            "compiler_control_synthesis_performed": False,
            "compiler_constraint_synthesis_performed": False,
            "constraint_execution_authority": "MECHANICS",
            "skin_binding_target": "DEFORM_STRUCTURE_HASH",
            "motion_editing_target": "CONTROL_PROGRAM",
        },
    )
    value = replace(value, deform_structure_hash=_hash_deform_structure(value))
    value = replace(value, control_program_hash=_hash_control_program(value))
    return replace(value, rig_lineage_hash=canonical_rig_lineage_hash_v2(value))


def canonical_rig_from_dict_v2(payload: Mapping[str, Any]) -> CanonicalRigIRV2:
    schema = str(payload.get("schema_version") or payload.get("schema") or "")
    if schema != "RealSaS.CanonicalRigIR.v2":
        raise QualificationError("CANONICAL_RIG_V2_SCHEMA_DRIFT")
    def vec3(v):
        out = tuple(map(float, v))
        if len(out) != 3 or not all(math.isfinite(x) for x in out):
            raise QualificationError("CANONICAL_RIG_V2_VECTOR_INVALID")
        return out
    joints = tuple(CanonicalDeformJointIRV2(
        canonical_joint_id=str(x["canonical_joint_id"]), P=vec3(x["P"]),
        parent_canonical_joint_id=None if x.get("parent_canonical_joint_id") is None else str(x["parent_canonical_joint_id"]),
        source_proposal_joint_id=str(x["source_proposal_joint_id"]),
        support_evidence_ids=tuple(map(str, x.get("support_evidence_ids") or ())), metadata=dict(x.get("metadata") or {}),
    ) for x in payload.get("joints") or ())
    frames = tuple(CanonicalJointFrameIRV2(
        canonical_joint_id=str(x["canonical_joint_id"]), basis=tuple(tuple(map(float, row)) for row in x["basis"]),  # type: ignore[arg-type]
        dof_labels=tuple(map(str, x.get("dof_labels") or ())), metadata=dict(x.get("metadata") or {}),
    ) for x in payload.get("local_frames") or ())
    controls = tuple(CanonicalRigControlIRV2(
        canonical_control_id=str(x["canonical_control_id"]), control_kind=str(x["control_kind"]),
        target_canonical_joint_ids=tuple(map(str, x.get("target_canonical_joint_ids") or ())),
        parent_canonical_control_id=None if x.get("parent_canonical_control_id") is None else str(x["parent_canonical_control_id"]),
        rest_P=vec3(x.get("rest_P") or (0, 0, 0)), source_proposal_control_id=str(x.get("source_proposal_control_id") or ""), metadata=dict(x.get("metadata") or {}),
    ) for x in payload.get("controls") or ())
    chains = tuple(CanonicalRigChainIRV2(str(x["canonical_chain_id"]), tuple(map(str, x.get("canonical_joint_ids") or ())), str(x["chain_kind"]), str(x["source_proposal_chain_id"]), dict(x.get("metadata") or {})) for x in payload.get("chains") or ())
    constraints = tuple(CanonicalRigConstraintIRV2(
        canonical_constraint_id=str(x["canonical_constraint_id"]), operator_id=str(x["operator_id"]),
        input_canonical_control_ids=tuple(map(str, x.get("input_canonical_control_ids") or ())), output_canonical_joint_ids=tuple(map(str, x.get("output_canonical_joint_ids") or ())),
        canonical_chain_id=None if x.get("canonical_chain_id") is None else str(x["canonical_chain_id"]), parameters=dict(x.get("parameters") or {}),
        source_proposal_constraint_id=str(x.get("source_proposal_constraint_id") or ""), metadata=dict(x.get("metadata") or {}),
    ) for x in payload.get("constraints") or ())
    anchors = tuple(CanonicalAttachmentAnchorIRV2(
        canonical_anchor_id=str(x["canonical_anchor_id"]), parent_canonical_joint_id=None if x.get("parent_canonical_joint_id") is None else str(x["parent_canonical_joint_id"]),
        parent_canonical_control_id=None if x.get("parent_canonical_control_id") is None else str(x["parent_canonical_control_id"]), local_P=vec3(x.get("local_P") or (0, 0, 0)),
        source_proposal_anchor_id=str(x.get("source_proposal_anchor_id") or ""), metadata=dict(x.get("metadata") or {}),
    ) for x in payload.get("attachment_anchors") or ())
    value = CanonicalRigIRV2(
        joints, frames, controls, chains, constraints, anchors,
        tuple(map(str, payload.get("deform_root_joint_ids") or ())), str(payload["mesh_lineage_hash"]), str(payload["feature_evidence_binding_hash"]),
        str(payload["proposal_evidence_hash"]), str(payload["rig_program_qualification_hash"]), str(payload["deform_structure_hash"]), str(payload["control_program_hash"]),
        str(payload["rig_lineage_hash"]), schema, dict(payload.get("metadata") or {}),
    )
    if value.deform_structure_hash != _hash_deform_structure(value):
        raise QualificationError("CANONICAL_RIG_V2_DEFORM_HASH_DRIFT")
    if value.control_program_hash != _hash_control_program(value):
        raise QualificationError("CANONICAL_RIG_V2_CONTROL_HASH_DRIFT")
    if value.rig_lineage_hash != canonical_rig_lineage_hash_v2(value):
        raise QualificationError("CANONICAL_RIG_V2_LINEAGE_HASH_DRIFT")
    return value


__all__ = [
    "CanonicalAttachmentAnchorIRV2", "CanonicalDeformJointIRV2", "CanonicalJointFrameIRV2",
    "CanonicalRigChainIRV2", "CanonicalRigConstraintIRV2", "CanonicalRigControlIRV2", "CanonicalRigIRV2",
    "canonical_rig_from_dict_v2", "canonical_rig_lineage_hash_v2", "mint_canonical_rig_v2",
]
