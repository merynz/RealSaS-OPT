from __future__ import annotations

"""AXIS V6 control-ready rig proposal boundary.

The learned provider owns proposal content. The Compiler owns only typed legality,
identity binding and canonical materialization. This module deliberately does not
solve IK, synthesize missing controls, reselect parents, repair frames or infer
semantic body-part labels.
"""

from dataclasses import asdict, dataclass, field, replace
import math
from typing import Any, Mapping

from .hashing import content_sha256
from .types import QualificationError

Json = dict[str, Any]
Vec3 = tuple[float, float, float]
Mat3 = tuple[Vec3, Vec3, Vec3]


@dataclass(frozen=True)
class RigProgramJointProposalIR:
    proposal_joint_id: str
    P: Vec3
    parent_proposal_joint_id: str | None
    support_evidence_ids: tuple[str, ...] = ()
    sequence_index: int = 0
    metadata: Json = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class RigProgramLocalFrameProposalIR:
    proposal_joint_id: str
    basis: Mat3
    dof_labels: tuple[str, ...] = ()
    metadata: Json = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class RigProgramControlProposalIR:
    proposal_control_id: str
    control_kind: str
    target_proposal_joint_ids: tuple[str, ...] = ()
    parent_proposal_control_id: str | None = None
    rest_P: Vec3 = (0.0, 0.0, 0.0)
    metadata: Json = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class RigProgramChainProposalIR:
    proposal_chain_id: str
    proposal_joint_ids: tuple[str, ...]
    chain_kind: str = "GENERIC_KINEMATIC_CHAIN"
    metadata: Json = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class RigProgramConstraintProposalIR:
    proposal_constraint_id: str
    operator_id: str
    input_proposal_control_ids: tuple[str, ...] = ()
    output_proposal_joint_ids: tuple[str, ...] = ()
    proposal_chain_id: str | None = None
    parameters: Json = field(default_factory=dict)
    metadata: Json = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class RigProgramAttachmentAnchorProposalIR:
    proposal_anchor_id: str
    parent_proposal_joint_id: str | None = None
    parent_proposal_control_id: str | None = None
    local_P: Vec3 = (0.0, 0.0, 0.0)
    metadata: Json = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class RigProgramProposalIR:
    joints: tuple[RigProgramJointProposalIR, ...]
    local_frames: tuple[RigProgramLocalFrameProposalIR, ...]
    controls: tuple[RigProgramControlProposalIR, ...]
    chains: tuple[RigProgramChainProposalIR, ...]
    constraints: tuple[RigProgramConstraintProposalIR, ...]
    attachment_anchors: tuple[RigProgramAttachmentAnchorProposalIR, ...]
    mesh_lineage_hash: str
    feature_evidence_binding_hash: str
    model_provenance: str
    proposal_lineage_hash: str = ""
    schema_version: str = "RealSaS.RigProgramProposalIR.v1"
    metadata: Json = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class QualifiedRigProgramIR:
    proposal: RigProgramProposalIR
    qualification_report: Json
    qualification_hash: str
    schema_version: str = "RealSaS.QualifiedRigProgramIR.v1"
    metadata: Json = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def rig_program_proposal_lineage_hash_v1(value: RigProgramProposalIR) -> str:
    payload = value.to_dict()
    payload.pop("proposal_lineage_hash", None)
    return content_sha256(payload)


def qualified_rig_program_hash_v1(value: QualifiedRigProgramIR) -> str:
    payload = value.to_dict()
    payload.pop("qualification_hash", None)
    return content_sha256(payload)


def _finite3(value: tuple[float, ...], code: str) -> tuple[float, float, float]:
    out = tuple(map(float, value))
    if len(out) != 3 or not all(math.isfinite(v) for v in out):
        raise QualificationError(code)
    return out  # type: ignore[return-value]


def _validate_id_set(values: tuple[str, ...], code: str) -> set[str]:
    cleaned = [str(v).strip() for v in values]
    if any(not v for v in cleaned) or len(cleaned) != len(set(cleaned)):
        raise QualificationError(code)
    return set(cleaned)


def _validate_parent_forest(joints: tuple[RigProgramJointProposalIR, ...]) -> tuple[str, ...]:
    ids = _validate_id_set(tuple(j.proposal_joint_id for j in joints), "RIG_PROGRAM_JOINT_ID_INVALID")
    if not ids:
        raise QualificationError("RIG_PROGRAM_JOINTS_EMPTY")
    by_id = {j.proposal_joint_id: j for j in joints}
    roots: list[str] = []
    seen_seq: set[int] = set()
    for joint in joints:
        _finite3(joint.P, "RIG_PROGRAM_JOINT_POSITION_INVALID")
        if joint.sequence_index < 0 or joint.sequence_index in seen_seq:
            raise QualificationError("RIG_PROGRAM_JOINT_SEQUENCE_INVALID")
        seen_seq.add(joint.sequence_index)
        parent = joint.parent_proposal_joint_id
        if parent is None:
            roots.append(joint.proposal_joint_id)
        elif parent not in ids or parent == joint.proposal_joint_id:
            raise QualificationError("RIG_PROGRAM_JOINT_PARENT_INVALID")
    if not roots:
        raise QualificationError("RIG_PROGRAM_DEFORM_ROOT_REQUIRED")
    for jid in ids:
        walk: set[str] = set()
        cur = jid
        while by_id[cur].parent_proposal_joint_id is not None:
            if cur in walk:
                raise QualificationError("RIG_PROGRAM_JOINT_CYCLE")
            walk.add(cur)
            cur = str(by_id[cur].parent_proposal_joint_id)
    return tuple(sorted(roots))


def _validate_control_graph(controls: tuple[RigProgramControlProposalIR, ...], joint_ids: set[str]) -> set[str]:
    control_ids = _validate_id_set(tuple(c.proposal_control_id for c in controls), "RIG_PROGRAM_CONTROL_ID_INVALID")
    by_id = {c.proposal_control_id: c for c in controls}
    for control in controls:
        if not str(control.control_kind).strip():
            raise QualificationError("RIG_PROGRAM_CONTROL_KIND_REQUIRED")
        _finite3(control.rest_P, "RIG_PROGRAM_CONTROL_REST_POSITION_INVALID")
        if any(j not in joint_ids for j in control.target_proposal_joint_ids):
            raise QualificationError("RIG_PROGRAM_CONTROL_TARGET_JOINT_UNKNOWN")
        parent = control.parent_proposal_control_id
        if parent is not None and (parent not in control_ids or parent == control.proposal_control_id):
            raise QualificationError("RIG_PROGRAM_CONTROL_PARENT_INVALID")
    for cid in control_ids:
        walk: set[str] = set()
        cur = cid
        while by_id[cur].parent_proposal_control_id is not None:
            if cur in walk:
                raise QualificationError("RIG_PROGRAM_CONTROL_CYCLE")
            walk.add(cur)
            cur = str(by_id[cur].parent_proposal_control_id)
    return control_ids


def qualify_rig_program_v1(proposal: RigProgramProposalIR) -> QualifiedRigProgramIR:
    """Validate learned rig-program structure without inventing or repairing it.

    This court is intentionally permissive about controller/constraint vocabulary:
    product execution support is a downstream capability question. Here we reject
    malformed references, cycles, degenerate frames and non-contiguous declared
    kinematic chains only.
    """
    if proposal.schema_version != "RealSaS.RigProgramProposalIR.v1":
        raise QualificationError("RIG_PROGRAM_SCHEMA_DRIFT")
    if not proposal.mesh_lineage_hash or not proposal.feature_evidence_binding_hash:
        raise QualificationError("RIG_PROGRAM_PARENT_BINDING_REQUIRED")
    if not proposal.model_provenance:
        raise QualificationError("RIG_PROGRAM_MODEL_PROVENANCE_REQUIRED")
    expected = rig_program_proposal_lineage_hash_v1(proposal)
    if proposal.proposal_lineage_hash != expected:
        raise QualificationError("RIG_PROGRAM_PROPOSAL_LINEAGE_DRIFT")

    roots = _validate_parent_forest(proposal.joints)
    joint_ids = {j.proposal_joint_id for j in proposal.joints}
    parent_by = {j.proposal_joint_id: j.parent_proposal_joint_id for j in proposal.joints}

    frame_ids = _validate_id_set(tuple(f.proposal_joint_id for f in proposal.local_frames), "RIG_PROGRAM_FRAME_JOINT_DUPLICATE")
    if frame_ids != joint_ids:
        raise QualificationError("RIG_PROGRAM_FRAME_COVERAGE_DRIFT")
    for frame in proposal.local_frames:
        rows = tuple(_finite3(row, "RIG_PROGRAM_FRAME_NONFINITE") for row in frame.basis)
        if len(rows) != 3:
            raise QualificationError("RIG_PROGRAM_FRAME_SHAPE_INVALID")
        a, b, c = rows
        det = (
            a[0] * (b[1] * c[2] - b[2] * c[1])
            - a[1] * (b[0] * c[2] - b[2] * c[0])
            + a[2] * (b[0] * c[1] - b[1] * c[0])
        )
        if not math.isfinite(det) or abs(det) <= 1e-8:
            raise QualificationError("RIG_PROGRAM_FRAME_DEGENERATE")
        if len(frame.dof_labels) != len(set(frame.dof_labels)) or any(not str(x).strip() for x in frame.dof_labels):
            raise QualificationError("RIG_PROGRAM_DOF_LABEL_INVALID")

    control_ids = _validate_control_graph(proposal.controls, joint_ids)

    chain_ids = _validate_id_set(tuple(c.proposal_chain_id for c in proposal.chains), "RIG_PROGRAM_CHAIN_ID_INVALID")
    for chain in proposal.chains:
        if not chain.proposal_joint_ids or len(chain.proposal_joint_ids) != len(set(chain.proposal_joint_ids)):
            raise QualificationError("RIG_PROGRAM_CHAIN_JOINT_SEQUENCE_INVALID")
        if any(j not in joint_ids for j in chain.proposal_joint_ids):
            raise QualificationError("RIG_PROGRAM_CHAIN_JOINT_UNKNOWN")
        for parent, child in zip(chain.proposal_joint_ids, chain.proposal_joint_ids[1:]):
            if parent_by.get(child) != parent:
                raise QualificationError("RIG_PROGRAM_CHAIN_NOT_PARENT_CONTIGUOUS")

    _validate_id_set(tuple(c.proposal_constraint_id for c in proposal.constraints), "RIG_PROGRAM_CONSTRAINT_ID_INVALID")
    for constraint in proposal.constraints:
        if not str(constraint.operator_id).strip():
            raise QualificationError("RIG_PROGRAM_CONSTRAINT_OPERATOR_REQUIRED")
        if any(cid not in control_ids for cid in constraint.input_proposal_control_ids):
            raise QualificationError("RIG_PROGRAM_CONSTRAINT_CONTROL_UNKNOWN")
        if any(jid not in joint_ids for jid in constraint.output_proposal_joint_ids):
            raise QualificationError("RIG_PROGRAM_CONSTRAINT_JOINT_UNKNOWN")
        if constraint.proposal_chain_id is not None and constraint.proposal_chain_id not in chain_ids:
            raise QualificationError("RIG_PROGRAM_CONSTRAINT_CHAIN_UNKNOWN")

    _validate_id_set(tuple(a.proposal_anchor_id for a in proposal.attachment_anchors), "RIG_PROGRAM_ANCHOR_ID_INVALID")
    for anchor in proposal.attachment_anchors:
        _finite3(anchor.local_P, "RIG_PROGRAM_ANCHOR_POSITION_INVALID")
        joint_parent = anchor.parent_proposal_joint_id is not None
        control_parent = anchor.parent_proposal_control_id is not None
        if joint_parent == control_parent:
            raise QualificationError("RIG_PROGRAM_ANCHOR_EXACTLY_ONE_PARENT_REQUIRED")
        if joint_parent and anchor.parent_proposal_joint_id not in joint_ids:
            raise QualificationError("RIG_PROGRAM_ANCHOR_JOINT_UNKNOWN")
        if control_parent and anchor.parent_proposal_control_id not in control_ids:
            raise QualificationError("RIG_PROGRAM_ANCHOR_CONTROL_UNKNOWN")

    report = {
        "status": "PASS_RIG_PROGRAM_VALIDATED",
        "joint_count": len(proposal.joints),
        "deform_root_count": len(roots),
        "control_count": len(proposal.controls),
        "chain_count": len(proposal.chains),
        "constraint_count": len(proposal.constraints),
        "attachment_anchor_count": len(proposal.attachment_anchors),
        "parent_reselection_performed": False,
        "control_synthesis_performed": False,
        "constraint_synthesis_performed": False,
        "frame_repair_performed": False,
        "constraint_solver_executed": False,
        "compiler_role": "VALIDATE_AND_MATERIALIZE_ONLY",
    }
    value = QualifiedRigProgramIR(
        proposal=proposal,
        qualification_report=report,
        qualification_hash="",
        metadata={
            "learned_program_preserved_exactly": True,
            "compiler_does_not_complete_missing_rig_semantics": True,
            "mechanics_execution_required_downstream": True,
        },
    )
    return replace(value, qualification_hash=qualified_rig_program_hash_v1(value))


def seal_rig_program_proposal_v1(value: RigProgramProposalIR) -> RigProgramProposalIR:
    """Convenience constructor step for model adapters before Compiler qualification."""
    return replace(value, proposal_lineage_hash=rig_program_proposal_lineage_hash_v1(value))


__all__ = [
    "QualifiedRigProgramIR",
    "RigProgramAttachmentAnchorProposalIR",
    "RigProgramChainProposalIR",
    "RigProgramConstraintProposalIR",
    "RigProgramControlProposalIR",
    "RigProgramJointProposalIR",
    "RigProgramLocalFrameProposalIR",
    "RigProgramProposalIR",
    "qualify_rig_program_v1",
    "qualified_rig_program_hash_v1",
    "rig_program_proposal_lineage_hash_v1",
    "seal_rig_program_proposal_v1",
]
