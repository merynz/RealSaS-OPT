from __future__ import annotations

"""Declared cross-parent relative 6D attachment bridge, research only.

A link between two distinct joint-owned grips is NOT inferred from proximity
or body part names. It exists only when supplied as a typed runtime contract
and SHA-bound to the exact canonical rig and control program.

The relative transform is in attachment A's declared local frame; it can
represent a shared rigid prop, but DOES NOT prove real source coupling,
material stiffness, collisions or friction. No source truth is minted.
"""

from dataclasses import dataclass, field
import math
from typing import Sequence

import numpy as np

from .canonical_rig_ir_v2 import CanonicalRigIRV2
from .mechanics_attachment_frames_v1 import _matrix
from .mechanics_control_solver_v1 import MechanicsPoseIRV1
from .types import QualificationError

Mat3 = tuple[tuple[float,float,float], ...]
Vec3 = tuple[float,float,float]


@dataclass(frozen=True)
class DeclaredRelativeAttachmentBridgeV1:
    bridge_id: str
    anchor_a_id: str
    anchor_b_id: str
    a_to_b_local_P: Vec3
    a_to_b_local_basis: Mat3
    declaration_evidence_id: str
    rig_lineage_hash: str
    control_program_hash: str


@dataclass(frozen=True)
class PreparedRelativeAttachmentBridgeV1:
    bridge_id: str
    anchor_a_id: str
    anchor_b_id: str
    parent_a_id: str
    parent_b_id: str
    local_a_P: np.ndarray
    local_b_P: np.ndarray
    local_a_basis: np.ndarray
    local_b_basis: np.ndarray
    a_to_b_local_P: np.ndarray
    a_to_b_local_basis: np.ndarray


@dataclass(frozen=True)
class RelativeAttachmentBridgeResidualV1:
    bridge_id: str
    anchor_a_id: str
    anchor_b_id: str
    expected_world_b_P: Vec3
    actual_world_b_P: Vec3
    expected_world_b_basis: Mat3
    actual_world_b_basis: Mat3
    position_residual_world: float
    orientation_residual_rad: float


@dataclass(frozen=True)
class RelativeAttachmentBridgeCourtV1:
    status: str
    residuals: tuple[RelativeAttachmentBridgeResidualV1, ...]
    max_position_residual_world: float
    max_orientation_residual_rad: float
    metadata: dict = field(default_factory=dict)


def _vector3(value: object, code: str) -> np.ndarray:
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (ValueError,TypeError) as exc:
        raise QualificationError(code) from exc
    if arr.shape!=(3,) or not np.isfinite(arr).all():
        raise QualificationError(code)
    return arr.copy()


def prepare_declared_relative_attachment_bridges_v1(
    rig: CanonicalRigIRV2,
    bridges: Sequence[DeclaredRelativeAttachmentBridgeV1],
    *,
    maximum_bridges: int = 16,
) -> tuple[PreparedRelativeAttachmentBridgeV1, ...]:
    if not isinstance(maximum_bridges,int) or not 1<=maximum_bridges<=64:
        raise QualificationError("RELATIVE_BRIDGE_RESOURCE_POLICY_INVALID")
    if not bridges or len(bridges)>maximum_bridges:
        raise QualificationError("RELATIVE_BRIDGE_COUNT_INVALID")
    anchors={a.canonical_anchor_id:a for a in rig.attachment_anchors}
    joint_ids={j.canonical_joint_id for j in rig.joints}
    if len(anchors)!=len(rig.attachment_anchors) or len(joint_ids)!=len(rig.joints):
        raise QualificationError("RELATIVE_BRIDGE_RIG_DUPLICATE_IDS")
    seen_bridges=set()
    seen_pairs=set()
    output=[]
    for row in bridges:
        if not isinstance(row,DeclaredRelativeAttachmentBridgeV1):
            raise QualificationError("RELATIVE_BRIDGE_TYPED_DECLARATION_REQUIRED")
        if (not row.bridge_id or not row.declaration_evidence_id
                or row.bridge_id in seen_bridges):
            raise QualificationError("RELATIVE_BRIDGE_DECLARATION_ID_INVALID")
        if (row.rig_lineage_hash!=rig.rig_lineage_hash
                or row.control_program_hash!=rig.control_program_hash):
            raise QualificationError("RELATIVE_BRIDGE_AUTHORITY_LINEAGE_DRIFT")
        a_id,b_id=row.anchor_a_id,row.anchor_b_id
        if a_id==b_id or a_id not in anchors or b_id not in anchors:
            raise QualificationError("RELATIVE_BRIDGE_ANCHOR_PAIR_INVALID")
        pair=tuple(sorted((a_id,b_id)))
        if pair in seen_pairs:
            raise QualificationError("RELATIVE_BRIDGE_DUPLICATE_PAIR")
        seen_pairs.add(pair)
        seen_bridges.add(row.bridge_id)
        a,b=anchors[a_id],anchors[b_id]
        pa,pb=a.parent_canonical_joint_id,b.parent_canonical_joint_id
        if (pa not in joint_ids or pb not in joint_ids
                or a.parent_canonical_control_id is not None
                or b.parent_canonical_control_id is not None):
            raise QualificationError("RELATIVE_BRIDGE_PARENT_NOT_EXECUTABLE")
        if pa==pb:
            raise QualificationError("RELATIVE_BRIDGE_SAME_PARENT_USE_RIGIDITY_COURT")
        if "local_basis" not in a.metadata or "local_basis" not in b.metadata:
            raise QualificationError("RELATIVE_BRIDGE_UNDECLARED_LOCAL_FRAME")
        output.append(PreparedRelativeAttachmentBridgeV1(
            bridge_id=row.bridge_id,anchor_a_id=a_id,anchor_b_id=b_id,
            parent_a_id=pa,parent_b_id=pb,
            local_a_P=_vector3(a.local_P,"RELATIVE_BRIDGE_LOCAL_POSITION_INVALID"),
            local_b_P=_vector3(b.local_P,"RELATIVE_BRIDGE_LOCAL_POSITION_INVALID"),
            local_a_basis=_matrix(
                a.metadata["local_basis"],"RELATIVE_BRIDGE_LOCAL_BASIS_INVALID",
            ),
            local_b_basis=_matrix(
                b.metadata["local_basis"],"RELATIVE_BRIDGE_LOCAL_BASIS_INVALID",
            ),
            a_to_b_local_P=_vector3(
                row.a_to_b_local_P,"RELATIVE_BRIDGE_DECLARED_OFFSET_INVALID",
            ),
            a_to_b_local_basis=_matrix(
                row.a_to_b_local_basis,"RELATIVE_BRIDGE_DECLARED_ROTATION_INVALID",
            ),
        ))
    return tuple(output)


def evaluate_declared_relative_attachment_bridges_v1(
    rig: CanonicalRigIRV2,
    pose: MechanicsPoseIRV1,
    bridges: Sequence[DeclaredRelativeAttachmentBridgeV1],
    *,
    position_tolerance_world: float = 1e-3,
    orientation_tolerance_rad: float = 1e-3,
) -> RelativeAttachmentBridgeCourtV1:
    for tol,code in (
        (position_tolerance_world,"RELATIVE_BRIDGE_POSITION_TOLERANCE_INVALID"),
        (orientation_tolerance_rad,"RELATIVE_BRIDGE_ORIENTATION_TOLERANCE_INVALID"),
    ):
        if not math.isfinite(float(tol)) or float(tol)<=0:
            raise QualificationError(code)
    if orientation_tolerance_rad>math.pi:
        raise QualificationError("RELATIVE_BRIDGE_ORIENTATION_TOLERANCE_INVALID")
    prepared=prepare_declared_relative_attachment_bridges_v1(rig,bridges)
    if (pose.rig_lineage_hash!=rig.rig_lineage_hash
            or pose.control_program_hash!=rig.control_program_hash):
        raise QualificationError("RELATIVE_BRIDGE_POSE_LINEAGE_DRIFT")
    posed={j.canonical_joint_id:j for j in pose.joints}
    if len(posed)!=len(pose.joints) or len(posed)!=len(rig.joints):
        raise QualificationError("RELATIVE_BRIDGE_POSE_JOINT_COVERAGE")
    output=[]
    for link in prepared:
        ja,jb=posed[link.parent_a_id],posed[link.parent_b_id]
        ra=_matrix(ja.rotation_basis,"RELATIVE_BRIDGE_POSE_ROTATION_INVALID")
        rb=_matrix(jb.rotation_basis,"RELATIVE_BRIDGE_POSE_ROTATION_INVALID")
        pa=_vector3(ja.solved_P,"RELATIVE_BRIDGE_POSE_POSITION_INVALID")
        pb=_vector3(jb.solved_P,"RELATIVE_BRIDGE_POSE_POSITION_INVALID")
        grip_a=pa+ra@link.local_a_P
        grip_b=pb+rb@link.local_b_P
        R_a=ra@link.local_a_basis
        R_b=rb@link.local_b_basis
        expected_P=grip_a+R_a@link.a_to_b_local_P
        expected_R=R_a@link.a_to_b_local_basis
        err_p=float(np.linalg.norm(grip_b-expected_P))
        err_R=math.acos(float(np.clip(
            (np.trace(expected_R.T@R_b)-1.)/2.,-1.,1.,
        )))
        output.append(RelativeAttachmentBridgeResidualV1(
            bridge_id=link.bridge_id,
            anchor_a_id=link.anchor_a_id,
            anchor_b_id=link.anchor_b_id,
            expected_world_b_P=tuple(map(float,expected_P)),
            actual_world_b_P=tuple(map(float,grip_b)),
            expected_world_b_basis=tuple(tuple(map(float,row)) for row in expected_R),
            actual_world_b_basis=tuple(tuple(map(float,row)) for row in R_b),
            position_residual_world=err_p,
            orientation_residual_rad=err_R,
        ))
    max_p=max((r.position_residual_world for r in output),default=0.)
    max_R=max((r.orientation_residual_rad for r in output),default=0.)
    return RelativeAttachmentBridgeCourtV1(
        status=("PASS_DECLARED_RELATIVE_BRIDGE_COURT"
                if max_p<=position_tolerance_world and max_R<=orientation_tolerance_rad
                else "FAIL_DECLARED_RELATIVE_BRIDGE_COURT"),
        residuals=tuple(output),max_position_residual_world=max_p,
        max_orientation_residual_rad=max_R,
        metadata={
            "rig_lineage_hash":rig.rig_lineage_hash,
            "control_program_hash":rig.control_program_hash,
            "authority":"EXPLICIT_RUNTIME_RELATIVE_ATTACHMENT_RESEARCH_ONLY",
            "independent_court_after_pose":True,
            "declared_coupling_does_not_prove_source_or_shared_object":True,
            "physical_contact_force_or_friction_claimed":False,
            "model_or_graphics_authority_minted":False,
        },
    )


__all__=[
    "DeclaredRelativeAttachmentBridgeV1",
    "PreparedRelativeAttachmentBridgeV1",
    "RelativeAttachmentBridgeResidualV1",
    "RelativeAttachmentBridgeCourtV1",
    "prepare_declared_relative_attachment_bridges_v1",
    "evaluate_declared_relative_attachment_bridges_v1",
]
