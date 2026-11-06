from __future__ import annotations

"""World-space attachment frames from declared canonical rig + primary pose.

A sword grip and an attached object require POSITION + ORIENTATION, not an
inferred body-part name. A CanonicalAttachmentAnchorIRV2 always declares a
local point; orientation is only *authored* when an explicit local_basis is
present in the anchor metadata. Otherwise the parent's basis may be reported
as a convenience but is NOT a source-attested grip orientation.

This is a read-only consequence evaluator, not an orientation IK solver,
constraint author, graphics draw-order system, or physical attachment force.
"""

from dataclasses import dataclass, field
import math
from typing import Mapping, Sequence

import numpy as np

from .canonical_rig_ir_v2 import CanonicalRigIRV2
from .mechanics_control_solver_v1 import MechanicsPoseIRV1
from .types import QualificationError

Mat3 = tuple[tuple[float,float,float], ...]
Vec3 = tuple[float,float,float]


@dataclass(frozen=True)
class AttachmentWorldFrameV1:
    canonical_anchor_id: str
    parent_canonical_joint_id: str
    world_P: Vec3
    world_basis: Mat3
    orientation_authority: str
    angular_target_residual_rad: float | None = None


@dataclass(frozen=True)
class AttachmentWorldFrameCourtV1:
    frames: tuple[AttachmentWorldFrameV1,...]
    max_target_angular_residual_rad: float
    status: str
    metadata: dict = field(default_factory=dict)


def _matrix(raw: object, code: str) -> np.ndarray:
    try:
        basis=np.asarray(raw,dtype=np.float64)
    except (ValueError,TypeError) as exc:
        raise QualificationError(code) from exc
    if (basis.shape!=(3,3) or not np.isfinite(basis).all() or
            np.max(np.abs(basis.T@basis-np.eye(3)))>1e-6 or
            abs(float(np.linalg.det(basis))-1.0)>1e-6):
        raise QualificationError(code)
    return basis


def evaluate_attachment_world_frames_v1(
    rig: CanonicalRigIRV2,
    primary_pose: MechanicsPoseIRV1,
    *,
    required_world_orientations: Mapping[str, Sequence[Sequence[float]]] | None=None,
    angular_tolerance_rad: float=1e-3,
) -> AttachmentWorldFrameCourtV1:
    if (primary_pose.rig_lineage_hash!=rig.rig_lineage_hash or
            primary_pose.control_program_hash!=rig.control_program_hash):
        raise QualificationError("ATTACHMENT_POSE_LINEAGE_DRIFT")
    if (not math.isfinite(float(angular_tolerance_rad))
            or not 0<=angular_tolerance_rad<=math.pi):
        raise QualificationError("ATTACHMENT_TOLERANCE_POLICY_INVALID")
    joint_ids={j.canonical_joint_id for j in rig.joints}
    by_joint={j.canonical_joint_id:j for j in primary_pose.joints}
    if (len(by_joint)!=len(primary_pose.joints) or set(by_joint)!=joint_ids
            or len(joint_ids)!=len(rig.joints)):
        raise QualificationError("ATTACHMENT_POSE_JOINT_COVERAGE_DRIFT")
    anchor_ids=tuple(a.canonical_anchor_id for a in rig.attachment_anchors)
    if len(set(anchor_ids))!=len(anchor_ids):
        raise QualificationError("ATTACHMENT_DUPLICATE_ANCHOR")
    required=dict(required_world_orientations or {})
    if not set(required).issubset(set(anchor_ids)):
        raise QualificationError("ATTACHMENT_TARGET_ANCHOR_UNKNOWN")
    frames=[]
    largest=0.0
    for anchor in rig.attachment_anchors:
        if (anchor.parent_canonical_joint_id is None
                or anchor.parent_canonical_control_id is not None):
            raise QualificationError("ATTACHMENT_CONTROL_PARENT_NOT_EXECUTABLE_V1")
        if anchor.parent_canonical_joint_id not in by_joint:
            raise QualificationError("ATTACHMENT_PARENT_JOINT_MISSING")
        joint=by_joint[anchor.parent_canonical_joint_id]
        R=_matrix(joint.rotation_basis,"ATTACHMENT_JOINT_BASIS_INVALID")
        try:
            local=np.asarray(anchor.local_P,dtype=np.float64)
            world_origin=np.asarray(joint.solved_P,dtype=np.float64)
        except (ValueError,TypeError) as exc:
            raise QualificationError("ATTACHMENT_POINT_NUMERIC_INVALID") from exc
        if (local.shape!=(3,) or world_origin.shape!=(3,) or
                not np.isfinite(local).all() or not np.isfinite(world_origin).all()):
            raise QualificationError("ATTACHMENT_POINT_NUMERIC_INVALID")
        p=world_origin+R@local
        authored_basis=anchor.metadata.get("local_basis")
        if authored_basis is None:
            if anchor.canonical_anchor_id in required:
                raise QualificationError("ATTACHMENT_ORIENTATION_NOT_AUTHORED")
            basis=R
            authority="PARENT_ORIENTATION_ONLY_NOT_AUTHORED_GRIP"
        else:
            local_basis=_matrix(
                authored_basis,"ATTACHMENT_LOCAL_GRIP_BASIS_INVALID")
            basis=R@local_basis
            authority="DECLARED_LOCAL_GRIP_BASIS_RESEARCH_ONLY"
        residual=None
        if anchor.canonical_anchor_id in required:
            target=_matrix(
                required[anchor.canonical_anchor_id],
                "ATTACHMENT_TARGET_BASIS_INVALID",
            )
            relative=target.T@basis
            residual=math.acos(float(np.clip(
                (np.trace(relative)-1.0)/2.0,-1.,1.,
            )))
            largest=max(largest,residual)
        frames.append(AttachmentWorldFrameV1(
            canonical_anchor_id=anchor.canonical_anchor_id,
            parent_canonical_joint_id=anchor.parent_canonical_joint_id,
            world_P=tuple(map(float,p)),
            world_basis=tuple(tuple(map(float,row)) for row in basis),
            orientation_authority=authority,
            angular_target_residual_rad=residual,
        ))
    return AttachmentWorldFrameCourtV1(
        frames=tuple(frames),
        max_target_angular_residual_rad=largest,
        status=("PASS_ATTACHMENT_FRAME_COURT"
                if largest<=angular_tolerance_rad
                else "FAIL_ATTACHMENT_ORIENTATION_TARGET"),
        metadata={
            "authority":"MECHANICS_DECLARED_ATTACHMENT_RESEARCH_ONLY",
            "pose_origin":"PRIMARY_BEFORE_SECONDARY",
            "orientation_ik_solver_executed":False,
            "local_grip_orientation_requires_explicit_authority":True,
            "source_physics_or_graphics_claimed":False,
        },
    )


__all__=[
    "AttachmentWorldFrameV1","AttachmentWorldFrameCourtV1",
    "evaluate_attachment_world_frames_v1",
]
