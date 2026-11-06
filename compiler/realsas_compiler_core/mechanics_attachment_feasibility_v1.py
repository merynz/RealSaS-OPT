from __future__ import annotations

"""Necessary-condition court for simultaneous declared 3D attachment targets.

Only exact rigid kinematics *on the same declared parent joint* are tested.
PASS does NOT mean that a rig has enough DOFs, that IK will converge, or
that different parents' frames are simultaneously reachable.

This court never invents attachment basis, changes CanonicalRig, imports
Graphics knowledge, or infers source-world physical joint limits.
"""

from dataclasses import dataclass, field
import math
from typing import Mapping, Sequence

import numpy as np

from .canonical_rig_ir_v2 import CanonicalRigIRV2
from .mechanics_attachment_frames_v1 import _matrix
from .types import QualificationError


@dataclass(frozen=True)
class AttachmentRigidConflictWitnessV1:
    anchor_ids: tuple[str, str]
    shared_parent_joint_id: str
    constraint_kind: str
    measured_mismatch: float
    allowable_mismatch: float
    measurement_unit: str
    explanation: str


@dataclass(frozen=True)
class AttachmentRigidFeasibilityCourtV1:
    status: str
    hard_anchor_ids: tuple[str, ...]
    witnesses: tuple[AttachmentRigidConflictWitnessV1, ...]
    metadata: dict = field(default_factory=dict)


def _position(value: object, label: str) -> np.ndarray:
    try:
        p=np.asarray(value,dtype=np.float64)
    except (ValueError,TypeError) as exc:
        raise QualificationError(label) from exc
    if p.shape!=(3,) or not np.isfinite(p).all():
        raise QualificationError(label)
    return p


def _geodesic(a: np.ndarray, b: np.ndarray) -> float:
    return math.acos(float(np.clip(
        (np.trace(a.T@b)-1.)/2.,-1.,1.,
    )))


def evaluate_declared_attachment_rigidity_v1(
    rig: CanonicalRigIRV2,
    *,
    required_world_positions: Mapping[str, Sequence[float]] | None = None,
    required_world_orientations: Mapping[str, Sequence[Sequence[float]]] | None = None,
    position_tolerance_world: float = 1e-3,
    orientation_tolerance_rad: float = 1e-3,
    maximum_anchors: int = 32,
) -> AttachmentRigidFeasibilityCourtV1:
    if (not math.isfinite(float(position_tolerance_world))
            or position_tolerance_world <= 0):
        raise QualificationError("ATTACHMENT_RIGIDITY_POSITION_TOLERANCE_INVALID")
    if (not math.isfinite(float(orientation_tolerance_rad))
            or not 0 < orientation_tolerance_rad <= math.pi):
        raise QualificationError("ATTACHMENT_RIGIDITY_ORIENTATION_TOLERANCE_INVALID")
    if not isinstance(maximum_anchors,int) or not 1 <= maximum_anchors <= 64:
        raise QualificationError("ATTACHMENT_RIGIDITY_RESOURCE_POLICY_INVALID")

    positions=dict(required_world_positions or {})
    orientations=dict(required_world_orientations or {})
    wanted=set(positions)|set(orientations)
    anchors={a.canonical_anchor_id:a for a in rig.attachment_anchors}
    joints={j.canonical_joint_id for j in rig.joints}
    if len(anchors)!=len(rig.attachment_anchors) or len(joints)!=len(rig.joints):
        raise QualificationError("ATTACHMENT_RIGIDITY_DUPLICATE_RIG_ID")
    if len(wanted)>maximum_anchors:
        raise QualificationError("ATTACHMENT_RIGIDITY_ANCHOR_RESOURCE_BOUND")
    if wanted-set(anchors):
        raise QualificationError("ATTACHMENT_RIGIDITY_UNKNOWN_ANCHOR")

    p_world={k:_position(v,"ATTACHMENT_RIGIDITY_WORLD_POSITION_INVALID")
             for k,v in positions.items()}
    R_world={k:_matrix(v,"ATTACHMENT_RIGIDITY_WORLD_ORIENTATION_INVALID")
             for k,v in orientations.items()}
    local_P={}
    local_R={}
    parent={}
    for aid in sorted(wanted):
        a=anchors[aid]
        if (a.parent_canonical_joint_id not in joints
                or a.parent_canonical_control_id is not None):
            raise QualificationError("ATTACHMENT_RIGIDITY_UNSUPPORTED_CONTROL_PARENT")
        local_P[aid]=_position(
            a.local_P,"ATTACHMENT_RIGIDITY_LOCAL_POSITION_INVALID"
        )
        parent[aid]=a.parent_canonical_joint_id
        if aid in R_world:
            if "local_basis" not in a.metadata:
                raise QualificationError("ATTACHMENT_RIGIDITY_UNDECLARED_LOCAL_ORIENTATION")
            local_R[aid]=_matrix(
                a.metadata["local_basis"],
                "ATTACHMENT_RIGIDITY_LOCAL_ORIENTATION_INVALID",
            )

    witnesses=[]
    ids=tuple(sorted(wanted))
    for i,a in enumerate(ids):
        for b in ids[i+1:]:
            if parent[a]!=parent[b]:
                continue
            same_parent=parent[a]
            delta=local_P[b]-local_P[a]
            # Distances between points rigidly attached to the SAME joint
            # do not change under any parent world transform. If two
            # tolerated target spheres cannot contain that distance, no
            # full-body IK solve can succeed.
            if a in p_world and b in p_world:
                distance_error=abs(
                    float(np.linalg.norm(p_world[b]-p_world[a]))
                    -float(np.linalg.norm(delta))
                )
                permitted=2.*position_tolerance_world
                if distance_error>permitted+1e-12:
                    witnesses.append(AttachmentRigidConflictWitnessV1(
                        (a,b),same_parent,"RIGID_POINT_DISTANCE",
                        distance_error,permitted,"world_distance",
                        "Same-parent attachment points cannot change pairwise distance.",
                    ))
            # Each independently authored orientation target determines
            # the SAME parent world rotation, up to tolerance.
            if a in R_world and b in R_world:
                parent_R_a=R_world[a]@local_R[a].T
                parent_R_b=R_world[b]@local_R[b].T
                orientation_error=_geodesic(parent_R_a,parent_R_b)
                permitted=2.*orientation_tolerance_rad
                if orientation_error>permitted+1e-10:
                    witnesses.append(AttachmentRigidConflictWitnessV1(
                        (a,b),same_parent,"RIGID_PARENT_ORIENTATION",
                        orientation_error,permitted,"rad",
                        "Two oriented anchors on one parent demand incompatible parent rotations.",
                    ))
            # One authored orientation plus two point targets predicts the
            # rigid displacement of the second point. Include an angular
            # tolerance radius, rather than falsely rejecting nearby poses.
            if a in p_world and b in p_world:
                for oriented in (a,b):
                    if oriented not in R_world:
                        continue
                    known=R_world[oriented]@local_R[oriented].T
                    predicted=known@delta
                    measured=p_world[b]-p_world[a]
                    mismatch=float(np.linalg.norm(measured-predicted))
                    permitted=(
                        2.*position_tolerance_world
                        +2.*float(np.linalg.norm(delta))
                            *math.sin(orientation_tolerance_rad/2.)
                    )
                    if mismatch>permitted+1e-12:
                        witnesses.append(AttachmentRigidConflictWitnessV1(
                            (a,b),same_parent,
                            "RIGID_FRAME_POINT_DISPLACEMENT",
                            mismatch,permitted,"world_distance",
                            "Declared parent orientation and rigid point displacement disagree.",
                        ))
    return AttachmentRigidFeasibilityCourtV1(
        status=("FAIL_DECLARED_RIGID_ATTACHMENT_CONFLICT"
                if witnesses else "NO_PAIRWISE_RIGID_CONFLICT_GLOBAL_IK_UNDETERMINED"),
        hard_anchor_ids=ids,
        witnesses=tuple(witnesses),
        metadata={
            "authority":"DECLARED_ATTACHMENT_RESEARCH_NECESSARY_CONDITION_ONLY",
            "rig_lineage_hash":rig.rig_lineage_hash,
            "control_program_hash":rig.control_program_hash,
            "pairwise_tested_anchor_count":len(ids),
            "witness_count":len(witnesses),
            "world_position_target_count":len(positions),
            "world_orientation_target_count":len(orientations),
            "per_parent_rigid_transform_only":True,
            "global_ik_feasibility_certified":False,
            "attachment_priorities_assumed":"ALL_PROVIDED_TARGETS_HARD",
            "source_physical_truth_minted":False,
            "graphics_or_model_authority_minted":False,
        },
    )


__all__=[
    "AttachmentRigidConflictWitnessV1",
    "AttachmentRigidFeasibilityCourtV1",
    "evaluate_declared_attachment_rigidity_v1",
]
