from __future__ import annotations

"""Canonical mechanical socket pose consumer shared by proof and Studio.

AXIS already chose each anchor's parent and local transform. The Compiler
MUST NOT infer a sword/hand relation, synthesize sockets, or repair parents.
This operator transports only explicitly decoded canonical anchors with the
EXACT same world joint/control transforms used for skinned geometry.
"""

from typing import Sequence

import numpy as np

from .canonical_rig_ir_v2 import CanonicalRigIRV2
from .mechanics_control_solver_v1 import (
    ControlTransformIRV1, MechanicsPoseIRV1, control_world_transforms_v1,
)
from .types import QualificationError


def solve_attachment_socket_world_v1(
    rig: CanonicalRigIRV2,
    pose: MechanicsPoseIRV1,
    state: Sequence[ControlTransformIRV1] = (),
) -> dict[str, tuple[float, float, float]]:
    if (
        type(rig) is not CanonicalRigIRV2
        or type(pose) is not MechanicsPoseIRV1
        or pose.rig_lineage_hash != rig.rig_lineage_hash
        or pose.control_program_hash != rig.control_program_hash
    ):
        raise QualificationError("MECHANICS_SOCKET_POSE_PARENT_DRIFT")
    joint_pose = {row.canonical_joint_id: row for row in pose.joints}
    expected_joint_ids = {joint.canonical_joint_id for joint in rig.joints}
    if len(joint_pose) != len(pose.joints) or set(joint_pose) != expected_joint_ids:
        raise QualificationError("MECHANICS_SOCKET_JOINT_COVERAGE_DRIFT")
    control_poses = control_world_transforms_v1(rig, tuple(state))
    out: dict[str, tuple[float, float, float]] = {}
    for anchor in rig.attachment_anchors:
        if anchor.canonical_anchor_id in out:
            raise QualificationError("MECHANICS_SOCKET_ID_DUPLICATE")
        local = np.asarray(anchor.local_P, dtype=np.float64)
        if local.shape != (3,) or not np.isfinite(local).all():
            raise QualificationError("MECHANICS_SOCKET_LOCAL_INVALID")
        joint_parent = anchor.parent_canonical_joint_id
        control_parent = anchor.parent_canonical_control_id
        if (joint_parent is None) == (control_parent is None):
            raise QualificationError("MECHANICS_SOCKET_PARENT_AMBIGUOUS")
        if joint_parent is not None:
            joint = joint_pose.get(joint_parent)
            if joint is None:
                raise QualificationError("MECHANICS_SOCKET_JOINT_PARENT_UNKNOWN")
            position = np.asarray(joint.solved_P, dtype=np.float64)
            rotation = np.asarray(joint.rotation_basis, dtype=np.float64)
            scale = np.asarray(joint.scale, dtype=np.float64)
        else:
            row = control_poses.get(control_parent)
            if row is None:
                raise QualificationError("MECHANICS_SOCKET_CONTROL_PARENT_UNKNOWN")
            position, rotation, scale = row
        if (
            position.shape != (3,) or rotation.shape != (3, 3)
            or scale.shape != (3,) or not np.isfinite(position).all()
            or not np.isfinite(rotation).all() or not np.isfinite(scale).all()
            or np.any(scale <= 0)
        ):
            raise QualificationError("MECHANICS_SOCKET_PARENT_TRANSFORM_INVALID")
        world = position + rotation @ (scale * local)
        if not np.isfinite(world).all():
            raise QualificationError("MECHANICS_SOCKET_WORLD_NONFINITE")
        out[anchor.canonical_anchor_id] = tuple(float(x) for x in world)
    return out


__all__ = ["solve_attachment_socket_world_v1"]
