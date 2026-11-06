from __future__ import annotations

"""Regression-only product-facing control→rig→skin→Motion→deform behavior.

Synthesized typed fixtures are not source truth, trained inference, or product
qualification. The point is to exercise real current Compiler consumers rather
than merely inspect modules and compare source strings.
"""

from dataclasses import replace
import math

import numpy as np
import pytest

from compiler.realsas_compiler_core.canonical_mechanics_ir_v1 import mint_canonical_mechanics_v1
from compiler.realsas_compiler_core.canonical_rig_ir_v2 import (
    CanonicalDeformJointIRV2, CanonicalJointFrameIRV2, CanonicalRigControlIRV2,
    CanonicalRigIRV2, _hash_control_program, _hash_deform_structure,
    canonical_rig_lineage_hash_v2,
)
from compiler.realsas_compiler_core.canonical_skin_ir_v2 import (
    CanonicalSkinRowIRV2, canonical_skin_lineage_hash_v2,
)
from compiler.realsas_compiler_core.canonical_motion_ir_v2 import mint_canonical_motion_v2
from compiler.realsas_compiler_core.control_motion_ir_v1 import (
    ControlMotionClipProposalIRV1, ControlMotionKeyProposalIRV1,
    ControlMotionTrackProposalIRV1, qualify_control_motion_v1,
    seal_control_motion_proposal_v1,
)
from compiler.realsas_compiler_core.control_motion_runtime_v2 import solve_motion_pose_v2
from compiler.realsas_compiler_core.mechanics_control_solver_v1 import (
    ControlTransformIRV1, deform_mesh_lbs_v1, solve_control_pose_v1,
)
from compiler.realsas_compiler_core.types import QualificationError
from tests.compiler.test_canonical_mechanics_ir_v1 import (
    _mesh as make_mesh, _skin as make_skin,
)


def fixture_pipeline(*, child_fk_first: bool = False):
    mesh = make_mesh()
    eye = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    root = CanonicalRigControlIRV2(
        "C_ROOT", "ROOT", ("J0",), None, (0.0, 0.0, 0.0), "P:ROOT",
    )
    child = CanonicalRigControlIRV2(
        "C_FK", "FK", ("J1",), "C_ROOT", (0.0, 1.0, 0.0), "P:FK",
    )
    controls = (child, root) if child_fk_first else (root, child)
    rig = CanonicalRigIRV2(
        joints=(
            CanonicalDeformJointIRV2("J0", (0.0, 0.0, 0.0), None, "P:J0"),
            CanonicalDeformJointIRV2("J1", (0.0, 1.0, 0.0), "J0", "P:J1"),
            CanonicalDeformJointIRV2("J2", (0.0, 2.0, 0.0), "J1", "P:J2"),
            CanonicalDeformJointIRV2("J3", (1.0, 0.0, 0.0), "J0", "P:J3"),
        ),
        local_frames=tuple(CanonicalJointFrameIRV2(j, eye, ("RX", "RY", "RZ")) for j in ("J0", "J1", "J2", "J3")),
        controls=controls,
        chains=(),
        constraints=(),
        attachment_anchors=(),
        deform_root_joint_ids=("J0",),
        mesh_lineage_hash=mesh.mesh_lineage_hash,
        feature_evidence_binding_hash="1" * 64,
        proposal_evidence_hash="2" * 64,
        rig_program_qualification_hash="3" * 64,
        deform_structure_hash="",
        control_program_hash="",
        rig_lineage_hash="",
        metadata={"authority": "COMPILER_CANONICAL_RIG"},
    )
    rig = replace(rig, deform_structure_hash=_hash_deform_structure(rig))
    rig = replace(rig, control_program_hash=_hash_control_program(rig))
    rig = replace(rig, rig_lineage_hash=canonical_rig_lineage_hash_v2(rig))
    old_skin = make_skin(mesh, rig)
    new_rows = (
        CanonicalSkinRowIRV2("MV:0", (("J0", 1.0),), "CV:0"),
        CanonicalSkinRowIRV2("MV:1", (("J1", 1.0),), "CV:1"),
        CanonicalSkinRowIRV2("MV:2", (("J2", 1.0),), "CV:2"),
    )
    skin = replace(old_skin, rows=new_rows, skin_lineage_hash="")
    skin = replace(skin, skin_lineage_hash=canonical_skin_lineage_hash_v2(skin))
    mechanics = mint_canonical_mechanics_v1(mesh, rig, skin)
    return mesh, rig, skin, mechanics


def positions(pose):
    return {
        row.canonical_joint_id: np.asarray(row.solved_P, dtype=np.float64)
        for row in pose.joints
    }


def test_root_drag_moves_entire_declared_subtree_and_deformed_vertices():
    mesh, rig, skin, mechanics = fixture_pipeline()
    override = ControlTransformIRV1("C_ROOT", translation=(1.0, 0.0, 0.0))
    pose = solve_control_pose_v1(rig, (override,))
    solved = positions(pose)
    for joint in rig.joints:
        assert np.allclose(solved[joint.canonical_joint_id], np.asarray(joint.P) + (1, 0, 0))
    deformed = deform_mesh_lbs_v1(mesh, rig, skin, pose)
    for original, result in zip(mesh.vertices, deformed.positions):
        assert np.allclose(result, np.asarray(original.P) + (1, 0, 0))
    assert mechanics.metadata["motion_authority_claimed"] is False


def test_child_fk_does_not_rotate_unrelated_sibling_and_overrides_root_subtree():
    _, rig, _, _ = fixture_pipeline(child_fk_first=True)
    q = (math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5))
    pose = solve_control_pose_v1(rig, (
        ControlTransformIRV1("C_FK", rotation_quat_wxyz=q),
        ControlTransformIRV1("C_ROOT", translation=(1.0, 0.0, 0.0)),
    ))
    p = positions(pose)
    assert np.allclose(p["J0"], (1, 0, 0))
    assert np.allclose(p["J1"], (1, 1, 0))
    assert np.allclose(p["J2"], (0, 1, 0), atol=1e-6)
    assert np.allclose(p["J3"], (2, 0, 0))


def test_motion_v2_control_curve_drives_the_same_deform_subtree():
    mesh, rig, skin, mechanics = fixture_pipeline()
    proposal = seal_control_motion_proposal_v1(ControlMotionClipProposalIRV1(
        proposal_clip_id="root-move",
        display_name="Root Move",
        intent="EDIT_INTERVENTION_REGRESSION",
        duration_seconds=1.0,
        loop=False,
        tracks=(ControlMotionTrackProposalIRV1(
            proposal_track_id="root", canonical_control_id="C_ROOT",
            keys=(
                ControlMotionKeyProposalIRV1(0.0, translation=(0, 0, 0)),
                ControlMotionKeyProposalIRV1(1.0, translation=(1, 0, 0)),
            ),
        ),),
        rig_lineage_hash=rig.rig_lineage_hash,
        control_program_hash=rig.control_program_hash,
        mechanics_lineage_hash=mechanics.mechanics_lineage_hash,
    ))
    motion = mint_canonical_motion_v2(
        qualify_control_motion_v1(proposal, rig=rig, mechanics=mechanics)
    )
    pose = solve_motion_pose_v2(
        motion, rig=rig, mechanics=mechanics, time_seconds=0.5,
    )
    for joint in rig.joints:
        assert np.allclose(positions(pose)[joint.canonical_joint_id], np.asarray(joint.P) + (0.5, 0, 0))
    moved = deform_mesh_lbs_v1(mesh, rig, skin, pose)
    for before, after in zip(mesh.vertices, moved.positions):
        assert np.allclose(after, np.asarray(before.P) + (0.5, 0, 0))


def test_joint_parent_cycle_is_a_hard_error_not_a_silent_repair():
    _, rig, _, _ = fixture_pipeline()
    changed_j0 = replace(rig.joints[0], parent_canonical_joint_id="J2")
    cyclic = replace(rig, joints=(changed_j0,) + rig.joints[1:])
    with pytest.raises(QualificationError, match="MECHANICS_JOINT_PARENT_CYCLE"):
        solve_control_pose_v1(cyclic, (ControlTransformIRV1("C_ROOT", translation=(1, 0, 0)),))


def test_unknown_control_or_missing_joint_parent_fails_closed():
    _, rig, _, _ = fixture_pipeline()
    with pytest.raises(QualificationError, match="MECHANICS_CONTROL_STATE_UNKNOWN"):
        solve_control_pose_v1(rig, (ControlTransformIRV1("AUTO_SYNTHESIZED"),))
    broken = replace(rig, joints=(rig.joints[0], replace(rig.joints[1], parent_canonical_joint_id="J_MISSING")) + rig.joints[2:])
    with pytest.raises(QualificationError, match="MECHANICS_JOINT_PARENT_UNKNOWN"):
        solve_control_pose_v1(broken, ())
