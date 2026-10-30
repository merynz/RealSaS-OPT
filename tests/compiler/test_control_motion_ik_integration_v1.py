from __future__ import annotations

import numpy as np

from compiler.realsas_compiler_core.canonical_mechanics_ir_v1 import CanonicalMechanicsIRV1
from compiler.realsas_compiler_core.canonical_motion_ir_v2 import mint_canonical_motion_v2
from compiler.realsas_compiler_core.canonical_rig_ir_v2 import (
    CanonicalDeformJointIRV2,
    CanonicalJointFrameIRV2,
    CanonicalRigChainIRV2,
    CanonicalRigConstraintIRV2,
    CanonicalRigControlIRV2,
    CanonicalRigIRV2,
)
from compiler.realsas_compiler_core.control_motion_ir_v1 import (
    ControlMotionClipProposalIRV1,
    ControlMotionKeyProposalIRV1,
    ControlMotionTrackProposalIRV1,
    qualify_control_motion_v1,
    seal_control_motion_proposal_v1,
)
from compiler.realsas_compiler_core.control_motion_runtime_v2 import solve_motion_pose_v2


def _rig() -> CanonicalRigIRV2:
    identity = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    return CanonicalRigIRV2(
        joints=(
            CanonicalDeformJointIRV2("J0", (0.0, 0.0, 0.0), None, "PJ0"),
            CanonicalDeformJointIRV2("J1", (0.0, 1.0, 0.0), "J0", "PJ1"),
            CanonicalDeformJointIRV2("J2", (0.0, 2.0, 0.0), "J1", "PJ2"),
        ),
        local_frames=(
            CanonicalJointFrameIRV2("J0", identity, ("RX", "RY", "RZ")),
            CanonicalJointFrameIRV2("J1", identity, ("RX", "RY", "RZ")),
            CanonicalJointFrameIRV2("J2", identity, ("RX", "RY", "RZ")),
        ),
        controls=(
            CanonicalRigControlIRV2("C_IK", "IK_EFFECTOR", ("J2",), None, (0.0, 2.0, 0.0), "PC0"),
            CanonicalRigControlIRV2("C_POLE", "IK_POLE", (), None, (1.0, 1.0, 0.0), "PC1"),
        ),
        chains=(
            CanonicalRigChainIRV2(
                "K0", ("J0", "J1", "J2"), "GENERIC_KINEMATIC_CHAIN", "PK0"
            ),
        ),
        constraints=(
            CanonicalRigConstraintIRV2(
                "Q0",
                "IK",
                ("C_IK", "C_POLE"),
                ("J2",),
                "K0",
                {"p0": 1.0, "p1": 1.0, "p2": 1.0},
                "PQ0",
            ),
        ),
        attachment_anchors=(),
        deform_root_joint_ids=("J0",),
        mesh_lineage_hash="1" * 64,
        feature_evidence_binding_hash="2" * 64,
        proposal_evidence_hash="3" * 64,
        rig_program_qualification_hash="4" * 64,
        deform_structure_hash="5" * 64,
        control_program_hash="6" * 64,
        rig_lineage_hash="7" * 64,
        metadata={"authority": "COMPILER_CANONICAL_RIG"},
    )


def _mechanics(rig: CanonicalRigIRV2) -> CanonicalMechanicsIRV1:
    return CanonicalMechanicsIRV1(
        mesh_lineage_hash=rig.mesh_lineage_hash,
        rig_lineage_hash=rig.rig_lineage_hash,
        deform_structure_hash=rig.deform_structure_hash,
        control_program_hash=rig.control_program_hash,
        skin_lineage_hash="8" * 64,
        constraint_schedule=("Q0",),
        executable_operator_ids=("IK",),
        rest_pose_hash="9" * 64,
        qualification_hash="a" * 64,
        mechanics_lineage_hash="b" * 64,
        metadata={"authority": "COMPILER_CANONICAL_MECHANICS"},
    )


def _motion(rig: CanonicalRigIRV2, mechanics: CanonicalMechanicsIRV1, *, pole_z: float):
    target = (1.0, 1.5, 0.0)
    pole = (0.0, 1.0, pole_z)
    proposal = seal_control_motion_proposal_v1(
        ControlMotionClipProposalIRV1(
            proposal_clip_id="ik-reach",
            display_name="IK Reach",
            intent="MECHANICS_REGRESSION",
            duration_seconds=1.0,
            loop=False,
            tracks=(
                ControlMotionTrackProposalIRV1(
                    proposal_track_id="ik-effector-track",
                    canonical_control_id="C_IK",
                    keys=(
                        ControlMotionKeyProposalIRV1(0.0),
                        ControlMotionKeyProposalIRV1(
                            1.0,
                            translation=(target[0], target[1] - 2.0, target[2]),
                        ),
                    ),
                ),
                ControlMotionTrackProposalIRV1(
                    proposal_track_id="ik-pole-track",
                    canonical_control_id="C_POLE",
                    keys=(
                        ControlMotionKeyProposalIRV1(0.0),
                        ControlMotionKeyProposalIRV1(
                            1.0,
                            translation=(pole[0] - 1.0, pole[1] - 1.0, pole[2]),
                        ),
                    ),
                ),
            ),
            rig_lineage_hash=rig.rig_lineage_hash,
            control_program_hash=rig.control_program_hash,
            mechanics_lineage_hash=mechanics.mechanics_lineage_hash,
        )
    )
    qualified = qualify_control_motion_v1(proposal, rig=rig, mechanics=mechanics)
    return mint_canonical_motion_v2(qualified), np.asarray(target, dtype=np.float64)


def _solve(pole_z: float):
    rig = _rig()
    mechanics = _mechanics(rig)
    motion, target = _motion(rig, mechanics, pole_z=pole_z)
    pose = solve_motion_pose_v2(
        motion,
        rig=rig,
        mechanics=mechanics,
        time_seconds=1.0,
    )
    by = {row.canonical_joint_id: np.asarray(row.solved_P) for row in pose.joints}
    return pose, by, target


def test_control_motion_effector_and_pole_execute_through_mechanics_constraint_program():
    positive, pos, target = _solve(1.0)
    negative, neg, _ = _solve(-1.0)

    # Motion owns only control tracks; Mechanics owns the IK solution.
    assert np.linalg.norm(pos["J2"] - target) < 1.0e-4
    assert np.linalg.norm(neg["J2"] - target) < 1.0e-4
    assert abs(np.linalg.norm(pos["J1"] - pos["J0"]) - 1.0) < 1.0e-4
    assert abs(np.linalg.norm(pos["J2"] - pos["J1"]) - 1.0) < 1.0e-4

    # Flipping the pole must flip the bend side without moving the effector.
    assert pos["J1"][2] > 1.0e-3
    assert neg["J1"][2] < -1.0e-3
    assert np.linalg.norm(pos["J2"] - neg["J2"]) < 1.0e-4

    assert positive.metadata["authority"] == "MECHANICS_CONSTRAINT_EXECUTION"
    assert negative.metadata["authority"] == "MECHANICS_CONSTRAINT_EXECUTION"
    assert positive.constraint_residuals[0][1] < 1.0e-4
    assert negative.constraint_residuals[0][1] < 1.0e-4
