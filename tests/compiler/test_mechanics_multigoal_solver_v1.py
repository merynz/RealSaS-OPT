from __future__ import annotations

"""Behavioral reference court for mechanics multi-goal articulation, not unseen."""

from dataclasses import replace
import math

import numpy as np
import pytest

from compiler.realsas_compiler_core.canonical_rig_ir_v2 import (
    CanonicalDeformJointIRV2,
    CanonicalJointFrameIRV2,
    CanonicalRigChainIRV2,
    CanonicalRigConstraintIRV2,
    CanonicalRigControlIRV2,
    CanonicalRigIRV2,
)
from compiler.realsas_compiler_core.mechanics_control_solver_v1 import (
    ControlTransformIRV1,
)
from compiler.realsas_compiler_core.mechanics_multigoal_solver_v1 import (
    solve_declared_multi_goal_v1,
)
from compiler.realsas_compiler_core.types import QualificationError


I = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def _branched_rig(*, root_dofs=("RZ",), right_limit=None):
    # Two independent branched two-segment links (not humanoid semantics).
    joints = (
        CanonicalDeformJointIRV2("BASE", (0., 0., 0.), None, "s0"),
        CanonicalDeformJointIRV2("A1", (1., 0., 0.), "BASE", "s1"),
        CanonicalDeformJointIRV2("A2", (2., 0., 0.), "A1", "s2"),
        CanonicalDeformJointIRV2("B1", (-1., 0., 0.), "BASE", "s3"),
        CanonicalDeformJointIRV2("B2", (-2., 0., 0.), "B1", "s4"),
    )
    limit_metadata = (
        {"rotation_limits_rad": {"RZ": right_limit}} if right_limit else {}
    )
    frames = (
        CanonicalJointFrameIRV2("BASE", I, root_dofs),
        CanonicalJointFrameIRV2("A1", I, ("RZ",), limit_metadata),
        CanonicalJointFrameIRV2("A2", I, ()),
        CanonicalJointFrameIRV2("B1", I, ("RZ",)),
        CanonicalJointFrameIRV2("B2", I, ()),
    )
    controls = (
        CanonicalRigControlIRV2("A_HANDLE", "IK_EFFECTOR", ("A2",), None, (2., 0., 0.), "ca"),
        CanonicalRigControlIRV2("B_HANDLE", "IK_EFFECTOR", ("B2",), None, (-2., 0., 0.), "cb"),
    )
    constraints = (
        CanonicalRigConstraintIRV2("A_IK", "IK", ("A_HANDLE",), ("A2",), "CHAIN_A",
                                   {"p0": 1., "p2": 1.}, "qa"),
        CanonicalRigConstraintIRV2("B_IK", "IK", ("B_HANDLE",), ("B2",), "CHAIN_B",
                                   {"p0": 1., "p2": 1.}, "qb"),
    )
    return CanonicalRigIRV2(
        joints=joints, local_frames=frames, controls=controls,
        chains=(
            CanonicalRigChainIRV2("CHAIN_A", ("BASE", "A1", "A2"), "GENERIC_KINEMATIC_CHAIN", "ka"),
            CanonicalRigChainIRV2("CHAIN_B", ("BASE", "B1", "B2"), "GENERIC_KINEMATIC_CHAIN", "kb"),
        ),
        constraints=constraints, attachment_anchors=(), deform_root_joint_ids=("BASE",),
        mesh_lineage_hash="1"*64, feature_evidence_binding_hash="2"*64,
        proposal_evidence_hash="3"*64, rig_program_qualification_hash="4"*64,
        deform_structure_hash="5"*64, control_program_hash="6"*64,
        rig_lineage_hash="7"*64, metadata={"authority": "COMPILER_CANONICAL_RIG"},
    )


def test_two_simultaneous_declared_targets_pin_b_while_dragging_a():
    rig = _branched_rig()
    result = solve_declared_multi_goal_v1(
        rig,
        (ControlTransformIRV1("A_HANDLE", translation=(-0.4, 0.8, 0.)),),
        pinned_contact_control_ids=("B_HANDLE",),
        goal_tolerance=1e-3,
        regularization=1e-9, max_nfev=400,
    )
    assert result.status == "PASS_MULTIGOAL_BEHAVIORAL_COURT", result
    assert result.max_drag_residual < 1e-3
    assert result.max_pin_residual < 1e-3
    by = {j.canonical_joint_id: np.asarray(j.solved_P) for j in result.pose.joints}
    assert np.allclose(by["A2"], (1.6, 0.8, 0.), atol=1e-3)
    assert np.allclose(by["B2"], (-2., 0., 0.), atol=1e-3)
    for j in rig.joints:
        if j.parent_canonical_joint_id is None:
            continue
        parent = next(x for x in rig.joints if x.canonical_joint_id == j.parent_canonical_joint_id)
        assert np.linalg.norm(by[j.canonical_joint_id] - by[parent.canonical_joint_id]) == pytest.approx(
            np.linalg.norm(np.asarray(j.P) - np.asarray(parent.P)), abs=1e-9
        )
    assert all(g.goal_kind in {"DRAG", "PIN"} for g in result.goals)
    assert result.pose.metadata["graphics_authority_claimed"] is False


def test_declared_angle_limit_is_enforced_and_unreachable_goal_not_certified():
    rig = _branched_rig(root_dofs=(), right_limit=(-0.1, 0.1))
    result = solve_declared_multi_goal_v1(
        rig, (ControlTransformIRV1("A_HANDLE", translation=(-0.4, 0.8, 0.)),),
        pinned_contact_control_ids=("B_HANDLE",),
        max_nfev=150, regularization=1e-10,
    )
    assert result.status == "FAIL_MULTIGOAL_BEHAVIORAL_COURT"
    q = {f"{jid}:{axis}": angle for jid, axis, angle in result.parameters_rad}
    assert abs(q["A1:RZ"]) <= 0.1000000001
    assert result.metadata["declared_angle_limit_count"] == 1
    assert result.max_drag_residual > 0.4


def test_unreachable_drag_produces_explicit_failure_with_finite_pose():
    rig = _branched_rig()
    result = solve_declared_multi_goal_v1(
        rig, (ControlTransformIRV1("A_HANDLE", translation=(4.0, 0., 0.)),),
        pinned_contact_control_ids=("B_HANDLE",), max_nfev=60,
    )
    assert result.status == "FAIL_MULTIGOAL_BEHAVIORAL_COURT"
    assert result.max_drag_residual > 3.5
    assert all(math.isfinite(float(v)) for row in result.pose.joints for v in row.solved_P)


def test_fail_closed_on_unsupported_operator_and_nonrotational_dofs():
    rig = _branched_rig()
    invalid_operator = replace(
        rig,
        constraints=(replace(rig.constraints[0], operator_id="PATH"), rig.constraints[1]),
    )
    with pytest.raises(QualificationError, match="OPERATOR_UNSUPPORTED"):
        solve_declared_multi_goal_v1(invalid_operator)
    invalid_dof = replace(
        rig,
        local_frames=(replace(rig.local_frames[0], dof_labels=("TX",)),)
        + rig.local_frames[1:],
    )
    with pytest.raises(QualificationError, match="UNSUPPORTED_DOF"):
        solve_declared_multi_goal_v1(invalid_dof)
    with pytest.raises(QualificationError, match="PIN_HAS_OVERRIDE"):
        solve_declared_multi_goal_v1(
            rig, (ControlTransformIRV1("B_HANDLE"),),
            pinned_contact_control_ids=("B_HANDLE",),
        )


def test_rest_pose_is_exact_and_source_independent():
    rig = _branched_rig()
    result = solve_declared_multi_goal_v1(rig, pinned_contact_control_ids=("B_HANDLE",))
    assert result.status == "PASS_MULTIGOAL_BEHAVIORAL_COURT"
    for j, original in zip(result.pose.joints, rig.joints):
        assert np.allclose(j.solved_P, original.P, atol=1e-8)
    assert result.metadata["training_authority_minted"] is False
    assert result.metadata["product_mechanics_release_minted"] is False
