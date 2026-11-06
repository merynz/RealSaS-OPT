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


def test_explicit_warm_start_preserves_solution_and_parameter_identity():
    rig = _branched_rig()
    first = solve_declared_multi_goal_v1(
        rig,
        (ControlTransformIRV1("A_HANDLE", translation=(-0.4, 0.8, 0.)),),
        pinned_contact_control_ids=("B_HANDLE",),
        regularization=1e-9, goal_tolerance=1e-3, max_nfev=400,
    )
    assert first.status == "PASS_MULTIGOAL_BEHAVIORAL_COURT"
    again = solve_declared_multi_goal_v1(
        rig,
        (ControlTransformIRV1("A_HANDLE", translation=(-0.4, 0.8, 0.)),),
        pinned_contact_control_ids=("B_HANDLE",),
        regularization=1e-9, goal_tolerance=1e-3, max_nfev=400,
        initial_parameters_rad=first.parameters_rad,
    )
    assert again.status == "PASS_MULTIGOAL_BEHAVIORAL_COURT"
    assert again.metadata["warm_start_supplied"] is True
    assert again.max_drag_residual < 1e-3
    assert again.max_pin_residual < 1e-3
    assert again.solver_evaluations <= first.solver_evaluations


def test_warm_start_cannot_change_dof_schema_or_violate_known_limits():
    rig = _branched_rig(root_dofs=(), right_limit=(-0.1, 0.1))
    base = solve_declared_multi_goal_v1(rig)
    params = base.parameters_rad
    assert params
    with pytest.raises(QualificationError, match="WARM_START_SCHEMA_DRIFT"):
        solve_declared_multi_goal_v1(rig, initial_parameters_rad=params[:-1])
    bad = list(params)
    jid, axis, value = bad[0]
    bad[0] = (jid, "TX", value)
    with pytest.raises(QualificationError, match="WARM_START_SCHEMA_DRIFT"):
        solve_declared_multi_goal_v1(rig, initial_parameters_rad=bad)
    bad[0] = (jid, axis, float("nan"))
    with pytest.raises(QualificationError, match="WARM_START_OUT_OF_BOUNDS"):
        solve_declared_multi_goal_v1(rig, initial_parameters_rad=bad)
    bad[0] = (jid, axis, 20.0)
    with pytest.raises(QualificationError, match="WARM_START_OUT_OF_BOUNDS"):
        solve_declared_multi_goal_v1(rig, initial_parameters_rad=bad)


@pytest.mark.parametrize("dof_count",[3,6,9,15])
def test_opt_in_analytic_jacobian_matches_numeric_goals_and_independent_central_difference(dof_count):
    rig=_branched_rig()
    # Generic label data, no teacher bone names or physics limits assumed.
    declared={
        "BASE":["RZ"],"A1":["RZ"],"A2":[],
        "B1":["RZ"],"B2":[],
    }
    extras=[
        ("BASE","RY"),("A1","RY"),("B1","RY"),
        ("BASE","RX"),("A1","RX"),("B1","RX"),
        ("A2","RZ"),("A2","RY"),("A2","RX"),
        ("B2","RZ"),("B2","RY"),("B2","RX"),
    ]
    for jid,axis in extras[:dof_count-3]:
        declared[jid].append(axis)
    rig=replace(rig,local_frames=tuple(
        replace(row,dof_labels=tuple(declared[row.canonical_joint_id]))
        for row in rig.local_frames
    ))
    angle=0.19
    edit=(ControlTransformIRV1(
        "A_HANDLE",translation=(math.cos(angle)-1.,math.sin(angle),0.),
    ),)
    baseline=solve_declared_multi_goal_v1(
        rig,edit,pinned_contact_control_ids=("B_HANDLE",),
        jacobian_mode="NUMERIC_V1",goal_tolerance=1e-3,max_nfev=400,
    )
    analytic=solve_declared_multi_goal_v1(
        rig,edit,pinned_contact_control_ids=("B_HANDLE",),
        jacobian_mode="ANALYTIC_RESEARCH_V1",verify_analytic_jacobian=True,
        goal_tolerance=1e-3,max_nfev=400,
    )
    assert analytic.status==baseline.status=="PASS_MULTIGOAL_BEHAVIORAL_COURT"
    assert analytic.max_drag_residual<1e-3
    assert analytic.max_pin_residual<1e-3
    assert analytic.metadata["jacobian_mode"]=="ANALYTIC_RESEARCH_V1"
    assert analytic.metadata["analytic_jacobian_finite_difference_max_error"]<1e-6
    assert abs(analytic.max_pin_residual-baseline.max_pin_residual)<1e-3
    assert abs(analytic.max_drag_residual-baseline.max_drag_residual)<1e-3
    assert len(analytic.parameters_rad)==dof_count


def test_opt_in_analytic_jacobian_fails_closed_for_unsupported_mode_and_court():
    rig=_branched_rig()
    with pytest.raises(QualificationError,match="JACOBIAN_MODE_UNSUPPORTED"):
        solve_declared_multi_goal_v1(rig,jacobian_mode="JAX_AUTOMATIC")
    with pytest.raises(QualificationError,match="JACOBIAN_COURT_REQUIRES_ANALYTIC"):
        solve_declared_multi_goal_v1(rig,verify_analytic_jacobian=True)


def test_analytic_jacobian_respects_declared_angle_limit_and_unreachable_goal():
    rig=_branched_rig(root_dofs=(),right_limit=(-0.1,0.1))
    result=solve_declared_multi_goal_v1(
        rig,
        (ControlTransformIRV1("A_HANDLE",translation=(-0.4,0.8,0.)),),
        pinned_contact_control_ids=("B_HANDLE",),
        jacobian_mode="ANALYTIC_RESEARCH_V1",
        verify_analytic_jacobian=True,
        max_nfev=150,regularization=1e-10,
    )
    assert result.status=="FAIL_MULTIGOAL_BEHAVIORAL_COURT"
    assert result.max_drag_residual>0.4
    q={f"{jid}:{axis}":angle for jid,axis,angle in result.parameters_rad}
    assert abs(q["A1:RZ"])<=.1000000001
    assert result.metadata["analytic_jacobian_finite_difference_max_error"]<1e-6
