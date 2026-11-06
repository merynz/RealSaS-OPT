from __future__ import annotations

"""Full 3D attachment orientation-IK research against declared source-free rig.

Proves the solver can simultaneously retain a world foot-contact position,
move a distal tip, and orient a declared 3D grip frame. Synthetic only:
this does not invent source-authored DOF/limits or real sword grip truth.
"""

from dataclasses import replace
import math

import numpy as np
import pytest

from compiler.realsas_compiler_core.canonical_rig_ir_v2 import CanonicalAttachmentAnchorIRV2
from compiler.realsas_compiler_core.mechanics_control_solver_v1 import ControlTransformIRV1
from compiler.realsas_compiler_core.mechanics_multigoal_solver_v1 import solve_declared_multi_goal_v1
from compiler.realsas_compiler_core.mechanics_live_runtime_v1 import evaluate_mechanics_intervention_frame_v1
from compiler.realsas_compiler_core.mechanics_attachment_frames_v1 import evaluate_attachment_world_frames_v1
from compiler.realsas_compiler_core.types import QualificationError
from tests.compiler.test_mechanics_multigoal_solver_v1 import _branched_rig
from tests.compiler.test_mechanics_live_multigoal_integration_v1 import _objects


I=((1.,0.,0.),(0.,1.,0.),(0.,0.,1.))


def _rz(rad:float):
    c,s=math.cos(rad),math.sin(rad)
    return ((c,-s,0.),(s,c,0.),(0.,0.,1.))


def _orientation_rig():
    rig=_branched_rig(root_dofs=())
    return replace(rig,attachment_anchors=(CanonicalAttachmentAnchorIRV2(
        canonical_anchor_id="GRIP:A",
        parent_canonical_joint_id="A1",
        local_P=(1.,0.,0.),
        source_proposal_anchor_id="SYNTHETIC_AUTHORED_RESEARCH_ONLY",
        metadata={"local_basis":I,"source_orientation_proven":False},
    ),))


def _goals(theta:float=.3):
    return (ControlTransformIRV1(
        "A_HANDLE",translation=(math.cos(theta)-1.,math.sin(theta),0.),
    ),)


def test_orientation_ik_changes_declared_joint_and_keeps_world_position_contact():
    rig=_orientation_rig()
    theta=.3
    # Important: root does not have rotational DOF. The A1 rotation
    # independently matches both the A2 tip and declared grip orientation;
    # B1/B2 contact stays planted at (-2,0,0).
    solved=solve_declared_multi_goal_v1(
        rig,_goals(theta),
        pinned_contact_control_ids=("B_HANDLE",),
        pinned_world_targets={"B_HANDLE":(-2.,0.,0.)},
        attachment_orientation_targets={"GRIP:A":_rz(theta)},
        jacobian_mode="ANALYTIC_RESEARCH_V1",
        verify_analytic_jacobian=True,
        regularization=1e-9,
        goal_tolerance=1e-3,
        orientation_tolerance_rad=1e-3,
    )
    assert solved.status=="PASS_MULTIGOAL_BEHAVIORAL_COURT"
    assert solved.max_orientation_residual_rad<1e-3
    assert solved.max_pin_residual<1e-3
    assert solved.max_drag_residual<1e-3
    assert solved.metadata["analytic_jacobian_finite_difference_max_error"]<1e-6
    assert len(solved.orientation_goals)==1
    pose=solved.pose
    court=evaluate_attachment_world_frames_v1(
        rig,pose,required_world_orientations={"GRIP:A":_rz(theta)},
    )
    assert court.status=="PASS_ATTACHMENT_FRAME_COURT"
    assert np.allclose(court.frames[0].world_P,
                       (1+math.cos(theta),math.sin(theta),0),atol=1e-3)
    by={j.canonical_joint_id:j for j in pose.joints}
    assert np.allclose(by["B2"].solved_P,(-2.,0.,0.),atol=1e-4)
    assert solved.metadata["declared_attachment_orientation_goal_count"]==1


def test_conflicting_grip_orientation_and_tip_drag_fail_without_fabrication():
    rig=_orientation_rig()
    t=.3
    solved=solve_declared_multi_goal_v1(
        rig,_goals(t),
        pinned_contact_control_ids=("B_HANDLE",),
        pinned_world_targets={"B_HANDLE":(-2.,0.,0.)},
        attachment_orientation_targets={"GRIP:A":_rz(-.8)},
        jacobian_mode="ANALYTIC_RESEARCH_V1",
        goal_tolerance=1e-3,
        orientation_tolerance_rad=1e-3,
        regularization=1e-10,
    )
    assert solved.status=="FAIL_MULTIGOAL_BEHAVIORAL_COURT"
    assert (solved.max_orientation_residual_rad>1e-3
            or solved.max_drag_residual>1e-3)
    assert solved.pose.metadata["rig_program_mutated"] is False


def test_unprovided_or_invalid_grip_orientation_is_not_source_truth():
    rig=_orientation_rig()
    missing=replace(rig,attachment_anchors=(
        replace(rig.attachment_anchors[0],metadata={}),
    ))
    for kwargs,reason in (
        ({"attachment_orientation_targets":{"MISSING":I}},"ANCHOR_UNKNOWN"),
        ({"attachment_orientation_targets":{"GRIP:A":((2.,0.,0.),(0.,1.,0.),(0.,0.,1.))}},"TARGET_BASIS_INVALID"),
    ):
        with pytest.raises(QualificationError,match=reason):
            solve_declared_multi_goal_v1(
                rig,_goals(.3),jacobian_mode="ANALYTIC_RESEARCH_V1",**kwargs,
            )
    with pytest.raises(QualificationError,match="UNDECLARED_GRIP_BASIS"):
        solve_declared_multi_goal_v1(
            missing,_goals(.3),
            attachment_orientation_targets={"GRIP:A":I},
            jacobian_mode="ANALYTIC_RESEARCH_V1",
        )
    with pytest.raises(QualificationError,match="ORIENTATION_REQUIRES_ANALYTIC"):
        solve_declared_multi_goal_v1(
            rig,_goals(.3),
            attachment_orientation_targets={"GRIP:A":I},
        )


def test_live_orientation_ik_is_explicit_opt_in_and_default_remains_evaluator():
    mesh,rig,skin,mechanics=_objects()
    rig=replace(rig,attachment_anchors=(CanonicalAttachmentAnchorIRV2(
        "GRIP:A",parent_canonical_joint_id="A1",local_P=(1.,0.,0.),
        source_proposal_anchor_id="SYNTHETIC_RESEARCH_FIXTURE",
        metadata={"local_basis":I},
    ),))
    angle=.28
    opts=dict(
        overrides=_goals(angle),
        solver_mode="DECLARED_MULTI_GOAL_V1",
        pinned_contact_control_ids=("B_HANDLE",),
        pinned_world_targets={"B_HANDLE":(-2.,0.,0.)},
        multi_goal_jacobian_mode="ANALYTIC_RESEARCH_V1",
        required_attachment_world_orientations={"GRIP:A":_rz(angle)},
        max_allowed_constraint_residual=1e-3,
    )
    solved=evaluate_mechanics_intervention_frame_v1(
        mesh,rig,skin,mechanics,
        attachment_orientation_mode="SOLVE_DECLARED_RESEARCH_V1",**opts,
    )
    assert solved.status=="PASS_MECHANICS_BEHAVIORAL_COURT"
    assert solved.attachment_frames.status=="PASS_ATTACHMENT_FRAME_COURT"
    assert solved.metadata["attachment_orientation_mode"]=="SOLVE_DECLARED_RESEARCH_V1"
    default=evaluate_mechanics_intervention_frame_v1(
        mesh,rig,skin,mechanics,
        attachment_orientation_mode="EVALUATE_ONLY",**opts,
    )
    # A coincidental match is not forbidden, but evaluator doesn't claim
    # to run or synthesize an orientation solver.
    assert default.metadata["attachment_orientation_mode"]=="EVALUATE_ONLY"
    assert default.metadata["attachment_orientation_solver_is_opt_in"] is True


def test_live_orientation_mode_requires_analytical_multigoal_and_target():
    mesh,rig,skin,mechanics=_objects()
    with pytest.raises(QualificationError,match="ORIENTATION_SOLVE_REQUIRES_DECLARED"):
        evaluate_mechanics_intervention_frame_v1(
            mesh,rig,skin,mechanics,
            attachment_orientation_mode="SOLVE_DECLARED_RESEARCH_V1",
        )
    with pytest.raises(QualificationError,match="ATTACHMENT_ORIENTATION_MODE_UNSUPPORTED"):
        evaluate_mechanics_intervention_frame_v1(
            mesh,rig,skin,mechanics,attachment_orientation_mode="INFER_SWORD_GRIP",
        )


def test_full_six_dimensional_grip_frame_meets_position_orientation_and_planted_contact():
    rig=_orientation_rig()
    theta=.31
    grip_target=(1.+math.cos(theta),math.sin(theta),0.)
    solved=solve_declared_multi_goal_v1(
        rig,_goals(theta),
        pinned_contact_control_ids=("B_HANDLE",),
        pinned_world_targets={"B_HANDLE":(-2.,0.,0.)},
        attachment_position_targets={"GRIP:A":grip_target},
        attachment_orientation_targets={"GRIP:A":_rz(theta)},
        jacobian_mode="ANALYTIC_RESEARCH_V1",
        verify_analytic_jacobian=True,
        regularization=1e-9,goal_tolerance=1e-3,
        orientation_tolerance_rad=1e-3,
    )
    assert solved.status=="PASS_MULTIGOAL_BEHAVIORAL_COURT"
    assert solved.max_orientation_residual_rad<1e-3
    assert solved.max_attachment_position_residual_world<1e-3
    assert solved.max_pin_residual<1e-3
    assert solved.max_drag_residual<1e-3
    assert solved.metadata["analytic_jacobian_finite_difference_max_error"]<1e-6
    assert solved.metadata["declared_attachment_position_goal_count"]==1
    assert len(solved.attachment_position_goals)==1
    checked=evaluate_attachment_world_frames_v1(
        rig,solved.pose,
        required_world_positions={"GRIP:A":grip_target},
        required_world_orientations={"GRIP:A":_rz(theta)},
    )
    assert checked.status=="PASS_ATTACHMENT_FRAME_COURT"
    assert checked.max_target_position_residual_world<1e-3
    assert checked.max_target_angular_residual_rad<1e-3


def test_contradictory_full_frame_position_fails_even_when_tip_and_rotation_are_satisfied():
    rig=_orientation_rig()
    theta=.3
    solved=solve_declared_multi_goal_v1(
        rig,_goals(theta),
        pinned_contact_control_ids=("B_HANDLE",),
        pinned_world_targets={"B_HANDLE":(-2.,0.,0.)},
        attachment_position_targets={"GRIP:A":(20.,0.,0.)},
        attachment_orientation_targets={"GRIP:A":_rz(theta)},
        jacobian_mode="ANALYTIC_RESEARCH_V1",
        regularization=1e-9,goal_tolerance=1e-3,
        orientation_tolerance_rad=1e-3,
    )
    assert solved.status=="FAIL_MULTIGOAL_BEHAVIORAL_COURT"
    assert solved.max_attachment_position_residual_world>1.
    assert solved.metadata["product_mechanics_release_minted"] is False


def test_position_only_attachment_frame_court_needs_no_fabricated_orientation():
    rig=_orientation_rig()
    unqualified=replace(rig,attachment_anchors=(
        replace(rig.attachment_anchors[0],metadata={}),
    ))
    solved=solve_declared_multi_goal_v1(
        unqualified,_goals(.25),
        pinned_contact_control_ids=("B_HANDLE",),
        pinned_world_targets={"B_HANDLE":(-2.,0.,0.)},
        attachment_position_targets={"GRIP:A":(1.+math.cos(.25),math.sin(.25),0.)},
        jacobian_mode="ANALYTIC_RESEARCH_V1",
        regularization=1e-9,goal_tolerance=1e-3,
    )
    assert solved.status=="PASS_MULTIGOAL_BEHAVIORAL_COURT"
    court=evaluate_attachment_world_frames_v1(
        unqualified,solved.pose,
        required_world_positions={"GRIP:A":(1.+math.cos(.25),math.sin(.25),0.)},
    )
    assert court.status=="PASS_ATTACHMENT_FRAME_COURT"
    assert court.frames[0].orientation_authority=="PARENT_ORIENTATION_ONLY_NOT_AUTHORED_GRIP"


def test_attachment_position_target_validation_and_analytic_mode_fail_closed():
    rig=_orientation_rig()
    for target,code in (
        ((1.,2.),"POSITION_INVALID"),
        ((float("nan"),0.,0.),"POSITION_INVALID"),
    ):
        with pytest.raises(QualificationError,match=code):
            solve_declared_multi_goal_v1(
                rig,_goals(.2),
                attachment_position_targets={"GRIP:A":target},
                jacobian_mode="ANALYTIC_RESEARCH_V1",
            )
    with pytest.raises(QualificationError,match="POSITION_ANCHOR_UNKNOWN"):
        solve_declared_multi_goal_v1(
            rig,_goals(.2),
            attachment_position_targets={"UNKNOWN":(0.,0.,0.)},
            jacobian_mode="ANALYTIC_RESEARCH_V1",
        )
    with pytest.raises(QualificationError,match="POSITION_REQUIRES_ANALYTIC"):
        solve_declared_multi_goal_v1(
            rig,_goals(.2),
            attachment_position_targets={"GRIP:A":(0.,0.,0.)},
        )


def test_live_full_grip_pose_solve_keeps_secondary_and_other_constraints_intact():
    mesh,rig,skin,mechanics=_objects()
    rig=replace(rig,attachment_anchors=(CanonicalAttachmentAnchorIRV2(
        "GRIP:A",parent_canonical_joint_id="A1",
        local_P=(1.,0.,0.),source_proposal_anchor_id="synthetic:grip",
        metadata={"local_basis":I},
    ),))
    angle=.24
    frame=evaluate_mechanics_intervention_frame_v1(
        mesh,rig,skin,mechanics,
        overrides=_goals(angle),
        solver_mode="DECLARED_MULTI_GOAL_V1",
        multi_goal_jacobian_mode="ANALYTIC_RESEARCH_V1",
        attachment_orientation_mode="SOLVE_DECLARED_RESEARCH_V1",
        required_attachment_world_positions={
            "GRIP:A":(1.+math.cos(angle),math.sin(angle),0.),
        },
        required_attachment_world_orientations={"GRIP:A":_rz(angle)},
        pinned_contact_control_ids=("B_HANDLE",),
        pinned_world_targets={"B_HANDLE":(-2.,0.,0.)},
        max_allowed_constraint_residual=1e-3,
    )
    assert frame.status=="PASS_MECHANICS_BEHAVIORAL_COURT",frame.failures
    assert frame.attachment_frames.status=="PASS_ATTACHMENT_FRAME_COURT"
    assert frame.metadata["attachment_max_position_residual_world"]<1e-3
    assert frame.metadata["attachment_target_orientation_status"]=="PASS_ATTACHMENT_FRAME_COURT"
    assert frame.metadata["rig_or_skin_mutated"] is False
