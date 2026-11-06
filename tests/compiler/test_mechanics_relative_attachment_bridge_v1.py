from __future__ import annotations

"""Cross-parent 6D two-grip bridge: real FK derivatives, no source truth claim."""

from dataclasses import replace
import math

import numpy as np
import pytest

from compiler.realsas_compiler_core.canonical_rig_ir_v2 import (
    CanonicalAttachmentAnchorIRV2,
)
from compiler.realsas_compiler_core.mechanics_control_solver_v1 import (
    ControlTransformIRV1,
)
from compiler.realsas_compiler_core.mechanics_multigoal_solver_v1 import (
    solve_declared_multi_goal_v1,
)
from compiler.realsas_compiler_core.mechanics_relative_attachment_bridge_v1 import (
    DeclaredRelativeAttachmentBridgeV1,
    evaluate_declared_relative_attachment_bridges_v1,
    prepare_declared_relative_attachment_bridges_v1,
)
from compiler.realsas_compiler_core.mechanics_live_runtime_v1 import (
    MechanicsInterventionStepV1,
    evaluate_mechanics_intervention_frame_v1,
    evaluate_mechanics_intervention_sequence_v1,
)
from compiler.realsas_compiler_core.types import QualificationError
from tests.compiler.test_mechanics_multigoal_solver_v1 import _branched_rig
from tests.compiler.test_mechanics_live_multigoal_integration_v1 import _objects

I=((1.,0.,0.),(0.,1.,0.),(0.,0.,1.))


def _rig_and_bridge(*,drag_only=False):
    rig=_branched_rig()
    anchors=(
        CanonicalAttachmentAnchorIRV2(
            canonical_anchor_id="G:A",parent_canonical_joint_id="A1",
            local_P=(1.,0.,0.),source_proposal_anchor_id="synthetic:grip_a",
            metadata={"local_basis":I,"source_authority_proven":False},
        ),
        CanonicalAttachmentAnchorIRV2(
            canonical_anchor_id="G:B",parent_canonical_joint_id="B1",
            local_P=(-1.,0.,0.),source_proposal_anchor_id="synthetic:grip_b",
            metadata={"local_basis":I,"source_authority_proven":False},
        ),
    )
    rig=replace(
        rig,attachment_anchors=anchors,
        constraints=rig.constraints[:1] if drag_only else rig.constraints,
    )
    bridge=DeclaredRelativeAttachmentBridgeV1(
        bridge_id="SHARED_RIGID_PROP_EXPLICIT",
        anchor_a_id="G:A",anchor_b_id="G:B",
        a_to_b_local_P=(-4.,0.,0.),
        a_to_b_local_basis=I,
        declaration_evidence_id="TEST_ONLY_EXPLICIT_SHARED_OBJECT_RELATION",
        rig_lineage_hash=rig.rig_lineage_hash,
        control_program_hash=rig.control_program_hash,
    )
    return rig,bridge


def _rotated_targets(theta):
    c,s=math.cos(theta),math.sin(theta)
    return (2.*c,2.*s,0.),(-2.*c,-2.*s,0.)


def _drag_to_world(target):
    return (ControlTransformIRV1(
        "A_HANDLE",translation=(target[0]-2.,target[1],target[2]),
    ),)


def test_two_distinct_parent_grips_hold_full_relative_frame_while_world_contact_moves():
    rig,bridge=_rig_and_bridge()
    theta=.16
    a,b=_rotated_targets(theta)
    out=solve_declared_multi_goal_v1(
        rig,_drag_to_world(a),
        pinned_contact_control_ids=("B_HANDLE",),
        pinned_world_targets={"B_HANDLE":b},
        relative_attachment_bridges=(bridge,),
        jacobian_mode="ANALYTIC_RESEARCH_V1",
        verify_analytic_jacobian=True,
        goal_tolerance=1e-3,
        orientation_tolerance_rad=1e-3,
        regularization=1e-9,max_nfev=250,
    )
    assert out.status=="PASS_MULTIGOAL_BEHAVIORAL_COURT",out
    assert out.max_drag_residual<1e-3
    assert out.max_pin_residual<1e-3
    assert out.max_relative_bridge_position_residual_world<1e-3
    assert out.max_relative_bridge_orientation_residual_rad<1e-3
    assert out.metadata["cross_parent_relative_bridge_count"]==1
    assert out.metadata["analytic_jacobian_finite_difference_max_error"]<1e-6
    assert len(out.relative_attachment_bridges)==1
    by={j.canonical_joint_id:j for j in out.pose.joints}
    assert np.allclose(by["A2"].solved_P,a,atol=1e-3)
    assert np.allclose(by["B2"].solved_P,b,atol=1e-3)
    independently=evaluate_declared_relative_attachment_bridges_v1(
        rig,out.pose,(bridge,),
    )
    assert independently.status=="PASS_DECLARED_RELATIVE_BRIDGE_COURT"
    assert independently.max_position_residual_world<1e-3
    assert independently.max_orientation_residual_rad<1e-3
    assert independently.metadata["model_or_graphics_authority_minted"] is False


def test_one_drag_drives_other_unconstrained_branch_through_declared_prop_bridge():
    rig,bridge=_rig_and_bridge(drag_only=True)
    theta=.13
    a,b=_rotated_targets(theta)
    solved=solve_declared_multi_goal_v1(
        rig,_drag_to_world(a),
        relative_attachment_bridges=(bridge,),
        jacobian_mode="ANALYTIC_RESEARCH_V1",
        regularization=1e-9,
        goal_tolerance=1e-3,orientation_tolerance_rad=1e-3,
    )
    assert solved.status=="PASS_MULTIGOAL_BEHAVIORAL_COURT",solved
    by={j.canonical_joint_id:j for j in solved.pose.joints}
    assert np.allclose(by["B2"].solved_P,b,atol=1e-3)
    assert solved.max_relative_bridge_position_residual_world<1e-3
    assert solved.max_relative_bridge_orientation_residual_rad<1e-3
    assert all(g.canonical_control_id=="A_HANDLE" for g in solved.goals)
    assert len(solved.parameters_rad)==3


def test_conflicting_planted_contact_and_two_hand_bridge_remains_fail():
    rig,bridge=_rig_and_bridge()
    a=(1.+math.cos(.22),math.sin(.22),0.)
    edit=_drag_to_world(a)
    baseline=solve_declared_multi_goal_v1(
        rig,edit,pinned_contact_control_ids=("B_HANDLE",),
        pinned_world_targets={"B_HANDLE":(-2.,0.,0.)},
        jacobian_mode="ANALYTIC_RESEARCH_V1",
        goal_tolerance=1e-3,regularization=1e-9,
    )
    assert baseline.status=="PASS_MULTIGOAL_BEHAVIORAL_COURT"
    incompatible=solve_declared_multi_goal_v1(
        rig,edit,pinned_contact_control_ids=("B_HANDLE",),
        pinned_world_targets={"B_HANDLE":(-2.,0.,0.)},
        relative_attachment_bridges=(bridge,),
        jacobian_mode="ANALYTIC_RESEARCH_V1",
        goal_tolerance=1e-3,orientation_tolerance_rad=1e-3,
        regularization=1e-9,
    )
    assert incompatible.status=="FAIL_MULTIGOAL_BEHAVIORAL_COURT"
    assert (
        incompatible.max_relative_bridge_position_residual_world>1e-3
        or incompatible.max_relative_bridge_orientation_residual_rad>1e-3
        or incompatible.max_drag_residual>1e-3
        or incompatible.max_pin_residual>1e-3
    )
    assert incompatible.pose.metadata["rig_program_mutated"] is False


def test_bridge_rejects_unknown_anchor_same_parent_stale_lineage_and_nonorthogonal_basis():
    rig,link=_rig_and_bridge()
    for invalid,match in (
        (replace(link,anchor_b_id="UNKNOWN"),"ANCHOR_PAIR_INVALID"),
        (replace(link,anchor_b_id="G:A"),"ANCHOR_PAIR_INVALID"),
        (replace(link,rig_lineage_hash="stale"),"AUTHORITY_LINEAGE_DRIFT"),
        (replace(link,a_to_b_local_P=(math.nan,0.,0.)),"DECLARED_OFFSET_INVALID"),
        (replace(link,a_to_b_local_basis=((2.,0.,0.),(0.,1.,0.),(0.,0.,1.))),
         "DECLARED_ROTATION_INVALID"),
    ):
        with pytest.raises(QualificationError,match=match):
            prepare_declared_relative_attachment_bridges_v1(rig,(invalid,))
    with pytest.raises(QualificationError,match="DUPLICATE_PAIR"):
        prepare_declared_relative_attachment_bridges_v1(rig,(link,replace(
            link,bridge_id="ANOTHER",anchor_a_id="G:B",anchor_b_id="G:A",
        )))
    new_b=replace(
        rig.attachment_anchors[1],parent_canonical_joint_id="A1",
    )
    with pytest.raises(QualificationError,match="SAME_PARENT_USE_RIGIDITY_COURT"):
        prepare_declared_relative_attachment_bridges_v1(
            replace(rig,attachment_anchors=(rig.attachment_anchors[0],new_b)),
            (link,),
        )
    with pytest.raises(QualificationError,match="UNDECLARED_LOCAL_FRAME"):
        prepare_declared_relative_attachment_bridges_v1(
            replace(rig,attachment_anchors=(
                replace(rig.attachment_anchors[0],metadata={}),
                rig.attachment_anchors[1],
            )),(link,),
        )
    with pytest.raises(QualificationError,match="REQUIRES_ANALYTIC"):
        solve_declared_multi_goal_v1(
            rig,relative_attachment_bridges=(link,),
            jacobian_mode="NUMERIC_V1",
        )


def test_live_world_bridge_runs_full_skin_and_5_frame_warm_motion_sequence():
    mesh,rig,skin,mechanics=_objects()
    authored,link=_rig_and_bridge()
    rig=replace(rig,attachment_anchors=authored.attachment_anchors)
    # Both synthetic rig fixtures intentionally use the same joint/lineage
    # naming and root-space geometry, so the declared link is still bound.
    theta=.1
    a,b=_rotated_targets(theta)
    live=evaluate_mechanics_intervention_frame_v1(
        mesh,rig,skin,mechanics,overrides=_drag_to_world(a),
        solver_mode="DECLARED_MULTI_GOAL_V1",
        multi_goal_jacobian_mode="ANALYTIC_RESEARCH_V1",
        pinned_contact_control_ids=("B_HANDLE",),
        pinned_world_targets={"B_HANDLE":b},
        relative_attachment_bridges=(link,),
        max_allowed_constraint_residual=1e-3,
    )
    assert live.status=="PASS_MECHANICS_BEHAVIORAL_COURT",live.failures
    assert live.relative_attachment_court.status=="PASS_DECLARED_RELATIVE_BRIDGE_COURT"
    assert live.metadata["relative_bridge_count"]==1
    sequence_angles=(.04,.06,.08,.10,.12)
    # The planted world contact is fixed across this sequence, so use a
    # rest-angle target only for a valid constant-pinned motion court.
    steps=tuple(MechanicsInterventionStepV1(
        time_seconds=i/30.,
        overrides=_drag_to_world(_rotated_targets(theta)[0]),
    ) for i,_ in enumerate(sequence_angles))
    seq=evaluate_mechanics_intervention_sequence_v1(
        mesh,rig,skin,mechanics,steps,
        solver_mode="DECLARED_MULTI_GOAL_V1",
        multi_goal_jacobian_mode="ANALYTIC_RESEARCH_V1",
        pinned_contact_control_ids=("B_HANDLE",),
        pinned_world_targets={"B_HANDLE":b},
        relative_attachment_bridges=(link,),
        max_allowed_constraint_residual=1e-3,
    )
    assert seq.status=="PASS_MECHANICS_SEQUENCE_DIAGNOSTIC"
    assert seq.metadata["multi_goal_warm_start_frame_count"]==len(steps)-1
    assert all(x.relative_attachment_court.status=="PASS_DECLARED_RELATIVE_BRIDGE_COURT"
               for x in seq.frames)


def test_live_bridge_requires_explicit_analytic_multi_goal():
    mesh,rig,skin,mechanics=_objects()
    _,link=_rig_and_bridge()
    for kwargs in (
        {"solver_mode":"FABRIK_V1"},
        {"solver_mode":"DECLARED_MULTI_GOAL_V1","multi_goal_jacobian_mode":"NUMERIC_V1"},
    ):
        with pytest.raises(QualificationError,match="RELATIVE_BRIDGE_REQUIRES_ANALYTIC_MULTI_GOAL"):
            evaluate_mechanics_intervention_frame_v1(
                mesh,rig,skin,mechanics,
                relative_attachment_bridges=(link,),**kwargs,
            )



def _three_branch_foot_contact_scene():
    """Two object grips on distinct branches; third branch plants a foot.

    The foot uses TWO hinges and a terminal link, so its fixed world
    target remains reachable as the other two branches jointly rotate.
    There is no semantic name requirement in Mechanics; these labels are
    synthetic readability, not source inferred ownership.
    """
    from compiler.realsas_compiler_core.canonical_rig_ir_v2 import (
        CanonicalDeformJointIRV2, CanonicalJointFrameIRV2,
        CanonicalRigChainIRV2, CanonicalRigControlIRV2,
        CanonicalRigConstraintIRV2,
    )
    mesh,rig,skin,mechanics=_objects()
    declaration,bridge=_rig_and_bridge()
    rig=replace(
        rig,
        joints=rig.joints+(
            CanonicalDeformJointIRV2("C1",(0.,-.8,0.),"BASE","synthetic:c1"),
            CanonicalDeformJointIRV2("C2",(0.,-1.6,0.),"C1","synthetic:c2"),
            CanonicalDeformJointIRV2("C3",(0.,-2.4,0.),"C2","synthetic:c3"),
        ),
        local_frames=rig.local_frames+(
            CanonicalJointFrameIRV2("C1",I,("RZ",)),
            CanonicalJointFrameIRV2("C2",I,("RZ",)),
            CanonicalJointFrameIRV2("C3",I,()),
        ),
        controls=rig.controls+(
            CanonicalRigControlIRV2(
                "C_CONTACT","IK_EFFECTOR",("C3",),None,
                (0.,-2.4,0.),"synthetic:contact",
            ),
        ),
        chains=rig.chains+(
            CanonicalRigChainIRV2(
                "CHAIN_C",("BASE","C1","C2","C3"),
                "GENERIC_KINEMATIC_CHAIN","synthetic:chain_c",
            ),
        ),
        constraints=(rig.constraints[0],CanonicalRigConstraintIRV2(
            "C_PIN","IK",("C_CONTACT",),("C3",),"CHAIN_C",
            {"p0":1.,"p2":1.},"synthetic:ground_contact",
        )),
        attachment_anchors=declaration.attachment_anchors,
    )
    mechanics=replace(
        mechanics,
        constraint_schedule=("A_IK","C_PIN"),
    )
    return mesh,rig,skin,mechanics,bridge


def test_two_handed_rigid_object_can_rotate_across_5_frames_while_independent_foot_is_planted():
    mesh,rig,skin,mechanics,bridge=_three_branch_foot_contact_scene()
    foot=(.6,-2.2,0.)
    angles=(.08,.10,.12,.14,.16)
    steps=tuple(MechanicsInterventionStepV1(
        time_seconds=i/30.,
        overrides=_drag_to_world(_rotated_targets(theta)[0]),
    ) for i,theta in enumerate(angles))
    scene=evaluate_mechanics_intervention_sequence_v1(
        mesh,rig,skin,mechanics,steps,
        solver_mode="DECLARED_MULTI_GOAL_V1",
        multi_goal_jacobian_mode="ANALYTIC_RESEARCH_V1",
        pinned_contact_control_ids=("C_CONTACT",),
        pinned_world_targets={"C_CONTACT":foot},
        relative_attachment_bridges=(bridge,),
        max_allowed_constraint_residual=1e-3,
    )
    assert scene.status=="PASS_MECHANICS_SEQUENCE_DIAGNOSTIC",[
        (f.status,f.failures,f.metadata["multi_goal_status"]) for f in scene.frames
    ]
    assert scene.metadata["multi_goal_warm_start_frame_count"]==len(steps)-1
    assert scene.metadata["relative_bridge_count"]==1
    assert scene.max_vertex_speed>0.
    for frame,theta in zip(scene.frames,angles):
        assert frame.status=="PASS_MECHANICS_BEHAVIORAL_COURT",frame.failures
        assert frame.relative_attachment_court.status=="PASS_DECLARED_RELATIVE_BRIDGE_COURT"
        assert frame.relative_attachment_court.max_position_residual_world<1e-3
        assert frame.relative_attachment_court.max_orientation_residual_rad<1e-3
        by={j.canonical_joint_id:np.asarray(j.solved_P) for j in frame.pose.joints}
        a,b=_rotated_targets(theta)
        assert np.allclose(by["A2"],a,atol=1e-3)
        assert np.allclose(by["B2"],b,atol=1e-3)
        assert np.allclose(by["C3"],foot,atol=1e-3)
        assert frame.max_segment_length_relative_error<1e-8
        assert frame.metadata["rig_or_skin_mutated"] is False


def test_two_handed_bridge_plus_physically_unreachable_foot_cannot_claim_success():
    mesh,rig,skin,mechanics,bridge=_three_branch_foot_contact_scene()
    a,_=_rotated_targets(.12)
    trial=evaluate_mechanics_intervention_frame_v1(
        mesh,rig,skin,mechanics,
        overrides=_drag_to_world(a),
        solver_mode="DECLARED_MULTI_GOAL_V1",
        multi_goal_jacobian_mode="ANALYTIC_RESEARCH_V1",
        pinned_contact_control_ids=("C_CONTACT",),
        pinned_world_targets={"C_CONTACT":(100.,100.,0.)},
        relative_attachment_bridges=(bridge,),
        max_allowed_constraint_residual=1e-3,
    )
    assert trial.status=="FAIL_MECHANICS_BEHAVIORAL_COURT"
    assert "MULTIGOAL_SOLVER_NOT_QUALIFIED" in trial.failures
    assert trial.metadata["control_synthesis_performed"] is False
