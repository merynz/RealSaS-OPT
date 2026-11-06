from __future__ import annotations

"""Multi-grip rigid feasibility: necessary contradictions, not global IK PASS."""

from dataclasses import replace
import math

import numpy as np
import pytest

from compiler.realsas_compiler_core.canonical_rig_ir_v2 import CanonicalAttachmentAnchorIRV2
from compiler.realsas_compiler_core.mechanics_attachment_feasibility_v1 import (
    evaluate_declared_attachment_rigidity_v1,
)
from compiler.realsas_compiler_core.mechanics_multigoal_solver_v1 import solve_declared_multi_goal_v1
from compiler.realsas_compiler_core.mechanics_attachment_frames_v1 import evaluate_attachment_world_frames_v1
from compiler.realsas_compiler_core.types import QualificationError
from tests.compiler.test_mechanics_oriented_ik_v1 import _orientation_rig,_goals,_rz,I


def _two_grips():
    rig=_orientation_rig()
    second=CanonicalAttachmentAnchorIRV2(
        canonical_anchor_id="GRIP:B",
        parent_canonical_joint_id="A1",
        local_P=(0.,1.,0.),
        source_proposal_anchor_id="synthetic:second-attachment",
        metadata={"local_basis":I,"source_orientation_proven":False},
    )
    return replace(rig,attachment_anchors=rig.attachment_anchors+(second,))


def _world_targets(angle:float):
    c,s=math.cos(angle),math.sin(angle)
    return {
        "GRIP:A":(1.+c,s,0.),
        "GRIP:B":(1.-s,c,0.),
    }


def test_two_declared_same_parent_6d_grips_are_executable_with_world_pin():
    rig=_two_grips()
    angle=.23
    targets=_world_targets(angle)
    orient={"GRIP:A":_rz(angle),"GRIP:B":_rz(angle)}
    court=evaluate_declared_attachment_rigidity_v1(
        rig,required_world_positions=targets,required_world_orientations=orient,
    )
    assert court.status=="NO_PAIRWISE_RIGID_CONFLICT_GLOBAL_IK_UNDETERMINED"
    assert not court.witnesses
    assert court.metadata["global_ik_feasibility_certified"] is False
    solved=solve_declared_multi_goal_v1(
        rig,_goals(angle),
        pinned_contact_control_ids=("B_HANDLE",),
        pinned_world_targets={"B_HANDLE":(-2.,0.,0.)},
        attachment_position_targets=targets,
        attachment_orientation_targets=orient,
        attachment_preflight_mode="RIGID_HARD_NECESSARY_V1",
        jacobian_mode="ANALYTIC_RESEARCH_V1",
        verify_analytic_jacobian=True,
        goal_tolerance=1e-3,orientation_tolerance_rad=1e-3,
        regularization=1e-9,
    )
    assert solved.status=="PASS_MULTIGOAL_BEHAVIORAL_COURT"
    assert solved.max_attachment_position_residual_world<1e-3
    assert solved.max_orientation_residual_rad<1e-3
    assert solved.max_pin_residual<1e-3
    assert len(solved.attachment_position_goals)==2
    assert len(solved.orientation_goals)==2
    assert solved.metadata["attachment_preflight_pairwise_witness_count"]==0
    assert solved.metadata["analytic_jacobian_finite_difference_max_error"]<1e-6
    independent=evaluate_attachment_world_frames_v1(
        rig,solved.pose,
        required_world_positions=targets,
        required_world_orientations=orient,
    )
    assert independent.status=="PASS_ATTACHMENT_FRAME_COURT"
    assert len(independent.frames)==2


def test_same_parent_world_point_distance_conflict_yields_stable_witness():
    rig=_two_grips()
    impossible={"GRIP:A":(2.,0.,0.),"GRIP:B":(5.,0.,0.)}
    r=evaluate_declared_attachment_rigidity_v1(
        rig,required_world_positions=impossible,
    )
    assert r.status=="FAIL_DECLARED_RIGID_ATTACHMENT_CONFLICT"
    distance=[w for w in r.witnesses if w.constraint_kind=="RIGID_POINT_DISTANCE"]
    assert len(distance)==1
    assert distance[0].anchor_ids==("GRIP:A","GRIP:B")
    assert distance[0].measured_mismatch>1.
    assert distance[0].allowable_mismatch==pytest.approx(.002)
    with pytest.raises(QualificationError,match="MULTIGOAL_RIGID_ATTACHMENT_CONFLICT:RIGID_POINT_DISTANCE"):
        solve_declared_multi_goal_v1(
            rig,_goals(.1),
            attachment_position_targets=impossible,
            attachment_preflight_mode="RIGID_HARD_NECESSARY_V1",
            jacobian_mode="ANALYTIC_RESEARCH_V1",
        )


def test_same_parent_declared_grips_cannot_demand_different_world_orientations():
    rig=_two_grips()
    r=evaluate_declared_attachment_rigidity_v1(
        rig,required_world_orientations={
            "GRIP:A":I,"GRIP:B":_rz(.6),
        },
    )
    assert r.status=="FAIL_DECLARED_RIGID_ATTACHMENT_CONFLICT"
    witness=next(w for w in r.witnesses
                 if w.constraint_kind=="RIGID_PARENT_ORIENTATION")
    assert witness.measured_mismatch==pytest.approx(.6,abs=1e-9)
    assert witness.allowable_mismatch==pytest.approx(.002)


def test_two_world_points_same_distance_can_still_conflict_with_authored_orientation():
    rig=_two_grips()
    positions={"GRIP:A":(2.,0.,0.),"GRIP:B":(1.,-1.,0.)}
    r=evaluate_declared_attachment_rigidity_v1(
        rig,required_world_positions=positions,
        required_world_orientations={"GRIP:A":I},
    )
    assert r.status=="FAIL_DECLARED_RIGID_ATTACHMENT_CONFLICT"
    assert not any(w.constraint_kind=="RIGID_POINT_DISTANCE" for w in r.witnesses)
    w=next(w for w in r.witnesses
           if w.constraint_kind=="RIGID_FRAME_POINT_DISPLACEMENT")
    assert w.measured_mismatch==pytest.approx(2.)
    assert w.allowable_mismatch<.01


def test_rigidity_court_respects_position_and_orientation_tolerance_near_boundary():
    rig=_two_grips()
    pos={"GRIP:A":(2.,0.,0.),"GRIP:B":(1.+.0003,1.,0.)}
    r=evaluate_declared_attachment_rigidity_v1(
        rig,
        required_world_positions=pos,
        required_world_orientations={"GRIP:A":_rz(.0003)},
        position_tolerance_world=.001,
        orientation_tolerance_rad=.001,
    )
    assert not r.witnesses
    assert r.status=="NO_PAIRWISE_RIGID_CONFLICT_GLOBAL_IK_UNDETERMINED"


def test_cross_parent_pairs_are_indeterminate_not_falsely_certified_or_rejected():
    rig=_two_grips()
    second=replace(rig.attachment_anchors[1],parent_canonical_joint_id="B1")
    rig=replace(rig,attachment_anchors=(rig.attachment_anchors[0],second))
    result=evaluate_declared_attachment_rigidity_v1(
        rig,required_world_positions={
            "GRIP:A":(2.,0.,0.),"GRIP:B":(50.,0.,0.),
        },
        required_world_orientations={"GRIP:A":I,"GRIP:B":_rz(1.)},
    )
    assert result.status=="NO_PAIRWISE_RIGID_CONFLICT_GLOBAL_IK_UNDETERMINED"
    assert result.metadata["global_ik_feasibility_certified"] is False


def test_preflight_rejects_undeclared_frame_and_bad_target_geometry():
    rig=_two_grips()
    without_basis=replace(rig.attachment_anchors[1],metadata={})
    no_orientation=replace(rig,attachment_anchors=(
        rig.attachment_anchors[0],without_basis,
    ))
    with pytest.raises(QualificationError,match="UNDECLARED_LOCAL_ORIENTATION"):
        evaluate_declared_attachment_rigidity_v1(
            no_orientation,required_world_orientations={"GRIP:B":I},
        )
    with pytest.raises(QualificationError,match="WORLD_POSITION_INVALID"):
        evaluate_declared_attachment_rigidity_v1(
            rig,required_world_positions={"GRIP:A":(float("nan"),0.,0.)},
        )
    with pytest.raises(QualificationError,match="UNKNOWN_ANCHOR"):
        evaluate_declared_attachment_rigidity_v1(
            rig,required_world_positions={"MISSING":(0.,0.,0.)},
        )
    with pytest.raises(QualificationError,match="RESOURCE_BOUND"):
        evaluate_declared_attachment_rigidity_v1(
            rig,required_world_positions=_world_targets(0.),
            maximum_anchors=1,
        )
    with pytest.raises(QualificationError,match="PREFLIGHT_MODE_UNSUPPORTED"):
        solve_declared_multi_goal_v1(
            rig,attachment_preflight_mode="ASSUME_GLOBAL_PASS",
        )
