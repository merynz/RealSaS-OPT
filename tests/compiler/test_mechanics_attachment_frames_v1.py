from __future__ import annotations

"""Attachment world-frame and declared orientation (not IK orientation solving)."""

from dataclasses import replace

import numpy as np
import pytest

from compiler.realsas_compiler_core.canonical_rig_ir_v2 import (
    CanonicalAttachmentAnchorIRV2,
)
from compiler.realsas_compiler_core.mechanics_control_solver_v1 import (
    ControlTransformIRV1, solve_control_pose_v1,
)
from compiler.realsas_compiler_core.mechanics_attachment_frames_v1 import (
    evaluate_attachment_world_frames_v1,
)
from compiler.realsas_compiler_core.mechanics_live_runtime_v1 import (
    evaluate_mechanics_intervention_frame_v1,
)
from compiler.realsas_compiler_core.types import QualificationError
from tests.compiler.test_mechanics_live_multigoal_integration_v1 import _objects


I=((1.,0.,0.),(0.,1.,0.),(0.,0.,1.))
RZ90=((0.,-1.,0.),(1.,0.,0.),(0.,0.,1.))


def _case():
    mesh,rig,skin,mechanics=_objects()
    anchor=CanonicalAttachmentAnchorIRV2(
        canonical_anchor_id="GRIP:0",
        parent_canonical_joint_id="BASE",
        local_P=(0.25,0.5,0.),
        source_proposal_anchor_id="fixture:grip",
        metadata={"local_basis":RZ90,
                  "source_orientation_proven":False,
                  "research_declared_grip":True},
    )
    rig=replace(rig,attachment_anchors=(anchor,))
    return mesh,rig,skin,mechanics


def test_declared_grip_orientation_tracks_joint_world_basis_and_offset():
    mesh,rig,skin,mechanics=_case()
    pose=solve_control_pose_v1(rig,(ControlTransformIRV1(
        "A_HANDLE",translation=(-0.2,0.2,0.),
    ),))
    root=next(j for j in pose.joints if j.canonical_joint_id=="BASE")
    R=np.asarray(root.rotation_basis)
    expected=R@np.asarray(RZ90)
    record=evaluate_attachment_world_frames_v1(
        rig,pose,required_world_orientations={"GRIP:0":expected.tolist()},
    )
    assert record.status=="PASS_ATTACHMENT_FRAME_COURT"
    assert record.max_target_angular_residual_rad<1e-6
    grip=record.frames[0]
    assert grip.orientation_authority=="DECLARED_LOCAL_GRIP_BASIS_RESEARCH_ONLY"
    assert np.allclose(grip.world_basis,expected,atol=1e-12)
    assert np.allclose(grip.world_P,
                       np.asarray(root.solved_P)+R@np.asarray((.25,.5,0.)))
    assert record.metadata["orientation_ik_solver_executed"] is False


def test_attachment_orientation_mismatch_not_minted_as_ik_success():
    _,rig,_,_= _case()
    pose=solve_control_pose_v1(rig,())
    result=evaluate_attachment_world_frames_v1(
        rig,pose,required_world_orientations={"GRIP:0":I},
        angular_tolerance_rad=.1,
    )
    assert result.status=="FAIL_ATTACHMENT_ORIENTATION_TARGET"
    assert result.max_target_angular_residual_rad>1.5


def test_unqualified_local_grip_orientation_cannot_be_used_as_target():
    _,rig,_,_=_case()
    anchor=replace(rig.attachment_anchors[0],metadata={})
    rig=replace(rig,attachment_anchors=(anchor,))
    pose=solve_control_pose_v1(rig,())
    observed=evaluate_attachment_world_frames_v1(rig,pose)
    assert observed.frames[0].orientation_authority=="PARENT_ORIENTATION_ONLY_NOT_AUTHORED_GRIP"
    with pytest.raises(QualificationError,match="ORIENTATION_NOT_AUTHORED"):
        evaluate_attachment_world_frames_v1(
            rig,pose,required_world_orientations={"GRIP:0":I},
        )


def test_bad_attachment_frames_and_parent_controls_fail_closed():
    _,rig,_,_=_case()
    pose=solve_control_pose_v1(rig,())
    with pytest.raises(QualificationError,match="TARGET_ANCHOR_UNKNOWN"):
        evaluate_attachment_world_frames_v1(
            rig,pose,required_world_orientations={"FAKE":I},
        )
    with pytest.raises(QualificationError,match="LOCAL_GRIP_BASIS_INVALID"):
        evaluate_attachment_world_frames_v1(
            replace(rig,attachment_anchors=(
                replace(rig.attachment_anchors[0],
                        metadata={"local_basis":((2.,0.,0.),(0.,1.,0.),(0.,0.,1.))}),
            )),pose,
        )
    with pytest.raises(QualificationError,match="CONTROL_PARENT_NOT_EXECUTABLE_V1"):
        evaluate_attachment_world_frames_v1(
            replace(rig,attachment_anchors=(
                replace(rig.attachment_anchors[0],
                        parent_canonical_joint_id=None,
                        parent_canonical_control_id="A_HANDLE"),
            )),pose,
        )
    with pytest.raises(QualificationError,match="POSE_LINEAGE_DRIFT"):
        evaluate_attachment_world_frames_v1(
            rig,replace(pose,control_program_hash="not-current"),
        )


def test_attachment_frame_is_evaluated_in_primary_mechanics_before_secondary():
    mesh,rig,skin,mechanics=_case()
    frame=evaluate_mechanics_intervention_frame_v1(
        mesh,rig,skin,mechanics,
        overrides=(ControlTransformIRV1("A_HANDLE",translation=(-0.2,0.2,0.)),),
    )
    assert frame.status=="PASS_MECHANICS_BEHAVIORAL_COURT"
    assert frame.attachment_frames is not None
    assert len(frame.attachment_frames.frames)==1
    assert frame.metadata["attachment_frames_evaluated_before_secondary"] is True
    assert frame.metadata["attachment_frame_count"]==1
    assert frame.attachment_frames.frames[0].orientation_authority=="DECLARED_LOCAL_GRIP_BASIS_RESEARCH_ONLY"


def test_runtime_orientation_requirement_is_real_fail_not_automatically_rotated_grip():
    mesh,rig,skin,mechanics=_case()
    frame=evaluate_mechanics_intervention_frame_v1(
        mesh,rig,skin,mechanics,
        required_attachment_world_orientations={"GRIP:0":I},
    )
    assert frame.status=="FAIL_MECHANICS_BEHAVIORAL_COURT"
    assert "ATTACHMENT_FRAME_NOT_QUALIFIED" in frame.failures
    assert frame.metadata["attachment_target_orientation_status"]=="FAIL_ATTACHMENT_ORIENTATION_TARGET"
    assert frame.attachment_frames.max_target_angular_residual_rad>1.5
    # This is an evaluator, never a disguised IK orientation optimizer:
    assert frame.attachment_frames.metadata["orientation_ik_solver_executed"] is False
