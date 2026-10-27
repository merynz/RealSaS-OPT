from __future__ import annotations

from copy import deepcopy

import pytest

from models.iris.truth.motion_truth_availability_v1 import (
    evaluate_motion_truth_availability_v1,
)


def _scene():
    return {
        "schema": "RealSaS.TeacherSceneIR.v1",
        "subject_key": "fixture",
        "qualification_minted": False,
        "student_input_allowed": False,
        "native_clip_refs": [{"path": "source_teacher.npz"}],
        "constraint_capabilities": {
            "local_native_rotation_parameter": "MEASURED_CANDIDATE",
            "physical_impulse": "UNAVAILABLE",
            "authored_physical_dof_or_joint_limit": "UNKNOWN",
        },
        "coordinate_contract": {
            "time_unit": "NO_DYNAMIC_TIME_IN_STATIC_CALIBRATION",
        },
        "calibration": {
            "actions": [
                {"kind": "ISOLATED_KINEMATIC_PARAMETER_PERTURBATION"}
            ]
        },
    }


def test_static_kinematic_calibration_does_not_mint_motion_truth_families():
    report = evaluate_motion_truth_availability_v1(_scene())
    assert report.kinematic_response_diagnostic.status == "MEASURED_DIAGNOSTIC_ONLY"
    assert report.kinematic_response_diagnostic.teacher_measured is True
    assert report.kinematic_response_diagnostic.student_target_allowed is False
    assert report.admissible_motion_subspace.status == "UNAVAILABLE"
    assert report.physical_impulse_response.status == "UNAVAILABLE"
    assert report.temporal_response.status == "UNAVAILABLE"
    assert report.secondary_motion_consumer.status == "UNAVAILABLE"
    for row in (
        report.admissible_motion_subspace,
        report.physical_impulse_response,
        report.temporal_response,
        report.secondary_motion_consumer,
    ):
        assert row.student_target_allowed is False
        assert row.compiler_consequence_allowed is False


def test_native_clip_reference_alone_is_not_temporal_measurement():
    scene = _scene()
    scene["native_clip_refs"] = [{"path": "many_clips.fbx"}]
    report = evaluate_motion_truth_availability_v1(scene)
    assert report.temporal_response.status == "UNAVAILABLE"
    assert report.temporal_response.reason == "STATIC_CALIBRATION_HAS_NO_DYNAMIC_TIME_RESPONSE"


def test_capability_flag_alone_cannot_mint_admissible_subspace_arrays():
    scene = _scene()
    scene["constraint_capabilities"]["authored_physical_dof_or_joint_limit"] = "MEASURED"
    report = evaluate_motion_truth_availability_v1(scene)
    assert report.admissible_motion_subspace.status == "UNAVAILABLE"
    assert report.admissible_motion_subspace.reason == "PHYSICAL_DOF_CAPABILITY_DECLARED_BUT_SUBSPACE_ARRAYS_NOT_PRODUCED"


def test_dynamic_time_declaration_alone_cannot_mint_temporal_response():
    scene = _scene()
    scene["coordinate_contract"]["time_unit"] = "SECOND"
    report = evaluate_motion_truth_availability_v1(scene)
    assert report.temporal_response.status == "UNAVAILABLE"
    assert report.temporal_response.reason == "DYNAMIC_TIME_DECLARED_BUT_TEMPORAL_RESPONSE_ARRAYS_NOT_PRODUCED"


def test_teacher_authority_boundary_is_fail_closed():
    scene = _scene()
    scene["qualification_minted"] = True
    with pytest.raises(ValueError, match="AUTHORITY_BOUNDARY"):
        evaluate_motion_truth_availability_v1(scene)

    scene = _scene()
    scene["student_input_allowed"] = True
    with pytest.raises(ValueError, match="PRIVILEGED_TEACHER_INPUT"):
        evaluate_motion_truth_availability_v1(scene)
