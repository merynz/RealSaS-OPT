from __future__ import annotations

import copy

import numpy as np
import pytest

from tools.corpus.build_corpus370_v5_source_native_canonical_animation_v1 import (
    MASKED, PASS, build_numeric, frame_schedule,
)


def data():
    rest=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]],np.float64)
    posed=np.stack([rest,rest+np.array([0.,0.,.25])],axis=0)
    source={
        "rest_vertices_source_world":rest,
        "faces":np.asarray([[0,1,2]],np.int32),
        "face_owner_privileged":np.asarray([0],np.int32),
    }
    native={
        "rest_vertices_source_world":rest.copy(),
        "faces":source["faces"].copy(),
        "face_owner":source["face_owner_privileged"].copy(),
        "posed_vertices_source_world":posed,
    }
    m=np.diag([2.,2.,2.,1.])
    m[:3,3]=np.array([1.,3.,5.])
    snap={"source_world_to_canonical":m,"canonical_to_source_world":np.linalg.inv(m)}
    receipt={
        "native_motion_inventory_exported":["Idle"],
        "sampled_frames":[{"action":"Idle","frame":0},{"action":"Idle","frame":8}],
    }
    return source,native,snap,receipt


def test_authored_motion_maps_to_canonical_not_physical_velocity():
    src,nat,snap,rec=data()
    out=build_numeric(src,nat,snap,rec)
    assert out["status"]==PASS
    assert out["measures"]["native_frame_to_seconds_authority"] is False
    assert out["measures"]["source_motion_is_causal_intervention"] is False
    assert out["measures"]["model_training_authorized"] is False
    np.testing.assert_allclose(
        out["arrays"]["posed_vertices_canonical"][1,0],
        [1.,3.,5.5])
    assert out["measures"]["measured_max_rest_relative_displacement_canonical"]==pytest.approx(.5)
    assert out["arrays"]["source_action_edge_frame_delta"].tolist()==[8]


def test_zero_movement_masks_without_false_negative_label():
    src,nat,snap,rec=data()
    nat["posed_vertices_source_world"][:]=src["rest_vertices_source_world"][None]
    out=build_numeric(src,nat,snap,rec)
    assert out["status"]==MASKED
    assert out["measures"]["measured_max_rest_relative_displacement_canonical"]==0.


def test_source_rest_import_mismatch_does_not_silently_transfer():
    src,nat,snap,rec=data()
    nat["rest_vertices_source_world"][0,0]=.01
    with pytest.raises(ValueError,match="NATIVE_REST_NOT_OBSERVATION_EQUIVALENT"):
        build_numeric(src,nat,snap,rec)


def test_owner_and_topology_cannot_reindex(tmp_path):
    src,nat,snap,rec=data()
    nat["face_owner"][0]=1
    with pytest.raises(ValueError,match="REST_TOPOLOGY_DRIFT"):
        build_numeric(src,nat,snap,rec)


def test_noninvertible_source_camera_gauge_rejected():
    src,nat,snap,rec=data()
    snap["canonical_to_source_world"]=np.eye(4)
    with pytest.raises(ValueError,match="SOURCE_FRAME_INVERSE"):
        build_numeric(src,nat,snap,rec)


def test_authored_frame_ordering_required():
    src,nat,snap,rec=data()
    rec["sampled_frames"][1]["frame"]=0
    with pytest.raises(ValueError,match="MONOTONIC_ACTION_FRAMES"):
        build_numeric(src,nat,snap,rec)


def test_action_boundaries_do_not_create_fictitious_cross_clip_motion():
    src,nat,snap,rec=data()
    nat["posed_vertices_source_world"]=np.concatenate([
        nat["posed_vertices_source_world"],
        nat["posed_vertices_source_world"]])
    rec["native_motion_inventory_exported"]=["Idle","Run"]
    rec["sampled_frames"]=[
        {"action":"Idle","frame":0},{"action":"Idle","frame":8},
        {"action":"Run","frame":0},{"action":"Run","frame":8},
    ]
    out=build_numeric(src,nat,snap,rec)
    assert out["arrays"]["valid_same_action_edge_ending_frame"].tolist()==[1,3]
    assert out["arrays"]["frame_action_ordinal_teacher_only"].tolist()==[0,0,1,1]


def test_cannot_promote_action_name_to_student_input_by_producer():
    src,nat,snap,rec=data()
    out=build_numeric(src,nat,snap,rec)
    assert "native_motion_inventory_exported" not in out["arrays"]
    assert "rest_vertices_canonical" in out["arrays"]
    assert "source_triangle_owner_teacher_only" in out["arrays"]
    assert out["measures"]["native_frame_to_seconds_authority"] is False
