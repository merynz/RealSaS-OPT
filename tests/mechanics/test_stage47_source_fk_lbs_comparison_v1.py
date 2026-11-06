from __future__ import annotations

"""Unit and negative courts for real Stage47 oracle-comparison implementation.

The small toy source is deliberately *not* a training or unseen fixture.
The independent forward reference below calculates each source bone's posed
matrix before LBS; production checker instead uses analytically grouped
descendant skin mass. This prevents tautologically testing the same kernel.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tools.mechanics.compare_stage47_source_fk_lbs_v1 import (
    Stage47CorrespondenceError,
    compare_stage47_source_fk_lbs_v1,
)


def _axis_matrix(axis: int, radians: float) -> np.ndarray:
    c, s = np.cos(radians), np.sin(radians)
    r=np.eye(4)
    if axis==0: r[1:3,1:3]=((c,-s),(s,c))
    elif axis==1: r[0,0],r[0,2],r[2,0],r[2,2]=c,s,-s,c
    else: r[0:2,0:2]=((c,-s),(s,c))
    return r


def _toy(tmp_path: Path, *, corrupt_response: bool=False):
    source_hash="a"*64
    frames=[
        {"name":"root","armature":"fixture","parent":None,
         "rest_matrix_world":np.eye(4).tolist()},
        {"name":"branch","armature":"fixture","parent":"root",
         "rest_matrix_world":np.array([[1.,0.,0.,1.],[0.,1.,0.,0.],[0.,0.,1.,0.],
                                       [0.,0.,0.,1.]]).tolist()},
    ]
    mats=[np.asarray(x["rest_matrix_world"]) for x in frames]
    rest=np.asarray([[0.,0.,0.],[1.,0.,0.],[2.,0.,0.],[1.,1.,0.]])
    faces=np.asarray([[0,1,3],[1,2,3]],dtype=np.int64)
    offsets=np.asarray([0,1,3,4,6],dtype=np.int64)
    bone_indices=np.asarray([0,0,1,1,0,1],dtype=np.int64)
    weights=np.asarray([1.,.25,.75,1.,.4,.6],dtype=np.float64)
    refs=[]
    actions=[]
    for bone, frame in enumerate(frames):
        for axis in range(3):
            cidx=3*bone+axis
            handle=hashlib.sha256(f"toy:{cidx}".encode()).hexdigest()
            refs.append({
                "kind":"NATIVE_POSE_BONE_LOCAL_ROTATION",
                "authored_capability":"KINEMATIC_PARAMETER__NOT_PHYSICAL_DOF_OR_JOINT_LIMIT",
                "axis_in_native_rest_frame":axis,
                "opaque_privileged_handle":handle,
                "units":"RADIAN",
                "source_lookup_privileged_only":{
                    "armature":"fixture","bone":frame["name"],
                    "axis":axis,"source_sha256":source_hash},
            })
            for amplitude_index, amplitude in enumerate((.002,.01,.05)):
                for sign in (-1,1):
                    actions.append({
                        "kind":"ISOLATED_KINEMATIC_PARAMETER_PERTURBATION",
                        "control_index_privileged":cidx,
                        "opaque_privileged_handle":handle,
                        "amplitude_index":amplitude_index,
                        "amplitude":amplitude,
                        "sign":sign,"units":"RADIAN",
                    })
    # Independent reference uses *each bone's individual* world posed matrix.
    observations=[]
    for action in actions:
        bone=action["control_index_privileged"]//3
        axis=action["control_index_privileged"]%3
        rot=_axis_matrix(axis,action["amplitude"]*action["sign"])
        posed=[]
        for idx, matrix in enumerate(mats):
            native=matrix.copy()
            if idx==bone: native=native@rot
            if idx==1:
                native=posed[0]@np.linalg.inv(mats[0])@native
            posed.append(native)
        deform=[posed[i]@np.linalg.inv(mats[i]) for i in range(2)]
        out=[]
        for vi,v in enumerate(rest):
            homo=np.append(v,1.)
            p=np.zeros(3)
            for off in range(offsets[vi],offsets[vi+1]):
                p+=weights[off]*(deform[bone_indices[off]]@homo)[:3]
            out.append(p)
        observations.append(out)
    observations=np.asarray(observations)
    if corrupt_response:
        observations[12,2,0]+=0.01
    npz=tmp_path/"source_fixture.npz"
    np.savez_compressed(npz,
        rest_vertices_source_world=rest,
        isolated_position=observations,
        faces=faces,
        source_weight_offsets=offsets,
        source_weight_values=weights,
        source_weight_bone_indices_privileged=bone_indices,
        source_weight_coverage=np.ones(len(rest),dtype=bool),
    )
    scene={
        "schema":"RealSaS.TeacherSceneIR.v2",
        "subject_key":"RESEARCH_EXPOSED_TOY_NOT_TRAINING",
        "source_sha256":source_hash,
        "student_input_allowed":False,
        "qualification_minted":False,
        "native_temporal_rest_compatible":False,
        "coordinate_contract":{"frame":"SOURCE_WORLD"},
        "constraint_capabilities":{"source_weight_lbs_equivalence":"NOT_QUALIFIED"},
        "geometry_payloads":[{"sha256":hashlib.sha256(npz.read_bytes()).hexdigest()}],
        "calibration":{"native_control_frames_privileged":frames,"actions":actions},
        "privileged_control_refs":refs,
    }
    teacher=tmp_path/"teacher.json"
    teacher.write_text(json.dumps(scene))
    return teacher,npz


def test_independent_source_fk_lbs_all_axes_and_oracle_product_deformer(tmp_path):
    teacher,npz=_toy(tmp_path)
    out=compare_stage47_source_fk_lbs_v1(
        teacher,npz,error_threshold_source_world=1e-9,
        product_probe_count=12,
        prepared_probe=True,
    )
    assert out["status"].startswith("PASS")
    assert out["action_count"]==36
    assert out["source_native_bone_count"]==2
    assert out["source_native_fk_lbs_max_vertex_error_world"]<1e-10
    probe=out["oracle_product_deformer_probe"]
    assert probe["product_deformer_invoked"] is True
    assert probe["probe_count"]==12
    assert probe["max_vertex_error_world"]<1e-10
    assert probe["prepared_kernel_probe"]["prepared_kernel_invoked"] is True
    assert probe["prepared_kernel_probe"]["prepared_once_for_all_actions"] is True
    assert probe["prepared_kernel_probe"]["max_parity_error_vs_legacy_world"]<1e-10
    assert probe["prepared_kernel_probe"]["max_error_vs_blender_world"]<1e-10
    assert probe["canonical_rig_minted"] is False
    assert out["source_to_canonical_geometry_binding_qualified"] is False


def test_corrupt_native_response_rejected_as_failed_comparison(tmp_path):
    teacher,npz=_toy(tmp_path,corrupt_response=True)
    result=compare_stage47_source_fk_lbs_v1(teacher,npz)
    assert result["status"].startswith("FAIL")
    assert result["source_native_fk_lbs_max_vertex_error_world"]>0.009


def test_corrupt_sha_and_teacher_control_binding_fail_closed(tmp_path):
    teacher,npz=_toy(tmp_path)
    with pytest.raises(Stage47CorrespondenceError,match="SHA_DRIFT"):
        compare_stage47_source_fk_lbs_v1(
            teacher,npz,expected_teacher_sha256="0"*64,
        )
    data=json.loads(teacher.read_text())
    data["privileged_control_refs"][2]["axis_in_native_rest_frame"]=0
    teacher.write_text(json.dumps(data))
    with pytest.raises(Stage47CorrespondenceError,match="CONTROL_FRAME_BINDING_DRIFT"):
        compare_stage47_source_fk_lbs_v1(teacher,npz)


def test_corrupt_parent_relationship_and_student_authority_fail_closed(tmp_path):
    teacher,npz=_toy(tmp_path)
    data=json.loads(teacher.read_text())
    data["calibration"]["native_control_frames_privileged"][0]["parent"]="branch"
    teacher.write_text(json.dumps(data))
    with pytest.raises(Stage47CorrespondenceError,match="PARENT_CYCLE"):
        compare_stage47_source_fk_lbs_v1(teacher,npz)
    data["calibration"]["native_control_frames_privileged"][0]["parent"]=None
    data["student_input_allowed"]=True
    teacher.write_text(json.dumps(data))
    with pytest.raises(Stage47CorrespondenceError,match="AUTHORITY_OR_GEOMETRY"):
        compare_stage47_source_fk_lbs_v1(teacher,npz)


def test_prepared_product_probe_requires_real_nonzero_product_probe_set(tmp_path):
    teacher,npz=_toy(tmp_path)
    with pytest.raises(Stage47CorrespondenceError,match="POLICY_INVALID"):
        compare_stage47_source_fk_lbs_v1(
            teacher,npz,prepared_probe=True,product_probe_count=0,
        )
