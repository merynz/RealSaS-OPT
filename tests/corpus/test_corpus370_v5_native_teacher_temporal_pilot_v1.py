import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tools.corpus.run_corpus370_v5_native_teacher_temporal_pilot_v1 import (
    PASS, court_teacher,
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def subject(tmp_path, *, moved):
    source_sha = "1"*64
    admission_sha = "2"*64
    row={"subject_id":"subject_fixture"}
    source={"source_sha256":source_sha}
    bindings={"admission":{"sha256":admission_sha}}
    rest=np.asarray([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]],np.float64)
    posed=rest[None].copy()
    if moved:
        posed[0,0,0]=0.25
    path=tmp_path/"teacher.npz"
    np.savez_compressed(path,rest_vertices_source_world=rest,
        faces=np.asarray([[0,1,2]],np.int32),
        face_owner=np.asarray([0],np.int32),
        face_material=np.asarray([0],np.int32),
        posed_vertices_source_world=posed)
    payload={
        "schema":"RealSaS.IRISVNextAuthoredTeacherExport.v1",
        "transport_revision":4, "subject_id":"subject_fixture",
        "source":{"sha256":source_sha},
        "observation_admission_sha256":admission_sha,
        "teacher_only":True,
        "canonical_M_G_W_minted":False,
        "teacher_rig_skin_motion_allowed_as_inference_input":False,
        "output":{"sha256":digest(path)},
        "sampled_frame_count":1,
        "native_motion_inventory_exported":["Action_1"],
    }
    receipt=path.with_suffix(path.suffix+".json")
    receipt.write_text(json.dumps(payload))
    return row,source,bindings,path,receipt,payload


def test_measured_authored_motion_is_candidate_only(tmp_path):
    row,source,bindings,path,*_=subject(tmp_path,moved=True)
    result=court_teacher(row,source,bindings,tmp_path,path)
    assert result["status"]==PASS
    assert result["causal_response_interventions_executed"] is False
    assert result["student_teacher_id_leakage_allowed"] is False
    assert result["family_authority_minted"] is False
    assert result["training_authorized"] is False
    assert result["measured_max_vertex_displacement_source_units"] == pytest.approx(.25)


def test_static_authored_clip_masked_not_false_negative(tmp_path):
    row,source,bindings,path,*_=subject(tmp_path,moved=False)
    result=court_teacher(row,source,bindings,tmp_path,path)
    assert result["status"]=="MASKED_STATIC_AUTHORED_CLIP"
    assert result["measured_max_vertex_displacement_source_units"]==0


def test_rejects_teacher_provenance_leak(tmp_path):
    row,source,bindings,path,receipt,payload=subject(tmp_path,moved=True)
    payload["teacher_rig_skin_motion_allowed_as_inference_input"]=True
    receipt.write_text(json.dumps(payload))
    with pytest.raises(ValueError,match="SOURCE_BINDING"):
        court_teacher(row,source,bindings,tmp_path,path)


def test_rejects_corrupted_teacher_payload(tmp_path):
    row,source,bindings,path,receipt,payload=subject(tmp_path,moved=True)
    with path.open("ab") as f: f.write(b"corrupted")
    with pytest.raises(ValueError,match="SOURCE_BINDING"):
        court_teacher(row,source,bindings,tmp_path,path)
