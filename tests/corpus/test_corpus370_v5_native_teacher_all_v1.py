from __future__ import annotations

import hashlib
import json

import pytest

from tools.corpus import run_corpus370_v5_native_teacher_all_v1 as worker


def fake_cohort(tmp_path,monkeypatch):
    observations=[];failures=[];roster=[]
    by_admission={}
    for i in range(370):
        sid=f"subject_{i:03d}"
        source_sha=hashlib.sha256(sid.encode()).hexdigest()
        roster.append({"subject_id":sid,"source_sha256":source_sha,
                       "source_relative_path":f"{sid}.gltf"})
        if i>=352:
            failures.append({"subject_id":sid,"source_sha256":source_sha})
            continue
        observations.append({"subject_id":sid,"source_sha256":source_sha})
        path=tmp_path/f"{sid}.json"
        path.write_text(json.dumps({
            "subject_id":sid,"source_sha256":source_sha,
            "status":"PASS_IRIS_BIND_OBSERVATIONS",
            "native_motion_inventory":["Action_1"] if i%3==0 else [],
        }))
        by_admission[sid]={"admission":{"path":str(path),"sha256":worker.sha(path)}}
    monkeypatch.setattr(worker,"row_input",lambda row:by_admission[row["subject_id"]])
    v5={"schema":"RealSaS.Corpus370PreparedObservationPackets.v5",
        "subject_count":370,"prepared_subject_count":352,"failure_count":18,
        "training_authorized":False,"subjects":observations,"failures":failures}
    sources={"schema":"RealSaS.Corpus370ExactSourceRoster.v1",
             "subject_count":370,"training_authorized":False,"subjects":roster}
    return v5,sources


def test_complete_native_and_missing_animation_accounting(tmp_path,monkeypatch):
    v5,sources=fake_cohort(tmp_path,monkeypatch)
    native,masked,excluded=worker.roster(v5,sources)
    assert len(native)==118
    assert len(masked)==234
    assert len(excluded)==18
    assert len(native)+len(masked)+len(excluded)==370
    assert all(row["authored_action_count"]==1 for row in native)
    assert all(row["status"]=="MASKED_NO_AUTHORED_NATIVE_ANIMATION" for row in masked)


def test_fail_closed_on_source_identity_drift(tmp_path,monkeypatch):
    v5,sources=fake_cohort(tmp_path,monkeypatch)
    sources["subjects"][0]["source_sha256"]="0"*64
    with pytest.raises(ValueError,match="ALL_NATIVE_SOURCE_IDENTITY"):
        worker.roster(v5,sources)


def test_fail_closed_on_duplicate_native_inventory(tmp_path,monkeypatch):
    v5,sources=fake_cohort(tmp_path,monkeypatch)
    path=tmp_path/"subject_000.json"
    d=json.loads(path.read_text())
    d["native_motion_inventory"]=["Idle","Idle"]
    path.write_text(json.dumps(d))
    with pytest.raises(ValueError,match="ALL_NATIVE_DUPLICATE_ACTION"):
        worker.roster(v5,sources)


def test_fail_closed_on_observation_failure_disappearance(tmp_path,monkeypatch):
    v5,sources=fake_cohort(tmp_path,monkeypatch)
    v5["failures"].pop()
    with pytest.raises(ValueError,match="ALL_NATIVE_COMPLETE_SOURCE_PARTITION"):
        worker.roster(v5,sources)


def test_receipt_stable_replay_ignores_timing_but_not_teacher_hash():
    original={"status":"PASS_SOURCE_AUTHORED_TEMPORAL_GEOMETRY_CANDIDATE__CAUSAL_RESPONSE_SEPARATE",
              "teacher_arrays":{"sha256":"1"*64},"elapsed_seconds":1.0}
    other={**original,"elapsed_seconds":200}
    assert worker._stable_receipt(original)==worker._stable_receipt(other)
    other["teacher_arrays"]={"sha256":"2"*64}
    assert worker._stable_receipt(original)!=worker._stable_receipt(other)
