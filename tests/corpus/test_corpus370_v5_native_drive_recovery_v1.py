from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

from tools.corpus import run_corpus370_v5_native_drive_recovery_v1 as recovery


def fixture(tmp_path, *, status=None):
    sid = "subject_a"
    folder = tmp_path / "subjects" / sid
    folder.mkdir(parents=True)
    arrays = folder / "teacher.npz"
    arrays.write_bytes(b"source-truth-bytes-fixture")
    manifest = folder / "teacher.npz.json"
    manifest.write_text('{"source":"fixture"}')
    status = status or "PASS_SOURCE_AUTHORED_TEMPORAL_GEOMETRY_CANDIDATE__CAUSAL_RESPONSE_SEPARATE"
    court = folder / "TEMPORAL_TEACHER_COURT_V1.json"
    payload = {
        "status": status, "subject_id": sid,
        "source_sha256": "a" * 64, "training_authorized": False,
        "teacher_arrays": {"path": str(arrays.resolve()), "sha256": recovery.sha(arrays)},
        "teacher_manifest": {"path": str(manifest.resolve()), "sha256": recovery.sha(manifest)},
    }
    court.write_text(json.dumps(payload))
    state = {
        "schema": recovery.SCHEMA,
        "status": "PARTIAL_NATIVE_TEACHER_SWEEP__NO_TRAINING",
        "training_authorized": False,
        "causal_response_authority_minted": False,
        "tessa_teacher_authority_minted": False,
        "results": {
            sid: {**payload, "court_sha256": recovery.sha(court)},
        },
    }
    (tmp_path / recovery.INDEX).write_text(json.dumps(state))
    return state, arrays, manifest, court


def test_valid_source_courts_can_be_recovered_without_new_truth(tmp_path):
    state, *_ = fixture(tmp_path)
    value, digest = recovery.snapshot(tmp_path)
    assert value == state
    assert len(digest) == 64
    assert recovery.checked_result_files(tmp_path, state) == ["subject_a"]
    assert state["training_authorized"] is False


def test_mutated_teacher_bytes_are_not_valid_resume(tmp_path):
    fixture(tmp_path)
    (tmp_path / "subjects" / "subject_a" / "teacher.npz").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="NATIVE_RECOVERY_LOCAL_CHECKPOINT_INVALID"):
        recovery.snapshot(tmp_path)


def test_mutated_court_bytes_are_not_valid_resume(tmp_path):
    fixture(tmp_path)
    (tmp_path / "subjects" / "subject_a" / "TEMPORAL_TEACHER_COURT_V1.json").write_text("{}")
    with pytest.raises(ValueError, match="NATIVE_RECOVERY_LOCAL_CHECKPOINT_INVALID"):
        recovery.snapshot(tmp_path)


def test_reject_fake_training_authority(tmp_path):
    state, *_ = fixture(tmp_path)
    state["training_authorized"] = True
    with pytest.raises(ValueError, match="NONAUTHORITATIVE_SCOPE"):
        recovery.checked_result_files(tmp_path, state)


def test_reject_subject_path_escape(tmp_path):
    state, *_ = fixture(tmp_path)
    state["results"]["../escape"] = state["results"].pop("subject_a")
    with pytest.raises(ValueError, match="NATIVE_RECOVERY_SUBJECT_ID"):
        recovery.checked_result_files(tmp_path, state)


def test_recover_never_overwrites_more_recent_local_checkpoint(tmp_path, monkeypatch):
    fixture(tmp_path)
    monkeypatch.setattr(recovery.subprocess, "run",
                        lambda *a, **kw: pytest.fail("Existing local checkpoint must not fetch remote"))
    monkeypatch.setattr(recovery, "rclone",
                        lambda *a, **kw: pytest.fail("Local state must never be downgraded"))
    recovery.recover_remote(tmp_path, "gdrive:corpus370_truth_v5_native_temporal_v4/pinned")


def test_frozen_checkpoint_remote_index_uploaded_last(tmp_path, monkeypatch):
    state, *_ = fixture(tmp_path)
    observed = []
    original = (tmp_path / recovery.INDEX).read_bytes()

    def fake_rclone(cmd, **kwargs):
        observed.append(list(cmd))
        if cmd[0] == "copy":
            # Simulate producer advancing live index after remote folder copy.
            live = json.loads((tmp_path / recovery.INDEX).read_text())
            live["status"] = "UPDATED_BY_CONCURRENT_WORKER"
            (tmp_path / recovery.INDEX).write_text(json.dumps(live))
        if cmd[0] == "copyto":
            assert Path(cmd[1]).read_bytes() == original

    monkeypatch.setattr(recovery, "rclone", fake_rclone)
    subjects, digest = recovery.sync_progress(
        tmp_path, "gdrive:corpus370_truth_v5_native_temporal_v4/pinned",
        set(), None)
    assert subjects == {"subject_a"}
    assert len(digest) == 64
    assert [item[0] for item in observed] == ["copy", "copyto"]


def test_court_failure_remains_explicit_without_teacher_files(tmp_path):
    state, *_ = fixture(tmp_path)
    sid = "subject_a"
    folder = tmp_path / "subjects" / sid
    court = folder / "TEMPORAL_TEACHER_COURT_V1.json"
    payload = json.loads(court.read_text())
    payload["status"] = recovery.FAIL_STATE
    payload.pop("teacher_arrays")
    payload.pop("teacher_manifest")
    court.write_text(json.dumps(payload))
    state["results"][sid] = {**payload, "court_sha256": recovery.sha(court)}
    assert recovery.checked_result_files(tmp_path, state) == [sid]
