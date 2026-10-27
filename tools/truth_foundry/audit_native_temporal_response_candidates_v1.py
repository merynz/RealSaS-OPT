#!/usr/bin/env python3
from __future__ import annotations

"""Independent replay court for unqualified native TEMPORAL_RESPONSE candidates."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

COURT_SCHEMA = "RealSaS.NativeTemporalResponseCandidateCourt.v1"
MANIFEST_SCHEMA = "RealSaS.NativeTemporalResponseCandidateManifest.v1"
ACTION_SCHEMA = "RealSaS.TemporalResponseCandidate.v1"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _bound(root: Path, ref: dict, code: str) -> Path:
    path = (root / str(ref["relative"])).resolve()
    if not path.is_file() or _sha256(path) != str(ref["sha256"]):
        raise RuntimeError(code)
    return path


def audit(
    *,
    candidate_root: Path,
    request_path: Path,
    authority_root: Path,
    timebase_inventory_path: Path,
    expected_timebase_sha256: str,
) -> dict:
    candidate_root = candidate_root.resolve()
    request = json.loads(request_path.read_text())
    if _sha256(timebase_inventory_path) != expected_timebase_sha256:
        raise RuntimeError("TEMPORAL_RESPONSE_COURT_TIMEBASE_HASH_DRIFT")
    timebase = json.loads(timebase_inventory_path.read_text())
    timebase_by_subject = {row["subject_id"]: row for row in timebase["subjects"]}
    subject_request = {row["subject_id"]: row for row in request["subjects"]}

    manifest_path = candidate_root / "TEMPORAL_RESPONSE_CANDIDATE_MANIFEST_V1.json"
    if not manifest_path.is_file():
        raise RuntimeError("TEMPORAL_RESPONSE_COURT_MANIFEST_MISSING")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise RuntimeError("TEMPORAL_RESPONSE_COURT_MANIFEST_SCHEMA")
    if manifest.get("timebase_inventory_sha256") != expected_timebase_sha256:
        raise RuntimeError("TEMPORAL_RESPONSE_COURT_TIMEBASE_BINDING_DRIFT")
    for key in (
        "physical_dof_authority",
        "physical_impulse_authority",
        "secondary_motion_physics_authority",
        "qualification_minted",
        "student_target_allowed",
        "compiler_consequence_allowed",
        "training_allowed",
    ):
        if manifest.get(key) is not False:
            raise RuntimeError("TEMPORAL_RESPONSE_COURT_AUTHORITY_VIOLATION:" + key)
    if manifest.get("geometry_resampled") is not False or manifest.get("new_interventions_executed") is not False:
        raise RuntimeError("TEMPORAL_RESPONSE_COURT_SCOPE_VIOLATION")

    total_actions = 0
    total_samples = 0
    max_response_abs_error = 0.0
    per_subject = []
    for subject_row in manifest.get("subjects") or ():
        subject_id = str(subject_row["subject_id"])
        if subject_id not in subject_request or subject_id not in timebase_by_subject:
            raise RuntimeError("TEMPORAL_RESPONSE_COURT_SUBJECT_UNKNOWN:" + subject_id)
        req = subject_request[subject_id]
        teacher_path = _bound(authority_root, req["expected_teacher"], "TEMPORAL_RESPONSE_COURT_TEACHER_DRIFT")
        subject_manifest_path = candidate_root / subject_row["manifest_relative"]
        if _sha256(subject_manifest_path) != subject_row["manifest_sha256"]:
            raise RuntimeError("TEMPORAL_RESPONSE_COURT_SUBJECT_MANIFEST_DRIFT:" + subject_id)
        subject_manifest = json.loads(subject_manifest_path.read_text())
        time_row = timebase_by_subject[subject_id]
        samples = tuple(time_row["samples"])

        with np.load(teacher_path, allow_pickle=False) as source:
            rest = np.asarray(source["rest_vertices_source_world"], dtype=np.float64)
            posed = np.asarray(source["posed_vertices_source_world"], dtype=np.float64)

        subject_actions = 0
        subject_samples = 0
        for action_meta in subject_manifest.get("actions") or ():
            if action_meta.get("schema") != ACTION_SCHEMA:
                raise RuntimeError("TEMPORAL_RESPONSE_COURT_ACTION_SCHEMA:" + subject_id)
            for key in (
                "physical_dof_authority",
                "physical_impulse_authority",
                "secondary_motion_physics_authority",
                "qualification_minted",
                "student_target_allowed",
                "compiler_consequence_allowed",
            ):
                if action_meta.get(key) is not False:
                    raise RuntimeError("TEMPORAL_RESPONSE_COURT_ACTION_AUTHORITY:" + subject_id + ":" + key)
            action = str(action_meta["teacher_action_identity"])
            action_rows = [row for row in samples if str(row["action"]) == action]
            action_rows.sort(key=lambda row: int(row["sample_index"]))
            indices = np.asarray([int(row["sample_index"]) for row in action_rows], dtype=np.int64)
            expected_times = np.asarray([float(row["time_seconds"]) for row in action_rows], dtype=np.float64)
            expected_response = np.transpose(posed[indices] - rest[None, :, :], (1, 0, 2))[:, None, :, :]

            payload_path = candidate_root / action_meta["payload_relative"]
            if _sha256(payload_path) != action_meta["payload_sha256"]:
                raise RuntimeError("TEMPORAL_RESPONSE_COURT_PAYLOAD_HASH_DRIFT:" + subject_id + ":" + action)
            metadata_path = candidate_root / action_meta["metadata_relative"]
            if _sha256(metadata_path) != action_meta["metadata_sha256"]:
                raise RuntimeError("TEMPORAL_RESPONSE_COURT_METADATA_HASH_DRIFT:" + subject_id + ":" + action)
            with np.load(payload_path, allow_pickle=False) as payload:
                response = np.asarray(payload["response"], dtype=np.float64)
                times = np.asarray(payload["timestamps_seconds"], dtype=np.float64)
                sample_validity = np.asarray(payload["sample_validity"], dtype=bool)
                interval_validity = np.asarray(payload["interval_validity"], dtype=bool)
            if response.shape != expected_response.shape or not np.array_equal(response, expected_response):
                error = float(np.max(np.abs(response - expected_response))) if response.shape == expected_response.shape else float("inf")
                raise RuntimeError("TEMPORAL_RESPONSE_COURT_RESPONSE_REPLAY_DRIFT:" + subject_id + ":" + action + f":{error}")
            if not np.array_equal(times, expected_times):
                raise RuntimeError("TEMPORAL_RESPONSE_COURT_TIMESTAMP_REPLAY_DRIFT:" + subject_id + ":" + action)
            if sample_validity.shape != response.shape[:-1] or not sample_validity.all():
                raise RuntimeError("TEMPORAL_RESPONSE_COURT_SAMPLE_VALIDITY:" + subject_id + ":" + action)
            if interval_validity.shape != (len(rest), 1, max(0, len(times) - 1)) or not interval_validity.all():
                raise RuntimeError("TEMPORAL_RESPONSE_COURT_INTERVAL_VALIDITY:" + subject_id + ":" + action)
            if len(times) and abs(float(times[0])) > 1.0e-12:
                raise RuntimeError("TEMPORAL_RESPONSE_COURT_LOCAL_ZERO:" + subject_id + ":" + action)
            if len(times) > 1 and not np.all(np.diff(times) > 0):
                raise RuntimeError("TEMPORAL_RESPONSE_COURT_TIME_ORDER:" + subject_id + ":" + action)
            max_response_abs_error = max(max_response_abs_error, float(np.max(np.abs(response - expected_response), initial=0.0)))
            subject_actions += 1
            subject_samples += len(times)

        if subject_actions != int(subject_row["action_count"]) or subject_samples != int(subject_row["posed_sample_count"]):
            raise RuntimeError("TEMPORAL_RESPONSE_COURT_SUBJECT_COUNT_DRIFT:" + subject_id)
        total_actions += subject_actions
        total_samples += subject_samples
        per_subject.append({"subject_id": subject_id, "action_count": subject_actions, "sample_count": subject_samples})

    if total_actions != int(manifest["action_count"]) or total_samples != int(manifest["posed_sample_count"]):
        raise RuntimeError("TEMPORAL_RESPONSE_COURT_GLOBAL_COUNT_DRIFT")
    if total_actions != int(timebase["action_count"]) or total_samples != int(timebase["posed_sample_count"]):
        raise RuntimeError("TEMPORAL_RESPONSE_COURT_TIMEBASE_COUNT_DRIFT")

    return {
        "schema": COURT_SCHEMA,
        "candidate_manifest_sha256": _sha256(manifest_path),
        "timebase_inventory_sha256": expected_timebase_sha256,
        "subject_count": len(per_subject),
        "action_count": total_actions,
        "posed_sample_count": total_samples,
        "max_response_abs_replay_error": max_response_abs_error,
        "all_interval_valid": True,
        "authored_action_sequences_independent": True,
        "physical_dof_authority": False,
        "physical_impulse_authority": False,
        "secondary_motion_physics_authority": False,
        "qualification_minted": False,
        "student_target_allowed": False,
        "compiler_consequence_allowed": False,
        "training_allowed": False,
        "court_status": "PASS_EXACT_TEMPORAL_RESPONSE_REPLAY",
        "subjects": per_subject,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-root", required=True, type=Path)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--authority-root", required=True, type=Path)
    parser.add_argument("--timebase-inventory", required=True, type=Path)
    parser.add_argument("--expected-timebase-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = audit(
        candidate_root=args.candidate_root,
        request_path=args.request,
        authority_root=args.authority_root,
        timebase_inventory_path=args.timebase_inventory,
        expected_timebase_sha256=args.expected_timebase_sha256,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "subjects"}, sort_keys=True))


if __name__ == "__main__":
    main()
