#!/usr/bin/env python3
from __future__ import annotations

"""Build unqualified TEMPORAL_RESPONSE candidates from sealed authored clips.

Inputs are the exact vNext teacher NPZ/receipt bytes and the exact timebase
inventory produced by the Motion timebase court. No Blender evaluation occurs
here. Each authored action becomes one independent `[L,1,T,3]` response payload.
The writer never mints student supervision, Compiler authority, physical DOF,
impulse, secondary-motion physics, Registry admission or product qualification.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from models.iris.truth.temporal_response_v1 import (
    DYNAMIC_AUTHORITY,
    RESPONSE_SEMANTICS,
    build_temporal_response_candidate_v1,
)

SCHEMA = "RealSaS.NativeTemporalResponseCandidateManifest.v1"
ACTION_SCHEMA = "RealSaS.TemporalResponseCandidate.v1"
SUBJECT_SCHEMA = "RealSaS.SubjectTemporalResponseCandidates.v1"
EXPECTED_TIMEBASE_SCHEMA = "RealSaS.NativeTemporalTimebaseInventory.v1"
EXPECTED_REQUEST_SCHEMA = "RealSaS.NativeTeacherReplayRequest.v1"
EXPECTED_TEACHER_SCHEMA = "RealSaS.IRISVNextAuthoredTeacherExport.v1"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(value.dtype.str.encode())
    h.update(json.dumps(list(value.shape), separators=(",", ":")).encode())
    h.update(value.tobytes(order="C"))
    return h.hexdigest()


def _bound(root: Path, ref: dict, code: str) -> Path:
    path = (root / str(ref["relative"])).resolve()
    if not path.is_file():
        raise RuntimeError(code + "_MISSING")
    if _sha256(path) != str(ref["sha256"]):
        raise RuntimeError(code + "_HASH_DRIFT")
    return path


def _safe_subject(value: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9_.-]+", value) is None:
        raise RuntimeError("TEMPORAL_RESPONSE_SUBJECT_PATH_UNSAFE")
    return value


def _metadata_sha(value: dict) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def build(
    *,
    request_path: Path,
    authority_root: Path,
    timebase_inventory_path: Path,
    expected_timebase_sha256: str,
    output: Path,
) -> dict:
    request_path = request_path.resolve()
    authority_root = authority_root.resolve()
    timebase_inventory_path = timebase_inventory_path.resolve()
    output = output.resolve()
    if not timebase_inventory_path.is_file() or _sha256(timebase_inventory_path) != expected_timebase_sha256:
        raise RuntimeError("TEMPORAL_RESPONSE_TIMEBASE_INVENTORY_HASH_DRIFT")

    request = json.loads(request_path.read_text(encoding="utf-8"))
    timebase = json.loads(timebase_inventory_path.read_text(encoding="utf-8"))
    if request.get("schema") != EXPECTED_REQUEST_SCHEMA:
        raise RuntimeError("TEMPORAL_RESPONSE_REQUEST_SCHEMA")
    if request.get("training_allowed") is not False or request.get("new_interventions_allowed") is not False:
        raise RuntimeError("TEMPORAL_RESPONSE_REQUEST_AUTHORITY")
    if timebase.get("schema") != EXPECTED_TIMEBASE_SCHEMA or timebase.get("court_status") != "PASS_EXACT_TIMEBASE_BOUND":
        raise RuntimeError("TEMPORAL_RESPONSE_TIMEBASE_NOT_PASSED")
    if timebase.get("qualification_minted") is not False or timebase.get("training_allowed") is not False:
        raise RuntimeError("TEMPORAL_RESPONSE_TIMEBASE_AUTHORITY_BOUNDARY")
    if timebase.get("geometry_resampled") is not False or timebase.get("new_interventions_executed") is not False:
        raise RuntimeError("TEMPORAL_RESPONSE_TIMEBASE_SCOPE_DRIFT")
    if timebase.get("request_sha256") != _sha256(request_path):
        raise RuntimeError("TEMPORAL_RESPONSE_TIMEBASE_REQUEST_DRIFT")

    subjects = tuple(request.get("subjects") or ())
    timebase_by_subject = {str(row["subject_id"]): row for row in timebase.get("subjects") or ()}
    if len(subjects) != 8 or set(timebase_by_subject) != {str(row["subject_id"]) for row in subjects}:
        raise RuntimeError("TEMPORAL_RESPONSE_FIT8_SUBJECT_BINDING_DRIFT")

    output.mkdir(parents=True, exist_ok=False)
    subject_manifests = []
    total_actions = 0
    total_samples = 0
    total_response_values = 0

    for subject in subjects:
        subject_id = _safe_subject(str(subject["subject_id"]))
        teacher_path = _bound(authority_root, subject["expected_teacher"], "TEMPORAL_RESPONSE_TEACHER_" + subject_id)
        receipt_path = _bound(authority_root, subject["expected_receipt"], "TEMPORAL_RESPONSE_RECEIPT_" + subject_id)
        _bound(authority_root, subject["source"], "TEMPORAL_RESPONSE_SOURCE_" + subject_id)
        _bound(authority_root, subject["observation_admission"], "TEMPORAL_RESPONSE_ADMISSION_" + subject_id)
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("schema") != EXPECTED_TEACHER_SCHEMA or receipt.get("subject_id") != subject_id:
            raise RuntimeError("TEMPORAL_RESPONSE_TEACHER_RECEIPT_DRIFT:" + subject_id)
        if receipt.get("teacher_only") is not True or receipt.get("canonical_M_G_W_minted") is not False:
            raise RuntimeError("TEMPORAL_RESPONSE_TEACHER_AUTHORITY_BOUNDARY:" + subject_id)

        time_row = timebase_by_subject[subject_id]
        if time_row.get("teacher_sha256") != subject["expected_teacher"]["sha256"]:
            raise RuntimeError("TEMPORAL_RESPONSE_TIMEBASE_TEACHER_DRIFT:" + subject_id)
        if time_row.get("teacher_receipt_sha256") != subject["expected_receipt"]["sha256"]:
            raise RuntimeError("TEMPORAL_RESPONSE_TIMEBASE_RECEIPT_DRIFT:" + subject_id)
        samples = tuple(time_row.get("samples") or ())
        if len(samples) != int(time_row.get("sample_count", -1)):
            raise RuntimeError("TEMPORAL_RESPONSE_TIMEBASE_SAMPLE_COUNT_DRIFT:" + subject_id)
        receipt_rows = tuple(receipt.get("sampled_frames") or ())
        if len(receipt_rows) != len(samples):
            raise RuntimeError("TEMPORAL_RESPONSE_RECEIPT_SAMPLE_COUNT_DRIFT:" + subject_id)
        for index, (receipt_row, time_sample) in enumerate(zip(receipt_rows, samples)):
            if int(time_sample.get("sample_index", -1)) != index:
                raise RuntimeError("TEMPORAL_RESPONSE_SAMPLE_INDEX_DRIFT:" + subject_id)
            for key in ("action", "armature", "frame"):
                if time_sample.get(key) != receipt_row.get(key):
                    raise RuntimeError("TEMPORAL_RESPONSE_SAMPLE_LINEAGE_DRIFT:" + subject_id + ":" + key)

        with np.load(teacher_path, allow_pickle=False) as payload:
            required = {"rest_vertices_source_world", "faces", "posed_vertices_source_world"}
            if not required.issubset(payload.files):
                raise RuntimeError("TEMPORAL_RESPONSE_TEACHER_ARRAYS_MISSING:" + subject_id)
            rest = np.asarray(payload["rest_vertices_source_world"], dtype=np.float64)
            faces = np.asarray(payload["faces"], dtype=np.int64)
            posed = np.asarray(payload["posed_vertices_source_world"], dtype=np.float64)
        if posed.shape[0] != len(samples) or posed.shape[1:] != rest.shape:
            raise RuntimeError("TEMPORAL_RESPONSE_TEACHER_LAYOUT_DRIFT:" + subject_id)
        if not np.isfinite(rest).all() or not np.isfinite(posed).all():
            raise RuntimeError("TEMPORAL_RESPONSE_TEACHER_NUMERIC_INVALID:" + subject_id)

        subject_dir = output / subject_id
        subject_dir.mkdir()
        actions = tuple(map(str, receipt.get("native_motion_inventory_exported") or ()))
        if len(actions) != int(time_row.get("action_count", -1)):
            raise RuntimeError("TEMPORAL_RESPONSE_ACTION_COUNT_DRIFT:" + subject_id)
        action_records = []
        for action_index, action in enumerate(actions):
            candidate = build_temporal_response_candidate_v1(
                rest,
                posed,
                samples,
                action_identity=action,
            )
            stem = f"action_{action_index:04d}"
            npz_path = subject_dir / f"{stem}.npz"
            np.savez_compressed(
                npz_path,
                response=candidate.response,
                timestamps_seconds=candidate.timestamps_seconds,
                sample_validity=candidate.sample_validity,
                interval_validity=candidate.interval_validity,
            )
            arrays = candidate.array_contract()
            metadata = {
                "schema": ACTION_SCHEMA,
                "family": "TEMPORAL_RESPONSE",
                "subject_id": subject_id,
                "action_index": action_index,
                "teacher_action_identity": action,
                "condition_axis_semantics": candidate.condition_axis_semantics,
                "coordinate_frame": candidate.coordinate_frame,
                "time_unit": candidate.time_unit,
                "response_semantics": candidate.response_semantics,
                "dynamic_authority": candidate.dynamic_authority,
                "source_sha256": subject["source"]["sha256"],
                "teacher_sha256": subject["expected_teacher"]["sha256"],
                "teacher_receipt_sha256": subject["expected_receipt"]["sha256"],
                "timebase_inventory_sha256": expected_timebase_sha256,
                "rest_vertices_array_sha256": _array_sha256(rest),
                "faces_array_sha256": _array_sha256(faces),
                "payload_relative": str(npz_path.relative_to(output)),
                "payload_sha256": _sha256(npz_path),
                "arrays": arrays,
                "sample_count": candidate.frame_count,
                "locus_count": candidate.locus_count,
                "root_motion_preserved": True,
                "physical_dof_authority": False,
                "physical_impulse_authority": False,
                "secondary_motion_physics_authority": False,
                "qualification_minted": False,
                "student_target_allowed": False,
                "compiler_consequence_allowed": False,
            }
            metadata_path = subject_dir / f"{stem}.json"
            metadata_path.write_text(json.dumps(metadata, sort_keys=True, indent=2, allow_nan=False) + "\n")
            metadata["metadata_relative"] = str(metadata_path.relative_to(output))
            metadata["metadata_sha256"] = _sha256(metadata_path)
            action_records.append(metadata)
            total_actions += 1
            total_samples += candidate.frame_count
            total_response_values += int(candidate.response.size)

        subject_manifest = {
            "schema": SUBJECT_SCHEMA,
            "subject_id": subject_id,
            "source_sha256": subject["source"]["sha256"],
            "teacher_sha256": subject["expected_teacher"]["sha256"],
            "teacher_receipt_sha256": subject["expected_receipt"]["sha256"],
            "timebase_inventory_sha256": expected_timebase_sha256,
            "rest_vertices_array_sha256": _array_sha256(rest),
            "faces_array_sha256": _array_sha256(faces),
            "locus_count": int(len(rest)),
            "action_count": len(action_records),
            "posed_sample_count": int(len(samples)),
            "actions": action_records,
            "qualification_minted": False,
            "training_allowed": False,
            "student_target_allowed": False,
            "compiler_consequence_allowed": False,
        }
        subject_manifest_path = subject_dir / "SUBJECT_TEMPORAL_RESPONSE_MANIFEST_V1.json"
        subject_manifest_path.write_text(json.dumps(subject_manifest, sort_keys=True, indent=2, allow_nan=False) + "\n")
        subject_manifests.append(
            {
                "subject_id": subject_id,
                "manifest_relative": str(subject_manifest_path.relative_to(output)),
                "manifest_sha256": _sha256(subject_manifest_path),
                "action_count": len(action_records),
                "posed_sample_count": len(samples),
                "locus_count": len(rest),
            }
        )

    if total_actions != int(timebase.get("action_count", -1)) or total_samples != int(timebase.get("posed_sample_count", -1)):
        raise RuntimeError("TEMPORAL_RESPONSE_GLOBAL_COUNT_DRIFT")

    manifest = {
        "schema": SCHEMA,
        "family": "TEMPORAL_RESPONSE",
        "request_sha256": _sha256(request_path),
        "timebase_inventory_sha256": expected_timebase_sha256,
        "source_subject_count": len(subject_manifests),
        "action_count": total_actions,
        "posed_sample_count": total_samples,
        "response_scalar_count": total_response_values,
        "response_semantics": RESPONSE_SEMANTICS,
        "dynamic_authority": DYNAMIC_AUTHORITY,
        "condition_policy": "ONE_AUTHORED_ACTION_PER_TEMPORAL_SEQUENCE",
        "coordinate_frame": "SOURCE_WORLD",
        "time_unit": "SECOND",
        "root_motion_preserved": True,
        "geometry_resampled": False,
        "new_interventions_executed": False,
        "physical_dof_authority": False,
        "physical_impulse_authority": False,
        "secondary_motion_physics_authority": False,
        "qualification_minted": False,
        "student_target_allowed": False,
        "compiler_consequence_allowed": False,
        "training_allowed": False,
        "subjects": subject_manifests,
    }
    manifest_path = output / "TEMPORAL_RESPONSE_CANDIDATE_MANIFEST_V1.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n")
    result = dict(manifest)
    result["manifest_sha256"] = _sha256(manifest_path)
    result["manifest_path"] = str(manifest_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--authority-root", required=True, type=Path)
    parser.add_argument("--timebase-inventory", required=True, type=Path)
    parser.add_argument("--expected-timebase-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = build(
        request_path=args.request,
        authority_root=args.authority_root,
        timebase_inventory_path=args.timebase_inventory,
        expected_timebase_sha256=args.expected_timebase_sha256,
        output=args.output,
    )
    print(json.dumps({k: v for k, v in result.items() if k != "subjects"}, sort_keys=True))


if __name__ == "__main__":
    main()
