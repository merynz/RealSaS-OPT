#!/usr/bin/env python3
from __future__ import annotations

"""Bind sealed vNext posed samples to exact native animation timebase metadata.

The court reuses immutable authored-teacher NPZ/receipts and a metadata-only
Blender timebase probe. It verifies byte lineage and determines whether every
existing posed sample can be assigned an explicit action-local timestamp without
re-evaluating geometry. FBX importer action-name normalization is accepted only
when the probe binds it through the exact sealed `(action, armature)` receipt.

This court does not mint Truth qualification or training data.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

SCHEMA = "RealSaS.NativeTemporalTimebaseInventory.v1"
PROBE_SCHEMA = "RealSaS.NativeTemporalTimebaseProbe.v1"
TEACHER_SCHEMA = "RealSaS.IRISVNextAuthoredTeacherExport.v1"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _bound(root: Path, ref: dict, code: str) -> Path:
    path = (root / str(ref["relative"])).resolve()
    if not path.is_file():
        raise RuntimeError(code + "_MISSING")
    if _sha256(path) != str(ref["sha256"]):
        raise RuntimeError(code + "_HASH_DRIFT")
    return path


def _version_triplet(label: str) -> tuple[int, int, int]:
    match = re.search(r"(?<!\d)(\d+)\.(\d+)\.(\d+)(?!\d)", str(label))
    if match is None:
        raise RuntimeError("TEMPORAL_TIMEBASE_BLENDER_VERSION_UNPARSEABLE:" + str(label))
    return tuple(int(value) for value in match.groups())


def audit(request_path: Path, authority_root: Path, probe_dir: Path) -> dict:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    if request.get("schema") != "RealSaS.NativeTeacherReplayRequest.v1":
        raise RuntimeError("TEMPORAL_INVENTORY_REQUEST_SCHEMA")
    if request.get("training_allowed") is not False or request.get("new_interventions_allowed") is not False:
        raise RuntimeError("TEMPORAL_INVENTORY_REQUEST_AUTHORITY")
    subjects = tuple(request.get("subjects") or ())
    if len(subjects) != 8:
        raise RuntimeError("TEMPORAL_INVENTORY_EXPECTED_FIT8")
    requested_blender_version = str(request.get("blender_version") or "")
    requested_blender_triplet = _version_triplet(requested_blender_version)

    rows = []
    total_samples = 0
    total_actions = 0
    total_normalized_action_identities = 0
    for subject in subjects:
        subject_id = str(subject["subject_id"])
        _source = _bound(authority_root, subject["source"], "TEMPORAL_SOURCE_" + subject_id)
        admission = _bound(authority_root, subject["observation_admission"], "TEMPORAL_ADMISSION_" + subject_id)
        teacher = _bound(authority_root, subject["expected_teacher"], "TEMPORAL_TEACHER_" + subject_id)
        receipt_path = _bound(authority_root, subject["expected_receipt"], "TEMPORAL_RECEIPT_" + subject_id)
        probe_path = probe_dir / f"{subject_id}.timebase.json"
        if not probe_path.is_file():
            raise RuntimeError("TEMPORAL_TIMEBASE_PROBE_MISSING:" + subject_id)

        admission_value = json.loads(admission.read_text(encoding="utf-8"))
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        probe = json.loads(probe_path.read_text(encoding="utf-8"))
        if receipt.get("schema") != TEACHER_SCHEMA or receipt.get("subject_id") != subject_id:
            raise RuntimeError("TEMPORAL_TEACHER_RECEIPT_SCHEMA_OR_SUBJECT:" + subject_id)
        if probe.get("schema") != PROBE_SCHEMA or probe.get("subject_id") != subject_id:
            raise RuntimeError("TEMPORAL_TIMEBASE_PROBE_SCHEMA_OR_SUBJECT:" + subject_id)
        if receipt.get("teacher_only") is not True or receipt.get("canonical_M_G_W_minted") is not False:
            raise RuntimeError("TEMPORAL_TEACHER_AUTHORITY_BOUNDARY:" + subject_id)
        if probe.get("mesh_geometry_evaluated") is not False or probe.get("new_interventions_executed") is not False:
            raise RuntimeError("TEMPORAL_TIMEBASE_PROBE_SCOPE_VIOLATION:" + subject_id)
        if probe.get("training_submitted") is not False or probe.get("qualification_minted") is not False:
            raise RuntimeError("TEMPORAL_TIMEBASE_PROBE_AUTHORITY_VIOLATION:" + subject_id)
        if probe.get("source_sha256") != subject["source"]["sha256"]:
            raise RuntimeError("TEMPORAL_TIMEBASE_PROBE_SOURCE_DRIFT:" + subject_id)
        if probe.get("observation_admission_sha256") != subject["observation_admission"]["sha256"]:
            raise RuntimeError("TEMPORAL_TIMEBASE_PROBE_ADMISSION_DRIFT:" + subject_id)
        if probe.get("teacher_receipt_sha256") != subject["expected_receipt"]["sha256"]:
            raise RuntimeError("TEMPORAL_TIMEBASE_PROBE_RECEIPT_DRIFT:" + subject_id)
        probe_blender_version = str(probe.get("blender_version") or "")
        if _version_triplet(probe_blender_version) != requested_blender_triplet:
            raise RuntimeError("TEMPORAL_TIMEBASE_BLENDER_VERSION_DRIFT:" + subject_id)
        if receipt.get("source", {}).get("sha256") != subject["source"]["sha256"]:
            raise RuntimeError("TEMPORAL_TEACHER_SOURCE_DRIFT:" + subject_id)
        if receipt.get("observation_admission_sha256") != subject["observation_admission"]["sha256"]:
            raise RuntimeError("TEMPORAL_TEACHER_ADMISSION_DRIFT:" + subject_id)
        if receipt.get("output", {}).get("sha256") != subject["expected_teacher"]["sha256"]:
            raise RuntimeError("TEMPORAL_TEACHER_OUTPUT_RECEIPT_DRIFT:" + subject_id)

        expected_actions = tuple(map(str, admission_value.get("native_motion_inventory") or ()))
        if not expected_actions:
            raise RuntimeError("TEMPORAL_NATIVE_ACTIONS_EMPTY:" + subject_id)
        if tuple(map(str, receipt.get("native_motion_inventory_exported") or ())) != expected_actions:
            raise RuntimeError("TEMPORAL_TEACHER_ACTION_INVENTORY_DRIFT:" + subject_id)
        if tuple(map(str, probe.get("native_motion_inventory_teacher_identity") or ())) != expected_actions:
            raise RuntimeError("TEMPORAL_PROBE_TEACHER_ACTION_INVENTORY_DRIFT:" + subject_id)

        timebase = dict(probe.get("scene_timebase") or {})
        fps = float(timebase.get("frames_per_second", float("nan")))
        fps_base = float(timebase.get("render_fps_base", float("nan")))
        fps_integer = int(timebase.get("render_fps", 0))
        if not math.isfinite(fps) or fps <= 0 or not math.isfinite(fps_base) or fps_base <= 0 or fps_integer <= 0:
            raise RuntimeError("TEMPORAL_TIMEBASE_INVALID:" + subject_id)
        if not math.isclose(fps, fps_integer / fps_base, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError("TEMPORAL_TIMEBASE_RATIO_DRIFT:" + subject_id)

        action_rows_from_probe = tuple(probe.get("actions") or ())
        action_time = {str(row["teacher_action_identity"]): row for row in action_rows_from_probe}
        if set(action_time) != set(expected_actions) or len(action_time) != len(action_rows_from_probe):
            raise RuntimeError("TEMPORAL_TIMEBASE_ACTION_ROWS_DRIFT:" + subject_id)
        normalized_count = 0
        for teacher_action, metadata in action_time.items():
            kind = str(metadata.get("identity_binding") or "")
            if kind not in {"EXACT_ACTION_NAME", "SEALED_RECEIPT_ARMATURE_PREFIX_NORMALIZATION"}:
                raise RuntimeError("TEMPORAL_ACTION_IDENTITY_BINDING_UNSUPPORTED:" + subject_id + ":" + teacher_action)
            if kind != "EXACT_ACTION_NAME":
                normalized_count += 1
            if not str(metadata.get("blender_action_name") or ""):
                raise RuntimeError("TEMPORAL_BLENDER_ACTION_NAME_MISSING:" + subject_id + ":" + teacher_action)
            if not str(metadata.get("sealed_receipt_armature") or ""):
                raise RuntimeError("TEMPORAL_SEALED_ARMATURE_MISSING:" + subject_id + ":" + teacher_action)
        total_normalized_action_identities += normalized_count

        frame_rows = tuple(receipt.get("sampled_frames") or ())
        with np.load(teacher, allow_pickle=False) as payload:
            required = {
                "rest_vertices_source_world",
                "faces",
                "face_owner",
                "face_material",
                "posed_vertices_source_world",
            }
            if not required.issubset(payload.files):
                raise RuntimeError("TEMPORAL_TEACHER_ARRAYS_MISSING:" + subject_id)
            posed_shape = tuple(payload["posed_vertices_source_world"].shape)
            rest_shape = tuple(payload["rest_vertices_source_world"].shape)
            if len(posed_shape) != 3 or posed_shape[2] != 3 or rest_shape != posed_shape[1:]:
                raise RuntimeError("TEMPORAL_TEACHER_ARRAY_LAYOUT:" + subject_id)
            sample_count = int(posed_shape[0])

        if sample_count != int(receipt.get("sampled_frame_count", -1)) or sample_count != len(frame_rows):
            raise RuntimeError("TEMPORAL_TEACHER_FRAME_ARRAY_COUNT_DRIFT:" + subject_id)

        samples = []
        by_action: dict[str, list[float]] = {name: [] for name in expected_actions}
        for sample_index, frame_row in enumerate(frame_rows):
            action = str(frame_row.get("action") or "")
            if action not in action_time:
                raise RuntimeError("TEMPORAL_SAMPLE_ACTION_UNKNOWN:" + subject_id)
            frame = int(frame_row["frame"])
            metadata = action_time[action]
            receipt_armature = str(frame_row.get("armature") or "")
            if receipt_armature != str(metadata["sealed_receipt_armature"]):
                raise RuntimeError("TEMPORAL_SAMPLE_ARMATURE_BINDING_DRIFT:" + subject_id + ":" + action)
            origin = int(metadata["teacher_time_origin_frame"])
            first = int(metadata["teacher_first_frame"])
            last = int(metadata["teacher_last_frame"])
            if frame < first or frame > last:
                raise RuntimeError("TEMPORAL_SAMPLE_FRAME_OUTSIDE_ACTION:" + subject_id + ":" + action)
            timestamp = (float(frame) - float(origin)) / fps
            if not math.isfinite(timestamp) or timestamp < -1e-12:
                raise RuntimeError("TEMPORAL_SAMPLE_TIMESTAMP_INVALID:" + subject_id)
            timestamp = max(0.0, timestamp)
            by_action[action].append(timestamp)
            samples.append(
                {
                    "sample_index": sample_index,
                    "action": action,
                    "blender_action_name": str(metadata["blender_action_name"]),
                    "action_identity_binding": str(metadata["identity_binding"]),
                    "armature": receipt_armature,
                    "frame": frame,
                    "time_seconds": timestamp,
                }
            )

        action_rows = []
        for action in expected_actions:
            times = by_action[action]
            if not times:
                raise RuntimeError("TEMPORAL_ACTION_HAS_NO_TEACHER_SAMPLE:" + subject_id + ":" + action)
            if any(b <= a for a, b in zip(times, times[1:])):
                raise RuntimeError("TEMPORAL_ACTION_TIMESTAMPS_NOT_STRICTLY_INCREASING:" + subject_id + ":" + action)
            if abs(times[0]) > 1e-12:
                raise RuntimeError("TEMPORAL_ACTION_LOCAL_ZERO_MISSING:" + subject_id + ":" + action)
            action_rows.append(
                {
                    "action": action,
                    "blender_action_name": str(action_time[action]["blender_action_name"]),
                    "identity_binding": str(action_time[action]["identity_binding"]),
                    "sample_count": len(times),
                    "start_seconds": times[0],
                    "end_seconds": times[-1],
                }
            )

        total_samples += sample_count
        total_actions += len(expected_actions)
        rows.append(
            {
                "subject_id": subject_id,
                "source_sha256": subject["source"]["sha256"],
                "teacher_sha256": subject["expected_teacher"]["sha256"],
                "teacher_receipt_sha256": subject["expected_receipt"]["sha256"],
                "blender_version_request_label": requested_blender_version,
                "blender_version_probe_label": probe_blender_version,
                "blender_version_triplet": list(requested_blender_triplet),
                "frames_per_second": fps,
                "render_fps": fps_integer,
                "render_fps_base": fps_base,
                "sample_count": sample_count,
                "action_count": len(expected_actions),
                "normalized_action_identity_count": normalized_count,
                "actions": action_rows,
                "samples": samples,
                "temporal_binding_feasible_without_geometry_resampling": True,
            }
        )

    return {
        "schema": SCHEMA,
        "request_sha256": _sha256(request_path),
        "blender_version_request_label": requested_blender_version,
        "blender_version_triplet": list(requested_blender_triplet),
        "subject_count": len(rows),
        "action_count": total_actions,
        "normalized_action_identity_count": total_normalized_action_identities,
        "posed_sample_count": total_samples,
        "time_formula": "(frame-teacher_time_origin_frame)/frames_per_second",
        "time_unit": "SECOND",
        "geometry_resampled": False,
        "new_interventions_executed": False,
        "training_allowed": False,
        "qualification_minted": False,
        "temporal_response_candidate_production_feasible": all(
            row["temporal_binding_feasible_without_geometry_resampling"] for row in rows
        ),
        "court_status": "PASS_EXACT_TIMEBASE_BOUND" if rows else "FAIL_EMPTY",
        "subjects": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--authority-root", required=True, type=Path)
    parser.add_argument("--probe-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.request, args.authority_root, args.probe_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "subjects"}, sort_keys=True))


if __name__ == "__main__":
    main()
