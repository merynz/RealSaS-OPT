#!/usr/bin/env python3
"""Build Truth139 IRIS response evidence from a sealed Stage47 TeacherSceneIR.v2.

This producer is deliberately renderer-independent.  It consumes the exact
observation-rest Stage47 isolated-response tensor and derives continuous,
semantics-free surface response statistics.  It does not use source joint names,
part classes, material ids, source skin weights as labels, Stage48 renderer
measurements, student likelihoods, or product authority.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from models.iris.truth.presentation_response_v1 import build_surface_response_arrays_v1

SCHEMA = "RealSaS.Truth139IrisResponseEvidence.v1"
STATUS = "PASS_IRIS_RESPONSE_EVIDENCE_CANDIDATE__INDEPENDENT_COURT_REQUIRED"
CONTRACT = "canonical/TRUTH139_IRIS_RESPONSE_EVIDENCE_TEACHER_V1.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def bound_ref(ref: dict, code: str) -> Path:
    path = Path(str(ref.get("path") or "")).resolve()
    expected = str(ref.get("sha256") or "")
    if not path.is_file() or len(expected) != 64 or sha256(path) != expected:
        raise ValueError(code)
    return path


def geometry_ref(scene: dict) -> dict:
    matches = [
        dict(row)
        for row in scene.get("geometry_payloads") or ()
        if Path(str(row.get("path") or "")).name == "geometry_and_response.npz"
    ]
    if len(matches) != 1:
        raise ValueError("TRUTH139_RESPONSE_STAGE47_GEOMETRY_PAYLOAD_CARDINALITY")
    return matches[0]


def save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    clean = {}
    for name, value in arrays.items():
        arr = np.asarray(value)
        if arr.dtype == object:
            raise ValueError("TRUTH139_RESPONSE_OBJECT_ARRAY_FORBIDDEN:" + name)
        if np.issubdtype(arr.dtype, np.floating) and not np.isfinite(arr).all():
            raise ValueError("TRUTH139_RESPONSE_NONFINITE_ARRAY:" + name)
        clean[name] = arr
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **clean)


def build(*, subject_id: str, teacher_scene: Path, output_dir: Path) -> dict:
    teacher_scene = teacher_scene.resolve()
    output_dir = output_dir.resolve()
    scene = json.loads(teacher_scene.read_text())
    if scene.get("schema") != "RealSaS.TeacherSceneIR.v2":
        raise ValueError("TRUTH139_RESPONSE_STAGE47_SCHEMA")
    if scene.get("subject_key") != subject_id:
        raise ValueError("TRUTH139_RESPONSE_SUBJECT_DRIFT")
    if scene.get("rest_contract") != "EXACT_ADMITTED_OBSERVATION_IMPORT_GEOMETRY":
        raise ValueError("TRUTH139_RESPONSE_REST_CONTRACT")
    binding = dict(scene.get("observation_rest_binding") or {})
    if binding.get("status") != "PASS_EXACT_ADMITTED_IMPORT_GEOMETRY_REPLAY":
        raise ValueError("TRUTH139_RESPONSE_REST_REPLAY_NOT_QUALIFIED")
    if scene.get("authority") != "UNQUALIFIED_DERIVED_MEASUREMENTS":
        raise ValueError("TRUTH139_RESPONSE_STAGE47_AUTHORITY_DRIFT")
    if scene.get("qualification_minted") is not False:
        raise ValueError("TRUTH139_RESPONSE_STAGE47_QUALIFICATION_DRIFT")

    gref = geometry_ref(scene)
    geometry_path = bound_ref(gref, "TRUTH139_RESPONSE_STAGE47_GEOMETRY_DRIFT")
    with np.load(geometry_path, allow_pickle=False) as z:
        required = {
            "rest_vertices_source_world",
            "faces",
            "isolated_position",
            "pose_identity_baseline",
            "sham_before",
            "sham_after",
            "root_control_position",
        }
        if not required.issubset(set(z.files)):
            raise ValueError("TRUTH139_RESPONSE_STAGE47_ARRAYS_MISSING")
        rest = np.asarray(z["rest_vertices_source_world"], dtype=np.float64)
        faces = np.asarray(z["faces"], dtype=np.int64)
        isolated = np.asarray(z["isolated_position"], dtype=np.float64)

    actions = list((scene.get("calibration") or {}).get("actions") or ())
    if len(actions) != isolated.shape[0] or not actions:
        raise ValueError("TRUTH139_RESPONSE_ACTION_BINDING")

    arrays = build_surface_response_arrays_v1(rest, faces, isolated, actions)
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays_path = output_dir / "IRIS_RESPONSE_EVIDENCE_V1.npz"
    save_npz(arrays_path, arrays)

    face_valid = np.asarray(arrays["surface_face_valid"], dtype=np.bool_)
    pair_valid = np.asarray(arrays["response_pair_valid"], dtype=np.bool_)
    nonrigid = np.asarray(arrays["surface_nonrigid_response"], dtype=np.float64)
    energy = np.asarray(arrays["surface_response_energy"], dtype=np.float64)
    if face_valid.shape != (len(faces),) or nonrigid.shape != face_valid.shape or energy.shape != face_valid.shape:
        raise ValueError("TRUTH139_RESPONSE_OUTPUT_LAYOUT")

    payload = {
        "schema": SCHEMA,
        "date": "2026-10-30",
        "status": STATUS,
        "subject_id": subject_id,
        "contract": CONTRACT,
        "source": {
            "teacher_scene": {"path": str(teacher_scene), "sha256": sha256(teacher_scene)},
            "teacher_scene_job_sha256": str(scene.get("job_sha256") or ""),
            "geometry_and_response": {"path": str(geometry_path), "sha256": sha256(geometry_path)},
            "observation_rest_binding": binding,
        },
        "output": {"path": str(arrays_path), "sha256": sha256(arrays_path)},
        "measurement_policy": {
            "action_measure": "CONTROL_BALANCED__EQUAL_MASS_PER_CONTROL_THEN_EQUAL_MASS_WITHIN_CONTROL",
            "semantic_part_labels_used": False,
            "source_skin_weights_used_as_labels": False,
            "renderer_measurement_used": False,
            "appearance_measurement_used": False,
            "student_likelihood_encoded": False,
            "stage48_required": False,
        },
        "counts": {
            "face_count": int(len(faces)),
            "valid_face_count": int(np.count_nonzero(face_valid)),
            "action_count": int(len(actions)),
            "response_pair_count": int(len(arrays["response_pair_face"])),
            "valid_response_pair_count": int(np.count_nonzero(pair_valid)),
        },
        "diagnostics": {
            "maximum_nonrigid_response": float(np.max(nonrigid[face_valid], initial=0.0)),
            "maximum_response_energy_source_units": float(np.max(energy[face_valid], initial=0.0)),
            "nonzero_response_face_count": int(np.count_nonzero(face_valid & ((nonrigid > 0.0) | (energy > 0.0)))),
        },
        "training_performed": False,
        "split_created": False,
        "qualification_minted": False,
    }
    payload["response_evidence_identity_sha256"] = digest(payload)
    manifest = output_dir / "IRIS_RESPONSE_EVIDENCE_MANIFEST.json"
    manifest.write_text(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject-id", required=True)
    parser.add_argument("--teacher-scene", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    result = build(subject_id=args.subject_id, teacher_scene=args.teacher_scene, output_dir=args.output_dir)
    print(json.dumps({
        "status": result["status"],
        "subject_id": result["subject_id"],
        "counts": result["counts"],
        "response_evidence_identity_sha256": result["response_evidence_identity_sha256"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
