#!/usr/bin/env python3
from __future__ import annotations

"""Measure temporal relative-transport evidence on sealed FIT8 authored motion.

Privileged source skin weights are used only by this outer evaluator to choose a
small deterministic set of mechanically informative surface pairs. They never
enter the temporal evidence kernel or the Compiler policy gate. No pair is
assigned an expected RIGID/NONRIGID label by this audit; the measured authored
motion decides the status.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from models.iris.truth.mechanics_oracle_v1 import (
    RelativeTransportPolicyV1,
    SurfacePairQueryV1,
)
from models.iris.truth.motion_temporal_oracle_v1 import (
    measure_temporal_relative_transport_v1,
)

SCHEMA = "RealSaS.MotionTemporalStaticPolicyOracleCourt.v1"
CANDIDATE_SCHEMA = "RealSaS.NativeTemporalResponseCandidateManifest.v1"
SUBJECT_SCHEMA = "RealSaS.SubjectTemporalResponseCandidates.v1"


def _array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(value.dtype.str.encode())
    h.update(json.dumps(list(value.shape), separators=(",", ":")).encode())
    h.update(value.tobytes(order="C"))
    return h.hexdigest()


def _load_calibration_subjects(root: Path):
    rows = {}
    for scene_path in sorted(root.rglob("TEACHER_SCENE.json")):
        scene = json.loads(scene_path.read_text())
        subject = str(scene.get("subject_key") or "")
        response_path = scene_path.with_name("geometry_and_response.npz")
        if not subject or not response_path.is_file():
            raise RuntimeError("MOTION_TEMPORAL_CALIBRATION_SUBJECT_INVALID")
        if subject in rows:
            raise RuntimeError("MOTION_TEMPORAL_CALIBRATION_SUBJECT_DUPLICATE:" + subject)
        with np.load(response_path, allow_pickle=False) as payload:
            arrays = {key: np.asarray(payload[key]) for key in payload.files}
        rows[subject] = (scene, arrays)
    if len(rows) != 8:
        raise RuntimeError(f"MOTION_TEMPORAL_EXPECTED_FIT8_CALIBRATION:{len(rows)}")
    return rows


def _strict_reference_faces(arrays: dict):
    rest = np.asarray(arrays["rest_vertices_source_world"], dtype=np.float64)
    faces = np.asarray(arrays["faces"], dtype=np.int64)
    offsets = np.asarray(arrays["source_weight_offsets"], dtype=np.int64)
    indices = np.asarray(arrays["source_weight_bone_indices_privileged"], dtype=np.int64)
    values = np.asarray(arrays["source_weight_values"], dtype=np.float64)
    if offsets.shape != (len(rest) + 1,) or offsets[0] != 0 or offsets[-1] != len(indices) or len(indices) != len(values):
        raise RuntimeError("MOTION_TEMPORAL_WEIGHT_CSR_INVALID")

    exact = np.zeros(len(rest), dtype=bool)
    dominant = np.full(len(rest), -1, dtype=np.int64)
    for vertex in range(len(rest)):
        lo, hi = int(offsets[vertex]), int(offsets[vertex + 1])
        if hi <= lo:
            continue
        local_values = values[lo:hi]
        local_indices = indices[lo:hi]
        best = int(np.argmax(local_values))
        dominant[vertex] = int(local_indices[best])
        exact[vertex] = len(local_values) == 1 and abs(float(local_values[0]) - 1.0) <= 1.0e-6

    tri = rest[faces]
    edge = np.stack(
        (
            np.linalg.norm(tri[:, 1] - tri[:, 0], axis=1),
            np.linalg.norm(tri[:, 2] - tri[:, 1], axis=1),
            np.linalg.norm(tri[:, 0] - tri[:, 2], axis=1),
        ),
        axis=1,
    )
    max_edge = np.max(edge, axis=1)
    area2 = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    quality = np.divide(area2, max_edge * max_edge, out=np.zeros_like(area2), where=max_edge > 0)
    scale = float(np.linalg.norm(rest.max(axis=0) - rest.min(axis=0)))

    reference = np.full(len(faces), -1, dtype=np.int64)
    strict = np.zeros(len(faces), dtype=bool)
    for face_index, face in enumerate(faces):
        if exact[face].all() and len({int(dominant[v]) for v in face}) == 1:
            strict[face_index] = True
            reference[face_index] = dominant[face[0]]
    robust = strict & (quality > 0.05) & (max_edge > scale * 1.0e-5)
    return rest, faces, robust, reference, scale


def _pair_queries(rest, faces, robust, reference, *, maximum_each=8):
    centres = rest[faces].mean(axis=1)
    positives = []
    representatives = []
    groups = sorted({int(v) for v in reference[robust]})
    for group in groups:
        members = np.where(robust & (reference == group))[0]
        if len(members):
            representatives.append(int(members[len(members) // 2]))
        if len(members) >= 2 and len(positives) < maximum_each:
            local = centres[members]
            first = int(np.argmax(np.linalg.norm(local - local.mean(axis=0), axis=1)))
            second = int(np.argmax(np.linalg.norm(local - local[first], axis=1)))
            if first != second:
                positives.append((int(members[first]), int(members[second]), "SAME_EXACT_SOURCE_INFLUENCE_REFERENCE"))
    negatives = []
    for i, left in enumerate(representatives):
        for right in representatives[i + 1 :]:
            negatives.append((left, right, "DISTINCT_EXACT_SOURCE_INFLUENCE_REFERENCE"))
            if len(negatives) >= maximum_each:
                break
        if len(negatives) >= maximum_each:
            break
    return positives + negatives


def _split(frame_count: int):
    if frame_count < 3:
        return None
    discovery_count = max(2, int(np.ceil(frame_count * (2.0 / 3.0))))
    discovery_count = min(discovery_count, frame_count - 1)
    return tuple(range(discovery_count)), tuple(range(discovery_count, frame_count))


def audit(*, calibration_root: Path, candidate_root: Path):
    calibration = _load_calibration_subjects(calibration_root)
    manifest_path = candidate_root / "TEMPORAL_RESPONSE_CANDIDATE_MANIFEST_V1.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != CANDIDATE_SCHEMA:
        raise RuntimeError("MOTION_TEMPORAL_CANDIDATE_SCHEMA")
    if manifest.get("qualification_minted") is not False or manifest.get("training_allowed") is not False:
        raise RuntimeError("MOTION_TEMPORAL_CANDIDATE_AUTHORITY_BOUNDARY")
    if manifest.get("physical_dof_authority") is not False or manifest.get("secondary_motion_physics_authority") is not False:
        raise RuntimeError("MOTION_TEMPORAL_CANDIDATE_PHYSICS_AUTHORITY_FORBIDDEN")

    status_counts = Counter()
    reference_status_counts = Counter()
    subjects = []
    evaluated = confirmed = skipped_short = 0
    policy = RelativeTransportPolicyV1()

    for subject_ref in manifest.get("subjects") or ():
        subject = str(subject_ref["subject_id"])
        if subject not in calibration:
            raise RuntimeError("MOTION_TEMPORAL_CALIBRATION_SUBJECT_MISSING:" + subject)
        scene, arrays = calibration[subject]
        rest, faces, robust, reference, scale = _strict_reference_faces(arrays)
        subject_manifest_path = candidate_root / subject_ref["manifest_relative"]
        subject_manifest = json.loads(subject_manifest_path.read_text())
        if subject_manifest.get("schema") != SUBJECT_SCHEMA:
            raise RuntimeError("MOTION_TEMPORAL_SUBJECT_MANIFEST_SCHEMA:" + subject)
        if subject_manifest.get("rest_vertices_array_sha256") != _array_sha256(rest):
            raise RuntimeError("MOTION_TEMPORAL_REST_TOPOLOGY_LINEAGE_DRIFT:" + subject)
        if subject_manifest.get("faces_array_sha256") != _array_sha256(faces):
            raise RuntimeError("MOTION_TEMPORAL_FACE_TOPOLOGY_LINEAGE_DRIFT:" + subject)

        queries = _pair_queries(rest, faces, robust, reference)
        subject_counts = Counter()
        action_rows = []
        degeneracy = max(scale * scale * 1.0e-12, 1.0e-14)
        for action_meta in subject_manifest.get("actions") or ():
            payload_path = candidate_root / action_meta["payload_relative"]
            with np.load(payload_path, allow_pickle=False) as payload:
                response = np.asarray(payload["response"], dtype=np.float64)
                timestamps = np.asarray(payload["timestamps_seconds"], dtype=np.float64)
            if response.ndim != 4 or response.shape[0] != len(rest) or response.shape[1] != 1 or response.shape[3] != 3:
                raise RuntimeError("MOTION_TEMPORAL_RESPONSE_LAYOUT_DRIFT:" + subject)
            split = _split(response.shape[2])
            if split is None:
                skipped_short += 1
                action_rows.append({"action": action_meta["teacher_action_identity"], "status": "SKIPPED_LT3_FRAMES"})
                continue
            discovery, validation = split
            positions = rest[None, :, :] + np.transpose(response[:, 0, :, :], (1, 0, 2))
            action_counter = Counter()
            for face_a, face_b, reference_kind in queries:
                evidence = measure_temporal_relative_transport_v1(
                    rest,
                    faces,
                    positions,
                    timestamps,
                    SurfacePairQueryV1(face_a, face_b),
                    condition_binding_hash=str(action_meta["metadata_sha256"]),
                    discovery_frame_indices=discovery,
                    validation_frame_indices=validation,
                    subject_scale=scale,
                    degeneracy_tolerance=degeneracy,
                    policy=policy,
                )
                evaluated += 1
                status_counts[evidence.final_status] += 1
                reference_status_counts[(reference_kind, evidence.final_status)] += 1
                subject_counts[evidence.final_status] += 1
                action_counter[evidence.final_status] += 1
                if evidence.heldout_confirmed:
                    confirmed += 1
            action_rows.append(
                {
                    "action": action_meta["teacher_action_identity"],
                    "frame_count": int(response.shape[2]),
                    "query_count": len(queries),
                    "status_counts": dict(sorted(action_counter.items())),
                }
            )
        subjects.append(
            {
                "subject_id": subject,
                "robust_exact_reference_faces": int(np.sum(robust)),
                "exact_source_influence_groups": int(len({int(v) for v in reference[robust]})),
                "query_count_per_action": len(queries),
                "status_counts": dict(sorted(subject_counts.items())),
                "actions": action_rows,
            }
        )

    if evaluated < 20:
        raise RuntimeError("MOTION_TEMPORAL_ORACLE_INSUFFICIENT_MEASURED_COVERAGE")
    if confirmed == 0:
        raise RuntimeError("MOTION_TEMPORAL_ORACLE_NO_HELDOUT_CONFIRMED_EVIDENCE")

    return {
        "schema": SCHEMA,
        "candidate_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "subject_count": len(subjects),
        "action_count": int(manifest["action_count"]),
        "evaluated_surface_condition_pairs": evaluated,
        "heldout_confirmed_count": confirmed,
        "skipped_actions_lt3_frames": skipped_short,
        "status_counts": dict(sorted(status_counts.items())),
        "reference_status_counts": {
            f"{kind}|{status}": count
            for (kind, status), count in sorted(reference_status_counts.items())
        },
        "policy": {
            "rigid_accept_angle_rad": policy.rigid_accept_angle_rad,
            "rigid_accept_translation_fraction": policy.rigid_accept_translation_fraction,
            "rigid_accept_strain_fro": policy.rigid_accept_strain_fro,
            "nonrigid_reject_angle_rad": policy.nonrigid_reject_angle_rad,
        },
        "privileged_reference_used_only_by_outer_evaluator": True,
        "expected_mechanism_labels_assigned": False,
        "finite_authored_conditions_mint_global_static_authority": False,
        "dynamic_generation_authorized": False,
        "qualification_minted": False,
        "student_target_allowed": False,
        "training_allowed": False,
        "court_status": "PASS_MEASURED_TEMPORAL_RELATIVE_TRANSPORT_COVERAGE",
        "subjects": subjects,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-root", required=True, type=Path)
    parser.add_argument("--candidate-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = audit(calibration_root=args.calibration_root, candidate_root=args.candidate_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "subjects"}, sort_keys=True))


if __name__ == "__main__":
    main()
