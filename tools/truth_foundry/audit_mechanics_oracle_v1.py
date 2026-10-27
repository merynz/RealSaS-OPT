#!/usr/bin/env python3
"""Audit Mechanics typed evidence against repaired FIT8 calibration bytes.

Privileged source weights are used only by this *outer evaluator* to construct
reference queries (same exact source influence vs different exact source
influence). They are never passed to ``measure_relative_transport_v1``. The
consumer-neutral evidence kernel therefore sees only canonical geometry and
finite intervention responses.

The calibration's action probability law is intentionally unsealed. This audit
uses a deterministic amplitude split and worst-case residuals; it does not mint
or assume a probability measure, truth qualification, training admission or a
product claim.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from io import BytesIO
import json
from pathlib import Path
import zipfile

import numpy as np

from models.iris.truth.mechanics_oracle_v1 import (
    RelativeTransportPolicyV1,
    SurfacePairQueryV1,
    measure_relative_transport_v1,
)


SCHEMA = "RealSaS.MechanicsOracleCourtCandidate.v1"
EXPECTED_ACTION_MEASURE = "UNSEALED__NO_IMPLICIT_UNIFORM_LAW"


def _load_json(archive: zipfile.ZipFile, name: str):
    return json.loads(archive.read(name))


def _subject_payloads(archive: zipfile.ZipFile):
    for name in sorted(n for n in archive.namelist() if n.endswith("/TEACHER_SCENE.json")):
        scene = _load_json(archive, name)
        base = name.rsplit("/", 1)[0]
        npz_name = base + "/geometry_and_response.npz"
        if npz_name not in archive.namelist():
            raise RuntimeError(f"MECHANICS_ORACLE_MISSING_RESPONSE:{scene.get('subject_key', name)}")
        with np.load(BytesIO(archive.read(npz_name))) as payload:
            arrays = {key: payload[key] for key in payload.files}
        yield scene, arrays


def _strict_vertex_reference(arrays):
    vertex_count = len(arrays["rest_vertices_source_world"])
    offsets = np.asarray(arrays["source_weight_offsets"], dtype=np.int64)
    indices = np.asarray(arrays["source_weight_bone_indices_privileged"], dtype=np.int64)
    values = np.asarray(arrays["source_weight_values"], dtype=np.float64)
    if offsets.shape != (vertex_count + 1,) or offsets[0] != 0 or offsets[-1] != len(indices) or len(indices) != len(values):
        raise RuntimeError("MECHANICS_ORACLE_WEIGHT_CSR_INVALID")

    exact = np.zeros(vertex_count, dtype=bool)
    dominant = np.full(vertex_count, -1, dtype=np.int64)
    for vertex in range(vertex_count):
        lo, hi = int(offsets[vertex]), int(offsets[vertex + 1])
        if hi <= lo:
            continue
        local_values = values[lo:hi]
        local_indices = indices[lo:hi]
        best = int(np.argmax(local_values))
        dominant[vertex] = int(local_indices[best])
        exact[vertex] = len(local_values) == 1 and abs(float(local_values[0]) - 1.0) <= 1.0e-6
    return exact, dominant


def _robust_exact_faces(arrays):
    rest = np.asarray(arrays["rest_vertices_source_world"], dtype=np.float64)
    faces = np.asarray(arrays["faces"], dtype=np.int64)
    exact, dominant = _strict_vertex_reference(arrays)
    p = rest[faces]
    edges = np.stack(
        (
            np.linalg.norm(p[:, 1] - p[:, 0], axis=1),
            np.linalg.norm(p[:, 2] - p[:, 1], axis=1),
            np.linalg.norm(p[:, 0] - p[:, 2], axis=1),
        ),
        axis=1,
    )
    max_edge = np.max(edges, axis=1)
    area2 = np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1)
    quality = np.divide(area2, max_edge * max_edge, out=np.zeros_like(area2), where=max_edge > 0)
    subject_scale = float(np.linalg.norm(rest.max(axis=0) - rest.min(axis=0)))

    face_reference = np.full(len(faces), -1, dtype=np.int64)
    strict = np.zeros(len(faces), dtype=bool)
    for face_index, face in enumerate(faces):
        if exact[face].all() and len(set(int(dominant[v]) for v in face)) == 1:
            strict[face_index] = True
            face_reference[face_index] = dominant[face[0]]
    robust = strict & (quality > 0.05) & (max_edge > subject_scale * 1.0e-5)
    return robust, face_reference, subject_scale


def _positive_reference_pairs(rest, faces, robust, face_reference):
    centres = rest[faces].mean(axis=1)
    pairs = []
    for reference in sorted(set(int(v) for v in face_reference[robust])):
        members = np.where(robust & (face_reference == reference))[0]
        if len(members) < 2:
            continue
        local = centres[members]
        first = int(np.argmax(np.linalg.norm(local - local.mean(axis=0), axis=1)))
        second = int(np.argmax(np.linalg.norm(local - local[first], axis=1)))
        if first != second:
            pairs.append((int(members[first]), int(members[second])))
        if len(members) >= 4:
            centre_member = int(np.argmin(np.linalg.norm(local - local.mean(axis=0), axis=1)))
            far = int(np.argmax(np.linalg.norm(local - local[centre_member], axis=1)))
            candidate = (int(members[centre_member]), int(members[far]))
            if candidate[0] != candidate[1] and candidate not in pairs:
                pairs.append(candidate)
    return pairs


def _negative_reference_pairs(robust, face_reference, *, maximum: int = 32):
    # One deterministic representative per exact source influence. Cross-product
    # construction is intentionally independent of geometry proximity; the
    # reference asks only whether distinct exact influences remain mechanically
    # distinguishable under the intervention bank.
    representatives = []
    for reference in sorted(set(int(v) for v in face_reference[robust])):
        members = np.where(robust & (face_reference == reference))[0]
        if len(members):
            representatives.append(int(members[len(members) // 2]))
    pairs = []
    for left_index, left in enumerate(representatives):
        for right in representatives[left_index + 1 :]:
            pairs.append((left, right))
            if len(pairs) >= maximum:
                return pairs
    return pairs


def _split_actions(scene):
    actions = scene.get("calibration", {}).get("actions", [])
    if not actions:
        raise RuntimeError("MECHANICS_ORACLE_ACTIONS_MISSING")
    amplitudes = np.asarray([float(action["amplitude"]) for action in actions], dtype=np.float64)
    discovery = np.where(amplitudes <= 0.01 + 1.0e-12)[0]
    validation = np.where(np.isclose(amplitudes, 0.05, atol=1.0e-12, rtol=0))[0]
    if len(discovery) < 2 or len(validation) < 1:
        raise RuntimeError("MECHANICS_ORACLE_ACTION_SPLIT_EMPTY")
    return discovery, validation


def _measure_pairs(arrays, scene, pairs, expected_status, policy):
    rest = np.asarray(arrays["rest_vertices_source_world"], dtype=np.float64)
    faces = np.asarray(arrays["faces"], dtype=np.int64)
    positions = np.asarray(arrays["isolated_position"], dtype=np.float64)
    robust, _reference, subject_scale = _robust_exact_faces(arrays)
    del robust
    discovery_indices, validation_indices = _split_actions(scene)
    if len(positions) != len(scene["calibration"]["actions"]):
        raise RuntimeError("MECHANICS_ORACLE_ACTION_RESPONSE_COUNT_MISMATCH")
    degeneracy_tolerance = max(subject_scale * subject_scale * 1.0e-12, 1.0e-14)

    records = []
    for face_a, face_b in pairs:
        evidence = measure_relative_transport_v1(
            rest,
            faces,
            positions[discovery_indices],
            positions[validation_indices],
            SurfacePairQueryV1(face_a, face_b),
            subject_scale=subject_scale,
            degeneracy_tolerance=degeneracy_tolerance,
            policy=policy,
        )
        records.append(
            {
                "expected_status_privileged_evaluator_only": expected_status,
                "observed_final_status": evidence.final_status,
                "heldout_confirmed": evidence.heldout_confirmed,
                "discovery": asdict(evidence.discovery),
                "validation": asdict(evidence.validation),
            }
        )
    return records


def audit(artifact_zip: Path):
    policy = RelativeTransportPolicyV1()
    with zipfile.ZipFile(artifact_zip) as archive:
        preparation = _load_json(archive, "PREPARATION.json")
        observation = _load_json(archive, "evidence/OBSERVATION.json")
        if observation.get("campaign_id") != preparation.get("campaign_id"):
            raise RuntimeError("MECHANICS_ORACLE_CAMPAIGN_MISMATCH")
        if observation.get("truth_qualification_inferred") is not False:
            raise RuntimeError("MECHANICS_ORACLE_INPUT_ALREADY_IMPLICITLY_QUALIFIED")
        if observation.get("training_submitted") is not False:
            raise RuntimeError("MECHANICS_ORACLE_UNEXPECTED_TRAINING")
        if not observation.get("all_measurements_committed_and_exported"):
            raise RuntimeError("MECHANICS_ORACLE_MEASUREMENTS_NOT_CLOSED")

        subjects = []
        total_rigid = total_nonrigid = rigid_failures = nonrigid_failures = 0
        rigid_discovery_angles = []
        rigid_validation_angles = []
        nonrigid_discovery_angles = []
        nonrigid_validation_angles = []

        for scene, arrays in _subject_payloads(archive):
            if scene.get("calibration", {}).get("action_measure") != EXPECTED_ACTION_MEASURE:
                raise RuntimeError("MECHANICS_ORACLE_ACTION_MEASURE_UNEXPECTED")
            rest = np.asarray(arrays["rest_vertices_source_world"], dtype=np.float64)
            faces = np.asarray(arrays["faces"], dtype=np.int64)
            robust, reference, subject_scale = _robust_exact_faces(arrays)
            positives = _positive_reference_pairs(rest, faces, robust, reference)
            negatives = _negative_reference_pairs(robust, reference)
            positive_records = _measure_pairs(
                arrays,
                scene,
                positives,
                "RIGID_RELATIVE_TRANSPORT_SUPPORTED",
                policy,
            )
            negative_records = _measure_pairs(
                arrays,
                scene,
                negatives,
                "NONRIGID_RELATIVE_TRANSPORT_SUPPORTED",
                policy,
            )

            positive_failures = sum(
                record["observed_final_status"] != record["expected_status_privileged_evaluator_only"]
                for record in positive_records
            )
            negative_failures = sum(
                record["observed_final_status"] != record["expected_status_privileged_evaluator_only"]
                for record in negative_records
            )
            total_rigid += len(positive_records)
            total_nonrigid += len(negative_records)
            rigid_failures += positive_failures
            nonrigid_failures += negative_failures
            rigid_discovery_angles.extend(record["discovery"]["max_relative_angle_rad"] for record in positive_records)
            rigid_validation_angles.extend(record["validation"]["max_relative_angle_rad"] for record in positive_records)
            nonrigid_discovery_angles.extend(record["discovery"]["max_relative_angle_rad"] for record in negative_records)
            nonrigid_validation_angles.extend(record["validation"]["max_relative_angle_rad"] for record in negative_records)

            subjects.append(
                {
                    "subject_key": scene["subject_key"],
                    "intervention_count": int(len(arrays["isolated_position"])),
                    "subject_scale": subject_scale,
                    "robust_exact_reference_faces": int(np.sum(robust)),
                    "exact_reference_groups": int(len(set(int(v) for v in reference[robust]))),
                    "rigid_reference_pairs": len(positive_records),
                    "nonrigid_reference_pairs": len(negative_records),
                    "rigid_reference_failures": int(positive_failures),
                    "nonrigid_reference_failures": int(negative_failures),
                    "evaluator_abstention": None if positive_records else "NO_STRICT_RIGID_REFERENCE_PAIR",
                }
            )

    minimum_reference_coverage_met = total_rigid >= 20 and total_nonrigid >= 20
    passed = minimum_reference_coverage_met and rigid_failures == 0 and nonrigid_failures == 0
    return {
        "schema": SCHEMA,
        "source_code_sha": preparation["source_code_sha"],
        "campaign_id": preparation["campaign_id"],
        "repair_of_campaign_id": preparation.get("repair_of_campaign_id"),
        "action_measure": EXPECTED_ACTION_MEASURE,
        "probability_law_assumed": False,
        "selection_reference_privileged": True,
        "privileged_reference_exposed_to_evidence_kernel": False,
        "qualification_minted": False,
        "training_allowed": False,
        "policy": asdict(policy),
        "court": {
            "discovery_amplitudes_rad": [0.002, 0.01],
            "heldout_validation_amplitudes_rad": [0.05],
            "rigid_reference_pairs": total_rigid,
            "nonrigid_reference_pairs": total_nonrigid,
            "rigid_reference_failures": rigid_failures,
            "nonrigid_reference_failures": nonrigid_failures,
            "minimum_reference_coverage_met": minimum_reference_coverage_met,
            "max_rigid_discovery_angle_rad": max(rigid_discovery_angles, default=None),
            "max_rigid_validation_angle_rad": max(rigid_validation_angles, default=None),
            "min_nonrigid_discovery_angle_rad": min(nonrigid_discovery_angles, default=None),
            "min_nonrigid_validation_angle_rad": min(nonrigid_validation_angles, default=None),
            "passed": passed,
        },
        "subjects": subjects,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-zip", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = audit(args.artifact_zip)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["court"], sort_keys=True), flush=True)
    if not report["court"]["passed"]:
        raise SystemExit("MECHANICS_ORACLE_COURT_FAILED")


if __name__ == "__main__":
    main()
