from __future__ import annotations

import hashlib
import json

import pytest

from tools.truth_foundry.build_source_ray_appearance_platform_bundle_v1 import build


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _measurement(subject_id):
    return {
        "schema": "RealSaS.SourceRayAppearanceMeasurementCandidate.v1",
        "subject_id": subject_id,
        "rays": [{
            "canonical_hits": [
                {"face_index": 1, "barycentric": [0.2, 0.3, 0.5], "depth": 1.0},
                {"face_index": 2, "barycentric": [0.3, 0.3, 0.4], "depth": 2.0},
            ]
        }],
    }


def _court(measurements):
    subjects = []
    for payload in measurements:
        sha = _digest(payload)
        subjects.append({
            "subject_id": payload["subject_id"],
            "parents": {"packed_a_sha256": sha, "packed_b_sha256": sha},
        })
    return {
        "schema": "RealSaS.SourceRayAppearanceSourceEquivalenceCourt.v1",
        "court_status": "PASS_SOURCE_RAY_APPEARANCE_LOCAL_QUALIFICATION_CANDIDATE",
        "subject_count": len(subjects),
        "ray_count": len(measurements),
        "hit_count": 2 * len(measurements),
        "subjects": subjects,
    }


def test_bundle_binds_exact_packed_parent_receipts():
    measurements = [_measurement("fit8_b_v1"), _measurement("fit8_a_v1")]
    value = build(_court(measurements), measurements)
    assert value["registry_qualification_minted"] is False
    assert value["canonical_mesh_binding_minted"] is False
    assert [row["subject_id"] for row in value["measurements"]] == ["fit8_a_v1", "fit8_b_v1"]
    assert all(row["sha256"] == _digest(row["payload"]) for row in value["measurements"])


def test_bundle_rejects_measurement_parent_hash_drift():
    measurement = _measurement("fit8_a_v1")
    court = _court([measurement])
    measurement["rays"][0]["canonical_hits"][0]["depth"] = 9.0
    with pytest.raises(RuntimeError, match="MEASUREMENT_PARENT_HASH_DRIFT"):
        build(court, [measurement])


def test_bundle_rejects_incomplete_cohort():
    a = _measurement("fit8_a_v1")
    b = _measurement("fit8_b_v1")
    with pytest.raises(RuntimeError, match="MEASUREMENT_COHORT_INCOMPLETE"):
        build(_court([a, b]), [a])
