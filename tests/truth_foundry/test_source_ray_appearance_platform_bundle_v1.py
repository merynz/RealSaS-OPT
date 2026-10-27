from __future__ import annotations

import hashlib
import json

import pytest

from tools.truth_foundry import build_source_ray_appearance_platform_bundle_v1 as bundle_builder


def _raw(value):
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _measurement(subject_id):
    return {
        "schema": bundle_builder.MEASUREMENT_SCHEMA,
        "subject_id": subject_id,
        "rays": [{
            "canonical_hits": [
                {"face_index": 1, "barycentric": [0.2, 0.3, 0.5], "depth": 1.0},
                {"face_index": 2, "barycentric": [0.3, 0.3, 0.4], "depth": 2.0},
            ]
        }],
    }


def _court(measurement_raws):
    subjects = []
    for raw in measurement_raws:
        payload = json.loads(raw)
        sha = _sha(raw)
        subjects.append({
            "subject_id": payload["subject_id"],
            "parents": {"packed_a_sha256": sha, "packed_b_sha256": sha},
        })
    return {
        "schema": bundle_builder.COURT_SCHEMA,
        "court_status": bundle_builder.PASS_COURT,
        "subject_count": len(subjects),
        "ray_count": len(measurement_raws),
        "hit_count": 2 * len(measurement_raws),
        "subjects": subjects,
    }


def _fixture(monkeypatch):
    measurements = [_raw(_measurement("fit8_b_v1")), _raw(_measurement("fit8_a_v1"))]
    court_raw = _raw(_court(measurements))
    monkeypatch.setattr(bundle_builder, "SOURCE_COURT_RAW_SHA256", _sha(court_raw))
    return court_raw, measurements


def test_bundle_preserves_exact_packed_parent_bytes(monkeypatch):
    court_raw, measurements = _fixture(monkeypatch)
    value = bundle_builder.build(court_raw, measurements)
    assert value["registry_qualification_minted"] is False
    assert value["canonical_mesh_binding_minted"] is False
    assert value["court"]["json_text"].encode() == court_raw
    assert [row["subject_id"] for row in value["measurements"]] == ["fit8_a_v1", "fit8_b_v1"]
    assert all(_sha(row["json_text"].encode()) == row["sha256"] for row in value["measurements"])


def test_bundle_rejects_measurement_parent_hash_drift(monkeypatch):
    measurement_raw = _raw(_measurement("fit8_a_v1"))
    court_raw = _raw(_court([measurement_raw]))
    monkeypatch.setattr(bundle_builder, "SOURCE_COURT_RAW_SHA256", _sha(court_raw))
    changed = json.loads(measurement_raw)
    changed["rays"][0]["canonical_hits"][0]["depth"] = 9.0
    with pytest.raises(RuntimeError, match="MEASUREMENT_PARENT_HASH_DRIFT"):
        bundle_builder.build(court_raw, [_raw(changed)])


def test_bundle_rejects_court_raw_byte_drift(monkeypatch):
    court_raw, measurements = _fixture(monkeypatch)
    with pytest.raises(RuntimeError, match="COURT_RAW_HASH_DRIFT"):
        bundle_builder.build(court_raw + b" ", measurements)


def test_bundle_rejects_incomplete_cohort(monkeypatch):
    court_raw, measurements = _fixture(monkeypatch)
    with pytest.raises(RuntimeError, match="MEASUREMENT_COHORT_INCOMPLETE"):
        bundle_builder.build(court_raw, measurements[:1])
