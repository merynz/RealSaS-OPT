from __future__ import annotations

import hashlib
import json

import pytest

from experiments.playback_stack_v1 import materialize_test_subject_001_sealed_fix1_v1 as seal


def _payload() -> dict:
    rows = []
    for i in range(20):
        rows.append({
            "canonical_joint_id": f"J:{i:02d}",
            "axis_xyz": [1.0, 0.0, 0.0],
            "legacy_scalar_to_semantic_sign": -1.0 if i in (13, 14, 15) else 1.0,
            "role": f"ROLE_{i:02d}",
            "positive_rotation_semantic": f"SEMANTIC_{i:02d}",
        })
    return {
        "schema": seal.EXPECTED_D0_AXIS_SCHEMA,
        "status": seal.EXPECTED_D0_AXIS_STATUS,
        "micro_pose_all_passed": True,
        "axis_semantics_are_explicit_contract_choices": True,
        "semantic_truth_claimed": False,
        "human_semantic_approval_required_before_D1_semantic_claim": True,
        "product_pass_claimed": False,
        "joint_axes": rows,
    }


def test_known_fix1_identity_is_pinned():
    assert seal.EXPECTED_D0_AXIS_CONTRACT_SHA256 == "8bfaad13ae4f0c7750756f96dd276246bd09d37c7fb51c0e6c17936721a6c43f"
    assert seal.EXPECTED_D0_AXIS_SCHEMA == "RealSaS.D0MotionAxisContract.v1"
    assert seal.EXPECTED_D0_JOINT_AXIS_COUNT == 20


def test_payload_requires_all_20_explicit_signs_and_preserves_unqualified_semantic_boundary():
    report = seal.validate_d0_fix1_payload_v1(_payload())
    assert report["joint_axis_count"] == 20
    assert report["explicit_sign_count"] == 20
    assert report["plus_sign_count"] == 17
    assert report["minus_sign_count"] == 3
    assert report["semantic_truth_claimed"] is False
    assert report["human_semantic_approval_required_before_D1_semantic_claim"] is True
    assert report["product_pass_claimed"] is False


def test_missing_sign_cannot_fall_through_to_adapter_default_plus_one():
    payload = _payload()
    payload["joint_axes"][4].pop("legacy_scalar_to_semantic_sign")
    with pytest.raises(RuntimeError, match="EXPLICIT_SIGN_REQUIRED:J:04"):
        seal.validate_d0_fix1_payload_v1(payload)


def test_semantic_truth_or_product_pass_promotion_is_rejected():
    payload = _payload()
    payload["semantic_truth_claimed"] = True
    with pytest.raises(RuntimeError, match="SEMANTIC_TRUTH_BOUNDARY_DRIFT"):
        seal.validate_d0_fix1_payload_v1(payload)

    payload = _payload()
    payload["product_pass_claimed"] = True
    with pytest.raises(RuntimeError, match="PRODUCT_PASS_BOUNDARY_DRIFT"):
        seal.validate_d0_fix1_payload_v1(payload)


def test_file_sha_is_checked_before_payload_is_accepted(tmp_path, monkeypatch):
    path = tmp_path / "D0_AXIS_CONTRACT.json"
    path.write_text(json.dumps(_payload(), sort_keys=True) + "\n", encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    with pytest.raises(RuntimeError, match="AXIS_CONTRACT_SHA_DRIFT"):
        seal.validate_sealed_d0_fix1_v1(path)

    monkeypatch.setattr(seal, "EXPECTED_D0_AXIS_CONTRACT_SHA256", digest)
    report = seal.validate_sealed_d0_fix1_v1(path)
    assert report["axis_contract_sha256"] == digest
