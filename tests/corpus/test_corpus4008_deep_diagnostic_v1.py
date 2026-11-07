from __future__ import annotations
import hashlib
from pathlib import Path

import pytest

from tools.corpus.validate_corpus4008_deep_diagnostic_v1 import verify_pair


def receipts(tmp_path):
    sha = "a" * 64
    ctrl = {
        "schema": "RealSaS.Corpus4008ImportedControlProfile.v1",
        "status": "PINNED_IMPORTED_CONTROL_OBSERVATION_ONLY",
        "asset_id": "asset_test", "candidate_id": "test:" + sha,
        "original_sha256": sha,
        "training_authorized": False,
        "model_family_authorized": False,
        "semantic_truth_authorized": False,
        "imported_control": {
            "training_authorized": False,
            "actual_ik_functionality": "UNKNOWN",
            "ik_absence_in_authored_source": "UNKNOWN",
            "controller_bone_semantics": "UNKNOWN",
            "control_group_semantics": "UNKNOWN",
            "fk_ik_switch_functionality": "UNKNOWN",
            "retargetability": "UNKNOWN",
            "source_anatomical_up": "UNKNOWN",
            "source_anatomical_front": "UNKNOWN",
            "imported_pose_bone_count": 0,
            "imported_armature_count": 0,
            "imported_constraint_count": 0,
            "imported_ik_constraint_count": 0,
            "imported_nondeform_bone_count": 0,
            "imported_driver_count": 0,
            "imported_nla_strip_count": 0,
            "imported_action_count": 0,
            "imported_action_curve_count": 0,
            "armatures": [], "object_constraints": [],
            "drivers": [], "nla_strips": [], "actions": [],
        },
    }
    data = {
        "schema": "RealSaS.Corpus4008LookdevStudy.v1",
        "status": "LOOKDEV_DIAGNOSTIC_ONLY",
        "asset_id": "asset_test", "candidate_id": "test:" + sha,
        "source_sha256": sha,
        "source_anatomical_up": "UNKNOWN",
        "source_anatomical_front": "UNKNOWN",
        "training_authorized": False,
        "stylization_training_authorized": False,
        "source_rgba_truth_authorized": False,
        "source_material_parity_courted": False,
        "texture_fidelity_courted": False,
        "outputs": [],
    }
    for mode in ("SOURCE_MATERIAL_DIAGNOSTIC", "STYLIZED_FLAT_DIAGNOSTIC",
                 "STYLIZED_CEL_3BAND_DIAGNOSTIC"):
        filename = mode.lower() + ".png"
        payload = ("test-" + mode).encode()
        (tmp_path / filename).write_bytes(payload)
        data["outputs"].append({
            "mode": mode, "path_private": filename,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "byte_count": len(payload),
            "render_approved_as_source_rgb": False,
            "render_approved_as_model_target": False,
        })
    return ctrl, data


def test_exact_same_source_can_verify_diagnostic_only(tmp_path):
    c, l = receipts(tmp_path)
    out = verify_pair(c, l, tmp_path)
    assert out["counts"]["render_studies"] == 3
    assert out["training_authorized"] is False


def test_ik_absence_cannot_be_minted_as_known(tmp_path):
    c, l = receipts(tmp_path)
    c["imported_control"]["ik_absence_in_authored_source"] = "ABSENT"
    with pytest.raises(ValueError, match="CONTROL_UNKNOWN"):
        verify_pair(c, l, tmp_path)


def test_cross_source_or_asset_mixup_is_rejected(tmp_path):
    c, l = receipts(tmp_path)
    l["source_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="CROSS_SOURCE_SHA"):
        verify_pair(c, l, tmp_path)


def test_stylized_render_cannot_claim_native_rgb(tmp_path):
    c, l = receipts(tmp_path)
    l["outputs"][1]["render_approved_as_source_rgb"] = True
    with pytest.raises(ValueError, match="LOOKDEV_RENDER_AUTHORITY"):
        verify_pair(c, l, tmp_path)


def test_pixel_or_receipt_tamper_is_rejected(tmp_path):
    c, l = receipts(tmp_path)
    (tmp_path / "stylized_cel_3band_diagnostic.png").write_bytes(b"changed")
    with pytest.raises(ValueError, match="LOOKDEV_BYTE_COUNT|LOOKDEV_SHA"):
        verify_pair(c, l, tmp_path)


def test_unsafe_output_path_is_rejected(tmp_path):
    c, l = receipts(tmp_path)
    l["outputs"][2]["path_private"] = "../outside.png"
    with pytest.raises(ValueError, match="LOOKDEV_PATH"):
        verify_pair(c, l, tmp_path)


def test_unconfirmed_controller_claim_is_rejected(tmp_path):
    c, l = receipts(tmp_path)
    c["imported_control"]["armatures"] = [{
        "bones": [{"use_deform": False, "constraints": [],
                   "is_control_bone_courted": True,
                   "ik_solver_functional_courted": False}]}]
    c["imported_control"]["imported_pose_bone_count"] = 1
    c["imported_control"]["imported_armature_count"] = 1
    c["imported_control"]["imported_nondeform_bone_count"] = 1
    with pytest.raises(ValueError, match="CONTROLLER_COURT_FORGERY"):
        verify_pair(c, l, tmp_path)
