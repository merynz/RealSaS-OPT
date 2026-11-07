"""Per-asset coverage MUST retain UNKNOWN and holdout for missing observations."""
import json
from pathlib import Path

import pytest

from tools.corpus.build_corpus4008_deep_control_matrix_v1 import run


def asset(aid, sha, split):
    return {
        "canonical_asset_id": aid,
        "candidate_id": "objaverse_animated_originals:" + sha,
        "source_registry_id": "objaverse_animated_originals",
        "split": split,
        "decision": "ADMIT",
        "capabilities": {"iris": True, "geppetto": True, "arachne": True},
    }


def control(aid, sha):
    c = {
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
        "armatures": [], "object_constraints": [], "drivers": [],
        "nla_strips": [], "actions": [],
    }
    return {
        "schema": "RealSaS.Corpus4008ImportedControlProfile.v1",
        "status": "PINNED_IMPORTED_CONTROL_OBSERVATION_ONLY",
        "asset_id": aid, "candidate_id": "objaverse_animated_originals:" + sha,
        "original_sha256": sha,
        "training_authorized": False,
        "model_family_authorized": False,
        "semantic_truth_authorized": False,
        "imported_control": c,
    }


def base(tmp_path):
    sel = tmp_path / "selection.json"
    src = tmp_path / "receipts"
    src.mkdir()
    sha_a = "a" * 64
    sha_b = "b" * 64
    sel.write_text(json.dumps({
        "asset_A": asset("asset_A", sha_a, "FIT"),
        "asset_B": asset("asset_B", sha_b, "EXTERNAL_HOLDOUT"),
    }))
    (src / "a.control.json").write_text(json.dumps(control("asset_A", sha_a)))
    return sel, src


def test_real_receipt_vs_unknown_and_holdout(tmp_path):
    sel, src = base(tmp_path)
    out = tmp_path / "out"
    report = run(sel, src, out, expected_count=2)
    assert report["control_receipt_status_counts"] == {
        "IMPORTED_CONTROL_OBSERVATION_ONLY": 1,
        "NO_IMPORTED_CONTROL_OBSERVATION": 1,
    }
    d = json.loads((out / "CORPUS4008_DEEP_CONTROL_MATRIX_V1.json").read_text())
    a, b = d["rows"]
    assert a["imported_ik_constraint_count"] == 0
    assert a["source_ik_functional"] == "UNKNOWN"
    assert b["imported_ik_constraint_count"] is None
    assert b["holdout_training_excluded"] is True
    assert b["training_authorized"] is False
    assert d["all_new_family_admissions_closed"] is True


def test_wrong_receipt_candidate_is_rejected(tmp_path):
    sel, src = base(tmp_path)
    path = src / "a.control.json"
    doc = json.loads(path.read_text())
    doc["candidate_id"] = "objaverse_animated_originals:" + "f"*64
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="CANDIDATE_ID_DRIFT"):
        run(sel, src, tmp_path / "out", expected_count=2)


def test_unknown_external_receipt_rejected(tmp_path):
    sel, src = base(tmp_path)
    (src / "unexpected.control.json").write_text(
        json.dumps(control("asset_NOT_IN_MANIFEST", "e"*64)))
    with pytest.raises(ValueError, match="UNSELECTED_CONTROL_PROFILE"):
        run(sel, src, tmp_path / "out", expected_count=2)


def test_duplicate_receipts_fail_closed(tmp_path):
    sel, src = base(tmp_path)
    (src / "duplicate.control.json").write_bytes(
        (src / "a.control.json").read_bytes())
    with pytest.raises(ValueError, match="DUPLICATE_OR_MALFORMED"):
        run(sel, src, tmp_path / "out", expected_count=2)
