"""No legacy label, importer observation or zero IK becomes verified teacher."""
import json

import pytest

from tools.corpus.build_corpus4008_typed_supervision_matrix_v1 import build


def fixture(tmp):
    sel = tmp / "selected.json"
    sha_a = "a" * 64
    sha_b = "b" * 64
    selection = {
        "a": {
            "canonical_asset_id": "a",
            "candidate_id": "src:" + sha_a,
            "source_registry_id": "src",
            "decision": "ADMIT",
            "split": "FIT",
        },
        "b": {
            "canonical_asset_id": "b",
            "candidate_id": "src:" + sha_b,
            "source_registry_id": "src",
            "decision": "ADMIT",
            "split": "EXTERNAL_HOLDOUT",
        },
    }
    sel.write_text(json.dumps(selection))
    src = tmp / "profiles"
    src.mkdir()
    ctrl = {
        "schema": "RealSaS.Corpus4008ImportedControlProfile.v1",
        "status": "PINNED_IMPORTED_CONTROL_OBSERVATION_ONLY",
        "asset_id": "a", "candidate_id": "src:" + sha_a,
        "original_sha256": sha_a,
        "semantic_truth_authorized": False,
        "model_family_authorized": False,
        "training_authorized": False,
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
            "imported_armature_count": 1,
            "imported_pose_bone_count": 1,
            "imported_nondeform_bone_count": 0,
            "imported_constraint_count": 0,
            "imported_ik_constraint_count": 0,
            "imported_driver_count": 0,
            "imported_nla_strip_count": 0,
            "imported_action_count": 0,
            "imported_action_curve_count": 0,
            "armatures": [{
                "bones": [{
                    "use_deform": True,
                    "constraints": [],
                    "is_control_bone_courted": False,
                    "ik_solver_functional_courted": False,
                }]
            }],
            "object_constraints": [], "drivers": [], "nla_strips": [],
            "actions": [],
        },
    }
    (src / "one.control.json").write_text(json.dumps(ctrl))
    return sel, src, ctrl


def test_3993_style_matrix_is_unknown_without_courts(tmp_path):
    sel, src, _ = fixture(tmp_path)
    report = build(sel, src, tmp_path / "out", expected_count=2)
    assert report["source_profile_counts"] == {
        "IMPORTED_PROFILE_PRESENT": 1, "NOT_IMPORTED": 1,
    }
    assert all(n == 0 for n in report["per_head_truth_pass"].values())
    assert all(n == 0 for n in report["per_head_training_admitted"].values())
    lines = (tmp_path / "out/TYPED_SUPERVISION_EVIDENCE_V1.jsonl").read_text().splitlines()
    a, b = map(json.loads, lines)
    assert a["evidence"]["ik_constraint_topology"]["observed_value"][
        "imported_ik_constraint_count"] == 0
    assert a["evidence"]["ik_constraint_topology"]["independent_truth_status"] == "UNKNOWN"
    assert a["evidence"]["ik_fk_switch_and_snap_response"]["observation_status"] == "UNKNOWN"
    assert a["evidence"]["rig_hierarchy_and_bind"]["observation_status"] == "IMPORTED_SOURCE_OBSERVATION"
    assert a["head_routes"]["AXIS"]["training_admitted"] is False
    assert a["head_routes"]["MIRA"]["training_admitted"] is False
    assert b["identity"]["holdout_excluded_from_training"] is True
    assert b["evidence"]["rig_hierarchy_and_bind"]["observed_value"] is None
    assert b["head_routes"]["AXIS"]["candidate_only"] is False


def test_one_source_cannot_supply_other_asset_evidence(tmp_path):
    sel, src, ctrl = fixture(tmp_path)
    ctrl["candidate_id"] = "src:" + "b"*64
    (src / "one.control.json").write_text(json.dumps(ctrl))
    with pytest.raises(ValueError, match="CANDIDATE_MISMATCH"):
        build(sel, src, tmp_path / "out", expected_count=2)


def test_duplicate_control_profile_rejected(tmp_path):
    sel, src, _ = fixture(tmp_path)
    (src / "duplicate.control.json").write_bytes(
        (src / "one.control.json").read_bytes())
    with pytest.raises(ValueError, match="DUPLICATE_CONTROL_RECEIPT"):
        build(sel, src, tmp_path / "out", expected_count=2)


def test_old_admit_cannot_become_full_control_truth(tmp_path):
    sel, src, _ = fixture(tmp_path)
    build(sel, src, tmp_path / "out", expected_count=2)
    a = json.loads((tmp_path / "out/TYPED_SUPERVISION_EVIDENCE_V1.jsonl")
                   .read_text().splitlines()[0])
    assert a["identity"]["historic_acceptance_not_authority"] == "ADMIT"
    for head in a["head_routes"].values():
        assert head["source_truth_court"] == "UNKNOWN"
        assert head["training_admitted"] is False
        assert head["supervision_suitable"] is False
