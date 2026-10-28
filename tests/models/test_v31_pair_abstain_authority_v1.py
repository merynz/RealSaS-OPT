import copy
import json
from pathlib import Path

import numpy as np
import pytest

from models.iris.v31.lofo_trainer_v3 import _pair_recipe_contract
from models.training_runtime.durable_state_v1 import file_sha256
from tools.training.prepare_iris_v31_lofo_inputs_v3 import _projection_qualification_v3


ROOT = Path(__file__).resolve().parents[2]


def _target(path: Path, *, pair_valid=False):
    arrays = {
        "mechanical_xyz": np.zeros((2, 3), np.float32),
        "mechanical_target": np.zeros((2, 2), np.float32),
        "mechanical_valid": np.ones(2, np.bool_),
        "appearance_xyz": np.zeros((2, 3), np.float32),
        "appearance_camera_index": np.zeros(2, np.int64),
        "appearance_rgba": np.zeros((2, 4), np.float32),
        "appearance_valid": np.ones(2, np.bool_),
        "pair_xyz_a": np.zeros((2, 3), np.float32),
        "pair_xyz_b": np.zeros((2, 3), np.float32),
        "pair_camera_index": np.zeros(2, np.int64),
        "pair_target": np.zeros((2, 1), np.float32),
        "pair_valid": np.full(2, pair_valid, np.bool_),
    }
    np.savez_compressed(path, **arrays)


def _manifest(tmp_path: Path, pair_subjects=0):
    rows = []
    for i in range(8):
        path = tmp_path / f"s{i}.npz"
        _target(path, pair_valid=i < pair_subjects)
        rows.append({
            "subject_id": f"fit8_{i}",
            "presentation_targets": {"path": str(path), "sha256": file_sha256(path)},
        })
    return {
        "subjects": rows,
        "teacher_inputs_in_forward": False,
        "semantic_part_taxonomy_used": False,
    }


def _bundle(source_sha: str, *, authorized=False, subjects=0, rows=0):
    return {
        "bundle_builder_revision": "V4_OBJECTIVE_EXACT_PAIR_OBSERVABILITY_ADMISSION",
        "subject_binding_revision": "V3_PLATFORM_UUID_TO_LOGICAL_KEY_FROM_EXACT_STAGE48_PAYLOAD",
        "source_code_sha": source_sha,
        "semantic_part_taxonomy_used": False,
        "knight_is_training_subject": False,
        "pair_student_supervision_authorized": authorized,
        "pair_evidence_subject_count": subjects,
        "pair_evidence_valid_rows": rows,
        "pair_observability": (
            "MECHANISM_CONDITIONAL__CROSS_FOLD_PAIR_EVIDENCE_AUTHORIZED"
            if authorized
            else "UNAVAILABLE__ZERO_JOINTLY_CONTRIBUTING_SAME_RAY_ROWS__ABSTAIN"
        ),
    }


def test_zero_pair_evidence_is_explicit_abstain_not_training_blocker(tmp_path):
    source_sha = "a" * 40
    qualification = _projection_qualification_v3(
        _manifest(tmp_path, pair_subjects=0),
        _bundle(source_sha, authorized=False, subjects=0, rows=0),
        source_sha,
    )
    assert qualification["status"] == "PASS_EXACT_STUDENT_PROJECTION__TRAINING_AUTHORIZED"
    assert qualification["pair_training_authorized"] is False
    assert qualification["pair_evidence_valid_rows"] == 0
    assert qualification["families"]["occlusion_attenuation"]["objective"] == "ABSTAIN"
    assert "NORMALIZED_PAIR_QUERY_XYZ" not in qualification["forward_input_contract"]
    assert qualification["missing_pair_truth_becomes_negative"] is False


def test_partial_cross_fold_pair_evidence_still_fails_closed(tmp_path):
    source_sha = "b" * 40
    with pytest.raises(ValueError, match="IRIS_V31_PROJECTION_CROSS_FOLD_PAIR_EVIDENCE_INSUFFICIENT"):
        _projection_qualification_v3(
            _manifest(tmp_path, pair_subjects=1),
            _bundle(source_sha, authorized=False, subjects=1, rows=2),
            source_sha,
        )


def test_checked_in_recipe_disables_all_pair_gradients_and_metrics():
    recipe = json.loads((ROOT / "canonical/IRIS_V31_LOFO_RECIPE_V1.json").read_text())
    _pair_recipe_contract(recipe, False)
    assert recipe["sampling"]["pair_queries_per_subject_step"] == 0
    assert all(stage["loss_weights"]["occlusion_attenuation"] == 0 for stage in recipe["schedule"])
    assert all("OCCLUSION_ATTENUATION" not in stage["families"] for stage in recipe["schedule"])
    assert "OCCLUSION_ATTENUATION_BCE" not in recipe["evaluation"]["heldout_metrics"]


def test_pair_abstain_recipe_rejects_silent_reenable():
    recipe = json.loads((ROOT / "canonical/IRIS_V31_LOFO_RECIPE_V1.json").read_text())
    bad = copy.deepcopy(recipe)
    bad["schedule"][-1]["loss_weights"]["occlusion_attenuation"] = 0.1
    with pytest.raises(ValueError, match="IRIS_V31_LOFO_PAIR_ABSTAIN_RECIPE_REQUIRED"):
        _pair_recipe_contract(bad, False)
