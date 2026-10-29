from __future__ import annotations

import hashlib
import json

from tools.training.build_truth139_tessa_recipe_v1 import build_recipe


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_recipe_derives_epochs_from_largest_teacher_window_set(tmp_path):
    teacher_rows = []
    conditioning_rows = []
    token_counts = {"a": 1000, "b": 5000, "c": 9000}
    for subject, count in token_counts.items():
        receipt = tmp_path / f"{subject}.json"
        receipt.write_text(json.dumps({
            "schema": "RealSaS.Truth139TESSATeacherMesh.v1",
            "subject_id": subject,
            "teacher_topology_target_only": True,
            "student_conditioning_allowed": False,
            "representability_court": "PASS_FULL_SOURCE_TOPOLOGY_SEQUENCE",
            "token_count": count,
        }))
        teacher_rows.append({
            "subject_id": subject,
            "teacher_receipt": {"path": str(receipt), "sha256": _sha(receipt)},
        })
        conditioning_rows.append({"subject_id": subject})
    teacher = {
        "schema": "RealSaS.Truth139TESSATeacherTrainManifest.v1",
        "split_identity_sha256": "1" * 64,
        "manifest_identity_sha256": "2" * 64,
        "contains_unseen_subject_ids": False,
        "contains_unseen_file_refs": False,
        "subjects": teacher_rows,
    }
    conditioning = {
        "schema": "RealSaS.Truth139TESSAConditioningIndex.v1",
        "scope": "TRAIN",
        "split_identity_sha256": "1" * 64,
        "index_identity_sha256": "3" * 64,
        "iris_checkpoint_sha256": "4" * 64,
        "iris_foundation_identity": "5" * 64,
        "teacher_topology_used": False,
        "subjects": conditioning_rows,
    }
    recipe = build_recipe(
        teacher_manifest=teacher,
        conditioning_index=conditioning,
        target_tokens_per_window=2048,
        coverage_cycles=4,
        minimum_epochs=12,
    )
    # ceil((9000-1)/2048) = 5 windows, then four full largest-subject cycles.
    assert recipe["teacher_windows"]["maximum_windows_per_subject"] == 5
    assert recipe["teacher_windows"]["epochs"] == 20
    assert recipe["teacher_windows"]["subject_exposure_is_equal"] is True
    assert recipe["architecture_id"] == "RealSaS.TESSA.DualStreamStructuralizer.v3_1"
    assert recipe["attention_semantics"] == "ROLLING_CAUSAL_SELF_INCLUSIVE_V31"
    assert recipe["presentation_stream_present"] is False
    assert recipe["firewall"]["conditioning_cache_contains_teacher_topology"] is False
