from __future__ import annotations

import hashlib

from tools.training.build_truth139_train_eval_manifests_v1 import build_manifests


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_train_manifest_contains_no_unseen_ids_or_paths(tmp_path):
    subjects = []
    for subject in ("a", "b", "c"):
        refs = {}
        for role in (
            "geometry_shard",
            "presentation_targets",
            "surface_normal_truth",
            "frozen_feature_cache",
        ):
            path = tmp_path / f"{subject}_{role}.bin"
            path.write_bytes((subject + role).encode())
            refs[role] = {"path": str(path), "sha256": _sha(path)}
        subjects.append({
            "subject_id": subject,
            "source_sha256": hashlib.sha256((subject + "source").encode()).hexdigest(),
            "observation_binding": hashlib.sha256((subject + "obs").encode()).hexdigest(),
            **refs,
        })

    split_identity = "1" * 64
    index = {
        "schema": "RealSaS.Truth139PreparedSubjectIndex.v1",
        "split_identity_sha256": split_identity,
        "subjects": subjects,
    }
    split = {
        "schema": "RealSaS.LineageSafeSubjectSplit.v1",
        "split_identity_sha256": split_identity,
        "train_subjects": ["a", "b"],
        "unseen_subjects": ["c"],
    }
    train, unseen = build_manifests(index, split)

    assert [row["subject_id"] for row in train["subjects"]] == ["a", "b"]
    assert [row["subject_id"] for row in unseen["subjects"]] == ["c"]
    assert train["contains_unseen_subject_ids"] is False
    assert train["contains_unseen_file_refs"] is False
    assert train["unseen_subject_set_commitment_sha256"] == unseen["unseen_subject_set_commitment_sha256"]
    train_text = repr(train)
    assert "c_geometry_shard" not in train_text
    assert "c_presentation_targets" not in train_text
    assert unseen["optimizer_updates_allowed"] is False
    assert unseen["checkpoint_selection_allowed"] is False
