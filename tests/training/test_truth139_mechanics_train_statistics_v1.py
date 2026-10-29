from __future__ import annotations

import hashlib

import numpy as np

from models.iris.v31.lofo_trainer_v1 import presentation_partition_v1
from tools.training.build_truth139_mechanics_train_statistics_v1 import (
    _row_partition,
    build_statistics,
)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_partition_matches_existing_presentation_contract():
    for subject in ("a", "fit8_bat_v1", "weird-subject"):
        for count in (1, 2, 7, 31):
            train, _ = presentation_partition_v1(subject, "mechanics", count, 1234, 0.8)
            assert np.array_equal(_row_partition(subject, count, 1234, 0.8), train)


def test_statistics_are_train_only_aggregate_without_subject_or_row_identity(tmp_path):
    subjects = []
    for index, subject in enumerate(("a", "b")):
        path = tmp_path / f"{subject}.npz"
        target = np.asarray(
            [[0.1 + index, 0.2], [0.3 + index, 0.5], [0.8 + index, 1.2]],
            dtype=np.float32,
        )
        valid = np.ones(3, dtype=np.bool_)
        np.savez(path, mechanical_target=target, mechanical_valid=valid)
        subjects.append({
            "subject_id": subject,
            "presentation_targets": {"path": str(path), "sha256": _sha(path)},
        })
    manifest = {
        "schema": "RealSaS.IRISV31Truth139TrainManifest.v1",
        "split_identity_sha256": "1" * 64,
        "manifest_identity_sha256": "2" * 64,
        "contains_unseen_subject_ids": False,
        "contains_unseen_file_refs": False,
        "subjects": subjects,
    }
    value = build_statistics(
        manifest,
        partition_seed=1234,
        train_fraction=0.8,
        minimum_scale=0.001,
    )
    assert value["train_row_count"] > 0
    assert len(value["transformed_median"]) == 2
    assert len(value["fixed_scale"]) == 2
    assert all(scale >= 0.001 for scale in value["fixed_scale"])
    assert value["contains_train_subject_ids"] is False
    assert value["contains_train_row_addresses"] is False
    assert value["contains_train_target_samples"] is False
    assert "a" not in repr(value)
    assert "b" not in repr(value)
