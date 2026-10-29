from __future__ import annotations

import copy

import pytest

from tools.training.build_lineage_safe_subject_split_v1 import build_split


def _row(subject: str, skeleton: str, topology: str, material: str, *, sha: str | None = None):
    return {
        "subject_id": subject,
        "admitted_for_truth139": True,
        "lineage": {
            "source_sha256": sha or (subject + "_sha"),
            "skeleton_hierarchy_signature": skeleton,
            "topology_lineage_signature": topology,
            "material_lineage_signature": material,
        },
    }


def _ledger():
    return {
        "schema": "RealSaS.TruthCorpusLineageLedger.v1",
        "subjects": [
            _row("a", "sk_shared", "top_a", "mat_a"),
            _row("b", "sk_shared", "top_b", "mat_b"),
            _row("c", "sk_c", "top_shared", "mat_c"),
            _row("d", "sk_d", "top_shared", "mat_d"),
            _row("e", "sk_e", "top_e", "mat_e"),
            _row("f", "sk_f", "top_f", "mat_f"),
        ],
    }


def test_shared_lineage_never_crosses_train_unseen():
    result = build_split(
        _ledger(), target_train_fraction=2.0 / 3.0, seed=7, expected_subject_count=6
    )
    side = {
        subject: component["split"]
        for component in result["components"]
        for subject in component["subject_ids"]
    }
    assert side["a"] == side["b"]
    assert side["c"] == side["d"]
    assert set(result["train_subjects"]).isdisjoint(result["unseen_subjects"])
    assert result["pack_level_split_used"] is False
    assert result["optimizer_may_read_unseen"] is False


def test_split_is_byte_identity_deterministic_for_same_ledger_and_seed():
    first = build_split(_ledger(), seed=123, expected_subject_count=6)
    second = build_split(copy.deepcopy(_ledger()), seed=123, expected_subject_count=6)
    assert first == second
    assert len(first["split_identity_sha256"]) == 64


def test_missing_required_lineage_fails_closed():
    ledger = _ledger()
    del ledger["subjects"][0]["lineage"]["material_lineage_signature"]
    with pytest.raises(ValueError, match="TRUTH139_REQUIRED_LINEAGE_FIELD_MISSING"):
        build_split(ledger, expected_subject_count=6)


def test_expected_subject_count_is_authoritative():
    with pytest.raises(ValueError, match="TRUTH139_EXPECTED_SUBJECT_COUNT_MISMATCH"):
        build_split(_ledger(), expected_subject_count=139)


def test_exact_source_duplicate_is_one_component():
    ledger = {
        "schema": "RealSaS.TruthCorpusLineageLedger.v1",
        "subjects": [
            _row("a", "s1", "t1", "m1", sha="same"),
            _row("b", "s2", "t2", "m2", sha="same"),
            _row("c", "s3", "t3", "m3"),
        ],
    }
    result = build_split(ledger, target_train_fraction=0.67, seed=1, expected_subject_count=3)
    component = next(row for row in result["components"] if "a" in row["subject_ids"])
    assert component["subject_ids"] == ["a", "b"]
