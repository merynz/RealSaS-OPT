from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools.corpus.seal_corpus370_lineage_split_v1 import build


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def fixtures(tmp_path: Path):
    groups = []
    subject_to_lineage = {}
    for index in range(370):
        sid = f"subject_{index:03d}"
        lineage_id = f"lineage_{index:03d}"
        source_id = f"pack_{index % 29:02d}"
        groups.append({
            "lineage_id": lineage_id,
            "subject_ids": [sid],
            "subject_count": 1,
            "source_sha256": [sha(f"source-{index}")],
            "source_ids": [source_id],
            "merge_evidence": [],
        })
        subject_to_lineage[sid] = lineage_id
    lineage = tmp_path / "lineage.json"
    lineage.write_text(json.dumps({
        "schema": "RealSaS.Corpus370LineageGroups.v1",
        "status": "PASS_LINEAGE_GROUPS_SEALED",
        "subject_count": 370,
        "lineage_count": 370,
        "review_required_count": 0,
        "groups": groups,
        "subject_to_lineage": subject_to_lineage,
        "split_authorized": True,
        "training_authorized": False,
    }))

    models = {}
    for model in ("IRIS", "TESSA", "MIRA", "AXIS"):
        eligible = []
        for index in range(370):
            eligible.append({
                "subject_id": f"subject_{index:03d}",
                "source_sha256": sha(f"source-{index}"),
                "origin": "ESTABLISHED_BASE" if index < 139 else "EXPANSION",
                "source_id": f"pack_{index % 29:02d}",
                "required_pass": [],
                "family_status": {},
            })
        models[model] = {"eligible_count": 370, "eligible_subjects": eligible, "blocked_subjects": []}
    rosters = tmp_path / "rosters.json"
    rosters.write_text(json.dumps({
        "schema": "RealSaS.Corpus370ModelRosters.v1",
        "subject_count": 370,
        "models": models,
        "training_authorized": False,
    }))
    return lineage, rosters


def test_global_unseen_is_sealed_first_and_never_leaks_into_any_model(tmp_path: Path):
    lineage, rosters = fixtures(tmp_path)
    result = build(lineage, rosters, Path("canonical/CORPUS370_LINEAGE_SPLIT_POLICY_V1.json"))
    assert result["status"] == "PASS_GLOBAL_PRODUCT_UNSEEN_AND_MODEL_TRAIN_VAL_SPLIT_SEALED"
    assert result["global_unseen"]["lineage_count"] == 46
    assert result["global_unseen"]["subject_count"] == 46
    global_subjects = set(result["global_unseen"]["subject_ids"])
    for model, row in result["models"].items():
        train = set(row["train_subject_ids"])
        val = set(row["val_subject_ids"])
        unseen = set(row["global_unseen_subject_ids"])
        assert not (train & val)
        assert not ((train | val) & global_subjects)
        assert unseen <= global_subjects
        assert len(train) + len(val) + len(unseen) == 370
        assert row["val_subject_count"] > 0
    assert result["training_authorized"] is False


def test_split_is_deterministic(tmp_path: Path):
    lineage, rosters = fixtures(tmp_path)
    policy = Path("canonical/CORPUS370_LINEAGE_SPLIT_POLICY_V1.json")
    first = build(lineage, rosters, policy)
    second = build(lineage, rosters, policy)
    assert first["global_unseen"] == second["global_unseen"]
    assert first["models"] == second["models"]


def test_unresolved_lineage_review_blocks_split(tmp_path: Path):
    lineage, rosters = fixtures(tmp_path)
    doc = json.loads(lineage.read_text())
    doc["status"] = "BLOCKED_LINEAGE_REVIEW_REQUIRED"
    doc["review_required_count"] = 1
    doc["split_authorized"] = False
    lineage.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="CORPUS370_SPLIT_LINEAGE_NOT_SEALED"):
        build(lineage, rosters, Path("canonical/CORPUS370_LINEAGE_SPLIT_POLICY_V1.json"))


def test_small_model_roster_is_not_starved_by_global_holdout(tmp_path: Path):
    lineage, rosters = fixtures(tmp_path)
    doc = json.loads(rosters.read_text())
    tiny = doc["models"]["TESSA"]["eligible_subjects"][:4]
    doc["models"]["TESSA"]["eligible_subjects"] = tiny
    doc["models"]["TESSA"]["eligible_count"] = 4
    rosters.write_text(json.dumps(doc))
    result = build(lineage, rosters, Path("canonical/CORPUS370_LINEAGE_SPLIT_POLICY_V1.json"))
    tessa = result["models"]["TESSA"]
    assert tessa["train_subject_count"] + tessa["val_subject_count"] >= 3
    assert tessa["global_unseen_subject_count"] <= 1


@pytest.mark.parametrize("mutation,code", [
    ("missing_member", "RECIPROCAL_COVERAGE"),
    ("cross_group", "RECIPROCAL_PARTITION"),
    ("duplicate_member", "GROUP_MEMBERS"),
    ("phantom_group", "GROUP_REF"),
])
def test_sealed_lineage_reciprocity_cannot_be_forged(tmp_path: Path, mutation: str, code: str):
    lineage, rosters = fixtures(tmp_path)
    doc = json.loads(lineage.read_text())
    if mutation == "missing_member":
        doc["groups"][0]["subject_ids"] = []
        doc["groups"][0]["subject_count"] = 0
    elif mutation == "cross_group":
        doc["groups"][0]["subject_ids"] = ["subject_001"]
    elif mutation == "duplicate_member":
        doc["groups"][0]["subject_ids"] = ["subject_000", "subject_000"]
        doc["groups"][0]["subject_count"] = 2
    else:
        doc["groups"].append({
            "lineage_id": "phantom", "subject_ids": ["subject_000"],
            "subject_count": 1, "source_ids": ["pack_00"],
        })
    lineage.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="CORPUS370_SPLIT_LINEAGE_" + code):
        build(lineage, rosters, Path("canonical/CORPUS370_LINEAGE_SPLIT_POLICY_V1.json"))


def test_duplicate_eligible_subject_cannot_be_silently_collapsed(tmp_path: Path):
    lineage, rosters = fixtures(tmp_path)
    doc = json.loads(rosters.read_text())
    doc["models"]["IRIS"]["eligible_subjects"].append(
        dict(doc["models"]["IRIS"]["eligible_subjects"][0]))
    rosters.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="CORPUS370_SPLIT_MODEL_DUPLICATE_SUBJECT:IRIS"):
        build(lineage, rosters, Path("canonical/CORPUS370_LINEAGE_SPLIT_POLICY_V1.json"))
