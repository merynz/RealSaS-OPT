from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tools.corpus.build_corpus370_lineage_groups_v1 import build


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def roster(tmp_path: Path, *, duplicate_bytes: bool = False, hint_collision: bool = False) -> Path:
    rows = []
    for index in range(370):
        source_sha = sha("same") if duplicate_bytes and index in {0, 1} else sha(f"source-{index}")
        source_id = "pack"
        if hint_collision and index in {0, 1}:
            path = f"assets/publishers/pack/raw/{'fbx' if index == 0 else 'gltf'}/Hero.{ 'fbx' if index == 0 else 'gltf'}"
        else:
            path = f"assets/publishers/pack/raw/model-{index}/subject-{index}.gltf"
        rows.append({
            "subject_id": f"subject_{index:03d}",
            "origin": "ESTABLISHED_BASE" if index < 139 else "EXPANSION",
            "source_id": source_id,
            "source_format": Path(path).suffix,
            "source_relative_path": path,
            "source_sha256": source_sha,
            "training_authorized": False,
        })
    path = tmp_path / "roster.json"
    path.write_text(json.dumps({
        "schema": "RealSaS.Corpus370ExactSourceRoster.v1",
        "subject_count": 370,
        "subjects": rows,
        "training_authorized": False,
    }))
    return path


def test_exact_source_bytes_merge_automatically(tmp_path: Path):
    result = build(roster(tmp_path, duplicate_bytes=True))
    assert result["status"] == "PASS_LINEAGE_GROUPS_SEALED"
    assert result["lineage_count"] == 369
    assert result["multi_subject_lineage_count"] == 1
    assert result["exact_source_duplicate_group_count"] == 1
    assert result["review_required_count"] == 0
    assert result["split_authorized"] is True
    assert result["training_authorized"] is False


def test_path_hint_collision_never_auto_merges_and_blocks_split(tmp_path: Path):
    result = build(roster(tmp_path, hint_collision=True))
    assert result["lineage_count"] == 370
    assert result["review_required_count"] == 1
    assert result["status"] == "BLOCKED_LINEAGE_REVIEW_REQUIRED"
    assert result["split_authorized"] is False
    candidate = result["review_required"][0]
    assert candidate["subject_ids"] == ["subject_000", "subject_001"]


def test_explicit_review_can_resolve_path_hint_as_distinct(tmp_path: Path):
    roster_path = roster(tmp_path, hint_collision=True)
    first = build(roster_path)
    candidate = first["review_required"][0]
    review = tmp_path / "review.json"
    review.write_text(json.dumps({
        "schema": "RealSaS.Corpus370LineageReviewDecisions.v1",
        "decisions": [{
            "candidate_id": candidate["candidate_id"],
            "subject_ids": candidate["subject_ids"],
            "decision": "DISTINCT_LINEAGES",
            "reviewer": "fixture-review",
            "rationale": "Independent authored variants despite export-lane-equivalent path hint.",
        }],
    }))
    result = build(roster_path, review_decisions_path=review)
    assert result["status"] == "PASS_LINEAGE_GROUPS_SEALED"
    assert result["lineage_count"] == 370
    assert result["review_required_count"] == 0
    assert result["split_authorized"] is True


def test_courted_high_precision_structural_fingerprint_can_merge(tmp_path: Path):
    roster_path = roster(tmp_path)
    roster_doc = json.loads(roster_path.read_text())
    fingerprint = sha("exact-structure")
    subjects = []
    for index in (0, 1):
        row = roster_doc["subjects"][index]
        subjects.append({
            "subject_id": row["subject_id"],
            "source_sha256": row["source_sha256"],
            "identity_strength": "EXACT_BIND_STRUCTURE_AND_RIG_SKIN",
            "fingerprint_sha256": fingerprint,
            "court": {"status": "PASS", "schema": "FixtureStructuralIdentityCourt.v1", "sha256": sha(f"court-{index}")},
            "evidence_refs": [{"schema": "FixtureImportedStructure.v1", "sha256": sha(f"evidence-{index}")}],
        })
    fp = tmp_path / "fingerprints.json"
    fp.write_text(json.dumps({"schema": "RealSaS.Corpus370StructuralFingerprintIndex.v1", "subjects": subjects}))
    result = build(roster_path, structural_fingerprint_path=fp)
    assert result["lineage_count"] == 369
    assert result["courted_structural_duplicate_group_count"] == 1
    assert result["review_required_count"] == 0
    assert result["split_authorized"] is True
