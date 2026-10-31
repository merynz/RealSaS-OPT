from __future__ import annotations

import json
from pathlib import Path

from tools.corpus.run_full139_truth_family_sweep_v1 import plan_only
from tools.corpus.run_full139_truth_family_sweep_v2 import aggregate


def write(path: Path, value) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def inventories(root: Path):
    subjects = {}
    source_rows = []
    required = {
        "geometry": {"path": "/x", "sha256": "1" * 64},
        "geometry_court": {"path": "/x", "sha256": "2" * 64},
        "rgb": {"path": "/x", "sha256": "3" * 64},
        "rgb_court": {"path": "/x", "sha256": "4" * 64},
        "stage48": {"path": "/x", "sha256": "5" * 64},
        "tessa_teacher": {"path": "/x", "sha256": "6" * 64},
        "tessa_court": {"path": "/x", "sha256": "7" * 64},
        "packet_lineage": "PREPARED_BASE",
    }
    for i in range(139):
        subject = f"truth139_{i:020x}"
        subjects[subject] = dict(required)
        source_rows.append({
            "subject_id": subject,
            "valid": True,
            "corpus_relative_path": f"raw/example/actor_{i}.gltf",
            "actual_sha256": f"{i:064x}"[-64:],
            "expected_sha256": f"{i:064x}"[-64:],
            "byte_size": 100 + i,
        })
    readiness = root / "readiness.json"
    cache = root / "cache.json"
    write(readiness, {"subject_count": 139, "subjects": subjects, "training_authorized": True, "split": {"historical": True}})
    write(cache, {"subject_count": 139, "source_failures": {}, "subjects": source_rows})
    return readiness, cache


def test_plan_counts_all_139_but_inherits_no_training_authority(tmp_path: Path):
    readiness, cache = inventories(tmp_path)
    plan = plan_only(readiness, cache)
    assert plan["status"] == "PASS_FULL139_SWEEP_PLAN__EXECUTION_NOT_PERFORMED"
    assert plan["subject_count"] == 139
    assert plan["stage48_inventory_count"] == 139
    assert plan["source_format_counts"] == {".gltf": 139}
    assert plan["historical_training_authority_ignored"] is True
    assert plan["historical_split_ignored"] is True
    assert plan["training_authorized"] is False


def test_aggregate_preserves_failure_without_minting_readiness():
    completed = {
        "status": "PASS_SUBJECT_TRUTH_FAMILY_SWEEP__FINAL_ADMISSION_PENDING",
        "canonical_family_candidates": {
            "SOURCE_IMPORT": "QUALIFIED_CORE_CANDIDATE",
            "SURFACE_GEOMETRY": "QUALIFIED_MASKED_CANDIDATE",
            "ANIMATION_KINEMATICS": "PENDING_CANONICAL_REPLAY_COURT",
        },
    }
    failure = {"status": "FAIL_SUBJECT_TRUTH_FAMILY_SWEEP", "error": "fixture"}
    result = aggregate(
        {"a": completed, "b": failure},
        historical_ref={"path": "/old", "sha256": "1" * 64},
        source_cache_ref={"path": "/cache", "sha256": "2" * 64},
        code_sha="3" * 40,
    )
    assert result["status"] == "INCOMPLETE_FULL139_TRUTH_FAMILY_SWEEP"
    assert result["completed_subject_count"] == 1
    assert result["failed_subject_count"] == 1
    assert result["family_candidate_counts"]["SOURCE_IMPORT"] == {"QUALIFIED_CORE_CANDIDATE": 1}
    assert result["authority"]["historical_training_authorization_inherited"] is False
    assert result["authority"]["training_authorized"] is False
