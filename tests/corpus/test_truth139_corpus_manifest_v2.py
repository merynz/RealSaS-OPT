from __future__ import annotations

import pytest

from tools.corpus.build_truth139_corpus_manifest_v2 import build


def _subject(index: int, *, dev: bool = False, stage48: bool = True) -> dict:
    sid = f"truth139_{index:03d}"
    caps = {
        "OBSERVATION_FSTAR": True,
        "SURFACE_NORMAL_TRUTH": True,
        "STAGE47_MECHANICS": True,
        "STAGE48_APPEARANCE_PRESENTATION": stage48,
        "TESSA_TOPOLOGY_TEACHER": True,
    }
    return {
        "subject_id": sid,
        "subject_identity_signature": f"identity-{index:03d}",
        "source": {
            "relative_path": f"raw/quaternius/pack/glTF/{sid}.gltf",
            "sha256": f"{index % 10}" * 64,
            "size_bytes": 1234,
            "format": "GLTF",
            "creator": "quaternius",
            "pack_id": "quaternius-pack",
            "dependencies": [],
            "dependency_count": 0,
            "source_page_snapshot": {"path": "provenance/source_pages/pack.html", "sha256": "a" * 64},
            "official_url": "https://example.invalid",
            "drive_folder_id": "folder",
            "source_family_signature": [f"SRCF:{index:03d}"],
        },
        "lineage": {"source_family_signature": [f"SRCF:{index:03d}"]},
        "raw_structure": {"skin_count": 1},
        "duplicate_source_file_count": 1,
        "duplicate_source_files": [],
        "capabilities": {**caps, "FULL_TRUTH139_RAW_ELIGIBLE": all(caps.values())},
        "raw_eligibility_blockers": [] if stage48 else ["STAGE48_APPEARANCE_PRESENTATION"],
        "development_exclusion": "FIT8" if dev else None,
        "duplicate_family_development_taint": ["FIT8"] if dev else [],
    }


def _parents(subjects: list[dict]):
    audit_id = "b" * 64
    audit = {
        "schema": "RealSaS.Truth139SealedRawAudit.v2",
        "status": "PASS_SEALED_RAW_CAPABILITY_AUDIT_V2__PLATFORM_ADMISSION_PENDING",
        "audit_identity_sha256": audit_id,
        "corpus": {
            "seal_verified": True,
            "root_sha256": "1" * 64,
            "manifest_sha256": "2" * 64,
            "ledger_sha256": "3" * 64,
            "seal_json_sha256": "4" * 64,
        },
        "subjects": subjects,
    }
    court = {
        "schema": "RealSaS.Truth139HistoricalReproductionCourt.v1",
        "status": "PASS_HISTORICAL_139_REPRODUCTION",
        "source_audit_identity_sha256": audit_id,
        "court_identity_sha256": "c" * 64,
    }
    return audit, court


def test_manifest_retains_all_139_and_tracks_capabilities_and_development_taint():
    subjects = [_subject(i) for i in range(139)]
    subjects[7] = _subject(7, dev=True)
    subjects[11] = _subject(11, stage48=False)
    audit, court = _parents(subjects)
    result = build(audit, court)
    assert result["schema"] == "RealSaS.Truth139CorpusManifest.v2"
    assert result["corpus"]["subject_count"] == 139
    assert result["corpus"]["development_or_regression_tainted_subject_count"] == 1
    assert result["corpus"]["capability_counts"]["STAGE48_APPEARANCE_PRESENTATION"] == 138
    assert result["corpus"]["clean_full_family_raw_candidate_count"] == 137
    assert len(result["subjects"]) == 139
    dev = next(row for row in result["subjects"] if row["subject_id"] == "truth139_007")
    partial = next(row for row in result["subjects"] if row["subject_id"] == "truth139_011")
    assert dev["development_or_regression_tainted"] is True
    assert dev["full_family_clean_experiment_candidate"] is False
    assert partial["materialization_plan"]["STAGE48_APPEARANCE_PRESENTATION_TRUTH"] is False
    assert partial["materialization_plan"]["CONTROLLED_8VIEW_OBSERVATION"] is True
    assert result["training_authorized"] is False


def test_manifest_rejects_parent_identity_mismatch():
    audit, court = _parents([_subject(i) for i in range(139)])
    court["source_audit_identity_sha256"] = "d" * 64
    with pytest.raises(ValueError, match="PARENT_IDENTITY_MISMATCH"):
        build(audit, court)


def test_manifest_rejects_non139_cardinality():
    audit, court = _parents([_subject(i) for i in range(138)])
    with pytest.raises(ValueError, match="SUBJECT_COUNT"):
        build(audit, court)
