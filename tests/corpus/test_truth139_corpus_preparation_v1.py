from __future__ import annotations

import json

import pytest

from tools.corpus.audit_truth139_gltf_capabilities_v1 import _norm_asset
from tools.corpus.build_truth139_apparatus_manifest_v1 import build
from tools.corpus.prepare_iris_observations_v1 import load_roster
from tools.training.prepare_iris_cohort_v1 import _subject_seed


def _subject(subject_id: str, *, dev=None, full=True):
    capabilities = {
        "OBSERVATION_FSTAR": full,
        "SURFACE_NORMAL_TRUTH": full,
        "STAGE47_MECHANICS": full,
        "STAGE48_APPEARANCE_PRESENTATION": full,
        "TESSA_TOPOLOGY_TEACHER": full,
        "FULL_TRUTH139_RAW_ELIGIBLE": full,
    }
    return {
        "subject_id": subject_id,
        "source": {
            "relative_path": f"raw/quaternius/pack/glTF/{subject_id}.gltf",
            "sha256": (subject_id[0] if subject_id else "a") * 64,
            "size_bytes": 123,
            "format": "GLTF",
            "creator": "quaternius",
            "pack_id": "pack",
            "dependencies": [],
            "dependency_count": 0,
            "source_page_snapshot": {"path": "provenance/source_pages/pack.html", "sha256": "c" * 64},
            "official_url": "https://example.invalid/pack",
            "drive_folder_id": "drive-folder-id",
        },
        "lineage": {
            "source_sha256": ["s:" + subject_id],
            "source_identity": ["i:" + subject_id],
            "skeleton_hierarchy_signature": ["k:" + subject_id],
            "topology_lineage_signature": ["t:" + subject_id],
            "material_lineage_signature": ["m:" + subject_id],
        },
        "subject_identity_signature": "identity-" + subject_id,
        "duplicate_source_file_count": 1,
        "raw_structure": {"skin_count": 1},
        "capabilities": capabilities,
        "development_exclusion": dev,
        "raw_eligible_for_truth139": bool(full and dev is None),
    }


def _audit(subjects):
    return {
        "schema": "RealSaS.Truth139SealedRawAudit.v1",
        "status": "PASS_SEALED_RAW_CAPABILITY_AUDIT__PLATFORM_ADMISSION_PENDING",
        "audit_identity_sha256": "a" * 64,
        "corpus": {
            "root_path": "/sealed/raw",
            "root_sha256": "b" * 64,
            "manifest_sha256": "d" * 64,
            "ledger_sha256": "e" * 64,
            "seal_verified": True,
        },
        "subjects": subjects,
    }


def test_apparatus_filters_development_and_partial_subjects(tmp_path):
    result = build(_audit([
        _subject("alpha"),
        _subject("beta", dev="FIT8_EXACT_SOURCE_SHA256"),
        _subject("gamma", full=False),
    ]))
    assert result["schema"] == "RealSaS.Truth139ApparatusManifest.v1"
    assert result["subject_count"] == 1
    assert [row["subject_key"] for row in result["selected"]] == ["alpha"]
    assert result["selected"][0]["source"]["dependencies"] == []
    assert result["split_created"] is False
    assert result["training_authorized"] is False

    path = tmp_path / "roster.json"
    path.write_text(json.dumps(result))
    _, loaded, identity = load_roster(path)
    assert loaded["subject_count"] == 1
    assert len(identity) == 64


def test_apparatus_rejects_nonpass_raw_audit():
    value = _audit([_subject("alpha")])
    value["status"] = "FAIL_OR_INCOMPLETE_SEALED_RAW_AUDIT__DO_NOT_BUILD_APPARATUS"
    with pytest.raises(ValueError, match="RAW_AUDIT_NOT_PASS"):
        build(value)


def test_apparatus_rejects_unverified_corpus_seal():
    value = _audit([_subject("alpha")])
    value["corpus"]["seal_verified"] = False
    with pytest.raises(ValueError, match="CORPUS_SEAL_NOT_VERIFIED"):
        build(value)


def test_truth139_subject_seed_is_order_independent():
    policy = {"seed": 100, "silhouette_pool_seed": 200}
    row = {"subject_key": "truth139_alpha"}
    a = _subject_seed(policy, row, 0, "SHA256_SUBJECT_ID_LOW32_PLUS_BASE_SEED__ROSTER_ORDER_INDEPENDENT")
    b = _subject_seed(policy, row, 91, "SHA256_SUBJECT_ID_LOW32_PLUS_BASE_SEED__ROSTER_ORDER_INDEPENDENT")
    assert a == b
    legacy_a = _subject_seed(policy, row, 0, "LEGACY_ROSTER_ORDINAL_OFFSET_1000000")
    legacy_b = _subject_seed(policy, row, 1, "LEGACY_ROSTER_ORDINAL_OFFSET_1000000")
    assert legacy_a != legacy_b


def test_normalized_asset_stem_collapses_flat_textured_suffixes():
    assert _norm_asset("GeorgeTextured") == _norm_asset("GeorgeFlat") == "george"
