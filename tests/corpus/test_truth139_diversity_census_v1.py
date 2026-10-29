from tools.corpus.build_truth139_diversity_census_v1 import build


def _subject(i: int, *, skeleton: str, topology: str, material: str, family: str) -> dict:
    return {
        "subject_id": f"s{i:03d}",
        "source": {"pack_id": "p0" if i < 70 else "p1"},
        "lineage": {
            "source_family_signature": [family],
            "skeleton_hierarchy_signature": [skeleton],
            "topology_lineage_signature": [topology],
            "material_lineage_signature": [material],
        },
        "raw_structure": {"animation_count": 1, "joint_count": 5},
    }


def _audit(rows: list[dict]) -> dict:
    return {
        "schema": "RealSaS.Truth139SealedRawAudit.v2",
        "status": "PASS_SEALED_RAW_CAPABILITY_AUDIT_V2__PLATFORM_ADMISSION_PENDING",
        "structural_core_passed": True,
        "audit_identity_sha256": "a" * 64,
        "subjects": rows,
    }


def test_diversity_census_counts_transitive_leakage_components():
    rows = []
    for i in range(139):
        rows.append(_subject(i, skeleton=f"sk{i}", topology=f"tp{i}", material=f"mt{i}", family=f"fm{i}"))
    rows[1]["lineage"]["skeleton_hierarchy_signature"] = ["sk0"]
    rows[2]["lineage"]["topology_lineage_signature"] = ["tp1"]
    rows[4]["lineage"]["material_lineage_signature"] = ["mt3"]
    result = build(_audit(rows))
    summary = result["leakage_component_summary"]
    assert summary["component_count"] == 136
    assert summary["largest_component_size"] == 3
    assert summary["component_size_histogram"]["3"] == 1
    assert summary["component_size_histogram"]["2"] == 1
    assert result["historical_subject_count_exact"] is True
    assert result["signature_multiplicity"]["skeleton_hierarchy_signature"]["maximum_subjects_per_token"] == 2


def test_non139_count_is_reported_but_does_not_block_structural_census():
    rows = [_subject(i, skeleton=f"sk{i}", topology=f"tp{i}", material=f"mt{i}", family=f"fm{i}") for i in range(138)]
    result = build(_audit(rows))
    assert result["subject_count"] == 138
    assert result["historical_subject_count_exact"] is False
    assert result["historical_count_mismatch_operational_blocker"] is False
    assert result["status"] == "PASS_TRUTH139_STRUCTURAL_DIVERSITY_CENSUS"
