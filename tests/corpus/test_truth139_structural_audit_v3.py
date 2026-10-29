from tools.corpus.run_truth139_structural_sealed_audit_v3 import (
    _collapse_historical_presentation_variants,
    _historical_variant_key,
)


def _row(path: str, *, stem: str, material: str, full: bool = True) -> dict:
    parts = path.split("/")
    assert parts[:2] == ["raw", "quaternius"]
    source = {
        "relative_path": path,
        "creator": "quaternius",
        "pack_id": parts[2],
        "normalized_asset_stem": stem,
    }
    return {
        "source": source,
        "subject_id": "old-" + material,
        "subject_identity_signature": "geom-" + material,
        "duplicate_source_files": [source],
        "capabilities": {
            "OBSERVATION_FSTAR": True,
            "SURFACE_NORMAL_TRUTH": True,
            "STAGE47_MECHANICS": full,
            "STAGE48_APPEARANCE_PRESENTATION": full,
            "TESSA_TOPOLOGY_TEACHER": True,
            "FULL_TRUTH139_RAW_ELIGIBLE": full,
        },
        "development_exclusion": None,
        "duplicate_family_development_taint": [],
        "raw_eligibility_blockers": [],
        "lineage": {
            "source_sha256": ["sha-" + material],
            "source_identity": ["id-" + material],
            "skeleton_hierarchy_signature": ["skel"],
            "topology_lineage_signature": ["topo-" + material],
            "material_lineage_signature": ["mat-" + material],
        },
    }


def test_flat_and_textured_style_directories_share_historical_variant_key():
    flat = _row(
        "raw/quaternius/quaternius-animated-mech/Flat Colors/glTF/George.gltf",
        stem="george",
        material="flat",
    )
    textured = _row(
        "raw/quaternius/quaternius-animated-mech/Textured/glTF/George.gltf",
        stem="george",
        material="textured",
    )
    assert _historical_variant_key(flat) == _historical_variant_key(textured)
    subjects, groups = _collapse_historical_presentation_variants([flat, textured])
    assert len(subjects) == 1
    assert len(subjects[0]["duplicate_source_files"]) == 2
    assert len(groups) == 1
    assert groups[0]["collapse_reason"] == "HISTORICAL_PRESENTATION_VARIANT_FAMILY"


def test_semantic_family_directories_are_not_stripped_or_overcollapsed():
    big = _row(
        "raw/quaternius/quaternius-ultimate-monsters/Big/glTF/Alien.gltf",
        stem="alien",
        material="big",
    )
    blob = _row(
        "raw/quaternius/quaternius-ultimate-monsters/Blob/glTF/Alien.gltf",
        stem="alien",
        material="blob",
    )
    assert _historical_variant_key(big) != _historical_variant_key(blob)
    subjects, groups = _collapse_historical_presentation_variants([big, blob])
    assert len(subjects) == 2
    assert groups == []
