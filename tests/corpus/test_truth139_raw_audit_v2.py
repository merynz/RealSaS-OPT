from __future__ import annotations

from tools.corpus.audit_truth139_gltf_capabilities_v2 import (
    _capability_rank,
    _group_development_taint,
    _total_tessa_raw_precheck,
    _union_lineage,
)
from tools.corpus.run_truth139_sealed_raw_audit_v2 import _source_family_tokens


def _row(name: str, *, vertices=100, faces=100, full=True, dev=None, material="m"):
    return {
        "source": {
            "relative_path": f"raw/quaternius/pack/glTF/{name}.gltf",
            "creator": "quaternius",
            "pack_id": "pack",
            "normalized_asset_stem": "asset",
        },
        "raw_structure": {"vertex_count": vertices, "face_count": faces},
        "capabilities": {
            "OBSERVATION_FSTAR": full,
            "SURFACE_NORMAL_TRUTH": full,
            "STAGE47_MECHANICS": full,
            "STAGE48_APPEARANCE_PRESENTATION": full,
            "TESSA_TOPOLOGY_TEACHER": full,
            "FULL_TRUTH139_RAW_ELIGIBLE": full,
        },
        "development_exclusion": dev,
        "lineage": {
            "source_sha256": ["s-" + name],
            "source_identity": ["i"],
            "skeleton_hierarchy_signature": ["k"],
            "topology_lineage_signature": ["t"],
            "material_lineage_signature": [material],
        },
    }


def test_tessa_scale_precheck_uses_total_asset_ceiling():
    ok, court = _total_tessa_raw_precheck(_row("ok", vertices=32000, faces=32000))
    assert ok is True
    assert court["total_vertex_count"] == 32000
    bad, court = _total_tessa_raw_precheck(_row("bad", vertices=33000, faces=100))
    assert bad is False
    assert court["max_vertices"] == 32768


def test_duplicate_family_lineage_is_union_and_taint_is_family_wide():
    a = _row("AssetFlat", dev="FIT8_EXACT_SOURCE_SHA256", material="flat")
    b = _row("AssetTextured", material="textured")
    lineage = _union_lineage([a, b])
    assert lineage["material_lineage_signature"] == ["flat", "textured"]
    assert set(lineage["source_sha256"]) == {"s-AssetFlat", "s-AssetTextured"}
    assert _group_development_taint([a, b]) == ["FIT8_EXACT_SOURCE_SHA256"]


def test_representative_selection_prefers_capability_before_textured_name():
    broken_textured = _row("AssetTextured", full=False)
    healthy_flat = _row("AssetFlat", full=True)
    assert sorted([broken_textured, healthy_flat], key=_capability_rank)[0] is healthy_flat


def test_source_family_token_is_creator_pack_normalized_stem_not_pack_only():
    a = _row("AssetFlat")
    a["duplicate_source_files"] = [
        {"creator": "quaternius", "pack_id": "pack", "normalized_asset_stem": "asset"},
        {"creator": "quaternius", "pack_id": "pack", "normalized_asset_stem": "asset"},
    ]
    tokens = _source_family_tokens(a)
    assert len(tokens) == 1
    assert tokens[0].startswith("SRCF:")

    other = _row("Other")
    other["source"]["normalized_asset_stem"] = "other"
    assert _source_family_tokens(other) != tokens
