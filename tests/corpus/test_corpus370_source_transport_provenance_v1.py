from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load(path: str) -> dict:
    return json.loads((ROOT / path).read_text())


def test_transport_census_and_provenance_pack_ids_are_exactly_aligned():
    transport = load("canonical/CORPUS370_SOURCE_BUNDLE_TRANSPORT_V1.json")
    provenance = load("canonical/CORPUS370_SOURCE_PACK_PROVENANCE_V1.json")

    assert transport["schema"] == "RealSaS.SourceBundleTransportParts.v1"
    assert transport["subject_count"] == 370
    assert transport["origin_counts"] == {"ESTABLISHED_BASE": 139, "EXPANSION": 231}
    assert transport["format_counts"] == {".fbx": 111, ".glb": 63, ".gltf": 196}
    assert sum(transport["source_id_counts"].values()) == 370
    assert len(transport["source_id_counts"]) == 29
    assert set(provenance["packs"]) == set(transport["source_id_counts"])
    assert provenance["training_authorized"] is False
    assert transport["training_authorized"] is False


def test_split_transport_is_byte_complete_and_hash_bound():
    transport = load("canonical/CORPUS370_SOURCE_BUNDLE_TRANSPORT_V1.json")
    parts = transport["parts"]
    assert len(parts) == 5
    assert len({row["filename"] for row in parts}) == 5
    assert len({row["drive_file_id"] for row in parts}) == 5
    assert sum(row["size_bytes"] for row in parts) == transport["archive_size_bytes"] == 452258567
    assert transport["archive_sha256"] == "5c92e77ae07b215fa4234d221faf27659bc0c47a54d2711d6e49db3d820b8fa9"
    assert transport["source_bundle_manifest_sha256"] == "99a3c98ea314ef4e4cb2b1fa62ac298128cc4d8d3c16b8c5c6da89449e57b61b"
    for row in parts:
        assert len(row["sha256"]) == 64
        assert row["size_bytes"] > 0
        assert row["drive_file_id"]


def test_historical_pack_aliases_cannot_reenter_canonical_provenance_keys():
    provenance = load("canonical/CORPUS370_SOURCE_PACK_PROVENANCE_V1.json")
    packs = set(provenance["packs"])
    forbidden_stale_aliases = {
        "quaternius-zombie-apocalypse",
        "quaternius-platformer-game-kit",
        "quaternius-rpg-character",
        "quaternius-cyberpunk",
        "quaternius-toon-shooter",
    }
    required_exact_ids = {
        "quaternius-zombie-apocalypse-kit",
        "quaternius-platformer-kit",
        "quaternius-rpg-characters",
        "quaternius-cyberpunk-game-kit",
        "quaternius-toon-shooter-kit",
    }
    assert not (packs & forbidden_stale_aliases)
    assert required_exact_ids <= packs
