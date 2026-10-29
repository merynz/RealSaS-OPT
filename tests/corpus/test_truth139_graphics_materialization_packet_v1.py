from __future__ import annotations

import json

import pytest

from tools.corpus.run_truth139_graphics_materialization_packet_v1 import (
    _persist_packet_bound_chain,
    packet_paths,
)


def _packet(tmp_path, subject_id="truth139_test"):
    packet = tmp_path / subject_id
    packet.mkdir()
    for name in ("REST_SURFACE_GEOMETRY.npz", "SOURCE_SCENE_SNAPSHOT.json", "camera_bundle.json"):
        (packet / name).write_bytes(b"x")
    for view in range(8):
        (packet / f"V{view}.png").write_bytes(b"png")
        (packet / f"V{view}.mask.bin").write_bytes(b"mask")
    return packet


def test_packet_paths_require_exact_subject_directory_and_complete_views(tmp_path):
    packet = _packet(tmp_path)
    paths = packet_paths(packet, "truth139_test")
    assert paths["geometry_npz"].name == "REST_SURFACE_GEOMETRY.npz"
    assert paths["source_scene_snapshot"].name == "SOURCE_SCENE_SNAPSHOT.json"
    assert paths["camera_bundle"].name == "camera_bundle.json"
    assert paths["observation_dir"] == packet.resolve()

    with pytest.raises(ValueError, match="SUBJECT_DIRECTORY"):
        packet_paths(packet, "another_subject")

    (packet / "V7.mask.bin").unlink()
    with pytest.raises(FileNotFoundError, match="VIEW_FILE"):
        packet_paths(packet, "truth139_test")


def test_packet_binding_is_persisted_inside_chain_identity(tmp_path):
    chain_path = tmp_path / "GRAPHICS_MATERIALIZATION_CHAIN.json"
    base = {
        "schema": "RealSaS.Truth139GraphicsMaterializationChain.v1",
        "status": "PASS_TRUTH139_GRAPHICS_MATERIALIZATION_CHAIN_EXECUTED__DERIVED_TRUTH_ADMISSION_SEPARATE",
        "subject_id": "truth139_test",
        "chain_identity_sha256": "old",
    }
    binding = {
        "path": "/packet/truth139_test",
        "observation_binding_sha256": "a" * 64,
        "observation_admission_sha256": "b" * 64,
        "surface_geometry_sha256": "c" * 64,
        "surface_geometry_manifest_sha256": "d" * 64,
    }
    result = _persist_packet_bound_chain(chain_path, base, binding)
    saved = json.loads(chain_path.read_text())
    assert saved == result
    assert saved["prepared_packet"] == binding
    assert saved["chain_identity_sha256"] != "old"
    before = saved["chain_identity_sha256"]
    binding2 = dict(binding)
    binding2["surface_geometry_sha256"] = "e" * 64
    changed = _persist_packet_bound_chain(chain_path, base, binding2)
    assert changed["chain_identity_sha256"] != before
