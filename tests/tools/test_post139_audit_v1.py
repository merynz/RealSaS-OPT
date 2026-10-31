from __future__ import annotations

import base64
import json
import struct
from pathlib import Path

from tools.corpus.post139_audit_v1 import (
    Candidate,
    _direct_gltf_structure,
    decide_admissions,
    technical_audit,
)


def _pack_floats(values):
    return struct.pack("<" + "f" * len(values), *values)


def _pack_u16(values):
    return struct.pack("<" + "H" * len(values), *values)


def _minimal_gltf(path: Path, *, position_delta=0.0, material_color=(1.0, 0.0, 0.0, 1.0)):
    positions = [0.0, 0.0, 0.0, 1.0 + position_delta, 0.0, 0.0, 0.0, 1.0, 0.0]
    indices = [0, 1, 2]
    joints = [0, 0, 0, 0] * 3
    weights = [1.0, 0.0, 0.0, 0.0] * 3
    chunks = [_pack_floats(positions), _pack_u16(indices), _pack_u16(joints), _pack_floats(weights)]
    offsets = []
    blob = bytearray()
    for chunk in chunks:
        offsets.append(len(blob))
        blob.extend(chunk)
    uri = "data:application/octet-stream;base64," + base64.b64encode(bytes(blob)).decode()
    doc = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [1]}],
        "nodes": [
            {"name": "Root"},
            {"name": "Mesh", "mesh": 0, "skin": 0, "children": [0]},
        ],
        "buffers": [{"byteLength": len(blob), "uri": uri}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": offsets[0], "byteLength": len(chunks[0])},
            {"buffer": 0, "byteOffset": offsets[1], "byteLength": len(chunks[1])},
            {"buffer": 0, "byteOffset": offsets[2], "byteLength": len(chunks[2])},
            {"buffer": 0, "byteOffset": offsets[3], "byteLength": len(chunks[3])},
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
            {"bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR"},
            {"bufferView": 2, "componentType": 5123, "count": 3, "type": "VEC4"},
            {"bufferView": 3, "componentType": 5126, "count": 3, "type": "VEC4"},
        ],
        "materials": [{"pbrMetallicRoughness": {"baseColorFactor": list(material_color)}}],
        "meshes": [
            {
                "primitives": [
                    {
                        "attributes": {"POSITION": 0, "JOINTS_0": 2, "WEIGHTS_0": 3},
                        "indices": 1,
                        "material": 0,
                    }
                ]
            }
        ],
        "skins": [{"joints": [0]}],
        "animations": [{"name": "Idle", "channels": [], "samplers": []}],
    }
    path.write_text(json.dumps(doc), encoding="utf-8")


def _row(path: Path, subject_id: str, structure: dict):
    return {
        "candidate": Candidate(path, subject_id, "Test", "Pack", "CLEAN_CC0_HUMAN_AUTHORED", "CC0"),
        "structure": structure,
        "technical_audit": technical_audit(structure),
    }


def test_direct_gltf_extracts_required_truth_and_animation(tmp_path):
    source = tmp_path / "actor.gltf"
    _minimal_gltf(source)
    structure = _direct_gltf_structure(source)
    assert structure["capabilities"] == {
        "geometry": True,
        "appearance": True,
        "rig": True,
        "skin": True,
        "animation": True,
    }
    audit = technical_audit(structure)
    assert audit["status"] == "PASS"
    assert audit["canonical_dedup_ready"] is False


def test_color_only_variant_has_same_structural_hash_but_different_appearance(tmp_path):
    red = tmp_path / "red.gltf"
    blue = tmp_path / "blue.gltf"
    _minimal_gltf(red, material_color=(1.0, 0.0, 0.0, 1.0))
    _minimal_gltf(blue, material_color=(0.0, 0.0, 1.0, 1.0))
    a = _direct_gltf_structure(red)
    b = _direct_gltf_structure(blue)
    assert a["fingerprints"]["structural_subject"] == b["fingerprints"]["structural_subject"]
    assert a["fingerprints"]["appearance"] != b["fingerprints"]["appearance"]
    decisions = decide_admissions([_row(red, "red", a), _row(blue, "blue", b)], {})
    by_id = {row["subject_id"]: row for row in decisions}
    assert by_id["blue"]["status"] == "AUDIT_READY_PENDING_CANONICAL_DEDUP"
    assert by_id["red"]["status"] == "DUPLICATE_COLLAPSE"
    assert by_id["red"]["duplicate_of"] == "blue"


def test_same_rig_different_geometry_is_not_duplicate(tmp_path):
    a_path = tmp_path / "a.gltf"
    b_path = tmp_path / "b.gltf"
    _minimal_gltf(a_path, position_delta=0.0)
    _minimal_gltf(b_path, position_delta=0.4)
    a = _direct_gltf_structure(a_path)
    b = _direct_gltf_structure(b_path)
    assert a["fingerprints"]["rig_rest"] == b["fingerprints"]["rig_rest"]
    assert a["fingerprints"]["structural_subject"] != b["fingerprints"]["structural_subject"]
    decisions = decide_admissions([_row(a_path, "a", a), _row(b_path, "b", b)], {})
    assert {row["status"] for row in decisions} == {"AUDIT_READY_PENDING_CANONICAL_DEDUP"}


def test_reference_structural_match_collapses(tmp_path):
    source = tmp_path / "actor.gltf"
    _minimal_gltf(source)
    structure = _direct_gltf_structure(source)
    fingerprint = structure["fingerprints"]["structural_subject"]
    decisions = decide_admissions(
        [_row(source, "candidate", structure)],
        {fingerprint: {"subject_id": "base139-actor", "source_sha256": "x", "canonical_space": "test"}},
    )
    assert decisions[0]["status"] == "DUPLICATE_COLLAPSE"
    assert decisions[0]["duplicate_of"] == "base139-actor"
