from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

from tools.corpus.build_gltf_animation_authority_v1 import build as build_animation
from tools.corpus.build_gltf_rig_skin_authority_v1 import build as build_rig_skin
from tools.corpus.validate_gltf_animation_authority_v1 import validate as validate_animation
from tools.corpus.validate_gltf_rig_skin_authority_v1 import validate as validate_rig_skin


def _align(blob: bytearray, value: int = 4) -> None:
    while len(blob) % value:
        blob.append(0)


def _write_glb(path: Path) -> None:
    blob = bytearray()
    views: list[dict] = []
    accessors: list[dict] = []

    def add(array: np.ndarray, component_type: int, kind: str) -> int:
        _align(blob)
        value = np.ascontiguousarray(array)
        offset = len(blob)
        blob.extend(value.tobytes())
        views.append({"buffer": 0, "byteOffset": offset, "byteLength": value.nbytes})
        accessors.append(
            {
                "bufferView": len(views) - 1,
                "byteOffset": 0,
                "componentType": component_type,
                "count": int(len(value)),
                "type": kind,
            }
        )
        return len(accessors) - 1

    position = add(np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32), 5126, "VEC3")
    joints = add(np.asarray([[0, 0, 0, 0]] * 3, np.uint16), 5123, "VEC4")
    weights = add(np.asarray([[1, 0, 0, 0]] * 3, np.float32), 5126, "VEC4")
    indices = add(np.asarray([0, 1, 2], np.uint16), 5123, "SCALAR")
    inverse_bind = add(np.eye(4, dtype=np.float32).reshape(1, 16), 5126, "MAT4")
    time = add(np.asarray([0.0, 0.5, 1.0], np.float32), 5126, "SCALAR")
    translation = add(np.asarray([[0, 0, 0], [0, 0.5, 0], [0, 1, 0]], np.float32), 5126, "VEC3")

    doc = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0, 1]}],
        "nodes": [
            {"name": "Root"},
            {"name": "Mesh", "mesh": 0, "skin": 0},
        ],
        "meshes": [
            {
                "primitives": [
                    {
                        "attributes": {
                            "POSITION": position,
                            "JOINTS_0": joints,
                            "WEIGHTS_0": weights,
                        },
                        "indices": indices,
                        "mode": 4,
                    }
                ]
            }
        ],
        "skins": [{"joints": [0], "inverseBindMatrices": inverse_bind}],
        "animations": [
            {
                "name": "Move",
                "samplers": [{"input": time, "output": translation, "interpolation": "LINEAR"}],
                "channels": [{"sampler": 0, "target": {"node": 0, "path": "translation"}}],
            }
        ],
        "accessors": accessors,
        "bufferViews": views,
        "buffers": [{"byteLength": len(blob)}],
    }

    json_bytes = json.dumps(doc, sort_keys=True, separators=(",", ":")).encode("utf-8")
    json_bytes += b" " * (-len(json_bytes) % 4)
    bin_bytes = bytes(blob) + b"\x00" * (-len(blob) % 4)
    total = 12 + 8 + len(json_bytes) + 8 + len(bin_bytes)
    payload = bytearray(struct.pack("<III", 0x46546C67, 2, total))
    payload.extend(struct.pack("<II", len(json_bytes), 0x4E4F534A))
    payload.extend(json_bytes)
    payload.extend(struct.pack("<II", len(bin_bytes), 0x004E4942))
    payload.extend(bin_bytes)
    path.write_bytes(payload)


def test_glb_uses_same_exact_gltf2_rig_skin_and_animation_courts(tmp_path: Path) -> None:
    source = tmp_path / "subject.glb"
    _write_glb(source)

    rig_manifest = build_rig_skin(source, tmp_path / "rig")
    assert rig_manifest["source_format"] == "GLTF_2_0"
    assert rig_manifest["source_container"] == "GLB"
    assert rig_manifest["counts"]["joint_count_total"] == 1
    assert rig_manifest["counts"]["render_vertex_count"] == 3
    rig_court = validate_rig_skin(rig_manifest)
    assert rig_court["status"] == "PASS_GLTF_RIG_SKIN_SOURCE_AUTHORITY_INDEPENDENT_COURT__IMPORT_EQUIVALENCE_REQUIRED"
    assert rig_court["authority"]["training_authorized"] is False

    animation_manifest = build_animation(source, tmp_path / "animation")
    assert animation_manifest["source_format"] == "GLTF_2_0"
    assert animation_manifest["source_container"] == "GLB"
    assert animation_manifest["counts"]["animation_count"] == 1
    assert animation_manifest["counts"]["joint_transform_channel_count"] == 1
    animation_court = validate_animation(animation_manifest)
    assert animation_court["status"] == "PASS_GLTF_ANIMATION_KINEMATICS_SOURCE_COURT__CANONICAL_REPLAY_REQUIRED"
    assert animation_court["authority"]["training_authorized"] is False
