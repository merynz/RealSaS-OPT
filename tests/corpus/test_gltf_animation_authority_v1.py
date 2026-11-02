from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
import pytest

from tools.corpus.build_gltf_animation_authority_v1 import build
from tools.corpus.validate_gltf_animation_authority_v1 import validate


def _align(blob: bytearray):
    while len(blob) % 4:
        blob.append(0)


def write_source(path: Path):
    blob = bytearray()
    views = []
    accessors = []

    def add(array, component_type, kind):
        _align(blob)
        start = len(blob)
        value = np.ascontiguousarray(array)
        blob.extend(value.tobytes())
        views.append({"buffer": 0, "byteOffset": start, "byteLength": value.nbytes})
        accessors.append({
            "bufferView": len(views) - 1,
            "byteOffset": 0,
            "componentType": component_type,
            "count": int(len(value)),
            "type": kind,
        })
        return len(accessors) - 1

    positions = add(np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32), 5126, "VEC3")
    joints = add(np.asarray([[0, 0, 0, 0]] * 3, np.uint16), 5123, "VEC4")
    weights = add(np.asarray([[1, 0, 0, 0]] * 3, np.float32), 5126, "VEC4")
    indices = add(np.asarray([0, 1, 2], np.uint16), 5123, "SCALAR")
    ibm = add(np.eye(4, dtype=np.float32).reshape(1, 16), 5126, "MAT4")
    time = add(np.asarray([0.0, 0.5, 1.0], np.float32), 5126, "SCALAR")
    translation = add(np.asarray([[0, 0, 0], [0, 0.5, 0], [0, 1, 0]], np.float32), 5126, "VEC3")
    rotation = add(np.asarray([[0, 0, 0, 1], [0, 0, 0.38268343, 0.9238795], [0, 0, 0.70710677, 0.70710677]], np.float32), 5126, "VEC4")

    doc = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0, 1]}],
        "nodes": [
            {"name": "Root"},
            {"name": "Mesh", "mesh": 0, "skin": 0},
        ],
        "meshes": [{"primitives": [{"attributes": {"POSITION": positions, "JOINTS_0": joints, "WEIGHTS_0": weights}, "indices": indices}]}],
        "skins": [{"joints": [0], "inverseBindMatrices": ibm}],
        "animations": [{
            "name": "MoveAndTurn",
            "samplers": [
                {"input": time, "output": translation, "interpolation": "LINEAR"},
                {"input": time, "output": rotation, "interpolation": "LINEAR"},
            ],
            "channels": [
                {"sampler": 0, "target": {"node": 0, "path": "translation"}},
                {"sampler": 1, "target": {"node": 0, "path": "rotation"}},
            ],
        }],
        "accessors": accessors,
        "bufferViews": views,
        "buffers": [{"byteLength": len(blob), "uri": "data:application/octet-stream;base64," + base64.b64encode(bytes(blob)).decode("ascii")}],
    }
    path.write_text(json.dumps(doc, sort_keys=True) + "\n")


def test_source_animation_authority_roundtrip(tmp_path):
    source = tmp_path / "animated.gltf"
    write_source(source)
    out = tmp_path / "out"
    manifest = build(source, out)
    assert manifest["counts"] == {
        "animation_count": 1,
        "channel_count": 2,
        "joint_transform_channel_count": 2,
        "sampler_key_count_sum": 6,
    }
    court = validate(manifest)
    assert court["status"] == "PASS_GLTF_ANIMATION_KINEMATICS_SOURCE_COURT__CANONICAL_REPLAY_REQUIRED"
    assert court["family_candidate_outcome"] == "SOURCE_CORE_CANDIDATE__CANONICAL_REPLAY_REQUIRED"
    assert court["authority"]["family_authority_minted"] is False


def test_source_animation_value_drift_is_rejected(tmp_path):
    source = tmp_path / "animated.gltf"
    write_source(source)
    out = tmp_path / "out"
    manifest = build(source, out)
    arrays_path = out / "GLTF_ANIMATION_KINEMATICS_V1.npz"
    with np.load(arrays_path, allow_pickle=False) as z:
        arrays = {name: np.asarray(z[name]) for name in z.files}
    key = "a0000_c0000_value"
    arrays[key] = arrays[key].copy()
    arrays[key][1, 1] += 0.01
    np.savez_compressed(arrays_path, **arrays)
    # Rebind producer hash so the independent court reaches the value comparison.
    import hashlib
    manifest["outputs"]["arrays"]["sha256"] = hashlib.sha256(arrays_path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="GLTF_ANIMATION_COURT_VALUE_DRIFT"):
        validate(manifest)
