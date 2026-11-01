from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import numpy as np

from tools.corpus.build_gltf_animation_authority_v1 import build as build_animation
from tools.corpus.build_gltf_animation_canonical_replay_v1 import build as build_replay
from tools.corpus.build_gltf_rig_skin_authority_v1 import build as build_rig_skin
from tools.corpus.validate_gltf_animation_authority_v1 import validate as validate_animation
from tools.corpus.validate_gltf_animation_canonical_replay_v1 import validate as validate_replay
from tools.corpus.validate_gltf_rig_skin_authority_v1 import validate as validate_rig_skin


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    return path


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


def test_glb_canonical_animation_replay_uses_binary_safe_source_reader(tmp_path: Path) -> None:
    source = tmp_path / "subject.glb"
    _write_glb(source)

    rig_dir = tmp_path / "rig"
    animation_dir = tmp_path / "animation"
    build_rig_skin(source, rig_dir)
    animation_manifest = build_animation(source, animation_dir)
    animation_court = validate_animation(animation_manifest)

    rig_manifest_path = rig_dir / "GLTF_RIG_SKIN_AUTHORITY_MANIFEST_V1.json"
    animation_manifest_path = animation_dir / "GLTF_ANIMATION_KINEMATICS_MANIFEST_V1.json"
    animation_court_path = _write_json(animation_dir / "GLTF_ANIMATION_KINEMATICS_COURT_V1.json", animation_court)
    teacher_path = _write_json(
        tmp_path / "teacher.json",
        {
            "schema": "RealSaS.TestTeacherScene.v1",
            "subject_key": "subject_glb",
            "calibration": {"native_control_frames_privileged": [{"name": "Root"}]},
        },
    )
    importer_court_path = tmp_path / "GLTF_IMPORTED_RIG_SKIN_EQUIVALENCE_COURT_V1.json"
    importer_court = {
        "schema": "RealSaS.GltfImportedRigSkinEquivalenceCourt.v1",
        "status": "PASS_GLTF_IMPORTED_RIG_SKIN_EQUIVALENCE__FINAL_ADMISSION_SEPARATE",
        "subject_id": "subject_glb",
        "inputs": {
            "raw_source": {"path": str(source.resolve()), "sha256": _sha(source)},
            "teacher_scene": {"path": str(teacher_path.resolve()), "sha256": _sha(teacher_path)},
        },
        "authority": {"importer_equivalence_passed": True, "training_authorized": False},
    }
    _write_json(importer_court_path, importer_court)

    replay_path = tmp_path / "GLTF_ANIMATION_CANONICAL_REPLAY_V1.json"
    replay = build_replay(
        animation_manifest_path,
        animation_court_path,
        rig_manifest_path,
        importer_court_path,
        teacher_path,
        replay_path,
    )
    assert replay["status"] == "PASS_GLTF_ANIMATION_CANONICAL_REPLAY_CANDIDATE__INDEPENDENT_COURT_REQUIRED"
    assert replay["subject_id"] == "subject_glb"
    assert replay["policy"]["gltf_and_glb_share_same_source_reader"] is True

    replay_court = validate_replay(replay_path)
    assert replay_court["status"] == "PASS_GLTF_ANIMATION_CANONICAL_REPLAY_INDEPENDENT_COURT__FINAL_ADMISSION_SEPARATE"
    assert replay_court["family_candidate_outcomes"]["ANIMATION_KINEMATICS"] == "QUALIFIED_CORE_CANDIDATE"
    assert replay_court["authority"]["training_authorized"] is False
