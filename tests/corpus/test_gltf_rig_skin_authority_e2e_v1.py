from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tools.corpus.build_gltf_rig_skin_authority_v1 import build
from tools.corpus.validate_gltf_rig_skin_authority_v1 import validate as validate_source
from tools.corpus.validate_gltf_imported_rig_skin_equivalence_v1 import validate as validate_import


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _align(blob: bytearray, alignment=4):
    while len(blob) % alignment:
        blob.append(0)


def _append(blob: bytearray, array: np.ndarray):
    _align(blob)
    start = len(blob)
    data = np.ascontiguousarray(array).tobytes()
    blob.extend(data)
    return start, len(data)


def write_synthetic_gltf(path: Path):
    positions = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32
    )
    joints = np.asarray(
        [[0, 0, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0]], dtype=np.uint16
    )
    weights = np.asarray(
        [[1.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32
    )
    indices = np.asarray([0, 1, 2], dtype=np.uint16)
    inverse_bind = np.repeat(np.eye(4, dtype=np.float32)[None], 2, axis=0)

    blob = bytearray()
    views = []
    accessors = []

    def add(array, component_type, kind, *, normalized=False):
        offset, length = _append(blob, array)
        view_index = len(views)
        views.append({"buffer": 0, "byteOffset": offset, "byteLength": length})
        accessor_index = len(accessors)
        row = {
            "bufferView": view_index,
            "byteOffset": 0,
            "componentType": component_type,
            "count": int(len(array)),
            "type": kind,
        }
        if normalized:
            row["normalized"] = True
        accessors.append(row)
        return accessor_index

    a_position = add(positions, 5126, "VEC3")
    a_joints = add(joints, 5123, "VEC4")
    a_weights = add(weights, 5126, "VEC4")
    a_indices = add(indices, 5123, "SCALAR")
    # glTF MAT4 bytes are column-major. Identity is invariant to transpose.
    a_ibm = add(inverse_bind.reshape(2, 16), 5126, "MAT4")

    encoded = base64.b64encode(bytes(blob)).decode("ascii")
    doc = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [2, 0]}],
        "nodes": [
            {"name": "Root", "children": [1]},
            {"name": "Child", "translation": [0.0, 1.0, 0.0]},
            {"name": "Mesh", "mesh": 0, "skin": 0},
        ],
        "meshes": [
            {
                "name": "Triangle",
                "primitives": [
                    {
                        "attributes": {
                            "POSITION": a_position,
                            "JOINTS_0": a_joints,
                            "WEIGHTS_0": a_weights,
                        },
                        "indices": a_indices,
                        "mode": 4,
                    }
                ],
            }
        ],
        "skins": [
            {
                "name": "Rig",
                "joints": [0, 1],
                "inverseBindMatrices": a_ibm,
            }
        ],
        "accessors": accessors,
        "bufferViews": views,
        "buffers": [
            {
                "byteLength": len(blob),
                "uri": "data:application/octet-stream;base64," + encoded,
            }
        ],
    }
    path.write_text(json.dumps(doc, sort_keys=True) + "\n")
    return positions, joints.astype(np.int64), weights.astype(np.float64), indices.astype(np.int64)


def write_imported_stage47(root: Path, source_path: Path, positions, weights, indices):
    # Blender glTF import basis used by the existing independent raw reader:
    # (x, y, z) -> (x, -z, y).
    imported = positions[:, [0, 2, 1]].astype(np.float64)
    imported[:, 1] *= -1.0
    faces = indices.reshape(-1, 3).astype(np.int64)

    # Teacher frame order is deliberately different from source joint-slot order.
    # source slots Root=0, Child=1 -> teacher frame Child=0, Root=1.
    offsets = np.asarray([0, 1, 3, 4], dtype=np.int64)
    bone_indices = np.asarray([1, 1, 0, 0], dtype=np.int64)
    values = np.asarray([1.0, 0.5, 0.5, 1.0], dtype=np.float64)
    coverage = np.asarray([True, True, True], dtype=np.bool_)

    geometry = root / "geometry_and_response.npz"
    np.savez_compressed(
        geometry,
        rest_vertices_source_world=imported,
        faces=faces,
        source_weight_offsets=offsets,
        source_weight_bone_indices_privileged=bone_indices,
        source_weight_values=values,
        source_weight_coverage=coverage,
    )
    teacher = root / "TEACHER_SCENE.json"
    teacher.write_text(
        json.dumps(
            {
                "schema": "RealSaS.TeacherSceneIR.v2",
                "subject_key": "synthetic-rig-skin",
                "authority": "UNQUALIFIED_DERIVED_MEASUREMENTS",
                "geometry_payloads": [
                    {
                        "path": str(geometry.resolve()),
                        "sha256": sha256(geometry),
                        "size_bytes": geometry.stat().st_size,
                    }
                ],
                "calibration": {
                    "native_control_frames_privileged": [
                        {
                            "armature": "Rig",
                            "name": "Child",
                            "parent": "Root",
                            "rest_matrix_world": np.eye(4).tolist(),
                        },
                        {
                            "armature": "Rig",
                            "name": "Root",
                            "parent": None,
                            "rest_matrix_world": np.eye(4).tolist(),
                        },
                    ]
                },
            },
            sort_keys=True,
        )
        + "\n"
    )
    return teacher, geometry


def test_raw_source_and_imported_skin_equivalence_end_to_end(tmp_path):
    source = tmp_path / "subject.gltf"
    positions, joints, weights, indices = write_synthetic_gltf(source)
    authority_dir = tmp_path / "authority"
    manifest = build(source, authority_dir)
    assert manifest["counts"]["joint_count_total"] == 2
    assert manifest["counts"]["render_vertex_count"] == 3
    assert manifest["counts"]["triangle_count"] == 1
    assert manifest["counts"]["nonzero_influence_count"] == 4

    source_court = validate_source(manifest)
    assert source_court["status"] == "PASS_GLTF_RIG_SKIN_SOURCE_AUTHORITY_INDEPENDENT_COURT__IMPORT_EQUIVALENCE_REQUIRED"
    source_court_path = authority_dir / "GLTF_RIG_SKIN_AUTHORITY_COURT_V1.json"
    source_court_path.write_text(json.dumps(source_court, sort_keys=True, indent=2) + "\n")

    teacher, geometry = write_imported_stage47(tmp_path, source, positions, weights, indices)
    import_court = validate_import(
        authority_dir / "GLTF_RIG_SKIN_AUTHORITY_MANIFEST_V1.json",
        source_court_path,
        teacher,
        geometry,
    )
    assert import_court["status"] == "PASS_GLTF_IMPORTED_RIG_SKIN_EQUIVALENCE__FINAL_ADMISSION_SEPARATE"
    assert import_court["family_candidate_outcomes"] == {
        "RIG_HIERARCHY": "QUALIFIED_CORE_CANDIDATE",
        "SKIN_WEIGHTS": "QUALIFIED_CORE_CANDIDATE",
    }
    assert import_court["equivalence"]["weight_indices_exact_after_joint_name_remap"] is True
    assert import_court["equivalence"]["semantic_skin_transfer_used"] is False
    assert import_court["authority"]["training_authorized"] is False


def test_imported_weight_value_drift_fails_closed(tmp_path):
    source = tmp_path / "subject.gltf"
    positions, joints, weights, indices = write_synthetic_gltf(source)
    authority_dir = tmp_path / "authority"
    manifest = build(source, authority_dir)
    source_court = validate_source(manifest)
    source_court_path = authority_dir / "GLTF_RIG_SKIN_AUTHORITY_COURT_V1.json"
    source_court_path.write_text(json.dumps(source_court, sort_keys=True, indent=2) + "\n")
    teacher, geometry = write_imported_stage47(tmp_path, source, positions, weights, indices)

    with np.load(geometry, allow_pickle=False) as z:
        arrays = {name: np.asarray(z[name]) for name in z.files}
    arrays["source_weight_values"] = arrays["source_weight_values"].copy()
    arrays["source_weight_values"][0] -= 0.01
    arrays["source_weight_values"][1] += 0.01
    np.savez_compressed(geometry, **arrays)
    # Rebind teacher geometry hash so the failure reaches the weight-equivalence court.
    payload = json.loads(teacher.read_text())
    payload["geometry_payloads"][0]["sha256"] = sha256(geometry)
    teacher.write_text(json.dumps(payload, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="GLTF_IMPORTED_EQUIVALENCE_WEIGHT_VALUE_DRIFT"):
        validate_import(
            authority_dir / "GLTF_RIG_SKIN_AUTHORITY_MANIFEST_V1.json",
            source_court_path,
            teacher,
            geometry,
        )
