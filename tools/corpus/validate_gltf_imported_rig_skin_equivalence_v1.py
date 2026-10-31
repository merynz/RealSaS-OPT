#!/usr/bin/env python3
from __future__ import annotations

"""Independent source -> imported rig/skin equivalence court.

This court compares immutable raw glTF skin truth to the imported Stage47 evidence
that downstream AXIS/MIRA use. It validates exact topology stream, coordinate-basis
conversion, native hierarchy by joint name, and per-render-vertex skin weights.
No semantic rebind or approximate nearest-neighbour transfer is allowed.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from tools.corpus.gltf_bind_scene_v1 import SourceGLTF, Y_TO_Z


SOURCE_MANIFEST_SCHEMA = "RealSaS.GltfRigSkinAuthority.v1"
SOURCE_COURT_SCHEMA = "RealSaS.GltfRigSkinAuthorityCourt.v1"
SOURCE_COURT_STATUS = "PASS_GLTF_RIG_SKIN_SOURCE_AUTHORITY_INDEPENDENT_COURT__IMPORT_EQUIVALENCE_REQUIRED"
SCHEMA = "RealSaS.GltfImportedRigSkinEquivalenceCourt.v1"
STATUS = "PASS_GLTF_IMPORTED_RIG_SKIN_EQUIVALENCE__FINAL_ADMISSION_SEPARATE"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def load_json(path: Path, code: str) -> dict:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(code + ":" + str(path))
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(code + "_NOT_OBJECT")
    return value


def bound(row: dict | None, code: str) -> Path:
    row = dict(row or {})
    path = Path(str(row.get("path") or "")).resolve()
    expected = str(row.get("sha256") or "")
    if not path.is_file() or len(expected) != 64 or sha256(path) != expected:
        raise ValueError(code)
    return path


def _source_streams(source: SourceGLTF):
    doc = source.doc
    skins = list(doc.get("skins") or ())
    instances = []
    for node_index, node in enumerate(doc.get("nodes") or ()):
        if "mesh" in node and "skin" in node:
            instances.append((node_index, int(node["mesh"]), int(node["skin"])))
    instances.sort(key=lambda row: (row[1], row[0], row[2]))

    vertices_world = []
    faces = []
    joints = []
    weights = []
    vertex_skin = []
    offset = 0
    for node_index, mesh_index, skin_index in instances:
        mesh = doc["meshes"][mesh_index]
        world = np.asarray(source.world(node_index), dtype=np.float64)
        for primitive in mesh.get("primitives") or ():
            if primitive.get("targets"):
                raise ValueError("GLTF_IMPORTED_EQUIVALENCE_MORPH_TARGETS_UNSUPPORTED_V1")
            attrs = dict(primitive.get("attributes") or {})
            local = np.asarray(source.accessor(attrs["POSITION"]), dtype=np.float64)
            world_xyz = local @ world[:3, :3].T + world[:3, 3]
            blender_xyz = world_xyz @ np.asarray(Y_TO_Z, dtype=np.float64).T
            source_joints = np.asarray(source.accessor(attrs["JOINTS_0"]), dtype=np.int64)
            source_weights = np.asarray(source.accessor(attrs["WEIGHTS_0"]), dtype=np.float64)
            if primitive.get("indices") is None:
                index = np.arange(len(local), dtype=np.int64)
            else:
                index = np.asarray(source.accessor(primitive["indices"]), dtype=np.int64).reshape(-1)
            vertices_world.append(blender_xyz)
            joints.append(source_joints)
            weights.append(source_weights)
            vertex_skin.extend([skin_index] * len(local))
            faces.append(index.reshape(-1, 3) + offset)
            offset += len(local)
    if not vertices_world:
        raise ValueError("GLTF_IMPORTED_EQUIVALENCE_NO_SKINNED_GEOMETRY")
    return (
        np.concatenate(vertices_world),
        np.concatenate(faces).astype(np.int64),
        np.concatenate(joints).astype(np.int64),
        np.concatenate(weights).astype(np.float64),
        np.asarray(vertex_skin, dtype=np.int32),
        skins,
    )


def _source_joint_records(source: SourceGLTF, skins: list[dict]):
    doc = source.doc
    rows = []
    for skin_index, skin in enumerate(skins):
        joints = list(map(int, skin.get("joints") or ()))
        lut = {node_index: slot for slot, node_index in enumerate(joints)}
        names = [str(doc["nodes"][node_index].get("name") or f"node_{node_index}") for node_index in joints]
        if len(set(names)) != len(names):
            raise ValueError("GLTF_IMPORTED_EQUIVALENCE_DUPLICATE_JOINT_NAMES")
        for slot, node_index in enumerate(joints):
            parent_node = source.parents.get(node_index)
            parent_slot = lut.get(parent_node, -1)
            parent_name = names[parent_slot] if parent_slot >= 0 else None
            rows.append(
                {
                    "skin_index": skin_index,
                    "joint_slot": slot,
                    "node_index": node_index,
                    "name": names[slot],
                    "parent_name": parent_name,
                }
            )
    return rows


def _expected_csr(joints, weights, vertex_skin, source_joint_records, frame_index_by_name):
    by_skin_slot = {
        (row["skin_index"], row["joint_slot"]): row["name"]
        for row in source_joint_records
    }
    offsets = [0]
    indices = []
    values = []
    coverage = []
    for vertex, (joint_row, weight_row) in enumerate(zip(joints, weights)):
        skin_index = int(vertex_skin[vertex])
        used = 0
        for slot, weight in zip(joint_row, weight_row):
            weight = float(weight)
            if weight <= 0.0:
                continue
            name = by_skin_slot.get((skin_index, int(slot)))
            if name is None:
                raise ValueError("GLTF_IMPORTED_EQUIVALENCE_JOINT_SLOT")
            if name not in frame_index_by_name:
                raise ValueError("GLTF_IMPORTED_EQUIVALENCE_FRAME_NAME_MISSING:" + name)
            indices.append(int(frame_index_by_name[name]))
            values.append(weight)
            used += 1
        coverage.append(used > 0)
        offsets.append(len(indices))
    return (
        np.asarray(offsets, dtype=np.int64),
        np.asarray(indices, dtype=np.int64),
        np.asarray(values, dtype=np.float64),
        np.asarray(coverage, dtype=np.bool_),
    )


def validate(source_manifest_path: Path, source_court_path: Path, teacher_scene_path: Path, geometry_response_path: Path) -> dict:
    source_manifest = load_json(source_manifest_path, "GLTF_IMPORTED_EQUIVALENCE_SOURCE_MANIFEST")
    source_court = load_json(source_court_path, "GLTF_IMPORTED_EQUIVALENCE_SOURCE_COURT")
    teacher = load_json(teacher_scene_path, "GLTF_IMPORTED_EQUIVALENCE_TEACHER")
    if source_manifest.get("schema") != SOURCE_MANIFEST_SCHEMA:
        raise ValueError("GLTF_IMPORTED_EQUIVALENCE_SOURCE_SCHEMA")
    if source_court.get("schema") != SOURCE_COURT_SCHEMA or source_court.get("status") != SOURCE_COURT_STATUS:
        raise ValueError("GLTF_IMPORTED_EQUIVALENCE_SOURCE_COURT_STATUS")
    source_path = bound(source_manifest.get("source"), "GLTF_IMPORTED_EQUIVALENCE_SOURCE_DRIFT")
    if (source_court.get("source") or {}).get("sha256") != sha256(source_path):
        raise ValueError("GLTF_IMPORTED_EQUIVALENCE_SOURCE_COURT_BINDING")
    geometry_ref = None
    for row in teacher.get("geometry_payloads") or ():
        if Path(str(row.get("path") or "")).name == Path(geometry_response_path).name:
            geometry_ref = row
            break
    if geometry_ref is not None and str(geometry_ref.get("sha256") or "") != sha256(geometry_response_path.resolve()):
        raise ValueError("GLTF_IMPORTED_EQUIVALENCE_TEACHER_GEOMETRY_DRIFT")
    if teacher.get("authority") not in {"UNQUALIFIED_DERIVED_MEASUREMENTS", "QUALIFIED_DERIVED_MEASUREMENTS"}:
        raise ValueError("GLTF_IMPORTED_EQUIVALENCE_TEACHER_AUTHORITY")

    source = SourceGLTF(source_path)
    vertices, faces, joints, weights, vertex_skin, skins = _source_streams(source)
    source_joint_records = _source_joint_records(source, skins)

    frames = list((teacher.get("calibration") or {}).get("native_control_frames_privileged") or ())
    if not frames:
        raise ValueError("GLTF_IMPORTED_EQUIVALENCE_NATIVE_FRAMES_MISSING")
    frame_names = [str(row.get("name") or "") for row in frames]
    if not all(frame_names) or len(set(frame_names)) != len(frame_names):
        raise ValueError("GLTF_IMPORTED_EQUIVALENCE_FRAME_NAMES")
    source_names = [row["name"] for row in source_joint_records]
    if set(frame_names) != set(source_names) or len(frame_names) != len(source_names):
        raise ValueError("GLTF_IMPORTED_EQUIVALENCE_JOINT_NAME_SET")
    frame_index_by_name = {name: index for index, name in enumerate(frame_names)}
    frame_by_name = {str(row["name"]): row for row in frames}
    for row in source_joint_records:
        frame = frame_by_name[row["name"]]
        if frame.get("parent") != row["parent_name"]:
            raise ValueError("GLTF_IMPORTED_EQUIVALENCE_PARENT_DRIFT:" + row["name"])
        matrix = np.asarray(frame.get("rest_matrix_world"), dtype=np.float64)
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all() or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-8, rtol=0):
            raise ValueError("GLTF_IMPORTED_EQUIVALENCE_FRAME_MATRIX:" + row["name"])

    expected_offsets, expected_indices, expected_values, expected_coverage = _expected_csr(
        joints, weights, vertex_skin, source_joint_records, frame_index_by_name
    )
    with np.load(geometry_response_path.resolve(), allow_pickle=False) as z:
        required = {
            "rest_vertices_source_world", "faces", "source_weight_offsets",
            "source_weight_bone_indices_privileged", "source_weight_values",
            "source_weight_coverage",
        }
        if not required.issubset(z.files):
            raise ValueError("GLTF_IMPORTED_EQUIVALENCE_STAGE47_ARRAYS")
        imported_vertices = np.asarray(z["rest_vertices_source_world"], dtype=np.float64)
        imported_faces = np.asarray(z["faces"], dtype=np.int64)
        offsets = np.asarray(z["source_weight_offsets"], dtype=np.int64)
        indices = np.asarray(z["source_weight_bone_indices_privileged"], dtype=np.int64)
        values = np.asarray(z["source_weight_values"], dtype=np.float64)
        coverage = np.asarray(z["source_weight_coverage"], dtype=np.bool_)

    if imported_vertices.shape != vertices.shape or not np.allclose(imported_vertices, vertices, atol=2.0e-6, rtol=0):
        max_abs = float(np.max(np.abs(imported_vertices - vertices))) if imported_vertices.shape == vertices.shape else float("inf")
        raise ValueError("GLTF_IMPORTED_EQUIVALENCE_VERTEX_DRIFT:" + str(max_abs))
    if not np.array_equal(imported_faces, faces):
        raise ValueError("GLTF_IMPORTED_EQUIVALENCE_FACE_DRIFT")
    if not np.array_equal(offsets, expected_offsets):
        raise ValueError("GLTF_IMPORTED_EQUIVALENCE_WEIGHT_OFFSET_DRIFT")
    if not np.array_equal(indices, expected_indices):
        raise ValueError("GLTF_IMPORTED_EQUIVALENCE_WEIGHT_INDEX_DRIFT")
    if not np.array_equal(values, expected_values):
        if values.shape != expected_values.shape or np.max(np.abs(values - expected_values)) > 1.0e-12:
            raise ValueError("GLTF_IMPORTED_EQUIVALENCE_WEIGHT_VALUE_DRIFT")
    if not np.array_equal(coverage, expected_coverage) or not np.all(coverage):
        raise ValueError("GLTF_IMPORTED_EQUIVALENCE_WEIGHT_COVERAGE_DRIFT")

    row_counts = np.diff(offsets)
    weight_sums = np.add.reduceat(values, offsets[:-1])
    payload = {
        "schema": SCHEMA,
        "status": STATUS,
        "subject_id": teacher.get("subject_key"),
        "inputs": {
            "source_manifest": {"path": str(source_manifest_path.resolve()), "sha256": sha256(source_manifest_path.resolve())},
            "source_court": {"path": str(source_court_path.resolve()), "sha256": sha256(source_court_path.resolve())},
            "raw_source": {"path": str(source_path), "sha256": sha256(source_path)},
            "teacher_scene": {"path": str(teacher_scene_path.resolve()), "sha256": sha256(teacher_scene_path.resolve())},
            "geometry_and_response": {"path": str(geometry_response_path.resolve()), "sha256": sha256(geometry_response_path.resolve())},
        },
        "counts": {
            "joint_count": len(source_joint_records),
            "render_vertex_count": int(len(vertices)),
            "triangle_count": int(len(faces)),
            "influence_count": int(len(values)),
            "maximum_influences_per_vertex": int(np.max(row_counts)),
        },
        "equivalence": {
            "source_to_imported_coordinate_basis": "GLTF_Y_UP_TO_BLENDER_Z_UP__X_NEGZ_Y",
            "maximum_vertex_abs_error": float(np.max(np.abs(imported_vertices - vertices))),
            "hierarchy_name_parent_exact": True,
            "faces_exact": True,
            "weight_offsets_exact": True,
            "weight_indices_exact_after_joint_name_remap": True,
            "weight_values_exact": bool(np.array_equal(values, expected_values)),
            "all_vertices_weight_covered": True,
            "maximum_weight_sum_error": float(np.max(np.abs(weight_sums - 1.0))),
            "semantic_skin_transfer_used": False,
            "nearest_neighbor_skin_transfer_used": False,
        },
        "family_candidate_outcomes": {
            "RIG_HIERARCHY": "QUALIFIED_CORE_CANDIDATE",
            "SKIN_WEIGHTS": "QUALIFIED_CORE_CANDIDATE",
        },
        "authority": {
            "source_court_passed": True,
            "importer_equivalence_passed": True,
            "final_subject_disposition_minted": False,
            "training_authorized": False,
        },
    }
    payload["court_identity_sha256"] = digest(payload)
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-court", type=Path, required=True)
    parser.add_argument("--teacher-scene", type=Path, required=True)
    parser.add_argument("--geometry-and-response", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = validate(args.source_manifest, args.source_court, args.teacher_scene, args.geometry_and_response)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": payload["status"], "subject_id": payload["subject_id"], "family_candidate_outcomes": payload["family_candidate_outcomes"], "court_identity_sha256": payload["court_identity_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
