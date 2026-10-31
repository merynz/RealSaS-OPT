#!/usr/bin/env python3
from __future__ import annotations

"""Independent raw-source court for RealSaS glTF rig/skin authority.

The producer does not import SourceGLTF. This court does, giving us a separate raw
file/accessor/world-transform implementation. The court rebuilds hierarchy, inverse
binds, primitive vertex order, joint slots, weights and triangle indices directly
from immutable source bytes and compares them to the producer artifact.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from tools.corpus.gltf_bind_scene_v1 import SourceGLTF


PARENT_SCHEMA = "RealSaS.GltfRigSkinAuthority.v1"
PARENT_STATUS = "PASS_GLTF_RIG_SKIN_SOURCE_AUTHORITY_CANDIDATE__INDEPENDENT_COURT_REQUIRED"
SCHEMA = "RealSaS.GltfRigSkinAuthorityCourt.v1"
STATUS = "PASS_GLTF_RIG_SKIN_SOURCE_AUTHORITY_INDEPENDENT_COURT__IMPORT_EQUIVALENCE_REQUIRED"


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


def bound(row: dict | None, code: str) -> Path:
    row = dict(row or {})
    path = Path(str(row.get("path") or "")).resolve()
    expected = str(row.get("sha256") or "")
    if not path.is_file() or len(expected) != 64 or sha256(path) != expected:
        raise ValueError(code)
    return path


def _matrix_from_accessor(source: SourceGLTF, accessor_index: int, count: int) -> np.ndarray:
    raw = np.asarray(source.accessor(int(accessor_index)), dtype=np.float64)
    if raw.shape != (count, 16):
        raise ValueError("GLTF_RIG_SKIN_COURT_INVERSE_BIND_LAYOUT")
    return raw.reshape(count, 4, 4).transpose(0, 2, 1)


def _local_from_world(source: SourceGLTF, node_index: int) -> np.ndarray:
    world = np.asarray(source.world(node_index), dtype=np.float64)
    parent = source.parents.get(node_index)
    if parent is None:
        return world
    return np.linalg.inv(np.asarray(source.world(parent), dtype=np.float64)) @ world


def _expected(source: SourceGLTF) -> tuple[list[dict], list[dict], dict[str, np.ndarray]]:
    doc = source.doc
    skins = list(doc.get("skins") or ())
    if not skins:
        raise ValueError("GLTF_RIG_SKIN_COURT_SKINS_MISSING")

    skin_rows = []
    joint_node_index = []
    parent_joint_slot = []
    joint_local_matrix = []
    joint_world_matrix = []
    inverse_bind_matrix = []
    skin_joint_offsets = [0]
    for skin_index, skin in enumerate(skins):
        joints = list(map(int, skin.get("joints") or ()))
        if not joints or len(set(joints)) != len(joints):
            raise ValueError("GLTF_RIG_SKIN_COURT_JOINT_SET")
        lut = {node_index: slot for slot, node_index in enumerate(joints)}
        if skin.get("inverseBindMatrices") is None:
            ibm = np.repeat(np.eye(4, dtype=np.float64)[None], len(joints), axis=0)
        else:
            ibm = _matrix_from_accessor(source, int(skin["inverseBindMatrices"]), len(joints))
        names = []
        parents = []
        for node_index in joints:
            node = doc["nodes"][node_index]
            names.append(str(node.get("name") or f"node_{node_index}"))
            parent_slot = lut.get(source.parents.get(node_index), -1)
            parents.append(parent_slot)
            joint_node_index.append(node_index)
            parent_joint_slot.append(parent_slot)
            joint_local_matrix.append(_local_from_world(source, node_index))
            joint_world_matrix.append(np.asarray(source.world(node_index), dtype=np.float64))
        inverse_bind_matrix.extend(ibm)
        skin_rows.append(
            {
                "skin_index": skin_index,
                "name": skin.get("name"),
                "joint_count": len(joints),
                "joint_node_indices": joints,
                "joint_names": names,
                "parent_joint_slots": parents,
                "inverse_bind_present": skin.get("inverseBindMatrices") is not None,
                "skeleton_node": skin.get("skeleton"),
            }
        )
        skin_joint_offsets.append(skin_joint_offsets[-1] + len(joints))

    instances = []
    for node_index, node in enumerate(doc.get("nodes") or ()):
        if "mesh" in node and "skin" in node:
            instances.append((node_index, int(node["mesh"]), int(node["skin"])))
    instances.sort(key=lambda row: (row[1], row[0], row[2]))

    primitive_rows = []
    position_rows = []
    joint_rows = []
    weight_rows = []
    vertex_skin_index = []
    triangle_rows = []
    primitive_vertex_offsets = [0]
    primitive_index_offsets = [0]
    for node_index, mesh_index, skin_index in instances:
        mesh = doc["meshes"][mesh_index]
        for primitive_index, primitive in enumerate(mesh.get("primitives") or ()):
            if primitive.get("targets"):
                raise ValueError("GLTF_RIG_SKIN_COURT_MORPH_TARGETS_UNSUPPORTED_V1")
            if primitive.get("mode", 4) != 4:
                raise ValueError("GLTF_RIG_SKIN_COURT_TRIANGLES_REQUIRED")
            attrs = dict(primitive.get("attributes") or {})
            if not {"POSITION", "JOINTS_0", "WEIGHTS_0"}.issubset(attrs):
                raise ValueError("GLTF_RIG_SKIN_COURT_ATTRIBUTES")
            position = np.asarray(source.accessor(attrs["POSITION"]), dtype=np.float64)
            joints = np.asarray(source.accessor(attrs["JOINTS_0"]), dtype=np.int32)
            weights = np.asarray(source.accessor(attrs["WEIGHTS_0"]), dtype=np.float64)
            if position.shape[1:] != (3,) or joints.shape != (len(position), 4) or weights.shape != (len(position), 4):
                raise ValueError("GLTF_RIG_SKIN_COURT_PRIMITIVE_LAYOUT")
            if primitive.get("indices") is None:
                indices = np.arange(len(position), dtype=np.int64)
            else:
                indices = np.asarray(source.accessor(primitive["indices"]), dtype=np.int64).reshape(-1)
            if len(indices) % 3 or np.any(indices < 0) or np.any(indices >= len(position)):
                raise ValueError("GLTF_RIG_SKIN_COURT_INDEX_LAYOUT")
            offset = primitive_vertex_offsets[-1]
            position_rows.append(position)
            joint_rows.append(joints)
            weight_rows.append(weights)
            vertex_skin_index.extend([skin_index] * len(position))
            triangle_rows.append(indices + offset)
            primitive_vertex_offsets.append(offset + len(position))
            primitive_index_offsets.append(primitive_index_offsets[-1] + len(indices))
            primitive_rows.append(
                {
                    "node_index": node_index,
                    "mesh_index": mesh_index,
                    "primitive_index": primitive_index,
                    "skin_index": skin_index,
                    "vertex_count": int(len(position)),
                    "triangle_count": int(len(indices) // 3),
                    "material_index": primitive.get("material"),
                }
            )
    if not position_rows:
        raise ValueError("GLTF_RIG_SKIN_COURT_NO_SKINNED_PRIMITIVES")
    arrays = {
        "source_position_local": np.concatenate(position_rows).astype(np.float64),
        "source_joint_slot": np.concatenate(joint_rows).astype(np.int32),
        "source_weight_value": np.concatenate(weight_rows).astype(np.float64),
        "primitive_vertex_offsets": np.asarray(primitive_vertex_offsets, dtype=np.int64),
        "primitive_index_offsets": np.asarray(primitive_index_offsets, dtype=np.int64),
        "triangle_index_stream": np.concatenate(triangle_rows).astype(np.int64),
        "vertex_skin_index": np.asarray(vertex_skin_index, dtype=np.int32),
        "skin_joint_offsets": np.asarray(skin_joint_offsets, dtype=np.int64),
        "joint_node_index": np.asarray(joint_node_index, dtype=np.int32),
        "parent_joint_slot": np.asarray(parent_joint_slot, dtype=np.int32),
        "joint_local_matrix": np.asarray(joint_local_matrix, dtype=np.float64),
        "joint_world_matrix": np.asarray(joint_world_matrix, dtype=np.float64),
        "inverse_bind_matrix": np.asarray(inverse_bind_matrix, dtype=np.float64),
    }
    return skin_rows, primitive_rows, arrays


def validate(manifest: dict) -> dict:
    if manifest.get("schema") != PARENT_SCHEMA or manifest.get("status") != PARENT_STATUS:
        raise ValueError("GLTF_RIG_SKIN_COURT_PARENT")
    source_path = bound(manifest.get("source"), "GLTF_RIG_SKIN_COURT_SOURCE_DRIFT")
    arrays_path = bound((manifest.get("outputs") or {}).get("arrays"), "GLTF_RIG_SKIN_COURT_ARRAY_DRIFT")
    source = SourceGLTF(source_path)
    skins, primitives, expected = _expected(source)
    if list(manifest.get("skins") or ()) != skins:
        raise ValueError("GLTF_RIG_SKIN_COURT_SKIN_MANIFEST")
    if list(manifest.get("primitives") or ()) != primitives:
        raise ValueError("GLTF_RIG_SKIN_COURT_PRIMITIVE_MANIFEST")

    with np.load(arrays_path, allow_pickle=False) as z:
        if set(z.files) != set(expected):
            raise ValueError("GLTF_RIG_SKIN_COURT_ARRAY_SET")
        for name, value in expected.items():
            actual = np.asarray(z[name])
            if np.issubdtype(value.dtype, np.floating):
                if not np.array_equal(actual, value):
                    max_abs = float(np.max(np.abs(actual.astype(np.float64) - value.astype(np.float64)))) if actual.shape == value.shape else float("inf")
                    if max_abs > 2.0e-12:
                        raise ValueError("GLTF_RIG_SKIN_COURT_ARRAY_MISMATCH:" + name)
            elif not np.array_equal(actual, value):
                raise ValueError("GLTF_RIG_SKIN_COURT_ARRAY_MISMATCH:" + name)

    weights = expected["source_weight_value"]
    active_count = np.count_nonzero(weights > 0, axis=1)
    weight_sum = weights.sum(axis=1)
    if not np.all(active_count > 0):
        raise ValueError("GLTF_RIG_SKIN_COURT_UNWEIGHTED_VERTEX")
    if np.max(np.abs(weight_sum - 1.0)) > 2.0e-5:
        raise ValueError("GLTF_RIG_SKIN_COURT_WEIGHT_SUM")
    if np.any(active_count > 4):
        raise ValueError("GLTF_RIG_SKIN_COURT_INFLUENCE_COUNT")
    if not np.isfinite(expected["inverse_bind_matrix"]).all():
        raise ValueError("GLTF_RIG_SKIN_COURT_INVERSE_BIND_FINITE")
    for matrix in expected["inverse_bind_matrix"]:
        if np.linalg.matrix_rank(matrix) != 4:
            raise ValueError("GLTF_RIG_SKIN_COURT_INVERSE_BIND_SINGULAR")

    counts = dict(manifest.get("counts") or {})
    expected_counts = {
        "skin_count": len(skins),
        "joint_count_total": int(len(expected["joint_node_index"])),
        "skinned_primitive_count": len(primitives),
        "render_vertex_count": int(len(expected["source_position_local"])),
        "triangle_count": int(len(expected["triangle_index_stream"]) // 3),
        "nonzero_influence_count": int(np.count_nonzero(weights > 0)),
        "maximum_influences_per_vertex": int(np.max(active_count)),
    }
    if counts != expected_counts:
        raise ValueError("GLTF_RIG_SKIN_COURT_COUNT_DRIFT")

    payload = {
        "schema": SCHEMA,
        "status": STATUS,
        "source": {"path": str(source_path), "sha256": sha256(source_path)},
        "producer_arrays": {"path": str(arrays_path), "sha256": sha256(arrays_path)},
        "independent_method": {
            "reader": "tools.corpus.gltf_bind_scene_v1.SourceGLTF",
            "producer_reader_reused": False,
            "hierarchy_reconstructed_from_nodes": True,
            "accessors_redecoded_from_source": True,
            "inverse_binds_redecoded_from_source": True,
        },
        "counts": expected_counts,
        "weight_quality": {
            "all_vertices_weighted": True,
            "maximum_influences_per_vertex": int(np.max(active_count)),
            "maximum_weight_sum_error": float(np.max(np.abs(weight_sum - 1.0))),
        },
        "authority": {
            "source_rig_skin_court_passed": True,
            "importer_equivalence_required": True,
            "family_authority_minted": False,
            "training_authorized": False,
        },
    }
    payload["court_identity_sha256"] = digest(payload)
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.resolve().read_text())
    payload = validate(manifest)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": payload["status"], "court_identity_sha256": payload["court_identity_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
