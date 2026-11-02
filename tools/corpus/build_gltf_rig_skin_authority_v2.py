#!/usr/bin/env python3
from __future__ import annotations

"""Build glTF2 rig/skin source authority candidate with semantic quaternion normalization.

V1 exposed raw quaternion magnitude to the TRS matrix formula.  The independent
SourceGLTF court (SciPy Rotation) interprets glTF rotations as unit quaternions,
which correctly exposed a systematic transform mismatch on exported rigs whose
quaternion rows are only approximately normalized.  V2 keeps the immutable V1
artifacts as history and makes the source semantic explicit: a finite non-zero
rotation quaternion is normalized to unit length before matrix construction.

No imported representation, Stage47 artifact, AXIS/MIRA output or semantic skin
transfer is consumed here.  Importer equivalence remains a separate required court.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from tools.corpus import build_gltf_rig_skin_authority_v1 as v1
from tools.observation.source_scene_contract_v2 import _json_and_chunks


SCHEMA = "RealSaS.GltfRigSkinAuthority.v2"
STATUS = "PASS_GLTF_RIG_SKIN_SOURCE_AUTHORITY_V2_CANDIDATE__INDEPENDENT_COURT_REQUIRED"


def _quat_matrix(value) -> np.ndarray:
    q = np.asarray(value, dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all():
        raise ValueError("GLTF_RIG_SKIN_V2_ROTATION_QUATERNION")
    norm = float(np.linalg.norm(q))
    if not np.isfinite(norm) or norm <= 1.0e-12:
        raise ValueError("GLTF_RIG_SKIN_V2_ROTATION_QUATERNION_ZERO_NORM")
    x, y, z, w = map(float, q / norm)
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y), 0],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x), 0],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y), 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float64,
    )


def _local_matrix(node: dict) -> np.ndarray:
    if "matrix" in node:
        value = np.asarray(node["matrix"], dtype=np.float64)
        if value.shape != (16,) or not np.isfinite(value).all():
            raise ValueError("GLTF_RIG_SKIN_V2_NODE_MATRIX")
        return value.reshape(4, 4).T
    translation = np.eye(4, dtype=np.float64)
    translation[:3, 3] = np.asarray(node.get("translation", [0, 0, 0]), dtype=np.float64)
    rotation = _quat_matrix(node.get("rotation", [0, 0, 0, 1]))
    scale = np.eye(4, dtype=np.float64)
    scale_value = np.asarray(node.get("scale", [1, 1, 1]), dtype=np.float64)
    if scale_value.shape != (3,) or not np.isfinite(scale_value).all():
        raise ValueError("GLTF_RIG_SKIN_V2_NODE_SCALE")
    scale[0, 0], scale[1, 1], scale[2, 2] = scale_value
    return translation @ rotation @ scale


def _world_matrices(nodes: list[dict], parents: dict[int, int]) -> list[np.ndarray]:
    cache: dict[int, np.ndarray] = {}
    visiting: set[int] = set()

    def solve(index: int) -> np.ndarray:
        if index in cache:
            return cache[index]
        if index in visiting:
            raise ValueError("GLTF_RIG_SKIN_V2_NODE_CYCLE")
        visiting.add(index)
        local = _local_matrix(nodes[index])
        result = solve(parents[index]) @ local if index in parents else local
        visiting.remove(index)
        cache[index] = result
        return result

    return [solve(index) for index in range(len(nodes))]


def _rotation_diagnostics(nodes: list[dict]) -> dict:
    norms = []
    for node in nodes:
        if "rotation" not in node:
            continue
        q = np.asarray(node["rotation"], dtype=np.float64)
        if q.shape != (4,) or not np.isfinite(q).all():
            raise ValueError("GLTF_RIG_SKIN_V2_ROTATION_QUATERNION")
        norm = float(np.linalg.norm(q))
        if norm <= 1.0e-12:
            raise ValueError("GLTF_RIG_SKIN_V2_ROTATION_QUATERNION_ZERO_NORM")
        norms.append(norm)
    return {
        "authored_rotation_quaternion_count": len(norms),
        "maximum_source_quaternion_norm_abs_error": max((abs(value - 1.0) for value in norms), default=0.0),
        "normalization_applied_before_matrix_construction": True,
    }


def build(source_path: Path, output_dir: Path) -> dict:
    source_path = source_path.resolve()
    output_dir = output_dir.resolve()
    gltf, chunks = _json_and_chunks(source_path)
    asset = dict(gltf.get("asset") or {})
    if not str(asset.get("version") or "").startswith("2"):
        raise ValueError("GLTF_RIG_SKIN_V2_GLTF2_REQUIRED")
    nodes = list(gltf.get("nodes") or ())
    skins = list(gltf.get("skins") or ())
    if not nodes or not skins:
        raise ValueError("GLTF_RIG_SKIN_NODES_OR_SKINS_MISSING")

    buffers = v1._buffer_bytes(gltf, source_path, chunks)
    parents = v1._parents(nodes)
    world = _world_matrices(nodes, parents)
    rotation_diagnostics = _rotation_diagnostics(nodes)

    skin_rows = []
    joint_local_matrices = []
    joint_world_matrices = []
    inverse_bind_matrices = []
    skin_joint_offsets = [0]
    joint_node_indices = []
    parent_joint_slots = []
    for skin_index, skin in enumerate(skins):
        joints = list(map(int, skin.get("joints") or ()))
        if not joints or len(set(joints)) != len(joints):
            raise ValueError("GLTF_RIG_SKIN_JOINT_SET")
        joint_lut = {node_index: slot for slot, node_index in enumerate(joints)}
        inverse_bind = v1._inverse_bind_matrices(gltf, buffers, skin)
        names = []
        local_parent = []
        for node_index in joints:
            node = nodes[node_index]
            name = str(node.get("name") or f"node_{node_index}")
            names.append(name)
            source_parent = parents.get(node_index)
            local_parent.append(joint_lut.get(source_parent, -1))
            joint_node_indices.append(node_index)
            parent_joint_slots.append(joint_lut.get(source_parent, -1))
            joint_local_matrices.append(_local_matrix(node))
            joint_world_matrices.append(world[node_index])
        inverse_bind_matrices.extend(inverse_bind)
        skin_rows.append(
            {
                "skin_index": skin_index,
                "name": skin.get("name"),
                "joint_count": len(joints),
                "joint_node_indices": joints,
                "joint_names": names,
                "parent_joint_slots": local_parent,
                "inverse_bind_present": skin.get("inverseBindMatrices") is not None,
                "skeleton_node": skin.get("skeleton"),
            }
        )
        skin_joint_offsets.append(skin_joint_offsets[-1] + len(joints))

    primitive_rows, arrays = v1._primitive_streams(gltf, buffers)
    vertex_skin_index = np.empty(len(arrays["source_position_local"]), dtype=np.int32)
    for row_index, row in enumerate(primitive_rows):
        start = int(arrays["primitive_vertex_offsets"][row_index])
        end = int(arrays["primitive_vertex_offsets"][row_index + 1])
        vertex_skin_index[start:end] = int(row["skin_index"])
    arrays.update(
        {
            "vertex_skin_index": vertex_skin_index,
            "skin_joint_offsets": np.asarray(skin_joint_offsets, dtype=np.int64),
            "joint_node_index": np.asarray(joint_node_indices, dtype=np.int32),
            "parent_joint_slot": np.asarray(parent_joint_slots, dtype=np.int32),
            "joint_local_matrix": np.asarray(joint_local_matrices, dtype=np.float64),
            "joint_world_matrix": np.asarray(joint_world_matrices, dtype=np.float64),
            "inverse_bind_matrix": np.asarray(inverse_bind_matrices, dtype=np.float64),
        }
    )

    weights = arrays["source_weight_value"]
    joints = arrays["source_joint_slot"]
    if np.any(weights < 0) or not np.isfinite(weights).all():
        raise ValueError("GLTF_RIG_SKIN_WEIGHT_VALUE")
    weight_sum = weights.sum(axis=1)
    if np.max(np.abs(weight_sum - 1.0)) > 2.0e-5:
        raise ValueError("GLTF_RIG_SKIN_WEIGHT_SUM")
    for row_index, row in enumerate(primitive_rows):
        start = int(arrays["primitive_vertex_offsets"][row_index])
        end = int(arrays["primitive_vertex_offsets"][row_index + 1])
        joint_count = int(skin_rows[int(row["skin_index"])]["joint_count"])
        active = weights[start:end] > 0
        if np.any(joints[start:end][active] < 0) or np.any(joints[start:end][active] >= joint_count):
            raise ValueError("GLTF_RIG_SKIN_JOINT_INDEX_RANGE")

    output_dir.mkdir(parents=True, exist_ok=True)
    arrays_path = output_dir / "GLTF_RIG_SKIN_AUTHORITY_V2.npz"
    v1._write_npz(arrays_path, arrays)
    manifest = {
        "schema": SCHEMA,
        "status": STATUS,
        "source": {"path": str(source_path), "sha256": v1.sha256(source_path)},
        "source_format": "GLTF_2_0",
        "source_container": "GLB" if source_path.suffix.lower() == ".glb" else "GLTF_JSON",
        "skins": skin_rows,
        "primitives": primitive_rows,
        "counts": {
            "skin_count": len(skin_rows),
            "joint_count_total": int(skin_joint_offsets[-1]),
            "skinned_primitive_count": len(primitive_rows),
            "render_vertex_count": int(len(arrays["source_position_local"])),
            "triangle_count": int(len(arrays["triangle_index_stream"]) // 3),
            "nonzero_influence_count": int(np.count_nonzero(weights > 0)),
            "maximum_influences_per_vertex": int(np.max(np.count_nonzero(weights > 0, axis=1))),
        },
        "weight_quality": {
            "all_vertices_have_positive_weight": bool(np.all(np.any(weights > 0, axis=1))),
            "maximum_weight_sum_error": float(np.max(np.abs(weight_sum - 1.0))),
        },
        "rotation_semantics": rotation_diagnostics,
        "outputs": {"arrays": {"path": str(arrays_path), "sha256": v1.sha256(arrays_path)}},
        "policy": {
            "joint_identity": "SKIN_INDEX_PLUS_SOURCE_JOINT_SLOT_PLUS_SOURCE_NODE_INDEX",
            "render_vertex_order": "SKINNED_MESH_INSTANCE_SORTED_BY_MESH_INDEX_THEN_NODE_INDEX__PRIMITIVE_SOURCE_ORDER__ACCESSOR_ROW_ORDER",
            "rotation_quaternion_policy": "FINITE_NONZERO_SOURCE_QUATERNION_NORMALIZED_TO_UNIT_BEFORE_MATRIX_CONSTRUCTION",
            "weight_zero_columns_preserved": True,
            "source_coordinates_preserved": True,
            "source_mutated": False,
            "importer_equivalence_required": True,
        },
        "authority": {
            "raw_source_candidate_only": True,
            "independent_source_court_required": True,
            "importer_equivalence_court_required": True,
            "family_authority_minted": False,
            "training_authorized": False,
        },
    }
    manifest["rig_skin_identity_sha256"] = v1.digest(manifest)
    manifest_path = output_dir / "GLTF_RIG_SKIN_AUTHORITY_MANIFEST_V2.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "status": STATUS,
                "joint_count_total": manifest["counts"]["joint_count_total"],
                "render_vertex_count": manifest["counts"]["render_vertex_count"],
                "nonzero_influence_count": manifest["counts"]["nonzero_influence_count"],
                "maximum_source_quaternion_norm_abs_error": rotation_diagnostics["maximum_source_quaternion_norm_abs_error"],
                "rig_skin_identity_sha256": manifest["rig_skin_identity_sha256"],
            },
            sort_keys=True,
        )
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    build(args.source, args.output_dir)


if __name__ == "__main__":
    main()
