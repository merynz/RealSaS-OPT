#!/usr/bin/env python3
from __future__ import annotations

"""Build raw-source rig/skin authority candidate from glTF 2.0 JSON or GLB.

The producer reads only immutable source bytes. It does not consume Blender,
TeacherSceneIR, Stage47, AXIS, MIRA or any imported representation. The output
therefore preserves source-native joint hierarchy, inverse binds and exact
JOINTS_0/WEIGHTS_0 rows before a separate importer-equivalence court.
"""

import argparse
import base64
import hashlib
import json
from pathlib import Path
from urllib.parse import unquote_to_bytes

import numpy as np

from tools.observation.source_scene_contract_v2 import _json_and_chunks


SCHEMA = "RealSaS.GltfRigSkinAuthority.v1"
STATUS = "PASS_GLTF_RIG_SKIN_SOURCE_AUTHORITY_CANDIDATE__INDEPENDENT_COURT_REQUIRED"

_COMPONENT_DTYPE = {
    5120: np.int8,
    5121: np.uint8,
    5122: np.int16,
    5123: np.uint16,
    5125: np.uint32,
    5126: np.float32,
}
_TYPE_COMPONENTS = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT2": 4,
    "MAT3": 9,
    "MAT4": 16,
}


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


def _buffer_bytes(gltf: dict, source_path: Path, chunks=()) -> list[bytes]:
    binary = [data for kind, data in chunks if kind == 0x004E4942]
    result = []
    for index, row in enumerate(gltf.get("buffers") or ()):
        uri = row.get("uri")
        if uri is None:
            if not binary:
                raise ValueError(f"GLTF_RIG_SKIN_EMBEDDED_BUFFER_REQUIRED:{index}")
            data = binary.pop(0)
        else:
            uri = str(uri)
            if uri.startswith("data:"):
                header, payload = uri.split(",", 1)
                data = base64.b64decode(payload) if ";base64" in header else unquote_to_bytes(payload)
            else:
                external = (source_path.parent / uri).resolve()
                if not external.is_relative_to(source_path.parent.resolve()):
                    raise ValueError("GLTF_RIG_SKIN_EXTERNAL_BUFFER_ESCAPE:" + str(external))
                if not external.is_file():
                    raise FileNotFoundError("GLTF_RIG_SKIN_EXTERNAL_BUFFER:" + str(external))
                data = external.read_bytes()
        expected = int(row.get("byteLength", -1))
        if expected < 0 or len(data) < expected:
            raise ValueError(f"GLTF_RIG_SKIN_BUFFER_LENGTH:{index}")
        result.append(data[:expected])
    if binary:
        raise ValueError("GLTF_RIG_SKIN_UNUSED_GLB_BINARY_CHUNK")
    return result


def _accessor(gltf: dict, buffers: list[bytes], accessor_index: int) -> np.ndarray:
    accessor = dict(gltf["accessors"][int(accessor_index)])
    if "sparse" in accessor:
        raise ValueError("GLTF_RIG_SKIN_SPARSE_ACCESSOR_UNSUPPORTED")
    view = dict(gltf["bufferViews"][int(accessor["bufferView"])])
    component_type = int(accessor["componentType"])
    dtype = np.dtype(_COMPONENT_DTYPE[component_type])
    count = int(accessor["count"])
    components = _TYPE_COMPONENTS[str(accessor["type"])]
    offset = int(view.get("byteOffset", 0)) + int(accessor.get("byteOffset", 0))
    stride = int(view.get("byteStride", dtype.itemsize * components))
    raw = buffers[int(view["buffer"])]
    item_bytes = dtype.itemsize * components
    if stride < item_bytes:
        raise ValueError("GLTF_RIG_SKIN_ACCESSOR_STRIDE")
    if stride == item_bytes:
        array = np.frombuffer(raw, dtype=dtype, count=count * components, offset=offset).reshape(count, components)
    else:
        array = np.empty((count, components), dtype=dtype)
        for row in range(count):
            array[row] = np.frombuffer(raw, dtype=dtype, count=components, offset=offset + row * stride)
    array = np.array(array)
    if accessor.get("normalized"):
        if np.issubdtype(dtype, np.unsignedinteger):
            array = array.astype(np.float64) / np.iinfo(dtype).max
        elif np.issubdtype(dtype, np.signedinteger):
            array = np.maximum(array.astype(np.float64) / np.iinfo(dtype).max, -1.0)
    if str(accessor["type"]) == "SCALAR":
        array = array[:, 0]
    return array


def _quat_matrix(value) -> np.ndarray:
    x, y, z, w = map(float, value)
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (y * z + w * x), 0],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x), 0],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y), 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float64,
    )


def _local_matrix(node: dict) -> np.ndarray:
    if "matrix" in node:
        value = np.asarray(node["matrix"], dtype=np.float64)
        if value.shape != (16,):
            raise ValueError("GLTF_RIG_SKIN_NODE_MATRIX")
        return value.reshape(4, 4).T
    translation = np.eye(4, dtype=np.float64)
    translation[:3, 3] = np.asarray(node.get("translation", [0, 0, 0]), dtype=np.float64)
    rotation = _quat_matrix(node.get("rotation", [0, 0, 0, 1]))
    scale = np.eye(4, dtype=np.float64)
    scale_value = np.asarray(node.get("scale", [1, 1, 1]), dtype=np.float64)
    scale[0, 0], scale[1, 1], scale[2, 2] = scale_value
    return translation @ rotation @ scale


def _parents(nodes: list[dict]) -> dict[int, int]:
    result = {}
    for parent, node in enumerate(nodes):
        for child in node.get("children") or ():
            child = int(child)
            if child in result:
                raise ValueError("GLTF_RIG_SKIN_MULTI_PARENT_NODE")
            result[child] = parent
    return result


def _world_matrices(nodes: list[dict], parents: dict[int, int]) -> list[np.ndarray]:
    cache: dict[int, np.ndarray] = {}
    visiting = set()

    def solve(index: int) -> np.ndarray:
        if index in cache:
            return cache[index]
        if index in visiting:
            raise ValueError("GLTF_RIG_SKIN_NODE_CYCLE")
        visiting.add(index)
        local = _local_matrix(nodes[index])
        result = solve(parents[index]) @ local if index in parents else local
        visiting.remove(index)
        cache[index] = result
        return result

    return [solve(index) for index in range(len(nodes))]


def _inverse_bind_matrices(gltf: dict, buffers: list[bytes], skin: dict) -> np.ndarray:
    joint_count = len(skin.get("joints") or ())
    accessor_index = skin.get("inverseBindMatrices")
    if accessor_index is None:
        return np.repeat(np.eye(4, dtype=np.float64)[None], joint_count, axis=0)
    raw = _accessor(gltf, buffers, int(accessor_index))
    if raw.shape != (joint_count, 16):
        raise ValueError("GLTF_RIG_SKIN_INVERSE_BIND_LAYOUT")
    return raw.reshape(joint_count, 4, 4).transpose(0, 2, 1).astype(np.float64)


def _primitive_streams(gltf: dict, buffers: list[bytes]) -> tuple[list[dict], dict[str, np.ndarray]]:
    nodes = list(gltf.get("nodes") or ())
    mesh_instances = []
    for node_index, node in enumerate(nodes):
        if "mesh" in node and "skin" in node:
            mesh_instances.append((node_index, int(node["mesh"]), int(node["skin"])))
    mesh_instances.sort(key=lambda row: (row[1], row[0], row[2]))

    positions = []
    joints = []
    weights = []
    primitive_vertex_offsets = [0]
    primitive_index_offsets = [0]
    index_stream = []
    records = []
    for node_index, mesh_index, skin_index in mesh_instances:
        mesh = gltf["meshes"][mesh_index]
        for primitive_index, primitive in enumerate(mesh.get("primitives") or ()):
            attrs = dict(primitive.get("attributes") or {})
            required = {"POSITION", "JOINTS_0", "WEIGHTS_0"}
            if not required.issubset(attrs):
                raise ValueError("GLTF_RIG_SKIN_PRIMITIVE_ATTRIBUTES")
            p = np.asarray(_accessor(gltf, buffers, attrs["POSITION"]), dtype=np.float64)
            j = np.asarray(_accessor(gltf, buffers, attrs["JOINTS_0"]), dtype=np.int64)
            w = np.asarray(_accessor(gltf, buffers, attrs["WEIGHTS_0"]), dtype=np.float64)
            if p.ndim != 2 or p.shape[1:] != (3,) or j.shape != (len(p), 4) or w.shape != (len(p), 4):
                raise ValueError("GLTF_RIG_SKIN_PRIMITIVE_LAYOUT")
            if primitive.get("mode", 4) != 4:
                raise ValueError("GLTF_RIG_SKIN_TRIANGLES_REQUIRED")
            if primitive.get("indices") is None:
                local_indices = np.arange(len(p), dtype=np.int64)
            else:
                local_indices = np.asarray(_accessor(gltf, buffers, primitive["indices"]), dtype=np.int64)
            if local_indices.ndim != 1 or len(local_indices) % 3 or np.any(local_indices < 0) or np.any(local_indices >= len(p)):
                raise ValueError("GLTF_RIG_SKIN_INDEX_LAYOUT")
            vertex_start = primitive_vertex_offsets[-1]
            positions.append(p)
            joints.append(j)
            weights.append(w)
            index_stream.append(local_indices + vertex_start)
            primitive_vertex_offsets.append(vertex_start + len(p))
            primitive_index_offsets.append(primitive_index_offsets[-1] + len(local_indices))
            records.append(
                {
                    "node_index": node_index,
                    "mesh_index": mesh_index,
                    "primitive_index": primitive_index,
                    "skin_index": skin_index,
                    "vertex_count": int(len(p)),
                    "triangle_count": int(len(local_indices) // 3),
                    "material_index": primitive.get("material"),
                }
            )
    if not positions:
        raise ValueError("GLTF_RIG_SKIN_NO_SKINNED_PRIMITIVES")
    return records, {
        "source_position_local": np.concatenate(positions).astype(np.float64),
        "source_joint_slot": np.concatenate(joints).astype(np.int32),
        "source_weight_value": np.concatenate(weights).astype(np.float64),
        "primitive_vertex_offsets": np.asarray(primitive_vertex_offsets, dtype=np.int64),
        "primitive_index_offsets": np.asarray(primitive_index_offsets, dtype=np.int64),
        "triangle_index_stream": np.concatenate(index_stream).astype(np.int64),
    }


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    for name, value in arrays.items():
        value = np.asarray(value)
        if value.dtype == object or (np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all()):
            raise ValueError("GLTF_RIG_SKIN_INVALID_ARRAY:" + name)
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_suffix(path.suffix + ".partial.npz")
    np.savez_compressed(staged, **arrays)
    staged.replace(path)


def build(source_path: Path, output_dir: Path) -> dict:
    source_path = source_path.resolve()
    output_dir = output_dir.resolve()
    gltf, chunks = _json_and_chunks(source_path)
    asset = dict(gltf.get("asset") or {})
    if not str(asset.get("version") or "").startswith("2"):
        raise ValueError("GLTF_RIG_SKIN_GLTF2_REQUIRED")
    nodes = list(gltf.get("nodes") or ())
    skins = list(gltf.get("skins") or ())
    if not nodes or not skins:
        raise ValueError("GLTF_RIG_SKIN_NODES_OR_SKINS_MISSING")
    buffers = _buffer_bytes(gltf, source_path, chunks)
    parents = _parents(nodes)
    world = _world_matrices(nodes, parents)

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
        inverse_bind = _inverse_bind_matrices(gltf, buffers, skin)
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

    primitive_rows, arrays = _primitive_streams(gltf, buffers)
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
    arrays_path = output_dir / "GLTF_RIG_SKIN_AUTHORITY_V1.npz"
    _write_npz(arrays_path, arrays)
    manifest = {
        "schema": SCHEMA,
        "status": STATUS,
        "source": {"path": str(source_path), "sha256": sha256(source_path)},
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
        "outputs": {"arrays": {"path": str(arrays_path), "sha256": sha256(arrays_path)}},
        "policy": {
            "joint_identity": "SKIN_INDEX_PLUS_SOURCE_JOINT_SLOT_PLUS_SOURCE_NODE_INDEX",
            "render_vertex_order": "SKINNED_MESH_INSTANCE_SORTED_BY_MESH_INDEX_THEN_NODE_INDEX__PRIMITIVE_SOURCE_ORDER__ACCESSOR_ROW_ORDER",
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
    manifest["rig_skin_identity_sha256"] = digest(manifest)
    manifest_path = output_dir / "GLTF_RIG_SKIN_AUTHORITY_MANIFEST_V1.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(json.dumps({
        "status": STATUS,
        "joint_count_total": manifest["counts"]["joint_count_total"],
        "render_vertex_count": manifest["counts"]["render_vertex_count"],
        "nonzero_influence_count": manifest["counts"]["nonzero_influence_count"],
        "rig_skin_identity_sha256": manifest["rig_skin_identity_sha256"],
    }, sort_keys=True))
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    build(args.source, args.output_dir)


if __name__ == "__main__":
    main()
