"""Generic, source-scene-aware glTF UV triangle inventory for independent verification.

Pure Python. No authored object/part names and no drawable owner inference.
The atlas coordinate frame is glTF TOP-LEFT, not Blender bottom-left.
"""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
import math
import numpy as np

from tools.corpus.compile_corpus370_original_source_uv_edit_plan_v2 import (
    glb_document, accessor,
)

SCHEMA = "RealSaS.SourceUVTriangleInventory.v0"
QUANTIZATION = 100000  # 1e-5 normalized UV, <= 0.00512 source texels / 512 texture


def _require(ok, reason):
    if not ok:
        raise ValueError("SOURCE_UV_INVENTORY_" + reason)


def uv_triangle_key(corners):
    a = np.asarray(corners, dtype=np.float64)
    _require(a.shape == (3, 2) and np.isfinite(a).all(),
             "TRIANGLE_UV_INVALID")
    _require(np.all(a >= -1e-7) and np.all(a <= 1 + 1e-7),
             "TRIANGLE_UV_DOMAIN")
    return tuple(sorted(tuple(int(np.rint(float(v) * QUANTIZATION))
                              for v in corner) for corner in a))


def _scene_mesh_instance_counts(document):
    nodes = document.get("nodes", [])
    scenes = document.get("scenes", [])
    _require(bool(scenes), "ACTIVE_SCENE_MISSING")
    scene = document.get("scene", 0)
    _require(type(scene) is int and 0 <= scene < len(scenes),
             "ACTIVE_SCENE_INVALID")
    counts = Counter()
    visited = set()
    active = set()

    def walk(index):
        _require(type(index) is int and 0 <= index < len(nodes),
                 "NODE_INVALID")
        _require(index not in active, "CYCLIC_NODE_GRAPH")
        _require(index not in visited, "NODE_SHARED_PARENT_INVALID")
        visited.add(index)
        active.add(index)
        n = nodes[index]
        if "mesh" in n:
            mesh = n["mesh"]
            _require(type(mesh) is int and 0 <= mesh < len(document["meshes"]),
                     "NODE_MESH_INVALID")
            counts[mesh] += 1
        for child in n.get("children", []):
            walk(child)
        active.remove(index)
    for root in scenes[scene].get("nodes", []):
        walk(root)
    _require(bool(counts), "NO_ACTIVE_RENDER_MESH")
    return counts


def original_gltf_uv_inventory(glb):
    doc, binary = glb_document(glb)
    counts = _scene_mesh_instance_counts(doc)
    triangles = Counter()
    primitive_rows = []
    for mesh_index, copies in sorted(counts.items()):
        mesh = doc["meshes"][mesh_index]
        for primitive_index, p in enumerate(mesh.get("primitives", [])):
            att = p.get("attributes", {})
            _require(p.get("mode", 4) == 4 and "indices" in p
                     and "TEXCOORD_0" in att and "POSITION" in att,
                     "SOURCE_TRIANGLE_ATTRIBUTE_MISSING")
            raw_uv = accessor(doc, binary, att["TEXCOORD_0"])
            raw_pos = accessor(doc, binary, att["POSITION"])
            raw_i = accessor(doc, binary, p["indices"]).reshape(-1)
            _require(raw_i.size % 3 == 0 and
                     len(raw_pos) == len(raw_uv) and
                     raw_i.size > 0 and
                     np.issubdtype(raw_i.dtype, np.integer),
                     "RAW_PRIMITIVE_TOPOLOGY_INVALID")
            _require(np.all(raw_i >= 0) and np.max(raw_i) < len(raw_uv),
                     "RAW_PRIMITIVE_INDEX_OOB")
            f = raw_i.reshape(-1, 3)
            for face in f:
                triangles[uv_triangle_key(raw_uv[face])] += copies
            primitive_rows.append({
                "anonymous_mesh_index": mesh_index,
                "anonymous_primitive_index": primitive_index,
                "scene_instance_count": copies,
                "triangle_count_per_instance": int(len(f)),
            })
    _require(bool(triangles), "ORIGINAL_SOURCE_EMPTY_UV_TRIANGLES")
    return triangles, primitive_rows


def compare_uv_inventories(source: Counter, imported_uv) -> dict:
    # A multiset comparison is independent of Blender mesh-object/face order.
    # Face-to-TEACHER identity is checked separately against the exact v4
    # frozen triangulation and source-world rest snapshot.
    a = np.asarray(imported_uv, dtype=np.float64)
    _require(a.ndim == 3 and a.shape[1:] == (3, 2),
             "IMPORTED_CORNER_UV_SHAPE")
    actual = Counter(uv_triangle_key(face) for face in a)
    delta = source - actual
    excess = actual - source
    return {
        "original_triangle_count": int(sum(source.values())),
        "imported_triangle_count": int(len(a)),
        "missing_gltf_triangle_uv_incidents": int(sum(delta.values())),
        "extra_imported_triangle_uv_incidents": int(sum(excess.values())),
        "matched": not delta and not excess,
        "uv_quantization_grid": QUANTIZATION,
        "quantization_coordinate_origin": "GLTF_IMAGE_TOP_LEFT",
    }
