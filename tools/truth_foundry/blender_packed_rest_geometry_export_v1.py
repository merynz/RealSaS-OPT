#!/usr/bin/env python3
from __future__ import annotations

"""Export evaluated geometry from an exact packed Truth Foundry source scene.

Diagnostic only.  The mesh selection deliberately reuses the native teacher
exporter's renderable-source selection so armature custom-shape helper meshes do
not re-enter a geometry comparison that the sealed teacher never contained.
This script does not alter materials, choose appearance semantics, qualify
geometry, or mint source equivalence.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import bpy

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.iris_vnext import blender_authored_teacher_export_v1 as source_transport

SCHEMA = "RealSaS.PackedRestGeometryDiagnostic.v1"


def _args() -> argparse.Namespace:
    raw = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--subject-id", required=True)
    p.add_argument("--expected-blend-sha256", required=True)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args(raw)


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    a = _args()
    blend = Path(bpy.data.filepath).resolve()
    if not blend.is_file() or _sha(blend) != a.expected_blend_sha256:
        raise RuntimeError("PACKED_REST_GEOMETRY_BLEND_HASH_DRIFT")

    meshes = source_transport._mesh_objects()
    deps = bpy.context.evaluated_depsgraph_get()
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    objects: list[dict] = []
    for obj in meshes:
        evaluated = obj.evaluated_get(deps)
        mesh = evaluated.to_mesh()
        try:
            offset = len(vertices)
            vertices.extend([list(evaluated.matrix_world @ vertex.co) for vertex in mesh.vertices])
            mesh.calc_loop_triangles()
            faces.extend([[offset + int(index) for index in tri.vertices] for tri in mesh.loop_triangles])
            objects.append({
                "name": obj.name,
                "vertex_offset": offset,
                "vertex_count": len(mesh.vertices),
                "triangle_count": len(mesh.loop_triangles),
            })
        finally:
            evaluated.to_mesh_clear()

    if not vertices or not faces:
        raise RuntimeError("PACKED_REST_GEOMETRY_EMPTY")
    receipt = {
        "schema": SCHEMA,
        "subject_id": a.subject_id,
        "packed_source_scene_blend_sha256": a.expected_blend_sha256,
        "blender_version": str(bpy.app.version_string),
        "coordinate_frame": "PACKED_SOURCE_SCENE_WORLD",
        "mesh_selection": "NATIVE_TEACHER_RENDERABLE_SOURCE_MESHES__CUSTOM_SHAPE_HELPERS_EXCLUDED",
        "vertices": vertices,
        "faces": faces,
        "objects": objects,
        "authority": "DIAGNOSTIC_ONLY__NO_GEOMETRY_OR_APPEARANCE_QUALIFICATION",
        "qualification_minted": False,
        "training_allowed": False,
        "product_pass_claimed": False,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    print(json.dumps({
        "subject_id": a.subject_id,
        "vertex_count": len(vertices),
        "triangle_count": len(faces),
        "object_count": len(objects),
        "mesh_selection": receipt["mesh_selection"],
        "authority": receipt["authority"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
