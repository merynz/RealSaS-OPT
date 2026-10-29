"""Export evaluated imported rest geometry for an independent raw-source court.

When requested, the same animation-free Blender import also emits a response-independent triangulated
surface package for canonical Graphics.  This package contains only rest geometry plus privileged
face-owner/material/UV addressing.  It contains no kinematic response, rig/skin authority, semantic
part labels, attachment labels or training authority.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import bpy
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.observation.blender_controlled_8view_v2 import import_rest, matrix_rows


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def arguments():
    raw = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--surface-geometry-output", type=Path)
    parser.add_argument("--surface-geometry-manifest", type=Path)
    args = parser.parse_args(raw)
    if (args.surface_geometry_output is None) != (args.surface_geometry_manifest is None):
        raise ValueError("PARITY_SURFACE_GEOMETRY_OUTPUTS_MUST_BE_PAIRED")
    return args


def main() -> None:
    args = arguments()
    source = args.source.resolve()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    receipt = import_rest(source, output.parent)
    deps = bpy.context.evaluated_depsgraph_get()
    vertices, faces, objects = [], [], []
    face_owner, face_material, triangle_uv, triangle_uv_valid = [], [], [], []
    mesh_owner = 0
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH" or obj.hide_render:
            continue
        evaluated = obj.evaluated_get(deps)
        mesh = evaluated.to_mesh(preserve_all_data_layers=True, depsgraph=deps)
        try:
            offset = len(vertices)
            vertices.extend([list(evaluated.matrix_world @ v.co) for v in mesh.vertices])
            mesh.calc_loop_triangles()
            uv_layer = mesh.uv_layers.active
            for tri in mesh.loop_triangles:
                faces.append([offset + int(i) for i in tri.vertices])
                face_owner.append(mesh_owner)
                face_material.append(int(mesh.polygons[tri.polygon_index].material_index))
                if uv_layer is None:
                    triangle_uv.append([[float("nan"), float("nan")]] * 3)
                    triangle_uv_valid.append(False)
                else:
                    uv = [[float(x) for x in uv_layer.data[loop_index].uv] for loop_index in tri.loops]
                    triangle_uv.append(uv)
                    triangle_uv_valid.append(bool(np.isfinite(uv).all()))
            objects.append({
                "name": obj.name,
                "vertex_count": len(mesh.vertices),
                "triangle_count": len(mesh.loop_triangles),
                "privileged_face_owner": mesh_owner,
                "parent": obj.parent.name if obj.parent else None,
                "parent_bone": obj.parent_bone,
                "world": matrix_rows(obj.matrix_world),
                "weights": [[{"group": obj.vertex_groups[g.group].name, "weight": g.weight}
                             for g in v.groups] for v in obj.data.vertices],
            })
            mesh_owner += 1
        finally:
            evaluated.to_mesh_clear()
    bones = sum(len(obj.data.bones) for obj in bpy.context.scene.objects if obj.type == "ARMATURE")
    imported = {
        "vertices": vertices,
        "faces": faces,
        "bone_count": bones,
        "objects": objects,
        "import": receipt,
    }
    output.write_text(json.dumps(imported))

    if args.surface_geometry_output is not None:
        geometry_output = args.surface_geometry_output.resolve()
        geometry_manifest = args.surface_geometry_manifest.resolve()
        geometry_output.parent.mkdir(parents=True, exist_ok=True)
        geometry_manifest.parent.mkdir(parents=True, exist_ok=True)
        v = np.asarray(vertices, dtype=np.float64)
        f = np.asarray(faces, dtype=np.int32)
        owners = np.asarray(face_owner, dtype=np.int32)
        materials = np.asarray(face_material, dtype=np.int32)
        uv = np.asarray(triangle_uv, dtype=np.float64)
        uv_valid = np.asarray(triangle_uv_valid, dtype=np.bool_)
        if v.ndim != 2 or v.shape[1:] != (3,) or f.ndim != 2 or f.shape[1:] != (3,) or len(v) == 0 or len(f) == 0:
            raise ValueError("PARITY_SURFACE_GEOMETRY_LAYOUT")
        if owners.shape != (len(f),) or materials.shape != (len(f),) or uv.shape != (len(f), 3, 2) or uv_valid.shape != (len(f),):
            raise ValueError("PARITY_SURFACE_METADATA_LAYOUT")
        staged = geometry_output.with_suffix(geometry_output.suffix + ".partial.npz")
        np.savez_compressed(
            staged,
            rest_vertices_source_world=v,
            faces=f,
            face_owner_privileged=owners,
            face_material_privileged=materials,
            triangle_uv=uv,
            triangle_uv_valid=uv_valid,
        )
        staged.replace(geometry_output)
        manifest = {
            "schema": "RealSaS.ResponseIndependentRestSurfaceGeometry.v1",
            "status": "PASS_IMPORTED_REST_SURFACE_GEOMETRY__RAW_PARITY_ADMISSION_SEPARATE",
            "source_sha256": receipt["source_sha256"],
            "source_path": str(source),
            "imported_geometry_json": {"path": str(output), "sha256": sha256(output)},
            "geometry_npz": {"path": str(geometry_output), "sha256": sha256(geometry_output)},
            "counts": {
                "vertex_count": int(len(v)),
                "triangle_count": int(len(f)),
                "privileged_face_owner_count": int(mesh_owner),
                "uv_valid_triangle_count": int(np.count_nonzero(uv_valid)),
            },
            "scope": {
                "rest_geometry": True,
                "privileged_face_owner_material_and_uv": True,
                "kinematic_response_measured": False,
                "rig_skin_authority_minted": False,
                "semantic_part_taxonomy_used": False,
                "training_authorized": False,
                "qualification_minted": False,
            },
        }
        staged_manifest = geometry_manifest.with_suffix(geometry_manifest.suffix + ".partial")
        staged_manifest.write_text(json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n")
        staged_manifest.replace(geometry_manifest)
        print(
            "[REST_SURFACE]",
            len(v),
            len(f),
            "geometry_sha256=" + manifest["geometry_npz"]["sha256"],
            flush=True,
        )
    print("[PARITY_EXPORT]", len(vertices), bones, flush=True)


if __name__ == "__main__":
    main()
