#!/usr/bin/env python3
"""Three-way original-byte visual diagnostic; never source RGB or training authority.

SOURCE_MATERIAL_DIAGNOSTIC and two Workbench stylizations use identical
source-space camera, rest, geometry and resolution. Workbench does not promise
texture parity and its output must not overwrite original source reference.
The source-space camera is NOT an anatomical front/up truth.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

# Blender bundled Python may ignore PYTHONPATH; bind package root explicitly.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.corpus.blender_corpus4008_native_source_probe_v1 import (
    import_source, finite_matrix, require, sha,
)

SCHEMA = "RealSaS.Corpus4008LookdevStudy.v1"
VIEWS = ("SOURCE_MATERIAL_DIAGNOSTIC",
         "STYLIZED_FLAT_DIAGNOSTIC", "STYLIZED_OUTLINE_DIAGNOSTIC")


def bound_world_points(bpy):
    from mathutils import Vector
    points = []
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        for corner in obj.bound_box:
            xyz = obj.matrix_world @ Vector(corner)
            require(all(math.isfinite(float(x)) for x in xyz),
                    "NONFINITE_WORLD_BOUND")
            points.append(xyz)
    require(points, "NO_MESH_BOUNDS")
    lo = Vector(tuple(min(v[i] for v in points) for i in range(3)))
    hi = Vector(tuple(max(v[i] for v in points) for i in range(3)))
    return (lo + hi) / 2, max((hi - lo).length / 2, 1e-4)


def diagnostic_camera(bpy, center, radius):
    from mathutils import Vector
    bpy.ops.object.camera_add()
    camera = bpy.context.object
    camera.name = "DIAGNOSTIC_SOURCE_SPACE_CAMERA__NOT_ANATOMICAL"
    # Deliberate source-space fixed diagnostic axis, with NO front/up claim.
    camera.location = center + Vector((1.8, -3.4, 1.3)).normalized() * max(
        radius * 4.5, 1.0)
    camera.rotation_euler = (center - camera.location).to_track_quat(
        "-Z", "Y").to_euler()
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = max(4 * radius, 0.5)
    bpy.context.scene.camera = camera
    return camera


def configure_scene(bpy, resolution):
    scene = bpy.context.scene
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = True
    scene.camera.data.lens = 50
    scene.view_settings.view_transform = "Standard"
    return scene


def configure_source_material_diagnostic(bpy, center):
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.render.image_settings.color_mode = "RGBA"
    # Lighting is not source-native lighting and does not claim RGB fidelity.
    from mathutils import Vector
    cam = scene.camera
    for name, shift, energy in (
            ("DIAGNOSTIC_KEY", (-0.7, 0.2, 0.5), 1300),
            ("DIAGNOSTIC_FILL", (0.7, -0.2, 0.2), 800)):
        bpy.ops.object.light_add(type="AREA")
        light = bpy.context.object
        light.name = name
        light.location = cam.location + Vector(shift)
        light.data.energy = energy
        light.data.shape = "DISK"
        light.data.size = 5
        light.rotation_euler = (
            center - light.location).to_track_quat(
            "-Z", "Y").to_euler()


def configure_flat_study(bpy, outline):
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_WORKBENCH"
    settings = scene.display.shading
    settings.light = "STUDIO"
    settings.color_type = "MATERIAL"
    settings.show_shadows = True
    settings.show_cavity = True
    settings.cavity_type = "BOTH"
    settings.show_object_outline = bool(outline)


def render_study(bpy, *, source, expected_sha256, asset_id,
                 candidate_id, output, resolution):
    require(source.is_file(), "SOURCE_NOT_FOUND")
    require(sha(source) == expected_sha256, "EXACT_ORIGINAL_SHA")
    require(candidate_id.endswith(expected_sha256) or
            candidate_id.endswith(expected_sha256[:24]),
            "SOURCE_CANDIDATE_SHA_DRIFT")
    require(256 <= resolution <= 1024, "RENDER_RESOLUTION_LIMIT")
    import_source(bpy, source, load_anim=False)
    meshes = sorted(o.name for o in bpy.data.objects if o.type == "MESH")
    center, radius = bound_world_points(bpy)
    camera = diagnostic_camera(bpy, center, radius)
    scene = configure_scene(bpy, resolution)
    output.mkdir(parents=True, exist_ok=True)
    outputs = []
    for mode in VIEWS:
        if mode == VIEWS[0]:
            configure_source_material_diagnostic(bpy, center)
        else:
            configure_flat_study(bpy, outline=(mode == VIEWS[2]))
        filename = mode.lower() + ".png"
        path = output / filename
        scene.render.filepath = str(path)
        bpy.ops.render.render(write_still=True)
        require(path.is_file() and path.stat().st_size > 0,
                "STUDY_RENDER_MISSING")
        outputs.append({
            "mode": mode,
            "path_private": filename,
            "sha256": sha(path),
            "byte_count": path.stat().st_size,
            "render_approved_as_source_rgb": False,
            "render_approved_as_model_target": False,
        })
    require(sorted(o.name for o in bpy.data.objects if o.type == "MESH") == meshes,
            "STUDY_SOURCE_MESH_IDENTITY_DRIFT")
    result = {
        "schema": SCHEMA,
        "status": "LOOKDEV_DIAGNOSTIC_ONLY",
        "asset_id": asset_id, "candidate_id": candidate_id,
        "source_sha256": expected_sha256,
        "importer_version": bpy.app.version_string,
        "camera_label": "SOURCE_SPACE_FIXED_UNQUALIFIED_DIAGNOSTIC",
        "source_anatomical_front": "UNKNOWN",
        "source_anatomical_up": "UNKNOWN",
        "pose_label": "IMPORTED_REST_NOT_INDEPENDENTLY_COURTED",
        "source_material_parity_courted": False,
        "texture_fidelity_courted": False,
        "camera_matrix_imported": finite_matrix(camera.matrix_world),
        "ortho_scale": float(camera.data.ortho_scale),
        "resolution_px": resolution,
        "mesh_object_count_before_and_after": len(meshes),
        "outputs": outputs,
        "source_rgba_truth_authorized": False,
        "stylization_training_authorized": False,
        "training_authorized": False,
    }
    (output / "LOOKDEV_STUDY_V1.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return result


def main():
    import bpy
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--expected-sha256", required=True)
    ap.add_argument("--asset-id", required=True)
    ap.add_argument("--candidate-id", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--resolution", type=int, default=640)
    args = ap.parse_args()
    output = render_study(
        bpy, source=args.source.resolve(), expected_sha256=args.expected_sha256,
        asset_id=args.asset_id, candidate_id=args.candidate_id,
        output=args.output, resolution=args.resolution)
    print(json.dumps({
        "status": output["status"],
        "study_modes": [x["mode"] for x in output["outputs"]],
        "training_authorized": False,
    }, sort_keys=True))


if __name__ == "__main__":
    import sys
    old = sys.argv
    try:
        sys.argv = [sys.argv[0]] + sys.argv[sys.argv.index("--") + 1:]
        main()
    finally:
        sys.argv = old
