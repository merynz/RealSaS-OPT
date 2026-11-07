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
         "STYLIZED_FLAT_DIAGNOSTIC", "STYLIZED_CEL_3BAND_DIAGNOSTIC")


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
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 12
    scene.cycles.use_denoising = False
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



def _flat_or_cel_material(original, *, cel):
    """Clone existing shader graph and reuse linked base color if present.

    The original material is retained untouched; lighting/style output differs.
    This diagnostic material is never a native appearance reference.
    """
    import bpy
    from mathutils import Vector
    material = original.copy()
    material.name = "STUDY_" + ("CEL" if cel else "FLAT") + "_" + original.name
    material.use_nodes = True
    tree = material.node_tree
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    output.is_active_output = True
    emission = tree.nodes.new("ShaderNodeEmission")
    base = None
    for n in tree.nodes:
        if n.type == "BSDF_PRINCIPLED":
            base = n.inputs.get("Base Color")
            break
    color_source = base.links[0].from_socket if base is not None and base.is_linked else None
    if color_source is not None:
        if cel:
            mix = tree.nodes.new("ShaderNodeMixRGB")
            mix.blend_type = "MULTIPLY"
            mix.inputs[0].default_value = 1.0
            tree.links.new(color_source, mix.inputs[1])
            tree.links.new(mix.outputs[0], emission.inputs["Color"])
        else:
            tree.links.new(color_source, emission.inputs["Color"])
    else:
        rgba = (tuple(base.default_value) if base is not None
                else tuple(original.diffuse_color))
        if cel:
            mix = tree.nodes.new("ShaderNodeMixRGB")
            mix.blend_type = "MULTIPLY"
            mix.inputs[0].default_value = 1.0
            mix.inputs[1].default_value = rgba
            tree.links.new(mix.outputs[0], emission.inputs["Color"])
        else:
            emission.inputs["Color"].default_value = rgba
    if cel:
        normal = tree.nodes.new("ShaderNodeNewGeometry")
        dot = tree.nodes.new("ShaderNodeVectorMath")
        dot.operation = "DOT_PRODUCT"
        dot.inputs[1].default_value = Vector((0.4, -0.5, 0.77)).normalized()
        tree.links.new(normal.outputs["Normal"], dot.inputs[0])
        shift = tree.nodes.new("ShaderNodeMapRange")
        shift.inputs["From Min"].default_value = -1.0
        shift.inputs["From Max"].default_value = 1.0
        tree.links.new(dot.outputs["Value"], shift.inputs["Value"])
        ramp = tree.nodes.new("ShaderNodeValToRGB")
        ramp.color_ramp.interpolation = "CONSTANT"
        ramp.color_ramp.elements.remove(ramp.color_ramp.elements[1])
        ramp.color_ramp.elements[0].position = 0
        ramp.color_ramp.elements[0].color = (0.50, 0.50, 0.50, 1)
        e1 = ramp.color_ramp.elements.new(0.35)
        e1.color = (0.77, 0.77, 0.77, 1)
        e2 = ramp.color_ramp.elements.new(0.72)
        e2.color = (1, 1, 1, 1)
        tree.links.new(shift.outputs["Result"], ramp.inputs["Fac"])
        tree.links.new(ramp.outputs["Color"], mix.inputs[2])
    tree.links.new(emission.outputs[0], output.inputs["Surface"])
    return material


def configure_flat_study(bpy, *, cel, original_materials):
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 8
    scene.cycles.use_denoising = False
    for obj, index, mat in original_materials:
        obj.material_slots[index].material = _flat_or_cel_material(mat, cel=cel)


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
    original_materials = [
        (o, i, slot.material)
        for o in bpy.data.objects if o.type == "MESH"
        for i, slot in enumerate(o.material_slots)
        if slot.material is not None
    ]
    outputs = []
    for mode in VIEWS:
        if mode == VIEWS[0]:
            configure_source_material_diagnostic(bpy, center)
        else:
            configure_flat_study(bpy, cel=(mode == VIEWS[2]),
                                 original_materials=original_materials)
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
