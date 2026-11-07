#!/usr/bin/env python3
"""SHA-locked source-space 8-view 2D-game-art render CANDIDATE factory.

2D artwork is *presentation policy*, not proof of source-authored 2D graphics.
No semantic labels, model truth, human approval, native alpha/material parity,
anatomical front/up or training permission are inferred from these renders.

Blender command must specify --python-exit-code 1. Original bytes untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.corpus.blender_corpus4008_native_source_probe_v1 import (
    import_source, sha, finite_matrix, require,
)
from tools.corpus.blender_corpus4008_lookdev_study_v1 import (
    bound_world_points, configure_scene, configure_source_material_diagnostic,
    configure_flat_study,
)

SCHEMA = "RealSaS.Graphics2DArtOrbitCandidate.v1"
VIEWS = 8
VERSION = "real-sas-toon-shader-plus-silhouette-ink-v1"


def choose_ortho_scale(base_scale, span_x, span_y, *, target_min=0.58,
                       target_max=0.86):
    """Correct source-space framing by actual projected source bounds.

    Object corner bounds can overestimate visible silhouette. Real PNG QA is
    still authoritative for legibility and crop, not these estimates.
    """
    require(all(math.isfinite(x) and x>0 for x in
                (base_scale,span_x,span_y)),"INVALID_PROJECTED_BOUNDS")
    # Scale is linear inverse of projected occupied fraction.
    # If both target spans cannot fit, favor no cropping over minimum span.
    ratio=max(min(span_x,span_y)/target_min,
              max(span_x,span_y)/target_max)
    require(math.isfinite(ratio) and ratio>0,"INVALID_ORTHO_RATIO")
    return max(0.20, min(10000., base_scale*ratio))


def set_source_space_orbit_frame(bpy, camera, meshes, base_scale):
    from mathutils import Vector
    cam_inv=camera.matrix_world.inverted()
    xyz=[]
    for obj in meshes:
        for co in obj.bound_box:
            xyz.append(cam_inv @ (obj.matrix_world @ Vector(co)))
    require(xyz,"NO_MESH_CAMERA_BOUNDS")
    # At square resolution with an orthographic camera, width and height
    # span fractions are both world-space projected span/ortho_scale.
    raw_span_x=(max(x.x for x in xyz)-min(x.x for x in xyz))/base_scale
    raw_span_y=(max(x.y for x in xyz)-min(x.y for x in xyz))/base_scale
    scale=choose_ortho_scale(base_scale,raw_span_x,raw_span_y)
    camera.data.ortho_scale=scale
    return {
        "projected_source_bbox_fraction_x_before_adjustment":raw_span_x,
        "projected_source_bbox_fraction_y_before_adjustment":raw_span_y,
        "source_bbox_correction_based_on_not_human_court":True,
        "camera_ortho_scale":scale,
    }


def image_ink(source: Path, output: Path):
    from PIL import Image, ImageChops, ImageEnhance, ImageFilter, ImageOps
    im = Image.open(source).convert("RGBA")
    alpha = im.getchannel("A")
    rgb = ImageEnhance.Color(im.convert("RGB")).enhance(1.10)
    rgb = ImageEnhance.Contrast(rgb).enhance(1.07)
    rgb = ImageOps.posterize(rgb, 5)

    # Ink only source-covered pixels: never create fake silhouette geometry.
    # Outside remains alpha=0, with no outline expansion/canvas invention.
    inner = alpha.filter(ImageFilter.MinFilter(5))
    outline = ImageChops.subtract(alpha, inner)
    # Stylized ink instead of pure black to preserve source palette.
    ink = Image.new("RGB", im.size, (26, 24, 34))
    rgb.paste(ink, (0, 0), outline)

    # Derive mild internal art strokes from tonal edges only inside the
    # visible source alpha. Source depth/face IDs remain an independent court.
    edges = ImageOps.grayscale(im.convert("RGB")).filter(
        ImageFilter.FIND_EDGES).point(lambda x: 255 if x >= 92 else 0)
    edges = ImageChops.multiply(edges, inner)
    edges = edges.point(lambda x: min(120, x))
    rgb.paste(ink, (0, 0), edges)

    out = Image.merge("RGBA", (*rgb.split(), alpha))
    require(out.getchannel("A").tobytes() == alpha.tobytes(),
            "STYLIZATION_CHANGED_ALPHA_MASK")
    out.save(output, optimize=True)
    return {
        "source_stylized_sha256": sha(source),
        "inked_target_sha256": sha(output),
        "pixel_width": im.width,
        "pixel_height": im.height,
        "style_alpha_identical_to_cel": True,
        "style_is_independently_visual_reviewed": False,
        "style_is_independently_source_courted": False,
    }


def render_orbit(bpy, *, source: Path, expected_sha256: str,
                 asset_id: str, candidate_id: str, output: Path,
                 resolution: int, reference_policy_sha256: str):
    require(source.is_file(), "SOURCE_NOT_FOUND")
    require(sha(source) == expected_sha256, "EXACT_ORIGINAL_SHA")
    require(candidate_id.endswith(expected_sha256) or
            candidate_id.endswith(expected_sha256[:24]),
            "CANDIDATE_SOURCE_SHA_DRIFT")
    require(256 <= resolution <= 1024, "RESOLUTION_BOUND")
    require(len(reference_policy_sha256) == 64 and
            all(c in "0123456789abcdef" for c in reference_policy_sha256),
            "STYLE_CONTRACT_SHA")
    import_source(bpy, source, load_anim=False)
    source_mesh_objects = sorted(
        ob.name for ob in bpy.data.objects if ob.type == "MESH")
    require(source_mesh_objects, "NO_VISIBLE_SOURCE_MESHES")
    center, radius = bound_world_points(bpy)

    from mathutils import Vector
    bpy.ops.object.camera_add()
    cam = bpy.context.object
    cam.name = "SOURCE_SPACE_ORBIT__ANATOMICAL_FRONT_UNKNOWN"
    cam.data.type = "ORTHO"
    cam.data.ortho_scale = max(2.15 * radius, 0.5)
    base_ortho_scale = float(cam.data.ortho_scale)
    scene_meshes = [ob for ob in bpy.data.objects if ob.type == "MESH"]
    bpy.context.scene.camera = cam
    scene = configure_scene(bpy, resolution)
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 8
    scene.render.film_transparent = True
    scene.camera = cam
    cam.data.clip_end = max(1000.0, radius*12)

    # Original bindings held immutable in memory and restored every view.
    original_materials = [
        (ob, idx, slot.material)
        for ob in bpy.data.objects if ob.type == "MESH"
        for idx, slot in enumerate(ob.material_slots)
        if slot.material is not None
    ]
    native_material_inventory = [
        {
            "owner_mesh_teacher_only": ob.name,
            "slot_index": idx,
            "original_material_name_teacher_only": mat.name,
        } for ob, idx, mat in original_materials
    ]
    output.mkdir(parents=True, exist_ok=True)
    results = []
    # Do NOT name anatomical front, side or back. This is only a geometric
    # orbit in raw imported source-space under a fixed Z-up convention.
    for i in range(VIEWS):
        theta = 2*math.pi*i/VIEWS
        direction = Vector((math.cos(theta), math.sin(theta), 0.45))
        cam.location = center + direction.normalized()*max(radius*3.7, 1.0)
        cam.rotation_euler = (center-cam.location).to_track_quat("-Z", "Y").to_euler()
        bpy.context.view_layer.update()
        framing = set_source_space_orbit_frame(
            bpy, cam, scene_meshes, base_ortho_scale)
        bpy.context.view_layer.update()
        matrix = finite_matrix(cam.matrix_world)
        for ob,idx,mat in original_materials:
            ob.material_slots[idx].material = mat
        if i == 0:
            configure_source_material_diagnostic(bpy, center)
        # The diagnostic camera is rotated, lighting remains fixed.
        # Both style + reference for this view share this same camera and pose.
        for ob,idx,mat in original_materials:
            ob.material_slots[idx].material = mat
        scene.render.engine = "CYCLES"
        scene.cycles.device = "CPU"
        scene.cycles.samples = 8
        source_path = output/("source_material_diagnostic_orbit_%02d.png" % i)
        scene.render.filepath = str(source_path)
        bpy.ops.render.render(write_still=True)
        require(source_path.is_file() and source_path.stat().st_size>0,
                "SOURCE_VIEW_RENDER_EMPTY")

        configure_flat_study(bpy, cel=True, original_materials=original_materials)
        cel_path = output/("cel_3band_orbit_%02d.png" % i)
        scene.render.filepath = str(cel_path)
        bpy.ops.render.render(write_still=True)
        require(cel_path.is_file() and cel_path.stat().st_size>0,
                "CEL_VIEW_RENDER_EMPTY")

        target_path = output/("inked_2d_orbit_%02d.png" % i)
        # Non-Blender postprocess generates inked 2D images using Pillow;
        # do not assume Blender's bundled Python has third-party PIL.
        results.append({
            "view_id": "SOURCE_ORBIT_%02d" % i,
            "source_space_orbit_degrees": 45*i,
            "camera_source_space_matrix": matrix,
            "camera_projection_details": framing,
            "camera_axis_authored_front_qualified": False,
            "camera_source_up_qualified": False,
            "source_reference_png": source_path.name,
            "source_reference_sha256": sha(source_path),
            "cel_candidate_png": cel_path.name,
            "cel_candidate_sha256": sha(cel_path),
            "inked_2d_candidate_png": target_path.name,
            "inked_2d_candidate_sha256": None,
            "inked_2d_alpha_identical_to_cel": None,
            "view_2d_art_truth_admitted": False,
            "source_material_parity_qualified": False,
        })
    require(sorted(ob.name for ob in bpy.data.objects if ob.type=="MESH")
            == source_mesh_objects, "SOURCE_MESH_CHANGED")
    result = {
        "schema": SCHEMA,
        "status": "EIGHT_SOURCE_SPACE_SOURCE_AND_CEL_READY_AWAITING_INK",
        "asset_id": asset_id, "candidate_id": candidate_id,
        "source_sha256": expected_sha256,
        "source_format": source.suffix.lower(),
        "blender_version": bpy.app.version_string,
        "style_recipe_id": VERSION,
        "style_contract_sha256": reference_policy_sha256,
        "style_class_candidate": "SOURCE_BOUND_2D_GAME_ART_INK_TOON",
        "style_origin": "DETERMINISTIC_NON_GENERATIVE_RENDER",
        "ink_postprocess_completed": False,
        "source_art_native_2d": False,
        "original_source_rgb_authority_qualified": False,
        "graphics_style_2d_truth_qualified": False,
        "image_source_legal_rights_revalidated": False,
        "source_character_class": "UNKNOWN",
        "source_anatomical_front": "UNKNOWN",
        "source_anatomical_up": "UNKNOWN",
        "source_pose": "IMPORTED_REST_UNQUALIFIED",
        "native_materials_teacher_only": native_material_inventory,
        "source_mesh_object_count": len(source_mesh_objects),
        "camera_orbit_space": "SOURCE_AXIS_Z_UP_UNQUALIFIED",
        "view_count": len(results),
        "views": results,
        "independent_visual_review": "PENDING",
        "source_face_to_render_pixel_courted": False,
        "edit_consequence_courted": False,
        "training_authorized": False,
    }
    manifest = output/"GRAPHICS_2D_ART_ORBIT_CANDIDATE_V1.json"
    manifest.write_text(json.dumps(result, indent=2, sort_keys=True,
                                   allow_nan=False)+"\n")
    print(json.dumps({
        "asset_id": asset_id,
        "views": len(results),
        "style": VERSION,
        "all_2d_style_supervision_unqualified": True,
        "ink_postprocess_completed": False,
        "training_authorized": False,
    }, sort_keys=True))
    return result


def main():
    import bpy
    p=argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--expected-sha256", required=True)
    p.add_argument("--asset-id", required=True)
    p.add_argument("--candidate-id", required=True)
    p.add_argument("--style-policy-sha256", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--resolution", type=int, default=512)
    a=p.parse_args()
    render_orbit(
        bpy,source=a.source.resolve(),expected_sha256=a.expected_sha256,
        asset_id=a.asset_id,candidate_id=a.candidate_id,
        reference_policy_sha256=a.style_policy_sha256,
        output=a.output,resolution=a.resolution,
    )


if __name__=="__main__":
    old=sys.argv
    try:
        sys.argv=[old[0]]+old[old.index("--")+1:]
        main()
    finally:
        sys.argv=old
