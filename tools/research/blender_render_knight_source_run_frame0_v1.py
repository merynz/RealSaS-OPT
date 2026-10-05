from __future__ import annotations

"""Render the exact source FBX RUN first sampled frame for retarget comparison.

The renderer uses the same canonical object basis as the motion extractor and
shows one research presentation loadout: 1H_Sword + Round_Shield. Geometry,
armature, skinning and authored animation all remain source-FBX exact.
"""

import argparse
import json
from pathlib import Path
import sys

import bpy
from mathutils import Matrix, Vector

KEEP_EQUIPMENT = {"1H_Sword", "Round_Shield"}
ALL_EQUIPMENT = {
    "1H_Sword",
    "1H_Sword_Offhand",
    "2H_Sword",
    "Spike_Shield",
    "Badge_Shield",
    "Round_Shield",
    "Rectangle_Shield",
}
REQUESTED_TAKE = "HumanArmature|HumanArmature|Run"


def find_named_bone(bones, *names):
    for name in names:
        if name in bones:
            return bones[name]
    return None


def normalize(v, label):
    x = Vector(v)
    if x.length <= 1e-10:
        raise RuntimeError(label)
    x.normalize()
    return x


def canonical_basis(armature):
    bones = armature.data.bones
    hips = find_named_bone(bones, "Hips", "hips", "Root", "root")
    head = find_named_bone(bones, "Head", "head")
    if hips is None or head is None:
        raise RuntimeError("SOURCE_COMPARE_REQUIRES_HIPS_HEAD")
    up = normalize(head.head_local - hips.head_local, "SOURCE_COMPARE_UP_DEGENERATE")
    lateral = []
    for base in ("Shoulder", "UpperArm", "Palm", "UpperLeg", "Foot"):
        left = find_named_bone(bones, base + ".L", base + "_L", base + "Left")
        right = find_named_bone(bones, base + ".R", base + "_R", base + "Right")
        if left is not None and right is not None:
            lateral.append(right.head_local - left.head_local)
    if not lateral:
        raise RuntimeError("SOURCE_COMPARE_REQUIRES_LR_PAIR")
    right = Vector((0.0, 0.0, 0.0))
    for row in lateral:
        right += row
    right = right - up * right.dot(up)
    right = normalize(right, "SOURCE_COMPARE_RIGHT_DEGENERATE")
    forward = normalize(up.cross(right), "SOURCE_COMPARE_FORWARD_DEGENERATE")
    right = normalize(forward.cross(up), "SOURCE_COMPARE_RIGHT_ORTHO_FAIL")
    B = Matrix((
        (right.x, forward.x, up.x),
        (right.y, forward.y, up.y),
        (right.z, forward.z, up.z),
    ))
    C = B.transposed()
    if C.determinant() < 0.999:
        raise RuntimeError("SOURCE_COMPARE_FRAME_NOT_RIGHT_HANDED")
    return B, C


def action_by_requested_take(requested):
    exact = [a for a in bpy.data.actions if a.name == requested]
    if len(exact) == 1:
        return exact[0]
    short = requested.split("|")[-1]
    matches = [a for a in bpy.data.actions if a.name == short or a.name.endswith("|" + short)]
    if len(matches) != 1:
        raise RuntimeError("SOURCE_COMPARE_ACTION_NOT_UNIQUE:" + requested + ":" + ",".join(a.name for a in bpy.data.actions))
    return matches[0]


def bind_action(armature, action):
    if armature.animation_data is None:
        armature.animation_data_create()
    armature.animation_data.use_nla = False
    armature.animation_data.action = action
    suitable = list(armature.animation_data.action_suitable_slots)
    by_user = [slot for slot in suitable if armature in tuple(slot.users())]
    by_name = [slot for slot in suitable if str(slot.name_display) == str(armature.name)]
    if len(by_user) == 1:
        slot = by_user[0]
    elif len(by_name) == 1:
        slot = by_name[0]
    elif len(suitable) == 1:
        slot = suitable[0]
    else:
        raise RuntimeError("SOURCE_COMPARE_ACTION_SLOT_NOT_UNIQUE")
    armature.animation_data.action_slot = slot


def visible_meshes():
    out = []
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH":
            continue
        if obj.name in ALL_EQUIPMENT and obj.name not in KEEP_EQUIPMENT:
            obj.hide_render = True
            obj.hide_viewport = True
            continue
        obj.hide_render = False
        obj.hide_viewport = False
        out.append(obj)
    return tuple(out)


def evaluated_bbox_world(meshes, depsgraph):
    points = []
    for obj in meshes:
        eval_obj = obj.evaluated_get(depsgraph)
        mesh = eval_obj.to_mesh()
        try:
            M = eval_obj.matrix_world
            for vertex in mesh.vertices:
                points.append(M @ vertex.co)
        finally:
            eval_obj.to_mesh_clear()
    if not points:
        raise RuntimeError("SOURCE_COMPARE_VISIBLE_MESH_EMPTY")
    return points


def setup_camera(*, B, bbox_points, out_png):
    right = Vector((B[0][0], B[1][0], B[2][0]))
    forward = Vector((B[0][1], B[1][1], B[2][1]))
    up = Vector((B[0][2], B[1][2], B[2][2]))

    xs = [float(p.dot(right)) for p in bbox_points]
    ys = [float(p.dot(up)) for p in bbox_points]
    ds = [float(p.dot(forward)) for p in bbox_points]
    cx = 0.5 * (min(xs) + max(xs))
    cy = 0.5 * (min(ys) + max(ys))
    cd = 0.5 * (min(ds) + max(ds))
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)
    span = max(width, height, 1e-6)

    center = right * cx + up * cy + forward * cd
    camera_data = bpy.data.cameras.new("SOURCE_COMPARE_CAMERA")
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = 1.12 * span
    camera = bpy.data.objects.new("SOURCE_COMPARE_CAMERA", camera_data)
    bpy.context.collection.objects.link(camera)
    camera.location = center + forward * (2.0 * span + 1.0)
    R = Matrix((
        (right.x, up.x, forward.x),
        (right.y, up.y, forward.y),
        (right.z, up.z, forward.z),
    ))
    camera.rotation_euler = R.to_euler()
    bpy.context.scene.camera = camera

    scene = bpy.context.scene
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.light = "STUDIO"
    scene.display.shading.color_type = "SINGLE"
    scene.display.shading.single_color = (0.66, 0.70, 0.76)
    scene.display.shading.show_shadows = True
    scene.display.shading.show_cavity = True
    scene.display.shading.cavity_type = "WORLD"
    scene.display.shading.show_specular_highlight = False
    scene.display.shading.background_type = "WORLD"
    scene.display.shading.background_color = (1.0, 1.0, 1.0)
    scene.render.resolution_x = 768
    scene.render.resolution_y = 768
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    scene.render.filepath = str(out_png)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-fbx", required=True)
    ap.add_argument("--out-png", required=True)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args(argv)

    source = Path(args.source_fbx).resolve()
    out_png = Path(args.out_png).resolve()
    out_json = Path(args.out_json).resolve()
    out_png.parent.mkdir(parents=True, exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    try:
        bpy.ops.import_scene.fbx(filepath=str(source), use_anim=True)
    except TypeError:
        bpy.ops.import_scene.fbx(filepath=str(source))

    arms = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]
    if len(arms) != 1:
        raise RuntimeError("SOURCE_COMPARE_REQUIRES_ONE_ARMATURE:" + str(len(arms)))
    armature = arms[0]
    action = action_by_requested_take(REQUESTED_TAKE)
    bind_action(armature, action)

    first = int(round(float(action.frame_range[0])))
    if first != 1:
        raise RuntimeError("SOURCE_COMPARE_RUN_FIRST_FRAME_DRIFT:" + str(first))
    scene = bpy.context.scene
    scene.frame_set(first)
    bpy.context.view_layer.update()

    meshes = visible_meshes()
    if not KEEP_EQUIPMENT.issubset({obj.name for obj in meshes}):
        raise RuntimeError("SOURCE_COMPARE_LOADOUT_MISSING")

    B, C = canonical_basis(armature)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    bbox_points = evaluated_bbox_world(meshes, depsgraph)
    setup_camera(B=B, bbox_points=bbox_points, out_png=out_png)
    bpy.ops.render.render(write_still=True)

    payload = {
        "schema": "RealSaS.KnightSourceRunFrame0Comparison.v1",
        "source_take": action.name,
        "source_frame": first,
        "presentation_loadout": sorted(KEEP_EQUIPMENT),
        "source_mesh_rig_skin_animation": "FBX_EXACT",
        "camera_frame": "REALSAS_OBJECT_FRAME_V1_FRONT_XZ",
        "canonical_basis_source_columns": [[float(B[r][c]) for c in range(3)] for r in range(3)],
        "canonical_transform_source_to_realsas": [[float(C[r][c]) for c in range(3)] for r in range(3)],
        "visible_mesh_objects": sorted(obj.name for obj in meshes),
        "render_png": out_png.name,
    }
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    print("KNIGHT_SOURCE_RUN_FRAME0_COMPARE_PASS", out_png)
    return 0


if __name__ == "__main__":
    args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    raise SystemExit(main(args))
