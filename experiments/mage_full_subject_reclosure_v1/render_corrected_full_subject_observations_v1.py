from __future__ import annotations

# Executed by Blender's Python:
# blender -b --factory-startup --python render_corrected_full_subject_observations_v1.py -- ...

import argparse
import json
import math
from pathlib import Path
import sys

import bpy
from mathutils import Matrix, Vector


SCHEMA = "RealSaS.MageCorrectedFullSubjectObservationRender.v1"
MARGIN_SCALE = 1.12
RESOLUTION = 1024


def parse_args():
    argv = sys.argv
    argv = argv[argv.index("--") + 1 :] if "--" in argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--source-fbx", required=True)
    p.add_argument("--camera-dir", required=True)
    p.add_argument("--output-dir", required=True)
    return p.parse_args(argv)


def primary_armature():
    arms = [o for o in bpy.context.scene.objects if o.type == "ARMATURE"]
    if not arms:
        raise RuntimeError("MAGE_REFRAME_NO_ARMATURE")
    return max(arms, key=lambda o: len(o.data.bones))


def mechanically_related_mesh(obj, arm) -> bool:
    if obj.type != "MESH":
        return False
    if obj.parent == arm:
        return True
    for mod in obj.modifiers:
        if mod.type == "ARMATURE" and mod.object == arm:
            return True
    return False


def admitted_subject(arm):
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    admitted = [o for o in meshes if mechanically_related_mesh(o, arm)]
    excluded = [o for o in meshes if o not in admitted]
    if not admitted:
        raise RuntimeError("MAGE_REFRAME_NO_ADMITTED_MESH")
    for o in admitted:
        o.hide_render = False
        o.hide_set(False)
    for o in excluded:
        o.hide_render = True
        o.hide_set(True)
    return admitted, excluded


def world_vertices(objects):
    deps = bpy.context.evaluated_depsgraph_get()
    out = []
    for obj in objects:
        ev = obj.evaluated_get(deps)
        mesh = ev.to_mesh()
        try:
            M = ev.matrix_world
            out.extend(tuple(M @ v.co) for v in mesh.vertices)
        finally:
            ev.to_mesh_clear()
    if not out:
        raise RuntimeError("MAGE_REFRAME_EMPTY_WORLD_VERTEX_SET")
    return out


def view_basis(view):
    t = math.radians(45.0 * int(view))
    f = Vector((math.sin(t), -math.cos(t), 0.0))
    r = Vector((-math.cos(t), -math.sin(t), 0.0))
    u = Vector((0.0, 0.0, 1.0))
    return r, u, f


def clip_polygon(poly, normal, bound):
    # Keep dot(normal, p) <= bound.
    if not poly:
        return []
    out = []
    eps = 1e-12
    for i in range(len(poly)):
        a = poly[i - 1]
        b = poly[i]
        fa = normal[0] * a[0] + normal[1] * a[1] - bound
        fb = normal[0] * b[0] + normal[1] * b[1] - bound
        ina = fa <= eps
        inb = fb <= eps
        if inb:
            if not ina:
                denom = fa - fb
                t = fa / denom if abs(denom) > 1e-15 else 0.0
                out.append((a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])))
            out.append(b)
        elif ina:
            denom = fa - fb
            t = fa / denom if abs(denom) > 1e-15 else 0.0
            out.append((a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])))
    return out


def feasible_xy(points_xy, half):
    M = 1000.0
    poly = [(-M, -M), (M, -M), (M, M), (-M, M)]
    for view in range(8):
        r, _u, _f = view_basis(view)
        rx, ry = float(r.x), float(r.y)
        vals = [p[0] * rx + p[1] * ry for p in points_xy]
        lo = min(vals)
        hi = max(vals)
        # hi-half <= dot(r,c) <= lo+half
        poly = clip_polygon(poly, (rx, ry), lo + half)
        poly = clip_polygon(poly, (-rx, -ry), -(hi - half))
        if not poly:
            return []
    return poly


def solve_common_frame(points):
    xy = [(float(p.x), float(p.y)) for p in points]
    zs = [float(p.z) for p in points]
    zmin, zmax = min(zs), max(zs)
    zcenter = 0.5 * (zmin + zmax)
    zhalf = 0.5 * (zmax - zmin)

    lower = 0.0
    for view in range(8):
        r, _u, _f = view_basis(view)
        vals = [x * r.x + y * r.y for x, y in xy]
        lower = max(lower, 0.5 * (max(vals) - min(vals)))
    upper = max(lower, zhalf, 1.0)
    while not feasible_xy(xy, upper):
        upper *= 2.0
    for _ in range(64):
        mid = 0.5 * (lower + upper)
        if feasible_xy(xy, mid):
            upper = mid
        else:
            lower = mid
    poly = feasible_xy(xy, upper * (1.0 + 1e-10))
    if not poly:
        raise RuntimeError("MAGE_REFRAME_XY_MINIMAX_SOLVE_FAIL")
    cx = sum(p[0] for p in poly) / len(poly)
    cy = sum(p[1] for p in poly) / len(poly)
    base_half = max(float(upper), float(zhalf))
    return {
        "center": [float(cx), float(cy), float(zcenter)],
        "base_half_extent": float(base_half),
        "half_extent": float(base_half * MARGIN_SCALE),
        "margin_scale": MARGIN_SCALE,
        "z_min": float(zmin),
        "z_max": float(zmax),
    }


def configure_render():
    scene = bpy.context.scene
    engine_ok = False
    for engine in ("BLENDER_WORKBENCH_NEXT", "BLENDER_WORKBENCH"):
        try:
            scene.render.engine = engine
            engine_ok = True
            break
        except Exception:
            pass
    if not engine_ok:
        raise RuntimeError("MAGE_REFRAME_WORKBENCH_ENGINE_UNAVAILABLE")

    scene.render.resolution_x = RESOLUTION
    scene.render.resolution_y = RESOLUTION
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = True

    sh = scene.display.shading
    sh.light = "FLAT"
    sh.color_type = "TEXTURE"
    sh.show_shadows = False
    sh.show_cavity = False
    if hasattr(sh, "show_specular_highlight"):
        sh.show_specular_highlight = False
    if hasattr(sh, "show_backface_culling"):
        sh.show_backface_culling = False

    # Workbench + texture should preserve source texture rather than relight it.
    scene.view_settings.view_transform = "Standard"
    try:\n        scene.view_settings.look = "Medium High Contrast"\n    except Exception:\n        pass\n    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0


def camera_object():
    data = bpy.data.cameras.new("REALSAS_ORTHO_CAMERA")
    data.type = "ORTHO"
    obj = bpy.data.objects.new("REALSAS_ORTHO_CAMERA", data)
    bpy.context.scene.collection.objects.link(obj)
    bpy.context.scene.camera = obj
    return obj


def set_camera(obj, *, origin, right, up, forward, half_extent):
    origin = Vector(origin)
    right = Vector(right).normalized()
    up = Vector(up).normalized()
    forward = Vector(forward).normalized()
    # Camera local X=screen right, local Y=screen up, local -Z=forward.
    obj.matrix_world = Matrix((
        (right.x, up.x, -forward.x, origin.x),
        (right.y, up.y, -forward.y, origin.y),
        (right.z, up.z, -forward.z, origin.z),
        (0.0, 0.0, 0.0, 1.0),
    ))
    obj.data.ortho_scale = 2.0 * float(half_extent)


def render_one(cam_obj, output, camera):
    set_camera(
        cam_obj,
        origin=camera["origin"],
        right=camera["right"],
        up=camera["screen_up"],
        forward=camera["forward"],
        half_extent=camera["half_extent"],
    )
    bpy.context.scene.render.filepath = str(output)
    bpy.ops.render.render(write_still=True)


def corrected_cameras(frame):
    center = Vector(frame["center"])
    half = float(frame["half_extent"])
    rows = []
    for view in range(8):
        r, u, f = view_basis(view)
        origin = center - f * (4.0 * half)
        rows.append({
            "contract": "realsas.level_orthographic_z_orbit.v1",
            "view_index": view,
            "yaw_deg": 45 * view,
            "origin": list(map(float, origin)),
            "right": list(map(float, r)),
            "screen_up": list(map(float, u)),
            "forward": list(map(float, f)),
            "half_extent": half,
            "resolution": RESOLUTION,
            "center": list(map(float, center)),
        })
    return rows


def main():
    args = parse_args()
    source = Path(args.source_fbx).resolve()
    camera_dir = Path(args.camera_dir).resolve()
    output = Path(args.output_dir).resolve()
    parity_dir = output / "parity_current_camera"
    corrected_dir = output / "corrected_full_subject"
    parity_dir.mkdir(parents=True, exist_ok=True)
    corrected_dir.mkdir(parents=True, exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=str(source))
    scene = bpy.context.scene
    scene.frame_set(0)

    arm = primary_armature()
    arm.data.pose_position = "REST"
    admitted, excluded = admitted_subject(arm)
    configure_render()
    points = world_vertices(admitted)
    frame = solve_common_frame(points)
    cam_obj = camera_object()

    current_rows = []
    for view in range(8):
        c = json.loads((camera_dir / f"V{view}.camera.json").read_text())
        render_one(cam_obj, parity_dir / f"V{view}.png", c)
        current_rows.append(c)

    new_rows = corrected_cameras(frame)
    for c in new_rows:
        view = int(c["view_index"])
        render_one(cam_obj, corrected_dir / f"V{view}.png", c)
        (corrected_dir / f"V{view}.camera.json").write_text(
            json.dumps(c, indent=2, sort_keys=True) + "\n"
        )

    xyz = [(float(p.x), float(p.y), float(p.z)) for p in points]
    lo = [min(p[i] for p in xyz) for i in range(3)]
    hi = [max(p[i] for p in xyz) for i in range(3)]
    report = {
        "schema": SCHEMA,
        "status": "PASS__DIAGNOSTIC_RERENDER_COMPLETED",
        "source_fbx": source.name,
        "primary_armature": arm.name,
        "primary_armature_bone_count": len(arm.data.bones),
        "admitted_objects": [o.name for o in admitted],
        "excluded_objects": [o.name for o in excluded],
        "world_vertex_count": len(points),
        "world_bbox_min": lo,
        "world_bbox_max": hi,
        "framing": frame,
        "parity_render": {
            "purpose": "REPRODUCE_CURRENT_BAD_CAMERA_BEFORE_ACCEPTING_RENDERER_PARITY",
            "camera_source": "CURRENT_EXACT_V0_V7_CAMERA_JSON",
        },
        "corrected_render": {
            "purpose": "SAME_SOURCE_AND_RENDERER_WITH_FULL_ADMITTED_SUBJECT_FRAMING",
            "camera_rule": "8_VIEW_HORIZONTAL_MINIMAX_PLUS_VERTICAL_HALF_SPAN_THEN_1P12_MARGIN",
        },
        "product_authority_promoted": False,
    }
    (output / "RENDER_REPORT.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print("MAGE_OBSERVATION_REFRAME_RENDER_PASS")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
