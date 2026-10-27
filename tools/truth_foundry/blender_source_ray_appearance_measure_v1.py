#!/usr/bin/env python3
from __future__ import annotations

"""Measure packed-source appearance at canonical geometric rays by intervention.

This producer deliberately does not read alpha/RGB truth out of material nodes.
It:

1. opens the exact packed ``source_scene.blend``;
2. replays the sealed source-observation material/opacity policy;
3. selects deterministic geometry-only canonical pixel rays;
4. reconstructs the sealed teacher triangles with exact face/material/UV binding;
5. proves local reconstructed-full parity to the packed evaluator;
6. measures full, standalone-hit and remove-hit renders at shrinking 1x1
   orthographic footprints centred on the exact half-integer source pixel ray;
7. asks ``qualify_measured_ray_appearance_v1`` whether those measured
   consequences are actually factorable by scalar source-over per-hit alpha/RGB.

A shrinking footprint is an explicit convergence experiment, not an assertion
that a finite anti-aliased raster pixel is a mathematical point sample.  Failure
or non-convergence abstains/fails closed.  This producer mints no qualification,
student target, training authority or product activation.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Iterable

import bpy
from mathutils import Vector
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.iris.truth.graphics_oracle_v1 import (
    camera_from_mapping,
    face_centroid_candidate_pixels_v1,
    measure_orthographic_ray_stack_v1,
)
from models.iris.truth.ray_appearance_measurement_v1 import (
    FAIL_REPLAY,
    PASS,
    qualify_measured_ray_appearance_v1,
)
from tools.iris_vnext import blender_authored_teacher_export_v1 as source_transport
from tools.observation import blender_controlled_8view_v2 as controlled
from tools.truth_foundry import blender_source_appearance_policy_replay_v1 as appearance_replay

SCHEMA = "RealSaS.SourceRayAppearanceMeasurementCandidate.v1"


def _args() -> argparse.Namespace:
    raw = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--subject-id", required=True)
    p.add_argument("--teacher-scene", type=Path, required=True)
    p.add_argument("--geometry-npz", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--camera-bundle", type=Path, required=True)
    p.add_argument("--expected-blend-sha256", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--maximum-rays", type=int, default=2)
    p.add_argument("--maximum-hits", type=int, default=2)
    p.add_argument("--cycles-samples", type=int, default=256)
    p.add_argument("--large-footprint-fraction", type=float, default=1.0 / 16.0)
    p.add_argument("--small-footprint-fraction", type=float, default=1.0 / 32.0)
    p.add_argument("--convergence-tolerance", type=float, default=5.0e-3)
    p.add_argument("--representation-tolerance", type=float, default=5.0e-3)
    p.add_argument("--measurement-tolerance", type=float, default=7.5e-3)
    return p.parse_args(raw)


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _finite_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return _finite_json(value.item())
    if isinstance(value, np.ndarray):
        return [_finite_json(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {str(k): _finite_json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_finite_json(v) for v in value]
    return value


def _validate_scalar_policy(a: argparse.Namespace) -> None:
    if a.maximum_rays < 1 or a.maximum_hits < 1 or a.cycles_samples < 16:
        raise RuntimeError("SOURCE_RAY_APPEARANCE_RENDER_POLICY")
    values = (
        a.large_footprint_fraction,
        a.small_footprint_fraction,
        a.convergence_tolerance,
        a.representation_tolerance,
        a.measurement_tolerance,
    )
    if not all(math.isfinite(float(v)) and float(v) > 0 for v in values):
        raise RuntimeError("SOURCE_RAY_APPEARANCE_NUMERIC_POLICY")
    if not 0 < a.small_footprint_fraction < a.large_footprint_fraction <= 1:
        raise RuntimeError("SOURCE_RAY_APPEARANCE_FOOTPRINT_ORDER")


def _teacher_arrays(path: Path) -> dict[str, np.ndarray]:
    required = {
        "rest_vertices_source_world",
        "faces",
        "face_owner_privileged",
        "face_material_privileged",
        "triangle_uv",
        "triangle_uv_valid",
    }
    with np.load(path, allow_pickle=False) as z:
        if not required.issubset(set(z.files)):
            raise RuntimeError("SOURCE_RAY_APPEARANCE_TEACHER_ARRAYS_MISSING")
        rows = {name: np.asarray(z[name]).copy() for name in required}
    vertices = np.asarray(rows["rest_vertices_source_world"], np.float64)
    faces = np.asarray(rows["faces"], np.int64)
    owners = np.asarray(rows["face_owner_privileged"], np.int64)
    materials = np.asarray(rows["face_material_privileged"], np.int64)
    uv = np.asarray(rows["triangle_uv"], np.float64)
    uv_valid = np.asarray(rows["triangle_uv_valid"])
    if (
        vertices.ndim != 2
        or vertices.shape[1:] != (3,)
        or not len(vertices)
        or not np.isfinite(vertices).all()
        or faces.ndim != 2
        or faces.shape[1:] != (3,)
        or not len(faces)
        or np.any(faces < 0)
        or np.any(faces >= len(vertices))
        or owners.shape != (len(faces),)
        or materials.shape != (len(faces),)
        or uv.shape != (len(faces), 3, 2)
        or uv_valid.shape != (len(faces),)
        or uv_valid.dtype != np.bool_
    ):
        raise RuntimeError("SOURCE_RAY_APPEARANCE_TEACHER_ARRAY_LAYOUT")
    rows["rest_vertices_source_world"] = vertices
    rows["faces"] = faces
    rows["face_owner_privileged"] = owners
    rows["face_material_privileged"] = materials
    rows["triangle_uv"] = uv
    rows["triangle_uv_valid"] = uv_valid
    return rows


def _tie_epsilon(vertices: np.ndarray) -> float:
    scale = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
    if not math.isfinite(scale) or scale <= 0:
        raise RuntimeError("SOURCE_RAY_APPEARANCE_ZERO_GEOMETRY_SCALE")
    return max(scale * 1.0e-7, 1.0e-9)


def _select_rays(
    vertices: np.ndarray,
    faces: np.ndarray,
    bundle: dict,
    *,
    maximum_rays: int,
    maximum_hits: int,
    tie_epsilon: float,
) -> list[dict]:
    cameras = tuple(bundle.get("cameras") or ())
    if len(cameras) != 8:
        raise RuntimeError("SOURCE_RAY_APPEARANCE_CAMERA_COUNT")
    candidates: list[tuple[int, int, int, object, object]] = []
    for row in sorted(cameras, key=lambda v: int(v["view_index"])):
        camera = camera_from_mapping(row)
        for pixel_x, pixel_y in face_centroid_candidate_pixels_v1(
            vertices, faces, camera, maximum=256
        ):
            stack = measure_orthographic_ray_stack_v1(
                vertices,
                faces,
                camera,
                int(pixel_x),
                int(pixel_y),
                tie_epsilon=tie_epsilon,
            )
            n = len(stack.hits)
            if not stack.complete or stack.has_depth_tie or n < 1 or n > maximum_hits:
                continue
            # Geometry-only preference: multi-hit rays first, then stable view/pixel order.
            candidates.append((-n, int(stack.view_index), int(pixel_y), int(pixel_x), stack))
    candidates.sort(key=lambda row: row[:4])
    selected = []
    used_views: set[int] = set()
    # Prefer distinct views without changing selection based on appearance outcomes.
    for _negative_hits, view, _y, _x, stack in candidates:
        if view in used_views:
            continue
        selected.append(stack)
        used_views.add(view)
        if len(selected) == maximum_rays:
            break
    if len(selected) < maximum_rays:
        selected_ids = {(s.view_index, s.pixel_x, s.pixel_y) for s in selected}
        for _negative_hits, _view, _y, _x, stack in candidates:
            key = (stack.view_index, stack.pixel_x, stack.pixel_y)
            if key in selected_ids:
                continue
            selected.append(stack)
            selected_ids.add(key)
            if len(selected) == maximum_rays:
                break
    if not selected:
        raise RuntimeError("SOURCE_RAY_APPEARANCE_NO_GEOMETRIC_RAY_CANDIDATE")
    return [
        {
            "stack": stack,
            "camera": next(row for row in cameras if int(row["view_index"]) == stack.view_index),
        }
        for stack in selected
    ]


def _transparent_material() -> bpy.types.Material:
    material = bpy.data.materials.new("RealSaS_RayMeasurement_Transparent")
    material.use_nodes = True
    tree = material.node_tree
    tree.nodes.clear()
    out = tree.nodes.new("ShaderNodeOutputMaterial")
    transparent = tree.nodes.new("ShaderNodeBsdfTransparent")
    tree.links.new(transparent.outputs[0], out.inputs["Surface"])
    return material


def _build_triangle_soup(
    arrays: dict[str, np.ndarray],
    owners: tuple[bpy.types.Object, ...],
    transparent: bpy.types.Material,
):
    faces = arrays["faces"]
    vertices = arrays["rest_vertices_source_world"]
    face_owner = arrays["face_owner_privileged"]
    face_material = arrays["face_material_privileged"]
    triangle_uv = arrays["triangle_uv"]
    triangle_uv_valid = arrays["triangle_uv_valid"]
    if len(owners) == 0 or np.any(face_owner < 0) or np.any(face_owner >= len(owners)):
        raise RuntimeError("SOURCE_RAY_APPEARANCE_OWNER_BINDING")

    global_map: dict[int, tuple[bpy.types.Mesh, int, int, int]] = {}
    soup_objects = []
    for owner_index, source_obj in enumerate(owners):
        global_faces = np.flatnonzero(face_owner == owner_index)
        if not len(global_faces):
            continue
        local_vertices = []
        local_faces = []
        for local_index, face_index in enumerate(global_faces):
            tri = faces[face_index]
            local_vertices.extend(vertices[tri].tolist())
            local_faces.append((3 * local_index, 3 * local_index + 1, 3 * local_index + 2))

        mesh = bpy.data.meshes.new("RealSaS_RaySoupMesh_" + source_obj.name)
        mesh.from_pydata(local_vertices, [], local_faces)
        mesh.update()
        for slot in source_obj.material_slots:
            if slot.material is None:
                raise RuntimeError("SOURCE_RAY_APPEARANCE_NULL_MATERIAL_SLOT:" + source_obj.name)
            mesh.materials.append(slot.material)
        transparent_index = len(mesh.materials)
        mesh.materials.append(transparent)
        uv_layer = mesh.uv_layers.new(name="UVMap")

        for local_index, face_index_raw in enumerate(global_faces):
            face_index = int(face_index_raw)
            material_index = int(face_material[face_index])
            if not 0 <= material_index < transparent_index:
                raise RuntimeError("SOURCE_RAY_APPEARANCE_FACE_MATERIAL_BINDING")
            polygon = mesh.polygons[local_index]
            polygon.material_index = material_index
            if bool(triangle_uv_valid[face_index]):
                for corner, loop_index in enumerate(polygon.loop_indices):
                    uv_layer.data[loop_index].uv = tuple(map(float, triangle_uv[face_index, corner]))
            global_map[face_index] = (mesh, local_index, material_index, transparent_index)

        obj = bpy.data.objects.new("RealSaS_RaySoup_" + source_obj.name, mesh)
        bpy.context.scene.collection.objects.link(obj)
        soup_objects.append(obj)

    if len(global_map) != len(faces):
        raise RuntimeError("SOURCE_RAY_APPEARANCE_SOUP_FACE_COVERAGE")
    bpy.context.view_layer.update()
    return tuple(soup_objects), global_map


def _set_soup_mode(global_map, *, mode: str, face_index: int | None = None) -> None:
    if mode not in {"FULL", "STANDALONE", "REMOVAL"}:
        raise RuntimeError("SOURCE_RAY_APPEARANCE_INTERVENTION_MODE")
    if mode != "FULL" and (type(face_index) is not int or face_index not in global_map):
        raise RuntimeError("SOURCE_RAY_APPEARANCE_INTERVENTION_FACE")
    touched = set()
    for index, (mesh, local_index, original, transparent) in global_map.items():
        if mode == "FULL":
            value = original
        elif mode == "STANDALONE":
            value = original if index == face_index else transparent
        else:
            value = transparent if index == face_index else original
        mesh.polygons[local_index].material_index = value
        touched.add(mesh)
    for mesh in touched:
        mesh.update()
    bpy.context.view_layer.update()


def _configure_measurement(samples: int) -> None:
    controlled.configure(1)
    scene = bpy.context.scene
    scene.cycles.samples = int(samples)
    scene.cycles.seed = 0
    scene.render.image_settings.file_format = "OPEN_EXR"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "32"
    scene.render.image_settings.exr_codec = "ZIP"


def _make_camera() -> bpy.types.Object:
    data = bpy.data.cameras.new("RealSaS_SourceRayAppearance_Camera")
    data.type = "ORTHO"
    camera = bpy.data.objects.new(data.name, data)
    bpy.context.scene.collection.objects.link(camera)
    bpy.context.scene.camera = camera
    return camera


def _set_micro_camera(camera: bpy.types.Object, row: dict, pixel_x: int, pixel_y: int, fraction: float) -> float:
    resolution = int(row["resolution"])
    half = float(row["half_extent"])
    right = Vector(tuple(map(float, row["right"])))
    up = Vector(tuple(map(float, row["screen_up"])))
    forward = Vector(tuple(map(float, row["forward"])))
    origin = Vector(tuple(map(float, row["origin"])))
    x = ((pixel_x + 0.5) / resolution * 2.0 - 1.0) * half
    y = (1.0 - (pixel_y + 0.5) / resolution * 2.0) * half
    ray_origin = origin + right * x + up * y
    camera.location = ray_origin
    camera.rotation_euler = forward.to_track_quat("-Z", "Y").to_euler()
    camera.data.ortho_scale = (2.0 * half / resolution) * float(fraction)
    camera.data.clip_start = float(row["clip_start"])
    camera.data.clip_end = float(row["clip_end"])
    bpy.context.view_layer.update()
    return 2.0 * half / resolution


def _render_rgba(path: Path) -> np.ndarray:
    scene = bpy.context.scene
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError("SOURCE_RAY_APPEARANCE_EXR_MISSING")
    image = bpy.data.images.load(str(path), check_existing=False)
    try:
        if tuple(map(int, image.size)) != (1, 1):
            raise RuntimeError("SOURCE_RAY_APPEARANCE_EXR_LAYOUT")
        pixels = np.empty(4, dtype=np.float32)
        image.pixels.foreach_get(pixels)
        value = pixels.astype(np.float64, copy=True)
    finally:
        bpy.data.images.remove(image)
        path.unlink(missing_ok=True)
    if not np.isfinite(value).all() or value[3] < -1e-6 or value[3] > 1 + 1e-6:
        raise RuntimeError("SOURCE_RAY_APPEARANCE_EXR_VALUE")
    value[3] = np.clip(value[3], 0.0, 1.0)
    return value


def _calibrate_rgb_carrier(camera: bpy.types.Object, row: dict, scratch: Path) -> dict:
    # Hide all existing meshes and render a known semi-transparent emission plane
    # that fully covers the micro camera. This determines whether reloaded EXR RGB
    # is straight or premultiplied; the producer does not assume Blender storage semantics.
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    hidden = {o: bool(o.hide_render) for o in meshes}
    for obj in meshes:
        obj.hide_render = True

    forward = Vector(tuple(map(float, row["forward"])))
    right = Vector(tuple(map(float, row["right"])))
    up = Vector(tuple(map(float, row["screen_up"])))
    depth = float(row["clip_start"]) + 0.25 * (float(row["clip_end"]) - float(row["clip_start"]))
    centre = camera.location + forward * depth
    extent = max(float(camera.data.ortho_scale) * 4.0, 1.0e-5)
    verts = [
        centre - right * extent - up * extent,
        centre + right * extent - up * extent,
        centre + right * extent + up * extent,
        centre - right * extent + up * extent,
    ]
    mesh = bpy.data.meshes.new("RealSaS_RGB_CarrierCalibrationMesh")
    mesh.from_pydata([list(v) for v in verts], [], [(0, 1, 2, 3)])
    material = bpy.data.materials.new("RealSaS_RGB_CarrierCalibrationMaterial")
    material.use_nodes = True
    tree = material.node_tree
    tree.nodes.clear()
    out = tree.nodes.new("ShaderNodeOutputMaterial")
    transparent = tree.nodes.new("ShaderNodeBsdfTransparent")
    emission = tree.nodes.new("ShaderNodeEmission")
    known = np.asarray([0.2, 0.4, 0.6], dtype=np.float64)
    emission.inputs["Color"].default_value = (*known.tolist(), 1.0)
    emission.inputs["Strength"].default_value = 1.0
    mix = tree.nodes.new("ShaderNodeMixShader")
    mix.inputs[0].default_value = 0.5
    tree.links.new(transparent.outputs[0], mix.inputs[1])
    tree.links.new(emission.outputs[0], mix.inputs[2])
    tree.links.new(mix.outputs[0], out.inputs["Surface"])
    mesh.materials.append(material)
    obj = bpy.data.objects.new("RealSaS_RGB_CarrierCalibration", mesh)
    bpy.context.scene.collection.objects.link(obj)
    bpy.context.view_layer.update()
    try:
        rgba = _render_rgba(scratch / "carrier.exr")
    finally:
        bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.meshes.remove(mesh, do_unlink=True)
        bpy.data.materials.remove(material, do_unlink=True)
        for existing, value in hidden.items():
            existing.hide_render = value
        bpy.context.view_layer.update()

    if abs(float(rgba[3]) - 0.5) > 2.5e-2:
        raise RuntimeError("SOURCE_RAY_APPEARANCE_CARRIER_ALPHA_CALIBRATION")
    straight_error = float(np.max(np.abs(rgba[:3] - known)))
    premul_error = float(np.max(np.abs(rgba[:3] - known * 0.5)))
    if min(straight_error, premul_error) > 2.5e-2 or abs(straight_error - premul_error) < 1.0e-2:
        raise RuntimeError("SOURCE_RAY_APPEARANCE_RGB_CARRIER_UNIDENTIFIED")
    mode = "STRAIGHT_LINEAR_RGB" if straight_error < premul_error else "PREMULTIPLIED_LINEAR_RGB"
    return {
        "mode": mode,
        "measured_rgba": rgba.tolist(),
        "known_straight_linear_rgb": known.tolist(),
        "known_alpha": 0.5,
        "straight_error": straight_error,
        "premultiplied_error": premul_error,
    }


def _premultiplied(rgba: np.ndarray, carrier_mode: str) -> tuple[float, np.ndarray]:
    alpha = float(rgba[3])
    if carrier_mode == "STRAIGHT_LINEAR_RGB":
        rgb = rgba[:3] * alpha
    elif carrier_mode == "PREMULTIPLIED_LINEAR_RGB":
        rgb = rgba[:3].copy()
    else:
        raise RuntimeError("SOURCE_RAY_APPEARANCE_RGB_CARRIER_MODE")
    return alpha, np.asarray(rgb, np.float64)


def _max_carrier_delta(a: np.ndarray, b: np.ndarray, mode: str) -> float:
    aa, ar = _premultiplied(a, mode)
    ba, br = _premultiplied(b, mode)
    return float(max(abs(aa - ba), np.max(np.abs(ar - br))))


def _measure_condition(
    camera,
    camera_row: dict,
    pixel_x: int,
    pixel_y: int,
    *,
    large_fraction: float,
    small_fraction: float,
    scratch: Path,
    tag: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    _set_micro_camera(camera, camera_row, pixel_x, pixel_y, large_fraction)
    large = _render_rgba(scratch / f"{tag}_large.exr")
    _set_micro_camera(camera, camera_row, pixel_x, pixel_y, small_fraction)
    small = _render_rgba(scratch / f"{tag}_small.exr")
    return large, small, 0.0


def main() -> None:
    a = _args()
    _validate_scalar_policy(a)
    teacher = _load(a.teacher_scene)
    candidate = _load(a.candidate)
    bundle = _load(a.camera_bundle)
    if teacher.get("schema") != "RealSaS.TeacherSceneIR.v1" or teacher.get("subject_key") != a.subject_id:
        raise RuntimeError("SOURCE_RAY_APPEARANCE_TEACHER_BINDING")
    if teacher.get("authority") != "UNQUALIFIED_DERIVED_MEASUREMENTS" or teacher.get("qualification_minted") is not False:
        raise RuntimeError("SOURCE_RAY_APPEARANCE_TEACHER_AUTHORITY")
    if candidate.get("schema") != "RealSaS.Controlled8ViewSourceRestCandidate.v2" or candidate.get("subject_id") != a.subject_id:
        raise RuntimeError("SOURCE_RAY_APPEARANCE_CANDIDATE_BINDING")
    if bundle.get("schema") != "RealSaS.CameraProjectionBundle.v1":
        raise RuntimeError("SOURCE_RAY_APPEARANCE_CAMERA_SCHEMA")
    if _sha(a.camera_bundle) != str(candidate.get("camera_bundle", {}).get("sha256")):
        raise RuntimeError("SOURCE_RAY_APPEARANCE_CAMERA_HASH_DRIFT")
    current_blend = Path(bpy.data.filepath).resolve()
    if not current_blend.is_file() or _sha(current_blend) != a.expected_blend_sha256:
        raise RuntimeError("SOURCE_RAY_APPEARANCE_BLEND_HASH_DRIFT")

    arrays = _teacher_arrays(a.geometry_npz)
    geometry_payloads = tuple(teacher.get("geometry_payloads") or ())
    if not any(Path(str(row.get("path"))).name == a.geometry_npz.name and row.get("sha256") == _sha(a.geometry_npz) for row in geometry_payloads):
        raise RuntimeError("SOURCE_RAY_APPEARANCE_GEOMETRY_PAYLOAD_LINEAGE")
    appearance_payloads = tuple(teacher.get("appearance_payloads") or ())
    if not any(Path(str(row.get("path"))).name == "source_scene.blend" and row.get("sha256") == a.expected_blend_sha256 for row in appearance_payloads):
        raise RuntimeError("SOURCE_RAY_APPEARANCE_BLEND_LINEAGE")

    vertices = arrays["rest_vertices_source_world"]
    faces = arrays["faces"]
    tie_epsilon = _tie_epsilon(vertices)
    selected = _select_rays(
        vertices,
        faces,
        bundle,
        maximum_rays=a.maximum_rays,
        maximum_hits=a.maximum_hits,
        tie_epsilon=tie_epsilon,
    )

    owners = source_transport._mesh_objects()
    render_materials = appearance_replay._materials_from_render_meshes(owners)
    material_receipts = appearance_replay._replay_transform(
        render_materials, list(candidate.get("material_transform") or ())
    )
    _configure_measurement(a.cycles_samples)
    camera = _make_camera()
    scratch = a.output.parent / (a.output.stem + ".scratch")
    scratch.mkdir(parents=True, exist_ok=True)

    # Calibrate EXR RGB storage semantics at the first exact ray location.
    first = selected[0]
    _set_micro_camera(
        camera,
        first["camera"],
        int(first["stack"].pixel_x),
        int(first["stack"].pixel_y),
        a.small_footprint_fraction,
    )
    carrier = _calibrate_rgb_carrier(camera, first["camera"], scratch)
    carrier_mode = str(carrier["mode"])

    # Measure the packed evaluator before replacing its geometry with the exact
    # teacher triangle soup used for face-level interventions.
    original_measurements: dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray]] = {}
    for row in selected:
        stack = row["stack"]
        large, small, _ = _measure_condition(
            camera,
            row["camera"],
            int(stack.pixel_x),
            int(stack.pixel_y),
            large_fraction=a.large_footprint_fraction,
            small_fraction=a.small_footprint_fraction,
            scratch=scratch,
            tag=f"original_V{stack.view_index}_{stack.pixel_x}_{stack.pixel_y}",
        )
        original_measurements[(stack.view_index, stack.pixel_x, stack.pixel_y)] = (large, small)

    transparent = _transparent_material()
    soup_objects, global_map = _build_triangle_soup(arrays, owners, transparent)
    original_meshes = [o for o in bpy.context.scene.objects if o.type == "MESH" and o not in soup_objects]
    original_visibility = {obj: bool(obj.hide_render) for obj in original_meshes}
    for obj in original_meshes:
        obj.hide_render = True
    for obj in soup_objects:
        obj.hide_render = False
    bpy.context.view_layer.update()

    ray_rows = []
    pass_count = fail_count = abstain_count = 0
    max_representation_error = 0.0
    max_footprint_delta = 0.0
    try:
        for ray_index, row in enumerate(selected):
            stack = row["stack"]
            camera_row = row["camera"]
            key = (stack.view_index, stack.pixel_x, stack.pixel_y)
            original_large, original_small = original_measurements[key]

            _set_soup_mode(global_map, mode="FULL")
            full_large, full_small, _ = _measure_condition(
                camera,
                camera_row,
                int(stack.pixel_x),
                int(stack.pixel_y),
                large_fraction=a.large_footprint_fraction,
                small_fraction=a.small_footprint_fraction,
                scratch=scratch,
                tag=f"ray{ray_index}_full",
            )
            representation_error = max(
                _max_carrier_delta(original_large, full_large, carrier_mode),
                _max_carrier_delta(original_small, full_small, carrier_mode),
            )
            max_representation_error = max(max_representation_error, representation_error)

            standalone_alpha = []
            standalone_rgb = []
            removal_alpha = []
            removal_rgb = []
            condition_deltas = [
                _max_carrier_delta(full_large, full_small, carrier_mode),
                _max_carrier_delta(original_large, original_small, carrier_mode),
            ]
            intervention_rows = []
            for hit_index, hit in enumerate(stack.hits):
                face_index = int(hit.face_index)
                _set_soup_mode(global_map, mode="STANDALONE", face_index=face_index)
                stand_large, stand_small, _ = _measure_condition(
                    camera,
                    camera_row,
                    int(stack.pixel_x),
                    int(stack.pixel_y),
                    large_fraction=a.large_footprint_fraction,
                    small_fraction=a.small_footprint_fraction,
                    scratch=scratch,
                    tag=f"ray{ray_index}_hit{hit_index}_standalone",
                )
                stand_delta = _max_carrier_delta(stand_large, stand_small, carrier_mode)
                condition_deltas.append(stand_delta)
                sa, srgb = _premultiplied(stand_small, carrier_mode)
                standalone_alpha.append(sa)
                standalone_rgb.append(srgb)

                _set_soup_mode(global_map, mode="REMOVAL", face_index=face_index)
                remove_large, remove_small, _ = _measure_condition(
                    camera,
                    camera_row,
                    int(stack.pixel_x),
                    int(stack.pixel_y),
                    large_fraction=a.large_footprint_fraction,
                    small_fraction=a.small_footprint_fraction,
                    scratch=scratch,
                    tag=f"ray{ray_index}_hit{hit_index}_removal",
                )
                remove_delta = _max_carrier_delta(remove_large, remove_small, carrier_mode)
                condition_deltas.append(remove_delta)
                ra, rrgb = _premultiplied(remove_small, carrier_mode)
                removal_alpha.append(ra)
                removal_rgb.append(rrgb)
                intervention_rows.append(
                    {
                        "hit_index": hit_index,
                        "face_index": face_index,
                        "barycentric": list(map(float, hit.barycentric)),
                        "depth": float(hit.depth),
                        "standalone_small_rgba": stand_small.tolist(),
                        "standalone_footprint_delta": stand_delta,
                        "removal_small_rgba": remove_small.tolist(),
                        "removal_footprint_delta": remove_delta,
                    }
                )

            _set_soup_mode(global_map, mode="FULL")
            full_alpha, full_rgb = _premultiplied(full_small, carrier_mode)
            footprint_delta = max(condition_deltas)
            max_footprint_delta = max(max_footprint_delta, footprint_delta)
            footprint_converged = (
                footprint_delta <= a.convergence_tolerance
                and representation_error <= a.representation_tolerance
            )
            measurement = qualify_measured_ray_appearance_v1(
                [float(hit.depth) for hit in stack.hits],
                standalone_alpha,
                np.asarray(standalone_rgb, np.float64),
                full_alpha,
                full_rgb,
                removal_alpha,
                np.asarray(removal_rgb, np.float64),
                complete=bool(stack.complete),
                tie_epsilon=float(stack.tie_epsilon),
                measurement_tolerance=a.measurement_tolerance,
                footprint_converged=bool(footprint_converged),
            )
            if measurement.status == PASS:
                pass_count += 1
            elif measurement.status == FAIL_REPLAY:
                fail_count += 1
            else:
                abstain_count += 1
            ray_rows.append(
                _finite_json(
                    {
                        "ray_index": ray_index,
                        "view_index": int(stack.view_index),
                        "pixel_x": int(stack.pixel_x),
                        "pixel_y": int(stack.pixel_y),
                        "resolution": int(camera_row["resolution"]),
                        "hit_count": len(stack.hits),
                        "complete": bool(stack.complete),
                        "has_depth_tie": bool(stack.has_depth_tie),
                        "tie_epsilon_source_units": float(stack.tie_epsilon),
                        "canonical_hits": [
                            {
                                "face_index": int(hit.face_index),
                                "barycentric": list(map(float, hit.barycentric)),
                                "depth": float(hit.depth),
                            }
                            for hit in stack.hits
                        ],
                        "original_packed_small_rgba": original_small.tolist(),
                        "reconstructed_full_small_rgba": full_small.tolist(),
                        "reconstructed_full_to_original_max_error": representation_error,
                        "maximum_footprint_convergence_delta": footprint_delta,
                        "footprint_converged": bool(footprint_converged),
                        "interventions": intervention_rows,
                        "measurement": {
                            **asdict(measurement),
                            "effective_alpha": measurement.effective_alpha,
                            "straight_linear_rgb": measurement.straight_linear_rgb,
                            "incoming_transmittance": measurement.incoming_transmittance,
                            "contribution": measurement.contribution,
                            "predicted_full_premultiplied_linear_rgb": measurement.predicted_full_premultiplied_linear_rgb,
                        },
                    }
                )
            )
    finally:
        for obj, value in original_visibility.items():
            obj.hide_render = value
        bpy.context.view_layer.update()
        for p in scratch.glob("*"):
            p.unlink(missing_ok=True)
        try:
            scratch.rmdir()
        except OSError:
            pass

    sampling_contract = {
        "kind": "HALF_INTEGER_ORTHOGRAPHIC_PIXEL_RAY__SHRINKING_1X1_BOX_FOOTPRINT",
        "candidate_policy": "GEOMETRY_ONLY_FACE_CENTROID__MULTIHIT_FIRST__DISTINCT_VIEW_FIRST",
        "large_footprint_fraction_of_source_pixel": float(a.large_footprint_fraction),
        "small_footprint_fraction_of_source_pixel": float(a.small_footprint_fraction),
        "cycles_samples": int(a.cycles_samples),
        "cycles_seed": 0,
        "pixel_filter": "BOX",
        "convergence_tolerance_premultiplied_linear_rgba": float(a.convergence_tolerance),
        "representation_tolerance_premultiplied_linear_rgba": float(a.representation_tolerance),
        "measurement_replay_tolerance": float(a.measurement_tolerance),
        "tie_epsilon_source_units": float(tie_epsilon),
    }
    payload = {
        "schema": SCHEMA,
        "subject_id": a.subject_id,
        "teacher_scene_sha256": _sha(a.teacher_scene),
        "geometry_npz_sha256": _sha(a.geometry_npz),
        "candidate_sha256": _sha(a.candidate),
        "camera_bundle_sha256": _sha(a.camera_bundle),
        "packed_source_scene_blend_sha256": a.expected_blend_sha256,
        "sampling_contract": sampling_contract,
        "sampling_contract_sha256": _digest(sampling_contract),
        "rgb_carrier_calibration": carrier,
        "material_policy_replay": material_receipts,
        "selected_ray_count": len(ray_rows),
        "pass_ray_count": pass_count,
        "fail_ray_count": fail_count,
        "abstain_ray_count": abstain_count,
        "maximum_reconstructed_full_to_original_error": max_representation_error,
        "maximum_footprint_convergence_delta": max_footprint_delta,
        "rays": ray_rows,
        "candidate_selection_uses_appearance_outcome": False,
        "material_node_values_used_as_per_hit_truth": False,
        "source_lit_rgb_parity_claimed": False,
        "qualified_sampler": False,
        "qualification_minted": False,
        "student_target_allowed": False,
        "training_allowed": False,
        "product_pass_claimed": False,
        "authority": "UNQUALIFIED_MEASURED_RENDERER_INTERVENTIONS",
        "court_status": (
            "FAIL_MEASURED_SOURCE_RAY_APPEARANCE" if fail_count
            else "PASS_MEASURED_SOURCE_RAY_APPEARANCE_CANDIDATE" if pass_count
            else "ABSTAIN_MEASURED_SOURCE_RAY_APPEARANCE"
        ),
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(_finite_json(payload), sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items() if k not in {"rays", "material_policy_replay"}}, sort_keys=True))
    if fail_count:
        raise RuntimeError("SOURCE_RAY_APPEARANCE_INTERVENTION_REPLAY_FAILED")


if __name__ == "__main__":
    main()
