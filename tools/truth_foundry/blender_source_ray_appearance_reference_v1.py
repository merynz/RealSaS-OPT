#!/usr/bin/env python3
from __future__ import annotations

"""Independent source-observation material reference for measured ray Appearance.

This process starts from the exact raw source, uses the admitted bind-rest import
and unlit-preserve-alpha material policy, then renders the *same teacher triangle,
UV and micro-camera footprint* as a packed measured-ray candidate.  Using the
teacher triangle geometry on both sides intentionally removes observation-rest vs
packed-rest geometry drift from this court: the measured variable is the source
material/texture evaluator only.

The reference is scoped to the admitted unlit source-observation policy.  It does
not claim native-lit parity, general sampler qualification, student supervision,
training authority or product activation.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import bpy
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.observation import blender_controlled_8view_v2 as controlled
from tools.observation import blender_controlled_8view_v1 as orbit
from tools.truth_foundry import blender_source_ray_appearance_measure_v1 as measured

SCHEMA = "RealSaS.SourceRayAppearanceReference.v1"


def _args() -> argparse.Namespace:
    raw = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--expected-source-sha256", required=True)
    p.add_argument("--subject-id", required=True)
    p.add_argument("--candidate", required=True, type=Path)
    p.add_argument("--camera-bundle", required=True, type=Path)
    p.add_argument("--appearance", required=True, type=Path)
    p.add_argument("--geometry-npz", required=True, type=Path)
    p.add_argument("--packed-measurement", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    return p.parse_args(raw)


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _material_lookup(appearance: dict) -> dict[tuple[int, int], str]:
    rows = list(appearance.get("materials") or ())
    out: dict[tuple[int, int], str] = {}
    for row in rows:
        key = (int(row["privileged_owner"]), int(row["slot"]))
        name = row.get("material_name")
        if key in out or not isinstance(name, str) or not name:
            raise RuntimeError("SOURCE_RAY_REFERENCE_MATERIAL_METADATA_INVALID")
        out[key] = name
    if not out:
        raise RuntimeError("SOURCE_RAY_REFERENCE_MATERIAL_METADATA_EMPTY")
    return out


def _load_geometry(path: Path):
    names = {
        "rest_vertices_source_world",
        "faces",
        "face_owner_privileged",
        "face_material_privileged",
        "triangle_uv",
        "triangle_uv_valid",
    }
    with np.load(path, allow_pickle=False) as z:
        if not names.issubset(set(z.files)):
            raise RuntimeError("SOURCE_RAY_REFERENCE_GEOMETRY_ARRAYS_MISSING")
        values = {name: np.asarray(z[name]).copy() for name in names}
    v = np.asarray(values["rest_vertices_source_world"], np.float64)
    f = np.asarray(values["faces"], np.int64)
    owner = np.asarray(values["face_owner_privileged"], np.int64)
    slot = np.asarray(values["face_material_privileged"], np.int64)
    uv = np.asarray(values["triangle_uv"], np.float64)
    valid = np.asarray(values["triangle_uv_valid"])
    if (
        v.ndim != 2 or v.shape[1:] != (3,) or not len(v) or not np.isfinite(v).all()
        or f.ndim != 2 or f.shape[1:] != (3,) or not len(f)
        or owner.shape != (len(f),) or slot.shape != (len(f),)
        or uv.shape != (len(f), 3, 2) or valid.shape != (len(f),)
        or valid.dtype != np.bool_
    ):
        raise RuntimeError("SOURCE_RAY_REFERENCE_GEOMETRY_LAYOUT")
    return v, f, owner, slot, uv, valid


def _make_triangle(
    vertices: np.ndarray,
    face: np.ndarray,
    triangle_uv: np.ndarray,
    uv_valid: bool,
    material: bpy.types.Material,
    name: str,
) -> bpy.types.Object:
    mesh = bpy.data.meshes.new(name + "_mesh")
    mesh.from_pydata(vertices[face].tolist(), [], [(0, 1, 2)])
    mesh.update()
    mesh.materials.append(material)
    if uv_valid:
        layer = mesh.uv_layers.new(name="UVMap")
        polygon = mesh.polygons[0]
        for corner, loop_index in enumerate(polygon.loop_indices):
            layer.data[loop_index].uv = tuple(map(float, triangle_uv[corner]))
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def _compare_transform_receipts(sealed: list[dict], actual: list[dict]) -> None:
    by_name = {str(row.get("material") or ""): row for row in actual}
    if len(by_name) != len(actual) or "" in by_name:
        raise RuntimeError("SOURCE_RAY_REFERENCE_TRANSFORM_DUPLICATE")
    for row in sealed:
        name = str(row.get("material") or "")
        got = by_name.get(name)
        if got is None or got.get("operation") != row.get("operation"):
            raise RuntimeError("SOURCE_RAY_REFERENCE_TRANSFORM_OPERATION_DRIFT:" + name)
        if row.get("operation") == "BSDF_LIGHT_RESPONSE_TO_EMISSION__ALPHA_GRAPH_RETAINED":
            if got.get("alpha_policy") != row.get("alpha_policy"):
                raise RuntimeError("SOURCE_RAY_REFERENCE_ALPHA_POLICY_DRIFT:" + name)


def main() -> None:
    a = _args()
    source = a.source.expanduser().resolve()
    if not source.is_file() or _sha(source) != a.expected_source_sha256:
        raise RuntimeError("SOURCE_RAY_REFERENCE_SOURCE_HASH_DRIFT")
    candidate = _json(a.candidate)
    bundle = _json(a.camera_bundle)
    appearance = _json(a.appearance)
    packed = _json(a.packed_measurement)
    if candidate.get("schema") != "RealSaS.Controlled8ViewSourceRestCandidate.v2":
        raise RuntimeError("SOURCE_RAY_REFERENCE_CANDIDATE_SCHEMA")
    if candidate.get("subject_id") != a.subject_id or candidate.get("source_sha256") != a.expected_source_sha256:
        raise RuntimeError("SOURCE_RAY_REFERENCE_CANDIDATE_BINDING")
    if bundle.get("schema") != "RealSaS.CameraProjectionBundle.v1" or _sha(a.camera_bundle) != candidate.get("camera_bundle", {}).get("sha256"):
        raise RuntimeError("SOURCE_RAY_REFERENCE_CAMERA_BINDING")
    if appearance.get("schema") != "RealSaS.SourceAppearanceSamplerInputs.v1" or appearance.get("qualified_sampler") is not False:
        raise RuntimeError("SOURCE_RAY_REFERENCE_APPEARANCE_BOUNDARY")
    if packed.get("schema") != "RealSaS.SourceRayAppearanceMeasurementCandidate.v1" or packed.get("subject_id") != a.subject_id:
        raise RuntimeError("SOURCE_RAY_REFERENCE_PACKED_SCHEMA")
    if packed.get("candidate_sha256") != _sha(a.candidate) or packed.get("camera_bundle_sha256") != _sha(a.camera_bundle):
        raise RuntimeError("SOURCE_RAY_REFERENCE_PACKED_LINEAGE")
    if packed.get("authority") != "UNQUALIFIED_MEASURED_RENDERER_INTERVENTIONS" or packed.get("qualification_minted") is not False:
        raise RuntimeError("SOURCE_RAY_REFERENCE_PACKED_AUTHORITY")

    vertices, faces, face_owner, face_slot, triangle_uv, uv_valid = _load_geometry(a.geometry_npz)
    materials_by_key = _material_lookup(appearance)
    cameras = {int(row["view_index"]): row for row in bundle.get("cameras") or ()}
    if set(cameras) != set(range(8)):
        raise RuntimeError("SOURCE_RAY_REFERENCE_CAMERA_SET")

    # Exact observation-lane import. The helper performs a factory reset.
    work = a.output.parent / (a.output.stem + ".source_import")
    work.mkdir(parents=True, exist_ok=True)
    import_receipt = controlled.import_rest(source, work)
    meshes, _points, _by_object = orbit.full_subject()
    actual_transform = controlled.preserve_alpha_unlit(meshes)
    _compare_transform_receipts(list(candidate.get("material_transform") or ()), actual_transform)

    source_materials = {m.name: m for m in bpy.data.materials if m is not None}
    needed_names = set()
    hit_requests = []
    for ray in packed.get("rays") or ():
        if ray.get("measurement", {}).get("status") != "PASS_MEASURED_RAY_APPEARANCE_FACTORIZATION":
            raise RuntimeError("SOURCE_RAY_REFERENCE_NONPASS_PACKED_RAY")
        for hit_index, hit in enumerate(ray.get("canonical_hits") or ()):
            face_index = int(hit["face_index"])
            if not 0 <= face_index < len(faces):
                raise RuntimeError("SOURCE_RAY_REFERENCE_FACE_RANGE")
            key = (int(face_owner[face_index]), int(face_slot[face_index]))
            material_name = materials_by_key.get(key)
            if material_name is None:
                raise RuntimeError("SOURCE_RAY_REFERENCE_MATERIAL_KEY_MISSING")
            needed_names.add(material_name)
            hit_requests.append((ray, hit_index, face_index, material_name))
    missing = sorted(name for name in needed_names if name not in source_materials)
    if missing:
        raise RuntimeError("SOURCE_RAY_REFERENCE_SOURCE_MATERIAL_MISSING:" + repr(missing))

    # Hide imported geometry; only isolated exact teacher triangles are rendered.
    imported_visibility = {obj: bool(obj.hide_render) for obj in bpy.context.scene.objects if obj.type == "MESH"}
    for obj in imported_visibility:
        obj.hide_render = True

    measured._configure_measurement(int(packed["sampling_contract"]["cycles_samples"]))
    camera = measured._make_camera()
    scratch = a.output.parent / (a.output.stem + ".scratch")
    scratch.mkdir(parents=True, exist_ok=True)
    first_ray = packed["rays"][0]
    first_camera = cameras[int(first_ray["view_index"])]
    measured._set_micro_camera(
        camera,
        first_camera,
        int(first_ray["pixel_x"]),
        int(first_ray["pixel_y"]),
        float(packed["sampling_contract"]["small_centre_pixel_footprint_fraction_of_source_pixel"]),
    )
    carrier = measured._calibrate_rgb_carrier(camera, first_camera, scratch)
    carrier_mode = str(carrier["mode"])
    if carrier_mode != packed.get("rgb_carrier_calibration", {}).get("mode"):
        raise RuntimeError("SOURCE_RAY_REFERENCE_RGB_CARRIER_MODE_DRIFT")

    rows = []
    try:
        for ray, hit_index, face_index, material_name in hit_requests:
            material = source_materials[material_name]
            obj = _make_triangle(
                vertices,
                faces[face_index],
                triangle_uv[face_index],
                bool(uv_valid[face_index]),
                material,
                f"RealSaS_SourceReference_F{face_index}_H{hit_index}",
            )
            bpy.context.view_layer.update()
            camera_row = cameras[int(ray["view_index"])]
            try:
                large, small = measured._measure_condition(
                    camera,
                    camera_row,
                    int(ray["pixel_x"]),
                    int(ray["pixel_y"]),
                    large_fraction=float(packed["sampling_contract"]["large_centre_pixel_footprint_fraction_of_source_pixel"]),
                    small_fraction=float(packed["sampling_contract"]["small_centre_pixel_footprint_fraction_of_source_pixel"]),
                    scratch=scratch,
                    tag=f"V{ray['view_index']}_P{ray['pixel_x']}_{ray['pixel_y']}_F{face_index}",
                )
            finally:
                mesh = obj.data
                bpy.data.objects.remove(obj, do_unlink=True)
                bpy.data.meshes.remove(mesh, do_unlink=True)
                bpy.context.view_layer.update()
            alpha, premul = measured._premultiplied(small, carrier_mode)
            footprint_delta = measured._max_carrier_delta(large, small, carrier_mode)
            rows.append(
                {
                    "ray_index": int(ray["ray_index"]),
                    "view_index": int(ray["view_index"]),
                    "pixel_x": int(ray["pixel_x"]),
                    "pixel_y": int(ray["pixel_y"]),
                    "hit_index": int(hit_index),
                    "face_index": int(face_index),
                    "material_name_privileged_reference_only": material_name,
                    "triangle_uv_valid": bool(uv_valid[face_index]),
                    "small_rgba_carrier": small.tolist(),
                    "effective_alpha": float(alpha),
                    "premultiplied_linear_rgb": premul.tolist(),
                    "footprint_convergence_delta": float(footprint_delta),
                }
            )
    finally:
        for obj, value in imported_visibility.items():
            if obj.name in bpy.data.objects:
                obj.hide_render = value
        for path in scratch.glob("*"):
            path.unlink(missing_ok=True)
        try:
            scratch.rmdir()
        except OSError:
            pass

    payload = {
        "schema": SCHEMA,
        "subject_id": a.subject_id,
        "source_sha256": a.expected_source_sha256,
        "candidate_sha256": _sha(a.candidate),
        "camera_bundle_sha256": _sha(a.camera_bundle),
        "appearance_sha256": _sha(a.appearance),
        "geometry_npz_sha256": _sha(a.geometry_npz),
        "packed_measurement_sha256": _sha(a.packed_measurement),
        "source_import": import_receipt,
        "source_material_transform": actual_transform,
        "rgb_carrier_calibration": carrier,
        "equivalence_scope": "ADMITTED_UNLIT_SOURCE_OBSERVATION_MATERIAL_POLICY__SAME_TEACHER_TRIANGLE_UV_AND_MICRO_CAMERA",
        "geometry_lane_equivalence_required": False,
        "native_lit_parity_claimed": False,
        "hit_count": len(rows),
        "hits": rows,
        "qualified_sampler": False,
        "qualification_minted": False,
        "student_target_allowed": False,
        "training_allowed": False,
        "product_pass_claimed": False,
        "authority": "SOURCE_OBSERVATION_APPEARANCE_REFERENCE_ONLY",
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items() if k not in {"hits", "source_material_transform", "source_import"}}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
