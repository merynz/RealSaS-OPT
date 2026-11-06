#!/usr/bin/env python3
"""Original GLB + frozen V4 teacher source-world face/UV binding, Blender v0.

Private source-backed oracle. The same imported Blender mesh/triangle order
frozen by native teacher V4 is checked EXACTLY against original teacher bytes.
The glTF top-left triangle UV multiset is independently recomputed from raw
source GLB bytes, not from Blender. Multi-primitive sources are supported
without guessing source primitive -> Blender object naming/order.

NO source artist ownership, no student inference/training authority.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

import bpy
import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.iris_vnext import blender_authored_teacher_export_v1 as base
from tools.iris_vnext import blender_authored_teacher_export_v4 as v4
from tools.corpus.qualify_source_uv_inventory_v0 import (
    original_gltf_uv_inventory, compare_uv_inventories,
)


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def _require(ok, reason):
    if not ok:
        raise ValueError("BLENDER_SOURCE_UV_" + reason)


def _blender_face_corner_uv(meshes):
    deps = bpy.context.evaluated_depsgraph_get()
    uv = []
    owner_rows = []
    for owner_index, obj in enumerate(meshes):
        evaluated = obj.evaluated_get(deps)
        mesh = evaluated.to_mesh()
        try:
            mesh.calc_loop_triangles()
            layers = mesh.uv_layers
            _require(len(layers) >= 1, "MISSING_SOURCE_UV_LAYER")
            # glTF TEXCOORD_0 enters Blender's first UV map. This selection
            # is independently checked against the raw GLB's TEXCOORD_0.
            layer = layers[0]
            for triangle in mesh.loop_triangles:
                corner_uv = []
                for loop_index in triangle.loops:
                    native_uv = layer.data[int(loop_index)].uv
                    # Blender UV origin: lower-left; glTF + PNG: top-left.
                    corner_uv.append((float(native_uv[0]), 1.0-float(native_uv[1])))
                uv.append(corner_uv)
            owner_rows.append({
                "anonymous_mesh_order": owner_index,
                "vertex_count": len(mesh.vertices),
                "triangle_count": len(mesh.loop_triangles),
                "uv_layer_count": len(layers),
            })
        finally:
            evaluated.to_mesh_clear()
    a = np.asarray(uv, dtype=np.float64)
    _require(a.ndim == 3 and a.shape[1:] == (3, 2),
             "IMPORTED_FACE_CORNER_SHAPE")
    return a, owner_rows


def inspect(*, source: Path, teacher: Path, expected_source_sha: str,
            expected_teacher_sha: str):
    _require(digest(source) == expected_source_sha and
             digest(teacher) == expected_teacher_sha,
             "ORIGINAL_BYTES_SHA_DRIFT")
    report = {
        "schema": "RealSaS.OriginalTeacherNativeUVAlignment.v0",
        "original_glb_sha256": expected_source_sha,
        "original_teacher_sha256": expected_teacher_sha,
        "blender_version": bpy.app.version_string,
        "source_to_teacher_face_uv_binding": "UNKNOWN",
        "reason": "NOT_EVALUATED",
        "source_gltf_primitive_count": None,
        "source_gltf_triangle_count": None,
        "teacher_face_count": None,
        "imported_face_count": None,
        "source_uv_comparison": None,
        "max_rest_source_world_vertex_error": None,
        "source_artist_owner": "UNKNOWN",
        "student_inference_input_authorized": False,
        "compiled_puppet_physical_parity": "UNKNOWN",
        "graphics_owner_authorized": False,
        "materialization_authorized": False,
        "training_authorized": False,
        "iris_training_authorized": False,
        "tessa_training_authorized": False,
    }
    _require(tuple(bpy.app.version) >= (4, 5, 0),
             "BLENDER_VERSION_TOO_OLD")
    with np.load(teacher, allow_pickle=False) as original:
        required = {"faces", "face_owner", "rest_vertices_source_world",
                    "posed_vertices_source_world"}
        if not required <= set(original.files):
            report["reason"] = "NATIVE_TEACHER_REQUIRED_ARRAY_MISSING"
            return report, None
        truth_faces = np.asarray(original["faces"], dtype=np.int64)
        truth_owner = np.asarray(original["face_owner"], dtype=np.int32)
        truth_rest = np.asarray(original["rest_vertices_source_world"], dtype=np.float64)
        poses = np.asarray(original["posed_vertices_source_world"], dtype=np.float64)
    report["teacher_face_count"] = int(len(truth_faces))
    if not (truth_faces.ndim == 2 and truth_faces.shape[1] == 3
            and truth_rest.ndim == 2 and truth_rest.shape[1] == 3
            and poses.ndim == 3 and poses.shape[1:] == truth_rest.shape):
        report["reason"] = "TEACHER_TOPOLOGY_OR_POSES_INVALID"
        return report, None

    raw_uv, primitive_rows = original_gltf_uv_inventory(source)
    report["source_gltf_primitive_count"] = len(primitive_rows)
    report["source_gltf_triangle_count"] = int(sum(raw_uv.values()))
    base._import_animated(source)
    meshes = base._mesh_objects()
    base._neutralize_animation()
    rest = v4._extract_snapshot(meshes, freeze_triangulation=True)
    report["imported_face_count"] = int(len(rest["faces"]))
    if rest["faces"].shape != truth_faces.shape or not np.array_equal(
            np.asarray(rest["faces"], dtype=np.int64), truth_faces):
        report["reason"] = "BLENDER_NATIVE_TEACHER_FACE_ORDER_DRIFT"
        return report, None
    if not np.array_equal(np.asarray(rest["face_owner"], dtype=np.int32), truth_owner):
        report["reason"] = "BLENDER_NATIVE_TEACHER_ANONYMOUS_OWNER_ORDER_DRIFT"
        return report, None
    if rest["vertices"].shape != truth_rest.shape:
        report["reason"] = "BLENDER_NATIVE_TEACHER_VERTEX_SHAPE_DRIFT"
        return report, None
    err = float(np.max(np.abs(np.asarray(rest["vertices"], dtype=np.float64) - truth_rest)))
    report["max_rest_source_world_vertex_error"] = err
    tolerance = 1e-6 * max(1., float(np.linalg.norm(np.ptp(truth_rest, axis=0))))
    if not np.isfinite(err) or err > tolerance:
        report["reason"] = "BLENDER_NATIVE_TEACHER_REST_GEOMETRY_DRIFT"
        return report, None

    imported_uv, owner_rows = _blender_face_corner_uv(meshes)
    if len(imported_uv) != len(truth_faces):
        report["reason"] = "UV_FACE_COUNT_DRIFT"
        return report, None
    compare = compare_uv_inventories(raw_uv, imported_uv)
    report["source_uv_comparison"] = compare
    if not compare["matched"]:
        report["reason"] = "RAW_GLTF_TO_NATIVE_BLENDER_UV_TRIANGLES_UNMATCHED"
        return report, None

    # The order of face-corner UVs uses EXACTLY v4's evaluated mesh object
    # order and loop-triangle order. All UV values are sourced from the same
    # untouched native GLB, not inferred from teacher pixels or target render.
    report["source_to_teacher_face_uv_binding"] = (
        "VERIFIED_NATIVE_BLENDER_FACE_ORDER_AND_RAW_GLTF_UV_MULTISET")
    report["reason"] = "NATIVE_SOURCE_ADDRESS_VERIFIED__NOT_PUPPET_RENDER_PARITY"
    report["imported_anonymous_mesh_rows"] = owner_rows
    report["face_uv_array_sha256"] = hashlib.sha256(
        np.ascontiguousarray(imported_uv.astype("<f8")).tobytes()
    ).hexdigest()
    return report, imported_uv


def main():
    args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    for name in ("source", "teacher", "expected-source-sha",
                 "expected-teacher-sha", "report", "private-face-uv"):
        p.add_argument("--" + name, required=True)
    a = p.parse_args(args)
    report, uv = inspect(
        source=Path(a.source), teacher=Path(a.teacher),
        expected_source_sha=a.expected_source_sha,
        expected_teacher_sha=a.expected_teacher_sha,
    )
    output = Path(a.report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True,
                                  allow_nan=False) + "\n")
    if uv is not None:
        dest = Path(a.private_face_uv)
        dest.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dest, face_uv=uv)
    print(json.dumps({
        "status": report["source_to_teacher_face_uv_binding"],
        "reason": report["reason"],
        "source_glb_primitive_count": report["source_gltf_primitive_count"],
        "source_glb_triangle_count": report["source_gltf_triangle_count"],
        "teacher_face_count": report["teacher_face_count"],
        "uv_comparison": report["source_uv_comparison"],
        "artist_owner": "UNKNOWN",
        "training_authorized": False,
    }, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
