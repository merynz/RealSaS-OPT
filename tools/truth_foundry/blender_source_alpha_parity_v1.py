#!/usr/bin/env python3
from __future__ import annotations

"""Measure pixel alpha parity for the controlled source-unlit transform.

The exact packed source evaluator scene is rendered twice under the same sealed
Cycles camera/sampling policy: first with its native material graph, then after
``preserve_alpha_unlit``. Only alpha is compared. RGB/source-lighting parity is
outside this court.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import bpy
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.iris.truth.source_alpha_parity_v1 import measure_alpha_transport_parity_v1
from tools.observation import blender_controlled_8view_v1 as orbit
from tools.observation import blender_controlled_8view_v2 as controlled

SCHEMA = "RealSaS.SourceAlphaParityCourtCandidate.v1"


def _args() -> argparse.Namespace:
    raw = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--subject-id", required=True)
    p.add_argument("--teacher-scene", required=True, type=Path)
    p.add_argument("--expected-blend-sha256", required=True)
    p.add_argument("--resolution", type=int, default=96)
    p.add_argument("--frame-scale", type=float, default=1.12)
    p.add_argument("--tolerance", type=float, default=1.0e-6)
    p.add_argument("--output", required=True, type=Path)
    return p.parse_args(raw)


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _render_alpha(camera, center, half: float, resolution: int) -> list[np.ndarray]:
    distance = max(4.0 * half, 1.0)
    rows = []
    for _index, yaw in enumerate(orbit.VIEW_YAW_DEG):
        radial, forward, _right, _up = orbit.basis(yaw)
        camera.location = center + radial * distance
        camera.rotation_euler = forward.to_track_quat("-Z", "Y").to_euler()
        bpy.context.view_layer.update()
        bpy.ops.render.render(write_still=False)
        rendered = bpy.data.images.get("Render Result")
        if rendered is None or tuple(rendered.size) != (resolution, resolution):
            raise RuntimeError("SOURCE_ALPHA_PARITY_RENDER_RESULT_LAYOUT")
        pixels = np.empty(resolution * resolution * 4, dtype=np.float32)
        rendered.pixels.foreach_get(pixels)
        alpha = pixels.reshape(resolution, resolution, 4)[:, :, 3].astype(np.float64, copy=True)
        if not np.isfinite(alpha).all() or np.any((alpha < 0) | (alpha > 1)):
            raise RuntimeError("SOURCE_ALPHA_PARITY_RENDER_ALPHA_INVALID")
        rows.append(alpha)
    return rows


def main() -> None:
    a = _args()
    if a.resolution < 16 or not np.isfinite(a.frame_scale) or a.frame_scale <= 1.0:
        raise RuntimeError("SOURCE_ALPHA_PARITY_RENDER_POLICY")
    teacher = json.loads(a.teacher_scene.read_text(encoding="utf-8"))
    if teacher.get("schema") != "RealSaS.TeacherSceneIR.v1" or teacher.get("subject_key") != a.subject_id:
        raise RuntimeError("SOURCE_ALPHA_PARITY_TEACHER_BINDING")
    if teacher.get("qualification_minted") is not False or teacher.get("authority") != "UNQUALIFIED_DERIVED_MEASUREMENTS":
        raise RuntimeError("SOURCE_ALPHA_PARITY_TEACHER_AUTHORITY")
    current_blend = Path(bpy.data.filepath).resolve()
    if not current_blend.is_file() or _sha(current_blend) != a.expected_blend_sha256:
        raise RuntimeError("SOURCE_ALPHA_PARITY_BLEND_HASH_DRIFT")

    meshes, points, _by_object = orbit.full_subject()
    center, half, _frame = orbit.solve_frame(points, a.frame_scale)
    controlled.configure(a.resolution)

    data = bpy.data.cameras.new("RealSaS_SourceAlphaParity_Camera")
    data.type = "ORTHO"
    data.ortho_scale = 2.0 * half
    distance = max(4.0 * half, 1.0)
    data.clip_start = max(1.0e-6, distance - 2.0 * half)
    data.clip_end = distance + 2.0 * half
    camera = bpy.data.objects.new(data.name, data)
    bpy.context.scene.collection.objects.link(camera)
    bpy.context.scene.camera = camera

    native = _render_alpha(camera, center, half, a.resolution)
    transform_receipts = controlled.preserve_alpha_unlit(meshes)
    transformed = _render_alpha(camera, center, half, a.resolution)

    views = []
    all_valid = True
    total_union = 0
    max_error = 0.0
    total_mismatch = 0
    for index, (label, left, right) in enumerate(zip(orbit.VIEW_ORDER, native, transformed)):
        measurement = measure_alpha_transport_parity_v1(left, right, tolerance=a.tolerance)
        all_valid = all_valid and measurement.valid
        total_union += measurement.union_foreground_sample_count
        max_error = max(max_error, measurement.maximum_absolute_error)
        total_mismatch += measurement.mismatch_count
        views.append({"view_index": index, "direction_label": label, **asdict(measurement)})

    payload = {
        "schema": SCHEMA,
        "subject_id": a.subject_id,
        "teacher_scene_sha256": _sha(a.teacher_scene),
        "source_scene_blend_sha256": a.expected_blend_sha256,
        "blender_version": str(bpy.app.version_string),
        "measurement_renderer": {
            "engine": "CYCLES", "device": "CPU", "samples": 16, "seed": 0,
            "denoising": False, "pixel_filter": "BOX", "filter_width": 1.0,
            "film_transparent": True, "view_transform": "Standard", "look": "None",
            "exposure": 0.0, "gamma": 1.0, "resolution": a.resolution,
            "view_count": 8, "frame_scale": a.frame_scale,
        },
        "alpha_tolerance": float(a.tolerance),
        "material_transform_receipts": transform_receipts,
        "union_foreground_sample_count": total_union,
        "maximum_absolute_alpha_error": max_error,
        "mismatch_count": total_mismatch,
        "all_views_pass": bool(all_valid),
        "views": views,
        "rgb_parity_claimed": False,
        "source_lighting_parity_claimed": False,
        "sampler_qualified": False,
        "qualification_minted": False,
        "student_target_allowed": False,
        "training_allowed": False,
        "court_status": "PASS_ALPHA_TRANSPORT_PARITY" if all_valid else "FAIL_ALPHA_TRANSPORT_DRIFT",
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items() if k not in {"views", "material_transform_receipts"}}, sort_keys=True))
    if not all_valid:
        raise RuntimeError("SOURCE_ALPHA_PARITY_COURT_FAILED")


if __name__ == "__main__":
    main()
