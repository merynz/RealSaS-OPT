#!/usr/bin/env python3
from __future__ import annotations

"""Read authored animation timebase metadata without evaluating mesh geometry.

This probe imports the exact admitted source under the same Blender 4.5 native
FBX/glTF semantics used by the vNext teacher lane, verifies the admitted native
action inventory, and records only scene/action timing metadata.  It does not
sample vertices, render, submit training, or mint Truth Foundry qualification.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import bpy

SCHEMA = "RealSaS.NativeTemporalTimebaseProbe.v1"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _args():
    raw = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--subject-id", required=True)
    parser.add_argument("--observation-admission", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(raw)


def _import_animated(source: Path) -> dict:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    suffix = source.suffix.lower()
    if suffix == ".fbx":
        bpy.ops.import_scene.fbx(
            filepath=str(source), automatic_bone_orientation=False, use_anim=True
        )
        return {
            "operation": "FBX_IMPORT_WITH_NATIVE_ANIMATION",
            "importer_options": {
                "automatic_bone_orientation": False,
                "use_anim": True,
            },
        }
    if suffix in {".gltf", ".glb"}:
        bpy.ops.import_scene.gltf(filepath=str(source), guess_original_bind_pose=True)
        return {
            "operation": "GLTF_IMPORT_WITH_NATIVE_ANIMATION",
            "importer_options": {"guess_original_bind_pose": True},
        }
    raise RuntimeError("TEMPORAL_TIMEBASE_SOURCE_FORMAT_UNSUPPORTED")


def main() -> None:
    args = _args()
    source = args.source.expanduser().resolve()
    admission_path = args.observation_admission.expanduser().resolve()
    output = args.output.expanduser().resolve()

    if not source.is_file() or _sha256(source) != args.expected_source_sha256:
        raise RuntimeError("TEMPORAL_TIMEBASE_EXACT_SOURCE_HASH_REQUIRED")
    admission = json.loads(admission_path.read_text(encoding="utf-8"))
    if admission.get("subject_id") != args.subject_id:
        raise RuntimeError("TEMPORAL_TIMEBASE_SUBJECT_ADMISSION_DRIFT")
    if admission.get("source_sha256") != args.expected_source_sha256:
        raise RuntimeError("TEMPORAL_TIMEBASE_SOURCE_ADMISSION_DRIFT")
    expected_actions = tuple(map(str, admission.get("native_motion_inventory") or ()))
    if not expected_actions:
        raise RuntimeError("TEMPORAL_TIMEBASE_NATIVE_MOTION_INVENTORY_EMPTY")

    import_receipt = _import_animated(source)
    actual = {str(action.name): action for action in bpy.data.actions}
    missing = sorted(set(expected_actions) - set(actual))
    if missing:
        raise RuntimeError("TEMPORAL_TIMEBASE_NATIVE_ACTION_MISSING:" + ",".join(missing))

    scene = bpy.context.scene
    fps_integer = int(scene.render.fps)
    fps_base = float(scene.render.fps_base or 1.0)
    fps = float(fps_integer) / fps_base
    if fps_integer <= 0 or not math.isfinite(fps_base) or fps_base <= 0 or not math.isfinite(fps) or fps <= 0:
        raise RuntimeError("TEMPORAL_TIMEBASE_FPS_INVALID")

    action_rows = []
    for name in expected_actions:
        action = actual[name]
        start, end = map(float, action.frame_range)
        if not math.isfinite(start) or not math.isfinite(end) or end < start:
            raise RuntimeError("TEMPORAL_TIMEBASE_ACTION_RANGE_INVALID:" + name)
        # vNext teacher sampling uses floor(start), ceil(end). MotionSource v2
        # uses an epsilon-protected equivalent for integral authored ranges. We
        # record both source range and the exact vNext teacher local time origin.
        teacher_first_frame = int(math.floor(start))
        teacher_last_frame = int(math.ceil(end))
        action_rows.append(
            {
                "action": name,
                "source_frame_range": [start, end],
                "teacher_first_frame": teacher_first_frame,
                "teacher_last_frame": teacher_last_frame,
                "teacher_time_origin_frame": teacher_first_frame,
                "time_seconds_formula": "(frame-teacher_time_origin_frame)/frames_per_second",
            }
        )

    payload = {
        "schema": SCHEMA,
        "subject_id": args.subject_id,
        "source_sha256": args.expected_source_sha256,
        "observation_admission_sha256": _sha256(admission_path),
        "blender_version": str(bpy.app.version_string),
        "import": import_receipt,
        "scene_timebase": {
            "render_fps": fps_integer,
            "render_fps_base": fps_base,
            "frames_per_second": fps,
            "time_unit": "SECOND",
        },
        "native_motion_inventory_expected": list(expected_actions),
        "native_motion_inventory_observed": list(expected_actions),
        "actions": action_rows,
        "mesh_geometry_evaluated": False,
        "render_executed": False,
        "new_interventions_executed": False,
        "training_submitted": False,
        "qualification_minted": False,
        "authority": "TIMEBASE_METADATA_CANDIDATE_ONLY",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(
        "TEMPORAL_TIMEBASE_PROBE_PASS",
        args.subject_id,
        "fps=", fps,
        "actions=", len(action_rows),
        flush=True,
    )


if __name__ == "__main__":
    main()
