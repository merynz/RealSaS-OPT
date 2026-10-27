#!/usr/bin/env python3
from __future__ import annotations

"""Read authored animation timebase metadata without evaluating mesh geometry.

This probe imports the exact admitted source under the same Blender 4.5 native
FBX/glTF semantics used by the vNext teacher lane, then binds importer-local
action names back to the exact sealed teacher action identity.  FBX may strip an
`Armature|` prefix during import, so that normalization is accepted only when it
is uniquely witnessed by the sealed teacher receipt's `(action, armature)` rows.

The probe does not sample vertices, render, submit training, or mint Truth
Foundry qualification.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import bpy

SCHEMA = "RealSaS.NativeTemporalTimebaseProbe.v1"
TEACHER_SCHEMA = "RealSaS.IRISVNextAuthoredTeacherExport.v1"


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
    parser.add_argument("--teacher-receipt", required=True, type=Path)
    parser.add_argument("--expected-teacher-receipt-sha256", required=True)
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


def _sealed_action_armature(receipt: dict, action: str) -> str:
    armatures = {
        str(row.get("armature") or "")
        for row in receipt.get("sampled_frames") or ()
        if str(row.get("action") or "") == action
    }
    armatures.discard("")
    if len(armatures) != 1:
        raise RuntimeError("TEMPORAL_TIMEBASE_SEALED_ACTION_ARMATURE_NOT_UNIQUE:" + action)
    return next(iter(armatures))


def _bind_actions(expected_actions: tuple[str, ...], actual: dict[str, object], receipt: dict):
    """Return exact teacher-identity -> Blender action binding.

    Exact name identity wins.  The only accepted normalization is the concrete
    FBX importer transformation witnessed by the sealed receipt:

        `<sealed armature>|<Blender action name>` == `<teacher action identity>`

    No suffix similarity, fuzzy matching, semantic aliases, or action ordering is
    used.  Ambiguous/missing bindings fail closed.
    """
    bound = {}
    used_actual = set()
    for teacher_name in expected_actions:
        if teacher_name in actual:
            blender_name = teacher_name
            armature = _sealed_action_armature(receipt, teacher_name)
            kind = "EXACT_ACTION_NAME"
        else:
            armature = _sealed_action_armature(receipt, teacher_name)
            candidates = [
                name for name in actual
                if f"{armature}|{name}" == teacher_name
            ]
            if len(candidates) != 1:
                raise RuntimeError(
                    "TEMPORAL_TIMEBASE_ACTION_IDENTITY_UNBOUND:"
                    + teacher_name
                    + ":candidates="
                    + ",".join(sorted(candidates))
                )
            blender_name = candidates[0]
            kind = "SEALED_RECEIPT_ARMATURE_PREFIX_NORMALIZATION"
        if blender_name in used_actual:
            raise RuntimeError("TEMPORAL_TIMEBASE_ACTION_IDENTITY_COLLISION:" + blender_name)
        used_actual.add(blender_name)
        bound[teacher_name] = {
            "teacher_action_identity": teacher_name,
            "blender_action_name": blender_name,
            "sealed_receipt_armature": armature,
            "identity_binding": kind,
        }
    return bound


def main() -> None:
    args = _args()
    source = args.source.expanduser().resolve()
    admission_path = args.observation_admission.expanduser().resolve()
    receipt_path = args.teacher_receipt.expanduser().resolve()
    output = args.output.expanduser().resolve()

    if not source.is_file() or _sha256(source) != args.expected_source_sha256:
        raise RuntimeError("TEMPORAL_TIMEBASE_EXACT_SOURCE_HASH_REQUIRED")
    if not receipt_path.is_file() or _sha256(receipt_path) != args.expected_teacher_receipt_sha256:
        raise RuntimeError("TEMPORAL_TIMEBASE_EXACT_TEACHER_RECEIPT_REQUIRED")

    admission = json.loads(admission_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if admission.get("subject_id") != args.subject_id:
        raise RuntimeError("TEMPORAL_TIMEBASE_SUBJECT_ADMISSION_DRIFT")
    if admission.get("source_sha256") != args.expected_source_sha256:
        raise RuntimeError("TEMPORAL_TIMEBASE_SOURCE_ADMISSION_DRIFT")
    if receipt.get("schema") != TEACHER_SCHEMA or receipt.get("subject_id") != args.subject_id:
        raise RuntimeError("TEMPORAL_TIMEBASE_TEACHER_RECEIPT_DRIFT")
    if receipt.get("teacher_only") is not True or receipt.get("canonical_M_G_W_minted") is not False:
        raise RuntimeError("TEMPORAL_TIMEBASE_TEACHER_AUTHORITY_BOUNDARY")

    expected_actions = tuple(map(str, admission.get("native_motion_inventory") or ()))
    if not expected_actions:
        raise RuntimeError("TEMPORAL_TIMEBASE_NATIVE_MOTION_INVENTORY_EMPTY")
    if tuple(map(str, receipt.get("native_motion_inventory_exported") or ())) != expected_actions:
        raise RuntimeError("TEMPORAL_TIMEBASE_TEACHER_ACTION_INVENTORY_DRIFT")

    import_receipt = _import_animated(source)
    actual = {str(action.name): action for action in bpy.data.actions}
    action_binding = _bind_actions(expected_actions, actual, receipt)

    scene = bpy.context.scene
    fps_integer = int(scene.render.fps)
    fps_base = float(scene.render.fps_base or 1.0)
    fps = float(fps_integer) / fps_base
    if fps_integer <= 0 or not math.isfinite(fps_base) or fps_base <= 0 or not math.isfinite(fps) or fps <= 0:
        raise RuntimeError("TEMPORAL_TIMEBASE_FPS_INVALID")

    action_rows = []
    for teacher_name in expected_actions:
        binding = action_binding[teacher_name]
        action = actual[binding["blender_action_name"]]
        start, end = map(float, action.frame_range)
        if not math.isfinite(start) or not math.isfinite(end) or end < start:
            raise RuntimeError("TEMPORAL_TIMEBASE_ACTION_RANGE_INVALID:" + teacher_name)
        teacher_first_frame = int(math.floor(start))
        teacher_last_frame = int(math.ceil(end))
        action_rows.append(
            {
                **binding,
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
        "teacher_receipt_sha256": args.expected_teacher_receipt_sha256,
        "blender_version": str(bpy.app.version_string),
        "import": import_receipt,
        "scene_timebase": {
            "render_fps": fps_integer,
            "render_fps_base": fps_base,
            "frames_per_second": fps,
            "time_unit": "SECOND",
        },
        "native_motion_inventory_teacher_identity": list(expected_actions),
        "blender_action_inventory_observed": sorted(actual),
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
    normalized = sum(
        row["identity_binding"] != "EXACT_ACTION_NAME" for row in action_rows
    )
    print(
        "TEMPORAL_TIMEBASE_PROBE_PASS",
        args.subject_id,
        "fps=", fps,
        "actions=", len(action_rows),
        "normalized_action_identities=", normalized,
        flush=True,
    )


if __name__ == "__main__":
    main()
