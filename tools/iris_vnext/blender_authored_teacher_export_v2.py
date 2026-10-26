from __future__ import annotations

"""Blender 4.4+ transport adapter for IRIS vNext authored teacher export.

Scientific semantics are unchanged from ``blender_authored_teacher_export_v1``.
V2 only makes the source-take transport exact across importers that strip an FBX
``Armature|`` prefix, and explicitly selects the Blender 4.4+ Action slot.
Requested/native inventory names remain the teacher identity written to receipts.
"""

import json
from pathlib import Path
import sys

import bpy
import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.iris_vnext import blender_authored_teacher_export_v1 as base


def _action_by_requested_take(requested: str):
    exact = [action for action in bpy.data.actions if action.name == requested]
    if len(exact) == 1:
        return exact[0]
    short = requested.split("|")[-1]
    matches = [
        action
        for action in bpy.data.actions
        if action.name == short or action.name.endswith("|" + short)
    ]
    if len(matches) != 1:
        raise RuntimeError(
            "VNEXT_TEACHER_TAKE_NOT_UNIQUE:"
            + requested
            + ":available="
            + ",".join(sorted(action.name for action in bpy.data.actions))
        )
    return matches[0]


def _armature_for_action(action):
    armatures = tuple(
        sorted(
            (obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"),
            key=lambda obj: obj.name,
        )
    )
    if not armatures:
        raise RuntimeError("VNEXT_TEACHER_ACTION_WITHOUT_ARMATURE")

    # Prefer the same pose-bone evidence used by V1 where Blender exposes legacy
    # FCurves.  Slotted Actions may not expose that collection directly.
    try:
        required = base._pose_bones_used(action)
    except (AttributeError, TypeError, RuntimeError):
        required = set()
    if required:
        compatible = [
            obj for obj in armatures if required.issubset(set(obj.pose.bones.keys()))
        ]
        if len(compatible) == 1:
            return compatible[0]
        if len(compatible) > 1:
            raise RuntimeError(
                "VNEXT_TEACHER_ACTION_OWNER_AMBIGUOUS:"
                + action.name
                + ":"
                + ",".join(obj.name for obj in compatible)
            )

    # Imported NLA ownership is strong evidence when present.
    nla_owners = []
    for obj in armatures:
        animation = obj.animation_data
        if animation is None:
            continue
        if animation.action == action or any(
            strip.action == action
            for track in animation.nla_tracks
            for strip in track.strips
        ):
            nla_owners.append(obj)
    if len(nla_owners) == 1:
        return nla_owners[0]
    if len(nla_owners) > 1:
        raise RuntimeError(
            "VNEXT_TEACHER_ACTION_OWNER_AMBIGUOUS:"
            + action.name
            + ":"
            + ",".join(obj.name for obj in nla_owners)
        )

    # With exactly one imported armature there is no competing animation owner.
    if len(armatures) == 1:
        return armatures[0]
    raise RuntimeError(
        "VNEXT_TEACHER_ACTION_OWNER_AMBIGUOUS:"
        + action.name
        + ":"
        + ",".join(obj.name for obj in armatures)
    )


def _select_action_slot(armature, action) -> dict:
    data = armature.animation_data
    if not hasattr(data, "action_suitable_slots"):
        return {
            "identifier": "LEGACY_ACTION_NO_SLOT_API",
            "name": str(armature.name),
            "target_id_type": "OBJECT",
        }
    suitable = list(data.action_suitable_slots)
    by_user = [slot for slot in suitable if armature in tuple(slot.users())]
    by_name = [
        slot for slot in suitable if str(slot.name_display) == str(armature.name)
    ]
    if len(by_user) == 1:
        slot = by_user[0]
    elif len(by_name) == 1:
        slot = by_name[0]
    elif len(suitable) == 1:
        slot = suitable[0]
    else:
        detail = ",".join(
            f"{slot.identifier}:{slot.name_display}:{slot.target_id_type}"
            for slot in suitable
        )
        raise RuntimeError(
            "VNEXT_TEACHER_ARMATURE_SLOT_NOT_UNIQUE:"
            + action.name
            + ":armature="
            + armature.name
            + ":slots="
            + detail
        )
    data.action_slot = slot
    return {
        "identifier": str(slot.identifier),
        "name": str(slot.name_display),
        "target_id_type": str(slot.target_id_type),
    }


def _activate_action(action):
    for owner in base._animation_owners():
        animation = getattr(owner, "animation_data", None)
        if animation is None:
            continue
        animation.action = None
        if hasattr(animation, "use_nla"):
            animation.use_nla = False
        for track in animation.nla_tracks:
            track.mute = True
    armature = _armature_for_action(action)
    armature.data.pose_position = "POSE"
    if armature.animation_data is None:
        armature.animation_data_create()
    if hasattr(armature.animation_data, "use_nla"):
        armature.animation_data.use_nla = False
    armature.animation_data.action = action
    slot = _select_action_slot(armature, action)
    bpy.context.view_layer.update()
    return armature.name, slot


def main():
    args = base._args()
    source = args.source.expanduser().resolve()
    admission_path = args.observation_admission.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not source.is_file() or base._sha256(source) != args.expected_source_sha256:
        raise RuntimeError("VNEXT_TEACHER_EXACT_SOURCE_HASH_REQUIRED")
    admission = json.loads(admission_path.read_text(encoding="utf-8"))
    if admission.get("subject_id") != args.subject_id:
        raise RuntimeError("VNEXT_TEACHER_SUBJECT_ADMISSION_DRIFT")
    if admission.get("source_sha256") != args.expected_source_sha256:
        raise RuntimeError("VNEXT_TEACHER_SOURCE_ADMISSION_HASH_DRIFT")
    expected_actions = tuple(map(str, admission.get("native_motion_inventory") or ()))
    if not expected_actions:
        raise RuntimeError("VNEXT_TEACHER_NATIVE_MOTION_INVENTORY_EMPTY")

    import_receipt = base._import_animated(source)
    resolved_actions = {
        requested: _action_by_requested_take(requested)
        for requested in expected_actions
    }
    if len({id(action) for action in resolved_actions.values()}) != len(resolved_actions):
        raise RuntimeError("VNEXT_TEACHER_TAKE_ALIAS_COLLISION")

    mesh_objects = base._mesh_objects()
    base._neutralize_animation()
    rest = base._extract_world_mesh(mesh_objects)
    if len(rest["vertices"]) == 0 or len(rest["faces"]) == 0:
        raise RuntimeError("VNEXT_TEACHER_REST_MESH_EMPTY")

    posed = []
    frame_rows = []
    action_resolution = []
    for requested_name in expected_actions:
        action = resolved_actions[requested_name]
        armature_name, slot = _activate_action(action)
        action_resolution.append(
            {
                "requested_take": requested_name,
                "imported_action": str(action.name),
                "armature": armature_name,
                "action_slot_identifier": slot["identifier"],
                "action_slot_name": slot["name"],
                "action_slot_target_id_type": slot["target_id_type"],
            }
        )
        for frame in base._sample_frames(action, args.max_frames_per_clip):
            bpy.context.scene.frame_set(int(frame))
            bpy.context.view_layer.update()
            current = base._extract_world_mesh(mesh_objects)
            if current["vertices"].shape != rest["vertices"].shape:
                raise RuntimeError(
                    "VNEXT_TEACHER_ANIMATED_VERTEX_COUNT_DRIFT:" + requested_name
                )
            if not np.array_equal(current["faces"], rest["faces"]):
                raise RuntimeError(
                    "VNEXT_TEACHER_ANIMATED_TOPOLOGY_DRIFT:" + requested_name
                )
            if not np.array_equal(current["face_owner"], rest["face_owner"]):
                raise RuntimeError(
                    "VNEXT_TEACHER_ANIMATED_OWNER_DRIFT:" + requested_name
                )
            posed.append(current["vertices"])
            frame_rows.append(
                {
                    "action": requested_name,
                    "imported_action": str(action.name),
                    "armature": armature_name,
                    "frame": int(frame),
                }
            )
        base._neutralize_animation()

    if not posed:
        raise RuntimeError("VNEXT_TEACHER_NO_POSED_FRAMES")
    posed_array = np.stack(posed, axis=0)
    if not np.isfinite(posed_array).all():
        raise RuntimeError("VNEXT_TEACHER_POSED_NONFINITE")

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        rest_vertices_source_world=rest["vertices"],
        faces=rest["faces"],
        face_owner=rest["face_owner"],
        face_material=rest["face_material"],
        posed_vertices_source_world=posed_array,
    )
    payload_sha = base._sha256(output)
    receipt = {
        "schema": "RealSaS.IRISVNextAuthoredTeacherExport.v1",
        "transport_revision": 2,
        "subject_id": args.subject_id,
        "source": {
            "path": str(source),
            "sha256": args.expected_source_sha256,
        },
        "observation_admission_sha256": base._sha256(admission_path),
        "output": {
            "path": str(output),
            "sha256": payload_sha,
            "size_bytes": output.stat().st_size,
        },
        "import": import_receipt,
        "action_resolution": action_resolution,
        "coordinate_domain": "SOURCE_WORLD__NOT_REALSAS_CANONICAL",
        "rest_vertex_count": int(rest["vertices"].shape[0]),
        "face_count": int(rest["faces"].shape[0]),
        "anonymous_owner_count": len(rest["owners"]),
        "owners": rest["owners"],
        "native_motion_inventory_expected": list(expected_actions),
        "native_motion_inventory_exported": list(expected_actions),
        "sampled_frame_count": int(posed_array.shape[0]),
        "sampled_frames": frame_rows,
        "max_frames_per_clip": int(args.max_frames_per_clip),
        "teacher_only": True,
        "teacher_rig_skin_motion_allowed_as_inference_input": False,
        "canonical_M_G_W_minted": False,
    }
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(receipt, sort_keys=True, indent=2), encoding="utf-8"
    )
    print(
        "VNEXT_AUTHORED_TEACHER_EXPORT_PASS",
        args.subject_id,
        "vertices=",
        rest["vertices"].shape[0],
        "faces=",
        rest["faces"].shape[0],
        "frames=",
        posed_array.shape[0],
        "sha256=",
        payload_sha,
        flush=True,
    )


if __name__ == "__main__":
    main()
