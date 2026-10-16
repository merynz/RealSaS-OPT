from __future__ import annotations

"""Export authored rest topology + native deformation sequence for IRIS vNext.

Run inside Blender:

  blender -b --python tools/iris_vnext/blender_authored_teacher_export_v1.py -- \
    --source ... --expected-source-sha256 ... --subject-id ... \
    --observation-admission ... --output ...

The exporter emits source-world geometry only.  It does not choose RealSaS
canonical coordinates and it never mints canonical M/G/W.  The vNext shard
builder later binds this evidence to the exact legacy IRIS normalization.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import bpy
import numpy as np


def _args():
    raw = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--expected-source-sha256", required=True)
    p.add_argument("--subject-id", required=True)
    p.add_argument("--observation-admission", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-frames-per-clip", type=int, default=16)
    return p.parse_args(raw)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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
    if suffix in (".gltf", ".glb"):
        bpy.ops.import_scene.gltf(filepath=str(source), guess_original_bind_pose=True)
        return {
            "operation": "GLTF_IMPORT_WITH_NATIVE_ANIMATION",
            "importer_options": {"guess_original_bind_pose": True},
        }
    raise RuntimeError("VNEXT_TEACHER_SOURCE_FORMAT_UNSUPPORTED")


def _custom_shape_helpers() -> set[bpy.types.Object]:
    return {
        bone.custom_shape
        for obj in bpy.context.scene.objects
        if obj.type == "ARMATURE"
        for bone in obj.pose.bones
        if bone.custom_shape is not None
    }


def _mesh_objects() -> tuple[bpy.types.Object, ...]:
    helpers = _custom_shape_helpers()
    rows = tuple(
        sorted(
            (
                obj
                for obj in bpy.context.scene.objects
                if obj.type == "MESH" and not obj.hide_render and obj not in helpers
            ),
            key=lambda obj: obj.name,
        )
    )
    if not rows:
        raise RuntimeError("VNEXT_TEACHER_NO_RENDERABLE_MESH")
    return rows


def _animation_owners():
    rows = []
    for obj in bpy.context.scene.objects:
        rows.append(obj)
        if getattr(obj, "data", None) is not None:
            rows.append(obj.data)
        shape_keys = getattr(getattr(obj, "data", None), "shape_keys", None)
        if shape_keys is not None:
            rows.append(shape_keys)
    # Preserve object identity while removing duplicates.
    return tuple(dict.fromkeys(rows))


def _neutralize_animation() -> None:
    for owner in _animation_owners():
        animation = getattr(owner, "animation_data", None)
        if animation is None:
            continue
        animation.action = None
        for track in animation.nla_tracks:
            track.mute = True
    for obj in bpy.context.scene.objects:
        if obj.type == "ARMATURE":
            obj.data.pose_position = "REST"
    bpy.context.scene.frame_set(0)
    bpy.context.view_layer.update()


def _pose_bones_used(action) -> set[str]:
    names = set()
    pattern = re.compile(r'^pose\.bones\["([^"]+)"\]')
    for curve in action.fcurves:
        match = pattern.match(curve.data_path)
        if match:
            names.add(match.group(1))
    return names


def _armature_for_action(action):
    required = _pose_bones_used(action)
    armatures = tuple(
        sorted(
            (obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"),
            key=lambda obj: obj.name,
        )
    )
    if not armatures:
        raise RuntimeError("VNEXT_TEACHER_ACTION_WITHOUT_ARMATURE")
    if required:
        compatible = [
            obj for obj in armatures if required.issubset(set(obj.pose.bones.keys()))
        ]
    else:
        compatible = [
            obj
            for obj in armatures
            if obj.animation_data is not None
            and (
                obj.animation_data.action == action
                or any(
                    strip.action == action
                    for track in obj.animation_data.nla_tracks
                    for strip in track.strips
                )
            )
        ]
    if len(compatible) != 1:
        raise RuntimeError(
            "VNEXT_TEACHER_ACTION_OWNER_AMBIGUOUS:"
            + action.name
            + ":"
            + ",".join(obj.name for obj in compatible)
        )
    return compatible[0]


def _activate_action(action) -> str:
    for owner in _animation_owners():
        animation = getattr(owner, "animation_data", None)
        if animation is None:
            continue
        animation.action = None
        for track in animation.nla_tracks:
            track.mute = True
    armature = _armature_for_action(action)
    armature.data.pose_position = "POSE"
    if armature.animation_data is None:
        armature.animation_data_create()
    armature.animation_data.action = action
    bpy.context.view_layer.update()
    return armature.name


def _sample_frames(action, maximum: int) -> tuple[int, ...]:
    if maximum <= 0:
        raise RuntimeError("VNEXT_TEACHER_MAX_FRAMES_INVALID")
    start, end = map(float, action.frame_range)
    lo, hi = int(math.floor(start)), int(math.ceil(end))
    if hi < lo:
        raise RuntimeError("VNEXT_TEACHER_ACTION_FRAME_RANGE_INVALID")
    count = min(maximum, hi - lo + 1)
    if count <= 1:
        return (lo,)
    values = {
        int(round(lo + (hi - lo) * index / float(count - 1)))
        for index in range(count)
    }
    return tuple(sorted(values))


def _extract_world_mesh(mesh_objects) -> dict:
    deps = bpy.context.evaluated_depsgraph_get()
    vertices = []
    faces = []
    face_owner = []
    face_material = []
    owner_rows = []
    vertex_offset = 0
    for owner_index, obj in enumerate(mesh_objects):
        evaluated = obj.evaluated_get(deps)
        mesh = evaluated.to_mesh()
        try:
            mesh.calc_loop_triangles()
            local_vertices = [
                tuple(float(x) for x in (evaluated.matrix_world @ vertex.co))
                for vertex in mesh.vertices
            ]
            triangles = [tuple(map(int, tri.vertices)) for tri in mesh.loop_triangles]
            materials = [int(tri.material_index) for tri in mesh.loop_triangles]
            vertices.extend(local_vertices)
            faces.extend(
                tuple(vertex_offset + index for index in triangle)
                for triangle in triangles
            )
            face_owner.extend([owner_index] * len(triangles))
            face_material.extend(materials)
            owner_rows.append(
                {
                    "anonymous_owner_index": owner_index,
                    "source_import_name": obj.name,
                    "vertex_count": len(local_vertices),
                    "triangle_count": len(triangles),
                    "parent_type": str(obj.parent_type),
                    "parent_object_name": obj.parent.name if obj.parent else None,
                    "parent_bone": obj.parent_bone or None,
                    "material_slot_names": [
                        slot.material.name if slot.material else None
                        for slot in obj.material_slots
                    ],
                }
            )
            vertex_offset += len(local_vertices)
        finally:
            evaluated.to_mesh_clear()
    return {
        "vertices": np.asarray(vertices, dtype=np.float64),
        "faces": np.asarray(faces, dtype=np.int64),
        "face_owner": np.asarray(face_owner, dtype=np.int32),
        "face_material": np.asarray(face_material, dtype=np.int32),
        "owners": owner_rows,
    }


def main():
    args = _args()
    source = args.source.expanduser().resolve()
    admission_path = args.observation_admission.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not source.is_file() or _sha256(source) != args.expected_source_sha256:
        raise RuntimeError("VNEXT_TEACHER_EXACT_SOURCE_HASH_REQUIRED")
    admission = json.loads(admission_path.read_text(encoding="utf-8"))
    if admission.get("subject_id") != args.subject_id:
        raise RuntimeError("VNEXT_TEACHER_SUBJECT_ADMISSION_DRIFT")
    if admission.get("source_sha256") != args.expected_source_sha256:
        raise RuntimeError("VNEXT_TEACHER_SOURCE_ADMISSION_HASH_DRIFT")
    expected_actions = tuple(map(str, admission.get("native_motion_inventory") or ()))
    if not expected_actions:
        raise RuntimeError("VNEXT_TEACHER_NATIVE_MOTION_INVENTORY_EMPTY")

    import_receipt = _import_animated(source)
    actual_actions = {action.name: action for action in bpy.data.actions}
    missing = sorted(set(expected_actions) - set(actual_actions))
    if missing:
        raise RuntimeError("VNEXT_TEACHER_NATIVE_ACTION_MISSING:" + ",".join(missing))

    mesh_objects = _mesh_objects()
    _neutralize_animation()
    rest = _extract_world_mesh(mesh_objects)
    if len(rest["vertices"]) == 0 or len(rest["faces"]) == 0:
        raise RuntimeError("VNEXT_TEACHER_REST_MESH_EMPTY")

    posed = []
    frame_rows = []
    for action_name in expected_actions:
        action = actual_actions[action_name]
        armature_name = _activate_action(action)
        for frame in _sample_frames(action, args.max_frames_per_clip):
            bpy.context.scene.frame_set(int(frame))
            bpy.context.view_layer.update()
            current = _extract_world_mesh(mesh_objects)
            if current["vertices"].shape != rest["vertices"].shape:
                raise RuntimeError("VNEXT_TEACHER_ANIMATED_VERTEX_COUNT_DRIFT:" + action_name)
            if not np.array_equal(current["faces"], rest["faces"]):
                raise RuntimeError("VNEXT_TEACHER_ANIMATED_TOPOLOGY_DRIFT:" + action_name)
            if not np.array_equal(current["face_owner"], rest["face_owner"]):
                raise RuntimeError("VNEXT_TEACHER_ANIMATED_OWNER_DRIFT:" + action_name)
            posed.append(current["vertices"])
            frame_rows.append(
                {
                    "action": action_name,
                    "armature": armature_name,
                    "frame": int(frame),
                }
            )
        _neutralize_animation()

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
    payload_sha = _sha256(output)
    receipt = {
        "schema": "RealSaS.IRISVNextAuthoredTeacherExport.v1",
        "subject_id": args.subject_id,
        "source": {
            "path": str(source),
            "sha256": args.expected_source_sha256,
        },
        "observation_admission_sha256": _sha256(admission_path),
        "output": {
            "path": str(output),
            "sha256": payload_sha,
            "size_bytes": output.stat().st_size,
        },
        "import": import_receipt,
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
        "vertices=", rest["vertices"].shape[0],
        "faces=", rest["faces"].shape[0],
        "frames=", posed_array.shape[0],
        "sha256=", payload_sha,
        flush=True,
    )


if __name__ == "__main__":
    main()
