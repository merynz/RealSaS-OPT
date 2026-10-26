from __future__ import annotations

"""Blender transport revision 3 for authored IRIS vNext teacher export.

V3 preserves the scientific teacher semantics of V1/V2 but fixes one Blender
transport artifact: ``calc_loop_triangles()`` may choose a different diagonal for
the same deformed quad/ngon at different poses. That is pose-time tessellation
drift, not source topology drift.

The rest pose therefore owns the one frozen triangleization used by the teacher.
Every sampled pose must still preserve evaluated vertex count, mesh-object order,
source polygon vertex cycles, polygon material indices, and owner census. Only
vertex positions are sampled from animation. Any real connectivity/owner change
continues to fail closed.
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
from tools.iris_vnext import blender_authored_teacher_export_v2 as transport


def _extract_snapshot(mesh_objects, *, freeze_triangulation: bool) -> dict:
    deps = bpy.context.evaluated_depsgraph_get()
    vertices = []
    faces = []
    face_owner = []
    face_material = []
    owner_rows = []
    polygon_signature = []
    vertex_offset = 0

    for owner_index, obj in enumerate(mesh_objects):
        evaluated = obj.evaluated_get(deps)
        mesh = evaluated.to_mesh()
        try:
            local_vertices = [
                tuple(float(x) for x in (evaluated.matrix_world @ vertex.co))
                for vertex in mesh.vertices
            ]
            polygon_cycles = tuple(
                tuple(map(int, polygon.vertices)) for polygon in mesh.polygons
            )
            polygon_materials = tuple(int(polygon.material_index) for polygon in mesh.polygons)
            polygon_signature.append(
                (
                    str(obj.name),
                    len(local_vertices),
                    polygon_cycles,
                    polygon_materials,
                )
            )
            vertices.extend(local_vertices)

            triangle_count = None
            if freeze_triangulation:
                mesh.calc_loop_triangles()
                triangles = [tuple(map(int, tri.vertices)) for tri in mesh.loop_triangles]
                materials = [int(tri.material_index) for tri in mesh.loop_triangles]
                faces.extend(
                    tuple(vertex_offset + index for index in triangle)
                    for triangle in triangles
                )
                face_owner.extend([owner_index] * len(triangles))
                face_material.extend(materials)
                triangle_count = len(triangles)

            owner_rows.append(
                {
                    "anonymous_owner_index": owner_index,
                    "source_import_name": obj.name,
                    "vertex_count": len(local_vertices),
                    "polygon_count": len(polygon_cycles),
                    "triangle_count": triangle_count,
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
        "polygon_signature": tuple(polygon_signature),
    }


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
        requested: transport._action_by_requested_take(requested)
        for requested in expected_actions
    }
    if len({id(action) for action in resolved_actions.values()}) != len(resolved_actions):
        raise RuntimeError("VNEXT_TEACHER_TAKE_ALIAS_COLLISION")

    mesh_objects = base._mesh_objects()
    base._neutralize_animation()
    rest = _extract_snapshot(mesh_objects, freeze_triangulation=True)
    if len(rest["vertices"]) == 0 or len(rest["faces"]) == 0:
        raise RuntimeError("VNEXT_TEACHER_REST_MESH_EMPTY")

    posed = []
    frame_rows = []
    action_resolution = []
    for requested_name in expected_actions:
        action = resolved_actions[requested_name]
        armature_name, slot = transport._activate_action(action)
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
            current = _extract_snapshot(mesh_objects, freeze_triangulation=False)
            if current["vertices"].shape != rest["vertices"].shape:
                raise RuntimeError(
                    "VNEXT_TEACHER_ANIMATED_VERTEX_COUNT_DRIFT:" + requested_name
                )
            if current["polygon_signature"] != rest["polygon_signature"]:
                raise RuntimeError(
                    "VNEXT_TEACHER_ANIMATED_POLYGON_CONNECTIVITY_DRIFT:" + requested_name
                )
            if len(current["owners"]) != len(rest["owners"]):
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
        "transport_revision": 3,
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
        "topology_validation_policy": "REST_TRIANGULATION_FROZEN__POSE_VERTEX_INDEX_AND_POLYGON_CONNECTIVITY_MATERIALS_VERIFIED",
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
