#!/usr/bin/env python3
from __future__ import annotations

"""Primary pinned-Blender FBX source inspection for Corpus370 rig/skin authority.

This producer inspects the exact immutable FBX bytes with animation disabled and
records the importer-decoded bind-rest hierarchy, mesh-to-armature bindings, and
authored vertex-group weights. It does not mint family or training authority.
A separately implemented Blender replay plus a non-Blender court must agree
before this evidence can participate in final admission.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import bpy
import numpy as np

SCHEMA = "RealSaS.ImportedTeacherSourceInspection.v1"
STATUS = "PASS_FBX_IMPORTED_TEACHER_SOURCE_INSPECTION__INDEPENDENT_REPLAY_REQUIRED"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def matrix_rows(value) -> list[list[float]]:
    return [[float(x) for x in row] for row in value]


def ref(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def require(condition: bool, code: str) -> None:
    if not condition:
        raise ValueError(code)


def build(source: Path, expected_source_sha256: str, subject_id: str, output_dir: Path) -> dict:
    source = source.resolve()
    output_dir = output_dir.resolve()
    require(source.suffix.lower() == ".fbx", "FBX_SOURCE_INSPECTION_FORMAT")
    require(source.is_file(), "FBX_SOURCE_INSPECTION_SOURCE_MISSING")
    require(sha256(source) == expected_source_sha256, "FBX_SOURCE_INSPECTION_SOURCE_DRIFT")

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(
        filepath=str(source),
        automatic_bone_orientation=False,
        use_anim=False,
    )
    for obj in bpy.context.scene.objects:
        if obj.type == "ARMATURE":
            obj.data.pose_position = "REST"
    bpy.context.view_layer.update()

    controls: list[dict] = []
    control_index: dict[tuple[str, str], int] = {}
    armatures = sorted(
        (obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"),
        key=lambda obj: obj.name,
    )
    for armature in armatures:
        for bone in sorted(armature.data.bones, key=lambda value: value.name):
            key = (armature.name, bone.name)
            require(key not in control_index, "FBX_SOURCE_INSPECTION_CONTROL_DUPLICATE")
            control_index[key] = len(controls)
            controls.append(
                {
                    "armature": armature.name,
                    "name": bone.name,
                    "parent": bone.parent.name if bone.parent else None,
                    "use_deform": bool(bone.use_deform),
                    "rest_matrix_world": matrix_rows(armature.matrix_world @ bone.matrix_local),
                }
            )
    require(bool(controls), "FBX_SOURCE_INSPECTION_NO_CONTROLS")

    meshes: list[dict] = []
    weight_vertex: list[int] = []
    weight_joint: list[int] = []
    weight_value: list[float] = []
    global_vertex_offset = 0
    skinned_mesh_count = 0

    mesh_objects = sorted(
        (obj for obj in bpy.context.scene.objects if obj.type == "MESH" and not obj.hide_render),
        key=lambda obj: obj.name,
    )
    require(bool(mesh_objects), "FBX_SOURCE_INSPECTION_NO_MESHES")

    for obj in mesh_objects:
        targets = sorted(
            {
                modifier.object.name
                for modifier in obj.modifiers
                if modifier.type == "ARMATURE" and modifier.object is not None
            }
        )
        require(len(targets) <= 1, "FBX_SOURCE_INSPECTION_MULTIPLE_ARMATURES:" + obj.name)
        target = targets[0] if targets else None
        weighted_vertices: set[int] = set()

        if target is not None:
            skinned_mesh_count += 1
            for vertex in obj.data.vertices:
                for membership in vertex.groups:
                    weight = float(membership.weight)
                    if weight <= 0.0:
                        continue
                    group_name = obj.vertex_groups[membership.group].name
                    joint = control_index.get((target, group_name))
                    if joint is None:
                        continue
                    weight_vertex.append(global_vertex_offset + int(vertex.index))
                    weight_joint.append(joint)
                    weight_value.append(weight)
                    weighted_vertices.add(int(vertex.index))

        meshes.append(
            {
                "name": obj.name,
                "vertex_count": len(obj.data.vertices),
                "armature_targets": targets,
                "weighted_vertex_count": len(weighted_vertices),
            }
        )
        global_vertex_offset += len(obj.data.vertices)

    require(skinned_mesh_count > 0, "FBX_SOURCE_INSPECTION_NO_SKINNED_MESH")
    require(bool(weight_value), "FBX_SOURCE_INSPECTION_NO_AUTHORED_WEIGHTS")

    output_dir.mkdir(parents=True, exist_ok=True)
    arrays_path = output_dir / "IMPORTED_TEACHER_SOURCE_REST_GEOMETRY_V1.npz"
    np.savez_compressed(
        arrays_path,
        weight_vertex=np.asarray(weight_vertex, dtype=np.int64),
        weight_joint=np.asarray(weight_joint, dtype=np.int32),
        weight_value=np.asarray(weight_value, dtype=np.float32),
    )

    payload = {
        "schema": SCHEMA,
        "status": STATUS,
        "subject_id": subject_id,
        "source": {"path": str(source), "sha256": expected_source_sha256},
        "runtime": {"blender": str(bpy.app.version_string)},
        "import": {
            "operation": "FBX_IMPORT_WITHOUT_ANIMATION",
            "importer_options": {"automatic_bone_orientation": False, "use_anim": False},
            "pose_policy": "SOURCE_BIND_REST__NO_ANIMATION",
        },
        "controls": controls,
        "meshes": meshes,
        "counts": {
            "control_count": len(controls),
            "mesh_count": len(meshes),
            "skinned_mesh_count": skinned_mesh_count,
            "weight_row_count": len(weight_value),
            "global_vertex_count": global_vertex_offset,
        },
        "outputs": {"rest_geometry": ref(arrays_path)},
        "authority": {
            "source_inspection_candidate_only": True,
            "independent_replay_required": True,
            "family_authority_minted": False,
            "training_authorized": False,
        },
    }
    manifest_path = output_dir / "IMPORTED_TEACHER_SOURCE_INSPECTION_V1.json"
    manifest_path.write_text(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {"status": STATUS, "subject_id": subject_id, "counts": payload["counts"]},
            sort_keys=True,
        ),
        flush=True,
    )
    return payload


def main() -> None:
    raw = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--subject-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(raw)
    build(args.source, args.expected_source_sha256, args.subject_id, args.output_dir)


if __name__ == "__main__":
    main()
