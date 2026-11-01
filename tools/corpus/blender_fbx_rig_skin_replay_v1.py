#!/usr/bin/env python3
from __future__ import annotations

"""Independent pinned-Blender replay of FBX rig/skin decoded source semantics.

This is deliberately not a raw FBX parser. It re-imports the immutable FBX bytes
with the declared animation-free Blender importer policy and extracts hierarchy,
rest matrices and source vertex-group weights without importing the primary
inspection producer. A separate non-Blender court compares both decodes.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import bpy
import numpy as np

SCHEMA = "RealSaS.FbxRigSkinImporterReplay.v1"
STATUS = "PASS_FBX_RIG_SKIN_IMPORTER_REPLAY_CANDIDATE__INDEPENDENT_COURT_REQUIRED"


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


def build(source: Path, expected_source_sha256: str, subject_id: str, output_dir: Path) -> dict:
    source = source.resolve()
    output_dir = output_dir.resolve()
    if source.suffix.lower() != ".fbx":
        raise ValueError("FBX_RIG_SKIN_REPLAY_FORMAT")
    if not source.is_file() or sha256(source) != expected_source_sha256:
        raise ValueError("FBX_RIG_SKIN_REPLAY_SOURCE_DRIFT")

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

    controls = []
    control_index = {}
    armatures = sorted(
        (obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"),
        key=lambda obj: obj.name,
    )
    for armature in armatures:
        for bone in sorted(armature.data.bones, key=lambda value: value.name):
            key = (armature.name, bone.name)
            if key in control_index:
                raise ValueError("FBX_RIG_SKIN_REPLAY_CONTROL_DUPLICATE")
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
    if not controls:
        raise ValueError("FBX_RIG_SKIN_REPLAY_NO_CONTROLS")

    meshes = []
    weight_mesh = []
    weight_vertex = []
    weight_control = []
    weight_value = []
    mesh_objects = sorted(
        (obj for obj in bpy.context.scene.objects if obj.type == "MESH" and not obj.hide_render),
        key=lambda obj: obj.name,
    )
    for mesh_index, obj in enumerate(mesh_objects):
        targets = [m.object for m in obj.modifiers if m.type == "ARMATURE" and m.object is not None]
        if len(targets) > 1:
            raise ValueError("FBX_RIG_SKIN_REPLAY_MULTIPLE_ARMATURES:" + obj.name)
        armature = targets[0] if targets else None
        weighted_vertices = set()
        if armature is not None:
            for vertex in obj.data.vertices:
                for group in vertex.groups:
                    if group.weight <= 0:
                        continue
                    group_name = obj.vertex_groups[group.group].name
                    control = control_index.get((armature.name, group_name))
                    if control is None:
                        continue
                    weight_mesh.append(mesh_index)
                    weight_vertex.append(int(vertex.index))
                    weight_control.append(control)
                    weight_value.append(float(group.weight))
                    weighted_vertices.add(int(vertex.index))
        meshes.append(
            {
                "mesh_index": mesh_index,
                "name": obj.name,
                "vertex_count": len(obj.data.vertices),
                "armature_target": armature.name if armature is not None else None,
                "weighted_vertex_count": len(weighted_vertices),
            }
        )

    if not any(row["armature_target"] for row in meshes):
        raise ValueError("FBX_RIG_SKIN_REPLAY_NO_SKINNED_MESH")

    arrays_path = output_dir / "FBX_RIG_SKIN_IMPORTER_REPLAY_V1.npz"
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        arrays_path,
        weight_mesh_index=np.asarray(weight_mesh, dtype=np.int32),
        weight_vertex_index=np.asarray(weight_vertex, dtype=np.int32),
        weight_control_index=np.asarray(weight_control, dtype=np.int32),
        weight_value=np.asarray(weight_value, dtype=np.float32),
    )
    payload = {
        "schema": SCHEMA,
        "status": STATUS,
        "subject_id": subject_id,
        "source": {"path": str(source), "sha256": expected_source_sha256},
        "authority_scope": "PINNED_BLENDER_FBX_DECODED_SOURCE_SEMANTICS__NOT_RAW_FBX_PARSER_PROOF",
        "blender_version": str(bpy.app.version_string),
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
            "skinned_mesh_count": sum(bool(row["armature_target"]) for row in meshes),
            "weight_row_count": len(weight_value),
        },
        "arrays": ref(arrays_path),
        "authority": {
            "importer_replay_candidate_only": True,
            "independent_court_required": True,
            "raw_fbx_parser_authority_claimed": False,
            "family_authority_minted": False,
            "training_authorized": False,
        },
    }
    manifest_path = output_dir / "FBX_RIG_SKIN_IMPORTER_REPLAY_V1.json"
    manifest_path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"status": STATUS, "subject_id": subject_id, "counts": payload["counts"]}, sort_keys=True))
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
