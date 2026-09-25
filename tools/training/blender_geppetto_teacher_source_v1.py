from __future__ import annotations

"""Extract training/evaluation-only Geppetto teacher arrays from an exact FBX.

This script is intended to run inside Blender. It mirrors the controlled 8-view
materializer's FBX import/rest-pose semantics and emits only source-rig teacher
arrays. It does not mint product joint IDs or perform Compiler qualification.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import bpy
import numpy as np


def _args():
    raw = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-fbx", required=True)
    parser.add_argument("--output-npz", required=True)
    parser.add_argument("--output-audit", required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    return parser.parse_args(raw)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _reset_and_import(path: Path) -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.import_scene.fbx(
        filepath=str(path),
        automatic_bone_orientation=False,
    )
    bpy.context.scene.frame_set(0)
    for obj in bpy.context.scene.objects:
        if obj.type == "ARMATURE":
            obj.data.pose_position = "REST"
    bpy.context.view_layer.update()


def _world_matrix_np(matrix) -> np.ndarray:
    return np.asarray([[float(v) for v in row] for row in matrix], dtype=np.float64)


def main() -> None:
    args = _args()
    source = Path(args.source_fbx).expanduser().resolve()
    out_npz = Path(args.output_npz).expanduser().resolve()
    out_audit = Path(args.output_audit).expanduser().resolve()
    if not source.is_file():
        raise RuntimeError("GEPPETTO_TEACHER_SOURCE_FBX_MISSING")
    source_sha = _sha256(source)
    if source_sha != str(args.expected_source_sha256):
        raise RuntimeError(
            f"GEPPETTO_TEACHER_SOURCE_SHA_DRIFT:{source_sha}!={args.expected_source_sha256}"
        )

    _reset_and_import(source)
    armatures = sorted(
        (obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"),
        key=lambda obj: obj.name,
    )
    if len(armatures) != 1:
        raise RuntimeError(
            f"GEPPETTO_TEACHER_REQUIRES_EXACT_ONE_ARMATURE:{len(armatures)}"
        )
    armature = armatures[0]
    bones = list(armature.data.bones)
    if not bones:
        raise RuntimeError("GEPPETTO_TEACHER_ARMATURE_EMPTY")

    bone_index = {bone.name: i for i, bone in enumerate(bones)}
    parents = np.asarray(
        [
            -1 if bone.parent is None else int(bone_index[bone.parent.name])
            for bone in bones
        ],
        dtype=np.int64,
    )
    deform_mask = np.asarray([bool(bone.use_deform) for bone in bones], dtype=bool)

    armature_world = armature.matrix_world.copy()
    rest_world_source = np.zeros((len(bones), 4, 4), dtype=np.float64)
    head_world = np.zeros((len(bones), 3), dtype=np.float64)
    tail_world = np.zeros((len(bones), 3), dtype=np.float64)
    max_head_matrix_error = 0.0
    for i, bone in enumerate(bones):
        world_matrix = armature_world @ bone.matrix_local
        rest_world_source[i] = _world_matrix_np(world_matrix)
        head = armature_world @ bone.head_local
        tail = armature_world @ bone.tail_local
        head_world[i] = np.asarray(tuple(map(float, head)), dtype=np.float64)
        tail_world[i] = np.asarray(tuple(map(float, tail)), dtype=np.float64)
        matrix_head = rest_world_source[i, :3, 3]
        max_head_matrix_error = max(
            max_head_matrix_error,
            float(np.max(np.abs(matrix_head - head_world[i]))),
        )
    if max_head_matrix_error > 1e-7:
        raise RuntimeError(
            f"GEPPETTO_TEACHER_REST_MATRIX_HEAD_DRIFT:{max_head_matrix_error}"
        )

    mesh_objects = sorted(
        (
            obj
            for obj in bpy.context.scene.objects
            if obj.type == "MESH" and not obj.hide_render
        ),
        key=lambda obj: obj.name,
    )
    if not mesh_objects:
        raise RuntimeError("GEPPETTO_TEACHER_NO_RENDERABLE_MESH")

    total_vertices = sum(len(obj.data.vertices) for obj in mesh_objects)
    skin = np.zeros((total_vertices, len(bones)), dtype=np.float32)
    row_offset = 0
    ignored_nonbone_group_memberships = 0
    matched_weight_memberships = 0
    mesh_rows = []
    for obj in mesh_objects:
        groups = {int(group.index): group.name for group in obj.vertex_groups}
        matched_groups = {
            index: bone_index[name]
            for index, name in groups.items()
            if name in bone_index
        }
        object_weight_sum = 0.0
        for local_index, vertex in enumerate(obj.data.vertices):
            row = row_offset + local_index
            for membership in vertex.groups:
                group_index = int(membership.group)
                weight = float(membership.weight)
                if not np.isfinite(weight) or weight < 0.0:
                    raise RuntimeError(
                        f"GEPPETTO_TEACHER_INVALID_WEIGHT:{obj.name}:{local_index}"
                    )
                if weight <= 0.0:
                    continue
                bone_col = matched_groups.get(group_index)
                if bone_col is None:
                    ignored_nonbone_group_memberships += 1
                    continue
                skin[row, bone_col] += np.float32(weight)
                object_weight_sum += weight
                matched_weight_memberships += 1
        mesh_rows.append(
            {
                "object_name": obj.name,
                "vertex_count": len(obj.data.vertices),
                "matched_bone_vertex_group_count": len(matched_groups),
                "matched_weight_sum": object_weight_sum,
            }
        )
        row_offset += len(obj.data.vertices)

    if row_offset != total_vertices:
        raise RuntimeError("GEPPETTO_TEACHER_SKIN_ROW_ACCOUNTING_DRIFT")
    if matched_weight_memberships <= 0 or not np.any(skin > 0.0):
        raise RuntimeError("GEPPETTO_TEACHER_SKIN_SUPPORT_EMPTY")
    if not np.isfinite(skin).all() or np.any(skin < 0.0):
        raise RuntimeError("GEPPETTO_TEACHER_SKIN_INVALID")

    skin_mass = skin.astype(np.float64).sum(axis=0)
    supported_deform = deform_mask & (skin_mass > 1e-8)
    if not np.any(supported_deform):
        raise RuntimeError("GEPPETTO_TEACHER_NO_SKIN_SUPPORTED_DEFORM_BONE")

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        parents=parents,
        deform_mask=deform_mask.astype(np.uint8),
        skin=skin,
        rest_world_source=rest_world_source,
        bone_heads_world=head_world,
        bone_tails_world=tail_world,
    )
    npz_sha = _sha256(out_npz)
    audit = {
        "schema": "RealSaS.KnightGeppettoTeacherSourceExtraction.v1",
        "status": "PASS",
        "authority_class": "TRAINING_EVALUATION_ONLY",
        "source_fbx": {
            "path": str(source),
            "sha256": source_sha,
            "bytes": source.stat().st_size,
        },
        "import_contract": {
            "automatic_bone_orientation": False,
            "frame": 0,
            "armature_pose_position": "REST",
            "matches_controlled_8view_materializer": True,
        },
        "armature": {
            "object_name_provenance_only": armature.name,
            "bone_count": len(bones),
            "deform_bone_count": int(np.count_nonzero(deform_mask)),
            "skin_supported_deform_bone_count": int(np.count_nonzero(supported_deform)),
            "root_bone_count": int(np.count_nonzero(parents < 0)),
            "max_rest_matrix_head_error": max_head_matrix_error,
        },
        "skin": {
            "mesh_vertex_row_count": total_vertices,
            "matched_weight_membership_count": matched_weight_memberships,
            "ignored_nonbone_group_membership_count": ignored_nonbone_group_memberships,
            "positive_skin_column_count": int(np.count_nonzero(skin_mass > 1e-8)),
            "mesh_objects": mesh_rows,
        },
        "output_npz": {
            "path": str(out_npz),
            "sha256": npz_sha,
            "arrays": {
                "parents": list(parents.shape),
                "deform_mask": list(deform_mask.shape),
                "skin": list(skin.shape),
                "rest_world_source": list(rest_world_source.shape),
                "bone_heads_world": list(head_world.shape),
                "bone_tails_world": list(tail_world.shape),
            },
        },
        "product_authority_claimed": False,
        "canonical_joint_ids_created": False,
        "teacher_inference_inputs_allowed": False,
    }
    out_audit.parent.mkdir(parents=True, exist_ok=True)
    out_audit.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("GEPPETTO_TEACHER_SOURCE_EXTRACTION_PASS")
    print(json.dumps(audit, sort_keys=True))


if __name__ == "__main__":
    main()
