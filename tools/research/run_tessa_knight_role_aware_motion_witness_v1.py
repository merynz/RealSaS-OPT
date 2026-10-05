from __future__ import annotations

"""Knight role-aware motion retarget witness.

Research-only adapter:
  sealed source motion clip
  -> source control/animation world pose
  -> role-aware world-rotation retarget
  -> target 23-joint deform skeleton
  -> rigid-equipment + continuous-skin carrier LBS.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from compiler.realsas_compiler_core.joint_frames_v1 import derive_joint_frames_from_rows
from compiler.realsas_compiler_core.motion_role_retarget_v1 import (
    INHERIT_PARENT,
    WORLD_ROTATION_DELTA,
    MotionRoleBindingV1IR,
    retarget_world_rotation_deltas_v1,
)

SCHEMA = "RealSaS.TESSAKnightRoleAwareMotionWitness.v1"
ROLE_MAP = {
    "root": "Bone",
    "hips": "Body",
    "spine": "Abdomen",
    "chest": "Torso",
    "upperarm.l": "UpperArm.L",
    "lowerarm.l": "LowerArm.L",
    "wrist.l": "Palm.L",
    "hand.l": "MiddleHand.L",
    "handslot.l": None,
    "upperarm.r": "UpperArm.R",
    "lowerarm.r": "LowerArm.R",
    "wrist.r": "Palm.R",
    "hand.r": "MiddleHand.R",
    "handslot.r": None,
    "head": "Head",
    "upperleg.l": "UpperLeg.L",
    "lowerleg.l": "LowerLeg.L",
    "foot.l": "Foot.L",
    "toes.l": None,
    "upperleg.r": "UpperLeg.R",
    "lowerleg.r": "LowerLeg.R",
    "foot.r": "Foot.R",
    "toes.r": None,
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def require(value: bool, code: str) -> None:
    if not value:
        raise RuntimeError(code)


def unit(v, code: str) -> np.ndarray:
    a = np.asarray(v, dtype=np.float64)
    require(a.shape == (3,) and np.isfinite(a).all(), code)
    n = float(np.linalg.norm(a))
    require(n > 1.0e-12, code)
    return a / n


def teacher_object_frame(bone_names, heads) -> np.ndarray:
    index = {str(name): i for i, name in enumerate(bone_names)}
    for required in (
        "hips", "head", "upperarm.l", "upperarm.r",
        "upperleg.l", "upperleg.r", "foot.l", "foot.r",
    ):
        require(required in index, "ROLE_WITNESS_TEACHER_FRAME_BONE_MISSING:" + required)
    up = unit(heads[index["head"]] - heads[index["hips"]], "ROLE_WITNESS_UP_DEGENERATE")
    lateral = [
        heads[index[base + ".r"]] - heads[index[base + ".l"]]
        for base in ("upperarm", "upperleg", "foot")
    ]
    right = np.sum(np.asarray(lateral, dtype=np.float64), axis=0)
    right = unit(right - up * float(np.dot(right, up)), "ROLE_WITNESS_RIGHT_DEGENERATE")
    forward = unit(np.cross(up, right), "ROLE_WITNESS_FORWARD_DEGENERATE")
    right = unit(np.cross(forward, up), "ROLE_WITNESS_RIGHT_ORTHO_FAIL")
    C = np.stack((right, forward, up), axis=0)
    require(float(np.linalg.det(C)) > 0.999999, "ROLE_WITNESS_OBJECT_FRAME_INVALID")
    return C


def matrices_from_rows(rows, *, id_key, parent_key, position_key):
    rows = tuple(dict(row) for row in rows)
    ids = tuple(str(row[id_key]) for row in rows)
    index = {jid: i for i, jid in enumerate(ids)}
    parents = np.asarray([
        -1 if row.get(parent_key) in (None, "") else index[str(row[parent_key])]
        for row in rows
    ], dtype=np.int64)
    frames = derive_joint_frames_from_rows(
        rows,
        joint_id_key=id_key,
        parent_id_key=parent_key,
        position_key=position_key,
    )
    global_rest = np.empty((len(rows), 4, 4), dtype=np.float64)
    for i, row in enumerate(rows):
        jid = ids[i]
        M = np.eye(4, dtype=np.float64)
        M[:3, :3] = np.asarray(frames[jid].rotation_matrix, dtype=np.float64)
        M[:3, 3] = np.asarray(row[position_key], dtype=np.float64)
        global_rest[i] = M
    local_rest = np.empty_like(global_rest)
    for i, parent in enumerate(parents):
        local_rest[i] = global_rest[i] if int(parent) < 0 else np.linalg.inv(global_rest[int(parent)]) @ global_rest[i]
    return ids, parents, global_rest, local_rest


def topological_order(parents):
    parents = np.asarray(parents, dtype=np.int64)
    remaining = set(range(len(parents)))
    out = []
    while remaining:
        progressed = False
        for i in sorted(tuple(remaining)):
            if int(parents[i]) < 0 or int(parents[i]) in out:
                out.append(i)
                remaining.remove(i)
                progressed = True
                break
        require(progressed, "ROLE_WITNESS_SOURCE_SKELETON_CYCLE")
    return tuple(out)


def source_pose_matrices(motion_payload: dict):
    rows = tuple(motion_payload["source_skeleton"])
    source_ids, parents, rest_global, rest_local = matrices_from_rows(
        rows,
        id_key="source_joint_id",
        parent_key="parent_source_joint_id",
        position_key="rest_position",
    )
    tracks = {
        str(row["source_joint_id"]): tuple(row["keyframes"])
        for row in motion_payload["tracks"]
    }
    require(set(tracks) == set(source_ids), "ROLE_WITNESS_SOURCE_TRACK_COVERAGE_DRIFT")
    reference = tracks[source_ids[0]]
    times = np.asarray([float(key["time_seconds"]) for key in reference], dtype=np.float64)
    for source_id in source_ids:
        keys = tracks[source_id]
        require(len(keys) == len(times), "ROLE_WITNESS_SOURCE_TRACK_LENGTH_DRIFT:" + source_id)
        require(
            max(abs(float(keys[i]["time_seconds"]) - float(times[i])) for i in range(len(times))) <= 1.0e-9,
            "ROLE_WITNESS_SOURCE_TRACK_TIME_DRIFT:" + source_id,
        )

    source_body_scale = float(motion_payload["metadata"]["source_body_scale"])
    require(math.isfinite(source_body_scale) and source_body_scale > 0.0, "ROLE_WITNESS_SOURCE_BODY_SCALE_INVALID")
    order = topological_order(parents)
    poses = []
    for frame_index in range(len(times)):
        posed = np.zeros_like(rest_global)
        posed[:, 3, 3] = 1.0
        for i in order:
            source_id = source_ids[i]
            key = tracks[source_id][frame_index]
            delta = np.eye(4, dtype=np.float64)
            quat = np.asarray(key["local_rotation_quat_xyzw"], dtype=np.float64)
            require(quat.shape == (4,) and np.isfinite(quat).all(), "ROLE_WITNESS_SOURCE_QUAT_INVALID")
            delta[:3, :3] = Rotation.from_quat(quat).as_matrix()
            translation = np.asarray(key["local_translation_xyz"], dtype=np.float64)
            require(translation.shape == (3,) and np.isfinite(translation).all(), "ROLE_WITNESS_SOURCE_TRANSLATION_INVALID")
            delta[:3, 3] = translation * source_body_scale
            parent = int(parents[i])
            posed[i] = rest_global[i] @ delta if parent < 0 else posed[parent] @ rest_local[i] @ delta
        poses.append(posed)
    return source_ids, parents, rest_global, np.asarray(poses), times


def target_rest_matrices(*, bone_names, raw_heads, raw_parents):
    C = teacher_object_frame(bone_names, raw_heads)
    target_names = tuple(map(str, bone_names[:23]))
    target_positions = (C @ np.asarray(raw_heads[:23], dtype=np.float64).T).T
    target_parents = np.asarray(raw_parents[:23], dtype=np.int64)
    rows = []
    for i, name in enumerate(target_names):
        parent = int(target_parents[i])
        rows.append({
            "joint_id": name,
            "parent_id": None if parent < 0 else target_names[parent],
            "position": target_positions[i].tolist(),
        })
    ids, parents, rest_global, _ = matrices_from_rows(
        rows,
        id_key="joint_id",
        parent_key="parent_id",
        position_key="position",
    )
    require(ids == target_names, "ROLE_WITNESS_TARGET_ORDER_DRIFT")
    return C, target_names, parents, rest_global


def bindings_for_knight(target_names, source_ids):
    source = set(map(str, source_ids))
    rows = []
    for target in target_names:
        source_id = ROLE_MAP[str(target)]
        if source_id is None:
            rows.append(
                MotionRoleBindingV1IR(
                    str(target),
                    None,
                    INHERIT_PARENT,
                    metadata={"role": "RIGID_ATTACHMENT_OR_TERMINAL_INHERIT"},
                )
            )
        else:
            require(source_id in source, "ROLE_WITNESS_SOURCE_ROLE_MISSING:" + source_id)
            rows.append(
                MotionRoleBindingV1IR(
                    str(target),
                    str(source_id),
                    WORLD_ROTATION_DELTA,
                    root_translation_from_source_world=False,
                    metadata={
                        "role": "DEFORM_ROTATION_EVIDENCE",
                        "source_control_topology_is_target_authority": False,
                    },
                )
            )
    return tuple(rows)


def lbs(vertices, weights, target_pose, target_rest):
    vertices = np.asarray(vertices, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    homo = np.concatenate((vertices, np.ones((len(vertices), 1))), axis=1)
    out = []
    skin_matrices_all = []
    for pose in np.asarray(target_pose, dtype=np.float64):
        skin_matrices = np.asarray([
            pose[j] @ np.linalg.inv(target_rest[j])
            for j in range(len(target_rest))
        ])
        transformed = np.einsum("jab,nb->jna", skin_matrices, homo, optimize=True)[:, :, :3]
        out.append(np.einsum("nj,jna->na", weights, transformed, optimize=True))
        skin_matrices_all.append(skin_matrices)
    poses = np.asarray(out)
    require(np.isfinite(poses).all(), "ROLE_WITNESS_LBS_NONFINITE")
    return poses, np.asarray(skin_matrices_all)


def dynamic_metrics(rest, poses, faces):
    faces = np.asarray(faces, dtype=np.int64)
    a, b, c = faces[:, 0], faces[:, 1], faces[:, 2]
    rest_edges = np.stack((
        np.linalg.norm(rest[b] - rest[a], axis=1),
        np.linalg.norm(rest[c] - rest[b], axis=1),
        np.linalg.norm(rest[a] - rest[c], axis=1),
    ), axis=1)
    rest_area2 = np.linalg.norm(np.cross(rest[b] - rest[a], rest[c] - rest[a]), axis=1)
    rows = []
    for frame_index, vertices in enumerate(poses):
        edges = np.stack((
            np.linalg.norm(vertices[b] - vertices[a], axis=1),
            np.linalg.norm(vertices[c] - vertices[b], axis=1),
            np.linalg.norm(vertices[a] - vertices[c], axis=1),
        ), axis=1)
        ratio = edges / np.maximum(rest_edges, 1.0e-12)
        symmetric = np.maximum(ratio, 1.0 / np.maximum(ratio, 1.0e-12))
        area2 = np.linalg.norm(np.cross(vertices[b] - vertices[a], vertices[c] - vertices[a]), axis=1)
        area_ratio = area2 / np.maximum(rest_area2, 1.0e-12)
        rows.append({
            "frame_index": int(frame_index),
            "edge_ratio_p99": float(np.percentile(symmetric, 99)),
            "edge_ratio_max": float(np.max(symmetric)),
            "area_ratio_min": float(np.min(area_ratio)),
        })
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rigid-witness-npz", required=True)
    ap.add_argument("--teacher-npz", required=True)
    ap.add_argument("--teacher-audit-json", required=True)
    ap.add_argument("--motion-json", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args(argv)

    rigid_path = Path(args.rigid_witness_npz).resolve()
    teacher_path = Path(args.teacher_npz).resolve()
    audit_path = Path(args.teacher_audit_json).resolve()
    motion_path = Path(args.motion_json).resolve()

    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    bone_names = tuple(map(str, audit["bone_names"]))
    motion_payload = json.loads(motion_path.read_text(encoding="utf-8"))
    require(str(motion_payload.get("schema") or "") == "RealSaS.MotionSourceClip.v2", "ROLE_WITNESS_MOTION_SCHEMA")
    require(str(motion_payload.get("coordinate_frame") or "") == "REALSAS_OBJECT_FRAME_V1", "ROLE_WITNESS_MOTION_FRAME")

    with np.load(teacher_path, allow_pickle=False) as z:
        raw_heads = np.asarray(z["bone_heads_source"], dtype=np.float64)
        raw_parents = np.asarray(z["parents"], dtype=np.int64)
        source_vertices_raw = np.asarray(z["vertices_source"], dtype=np.float64)
        source_faces = np.asarray(z["faces"], dtype=np.int64)
        source_skin = np.asarray(z["skin"], dtype=np.float64)[:, :23]
    with np.load(rigid_path, allow_pickle=False) as z:
        vertices = np.asarray(z["vertices_rest"], dtype=np.float64)
        faces = np.asarray(z["faces"], dtype=np.int64)
        weights = np.asarray(z["weights_mechanical"], dtype=np.float64)
        presentation_face_visible = np.asarray(z["presentation_face_visible"], dtype=bool)

    source_ids, _, source_rest, source_pose, times = source_pose_matrices(motion_payload)
    C, target_names, target_parents, target_rest = target_rest_matrices(
        bone_names=bone_names,
        raw_heads=raw_heads,
        raw_parents=raw_parents,
    )
    bindings = bindings_for_knight(target_names, source_ids)
    source_extent = max(
        float(np.linalg.norm(source_rest[i, :3, 3] - source_rest[0, :3, 3]))
        for i in range(len(source_rest))
    )
    target_extent = max(
        float(np.linalg.norm(target_rest[i, :3, 3] - target_rest[0, :3, 3]))
        for i in range(len(target_rest))
    )
    target_pose, retarget_report = retarget_world_rotation_deltas_v1(
        source_rest_global=source_rest,
        source_pose_global=source_pose,
        source_joint_ids=source_ids,
        target_rest_global=target_rest,
        target_joint_ids=target_names,
        target_parent_indices=target_parents,
        bindings=bindings,
        root_translation_scale=target_extent / max(source_extent, 1.0e-12),
    )
    poses, skin_matrices = lbs(vertices, weights, target_pose, target_rest)

    source_vertices = (C @ source_vertices_raw.T).T
    source_poses, _ = lbs(source_vertices, source_skin, target_pose, target_rest)
    tessa_dynamic = dynamic_metrics(vertices, poses, faces)
    source_dynamic = dynamic_metrics(source_vertices, source_poses, source_faces)

    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "TESSA_KNIGHT_ROLE_AWARE_MOTION_WITNESS.npz",
        vertices_rest=vertices,
        faces=faces,
        poses=poses,
        target_pose_global=target_pose,
        target_rest_global=target_rest,
        skin_matrices=skin_matrices,
        times=times,
        presentation_face_visible=presentation_face_visible,
        target_parent_indices=target_parents,
    )
    report = {
        "schema": SCHEMA,
        "status": "PASS_RESEARCH_WITNESS__NOT_PRODUCT_AUTHORITY",
        "input_sha256": {
            "rigid_witness_npz": sha256(rigid_path),
            "teacher_npz": sha256(teacher_path),
            "teacher_audit_json": sha256(audit_path),
            "motion_json": sha256(motion_path),
        },
        "retarget_report": retarget_report,
        "role_map": ROLE_MAP,
        "source_teacher_dynamic": {
            "worst_edge_ratio_p99": float(max(row["edge_ratio_p99"] for row in source_dynamic)),
            "worst_edge_ratio_max": float(max(row["edge_ratio_max"] for row in source_dynamic)),
            "minimum_area_ratio": float(min(row["area_ratio_min"] for row in source_dynamic)),
        },
        "tessa_dynamic": {
            "worst_edge_ratio_p99": float(max(row["edge_ratio_p99"] for row in tessa_dynamic)),
            "worst_edge_ratio_max": float(max(row["edge_ratio_max"] for row in tessa_dynamic)),
            "minimum_area_ratio": float(min(row["area_ratio_min"] for row in tessa_dynamic)),
            "source_to_tessa_worst_edge_ratio_p99_delta": float(
                max(abs(a["edge_ratio_p99"] - b["edge_ratio_p99"]) for a, b in zip(source_dynamic, tessa_dynamic))
            ),
        },
        "claims": {
            "source_control_topology_used_as_target_deform_topology": False,
            "target_bone_lengths_preserved": bool(retarget_report["max_bone_length_relative_residual"] <= 1.0e-8),
            "rigid_equipment_contract_inherited_from_input": True,
            "product_authority": False,
            "motion_quality_pass": False,
        },
    }
    report_path = out / "TESSA_KNIGHT_ROLE_AWARE_MOTION_WITNESS.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    print("TESSA_KNIGHT_ROLE_AWARE_MOTION_WITNESS_PASS", report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
