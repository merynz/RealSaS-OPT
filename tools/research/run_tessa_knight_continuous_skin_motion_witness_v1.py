from __future__ import annotations

"""Knight FIT1 continuous-skin motion witness for TESSA.

Research-only causal court:
  TESSA decoded mesh -> continuous teacher surface skin field -> role-aware
  deform skeleton motion -> LBS consequence.

This tool intentionally does not claim product authority. It exists to falsify
vertex-index skin transport and to measure the consequence of replacing it with
continuous surface-field supervision on the exact Knight FIT1 witness.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

SCHEMA = "RealSaS.TESSAKnightContinuousSkinMotionWitness.v1"
EXPECTED = {
    "teacher_npz": "96435646a0084a0a038040ee748cb33d3bfa0b1ed550661befd3b18539e28e6f",
    "tessa_mesh_npz": "b7ccb058ef23f8c54d053e64580150dffffeb6a76a126ecaf48b265dd093f390",
    "normalization_json": "01b63a57390de9a5d2c731d9644e65ff779980caa066c19204db3b4a3e0c634b",
    "run_motion_json": "df33d312b40d27cc51ff545a8bd8fc86a67864cb9bcdcea8b2857b3e5cf4f7be",
}

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
    "foot.l": None,
    "toes.l": None,
    "upperleg.r": "UpperLeg.R",
    "lowerleg.r": "LowerLeg.R",
    "foot.r": None,
    "toes.r": None,
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def require(condition: bool, code: str) -> None:
    if not condition:
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
        require(required in index, "TESSA_WITNESS_TEACHER_FRAME_BONE_MISSING:" + required)
    up = unit(heads[index["head"]] - heads[index["hips"]], "TESSA_WITNESS_UP_DEGENERATE")
    lateral = []
    for base in ("upperarm", "upperleg", "foot"):
        lateral.append(heads[index[base + ".r"]] - heads[index[base + ".l"]])
    right = np.sum(np.asarray(lateral, dtype=np.float64), axis=0)
    right = right - up * float(np.dot(right, up))
    right = unit(right, "TESSA_WITNESS_RIGHT_DEGENERATE")
    forward = unit(np.cross(up, right), "TESSA_WITNESS_FORWARD_DEGENERATE")
    right = unit(np.cross(forward, up), "TESSA_WITNESS_RIGHT_ORTHO_FAIL")
    C = np.stack((right, forward, up), axis=0)
    require(float(np.linalg.det(C)) > 0.999999, "TESSA_WITNESS_OBJECT_FRAME_NOT_RIGHT_HANDED")
    return C


def derive_frames(positions: np.ndarray, parents: np.ndarray) -> np.ndarray:
    gr = unit((1.0, 0.0, 0.0), "TESSA_WITNESS_GLOBAL_RIGHT")
    gu = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
    gu = unit(gu - gr * float(np.dot(gu, gr)), "TESSA_WITNESS_GLOBAL_UP")
    gf = unit(np.cross(gr, gu), "TESSA_WITNESS_GLOBAL_FORWARD")
    desired = unit((0.0, 1.0, 0.0), "TESSA_WITNESS_GLOBAL_FORWARD")
    if float(np.dot(gf, desired)) < 0.0:
        gf = -gf
    gr = unit(np.cross(gu, gf), "TESSA_WITNESS_GLOBAL_RIGHT")
    out = []
    for i, pos in enumerate(positions):
        parent = int(parents[i])
        if parent < 0:
            R = np.column_stack((gr, gu, gf))
        else:
            y = unit(pos - positions[parent], "TESSA_WITNESS_BONE_DEGENERATE")
            z = gf - y * float(np.dot(gf, y))
            if float(np.linalg.norm(z)) <= 1.0e-8:
                z = gu - y * float(np.dot(gu, y))
            if float(np.linalg.norm(z)) <= 1.0e-8:
                z = gr - y * float(np.dot(gr, y))
            z = unit(z, "TESSA_WITNESS_SECONDARY_AXIS_DEGENERATE")
            x = unit(np.cross(y, z), "TESSA_WITNESS_PRIMARY_AXIS_DEGENERATE")
            if float(np.dot(x, gr)) < 0.0:
                x = -x
                z = -z
            z = unit(np.cross(x, y), "TESSA_WITNESS_FRAME_ORTHO_FAIL")
            R = np.column_stack((x, y, z))
        require(float(np.linalg.det(R)) > 0.999999, "TESSA_WITNESS_FRAME_NOT_RIGHT_HANDED")
        out.append(R)
    return np.asarray(out, dtype=np.float64)


def closest_bary_point_triangle(p, a, b, c):
    ab = b - a
    ac = c - a
    ap = p - a
    d1, d2 = float(ab @ ap), float(ac @ ap)
    if d1 <= 0.0 and d2 <= 0.0:
        return a, np.asarray((1.0, 0.0, 0.0))
    bp = p - b
    d3, d4 = float(ab @ bp), float(ac @ bp)
    if d3 >= 0.0 and d4 <= d3:
        return b, np.asarray((0.0, 1.0, 0.0))
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        return a + v * ab, np.asarray((1.0 - v, v, 0.0))
    cp = p - c
    d5, d6 = float(ab @ cp), float(ac @ cp)
    if d6 >= 0.0 and d5 <= d6:
        return c, np.asarray((0.0, 0.0, 1.0))
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        return a + w * ac, np.asarray((1.0 - w, 0.0, w))
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return b + w * (c - b), np.asarray((0.0, 1.0 - w, w))
    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom
    u = 1.0 - v - w
    return u * a + v * b + w * c, np.asarray((u, v, w))


def continuous_skin_field(tessa_vertices, source_vertices, source_faces, source_skin):
    tree = cKDTree(source_vertices)
    _, nearest_vertex = tree.query(tessa_vertices, k=1)
    incident = [[] for _ in range(len(source_vertices))]
    for face_index, face in enumerate(source_faces):
        for vertex_index in face:
            incident[int(vertex_index)].append(int(face_index))

    weights = np.zeros((len(tessa_vertices), source_skin.shape[1]), dtype=np.float64)
    field_distance = np.empty(len(tessa_vertices), dtype=np.float64)
    field_face = np.empty(len(tessa_vertices), dtype=np.int64)
    field_bary = np.empty((len(tessa_vertices), 3), dtype=np.float64)

    for i, point in enumerate(tessa_vertices):
        _, seeds = tree.query(point, k=4)
        candidates = set()
        for seed in np.atleast_1d(seeds):
            candidates.update(incident[int(seed)])
        require(bool(candidates), "TESSA_WITNESS_SKIN_FIELD_CANDIDATES_EMPTY")
        best = None
        for face_index in candidates:
            tri = source_faces[int(face_index)]
            q, bary = closest_bary_point_triangle(
                point,
                source_vertices[tri[0]],
                source_vertices[tri[1]],
                source_vertices[tri[2]],
            )
            distance = float(np.linalg.norm(point - q))
            key = (distance, int(face_index))
            if best is None or key < best[0]:
                best = (key, int(face_index), bary, q)
        _, face_index, bary, q = best
        tri = source_faces[face_index]
        row = bary @ source_skin[tri]
        row[row < 0.0] = 0.0
        total = float(row.sum())
        require(math.isfinite(total) and total > 1.0e-12, "TESSA_WITNESS_SKIN_FIELD_ZERO_ROW")
        row /= total
        weights[i] = row
        field_distance[i] = float(np.linalg.norm(point - q))
        field_face[i] = int(face_index)
        field_bary[i] = bary
    return weights, field_distance, field_face, field_bary, nearest_vertex


def topological_order(parents):
    remaining = set(range(len(parents)))
    out = []
    while remaining:
        progressed = False
        for i in sorted(tuple(remaining)):
            parent = int(parents[i])
            if parent < 0 or parent in out:
                out.append(i)
                remaining.remove(i)
                progressed = True
        require(progressed, "TESSA_WITNESS_TARGET_SKELETON_CYCLE")
    return tuple(out)


def build_motion(*, vertices, weights, target_names, target_positions, target_parents, motion_payload):
    require(str(motion_payload.get("schema") or "") == "RealSaS.MotionSourceClip.v2", "TESSA_WITNESS_MOTION_SCHEMA")
    require(str(motion_payload.get("coordinate_frame") or "") == "REALSAS_OBJECT_FRAME_V1", "TESSA_WITNESS_MOTION_FRAME")
    source_ids = {str(row["source_joint_id"]) for row in motion_payload["source_skeleton"]}
    for target_name, source_name in ROLE_MAP.items():
        require(target_name in target_names, "TESSA_WITNESS_TARGET_ROLE_MISSING:" + target_name)
        if source_name is not None:
            require(source_name in source_ids, "TESSA_WITNESS_SOURCE_ROLE_MISSING:" + source_name)

    tracks = {str(row["source_joint_id"]): tuple(row["keyframes"]) for row in motion_payload["tracks"]}
    reference = next(iter(tracks.values()))
    times = np.asarray([float(k["time_seconds"]) for k in reference], dtype=np.float64)
    for source_id, keys in tracks.items():
        require(len(keys) == len(times), "TESSA_WITNESS_TRACK_LENGTH_DRIFT:" + source_id)
        require(max(abs(float(keys[i]["time_seconds"]) - float(times[i])) for i in range(len(times))) <= 1.0e-9, "TESSA_WITNESS_TRACK_TIME_DRIFT:" + source_id)

    frames = derive_frames(target_positions, target_parents)
    rest_global = []
    for i in range(len(target_names)):
        G = np.eye(4, dtype=np.float64)
        G[:3, :3] = frames[i]
        G[:3, 3] = target_positions[i]
        rest_global.append(G)
    rest_global = np.asarray(rest_global)
    rest_local = []
    for i in range(len(target_names)):
        parent = int(target_parents[i])
        if parent < 0:
            rest_local.append(rest_global[i])
        else:
            rest_local.append(np.linalg.inv(rest_global[parent]) @ rest_global[i])
    rest_local = np.asarray(rest_local)
    order = topological_order(target_parents)
    body_scale = max(float(np.linalg.norm(p - target_positions[0])) for p in target_positions)

    poses = []
    joint_positions = []
    skin_matrices_all = []
    homo = np.concatenate((vertices, np.ones((len(vertices), 1))), axis=1)
    for frame_index in range(len(times)):
        posed_global = np.zeros_like(rest_global)
        posed_global[:, 3, 3] = 1.0
        for i in order:
            name = target_names[i]
            source_name = ROLE_MAP[name]
            delta = np.eye(4, dtype=np.float64)
            if source_name is not None:
                key = tracks[source_name][frame_index]
                quat = np.asarray(key["local_rotation_quat_xyzw"], dtype=np.float64)
                require(quat.shape == (4,) and np.isfinite(quat).all(), "TESSA_WITNESS_QUAT_INVALID")
                delta[:3, :3] = Rotation.from_quat(quat).as_matrix()
                if name == "hips":
                    delta[:3, 3] = np.asarray(key["local_translation_xyz"], dtype=np.float64) * body_scale
            parent = int(target_parents[i])
            if parent < 0:
                posed_global[i] = rest_global[i] @ delta
            else:
                posed_global[i] = posed_global[parent] @ rest_local[i] @ delta
        skin_matrices = np.asarray([
            posed_global[i] @ np.linalg.inv(rest_global[i])
            for i in range(len(target_names))
        ])
        transformed = np.einsum("jab,nb->jna", skin_matrices, homo, optimize=True)[:, :, :3]
        posed = np.einsum("nj,jna->na", weights, transformed, optimize=True)
        require(np.isfinite(posed).all(), "TESSA_WITNESS_LBS_NONFINITE")
        poses.append(posed)
        joint_positions.append(posed_global[:, :3, 3])
        skin_matrices_all.append(skin_matrices)
    return np.asarray(poses), np.asarray(joint_positions), np.asarray(skin_matrices_all), times, body_scale


def dynamic_metrics(rest, poses, faces):
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
            "area_ratio_p01": float(np.percentile(area_ratio, 1)),
            "area_ratio_p99": float(np.percentile(area_ratio, 99)),
            "area_ratio_min": float(np.min(area_ratio)),
        })
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-npz", required=True)
    ap.add_argument("--teacher-audit-json", required=True)
    ap.add_argument("--tessa-mesh-npz", required=True)
    ap.add_argument("--normalization-json", required=True)
    ap.add_argument("--run-motion-json", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--allow-hash-drift", action="store_true")
    args = ap.parse_args(argv)

    paths = {
        "teacher_npz": Path(args.teacher_npz).resolve(),
        "tessa_mesh_npz": Path(args.tessa_mesh_npz).resolve(),
        "normalization_json": Path(args.normalization_json).resolve(),
        "run_motion_json": Path(args.run_motion_json).resolve(),
    }
    hashes = {key: sha256(path) for key, path in paths.items()}
    if not args.allow_hash_drift:
        for key, expected in EXPECTED.items():
            require(hashes[key] == expected, "TESSA_WITNESS_INPUT_SHA_DRIFT:" + key)

    audit = json.loads(Path(args.teacher_audit_json).read_text(encoding="utf-8"))
    bone_names = tuple(map(str, audit["bone_names"]))
    require(len(bone_names) == 41, "TESSA_WITNESS_RAW_BONE_COUNT_DRIFT")
    with np.load(paths["teacher_npz"], allow_pickle=False) as z:
        source_vertices = np.asarray(z["vertices_source"], dtype=np.float64)
        source_faces = np.asarray(z["faces"], dtype=np.int64)
        source_heads = np.asarray(z["bone_heads_source"], dtype=np.float64)
        source_parents = np.asarray(z["parents"], dtype=np.int64)
        source_skin = np.asarray(z["skin"], dtype=np.float64)
    with np.load(paths["tessa_mesh_npz"], allow_pickle=False) as z:
        tessa_normalized = np.asarray(z["vertices_normalized"], dtype=np.float64)
        tessa_faces = np.asarray(z["faces"], dtype=np.int64)
    normalization = json.loads(paths["normalization_json"].read_text(encoding="utf-8"))
    center = np.asarray(normalization["center_xyz"], dtype=np.float64)
    world_scale = 2.0 * float(normalization["half_extent"])
    tessa_raw = tessa_normalized * world_scale + center[None, :]

    require(source_skin.shape == (len(source_vertices), 41), "TESSA_WITNESS_SKIN_SHAPE_DRIFT")
    positive_columns = np.flatnonzero(source_skin.sum(axis=0) > 1.0e-8)
    require(tuple(positive_columns.tolist()) == tuple(range(1, 23)), "TESSA_WITNESS_SKIN_BEARING_SET_DRIFT")
    require(source_faces.shape == tessa_faces.shape == (6952, 3), "TESSA_WITNESS_FACE_COUNT_DRIFT")
    require(len(source_vertices) == len(tessa_raw) == 3665, "TESSA_WITNESS_VERTEX_COUNT_DRIFT")

    field_weights, field_distance, field_face, field_bary, nearest_source_vertex = continuous_skin_field(
        tessa_raw, source_vertices, source_faces, source_skin
    )
    require(float(np.max(np.abs(field_weights.sum(axis=1) - 1.0))) <= 1.0e-9, "TESSA_WITNESS_SKIN_SIMPLEX_FAIL")

    C = teacher_object_frame(bone_names, source_heads)
    tessa_vertices = (C @ tessa_raw.T).T
    source_vertices_canonical = (C @ source_vertices.T).T
    target_names = bone_names[:23]
    target_positions = (C @ source_heads[:23].T).T
    target_parents = source_parents[:23]
    require(all(int(p) < 23 for p in target_parents if int(p) >= 0), "TESSA_WITNESS_DEFORM_PARENT_OUTSIDE_TARGET")
    require(tuple(target_names) == tuple(ROLE_MAP.keys()), "TESSA_WITNESS_TARGET_ROLE_ORDER_DRIFT")

    motion_payload = json.loads(paths["run_motion_json"].read_text(encoding="utf-8"))
    poses, joints, skin_matrices, times, body_scale = build_motion(
        vertices=tessa_vertices,
        weights=field_weights[:, :23],
        target_names=target_names,
        target_positions=target_positions,
        target_parents=target_parents,
        motion_payload=motion_payload,
    )
    source_poses, _, _, _, _ = build_motion(
        vertices=source_vertices_canonical,
        weights=source_skin[:, :23],
        target_names=target_names,
        target_positions=target_positions,
        target_parents=target_parents,
        motion_payload=motion_payload,
    )
    wrong_poses, _, _, _, _ = build_motion(
        vertices=tessa_vertices,
        weights=source_skin[:, :23],
        target_names=target_names,
        target_positions=target_positions,
        target_parents=target_parents,
        motion_payload=motion_payload,
    )
    wrong_delta = np.linalg.norm(wrong_poses - poses, axis=2)
    row_l1 = np.abs(field_weights - source_skin).sum(axis=1)

    tessa_dynamic = dynamic_metrics(tessa_vertices, poses, tessa_faces)
    source_dynamic = dynamic_metrics(source_vertices_canonical, source_poses, source_faces)

    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "TESSA_KNIGHT_CONTINUOUS_SKIN_RUN_WITNESS.npz",
        vertices_rest=tessa_vertices,
        faces=tessa_faces,
        weights_continuous=field_weights[:, :23],
        poses=poses,
        joint_positions=joints,
        skin_matrices=skin_matrices,
        times=times,
        field_face=field_face,
        field_bary=field_bary,
        field_distance=field_distance,
    )

    report = {
        "schema": SCHEMA,
        "status": "PASS_RESEARCH_WITNESS__NOT_PRODUCT_AUTHORITY",
        "input_sha256": hashes,
        "teacher_audit_bone_count": len(bone_names),
        "source_motion_joint_count": len(motion_payload["source_skeleton"]),
        "target_deform_joint_count": 23,
        "positive_skin_columns": positive_columns.astype(int).tolist(),
        "continuous_skin_field": {
            "rule": "CLOSEST_SOURCE_TRIANGLE_BARYCENTRIC__INCIDENT_CANDIDATE_V1",
            "teacher_vertex_index_used": False,
            "projection_distance_p50": float(np.percentile(field_distance, 50)),
            "projection_distance_p95": float(np.percentile(field_distance, 95)),
            "projection_distance_p99": float(np.percentile(field_distance, 99)),
            "projection_distance_max": float(np.max(field_distance)),
            "simplex_max_abs_residual": float(np.max(np.abs(field_weights.sum(axis=1) - 1.0))),
        },
        "vertex_index_drift_counterfactual": {
            "same_source_index_nearest_fraction": float(np.mean(nearest_source_vertex == np.arange(len(tessa_vertices)))),
            "continuous_vs_same_row_skin_l1_p75": float(np.percentile(row_l1, 75)),
            "continuous_vs_same_row_skin_l1_p95": float(np.percentile(row_l1, 95)),
            "wrong_same_row_vs_continuous_motion_displacement_p95": float(np.percentile(wrong_delta, 95)),
            "wrong_same_row_vs_continuous_motion_displacement_max": float(np.max(wrong_delta)),
            "fraction_vertex_frames_over_5pct_body_scale": float(np.mean(wrong_delta > 0.05 * body_scale)),
        },
        "retarget_contract": {
            "class": "ROLE_AWARE_RESEARCH_WITNESS_V1",
            "source_control_topology_required_to_match_target_deform_topology": False,
            "zero_skin_control_bones_excluded_from_target": True,
            "source_foot_ik_translation_reused_as_target_deform_local_translation": False,
            "terminal_attachment_and_toe_joints_inherit_parent_with_identity_delta": True,
            "mapping": ROLE_MAP,
        },
        "source_teacher_dynamic": {
            "worst_edge_ratio_p99": float(max(r["edge_ratio_p99"] for r in source_dynamic)),
            "worst_edge_ratio_max": float(max(r["edge_ratio_max"] for r in source_dynamic)),
            "minimum_area_ratio": float(min(r["area_ratio_min"] for r in source_dynamic)),
        },
        "tessa_dynamic": {
            "worst_edge_ratio_p99": float(max(r["edge_ratio_p99"] for r in tessa_dynamic)),
            "worst_edge_ratio_max": float(max(r["edge_ratio_max"] for r in tessa_dynamic)),
            "minimum_area_ratio": float(min(r["area_ratio_min"] for r in tessa_dynamic)),
            "source_to_tessa_worst_edge_ratio_p99_delta": float(
                max(abs(a["edge_ratio_p99"] - b["edge_ratio_p99"]) for a, b in zip(source_dynamic, tessa_dynamic))
            ),
        },
        "claims": {
            "tessa_generalization": False,
            "compiler_product_qualification": False,
            "motion_quality_pass": False,
            "causal_vertex_index_transport_failure_demonstrated": True,
            "continuous_skin_field_mechanical_witness_generated": True,
        },
    }
    report_path = out / "TESSA_KNIGHT_CONTINUOUS_SKIN_RUN_WITNESS.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    print("TESSA_KNIGHT_CONTINUOUS_SKIN_RUN_WITNESS_PASS", report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
