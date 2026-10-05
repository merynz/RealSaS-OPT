from __future__ import annotations

"""Evaluation-only ATLAS core alignment court against raw Knight teacher23.

The learned/Compiler-qualified ATLAS skeleton contains a historical 28-joint
mechanically meaningful projection. Raw source authority is 41 bones, with the
fixed deform-motion test chain defined as root0 + skin-positive raw bones 1..22.
This court does not rename ATLAS joints or mint product authority. It asks only:
can 23 learned qualified joints be put in a global one-to-one geometric
correspondence with teacher23, and does that correspondence preserve the exact
22 parent edges? The remaining five ATLAS joints must be leaves, consistent with
the historical synthetic-tip role count.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


SCHEMA = "RealSaS.ATLASTeacher23CoreAlignmentAudit.v1"
EXPECTED_ATLAS_JOINTS = 28
EXPECTED_TEACHER_RAW_BONES = 41
EXPECTED_TEACHER23 = 23
EXPECTED_SYNTHETIC_TIPS = 5


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--atlas-skeleton-json", type=Path, required=True)
    ap.add_argument("--teacher-npz", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    atlas_path = args.atlas_skeleton_json.resolve()
    teacher_path = args.teacher_npz.resolve()
    atlas = json.loads(atlas_path.read_text(encoding="utf-8"))
    joints = tuple(atlas["joints"])
    if len(joints) != EXPECTED_ATLAS_JOINTS:
        raise RuntimeError(f"ATLAS_JOINT_COUNT_DRIFT::{len(joints)}")

    atlas_ids = tuple(str(row["canonical_joint_id"]) for row in joints)
    if len(set(atlas_ids)) != len(atlas_ids):
        raise RuntimeError("ATLAS_DUPLICATE_JOINT_ID")
    atlas_index = {jid: i for i, jid in enumerate(atlas_ids)}
    atlas_pos = np.asarray([row["position"] for row in joints], dtype=np.float64)
    atlas_parent = np.asarray([
        -1 if row.get("parent_canonical_id") is None else atlas_index[str(row["parent_canonical_id"])]
        for row in joints
    ], dtype=np.int64)
    if np.sum(atlas_parent < 0) != 1:
        raise RuntimeError("ATLAS_ROOT_COUNT_DRIFT")

    with np.load(teacher_path, allow_pickle=False) as z:
        heads = np.asarray(z["bone_heads_source"], dtype=np.float64)
        parents = np.asarray(z["parents"], dtype=np.int64)
        skin = np.asarray(z["skin"], dtype=np.float64)
    if heads.shape != (EXPECTED_TEACHER_RAW_BONES, 3):
        raise RuntimeError(f"TEACHER_HEAD_SHAPE_DRIFT::{heads.shape}")
    if parents.shape != (EXPECTED_TEACHER_RAW_BONES,) or skin.shape[1] != EXPECTED_TEACHER_RAW_BONES:
        raise RuntimeError("TEACHER_RIG_SHAPE_DRIFT")

    positive = tuple(map(int, np.flatnonzero(skin.sum(axis=0) > 0.0)))
    if positive != tuple(range(1, 23)):
        raise RuntimeError(f"TEACHER_SKIN_POSITIVE_COLUMNS_DRIFT::{positive}")
    teacher_raw_indices = (0,) + positive
    teacher_pos = heads[np.asarray(teacher_raw_indices, dtype=np.int64)]
    teacher_parent_raw = parents[np.asarray(teacher_raw_indices, dtype=np.int64)]
    raw_to_local = {raw: local for local, raw in enumerate(teacher_raw_indices)}
    teacher_parent = np.asarray([
        -1 if int(raw_parent) < 0 else raw_to_local[int(raw_parent)]
        for raw_parent in teacher_parent_raw
    ], dtype=np.int64)

    cost = np.linalg.norm(atlas_pos[:, None, :] - teacher_pos[None, :, :], axis=2)
    atlas_rows, teacher_cols = linear_sum_assignment(cost)
    if len(atlas_rows) != EXPECTED_TEACHER23 or len(set(map(int, teacher_cols))) != EXPECTED_TEACHER23:
        raise RuntimeError("ATLAS_TEACHER23_ASSIGNMENT_CARDINALITY_FAIL")

    atlas_to_teacher = {int(a): int(t) for a, t in zip(atlas_rows, teacher_cols)}
    matched_distance = np.asarray([cost[a, t] for a, t in zip(atlas_rows, teacher_cols)], dtype=np.float64)

    edge_rows = []
    edge_match = 0
    edge_total = 0
    for atlas_i, teacher_i in sorted(atlas_to_teacher.items(), key=lambda x: x[1]):
        tp = int(teacher_parent[teacher_i])
        ap = int(atlas_parent[atlas_i])
        if tp < 0:
            ok = ap < 0
        else:
            edge_total += 1
            mapped_parent = atlas_to_teacher.get(ap)
            ok = mapped_parent == tp
            edge_match += int(ok)
        edge_rows.append({
            "teacher23_local_index": int(teacher_i),
            "teacher_raw_bone_index": int(teacher_raw_indices[teacher_i]),
            "atlas_joint_index": int(atlas_i),
            "atlas_joint_id": atlas_ids[atlas_i],
            "euclidean_distance": float(cost[atlas_i, teacher_i]),
            "teacher_parent_local_index": None if tp < 0 else int(tp),
            "atlas_parent_index": None if ap < 0 else int(ap),
            "parent_edge_preserved": bool(ok),
        })

    unmatched = tuple(sorted(set(range(EXPECTED_ATLAS_JOINTS)).difference(atlas_to_teacher)))
    children = {i: [] for i in range(EXPECTED_ATLAS_JOINTS)}
    for child, parent in enumerate(atlas_parent):
        if int(parent) >= 0:
            children[int(parent)].append(int(child))
    unmatched_are_leaves = all(len(children[i]) == 0 for i in unmatched)

    structural_pass = bool(
        edge_total == 22
        and edge_match == 22
        and len(unmatched) == EXPECTED_SYNTHETIC_TIPS
        and unmatched_are_leaves
        and bool(edge_rows[0]["parent_edge_preserved"])
    )

    result = {
        "schema": SCHEMA,
        "status": "PASS" if structural_pass else "FAIL",
        "scope": "EVALUATION_ONLY__NO_PRODUCT_AUTHORITY__NO_RETRAIN",
        "inputs": {
            "atlas_skeleton_sha256": sha256(atlas_path),
            "atlas_skeleton_lineage_hash": atlas.get("skeleton_lineage_hash"),
            "teacher_npz_sha256": sha256(teacher_path),
        },
        "authority": {
            "raw_teacher_bone_count": EXPECTED_TEACHER_RAW_BONES,
            "teacher23_rule": "ROOT0_PLUS_SKIN_POSITIVE_RAW_BONES_1_TO_22",
            "atlas_historical_joint_count": EXPECTED_ATLAS_JOINTS,
            "atlas_extra_joint_interpretation": "DERIVED_TERMINAL_TIPS__NOT_RAW41_TRUTH",
            "fbx_object_packaging_authority": False,
            "geometry_assignment_is_evaluation_correspondence_not_identity_authority": True,
        },
        "geometry_assignment": {
            "matched_joint_count": int(len(atlas_rows)),
            "distance_mean": float(matched_distance.mean()),
            "distance_p50": float(np.percentile(matched_distance, 50)),
            "distance_p95": float(np.percentile(matched_distance, 95)),
            "distance_max": float(matched_distance.max()),
            "rows": edge_rows,
        },
        "topology": {
            "teacher_nonroot_edge_count": int(edge_total),
            "parent_edges_preserved": int(edge_match),
            "parent_edge_recall": float(edge_match / max(edge_total, 1)),
        },
        "unmatched_atlas": {
            "count": int(len(unmatched)),
            "joint_indices": list(map(int, unmatched)),
            "joint_ids": [atlas_ids[i] for i in unmatched],
            "all_are_leaves": bool(unmatched_are_leaves),
        },
        "structural_core_alignment_pass": structural_pass,
        "claims": {
            "atlas_product_authority": False,
            "atlas_generalization": False,
            "teacher_identity_minted_from_geometry": False,
            "retrain_required_by_this_court": False if structural_pass else None,
        },
        "next": "ATLAS_ONLY_DYNAMIC_SUBSTITUTION_ON_TESSA_CARRIER" if structural_pass else "ATLAS_ALIGNMENT_FAILURE_DIAGNOSIS",
    }
    args.out.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.out.resolve().write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "matched": result["geometry_assignment"]["matched_joint_count"],
        "edge_recall": result["topology"]["parent_edge_recall"],
        "distance_p95": result["geometry_assignment"]["distance_p95"],
        "distance_max": result["geometry_assignment"]["distance_max"],
        "unmatched_leaves": result["unmatched_atlas"]["count"],
        "next": result["next"],
    }, sort_keys=True), flush=True)
    if not structural_pass:
        raise AssertionError("ATLAS_TEACHER23_CORE_ALIGNMENT_FAIL")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
