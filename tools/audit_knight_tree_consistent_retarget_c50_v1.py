from __future__ import annotations

"""C5.0 tree-consistent retarget ceiling.

Research-only owner attribution. No model fit, no product promotion.

Compare the current independent nearest-cost source->target retarget mapping against
a target-tree-consistent dynamic-programming mapping. The constrained solver uses
the same anonymous geometry/side unary cost as the current demo mapper, but enforces:

    target parent maps to the same source control OR
    target parent maps to a source ancestor of the target child's source control.

The court evaluates both:
- frozen projected MIRA carrier weights;
- the best C4.1 actual-motion optimized skin state.

This isolates the motion-adapter contribution before changing ATLAS.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from compiler.realsas_compiler_core.artifact_codec_v2 import (
    canonical_mesh_candidate_from_dict,
    mesh_policy_from_dict,
    qualified_camera_set_from_dict,
    qualified_skeleton_from_dict,
)
from compiler.realsas_compiler_core.joint_frames_v1 import (
    derive_joint_frames_from_rows,
    derive_joint_frames_from_skeleton,
    object_vector_to_joint_local,
)
from compiler.realsas_compiler_core.motion_compile_v2 import (
    CanonicalJointTrack3DIR,
    MotionKeyframe3DIR,
    _target_body_scale,
    _tree,
)
from compiler.realsas_compiler_core.motion_dynamic_proof_v2 import (
    _joint_pose_v2,
    _quat_matrix_xyzw,
    _slerp,
)
from compiler.realsas_compiler_services.orchestrator.adapters.adapter_io import (
    stage_output_payload,
)
from tools.audit_knight_canonical_caa_mechanics_geometry_lock_v1 import (
    CLIPS,
    FULL_MOTION_SAMPLES,
)
from tools.audit_knight_mira_actual_motion_weight_oracle_c41_v1 import (
    _faces,
    _hard_actual_motion,
)
from tools.demo.render_knight_motion_preview_v1 import (
    _ctx,
    _mapping,
    _matrix_to_quat_xyzw,
    _same_source_chain_group_sizes,
    _tracks_for_clip,
)
from tools.inference.refined_surface_rig_skin_v1 import write


def _ancestors(parent: dict[str, str | None], node: str) -> set[str]:
    out = set()
    cur = parent.get(node)
    while cur is not None and cur not in out:
        out.add(cur)
        cur = parent.get(cur)
    return out


def _unary_cost(tf: np.ndarray, sf: np.ndarray) -> float:
    c = 2.0 * float(np.linalg.norm(tf[:3] - sf[:3]))
    c += 0.45 * abs(float(tf[3] - sf[3]))
    c += 0.35 * abs(float(tf[4] - sf[4]))
    c += 0.35 * abs(float(tf[5] - sf[5]))
    if (
        abs(float(tf[0])) > 0.05
        and abs(float(sf[0])) > 0.05
        and math.copysign(1.0, float(tf[0]))
        != math.copysign(1.0, float(sf[0]))
    ):
        c += 4.0
    return float(c)


def _tree_consistent_mapping(payload: dict, skeleton, source_report: dict):
    srows = tuple(payload.get("source_skeleton") or ())
    sids, spar, schildren, spos, sfeat, sroot = _tree(
        srows,
        id_key="source_joint_id",
        parent_key="parent_source_joint_id",
        pos_key="rest_position",
    )
    roles = {
        str(row["source_joint_id"]): row
        for row in source_report.get("bone_roles") or ()
    }
    candidates = tuple(
        sid
        for sid in sids
        if sid == sroot or bool((roles.get(sid) or {}).get("has_mesh_influence"))
    )
    nonroot_candidates = tuple(sid for sid in candidates if sid != sroot)

    trows = tuple(
        {
            "joint_id": j.canonical_joint_id,
            "parent_id": j.parent_canonical_id,
            "position": j.position,
        }
        for j in skeleton.joints
    )
    tids, tpar, tchildren, tpos, tfeat, troot = _tree(
        trows,
        id_key="joint_id",
        parent_key="parent_id",
        pos_key="position",
    )

    sanc = {sid: _ancestors(spar, sid) for sid in sids}

    def allowed(parent_sid: str, child_sid: str) -> bool:
        return parent_sid == child_sid or parent_sid in sanc[child_sid]

    unary = {
        tid: {
            sid: _unary_cost(np.asarray(tfeat[tid]), np.asarray(sfeat[sid]))
            for sid in nonroot_candidates
        }
        for tid in tids
        if tid != troot
    }

    postorder = []
    def visit(tid: str):
        for child in sorted(tchildren.get(tid, ())):
            visit(str(child))
        postorder.append(tid)
    visit(troot)

    dp: dict[str, dict[str, float]] = {}
    choice: dict[tuple[str, str, str], str] = {}

    for tid in postorder:
        if tid == troot:
            continue
        dp[tid] = {}
        for sid in nonroot_candidates:
            total = unary[tid][sid]
            feasible = True
            for child in sorted(tchildren.get(tid, ())):
                child = str(child)
                options = [
                    (dp[child][csid], csid)
                    for csid in nonroot_candidates
                    if allowed(sid, csid) and np.isfinite(dp[child][csid])
                ]
                if not options:
                    feasible = False
                    break
                best_cost, best_sid = min(options, key=lambda row: (row[0], row[1]))
                total += float(best_cost)
                choice[(tid, sid, child)] = str(best_sid)
            dp[tid][sid] = float(total) if feasible else float("inf")

    mapping = {troot: sroot}
    details = []
    total = 0.0
    for child in sorted(tchildren.get(troot, ())):
        child = str(child)
        options = [
            (dp[child][sid], sid)
            for sid in nonroot_candidates
            if allowed(sroot, sid) and np.isfinite(dp[child][sid])
        ]
        if not options:
            raise RuntimeError("C50_TREE_CONSISTENT_ROOT_CHILD_INFEASIBLE")
        cost, sid = min(options, key=lambda row: (row[0], row[1]))
        total += float(cost)

        stack = [(child, str(sid))]
        while stack:
            tid, tsid = stack.pop()
            if tid in mapping:
                if mapping[tid] != tsid:
                    raise RuntimeError("C50_TREE_MAPPING_CONFLICT")
                continue
            mapping[tid] = tsid
            details.append(
                {
                    "target_joint_id": tid,
                    "source_joint_id": tsid,
                    "unary_cost": float(unary[tid][tsid]),
                }
            )
            for grandchild in sorted(tchildren.get(tid, ()), reverse=True):
                grandchild = str(grandchild)
                csid = choice.get((tid, tsid, grandchild))
                if csid is None:
                    opts = [
                        (dp[grandchild][x], x)
                        for x in nonroot_candidates
                        if allowed(tsid, x) and np.isfinite(dp[grandchild][x])
                    ]
                    if not opts:
                        raise RuntimeError("C50_TREE_MAPPING_CHILD_INFEASIBLE")
                    _, csid = min(opts, key=lambda row: (row[0], row[1]))
                stack.append((grandchild, str(csid)))

    if set(mapping) != set(tids):
        raise RuntimeError("C50_TREE_MAPPING_TARGET_ACCOUNTING_DRIFT")

    violations = []
    for tid, parent_tid in tpar.items():
        tid = str(tid)
        if parent_tid is None:
            continue
        parent_tid = str(parent_tid)
        ps = mapping[parent_tid]
        cs = mapping[tid]
        if not allowed(ps, cs):
            violations.append(
                {
                    "target_parent": parent_tid,
                    "target_child": tid,
                    "source_parent": ps,
                    "source_child": cs,
                }
            )
    if violations:
        raise RuntimeError("C50_TREE_CONSISTENT_SOLVER_VIOLATION")

    return mapping, details, srows, trows, tpar, float(total)


def _tracks_with_mapping(payload, skeleton, cameras, mapping, srows, tpar):
    source_frames = derive_joint_frames_from_rows(
        srows,
        joint_id_key="source_joint_id",
        parent_id_key="parent_source_joint_id",
        position_key="rest_position",
    )
    target_frames = derive_joint_frames_from_skeleton(skeleton, cameras=cameras)
    group_size = _same_source_chain_group_sizes(mapping, tpar)
    source_tracks = {
        str(row["source_joint_id"]): row
        for row in payload.get("tracks") or ()
    }
    target_scale = _target_body_scale(skeleton)
    out = {}

    for tid in sorted(mapping):
        sid = mapping[tid]
        raw = source_tracks.get(sid)
        if raw is None:
            raise RuntimeError(f"C50_SOURCE_TRACK_MISSING:{sid}")
        Rs = np.asarray(source_frames[sid].rotation_matrix, dtype=np.float64)
        Rt = np.asarray(target_frames[tid].rotation_matrix, dtype=np.float64)
        n = int(group_size.get(tid, 1))
        keys = []
        previous = None
        for row in raw.get("keyframes") or ():
            q_source = tuple(map(float, row["local_rotation_quat_xyzw"]))
            Qs = _quat_matrix_xyzw(q_source)
            Qobj = Rs @ Qs @ Rs.T
            Qt_full = Rt.T @ Qobj @ Rt
            qt_full = _matrix_to_quat_xyzw(Qt_full)
            qt = _slerp((0.0, 0.0, 0.0, 1.0), qt_full, 1.0 / n)
            if previous is not None and float(np.dot(previous, qt)) < 0.0:
                qt = tuple(-float(x) for x in qt)
            previous = np.asarray(qt, dtype=np.float64)

            source_local = np.asarray(
                row.get("local_translation_xyz") or (0.0, 0.0, 0.0),
                dtype=np.float64,
            )
            object_delta = Rs @ (source_local * float(target_scale))
            target_local = (
                np.asarray(
                    object_vector_to_joint_local(target_frames[tid], object_delta),
                    dtype=np.float64,
                )
                / float(n)
            )
            keys.append(
                MotionKeyframe3DIR(
                    time_seconds=float(row["time_seconds"]),
                    local_rotation_quat_xyzw=tuple(map(float, qt)),
                    local_translation_xyz=tuple(map(float, target_local)),
                    local_scale_xyz=(1.0, 1.0, 1.0),
                    metadata={
                        "research_only": True,
                        "source_joint_id": sid,
                        "shared_chain_divisor": n,
                        "frame_rebased": True,
                    },
                )
            )
        out[tid] = CanonicalJointTrack3DIR(
            canonical_joint_id=tid,
            source_joint_id=sid,
            channel_contract=(
                "LOCAL_ROTATION_QUAT_XYZW",
                "LOCAL_TRANSLATION_XYZ",
                "LOCAL_SCALE_XYZ",
            ),
            keyframes=tuple(keys),
            metadata={
                "research_only": True,
                "retarget_kind": "TREE_CONSISTENT_GEOMETRY_DP_V1",
                "source_delta_rebased_through_object_frame": True,
                "shared_source_chain_rotation_distributed": True,
            },
        )
    return out


def _hierarchy_violations(payload, skeleton, source_report, mapping):
    srows = tuple(payload.get("source_skeleton") or ())
    _, spar, _, _, _, _ = _tree(
        srows,
        id_key="source_joint_id",
        parent_key="parent_source_joint_id",
        pos_key="rest_position",
    )
    trows = tuple(
        {
            "joint_id": j.canonical_joint_id,
            "parent_id": j.parent_canonical_id,
            "position": j.position,
        }
        for j in skeleton.joints
    )
    _, tpar, _, _, _, _ = _tree(
        trows,
        id_key="joint_id",
        parent_key="parent_id",
        pos_key="position",
    )
    anc = {sid: _ancestors(spar, sid) for sid in spar}
    rows = []
    for child, parent in tpar.items():
        if parent is None:
            continue
        ps = mapping[str(parent)]
        cs = mapping[str(child)]
        if ps != cs and ps not in anc.get(cs, set()):
            rows.append(
                {
                    "target_parent": str(parent),
                    "target_child": str(child),
                    "source_parent": ps,
                    "source_child": cs,
                }
            )
    return rows


def _motion_matrices(*, payload, tracks, skeleton, cameras, joint_ids, clip_id):
    duration = float(payload["duration_seconds"])
    times = np.linspace(
        0.0,
        duration,
        FULL_MOTION_SAMPLES,
        endpoint=not bool(payload.get("loop")),
    )
    mats = []
    rows = []
    for frame_index, t in enumerate(times):
        skin, _positions, _frame_hash = _joint_pose_v2(
            skeleton=skeleton,
            tracks=tracks,
            time_seconds=float(t),
            cameras=cameras,
        )
        mats.append(
            np.stack([np.asarray(skin[jid], dtype=np.float64) for jid in joint_ids])
        )
        rows.append(
            {
                "clip_id": str(clip_id),
                "frame_index": int(frame_index),
                "time_seconds": float(t),
            }
        )
    return np.asarray(mats, np.float64), tuple(rows)


def _summarize(actual):
    return {
        k: v
        for k, v in actual.items()
        if k not in {"union_bad", "per_frame"}
    }


def main(args):
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ctx = _ctx(args.authority_root, args.run_id)
    rr = ctx["run_root"]

    candidate = canonical_mesh_candidate_from_dict(
        stage_output_payload(
            ctx,
            "18_CANONICAL_MESH_ADDRESSING_BUILD",
            "RealSaS.CanonicalMeshCandidateIR.v1",
        )
    )
    skeleton = qualified_skeleton_from_dict(
        stage_output_payload(
            ctx,
            "28_SKELETON_QUALIFIED",
            "RealSaS.QualifiedSkeletonIR.v1",
        )
    )
    policy = mesh_policy_from_dict(
        stage_output_payload(
            ctx,
            "18_CANONICAL_MESH_ADDRESSING_BUILD",
            "RealSaS.MeshQualificationPolicyIR.v1",
        )
    )
    camera_set = qualified_camera_set_from_dict(
        stage_output_payload(
            ctx,
            "05_CAMERA_CONTRACT_SOLVED",
            "RealSaS.QualifiedCameraSetIR.v1",
        )
    )
    cameras = tuple(sorted(camera_set.cameras, key=lambda x: int(x.view_index)))
    source_report = json.loads(
        Path("canonical/KNIGHT_MOTION_SOURCE_ACTION_DIAGNOSTIC_20260927.json").read_text()
    )

    c41_report = json.loads((args.c41_dir / "REPORT.json").read_text())
    if c41_report.get("run_id") != args.run_id:
        raise RuntimeError("C50_C41_RUN_ID_DRIFT")
    with np.load(
        args.c41_dir / "OPTIMIZED_ACTUAL_MOTION_WEIGHTS.npz",
        allow_pickle=False,
    ) as z:
        base_weights = np.asarray(z["base_weights"], np.float64)
        optimized_weights = np.asarray(z["optimized_weights"], np.float64)
        vertex_ids = tuple(map(str, z["vertex_ids"].tolist()))
        joint_ids = tuple(map(str, z["joint_ids"].tolist()))

    if set(joint_ids) != {j.canonical_joint_id for j in skeleton.joints}:
        raise RuntimeError("C50_C41_JOINT_AXIS_DRIFT")
    rest = np.asarray([v.P for v in candidate.vertices], dtype=np.float64)
    faces = _faces(candidate, vertex_ids)

    arms = {
        "BASE_PROJECTED_SKIN": base_weights,
        "C41_OPTIMIZED_SKIN": optimized_weights,
    }
    report = {
        "schema": "RealSaS.TreeConsistentRetargetCeiling.v1",
        "status": "MEASURED__NO_MODEL_FIT__NO_PROMOTION_CLAIM",
        "run_id": args.run_id,
        "c41_status": c41_report.get("status"),
        "clips": {},
        "training_used": False,
        "teacher_weights_used": False,
        "product_authority_minted": False,
    }

    for clip in CLIPS:
        payload = json.loads(
            (
                rr
                / "inputs"
                / "motion"
                / "quaternius_knight_v1"
                / f"{clip}.motion.json"
            ).read_text()
        )

        current_mapping, current_details, _srows, _trows, _tpar = _mapping(
            payload, skeleton, source_report
        )
        current_tracks, _ = _tracks_for_clip(
            payload, skeleton, cameras, source_report
        )

        constrained_mapping, constrained_details, srows, _trows2, tpar, total_cost = (
            _tree_consistent_mapping(payload, skeleton, source_report)
        )
        constrained_tracks = _tracks_with_mapping(
            payload,
            skeleton,
            cameras,
            constrained_mapping,
            srows,
            tpar,
        )

        current_violations = _hierarchy_violations(
            payload, skeleton, source_report, current_mapping
        )
        constrained_violations = _hierarchy_violations(
            payload, skeleton, source_report, constrained_mapping
        )
        if constrained_violations:
            raise RuntimeError("C50_CONSTRAINED_MAPPING_HAS_HIERARCHY_VIOLATIONS")

        cmats, crows = _motion_matrices(
            payload=payload,
            tracks=current_tracks,
            skeleton=skeleton,
            cameras=cameras,
            joint_ids=joint_ids,
            clip_id=clip,
        )
        tmats, trows = _motion_matrices(
            payload=payload,
            tracks=constrained_tracks,
            skeleton=skeleton,
            cameras=cameras,
            joint_ids=joint_ids,
            clip_id=clip,
        )

        skin_rows = {}
        for skin_name, W in arms.items():
            current = _hard_actual_motion(
                rest=rest,
                faces=faces,
                weights=W,
                matrices=cmats,
                frame_rows=crows,
                policy=policy,
            )
            constrained = _hard_actual_motion(
                rest=rest,
                faces=faces,
                weights=W,
                matrices=tmats,
                frame_rows=trows,
                policy=policy,
            )
            skin_rows[skin_name] = {
                "current_mapping": _summarize(current),
                "tree_consistent_mapping": _summarize(constrained),
                "delta_tree_minus_current": {
                    "failed_frame_count": int(
                        constrained["failed_frame_count"] - current["failed_frame_count"]
                    ),
                    "unique_bad_face_count": int(
                        constrained["unique_bad_face_count"] - current["unique_bad_face_count"]
                    ),
                    "maximum_condition": float(
                        constrained["maximum_condition"] - current["maximum_condition"]
                    ),
                    "maximum_edge_ratio": float(
                        constrained["maximum_edge_ratio"] - current["maximum_edge_ratio"]
                    ),
                },
            }

        report["clips"][clip] = {
            "current_mapping_hierarchy_violation_count": int(len(current_violations)),
            "current_mapping_hierarchy_violations": current_violations,
            "tree_consistent_mapping_hierarchy_violation_count": 0,
            "tree_consistent_total_unary_plus_tree_cost": float(total_cost),
            "current_mapping": current_mapping,
            "tree_consistent_mapping": constrained_mapping,
            "tree_consistent_details": constrained_details,
            "skin_arms": skin_rows,
        }

    current_total_bad = sum(
        report["clips"][clip]["skin_arms"]["C41_OPTIMIZED_SKIN"]["current_mapping"][
            "unique_bad_face_count"
        ]
        for clip in CLIPS
    )
    tree_total_bad = sum(
        report["clips"][clip]["skin_arms"]["C41_OPTIMIZED_SKIN"][
            "tree_consistent_mapping"
        ]["unique_bad_face_count"]
        for clip in CLIPS
    )
    current_failed = sum(
        report["clips"][clip]["skin_arms"]["C41_OPTIMIZED_SKIN"]["current_mapping"][
            "failed_frame_count"
        ]
        for clip in CLIPS
    )
    tree_failed = sum(
        report["clips"][clip]["skin_arms"]["C41_OPTIMIZED_SKIN"][
            "tree_consistent_mapping"
        ]["failed_frame_count"]
        for clip in CLIPS
    )

    materially_better = (
        tree_total_bad < current_total_bad
        or tree_failed < current_failed
    )
    all_pass = all(
        report["clips"][clip]["skin_arms"]["C41_OPTIMIZED_SKIN"][
            "tree_consistent_mapping"
        ]["passed"]
        for clip in CLIPS
    )
    report["decision"] = (
        "TREE_CONSISTENT_RETARGET_CLOSES_ACTUAL_MOTION__REPAIR_RETARGET_FIRST"
        if all_pass
        else (
            "TREE_CONSISTENT_RETARGET_MATERIAL_CONTRIBUTOR__RESIDUAL_REMAINS"
            if materially_better
            else "TREE_CONSISTENT_RETARGET_NOT_PRIMARY_OWNER__OPEN_C5_1"
        )
    )
    report["optimized_skin_aggregate"] = {
        "current_mapping_unique_bad_face_sum_across_clips": int(current_total_bad),
        "tree_consistent_unique_bad_face_sum_across_clips": int(tree_total_bad),
        "current_mapping_failed_frame_sum_across_clips": int(current_failed),
        "tree_consistent_failed_frame_sum_across_clips": int(tree_failed),
    }
    report["claim_boundary"] = [
        "This court changes only the motion retarget mapping contract.",
        "Carrier, ATLAS rig, and each evaluated skin arm are frozen.",
        "No source joint names are hard-coded in the constrained solver.",
        "A better retarget arm does not by itself prove ATLAS is optimal.",
        "Residual after the best retarget arm proceeds to subject-agnostic C5.1/C5.2 rig-basis search.",
    ]

    write(args.out_dir / "REPORT.json", report)
    print("C50_TREE_RETARGET_RESULT=" + json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--authority-root", type=Path, required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--c41-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    try:
        main(args)
    except Exception as exc:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        write(
            args.out_dir / "ERROR.json",
            {"type": type(exc).__name__, "message": str(exc)},
        )
        raise
