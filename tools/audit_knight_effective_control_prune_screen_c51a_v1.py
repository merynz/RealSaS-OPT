from __future__ import annotations

"""C5.1a effective-control prune screen.

Research-only, subject-agnostic screen.

For every non-root control in the current rig:
- remove its independent motion DOF by replacing its track with identity;
- move its skin-weight mass exactly to its parent control;
- keep carrier, remaining rig geometry, tree-consistent retarget, and motion fixed;
- replay the exact 51-frame idle/run/slash hard triangle court.

This is only a screening operator. A control is not product-pruned here.
Promising candidates advance to C5.1b, where skin is reoptimized and full
editability/proof gates are rerun.
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from compiler.realsas_compiler_core.artifact_codec_v2 import (
    canonical_mesh_candidate_from_dict,
    mesh_policy_from_dict,
    qualified_camera_set_from_dict,
    qualified_skeleton_from_dict,
)
from compiler.realsas_compiler_services.orchestrator.adapters.adapter_io import (
    stage_output_payload,
)
from tools.audit_knight_canonical_caa_mechanics_geometry_lock_v1 import (
    CLIPS,
)
from tools.audit_knight_mira_actual_motion_weight_oracle_c41_v1 import (
    _faces,
    _hard_actual_motion,
)
from tools.audit_knight_tree_consistent_retarget_c50_v1 import (
    _motion_matrices,
    _tracks_with_mapping,
    _tree_consistent_mapping,
)
from tools.demo.render_knight_motion_preview_v1 import _ctx
from tools.inference.refined_surface_rig_skin_v1 import write


def _identity_track(track):
    keys = tuple(
        replace(
            key,
            local_rotation_quat_xyzw=(0.0, 0.0, 0.0, 1.0),
            local_translation_xyz=(0.0, 0.0, 0.0),
            local_scale_xyz=(1.0, 1.0, 1.0),
        )
        for key in track.keyframes
    )
    return replace(
        track,
        keyframes=keys,
        metadata={
            **dict(track.metadata or {}),
            "research_only_effective_control_pruned": True,
        },
    )


def _summarize(actual):
    return {
        k: v for k, v in actual.items() if k not in {"union_bad", "per_frame"}
    }


def _aggregate(clip_rows):
    return {
        "failed_frame_sum": int(
            sum(row["failed_frame_count"] for row in clip_rows.values())
        ),
        "unique_bad_face_sum": int(
            sum(row["unique_bad_face_count"] for row in clip_rows.values())
        ),
        "maximum_condition": float(
            max(row["maximum_condition"] for row in clip_rows.values())
        ),
        "maximum_edge_ratio": float(
            max(row["maximum_edge_ratio"] for row in clip_rows.values())
        ),
        "minimum_area_ratio": float(
            min(row["minimum_area_ratio"] for row in clip_rows.values())
        ),
        "all_clips_passed": bool(all(row["passed"] for row in clip_rows.values())),
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
    cameras = tuple(
        sorted(
            qualified_camera_set_from_dict(
                stage_output_payload(
                    ctx,
                    "05_CAMERA_CONTRACT_SOLVED",
                    "RealSaS.QualifiedCameraSetIR.v1",
                )
            ).cameras,
            key=lambda x: int(x.view_index),
        )
    )
    source_report = json.loads(
        Path(
            "canonical/KNIGHT_MOTION_SOURCE_ACTION_DIAGNOSTIC_20260927.json"
        ).read_text()
    )

    c41_report = json.loads((args.c41_dir / "REPORT.json").read_text())
    if c41_report.get("run_id") != args.run_id:
        raise RuntimeError("C51A_C41_RUN_ID_DRIFT")
    with np.load(
        args.c41_dir / "OPTIMIZED_ACTUAL_MOTION_WEIGHTS.npz",
        allow_pickle=False,
    ) as z:
        W = np.asarray(z["optimized_weights"], np.float64)
        vertex_ids = tuple(map(str, z["vertex_ids"].tolist()))
        joint_ids = tuple(map(str, z["joint_ids"].tolist()))

    if set(joint_ids) != {j.canonical_joint_id for j in skeleton.joints}:
        raise RuntimeError("C51A_JOINT_AXIS_DRIFT")

    ji = {jid: i for i, jid in enumerate(joint_ids)}
    parent = {
        str(j.canonical_joint_id):
            (None if j.parent_canonical_id is None else str(j.parent_canonical_id))
        for j in skeleton.joints
    }
    if skeleton.root_id not in ji:
        raise RuntimeError("C51A_ROOT_AXIS_DRIFT")

    rest = np.asarray([v.P for v in candidate.vertices], np.float64)
    faces = _faces(candidate, vertex_ids)

    clip_payload = {}
    clip_tracks = {}
    clip_rows = {}
    mapping_receipts = {}
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
        mapping, details, srows, _trows, tpar, total_cost = (
            _tree_consistent_mapping(payload, skeleton, source_report)
        )
        tracks = _tracks_with_mapping(
            payload, skeleton, cameras, mapping, srows, tpar
        )
        mats, rows = _motion_matrices(
            payload=payload,
            tracks=tracks,
            skeleton=skeleton,
            cameras=cameras,
            joint_ids=joint_ids,
            clip_id=clip,
        )
        clip_payload[clip] = payload
        clip_tracks[clip] = tracks
        clip_rows[clip] = rows
        mapping_receipts[clip] = {
            "mapping": mapping,
            "total_cost": float(total_cost),
            "detail_count": int(len(details)),
        }

    baseline_by_clip = {}
    for clip in CLIPS:
        mats, _ = _motion_matrices(
            payload=clip_payload[clip],
            tracks=clip_tracks[clip],
            skeleton=skeleton,
            cameras=cameras,
            joint_ids=joint_ids,
            clip_id=clip,
        )
        baseline_by_clip[clip] = _summarize(
            _hard_actual_motion(
                rest=rest,
                faces=faces,
                weights=W,
                matrices=mats,
                frame_rows=clip_rows[clip],
                policy=policy,
            )
        )
    baseline = _aggregate(baseline_by_clip)

    results = []
    for jid in sorted(joint_ids):
        p = parent.get(jid)
        if p is None:
            continue
        if p not in ji:
            raise RuntimeError("C51A_PARENT_AXIS_DRIFT")

        Wc = W.copy()
        child_col = ji[jid]
        parent_col = ji[p]
        Wc[:, parent_col] += Wc[:, child_col]
        Wc[:, child_col] = 0.0
        if not np.allclose(Wc.sum(axis=1), 1.0, atol=1e-10, rtol=0.0):
            raise RuntimeError("C51A_WEIGHT_MASS_DRIFT")

        by_clip = {}
        for clip in CLIPS:
            tracks = dict(clip_tracks[clip])
            tracks[jid] = _identity_track(tracks[jid])
            mats, rows = _motion_matrices(
                payload=clip_payload[clip],
                tracks=tracks,
                skeleton=skeleton,
                cameras=cameras,
                joint_ids=joint_ids,
                clip_id=clip,
            )
            by_clip[clip] = _summarize(
                _hard_actual_motion(
                    rest=rest,
                    faces=faces,
                    weights=Wc,
                    matrices=mats,
                    frame_rows=rows,
                    policy=policy,
                )
            )

        agg = _aggregate(by_clip)
        delta = {
            "failed_frame_sum": int(
                agg["failed_frame_sum"] - baseline["failed_frame_sum"]
            ),
            "unique_bad_face_sum": int(
                agg["unique_bad_face_sum"] - baseline["unique_bad_face_sum"]
            ),
            "maximum_condition": float(
                agg["maximum_condition"] - baseline["maximum_condition"]
            ),
            "maximum_edge_ratio": float(
                agg["maximum_edge_ratio"] - baseline["maximum_edge_ratio"]
            ),
            "minimum_area_ratio": float(
                agg["minimum_area_ratio"] - baseline["minimum_area_ratio"]
            ),
        }
        nonregressive_screen = bool(
            delta["failed_frame_sum"] <= 0
            and delta["unique_bad_face_sum"] <= 0
            and delta["maximum_condition"] <= 1e-6
            and delta["maximum_edge_ratio"] <= 1e-6
            and delta["minimum_area_ratio"] >= -1e-6
        )
        results.append(
            {
                "control_id": jid,
                "parent_control_id": p,
                "effective_control_count_after_prune": int(len(joint_ids) - 1),
                "metrics_by_clip": by_clip,
                "aggregate": agg,
                "delta_vs_baseline": delta,
                "nonregressive_screen": nonregressive_screen,
            }
        )

    results.sort(
        key=lambda row: (
            row["delta_vs_baseline"]["failed_frame_sum"],
            row["delta_vs_baseline"]["unique_bad_face_sum"],
            row["delta_vs_baseline"]["maximum_condition"],
            row["delta_vs_baseline"]["maximum_edge_ratio"],
            row["control_id"],
        )
    )
    strong = [row for row in results if row["nonregressive_screen"]]

    report = {
        "schema": "RealSaS.EffectiveControlPruneScreen.v1",
        "status": "MEASURED__SCREEN_ONLY__NO_PRODUCT_PRUNE",
        "run_id": args.run_id,
        "current_effective_control_count": int(len(joint_ids)),
        "baseline": baseline,
        "baseline_by_clip": baseline_by_clip,
        "tree_consistent_mapping_receipts": mapping_receipts,
        "candidate_count": int(len(results)),
        "nonregressive_screen_count": int(len(strong)),
        "nonregressive_control_ids": [
            row["control_id"] for row in strong
        ],
        "ranked_candidates": results,
        "training_used": False,
        "teacher_weights_used": False,
        "teacher_joint_count_used": False,
        "product_authority_minted": False,
        "next_gate": (
            "C5_1B_FULL_PRUNE_TO_PROOF_ON_SCREENED_CANDIDATES"
            if strong
            else "C5_2_GROW_TO_NEED_OR_LOCUS_OWNER"
        ),
        "claim_boundary": [
            "This is a cheap effective-DOF prune screen, not structural skeleton deletion.",
            "A candidate passes only if collapsing its motion DOF and moving all skin mass to its parent is non-regressive without skin reoptimization.",
            "Failing this screen does not prove a control is necessary; C5.1b may still test selected candidates with skin reoptimization.",
            "Passing this screen does not authorize product pruning; editability and hard proof remain required.",
            "No requested or teacher joint count is used.",
        ],
    }
    write(args.out_dir / "REPORT.json", report)
    print("C51A_CONTROL_PRUNE_SCREEN_RESULT=" + json.dumps(report, sort_keys=True), flush=True)


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
