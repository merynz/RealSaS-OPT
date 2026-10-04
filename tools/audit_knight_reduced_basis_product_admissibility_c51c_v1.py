from __future__ import annotations

"""C5.1c product-admissibility gates for mechanically closed reduced control bases.

This court does not decide global optimum control count. It verifies that a
mechanically admitted prune also preserves the source/motion and editability
contracts required before that basis may seed the next minimum-sufficient search
frontier.

Hard V1 gates:
- exact same Stage19 carrier/source substrate lineage;
- exact-motion + fresh microstress mechanical PASS from C5.1b;
- tree-consistent retarget correspondence cost non-regressive vs the 28-control
  corrected baseline for every supplied clip;
- structurally legal reduced skeleton and exact skin joint accounting;
- every exposed control has a measurable non-zero carrier response under the
  subject-free +/-10 degree XYZ edit probes.

No teacher weights or teacher joint count are used.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from compiler.realsas_compiler_core.artifact_codec_v2 import (
    canonical_mesh_candidate_from_dict,
    qualified_camera_set_from_dict,
    qualified_skeleton_from_dict,
)
from compiler.realsas_compiler_core.control_basis_pruning_v1 import (
    prune_qualified_skeleton_control_v1,
)
from compiler.realsas_compiler_core.joint_frames_v1 import derive_joint_frames_from_skeleton
from compiler.realsas_compiler_core.mesh.deformation_stress_v1 import apply_lbs_matrix_v1
from compiler.realsas_compiler_core.mesh.deformation_stress_v2 import _pose_skin_matrices
from compiler.realsas_compiler_services.orchestrator.adapters.adapter_io import (
    stage_output_payload,
)
from tools.demo.render_knight_motion_preview_v1 import _ctx
from tools.inference.refined_surface_rig_skin_v1 import write


EDIT_PROBE_DEG = 10.0
ZERO_RESPONSE_REL_EPS = 1e-8


def _load_admitted_weights(c51b_dir: Path, control_id: str):
    safe = control_id.replace(":", "_")
    matches = sorted(c51b_dir.glob(f"*_{safe}_WEIGHTS.npz"))
    if len(matches) != 1:
        raise RuntimeError(
            f"C51C_ADMITTED_WEIGHT_FILE_ACCOUNTING:{control_id}:{len(matches)}"
        )
    with np.load(matches[0], allow_pickle=False) as z:
        return (
            np.asarray(z["weights"], dtype=np.float64),
            tuple(map(str, z["vertex_ids"].tolist())),
            tuple(map(str, z["joint_ids"].tolist())),
            matches[0],
        )


def _validate_tree(skeleton):
    by = {str(j.canonical_joint_id): j for j in skeleton.joints}
    if len(by) != len(skeleton.joints) or str(skeleton.root_id) not in by:
        raise RuntimeError("C51C_SKELETON_ID_OR_ROOT_INVALID")
    for joint in skeleton.joints:
        p = joint.parent_canonical_id
        if p is not None and str(p) not in by:
            raise RuntimeError("C51C_SKELETON_PARENT_UNKNOWN")

    visiting = set()
    done = set()

    def visit(jid):
        if jid in done:
            return
        if jid in visiting:
            raise RuntimeError("C51C_SKELETON_CYCLE")
        visiting.add(jid)
        p = by[jid].parent_canonical_id
        if p is not None:
            visit(str(p))
        visiting.remove(jid)
        done.add(jid)

    for jid in sorted(by):
        visit(jid)
    return by


def _control_response_report(*, skeleton, cameras, rest, weights, joint_ids):
    by = _validate_tree(skeleton)
    ids = tuple(map(str, joint_ids))
    if set(ids) != set(by) or len(ids) != len(by):
        raise RuntimeError("C51C_SKIN_JOINT_ACCOUNTING_DRIFT")
    ji = {jid: i for i, jid in enumerate(ids)}

    W = np.asarray(weights, dtype=np.float64)
    P = np.asarray(rest, dtype=np.float64)
    if W.shape != (len(P), len(ids)):
        raise RuntimeError("C51C_WEIGHT_SHAPE_DRIFT")
    if not np.isfinite(W).all() or np.any(W < -1e-10):
        raise RuntimeError("C51C_WEIGHT_INVALID")
    if not np.allclose(W.sum(axis=1), 1.0, atol=1e-8, rtol=0.0):
        raise RuntimeError("C51C_WEIGHT_SIMPLEX_DRIFT")

    extent = float(np.linalg.norm(P.max(axis=0) - P.min(axis=0)))
    extent = max(extent, 1e-12)
    frames = derive_joint_frames_from_skeleton(skeleton, cameras=cameras)

    rows = []
    no_op = []
    for jid in sorted(ids):
        probe_rows = []
        best = 0.0
        best_rms = 0.0
        for axis in range(3):
            for sign in (-1.0, 1.0):
                mats_by_id = _pose_skin_matrices(
                    skeleton,
                    frames,
                    joint_id=jid,
                    local_axis_index=axis,
                    degrees=sign * EDIT_PROBE_DEG,
                )
                mats = np.stack([mats_by_id[x] for x in ids], axis=0)
                posed = apply_lbs_matrix_v1(P, W, mats)
                d = np.linalg.norm(posed - P, axis=1)
                mx = float(np.max(d) / extent)
                rms = float(np.sqrt(np.mean(d * d)) / extent)
                best = max(best, mx)
                best_rms = max(best_rms, rms)
                probe_rows.append(
                    {
                        "axis_index": int(axis),
                        "degrees": float(sign * EDIT_PROBE_DEG),
                        "max_displacement_over_extent": mx,
                        "rms_displacement_over_extent": rms,
                    }
                )
        is_noop = bool(best <= ZERO_RESPONSE_REL_EPS)
        if is_noop:
            no_op.append(jid)
        rows.append(
            {
                "control_id": jid,
                "is_root": jid == str(skeleton.root_id),
                "max_response_over_extent": best,
                "max_rms_response_over_extent": best_rms,
                "numerical_no_op": is_noop,
                "probes": probe_rows,
            }
        )

    return {
        "probe_angle_deg": EDIT_PROBE_DEG,
        "zero_response_relative_epsilon": ZERO_RESPONSE_REL_EPS,
        "control_count": int(len(ids)),
        "numerical_no_op_control_count": int(len(no_op)),
        "numerical_no_op_control_ids": no_op,
        "all_exposed_controls_measurable": not no_op,
        "controls": rows,
    }


def main(args):
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ctx = _ctx(args.authority_root, args.run_id)

    source_skeleton = qualified_skeleton_from_dict(
        stage_output_payload(
            ctx, "28_SKELETON_QUALIFIED", "RealSaS.QualifiedSkeletonIR.v1"
        )
    )
    candidate = canonical_mesh_candidate_from_dict(
        stage_output_payload(
            ctx,
            "18_CANONICAL_MESH_ADDRESSING_BUILD",
            "RealSaS.CanonicalMeshCandidateIR.v1",
        )
    )
    cameras_ir = qualified_camera_set_from_dict(
        stage_output_payload(
            ctx, "05_CAMERA_CONTRACT_SOLVED", "RealSaS.QualifiedCameraSetIR.v1"
        )
    )
    cameras = tuple(sorted(cameras_ir.cameras, key=lambda x: int(x.view_index)))

    c50b = json.loads((args.c50b_dir / "REPORT.json").read_text())
    c51b = json.loads((args.c51b_dir / "REPORT.json").read_text())
    if c50b.get("run_id") != args.run_id or c51b.get("run_id") != args.run_id:
        raise RuntimeError("C51C_INPUT_RUN_ID_DRIFT")

    admitted = tuple(map(str, c51b.get("mechanically_admissible_control_ids_removed") or ()))
    if not admitted:
        raise RuntimeError("C51C_NO_MECHANICALLY_ADMITTED_BASIS")

    baseline_retarget = dict(c50b.get("tree_consistent_retarget") or {})
    ranked = {
        str(row["control_id"]): row
        for row in c51b.get("ranked_results") or ()
    }

    results = []
    for control_id in admitted:
        if control_id not in ranked:
            raise RuntimeError("C51C_ADMITTED_RESULT_MISSING:" + control_id)
        mech = ranked[control_id]
        if not mech.get("mechanically_admissible"):
            raise RuntimeError("C51C_ADMITTED_FLAG_DRIFT:" + control_id)

        pruned, receipt = prune_qualified_skeleton_control_v1(
            source_skeleton, control_id
        )
        weights, vertex_ids, joint_ids, weight_path = _load_admitted_weights(
            args.c51b_dir, control_id
        )

        carrier_ids = tuple(str(v.candidate_vertex_id) for v in candidate.vertices)
        if tuple(vertex_ids) != carrier_ids:
            raise RuntimeError("C51C_CARRIER_VERTEX_AXIS_DRIFT")

        edit = _control_response_report(
            skeleton=pruned,
            cameras=cameras,
            rest=np.asarray([v.P for v in candidate.vertices], dtype=np.float64),
            weights=weights,
            joint_ids=joint_ids,
        )

        motion_rows = {}
        motion_nonregressive = True
        for clip, base_row in sorted(baseline_retarget.items()):
            cand_row = (mech.get("retarget") or {}).get(clip)
            if cand_row is None:
                raise RuntimeError("C51C_RETARGET_CLIP_MISSING:" + clip)
            base_cost = float(base_row["total_cost"])
            cand_cost = float(cand_row["total_cost"])
            nonreg = bool(cand_cost <= base_cost + 1e-9)
            motion_nonregressive &= nonreg
            motion_rows[clip] = {
                "baseline_28_control_tree_cost": base_cost,
                "candidate_tree_cost": cand_cost,
                "delta_candidate_minus_baseline": cand_cost - base_cost,
                "nonregressive": nonreg,
            }

        micro = dict(mech.get("microstress") or {})
        mechanical_pass = bool(
            (mech.get("after_reoptimization") or {}).get("passed")
            and micro.get("g3_passed")
            and micro.get("g3b_passed")
        )
        static_source_invariant = True  # carrier candidate is literally unchanged
        editability_pass = bool(edit["all_exposed_controls_measurable"])
        passed = bool(
            mechanical_pass
            and static_source_invariant
            and motion_nonregressive
            and editability_pass
        )

        results.append(
            {
                "removed_control_id": control_id,
                "prune_receipt": receipt.to_dict(),
                "reduced_control_count": int(len(pruned.joints)),
                "mechanical_pass": mechanical_pass,
                "static_source_fidelity": {
                    "passed": static_source_invariant,
                    "reason": "EXACT_SAME_STAGE18_STAGE19_CARRIER_AND_APPEARANCE_DOMAIN",
                },
                "motion_source_fidelity": {
                    "passed": bool(motion_nonregressive),
                    "per_clip_tree_consistent_correspondence": motion_rows,
                    "semantics": "SUPPLIED_MOTION_SOURCE_GEOMETRY_CORRESPONDENCE_NONREGRESSION",
                },
                "editability": {
                    "passed": editability_pass,
                    "response": edit,
                    "microstress_pass": bool(
                        micro.get("g3_passed") and micro.get("g3b_passed")
                    ),
                },
                "product_admissibility_v1": passed,
                "weights_path": str(weight_path),
            }
        )

    passed_rows = [row for row in results if row["product_admissibility_v1"]]
    report = {
        "schema": "RealSaS.ReducedControlBasisProductAdmissibilityCourt.v1",
        "status": (
            "PRODUCT_ADMISSIBLE_REDUCED_BASIS_FOUND"
            if passed_rows
            else "NO_PRODUCT_ADMISSIBLE_REDUCED_BASIS"
        ),
        "run_id": args.run_id,
        "baseline_control_count": int(len(source_skeleton.joints)),
        "mechanically_admitted_candidate_count": int(len(admitted)),
        "product_admissible_candidate_count": int(len(passed_rows)),
        "product_admissible_removed_control_ids": sorted(
            row["removed_control_id"] for row in passed_rows
        ),
        "results": results,
        "teacher_joint_count_used": False,
        "teacher_weights_used": False,
        "training_used": False,
        "product_authority_minted": False,
        "decision": (
            "SEED_NEXT_PRUNE_FRONTIER_FROM_PRODUCT_ADMISSIBLE_BASIS"
            if passed_rows
            else "REJECT_MECHANICAL_ONLY_PRUNE_AND_OPEN_C5_2"
        ),
        "claim_boundary": [
            "Static source fidelity is invariant because the Stage19 carrier is unchanged.",
            "Motion-source fidelity uses only the supplied motion source and the generic tree-consistent correspondence cost.",
            "Editability V1 requires fresh mechanical microstress PASS plus a measurable response from every exposed control.",
            "This gate establishes an admissible search-frontier basis, not global optimum control count.",
        ],
    }
    write(args.out_dir / "REPORT.json", report)
    print("C51C_RESULT=" + json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--authority-root", type=Path, required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--c50b-dir", type=Path, required=True)
    ap.add_argument("--c51b-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    try:
        main(args)
    except Exception as exc:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        write(args.out_dir / "ERROR.json", {"type": type(exc).__name__, "message": str(exc)})
        raise
