from __future__ import annotations

"""C5.1f recursive Pareto prune frontier.

Consumes the sealed C5.1e candidate family and recursively expands every
non-dominated admitted parent under the same single-control structural-prune
neighborhood. This is a research/compiler court; it does not mint product
authority and never turns the Knight count into a generic target.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from compiler.realsas_compiler_core.artifact_codec_v2 import (
    canonical_mesh_candidate_from_dict,
    mesh_policy_from_dict,
    qualified_camera_set_from_dict,
    qualified_skeleton_from_dict,
    rigging_surface_from_dict,
)
from compiler.realsas_compiler_core.control_basis_frontier_v1 import (
    build_control_basis_pareto_frontier_v1,
)
from compiler.realsas_compiler_core.control_basis_pruning_v1 import (
    collapse_weight_axis_to_parent_v1,
    prune_qualified_skeleton_control_v1,
)
from compiler.realsas_compiler_core.deformation_envelope_derivation_v1 import (
    derive_deformation_envelope_v1,
)
from compiler.realsas_compiler_core.mechanical_carrier_evidence_v1 import (
    build_mechanical_carrier_evidence_v1,
)
from compiler.realsas_compiler_core.mechanical_carrier_skin_v1 import (
    qualify_mechanical_carrier_skin_v1,
)
from compiler.realsas_compiler_core.mesh.deformation_stress_v2 import (
    run_g3_local_frame_micro_stress_v2,
)
from compiler.realsas_compiler_core.mesh.skin_topology_compatibility_v1 import (
    run_skin_topology_compatibility_v1,
)
from compiler.realsas_compiler_core.surface_addressing_v1 import (
    static_mesh_qualification_from_dict,
    surface_addressing_from_dict,
)
from compiler.realsas_compiler_services.orchestrator.adapters.adapter_io import (
    stage_output_payload,
)
from tools.audit_knight_carrier_first_c3_v1 import _summarize_g3
from tools.audit_knight_control_basis_local_irreducibility_c51e_v1 import (
    _frontier_row,
    _summary,
)
from tools.audit_knight_mira_actual_motion_weight_oracle_c41_v1 import _faces
from tools.audit_knight_reduced_basis_product_admissibility_c51c_v1 import (
    _control_response_report,
)
from tools.audit_knight_structural_prune_to_proof_c51b_v1 import (
    _optimize_skin,
    _tree_motion_frames,
)
from tools.audit_knight_targeted_noop_prune_frontier_c51d_v1 import (
    _motion_nonregression,
)
from tools.demo.render_knight_motion_preview_v1 import _ctx
from tools.inference.refined_surface_rig_skin_v1 import write


@dataclass
class ParentState:
    skeleton: object
    weights: np.ndarray
    vertex_ids: tuple[str, ...]
    joint_ids: tuple[str, ...]
    removed_path: tuple[str, ...]
    source_basis_id: str


def _safe(value: str) -> str:
    return str(value).replace(":", "_")


def _load_c51e_frontier(c51e_dir: Path, report: dict) -> list[ParentState]:
    results = {
        str(row["skeleton_lineage_hash"]): row
        for row in report.get("results") or ()
    }
    candidate_ids = tuple(map(str, report.get("candidate_control_ids") or ()))
    base_path = tuple(map(str, report.get("base_removed_path") or ()))
    frontier = dict(report.get("pareto_frontier") or {})
    parent_rows = [
        dict(row)
        for row in frontier.get("candidates") or ()
        if bool(row.get("hard_admissible"))
    ]
    if not parent_rows:
        raise RuntimeError("C51F_C51E_ADMITTED_PARETO_PARENT_MISSING")

    out = []
    for parent in parent_rows:
        basis_id = str(parent["basis_id"])
        row = results.get(basis_id)
        if row is None:
            raise RuntimeError("C51F_C51E_RESULT_FOR_FRONTIER_BASIS_MISSING:" + basis_id)
        removed = str(row["removed_control_id"])
        try:
            rank = candidate_ids.index(removed)
        except ValueError as exc:
            raise RuntimeError("C51F_C51E_REMOVED_CONTROL_AXIS_MISSING:" + removed) from exc
        skeleton_path = c51e_dir / f"{rank:02d}_{_safe(removed)}_SKELETON.json"
        weights_path = c51e_dir / f"{rank:02d}_{_safe(removed)}_WEIGHTS.npz"
        if not skeleton_path.is_file() or not weights_path.is_file():
            raise RuntimeError(
                "C51F_C51E_FRONTIER_ARTIFACT_MISSING:"
                + str(skeleton_path)
                + ":"
                + str(weights_path)
            )
        skeleton = qualified_skeleton_from_dict(json.loads(skeleton_path.read_text()))
        with np.load(weights_path, allow_pickle=False) as z:
            weights = np.asarray(z["weights"], np.float64)
            vertex_ids = tuple(map(str, z["vertex_ids"].tolist()))
            joint_ids = tuple(map(str, z["joint_ids"].tolist()))
        if str(skeleton.skeleton_lineage_hash) != basis_id:
            raise RuntimeError("C51F_C51E_FRONTIER_SKELETON_HASH_DRIFT")
        if set(joint_ids) != {j.canonical_joint_id for j in skeleton.joints}:
            raise RuntimeError("C51F_C51E_FRONTIER_JOINT_AXIS_DRIFT")
        out.append(
            ParentState(
                skeleton=skeleton,
                weights=weights,
                vertex_ids=vertex_ids,
                joint_ids=joint_ids,
                removed_path=base_path + (removed,),
                source_basis_id=basis_id,
            )
        )
    return out


def _evaluate_child(
    *,
    parent: ParentState,
    control_id: str,
    rank_seed: int,
    candidate,
    carrier,
    surface,
    cameras,
    camera_set,
    policy,
    source_report,
    run_root,
    rest,
    faces,
    baseline_retarget,
    args,
):
    pruned, receipt = prune_qualified_skeleton_control_v1(
        parent.skeleton,
        control_id,
    )
    W0, child_joint_ids = collapse_weight_axis_to_parent_v1(
        parent.weights,
        parent.joint_ids,
        removed_control_id=receipt.removed_control_id,
        replacement_parent_id=receipt.replacement_parent_id,
    )
    matrices, frame_rows, retarget = _tree_motion_frames(
        run_root=run_root,
        skeleton=pruned,
        cameras=cameras,
        joint_ids=child_joint_ids,
        source_report=source_report,
    )
    Wopt, before, after, opt_report = _optimize_skin(
        rest=rest,
        faces=faces,
        base_weights=W0,
        matrices=matrices,
        frame_rows=frame_rows,
        policy=policy,
        device=args.device,
        seed=args.seed + rank_seed,
        steps=args.steps,
        lr=args.lr,
        frame_batch=args.frame_batch,
        trust_weight=args.trust_weight,
        log_every=args.log_every,
        label=control_id,
    )
    motion_ok, motion_rows = _motion_nonregression(
        retarget=retarget,
        baseline_retarget=baseline_retarget,
    )
    edit = _control_response_report(
        skeleton=pruned,
        cameras=cameras,
        rest=rest,
        weights=Wopt,
        joint_ids=child_joint_ids,
    )

    micro = None
    micro_ok = False
    skin_hash = None
    if after["passed"]:
        skin = qualify_mechanical_carrier_skin_v1(
            candidate=candidate,
            carrier_evidence=carrier,
            skeleton=pruned,
            vertex_ids=parent.vertex_ids,
            joint_ids=child_joint_ids,
            weights=Wopt,
            max_simplex_repair_l1=1e-5,
            max_total_correction_l1=0.1,
        )
        _axis, envelope = derive_deformation_envelope_v1(
            skeleton=pruned,
            camera_set=camera_set,
        )
        g3 = run_g3_local_frame_micro_stress_v2(
            candidate,
            surface=surface,
            skeleton=pruned,
            skin=skin,
            envelope=envelope,
            cameras=cameras,
            policy=policy,
        )
        g3b = run_skin_topology_compatibility_v1(
            candidate,
            surface=surface,
            skeleton=pruned,
            skin=skin,
            envelope=envelope,
            cameras=cameras,
            policy=policy,
            stress_all_faces=True,
        )
        micro = _summarize_g3(g3, g3b)
        micro_ok = bool(micro["g3_passed"] and micro["g3b_passed"])
        skin_hash = str(skin.skin_lineage_hash)

    product_pass = bool(
        after["passed"]
        and micro_ok
        and motion_ok
        and edit["all_exposed_controls_measurable"]
    )
    removed_path = parent.removed_path + (control_id,)
    row = {
        "parent_basis_id": parent.source_basis_id,
        "removed_control_id": str(control_id),
        "removed_path": list(removed_path),
        "control_count": int(len(child_joint_ids)),
        "parent_skeleton_lineage_hash": str(parent.skeleton.skeleton_lineage_hash),
        "skeleton_lineage_hash": str(pruned.skeleton_lineage_hash),
        "skin_lineage_hash": skin_hash,
        "prune_receipt": receipt.to_dict(),
        "before_reoptimization": _summary(before),
        "after_reoptimization": _summary(after),
        "optimization": opt_report,
        "motion_source_fidelity": {
            "passed": motion_ok,
            "per_clip": motion_rows,
        },
        "editability": edit,
        "microstress": micro,
        "microstress_pass": micro_ok,
        "product_admissible": product_pass,
    }
    child = ParentState(
        skeleton=pruned,
        weights=Wopt,
        vertex_ids=parent.vertex_ids,
        joint_ids=tuple(child_joint_ids),
        removed_path=removed_path,
        source_basis_id=str(pruned.skeleton_lineage_hash),
    )
    return child, row


def main(args):
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("C51F_CUDA_NOT_AVAILABLE")
    if args.max_depth < 1:
        raise ValueError("C51F_MAX_DEPTH_INVALID")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    ctx = _ctx(args.authority_root, args.run_id)
    rr = ctx["run_root"]
    surface = rigging_surface_from_dict(
        stage_output_payload(
            ctx, "15_RIGGING_SURFACE_QUALIFIED", "RealSaS.RiggingSurfaceIR.v1"
        )
    )
    candidate = canonical_mesh_candidate_from_dict(
        stage_output_payload(
            ctx,
            "18_CANONICAL_MESH_ADDRESSING_BUILD",
            "RealSaS.CanonicalMeshCandidateIR.v1",
        )
    )
    addressing = surface_addressing_from_dict(
        stage_output_payload(
            ctx,
            "18_CANONICAL_MESH_ADDRESSING_BUILD",
            "RealSaS.SurfaceAddressingIR.v1",
        )
    )
    static = static_mesh_qualification_from_dict(
        stage_output_payload(
            ctx,
            "19_STATIC_CANONICAL_MESH_QUALIFIED",
            "RealSaS.StaticCanonicalMeshQualificationIR.v1",
        )
    )
    carrier = build_mechanical_carrier_evidence_v1(
        candidate,
        static_qualification=static,
        surface_addressing=addressing,
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
    c50b = json.loads((args.c50b_dir / "REPORT.json").read_text())
    c51e = json.loads((args.c51e_dir / "REPORT.json").read_text())
    if c50b.get("run_id") != args.run_id or c51e.get("run_id") != args.run_id:
        raise RuntimeError("C51F_INPUT_RUN_ID_DRIFT")
    if c51e.get("status") != "PRODUCT_ADMISSIBLE_24_CONTROL_CHILD_FOUND":
        raise RuntimeError("C51F_REQUIRES_C51E_ADMITTED_24_FRONTIER")

    parents = _load_c51e_frontier(args.c51e_dir, c51e)
    if len(parents) > args.max_parent_count:
        raise RuntimeError(
            f"C51F_NONDOMINATED_PARENT_BOUND_EXCEEDED:{len(parents)}>{args.max_parent_count}"
        )

    carrier_ids = tuple(str(v.candidate_vertex_id) for v in candidate.vertices)
    for parent in parents:
        if tuple(parent.vertex_ids) != carrier_ids:
            raise RuntimeError("C51F_CARRIER_VERTEX_AXIS_DRIFT")

    baseline_retarget = dict(c50b.get("tree_consistent_retarget") or {})
    rest = np.asarray(carrier.positions, np.float64)
    faces = _faces(candidate, carrier_ids)

    rung_reports = []
    explored = []
    final_frontier = parents
    fixed_point = False
    seed_counter = 0

    for depth in range(1, args.max_depth + 1):
        child_states = {}
        child_rows = []
        frontier_rows = []
        parent_basis_ids = [p.source_basis_id for p in final_frontier]

        for parent in final_frontier:
            candidate_ids = tuple(
                sorted(
                    j.canonical_joint_id
                    for j in parent.skeleton.joints
                    if j.canonical_joint_id != parent.skeleton.root_id
                )
            )
            for control_id in candidate_ids:
                seed_counter += 1
                child, row = _evaluate_child(
                    parent=parent,
                    control_id=control_id,
                    rank_seed=depth * 10000 + seed_counter,
                    candidate=candidate,
                    carrier=carrier,
                    surface=surface,
                    cameras=cameras,
                    camera_set=camera_set,
                    policy=policy,
                    source_report=source_report,
                    run_root=rr,
                    rest=rest,
                    faces=faces,
                    baseline_retarget=baseline_retarget,
                    args=args,
                )
                child_rows.append(row)
                explored.append(row)
                child_states[child.source_basis_id] = child
                frontier_rows.append(_frontier_row(row, child.skeleton))

                safe = "__".join(_safe(x) for x in child.removed_path)
                stem = f"d{depth}_{safe}"
                np.savez_compressed(
                    args.out_dir / f"{stem}_WEIGHTS.npz",
                    weights=child.weights,
                    vertex_ids=np.asarray(child.vertex_ids),
                    joint_ids=np.asarray(child.joint_ids),
                )
                write(
                    args.out_dir / f"{stem}_SKELETON.json",
                    child.skeleton.to_dict(),
                )
                print(
                    "C51F_CANDIDATE_RESULT=" + json.dumps(row, sort_keys=True),
                    flush=True,
                )

        if not frontier_rows:
            fixed_point = True
            rung_reports.append(
                {
                    "depth": depth,
                    "parent_basis_ids": parent_basis_ids,
                    "candidate_count": 0,
                    "admitted_count": 0,
                    "status": "NO_CHILDREN",
                }
            )
            break

        frontier = build_control_basis_pareto_frontier_v1(frontier_rows)
        admitted_ids = tuple(
            row.basis_id
            for row in frontier.candidates
            if row.hard_admissible
        )
        admitted_rows = [
            row for row in child_rows if row["skeleton_lineage_hash"] in admitted_ids
        ]
        rung_reports.append(
            {
                "depth": depth,
                "parent_control_count": int(len(final_frontier[0].joint_ids)),
                "child_control_count": (
                    int(admitted_rows[0]["control_count"]) if admitted_rows else None
                ),
                "parent_basis_ids": parent_basis_ids,
                "candidate_count": int(len(child_rows)),
                "product_admissible_child_count": int(
                    sum(bool(row["product_admissible"]) for row in child_rows)
                ),
                "non_dominated_admitted_count": int(len(admitted_ids)),
                "non_dominated_admitted_basis_ids": list(admitted_ids),
                "pareto_frontier": frontier.to_dict(),
                "status": (
                    "ADMITTED_FRONTIER_CONTINUES"
                    if admitted_ids
                    else "LOCAL_PRUNE_FIXED_POINT"
                ),
            }
        )
        if not admitted_ids:
            fixed_point = True
            break
        if len(admitted_ids) > args.max_parent_count:
            raise RuntimeError(
                f"C51F_NONDOMINATED_PARENT_BOUND_EXCEEDED:{len(admitted_ids)}>{args.max_parent_count}"
            )
        final_frontier = [child_states[basis_id] for basis_id in admitted_ids]

    min_control = min(len(p.joint_ids) for p in final_frontier)
    report = {
        "schema": "RealSaS.ControlBasisRecursiveParetoPruneFrontier.v1",
        "status": (
            "LOCAL_PRUNE_FIXED_POINT_FOUND"
            if fixed_point
            else "BOUNDED_DEPTH_REACHED_WITH_ADMITTED_FRONTIER"
        ),
        "decision": (
            "ROUTE_FIXED_POINT_TO_C5_2_CAPABILITY_ATTRIBUTION"
            if fixed_point
            else "CONTINUE_C5_1F_FROM_SEALED_FRONTIER"
        ),
        "run_id": args.run_id,
        "carrier_evidence_hash": carrier.carrier_evidence_hash,
        "initial_parent_count": int(len(parents)),
        "initial_parent_control_count": int(len(parents[0].joint_ids)),
        "max_depth": int(args.max_depth),
        "max_parent_count": int(args.max_parent_count),
        "rungs": rung_reports,
        "explored_candidate_count": int(len(explored)),
        "final_non_dominated_admitted_parent_count": int(len(final_frontier)),
        "final_control_count": int(min_control),
        "final_basis_ids": [p.source_basis_id for p in final_frontier],
        "local_prune_fixed_point_found": bool(fixed_point),
        "teacher_joint_count_used": False,
        "teacher_weights_used": False,
        "training_used": False,
        "product_authority_minted": False,
        "claim_boundary": [
            "Only non-dominated admitted parents are recursively expanded.",
            "Every child is a true reduced skeleton with fresh retarget, fresh skin reoptimization and fresh hard proof.",
            "Knight control count is evidence for this subject only and is never a generic target.",
            "A local fixed point is under this sealed structural-prune and optimization policy, not a global mathematical optimum.",
            "If bounded depth is reached with an admitted frontier, C5.2 remains unauthorized.",
        ],
    }
    write(args.out_dir / "REPORT.json", report)
    print("C51F_RESULT=" + json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--authority-root", type=Path, required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--c50b-dir", type=Path, required=True)
    ap.add_argument("--c51e-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    ap.add_argument("--seed", type=int, default=20261004)
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--frame-batch", type=int, default=9)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--trust-weight", type=float, default=0.0005)
    ap.add_argument("--max-depth", type=int, default=2)
    ap.add_argument("--max-parent-count", type=int, default=8)
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
