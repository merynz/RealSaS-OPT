from __future__ import annotations

"""C5.1e full single-prune local-irreducibility court.

Starting from the exact product-admissible 25-control C5.1d basis, enumerate every
remaining non-root single-control structural deletion. Every 24-control child gets:
- true skeleton prune + child reparenting;
- fresh tree-consistent retarget;
- independent exact-motion skin reoptimization;
- hard 51-frame idle/run/slash proof;
- fresh reduced-skeleton microstress when exact motion closes;
- motion-source nonregression;
- editability/no-op response gate.

This seals a finite candidate family and can prove only local irreducibility under
this exact single-prune + optimization policy. It never claims global optimum.
"""

import argparse
import json
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
    ControlBasisSearchCandidateV1,
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


def _safe_path(parts):
    return "__".join(str(x).replace(":", "_") for x in parts)


def _load_best_c51d(c51d_dir: Path, report: dict):
    best = dict(report.get("best_product_admissible") or {})
    path = tuple(map(str, best.get("removed_path") or ()))
    if not path:
        raise RuntimeError("C51E_C51D_BEST_PATH_MISSING")
    depth = len(path) - 1
    safe = _safe_path(path)
    skeleton_path = c51d_dir / f"depth{depth}_{safe}_SKELETON.json"
    weight_path = c51d_dir / f"depth{depth}_{safe}_WEIGHTS.npz"
    if not skeleton_path.is_file() or not weight_path.is_file():
        raise RuntimeError(
            "C51E_C51D_BEST_ARTIFACT_MISSING:"
            + str(skeleton_path)
            + ":"
            + str(weight_path)
        )
    skeleton = qualified_skeleton_from_dict(json.loads(skeleton_path.read_text()))
    with np.load(weight_path, allow_pickle=False) as z:
        weights = np.asarray(z["weights"], np.float64)
        vertex_ids = tuple(map(str, z["vertex_ids"].tolist()))
        joint_ids = tuple(map(str, z["joint_ids"].tolist()))
    if int(best.get("control_count", -1)) != len(joint_ids):
        raise RuntimeError("C51E_C51D_CONTROL_COUNT_DRIFT")
    if set(joint_ids) != {j.canonical_joint_id for j in skeleton.joints}:
        raise RuntimeError("C51E_C51D_JOINT_AXIS_DRIFT")
    return best, skeleton, weights, vertex_ids, joint_ids, skeleton_path, weight_path


def _summary(actual):
    return {k: v for k, v in actual.items() if k not in {"union_bad", "per_frame"}}


def _tree_depth_complexity(skeleton) -> float:
    by = {str(j.canonical_joint_id): j for j in skeleton.joints}
    cache = {}

    def depth(jid: str) -> int:
        if jid in cache:
            return cache[jid]
        row = by[jid]
        parent = row.parent_canonical_id
        value = 0 if parent is None else 1 + depth(str(parent))
        cache[jid] = value
        return value

    values = [depth(jid) for jid in by]
    return float(sum(values) / max(1, len(values)))


def _frontier_row(row: dict, skeleton) -> ControlBasisSearchCandidateV1:
    after = dict(row["after_reoptimization"])
    edit = dict(row["editability"])
    motion = dict(row["motion_source_fidelity"])
    micro_pass = bool(row.get("microstress_pass"))
    mechanical_defect = float(after["failed_frame_count"]) + float(
        after["unique_bad_face_count"]
    )
    if after["passed"] and not micro_pass:
        mechanical_defect += 1.0
    source_deltas = [
        max(0.0, float(x["delta_candidate_minus_baseline"]))
        for x in dict(motion.get("per_clip") or {}).values()
    ]
    source_defect = max(source_deltas, default=0.0)
    editability_defect = float(edit["numerical_no_op_control_count"])
    admissible = bool(row["product_admissible"])
    margin = 0.0 if admissible else -max(
        1.0e-12,
        mechanical_defect,
        source_defect,
        editability_defect,
    )
    return ControlBasisSearchCandidateV1(
        basis_id=str(row["skeleton_lineage_hash"]),
        control_ids=tuple(sorted(j.canonical_joint_id for j in skeleton.joints)),
        parent_basis_id=str(row["parent_skeleton_lineage_hash"]),
        mechanical_defect=mechanical_defect,
        source_defect=source_defect,
        editability_defect=editability_defect,
        secondary_complexity=_tree_depth_complexity(skeleton),
        hard_admissible=admissible,
        minimum_hard_margin=float(margin),
        metadata={
            "removed_control_id": str(row["removed_control_id"]),
            "court": "C5.1e",
        },
    )


def main(args):
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("C51E_CUDA_NOT_AVAILABLE")
    if args.steps < 1 or args.frame_batch < 1:
        raise ValueError("C51E_OPTIMIZATION_CONFIG_INVALID")

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
    c51d = json.loads((args.c51d_dir / "REPORT.json").read_text())
    if c50b.get("run_id") != args.run_id or c51d.get("run_id") != args.run_id:
        raise RuntimeError("C51E_INPUT_RUN_ID_DRIFT")
    if c51d.get("status") != "PRODUCT_ADMISSIBLE_DESCENDANT_FOUND":
        raise RuntimeError("C51E_REQUIRES_C51D_PRODUCT_ADMISSIBLE_BASE")

    (
        best,
        base_skeleton,
        base_weights,
        vertex_ids,
        joint_ids,
        skeleton_path,
        weight_path,
    ) = _load_best_c51d(args.c51d_dir, c51d)

    carrier_ids = tuple(str(v.candidate_vertex_id) for v in candidate.vertices)
    if tuple(vertex_ids) != carrier_ids:
        raise RuntimeError("C51E_CARRIER_VERTEX_AXIS_DRIFT")
    if len(joint_ids) != 25:
        raise RuntimeError(f"C51E_EXPECT_25_CONTROL_BASE:{len(joint_ids)}")

    baseline_retarget = dict(c50b.get("tree_consistent_retarget") or {})
    rest = np.asarray(carrier.positions, np.float64)
    faces = _faces(candidate, vertex_ids)

    candidate_ids = tuple(
        sorted(
            j.canonical_joint_id
            for j in base_skeleton.joints
            if j.canonical_joint_id != base_skeleton.root_id
        )
    )
    results = []
    frontier_rows = []

    for rank, control_id in enumerate(candidate_ids):
        pruned, receipt = prune_qualified_skeleton_control_v1(
            base_skeleton,
            control_id,
        )
        W0, child_joint_ids = collapse_weight_axis_to_parent_v1(
            base_weights,
            joint_ids,
            removed_control_id=receipt.removed_control_id,
            replacement_parent_id=receipt.replacement_parent_id,
        )
        matrices, frame_rows, retarget = _tree_motion_frames(
            run_root=rr,
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
            seed=args.seed + rank,
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
                vertex_ids=vertex_ids,
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
        row = {
            "removed_control_id": control_id,
            "control_count": int(len(child_joint_ids)),
            "parent_skeleton_lineage_hash": str(base_skeleton.skeleton_lineage_hash),
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
        results.append(row)
        frontier_rows.append(_frontier_row(row, pruned))

        safe = control_id.replace(":", "_")
        np.savez_compressed(
            args.out_dir / f"{rank:02d}_{safe}_WEIGHTS.npz",
            weights=Wopt,
            vertex_ids=np.asarray(vertex_ids),
            joint_ids=np.asarray(child_joint_ids),
        )
        write(
            args.out_dir / f"{rank:02d}_{safe}_SKELETON.json",
            pruned.to_dict(),
        )
        print(
            "C51E_CANDIDATE_RESULT=" + json.dumps(row, sort_keys=True),
            flush=True,
        )

    frontier = build_control_basis_pareto_frontier_v1(frontier_rows)
    admitted = [row for row in results if row["product_admissible"]]
    admitted.sort(
        key=lambda row: (
            row["after_reoptimization"]["maximum_condition"],
            row["removed_control_id"],
        )
    )
    locally_irreducible = not admitted

    report = {
        "schema": "RealSaS.ControlBasisLocalIrreducibilityCourt.v1",
        "status": (
            "LOCALLY_IRREDUCIBLE_UNDER_SEALED_SINGLE_PRUNE_POLICY"
            if locally_irreducible
            else "PRODUCT_ADMISSIBLE_24_CONTROL_CHILD_FOUND"
        ),
        "run_id": args.run_id,
        "carrier_evidence_hash": carrier.carrier_evidence_hash,
        "base_control_count": int(len(joint_ids)),
        "base_skeleton_lineage_hash": str(base_skeleton.skeleton_lineage_hash),
        "base_removed_path": list(best["removed_path"]),
        "base_skeleton_path": str(skeleton_path),
        "base_weights_path": str(weight_path),
        "candidate_count": int(len(candidate_ids)),
        "candidate_control_ids": list(candidate_ids),
        "optimization_policy": {
            "steps": int(args.steps),
            "lr": float(args.lr),
            "frame_batch": int(args.frame_batch),
            "trust_weight": float(args.trust_weight),
            "seed": int(args.seed),
        },
        "product_admissible_24_control_count": int(len(admitted)),
        "product_admissible_removed_control_ids": [
            row["removed_control_id"] for row in admitted
        ],
        "locally_irreducible_under_sealed_policy": bool(locally_irreducible),
        "pareto_frontier": frontier.to_dict(),
        "results": results,
        "teacher_joint_count_used": False,
        "teacher_weights_used": False,
        "training_used": False,
        "product_authority_minted": False,
        "decision": (
            "SEAL_25_CONTROL_BASIS_AS_LOCAL_PRUNE_FIXED_POINT_AND_ROUTE_TO_COMPILER_REPAIR_SUPERVISION"
            if locally_irreducible
            else "CONTINUE_PRUNE_FRONTIER_FROM_ALL_ADMITTED_24_CONTROL_CHILDREN"
        ),
        "claim_boundary": [
            "All remaining non-root single-control deletions from the admitted 25-control basis were enumerated.",
            "Every child received fresh retarget and independent skin reoptimization.",
            "No admissible-child finding is interpreted as a generic joint-count rule.",
            "No-child PASS proves only local irreducibility under this exact structural-prune and optimization policy.",
            "This is research/compiler supervision evidence, not ATLAS or MIRA product success.",
        ],
    }
    write(args.out_dir / "REPORT.json", report)
    print("C51E_RESULT=" + json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--authority-root", type=Path, required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--c50b-dir", type=Path, required=True)
    ap.add_argument("--c51d-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    ap.add_argument("--seed", type=int, default=20261004)
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--frame-batch", type=int, default=9)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--trust-weight", type=float, default=0.0005)
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
