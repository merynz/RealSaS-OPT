from __future__ import annotations

"""C5.1d targeted no-op prune frontier from a mechanically closed reduced basis.

Research/execution court, no model fit and no product promotion.

Starting point:
- C5.1b mechanically closed reduced skeleton/skin
- C5.1c product-admissibility failure caused by exposed numerical no-op controls

For each frontier basis:
1) measure exposed control response;
2) select numerical no-op non-root controls;
3) structurally prune one control;
4) rebuild tree-consistent retarget;
5) reoptimize skin for the exact reduced rig + exact 51-frame motion domain;
6) run exact motion hard proof;
7) if exact motion closes, run fresh reduced-skeleton microstress;
8) run motion-source nonregression + editability;
9) keep provenance for all explored children.

This court finds a product-admissible descendant if one exists in the bounded
no-op-prune neighborhood. It does not claim global minimum control count.
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
from tools.demo.render_knight_motion_preview_v1 import _ctx
from tools.inference.refined_surface_rig_skin_v1 import write


@dataclass
class FrontierNode:
    skeleton: object
    weights: np.ndarray
    joint_ids: tuple[str, ...]
    removed_path: tuple[str, ...]
    depth: int
    lineage_key: str


def _load_c51b_base(*, source_skeleton, c51b_dir: Path, c51b_report: dict):
    admitted = tuple(
        map(str, c51b_report.get("mechanically_admissible_control_ids_removed") or ())
    )
    if len(admitted) != 1:
        raise RuntimeError(
            f"C51D_EXPECT_EXACTLY_ONE_MECHANICAL_BASE:{len(admitted)}"
        )
    removed = admitted[0]
    skeleton, receipt = prune_qualified_skeleton_control_v1(source_skeleton, removed)

    safe = removed.replace(":", "_")
    matches = sorted(c51b_dir.glob(f"*_{safe}_WEIGHTS.npz"))
    if len(matches) != 1:
        raise RuntimeError(f"C51D_BASE_WEIGHT_FILE_ACCOUNTING:{removed}:{len(matches)}")
    with np.load(matches[0], allow_pickle=False) as z:
        weights = np.asarray(z["weights"], np.float64)
        vertex_ids = tuple(map(str, z["vertex_ids"].tolist()))
        joint_ids = tuple(map(str, z["joint_ids"].tolist()))

    if set(joint_ids) != {j.canonical_joint_id for j in skeleton.joints}:
        raise RuntimeError("C51D_BASE_JOINT_AXIS_DRIFT")
    return skeleton, weights, vertex_ids, joint_ids, receipt


def _motion_nonregression(*, retarget: dict, baseline_retarget: dict):
    rows = {}
    passed = True
    for clip, base_row in sorted(baseline_retarget.items()):
        cand = retarget.get(clip)
        if cand is None:
            raise RuntimeError("C51D_RETARGET_CLIP_MISSING:" + clip)
        base_cost = float(base_row["total_cost"])
        cand_cost = float(cand["total_cost"])
        nonreg = bool(cand_cost <= base_cost + 1e-9)
        passed &= nonreg
        rows[clip] = {
            "baseline_tree_cost": base_cost,
            "candidate_tree_cost": cand_cost,
            "delta_candidate_minus_baseline": cand_cost - base_cost,
            "nonregressive": nonreg,
        }
    return bool(passed), rows


def _summarize_actual(value):
    return {k: v for k, v in value.items() if k not in {"union_bad", "per_frame"}}


def _evaluate_node(
    *,
    node: FrontierNode,
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
    vertex_ids,
    baseline_retarget,
    device,
):
    edit = _control_response_report(
        skeleton=node.skeleton,
        cameras=cameras,
        rest=rest,
        weights=node.weights,
        joint_ids=node.joint_ids,
    )
    noops = tuple(
        cid
        for cid in map(str, edit["numerical_no_op_control_ids"])
        if cid != str(node.skeleton.root_id)
    )

    matrices, frame_rows, retarget = _tree_motion_frames(
        run_root=run_root,
        skeleton=node.skeleton,
        cameras=cameras,
        joint_ids=node.joint_ids,
        source_report=source_report,
    )
    motion_ok, motion_rows = _motion_nonregression(
        retarget=retarget,
        baseline_retarget=baseline_retarget,
    )

    skin = qualify_mechanical_carrier_skin_v1(
        candidate=candidate,
        carrier_evidence=carrier,
        skeleton=node.skeleton,
        vertex_ids=vertex_ids,
        joint_ids=node.joint_ids,
        weights=node.weights,
        max_simplex_repair_l1=1e-5,
        max_total_correction_l1=0.1,
    )
    _axis, envelope = derive_deformation_envelope_v1(
        skeleton=node.skeleton,
        camera_set=camera_set,
    )
    g3 = run_g3_local_frame_micro_stress_v2(
        candidate,
        surface=surface,
        skeleton=node.skeleton,
        skin=skin,
        envelope=envelope,
        cameras=cameras,
        policy=policy,
    )
    g3b = run_skin_topology_compatibility_v1(
        candidate,
        surface=surface,
        skeleton=node.skeleton,
        skin=skin,
        envelope=envelope,
        cameras=cameras,
        policy=policy,
        stress_all_faces=True,
    )
    micro = _summarize_g3(g3, g3b)
    micro_ok = bool(micro["g3_passed"] and micro["g3b_passed"])

    return {
        "control_count": int(len(node.joint_ids)),
        "removed_path": list(node.removed_path),
        "skeleton_lineage_hash": str(node.skeleton.skeleton_lineage_hash),
        "skin_lineage_hash": str(skin.skin_lineage_hash),
        "editability": edit,
        "numerical_no_op_control_ids": list(noops),
        "motion_source_fidelity": {
            "passed": motion_ok,
            "per_clip": motion_rows,
        },
        "microstress": micro,
        "microstress_pass": micro_ok,
        "product_admissible": bool(
            motion_ok
            and micro_ok
            and edit["all_exposed_controls_measurable"]
        ),
        "_matrices": matrices,
        "_frame_rows": frame_rows,
        "_retarget": retarget,
    }


def _child_rank(row):
    after = row["after_reoptimization"]
    return (
        0 if row["product_admissible"] else 1,
        int(row["editability"]["numerical_no_op_control_count"]),
        int(after["failed_frame_count"]),
        int(after["unique_bad_face_count"]),
        float(after["maximum_condition"]),
        int(row["control_count"]),
        tuple(row["removed_path"]),
    )


def main(args):
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("C51D_CUDA_NOT_AVAILABLE")
    if args.max_depth < 1 or args.beam_width < 1:
        raise ValueError("C51D_SEARCH_CONFIG_INVALID")

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
            ctx, "05_CAMERA_CONTRACT_SOLVED", "RealSaS.QualifiedCameraSetIR.v1"
        )
    )
    cameras = tuple(sorted(camera_set.cameras, key=lambda x: int(x.view_index)))
    source_report = json.loads(
        Path("canonical/KNIGHT_MOTION_SOURCE_ACTION_DIAGNOSTIC_20260927.json").read_text()
    )

    c50b = json.loads((args.c50b_dir / "REPORT.json").read_text())
    c51b = json.loads((args.c51b_dir / "REPORT.json").read_text())
    c51c = json.loads((args.c51c_dir / "REPORT.json").read_text())
    for name, report in (("C50B", c50b), ("C51B", c51b), ("C51C", c51c)):
        if report.get("run_id") != args.run_id:
            raise RuntimeError(f"C51D_{name}_RUN_ID_DRIFT")

    baseline_retarget = dict(c50b.get("tree_consistent_retarget") or {})
    base_skeleton, base_weights, vertex_ids, base_joint_ids, base_receipt = _load_c51b_base(
        source_skeleton=source_skeleton,
        c51b_dir=args.c51b_dir,
        c51b_report=c51b,
    )
    carrier_ids = tuple(str(v.candidate_vertex_id) for v in candidate.vertices)
    if tuple(vertex_ids) != carrier_ids:
        raise RuntimeError("C51D_CARRIER_VERTEX_AXIS_DRIFT")

    c51c_rows = {
        str(row["removed_control_id"]): row
        for row in c51c.get("results") or ()
    }
    base_removed = str(base_receipt.removed_control_id)
    if base_removed not in c51c_rows:
        raise RuntimeError("C51D_BASE_C51C_RESULT_MISSING")
    initial_noops = tuple(
        map(
            str,
            (
                c51c_rows[base_removed]
                .get("editability", {})
                .get("response", {})
                .get("numerical_no_op_control_ids", ())
            ),
        )
    )
    if not initial_noops:
        raise RuntimeError("C51D_NO_INITIAL_NOOP_CONTROLS")

    rest = np.asarray(carrier.positions, np.float64)
    faces = _faces(candidate, vertex_ids)

    base = FrontierNode(
        skeleton=base_skeleton,
        weights=base_weights,
        joint_ids=base_joint_ids,
        removed_path=(base_removed,),
        depth=0,
        lineage_key=str(base_skeleton.skeleton_lineage_hash),
    )

    explored = []
    product_admissible = []
    frontier = [base]
    seen = {base.lineage_key}

    base_eval = _evaluate_node(
        node=base,
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
        vertex_ids=vertex_ids,
        baseline_retarget=baseline_retarget,
        device=args.device,
    )
    base_public = {k: v for k, v in base_eval.items() if not k.startswith("_")}
    base_public["expected_initial_noops_from_c51c"] = list(initial_noops)
    explored.append({"kind": "BASE_27", **base_public})

    for depth in range(1, args.max_depth + 1):
        candidates = []
        for parent_index, parent in enumerate(frontier):
            parent_eval = _evaluate_node(
                node=parent,
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
                vertex_ids=vertex_ids,
                baseline_retarget=baseline_retarget,
                device=args.device,
            )
            noops = tuple(map(str, parent_eval["numerical_no_op_control_ids"]))
            for control_id in noops:
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
                lineage = str(pruned.skeleton_lineage_hash)
                if lineage in seen:
                    continue
                seen.add(lineage)

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
                    seed=args.seed + depth * 100 + len(candidates),
                    steps=args.steps,
                    lr=args.lr,
                    frame_batch=args.frame_batch,
                    trust_weight=args.trust_weight,
                    log_every=args.log_every,
                    label=control_id,
                )
                child = FrontierNode(
                    skeleton=pruned,
                    weights=Wopt,
                    joint_ids=tuple(child_joint_ids),
                    removed_path=parent.removed_path + (control_id,),
                    depth=depth,
                    lineage_key=lineage,
                )

                edit = _control_response_report(
                    skeleton=pruned,
                    cameras=cameras,
                    rest=rest,
                    weights=Wopt,
                    joint_ids=child_joint_ids,
                )
                motion_ok, motion_rows = _motion_nonregression(
                    retarget=retarget,
                    baseline_retarget=baseline_retarget,
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
                    "kind": "CHILD",
                    "depth": depth,
                    "parent_lineage_hash": parent.lineage_key,
                    "removed_control_id": control_id,
                    "removed_path": list(child.removed_path),
                    "control_count": int(len(child_joint_ids)),
                    "prune_receipt": receipt.to_dict(),
                    "before_reoptimization": _summarize_actual(before),
                    "after_reoptimization": _summarize_actual(after),
                    "optimization": opt_report,
                    "motion_source_fidelity": {
                        "passed": motion_ok,
                        "per_clip": motion_rows,
                    },
                    "editability": edit,
                    "microstress": micro,
                    "microstress_pass": micro_ok,
                    "skin_lineage_hash": skin_hash,
                    "skeleton_lineage_hash": lineage,
                    "product_admissible": product_pass,
                }
                explored.append(row)
                candidates.append((child, row))

                safe = "__".join(x.replace(":", "_") for x in child.removed_path)
                np.savez_compressed(
                    args.out_dir / f"depth{depth}_{safe}_WEIGHTS.npz",
                    weights=Wopt,
                    vertex_ids=np.asarray(vertex_ids),
                    joint_ids=np.asarray(child_joint_ids),
                )
                write(
                    args.out_dir / f"depth{depth}_{safe}_SKELETON.json",
                    pruned.to_dict(),
                )
                print(
                    "C51D_CANDIDATE_RESULT=" + json.dumps(row, sort_keys=True),
                    flush=True,
                )
                if product_pass:
                    product_admissible.append((child, row))

        if not candidates:
            break

        candidates.sort(key=lambda pair: _child_rank(pair[1]))
        frontier = [node for node, _row in candidates[: args.beam_width]]

    product_rows = [row for _node, row in product_admissible]
    product_rows.sort(
        key=lambda row: (
            int(row["control_count"]),
            int(row["editability"]["numerical_no_op_control_count"]),
            float(row["after_reoptimization"]["maximum_condition"]),
            tuple(row["removed_path"]),
        )
    )

    report = {
        "schema": "RealSaS.TargetedNoOpPruneFrontier.v1",
        "status": (
            "PRODUCT_ADMISSIBLE_DESCENDANT_FOUND"
            if product_rows
            else "NO_PRODUCT_ADMISSIBLE_DESCENDANT_IN_BOUNDED_NOOP_FRONTIER"
        ),
        "run_id": args.run_id,
        "carrier_evidence_hash": carrier.carrier_evidence_hash,
        "source_control_count": int(len(source_skeleton.joints)),
        "mechanically_closed_base_control_count": int(len(base_joint_ids)),
        "mechanically_closed_base_removed_path": list(base.removed_path),
        "initial_noop_control_ids": list(initial_noops),
        "max_depth": int(args.max_depth),
        "beam_width": int(args.beam_width),
        "explored_candidate_count": int(sum(1 for row in explored if row["kind"] == "CHILD")),
        "product_admissible_candidate_count": int(len(product_rows)),
        "best_product_admissible": product_rows[0] if product_rows else None,
        "explored": explored,
        "teacher_joint_count_used": False,
        "teacher_weights_used": False,
        "training_used": False,
        "product_authority_minted": False,
        "decision": (
            "SEED_FULL_LOCAL_IRREDUCIBILITY_SEARCH_FROM_BEST_ADMISSIBLE_DESCENDANT"
            if product_rows
            else "OPEN_C5_2_GROW_LOCUS_OR_CARRIER_ATTRIBUTION"
        ),
        "claim_boundary": [
            "The search starts from the C5.1b mechanically closed 27-control basis.",
            "Only controls measured as numerical no-ops in the current frontier state are targeted.",
            "Every child is a true reduced skeleton with fresh retarget and skin reoptimization.",
            "Product admissibility requires exact-motion PASS, fresh microstress PASS, motion-source nonregression, and measurable response from every remaining exposed control.",
            "A PASS is a bounded-frontier result, not proof of global optimum count.",
        ],
    }
    write(args.out_dir / "REPORT.json", report)
    print("C51D_RESULT=" + json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--authority-root", type=Path, required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--c50b-dir", type=Path, required=True)
    ap.add_argument("--c51b-dir", type=Path, required=True)
    ap.add_argument("--c51c-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    ap.add_argument("--seed", type=int, default=20261004)
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--frame-batch", type=int, default=9)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--trust-weight", type=float, default=0.0005)
    ap.add_argument("--max-depth", type=int, default=2)
    ap.add_argument("--beam-width", type=int, default=4)
    args = ap.parse_args()
    try:
        main(args)
    except Exception as exc:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        write(args.out_dir / "ERROR.json", {"type": type(exc).__name__, "message": str(exc)})
        raise
