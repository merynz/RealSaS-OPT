from __future__ import annotations

"""C5.1b structural prune-to-proof mechanical ceiling.

For each control that passed C5.1a's cheap screen:
- truly remove the non-root control from the skeleton;
- reparent its children to the removed control's parent;
- remove the weight column and conserve mass at the parent;
- recompute tree-consistent retarget on the reduced skeleton;
- reoptimize skin on the exact motion offender closure;
- replay all 51 idle/run/slash frames;
- only if exact motion closes, run fresh reduced-skeleton Stage34-style microstress.

No teacher joint count or teacher weights are used. This court only establishes
mechanical admissibility; editability/source-fidelity gates remain separate.
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
    DEFAULT_MAX_EDGE_RATIO,
    run_skin_topology_compatibility_v1,
)
from compiler.realsas_compiler_core.surface_addressing_v1 import (
    static_mesh_qualification_from_dict,
    surface_addressing_from_dict,
)
from compiler.realsas_compiler_services.orchestrator.adapters.adapter_io import (
    stage_output_payload,
)
from models.shared.joint_mechanical_loss_v1 import (
    JointMechanicalLossConfigV1,
    joint_mechanical_loss_v1,
)
from tools.audit_knight_carrier_first_c3_v1 import _summarize_g3
from tools.audit_knight_mira_actual_motion_weight_oracle_c41_v1 import (
    _compose,
    _faces,
    _hard_actual_motion,
    _local_problem,
    _one_ring_active,
    _surrogate_eval,
)
from tools.audit_knight_tree_consistent_retarget_c50_v1 import (
    _motion_matrices,
    _tracks_with_mapping,
    _tree_consistent_mapping,
)
from tools.audit_knight_canonical_caa_mechanics_geometry_lock_v1 import CLIPS
from tools.demo.render_knight_motion_preview_v1 import _ctx
from tools.inference.refined_surface_rig_skin_v1 import write


def _tree_motion_frames(*, run_root, skeleton, cameras, joint_ids, source_report):
    mats = []
    rows = []
    receipts = {}
    for clip in CLIPS:
        payload = json.loads(
            (
                run_root
                / "inputs"
                / "motion"
                / "quaternius_knight_v1"
                / f"{clip}.motion.json"
            ).read_text()
        )
        mapping, details, srows, _trows, tpar, total_cost = _tree_consistent_mapping(
            payload, skeleton, source_report
        )
        tracks = _tracks_with_mapping(
            payload, skeleton, cameras, mapping, srows, tpar
        )
        cmats, crows = _motion_matrices(
            payload=payload,
            tracks=tracks,
            skeleton=skeleton,
            cameras=cameras,
            joint_ids=joint_ids,
            clip_id=clip,
        )
        mats.append(cmats)
        rows.extend(crows)
        receipts[clip] = {
            "mapping": mapping,
            "detail_count": int(len(details)),
            "total_cost": float(total_cost),
        }
    return np.concatenate(mats, axis=0), tuple(rows), receipts


def _optimize_skin(
    *,
    rest,
    faces,
    base_weights,
    matrices,
    frame_rows,
    policy,
    device,
    seed,
    steps,
    lr,
    frame_batch,
    trust_weight,
    log_every,
    label,
):
    baseline = _hard_actual_motion(
        rest=rest,
        faces=faces,
        weights=base_weights,
        matrices=matrices,
        frame_rows=frame_rows,
        policy=policy,
    )
    if baseline["passed"]:
        return base_weights.copy(), baseline, baseline, {
            "steps": 0,
            "baseline_already_passed": True,
            "history": [],
        }

    active, closure_face_mask = _one_ring_active(faces, baseline["union_bad"])
    local = _local_problem(
        rest, faces, base_weights, active, closure_face_mask, baseline["union_bad"]
    )

    dev = torch.device(device)
    rest_t = torch.as_tensor(local["rest"], device=dev, dtype=torch.float32)
    closure_faces_t = torch.as_tensor(local["closure_faces"], device=dev, dtype=torch.long)
    seed_faces_t = torch.as_tensor(local["seed_faces"], device=dev, dtype=torch.long)
    base_t = torch.as_tensor(local["base_weights"], device=dev, dtype=torch.float32)
    active_idx = torch.as_tensor(local["active_local"], device=dev, dtype=torch.long)
    base_active = base_t[active_idx].detach()
    logits = torch.nn.Parameter(torch.log(base_active.clamp_min(1e-12)))
    mats_t = torch.as_tensor(matrices, device=dev, dtype=torch.float32)

    cfg = JointMechanicalLossConfigV1(
        max_condition=float(policy.g3_max_dynamic_condition_number),
        min_area_ratio=float(policy.g3_min_dynamic_area_ratio),
        max_area_ratio=float(policy.g3_max_dynamic_area_ratio),
        max_edge_ratio=float(DEFAULT_MAX_EDGE_RATIO),
    )
    opt = torch.optim.Adam([logits], lr=float(lr))
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    order = torch.randperm(len(mats_t), generator=generator).tolist()
    cursor = 0
    best = None
    history = []

    for step in range(1, int(steps) + 1):
        if cursor + frame_batch > len(order):
            order = torch.randperm(len(mats_t), generator=generator).tolist()
            cursor = 0
        ids = order[cursor:cursor + frame_batch]
        cursor += frame_batch
        batch = mats_t[torch.as_tensor(ids, device=dev, dtype=torch.long)]

        opt.zero_grad(set_to_none=True)
        Wlocal, active_weights = _compose(base_t, active_idx, logits)
        seed_loss = joint_mechanical_loss_v1(
            rest_t, seed_faces_t, Wlocal, batch, config=cfg
        )
        closure_loss = joint_mechanical_loss_v1(
            rest_t, closure_faces_t, Wlocal, batch, config=cfg
        )
        trust = (active_weights - base_active).abs().sum(dim=-1).mean()
        loss = (
            seed_loss["loss"]
            + 0.25 * closure_loss["loss"]
            + float(trust_weight) * trust
        )
        if not torch.isfinite(loss):
            raise RuntimeError(f"C51B_NONFINITE_LOSS:{label}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_([logits], 10.0)
        opt.step()

        if step == 1 or step % log_every == 0 or step == steps:
            current, _ = _compose(base_t, active_idx, logits)
            ev = _surrogate_eval(
                rest_t,
                closure_faces_t,
                seed_faces_t,
                current,
                mats_t,
                cfg,
                frame_batch,
            )
            with torch.no_grad():
                delta = (current[active_idx] - base_active).abs().sum(dim=-1)
                ev["trust_mean_row_l1"] = float(delta.mean().cpu())
            row = {"step": int(step), **ev}
            history.append(row)
            key = (ev["mean_loss"], ev["trust_mean_row_l1"], step)
            if best is None or key < best[0]:
                best = (key, logits.detach().cpu().clone(), row)
            print(
                "C51B_STEP="
                + json.dumps({"control_id": label, **row}, sort_keys=True),
                flush=True,
            )

    if best is None:
        raise RuntimeError(f"C51B_NO_BEST_STATE:{label}")

    with torch.no_grad():
        logits.copy_(best[1].to(dev))
        local_final, _ = _compose(base_t, active_idx, logits)

    full = base_weights.copy()
    active_global = np.asarray(local["active_global"], np.int64)
    full[active_global] = (
        local_final[active_idx].detach().cpu().numpy().astype(np.float64)
    )
    full /= full.sum(axis=1, keepdims=True)

    final = _hard_actual_motion(
        rest=rest,
        faces=faces,
        weights=full,
        matrices=matrices,
        frame_rows=frame_rows,
        policy=policy,
    )
    return full, baseline, final, {
        "steps": int(steps),
        "active_vertex_count": int(len(active_global)),
        "seed_bad_face_count": int(local["seed_face_count"]),
        "closure_face_count": int(local["closure_face_count"]),
        "best_surrogate": best[2],
        "history": history,
    }


def _summary(actual):
    return {
        k: v for k, v in actual.items() if k not in {"union_bad", "per_frame"}
    }


def main(args):
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("C51B_CUDA_NOT_AVAILABLE")
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
        stage_output_payload(ctx, "15_RIGGING_SURFACE_QUALIFIED", "RealSaS.RiggingSurfaceIR.v1")
    )
    source_skeleton = qualified_skeleton_from_dict(
        stage_output_payload(ctx, "28_SKELETON_QUALIFIED", "RealSaS.QualifiedSkeletonIR.v1")
    )
    candidate = canonical_mesh_candidate_from_dict(
        stage_output_payload(
            ctx, "18_CANONICAL_MESH_ADDRESSING_BUILD", "RealSaS.CanonicalMeshCandidateIR.v1"
        )
    )
    addressing = surface_addressing_from_dict(
        stage_output_payload(
            ctx, "18_CANONICAL_MESH_ADDRESSING_BUILD", "RealSaS.SurfaceAddressingIR.v1"
        )
    )
    static = static_mesh_qualification_from_dict(
        stage_output_payload(
            ctx, "19_STATIC_CANONICAL_MESH_QUALIFIED",
            "RealSaS.StaticCanonicalMeshQualificationIR.v1"
        )
    )
    carrier = build_mechanical_carrier_evidence_v1(
        candidate, static_qualification=static, surface_addressing=addressing
    )
    policy = mesh_policy_from_dict(
        stage_output_payload(
            ctx, "18_CANONICAL_MESH_ADDRESSING_BUILD", "RealSaS.MeshQualificationPolicyIR.v1"
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

    c50b_report = json.loads((args.c50b_dir / "REPORT.json").read_text())
    c51a_report = json.loads((args.c51a_dir / "REPORT.json").read_text())
    if c50b_report.get("run_id") != args.run_id or c51a_report.get("run_id") != args.run_id:
        raise RuntimeError("C51B_INPUT_RUN_ID_DRIFT")

    with np.load(
        args.c50b_dir / "OPTIMIZED_TREE_RETARGET_WEIGHTS.npz",
        allow_pickle=False,
    ) as z:
        base_weights = np.asarray(z["optimized_weights"], np.float64)
        vertex_ids = tuple(map(str, z["vertex_ids"].tolist()))
        joint_ids = tuple(map(str, z["joint_ids"].tolist()))

    source_joint_ids = {j.canonical_joint_id for j in source_skeleton.joints}
    if set(joint_ids) != source_joint_ids:
        raise RuntimeError("C51B_SOURCE_JOINT_AXIS_DRIFT")

    screened = tuple(map(str, c51a_report.get("nonregressive_control_ids") or ()))
    if not screened:
        raise RuntimeError("C51B_NO_SCREENED_CONTROLS")
    if args.max_candidates > 0:
        screened = screened[: args.max_candidates]

    rest = np.asarray(carrier.positions, np.float64)
    faces = _faces(candidate, vertex_ids)
    results = []

    for rank, control_id in enumerate(screened):
        pruned, receipt = prune_qualified_skeleton_control_v1(
            source_skeleton, control_id
        )
        W0, pruned_joint_ids = collapse_weight_axis_to_parent_v1(
            base_weights,
            joint_ids,
            removed_control_id=receipt.removed_control_id,
            replacement_parent_id=receipt.replacement_parent_id,
        )

        matrices, frame_rows, retarget = _tree_motion_frames(
            run_root=rr,
            skeleton=pruned,
            cameras=cameras,
            joint_ids=pruned_joint_ids,
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

        micro = None
        mechanically_admissible = False
        skin_hash = None
        if after["passed"]:
            _axis, envelope = derive_deformation_envelope_v1(
                skeleton=pruned, camera_set=camera_set
            )
            skin = qualify_mechanical_carrier_skin_v1(
                candidate=candidate,
                carrier_evidence=carrier,
                skeleton=pruned,
                vertex_ids=vertex_ids,
                joint_ids=pruned_joint_ids,
                weights=Wopt,
                max_simplex_repair_l1=1e-5,
                max_total_correction_l1=0.1,
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
            mechanically_admissible = bool(micro["g3_passed"] and micro["g3b_passed"])
            skin_hash = skin.skin_lineage_hash
            write(
                args.out_dir / f"{rank:02d}_{control_id.replace(':','_')}_SKIN.json",
                skin.to_dict(),
            )

        result = {
            "control_id": control_id,
            "prune_receipt": receipt.to_dict(),
            "reduced_control_count": int(len(pruned_joint_ids)),
            "retarget": retarget,
            "before_reoptimization": _summary(before),
            "after_reoptimization": _summary(after),
            "optimization": opt_report,
            "microstress": micro,
            "mechanically_admissible": bool(mechanically_admissible),
            "skin_lineage_hash": skin_hash,
        }
        results.append(result)
        np.savez_compressed(
            args.out_dir / f"{rank:02d}_{control_id.replace(':','_')}_WEIGHTS.npz",
            weights=Wopt,
            vertex_ids=np.asarray(vertex_ids),
            joint_ids=np.asarray(pruned_joint_ids),
        )
        print(
            "C51B_CANDIDATE_RESULT=" + json.dumps(result, sort_keys=True),
            flush=True,
        )

    admitted = [row for row in results if row["mechanically_admissible"]]
    results.sort(
        key=lambda row: (
            not row["mechanically_admissible"],
            row["after_reoptimization"]["failed_frame_count"],
            row["after_reoptimization"]["unique_bad_face_count"],
            row["after_reoptimization"]["maximum_condition"],
            row["control_id"],
        )
    )

    report = {
        "schema": "RealSaS.StructuralPruneToProofMechanicalCeiling.v1",
        "status": (
            "MECHANICALLY_ADMISSIBLE_REDUCED_BASIS_FOUND"
            if admitted
            else "NO_SINGLE_PRUNE_MECHANICAL_CLOSURE"
        ),
        "run_id": args.run_id,
        "source_control_count": int(len(joint_ids)),
        "screened_candidate_count": int(len(screened)),
        "mechanically_admissible_candidate_count": int(len(admitted)),
        "mechanically_admissible_control_ids_removed": sorted(
            row["control_id"] for row in admitted
        ),
        "ranked_results": results,
        "training_used": False,
        "teacher_weights_used": False,
        "teacher_joint_count_used": False,
        "product_authority_minted": False,
        "decision": (
            "OPEN_EDITABILITY_SOURCE_GATES_THEN_CONTINUE_PRUNE_FIXED_POINT"
            if admitted
            else "OPEN_C5_2_GROW_TO_NEED_LOCUS_OR_CARRIER"
        ),
        "claim_boundary": [
            "Each candidate is a true reduced skeleton with the joint removed and children reparented.",
            "Skin is reoptimized independently for every reduced basis.",
            "Exact motion must pass before expensive microstress is run.",
            "Mechanical admissibility alone does not authorize product pruning; source fidelity/editability gates remain required.",
            "No requested joint count is used.",
        ],
    }
    write(args.out_dir / "REPORT.json", report)
    print("C51B_RESULT=" + json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--authority-root", type=Path, required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--c50b-dir", type=Path, required=True)
    ap.add_argument("--c51a-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    ap.add_argument("--seed", type=int, default=20261004)
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--frame-batch", type=int, default=9)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--trust-weight", type=float, default=0.0005)
    ap.add_argument("--max-candidates", type=int, default=0)
    args = ap.parse_args()
    try:
        main(args)
    except Exception as exc:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        write(args.out_dir / "ERROR.json", {"type": type(exc).__name__, "message": str(exc)})
        raise
