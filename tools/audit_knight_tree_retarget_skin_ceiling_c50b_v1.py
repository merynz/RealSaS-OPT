from __future__ import annotations

"""C5.0b: reoptimize skin after tree-consistent retarget changes Q.

Research-only ceiling. No model fit and no product promotion.

This closes the stale-W hole between C4.1 and C5.0:
  C4.1 solved W*(R,Q_old)
  C5.0 changed Q -> Q_tree
  C5.0b solves W*(R,Q_tree)

Carrier and ATLAS rig remain frozen.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from compiler.realsas_compiler_core.artifact_codec_v2 import (
    canonical_mesh_candidate_from_dict,
    deformation_envelope_from_dict,
    mesh_policy_from_dict,
    qualified_camera_set_from_dict,
    qualified_skeleton_from_dict,
    rigging_surface_from_dict,
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
from models.geppetto.reference_strength_v1.rigging_surface_tensorization_v1 import (
    tensorize_rigging_surface_v1,
)
from models.mira.carrier_query_v1 import build_mira_mechanical_carrier_query_v1
from models.shared.joint_mechanical_loss_v1 import (
    JointMechanicalLossConfigV1,
    joint_mechanical_loss_v1,
)
from tools.audit_knight_canonical_caa_mechanics_geometry_lock_v1 import CLIPS
from tools.audit_knight_carrier_first_c3_v1 import (
    _decode_legacy,
    _run_mira_backbone,
    _summarize_g3,
)
from tools.audit_knight_mira_actual_motion_weight_oracle_c41_v1 import (
    _compose,
    _faces,
    _hard_actual_motion,
    _local_problem,
    _one_ring_active,
    _surrogate_eval,
)
from tools.audit_knight_mira_support_mixture_c32_v1 import _legacy_transfer_ceiling
from tools.audit_knight_tree_consistent_retarget_c50_v1 import (
    _motion_matrices,
    _tracks_with_mapping,
    _tree_consistent_mapping,
)
from tools.demo.render_knight_motion_preview_v1 import _ctx
from tools.inference.refined_surface_rig_skin_v1 import write


def _tree_motion_frames(*, run_root, skeleton, cameras, joint_ids, source_report):
    mats = []
    rows = []
    mappings = {}
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
        clip_mats, clip_rows = _motion_matrices(
            payload=payload,
            tracks=tracks,
            skeleton=skeleton,
            cameras=cameras,
            joint_ids=joint_ids,
            clip_id=clip,
        )
        mats.append(clip_mats)
        rows.extend(clip_rows)
        mappings[clip] = {
            "mapping": mapping,
            "detail_count": int(len(details)),
            "total_cost": float(total_cost),
        }
    return np.concatenate(mats, axis=0), tuple(rows), mappings


def main(args):
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("C50B_CUDA_NOT_AVAILABLE")
    dev = torch.device(args.device)

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
    skeleton = qualified_skeleton_from_dict(
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
    envelope = deformation_envelope_from_dict(
        stage_output_payload(
            ctx, "34_DEFORMATION_CAPABILITY_ENVELOPE",
            "RealSaS.DeformationCapabilityEnvelopeIR.v1"
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

    surface_tensor = tensorize_rigging_surface_v1(surface)
    conditioning, ci, memory, tokens, decoder, mira_result, telemetry = _run_mira_backbone(
        surface=surface,
        skeleton=skeleton,
        arm_dir=args.arm_dir,
        device=args.device,
        query_chunk=args.backbone_query_chunk,
    )
    legacy = _decode_legacy(
        conditioning=conditioning,
        ci=ci,
        memory=memory,
        tokens=tokens,
        decoder=decoder,
        device=args.device,
        chunk=args.readout_chunk,
    )
    query = build_mira_mechanical_carrier_query_v1(
        candidate=candidate,
        carrier_evidence=carrier,
        surface_tensor=surface_tensor,
        conditioning=conditioning,
    )
    base_weights = np.asarray(
        _legacy_transfer_ceiling(query=query, legacy_weights=legacy), np.float64
    )

    rest = np.asarray(carrier.positions, np.float64)
    faces = _faces(candidate, query.carrier_vertex_ids)
    matrices, frame_rows, mappings = _tree_motion_frames(
        run_root=rr,
        skeleton=skeleton,
        cameras=cameras,
        joint_ids=query.joint_ids,
        source_report=source_report,
    )
    base_actual = _hard_actual_motion(
        rest=rest,
        faces=faces,
        weights=base_weights,
        matrices=matrices,
        frame_rows=frame_rows,
        policy=policy,
    )

    active, closure_face_mask = _one_ring_active(faces, base_actual["union_bad"])
    local = _local_problem(
        rest, faces, base_weights, active, closure_face_mask, base_actual["union_bad"]
    )

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
    optimizer = torch.optim.Adam([logits], lr=float(args.lr))
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    order = torch.randperm(len(mats_t), generator=generator).tolist()
    cursor = 0

    initial_weights, _ = _compose(base_t, active_idx, logits)
    initial = _surrogate_eval(
        rest_t, closure_faces_t, seed_faces_t, initial_weights, mats_t, cfg, args.frame_batch
    )
    print("C50B_INITIAL=" + json.dumps(initial, sort_keys=True), flush=True)

    best = None
    history = []
    for step in range(1, args.steps + 1):
        if cursor + args.frame_batch > len(order):
            order = torch.randperm(len(mats_t), generator=generator).tolist()
            cursor = 0
        ids = order[cursor:cursor + args.frame_batch]
        cursor += args.frame_batch
        batch = mats_t[torch.as_tensor(ids, device=dev, dtype=torch.long)]

        optimizer.zero_grad(set_to_none=True)
        Wlocal, active_weights = _compose(base_t, active_idx, logits)
        seed = joint_mechanical_loss_v1(rest_t, seed_faces_t, Wlocal, batch, config=cfg)
        closure = joint_mechanical_loss_v1(
            rest_t, closure_faces_t, Wlocal, batch, config=cfg
        )
        trust = (active_weights - base_active).abs().sum(dim=-1).mean()
        loss = seed["loss"] + 0.25 * closure["loss"] + float(args.trust_weight) * trust
        if not torch.isfinite(loss):
            raise RuntimeError("C50B_NONFINITE_LOSS")
        loss.backward()
        torch.nn.utils.clip_grad_norm_([logits], 10.0)
        optimizer.step()

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            current, _ = _compose(base_t, active_idx, logits)
            ev = _surrogate_eval(
                rest_t, closure_faces_t, seed_faces_t, current, mats_t, cfg, args.frame_batch
            )
            with torch.no_grad():
                delta = (current[active_idx] - base_active).abs().sum(dim=-1)
                ev["trust_mean_row_l1"] = float(delta.mean().cpu())
                ev["trust_max_row_l1"] = float(delta.max().cpu())
            row = {"step": int(step), **ev}
            history.append(row)
            key = (ev["mean_loss"], ev["trust_mean_row_l1"], step)
            if best is None or key < best[0]:
                best = (key, logits.detach().cpu().clone(), row)
            print("C50B_STEP=" + json.dumps(row, sort_keys=True), flush=True)

    if best is None:
        raise RuntimeError("C50B_NO_BEST_STATE")

    with torch.no_grad():
        logits.copy_(best[1].to(dev))
        local_final, _ = _compose(base_t, active_idx, logits)

    full = base_weights.copy()
    active_global = np.asarray(local["active_global"], np.int64)
    full[active_global] = local_final[active_idx].detach().cpu().numpy().astype(np.float64)
    full /= full.sum(axis=1, keepdims=True)

    final_actual = _hard_actual_motion(
        rest=rest,
        faces=faces,
        weights=full,
        matrices=matrices,
        frame_rows=frame_rows,
        policy=policy,
    )
    skin = qualify_mechanical_carrier_skin_v1(
        candidate=candidate,
        carrier_evidence=carrier,
        skeleton=skeleton,
        vertex_ids=query.carrier_vertex_ids,
        joint_ids=query.joint_ids,
        weights=full,
        max_simplex_repair_l1=1e-5,
        max_total_correction_l1=0.1,
    )
    g3 = run_g3_local_frame_micro_stress_v2(
        candidate,
        surface=surface,
        skeleton=skeleton,
        skin=skin,
        envelope=envelope,
        cameras=cameras,
        policy=policy,
    )
    g3b = run_skin_topology_compatibility_v1(
        candidate,
        surface=surface,
        skeleton=skeleton,
        skin=skin,
        envelope=envelope,
        cameras=cameras,
        policy=policy,
        stress_all_faces=True,
    )
    micro = _summarize_g3(g3, g3b)

    passed = bool(final_actual["passed"] and micro["g3_passed"] and micro["g3b_passed"])
    report = {
        "schema": "RealSaS.TreeConsistentRetargetSkinReoptimizationCeiling.v1",
        "status": (
            "PASS_CURRENT_RIG_SUFFICIENT_UNDER_TREE_RETARGET"
            if passed
            else "FAIL_CURRENT_RIG_STILL_NOT_SUFFICIENT"
        ),
        "run_id": args.run_id,
        "carrier_evidence_hash": carrier.carrier_evidence_hash,
        "carrier_topology_hash": carrier.topology_hash,
        "atlas_rig_binding_hash": skeleton.skeleton_lineage_hash,
        "current_effective_control_count": int(len(query.joint_ids)),
        "tree_consistent_retarget": mappings,
        "base_actual_motion": {
            k: v for k, v in base_actual.items() if k not in {"union_bad", "per_frame"}
        },
        "active_region": {
            "seed_bad_face_count": int(local["seed_face_count"]),
            "closure_face_count": int(local["closure_face_count"]),
            "active_vertex_count": int(len(active_global)),
        },
        "optimization": {
            "steps": int(args.steps),
            "lr": float(args.lr),
            "frame_batch": int(args.frame_batch),
            "trust_weight": float(args.trust_weight),
            "initial_surrogate": initial,
            "best_surrogate": best[2],
            "history": history,
        },
        "final_actual_motion": {
            k: v for k, v in final_actual.items() if k not in {"union_bad", "per_frame"}
        },
        "microstress_nonregression": micro,
        "training_used": False,
        "teacher_weights_used": False,
        "product_authority_minted": False,
        "decision": (
            "OPEN_C5_1_PRUNE_FROM_ADMISSIBLE_BASELINE"
            if passed
            else "OPEN_C5_1_REDUCED_BASIS_CEILINGS_THEN_C5_2_GROW_OR_CARRIER"
        ),
        "claim_boundary": [
            "Carrier and ATLAS rig are exact-frozen.",
            "The motion domain is the tree-consistent retarget domain.",
            "No teacher skin weights enter the objective.",
            "This is a direct-weight ceiling, not a MIRA fit.",
            "Hard exact-motion replay plus microstress is the verdict.",
        ],
    }

    write(args.out_dir / "REPORT.json", report)
    write(args.out_dir / "OPTIMIZED_CARRIER_SKIN.json", skin.to_dict())
    write(
        args.out_dir / "BASE_ACTUAL_MOTION.json",
        {"summary": report["base_actual_motion"], "frames": base_actual["per_frame"]},
    )
    write(
        args.out_dir / "FINAL_ACTUAL_MOTION.json",
        {"summary": report["final_actual_motion"], "frames": final_actual["per_frame"]},
    )
    np.savez_compressed(
        args.out_dir / "OPTIMIZED_TREE_RETARGET_WEIGHTS.npz",
        base_weights=base_weights,
        optimized_weights=full,
        active_mask=np.isin(np.arange(len(full)), active_global).astype(np.uint8),
        vertex_ids=np.asarray(query.carrier_vertex_ids),
        joint_ids=np.asarray(query.joint_ids),
    )
    print("C50B_RESULT=" + json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--authority-root", type=Path, required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--arm-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    ap.add_argument("--seed", type=int, default=20261004)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--frame-batch", type=int, default=9)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--trust-weight", type=float, default=0.0005)
    ap.add_argument("--backbone-query-chunk", type=int, default=256)
    ap.add_argument("--readout-chunk", type=int, default=128)
    args = ap.parse_args()
    try:
        main(args)
    except Exception as exc:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        write(args.out_dir / "ERROR.json", {"type": type(exc).__name__, "message": str(exc)})
        raise
