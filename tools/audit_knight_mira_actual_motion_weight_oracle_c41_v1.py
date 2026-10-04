from __future__ import annotations

"""C4.1 exact-motion skin-only direct-weight ceiling.

Research-only owner-attribution oracle. This is NOT a model fit and does not mint
product authority.

Question:
    On the exact Stage19 carrier and frozen A2-verified ATLAS rig, can skin weights
    alone close the exact current 51-frame idle/run/slash triangle-conditioning
    court when optimization sees those actual retargeted transforms?

No teacher weights enter the objective. The starting field is the frozen MIRA GSA
semantic field deterministically projected onto the exact carrier. Only rows in the
actual-motion offender region plus one carrier 1-ring closure are allowed to move.

If this oracle cannot close the exact-motion court, a MIRA fit is not justified:
the next owner must be rig/control basis and/or carrier repair.
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
    _triangle_metrics_batch_exact,
    run_g3_local_frame_micro_stress_v2,
)
from compiler.realsas_compiler_core.mesh.skin_topology_compatibility_v1 import (
    DEFAULT_MAX_EDGE_RATIO,
    run_skin_topology_compatibility_v1,
)
from compiler.realsas_compiler_core.motion_dynamic_proof_v2 import _joint_pose_v2
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
from tools.audit_knight_canonical_caa_mechanics_geometry_lock_v1 import (
    CLIPS,
    FULL_MOTION_SAMPLES,
)
from tools.audit_knight_carrier_first_c3_v1 import (
    _decode_legacy,
    _run_mira_backbone,
    _summarize_g3,
)
from tools.audit_knight_mira_support_mixture_c32_v1 import (
    _legacy_transfer_ceiling,
)
from tools.demo.render_knight_motion_preview_v1 import (
    _ctx,
    _skin,
    _tracks_for_clip,
)
from tools.inference.refined_surface_rig_skin_v1 import write


def _faces(candidate, vertex_ids):
    row = {str(v): i for i, v in enumerate(vertex_ids)}
    out = []
    for face in candidate.faces:
        ids = tuple(map(str, face))
        if len(ids) != 3 or len(set(ids)) != 3 or any(v not in row for v in ids):
            raise RuntimeError("C41_FACE_AXIS_DRIFT")
        out.append(tuple(row[v] for v in ids))
    return np.asarray(out, dtype=np.int64)


def _actual_motion_frames(*, run_root, skeleton, cameras, joint_ids):
    source_report = json.loads(
        Path("canonical/KNIGHT_MOTION_SOURCE_ACTION_DIAGNOSTIC_20260927.json").read_text()
    )
    matrices = []
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
        tracks, mapping = _tracks_for_clip(payload, skeleton, cameras, source_report)
        mappings[clip] = mapping
        times = np.linspace(
            0.0,
            float(payload["duration_seconds"]),
            FULL_MOTION_SAMPLES,
            endpoint=not bool(payload.get("loop")),
        )
        for frame_index, t in enumerate(times):
            skin, _positions, _frame_hash = _joint_pose_v2(
                skeleton=skeleton,
                tracks=tracks,
                time_seconds=float(t),
                cameras=cameras,
            )
            matrices.append(
                np.stack([np.asarray(skin[jid], dtype=np.float64) for jid in joint_ids])
            )
            rows.append(
                {
                    "clip_id": str(clip),
                    "frame_index": int(frame_index),
                    "time_seconds": float(t),
                }
            )
    if len(matrices) != len(CLIPS) * FULL_MOTION_SAMPLES:
        raise RuntimeError("C41_ACTUAL_FRAME_COUNT_DRIFT")
    return np.asarray(matrices, dtype=np.float64), tuple(rows), mappings


def _hard_actual_motion(*, rest, faces, weights, matrices, frame_rows, policy):
    union_bad = np.zeros(len(faces), dtype=bool)
    union_edge4 = np.zeros(len(faces), dtype=bool)
    union_edge10 = np.zeros(len(faces), dtype=bool)
    worst_condition = np.ones(len(faces), dtype=np.float64)
    worst_edge = np.ones(len(faces), dtype=np.float64)
    min_area = np.ones(len(faces), dtype=np.float64)
    max_area = np.ones(len(faces), dtype=np.float64)
    per_frame = []

    hom = np.concatenate(
        [np.asarray(rest, np.float64), np.ones((len(rest), 1), dtype=np.float64)],
        axis=1,
    )
    W = np.asarray(weights, np.float64)
    for qi, mats in enumerate(np.asarray(matrices, np.float64)):
        per_joint = np.stack(
            [(hom @ np.asarray(mats[j]).T)[:, :3] for j in range(mats.shape[0])],
            axis=1,
        )
        posed = np.sum(per_joint * W[:, :, None], axis=1)
        area, condition, edge_min, edge_max, smin = _triangle_metrics_batch_exact(
            np.asarray(rest, np.float64), posed, faces
        )
        finite = (
            np.isfinite(area)
            & np.isfinite(condition)
            & np.isfinite(edge_min)
            & np.isfinite(edge_max)
            & np.isfinite(smin)
        )
        bad = (
            (~finite)
            | (area < float(policy.g3_min_dynamic_area_ratio))
            | (area > float(policy.g3_max_dynamic_area_ratio))
            | (condition > float(policy.g3_max_dynamic_condition_number))
        )
        edge4 = edge_max > float(DEFAULT_MAX_EDGE_RATIO)
        edge10 = edge_max > 10.0
        union_bad |= bad
        union_edge4 |= edge4
        union_edge10 |= edge10
        worst_condition = np.maximum(worst_condition, condition)
        worst_edge = np.maximum(worst_edge, edge_max)
        min_area = np.minimum(min_area, area)
        max_area = np.maximum(max_area, area)
        per_frame.append(
            {
                **frame_rows[qi],
                "bad_face_count": int(np.count_nonzero(bad)),
                "edge_gt4_count": int(np.count_nonzero(edge4)),
                "edge_gt10_count": int(np.count_nonzero(edge10)),
                "maximum_condition": float(np.max(condition)),
                "maximum_edge_ratio": float(np.max(edge_max)),
                "minimum_area_ratio": float(np.min(area)),
                "maximum_area_ratio": float(np.max(area)),
            }
        )

    return {
        "passed": bool(not np.any(union_bad)),
        "failed_frame_count": int(sum(row["bad_face_count"] > 0 for row in per_frame)),
        "unique_bad_face_count": int(np.count_nonzero(union_bad)),
        "unique_edge_gt4_face_count": int(np.count_nonzero(union_edge4)),
        "unique_edge_gt10_face_count": int(np.count_nonzero(union_edge10)),
        "maximum_condition": float(np.max(worst_condition)),
        "maximum_edge_ratio": float(np.max(worst_edge)),
        "minimum_area_ratio": float(np.min(min_area)),
        "maximum_area_ratio": float(np.max(max_area)),
        "union_bad": union_bad,
        "per_frame": per_frame,
    }


def _one_ring_active(faces, seed_bad):
    seed_faces = np.asarray(seed_bad, bool)
    if not np.any(seed_faces):
        return np.zeros(int(faces.max()) + 1, bool), seed_faces
    seed_vertices = np.unique(faces[seed_faces].reshape(-1))
    touched = np.isin(faces, seed_vertices).any(axis=1)
    closure_vertices = np.unique(faces[touched].reshape(-1))
    active = np.zeros(int(faces.max()) + 1, bool)
    active[closure_vertices] = True
    return active, touched


def _local_problem(rest, faces, base_weights, active, closure_faces, seed_bad):
    selected_faces = faces[closure_faces]
    used = np.asarray(sorted(set(selected_faces.reshape(-1).tolist())), np.int64)
    remap = np.full(len(rest), -1, np.int64)
    remap[used] = np.arange(len(used), dtype=np.int64)
    local_faces = remap[selected_faces]

    global_to_local = {int(g): i for i, g in enumerate(used.tolist())}
    active_global = np.flatnonzero(active).astype(np.int64)
    active_local = np.asarray([global_to_local[int(g)] for g in active_global], np.int64)

    seed_global_faces = faces[seed_bad]
    local_seed_faces = np.asarray(
        [[global_to_local[int(v)] for v in face] for face in seed_global_faces],
        np.int64,
    )
    return {
        "used_global": used,
        "active_global": active_global,
        "active_local": active_local,
        "rest": np.asarray(rest, np.float64)[used],
        "base_weights": np.asarray(base_weights, np.float64)[used],
        "closure_faces": local_faces,
        "seed_faces": local_seed_faces,
        "closure_face_count": int(len(local_faces)),
        "seed_face_count": int(len(local_seed_faces)),
    }


def _compose(base, active_idx, logits):
    active = torch.softmax(logits, dim=-1)
    inserted = torch.zeros_like(base).index_copy(0, active_idx, active)
    mask = torch.zeros((base.shape[0], 1), device=base.device, dtype=base.dtype)
    mask = mask.index_fill(0, active_idx, 1.0)
    return base * (1.0 - mask) + inserted, active


def _surrogate_eval(rest, closure_faces, seed_faces, weights, mats, cfg, batch):
    rows = []
    with torch.no_grad():
        for start in range(0, len(mats), batch):
            q = mats[start : start + batch]
            seed = joint_mechanical_loss_v1(
                rest, seed_faces, weights, q, config=cfg
            )
            closure = joint_mechanical_loss_v1(
                rest, closure_faces, weights, q, config=cfg
            )
            rows.append(
                {
                    "loss": float((seed["loss"] + 0.25 * closure["loss"]).cpu()),
                    "max_condition": max(
                        float(seed["max_condition"].cpu()),
                        float(closure["max_condition"].cpu()),
                    ),
                    "min_area": min(
                        float(seed["min_area_ratio"].cpu()),
                        float(closure["min_area_ratio"].cpu()),
                    ),
                    "max_area": max(
                        float(seed["max_area_ratio"].cpu()),
                        float(closure["max_area_ratio"].cpu()),
                    ),
                    "max_edge": max(
                        float(seed["max_edge_ratio"].cpu()),
                        float(closure["max_edge_ratio"].cpu()),
                    ),
                }
            )
    return {
        "mean_loss": float(np.mean([x["loss"] for x in rows])),
        "max_condition": float(max(x["max_condition"] for x in rows)),
        "min_area_ratio": float(min(x["min_area"] for x in rows)),
        "max_area_ratio": float(max(x["max_area"] for x in rows)),
        "max_edge_ratio": float(max(x["max_edge"] for x in rows)),
    }


def main(args):
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("C41_CUDA_NOT_AVAILABLE")
    if args.steps < 1 or args.lr <= 0 or args.frame_batch < 1 or args.log_every < 1:
        raise ValueError("C41_HYPERPARAM_INVALID")
    if args.trust_weight < 0:
        raise ValueError("C41_TRUST_WEIGHT_NEGATIVE")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = torch.device(args.device)

    ctx = _ctx(args.authority_root, args.run_id)
    rr = ctx["run_root"]
    surface = rigging_surface_from_dict(
        stage_output_payload(
            ctx, "15_RIGGING_SURFACE_QUALIFIED", "RealSaS.RiggingSurfaceIR.v1"
        )
    )
    skeleton = qualified_skeleton_from_dict(
        stage_output_payload(
            ctx, "28_SKELETON_QUALIFIED", "RealSaS.QualifiedSkeletonIR.v1"
        )
    )
    candidate = canonical_mesh_candidate_from_dict(
        stage_output_payload(
            ctx, "18_CANONICAL_MESH_ADDRESSING_BUILD",
            "RealSaS.CanonicalMeshCandidateIR.v1",
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
            "RealSaS.StaticCanonicalMeshQualificationIR.v1",
        )
    )
    carrier = build_mechanical_carrier_evidence_v1(
        candidate, static_qualification=static, surface_addressing=addressing
    )
    policy = mesh_policy_from_dict(
        stage_output_payload(
            ctx, "18_CANONICAL_MESH_ADDRESSING_BUILD",
            "RealSaS.MeshQualificationPolicyIR.v1",
        )
    )
    envelope = deformation_envelope_from_dict(
        stage_output_payload(
            ctx, "34_DEFORMATION_CAPABILITY_ENVELOPE",
            "RealSaS.DeformationCapabilityEnvelopeIR.v1",
        )
    )
    camera_set = qualified_camera_set_from_dict(
        stage_output_payload(
            ctx, "05_CAMERA_CONTRACT_SOLVED", "RealSaS.QualifiedCameraSetIR.v1"
        )
    )
    cameras = tuple(sorted(camera_set.cameras, key=lambda x: int(x.view_index)))
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
    base_weights = _legacy_transfer_ceiling(query=query, legacy_weights=legacy)
    base_weights = np.asarray(base_weights, np.float64)

    rest = np.asarray(carrier.positions, np.float64)
    faces = _faces(candidate, query.carrier_vertex_ids)
    matrices, frame_rows, mappings = _actual_motion_frames(
        run_root=rr,
        skeleton=skeleton,
        cameras=cameras,
        joint_ids=query.joint_ids,
    )
    base_actual = _hard_actual_motion(
        rest=rest,
        faces=faces,
        weights=base_weights,
        matrices=matrices,
        frame_rows=frame_rows,
        policy=policy,
    )
    if base_actual["passed"]:
        raise RuntimeError("C41_BASE_ALREADY_ACTUAL_MOTION_PASS")

    active, closure_face_mask = _one_ring_active(faces, base_actual["union_bad"])
    local = _local_problem(
        rest,
        faces,
        base_weights,
        active,
        closure_face_mask,
        base_actual["union_bad"],
    )

    rest_t = torch.as_tensor(local["rest"], device=dev, dtype=torch.float32)
    closure_faces_t = torch.as_tensor(
        local["closure_faces"], device=dev, dtype=torch.long
    )
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
    permutation = torch.randperm(len(mats_t), generator=generator).tolist()
    cursor = 0

    initial_weights, _ = _compose(base_t, active_idx, logits)
    initial_surrogate = _surrogate_eval(
        rest_t,
        closure_faces_t,
        seed_faces_t,
        initial_weights,
        mats_t,
        cfg,
        args.frame_batch,
    )
    print("C41_INITIAL=" + json.dumps(initial_surrogate, sort_keys=True), flush=True)

    best = None
    history = []
    for step in range(1, args.steps + 1):
        if cursor + args.frame_batch > len(permutation):
            permutation = torch.randperm(len(mats_t), generator=generator).tolist()
            cursor = 0
        ids = permutation[cursor : cursor + args.frame_batch]
        cursor += args.frame_batch
        batch = mats_t[torch.as_tensor(ids, device=dev, dtype=torch.long)]

        optimizer.zero_grad(set_to_none=True)
        weights, active_weights = _compose(base_t, active_idx, logits)
        seed = joint_mechanical_loss_v1(
            rest_t, seed_faces_t, weights, batch, config=cfg
        )
        closure = joint_mechanical_loss_v1(
            rest_t, closure_faces_t, weights, batch, config=cfg
        )
        trust = (active_weights - base_active).abs().sum(dim=-1).mean()
        total = seed["loss"] + 0.25 * closure["loss"] + float(args.trust_weight) * trust
        if not bool(torch.isfinite(total)):
            raise RuntimeError("C41_NONFINITE_LOSS")
        total.backward()
        torch.nn.utils.clip_grad_norm_([logits], max_norm=10.0)
        optimizer.step()

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            current, _ = _compose(base_t, active_idx, logits)
            ev = _surrogate_eval(
                rest_t,
                closure_faces_t,
                seed_faces_t,
                current,
                mats_t,
                cfg,
                args.frame_batch,
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
            print("C41_STEP=" + json.dumps(row, sort_keys=True), flush=True)

    if best is None:
        raise RuntimeError("C41_NO_BEST_STATE")

    with torch.no_grad():
        logits.copy_(best[1].to(dev))
        local_final, _ = _compose(base_t, active_idx, logits)

    full = base_weights.copy()
    active_global = np.asarray(local["active_global"], np.int64)
    full[active_global] = (
        local_final[active_idx].detach().cpu().numpy().astype(np.float64)
    )
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

    row_delta = np.abs(full - base_weights).sum(axis=1)
    active_set = np.zeros(len(full), bool)
    active_set[active_global] = True
    passed = (
        bool(final_actual["passed"])
        and bool(micro["g3_passed"])
        and bool(micro["g3b_passed"])
    )
    report = {
        "schema": "RealSaS.MIRAActualMotionSkinOnlyCeiling.v1",
        "status": (
            "PASS_SKIN_ONLY_ACTUAL_MOTION_CEILING__MINIMAL_MIRA_FIT_AUTHORIZED"
            if passed
            else "FAIL_SKIN_ONLY_ACTUAL_MOTION_CEILING__DO_NOT_FIT_MIRA"
        ),
        "run_id": args.run_id,
        "carrier_evidence_hash": carrier.carrier_evidence_hash,
        "carrier_topology_hash": carrier.topology_hash,
        "atlas_rig_binding_hash": skeleton.skeleton_lineage_hash,
        "mira_checkpoint_sha256": mira_result["model_sha256"],
        "query_hash": query.query_hash,
        "training_used": False,
        "teacher_weights_used": False,
        "direct_weight_optimization_used": True,
        "product_authority_minted": False,
        "actual_motion_frame_count": int(len(frame_rows)),
        "base_actual_motion": {
            k: v
            for k, v in base_actual.items()
            if k not in {"union_bad", "per_frame"}
        },
        "active_region": {
            "seed_bad_face_count": int(local["seed_face_count"]),
            "closure_face_count": int(local["closure_face_count"]),
            "active_vertex_count": int(len(active_global)),
            "active_vertex_fraction": float(len(active_global) / len(full)),
            "one_ring_closure": True,
        },
        "optimization": {
            "steps": int(args.steps),
            "lr": float(args.lr),
            "frame_batch": int(args.frame_batch),
            "trust_weight": float(args.trust_weight),
            "initial_surrogate": initial_surrogate,
            "best_surrogate": best[2],
            "history": history,
        },
        "correction": {
            "active_mean_row_l1": float(row_delta[active_set].mean()),
            "active_p95_row_l1": float(np.quantile(row_delta[active_set], 0.95)),
            "active_max_row_l1": float(row_delta[active_set].max()),
            "frozen_outside_max_row_l1": float(
                row_delta[~active_set].max(initial=0.0)
            ),
        },
        "final_actual_motion": {
            k: v
            for k, v in final_actual.items()
            if k not in {"union_bad", "per_frame"}
        },
        "microstress_nonregression": micro,
        "retarget_mapping": mappings,
        "backbone_query_chunk_telemetry": telemetry,
        "decision": (
            "AUTHORIZE_C4_2_MINIMAL_CARRIER_MOTION_MIRA_ADAPTER"
            if passed
            else "REATTRIBUTE_TO_RIG_AND_OR_CARRIER__MIRA_FIT_BLOCKED"
        ),
        "claim_boundary": [
            "The carrier and A2-verified ATLAS rig are frozen.",
            "No teacher skin weights enter initialization or objective.",
            "Only rows in the exact-motion offender one-ring closure are optimized.",
            "Rows outside the closure are exact-frozen.",
            "Hard exact-motion replay plus microstress G3/G3B is the verdict.",
            "This is an optimization ceiling, not a learned-model result.",
        ],
    }

    write(args.out_dir / "OPTIMIZED_CARRIER_SKIN.json", skin.to_dict())
    write(args.out_dir / "MICRO_G3.json", g3.to_dict())
    write(args.out_dir / "MICRO_G3B.json", g3b)
    write(
        args.out_dir / "BASE_ACTUAL_MOTION.json",
        {
            "summary": report["base_actual_motion"],
            "frames": base_actual["per_frame"],
        },
    )
    write(
        args.out_dir / "FINAL_ACTUAL_MOTION.json",
        {
            "summary": report["final_actual_motion"],
            "frames": final_actual["per_frame"],
        },
    )
    np.savez_compressed(
        args.out_dir / "OPTIMIZED_ACTUAL_MOTION_WEIGHTS.npz",
        base_weights=base_weights,
        optimized_weights=full,
        active_mask=active_set.astype(np.uint8),
        vertex_ids=np.asarray(query.carrier_vertex_ids),
        joint_ids=np.asarray(query.joint_ids),
    )
    write(args.out_dir / "REPORT.json", report)
    print("MIRA_C41_RESULT=" + json.dumps(report, sort_keys=True), flush=True)


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
        write(
            args.out_dir / "ERROR.json",
            {"type": type(exc).__name__, "message": str(exc)},
        )
        raise
