from __future__ import annotations

"""Sealed fail-closed Mage FIT1 runner for audited Arachne A1 v4.

This is the only V4 execution candidate intended for the A100 Run-All notebook.
The superseded V3 predictor/runtime is not imported. Product-time neural input is
restricted to rich RiggingSurfaceIR + Compiler-qualified skeleton + exact camera
binding. Teacher skin appears only in objective/evaluation targets.
"""

import argparse
import inspect
import json
import math
import random
from pathlib import Path

import numpy as np
import torch

from compiler.realsas_compiler_core.skin import qualify_skin
from experiments.arachne_a1_v4_fit1.runtime_support_v4 import (
    EXPECTED_A0_CONFIG_HASH,
    EXPECTED_A0_MODEL_SHA,
    EXPECTED_BANK_SHA,
    EXPECTED_SKELETON_LINEAGE,
    EXPECTED_SURFACE_LINEAGE,
    atomic_torch_save,
    build_surface,
    conditioning_to_torch,
    decode_logits,
    load_bank,
    load_cameras,
    load_frozen_codec,
    load_skeleton,
    metrics,
    sha_file,
    write_json,
)
from models.arachne.v3.conditioning_v3 import ArachneRichConditioningAdapterV3
from models.arachne.v4.arachne_candidate_v4 import ArachneA1ConfigV4, ArachneA1V4
from models.arachne.v4.articulated_probe_v1 import build_articulated_probe_transforms
from models.arachne.v4.loss_v4 import (
    ArachneA1LossConfigV4,
    arachne_a1_behavior_loss_v4,
    articulated_deformation_ratio_loss,
    normalize_joint_fields,
)
from models.arachne.v4.proposal_v4 import make_skin_proposal_v4


SCHEMA = "RealSaS.ArachneA1V7NativeFIT1.v2"
SEED = 20260911

EXPECTED_A1_CONFIG_HASH = "738528eb7d94d8c8c69c9caf5838e4cf01341ddad287d5e29a78b735a53d3f96"
EXPECTED_A1_PARAMETER_COUNT = 138_053_153
EXPECTED_LOSS_CONFIG_HASH = "1e609f04bafcfaa592c732503e4d3054a614cd6e72dda5c7fad911f34e6efae6"

MAX_STEPS = 12_288
MIN_CLOSURE_STEP = 2_048
CHECK_EVERY = 256
REQUIRED_STABLE = 3
WARMUP_STEPS = 512
LR = 5.0e-5
LR_FLOOR = 0.1
WEIGHT_DECAY = 1.0e-4
GRAD_CLIP = 1.0
TRAIN_GSA_ROWS = 192
TRAIN_DENSE_ROWS = 192
FIT1_P95_MAX = 0.05
FIT1_LEGACY_DEFORM_MAX = 0.05
FIT1_ARTICULATED_DEFORM_MAX = 0.05
COMPILER_TOTAL_CORR_MAX = 1.0e-4
MIN_ARTICULATED_TEACHER_MOTION_RMS = 1.0e-6


def lr_factor(step: int) -> float:
    s = max(1, int(step))
    if s <= WARMUP_STEPS:
        return float(s) / float(WARMUP_STEPS)
    q = (s - WARMUP_STEPS) / float(MAX_STEPS - WARMUP_STEPS)
    q = min(max(q, 0.0), 1.0)
    return LR_FLOOR + (1.0 - LR_FLOOR) * 0.5 * (1.0 + math.cos(math.pi * q))


def _expect(obj: dict, path: tuple[str, ...], expected) -> None:
    cur = obj
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            raise RuntimeError("A1_V4_PREREG_FIELD_MISSING::" + ".".join(path))
        cur = cur[key]
    if cur != expected:
        raise RuntimeError(
            "A1_V4_PREREG_CONTRACT_DRIFT::"
            + ".".join(path)
            + f"::expected={expected!r}::got={cur!r}"
        )


def validate_prereg(prereg: dict) -> None:
    _expect(prereg, ("schema",), SCHEMA + ".Preregistration.v1")
    _expect(prereg, ("status",), "AUTHORIZED")
    _expect(prereg, ("architecture", "architecture_id"), "RealSaS.Arachne.A1.RichQualifiedBidirectional.v4")
    _expect(prereg, ("architecture", "config_hash"), EXPECTED_A1_CONFIG_HASH)
    _expect(prereg, ("architecture", "parameter_count"), EXPECTED_A1_PARAMETER_COUNT)
    _expect(prereg, ("architecture", "field_tokens"), 4)
    _expect(prereg, ("architecture", "latent_channels"), 512)
    _expect(prereg, ("frozen_a0", "model_sha256"), EXPECTED_A0_MODEL_SHA)
    _expect(prereg, ("frozen_a0", "supervision_bank_sha256"), EXPECTED_BANK_SHA)
    _expect(prereg, ("frozen_a0", "config_hash"), EXPECTED_A0_CONFIG_HASH)
    _expect(prereg, ("objective", "loss_config_hash"), EXPECTED_LOSS_CONFIG_HASH)
    _expect(prereg, ("objective", "ordered_latent_alignment_weight"), 0.0)
    _expect(prereg, ("objective", "articulated_deformation_weight"), 0.25)
    _expect(prereg, ("teacher_boundary",), "OBJECTIVE_AND_EVALUATION_ONLY__NEVER_PREDICTOR_INPUT")
    _expect(prereg, ("training", "max_steps"), MAX_STEPS)
    _expect(prereg, ("training", "min_closure_step"), MIN_CLOSURE_STEP)
    _expect(prereg, ("training", "check_every_steps"), CHECK_EVERY)
    _expect(prereg, ("fit1_gate", "gsa950_row_l1_p95_max"), FIT1_P95_MAX)
    _expect(prereg, ("fit1_gate", "gsa950_legacy_deformation_ratio_max"), FIT1_LEGACY_DEFORM_MAX)
    _expect(prereg, ("fit1_gate", "gsa950_articulated_deformation_ratio_max"), FIT1_ARTICULATED_DEFORM_MAX)
    _expect(prereg, ("fit1_gate", "stable_observations"), REQUIRED_STABLE)


def camera_yaw_fourier(cameras, device: torch.device) -> torch.Tensor:
    rows = []
    for i, camera in enumerate(cameras):
        if int(camera["view_index"]) != i:
            raise RuntimeError("CAMERA_ORDER_DRIFT")
        yaw = math.radians(float(camera["yaw_deg"]))
        rows.append([math.sin(yaw), math.cos(yaw), math.sin(2.0 * yaw), math.cos(2.0 * yaw)])
    return torch.tensor(rows, device=device, dtype=torch.float32)[None]


def joint_world_and_parent(skeleton, conditioning, device: torch.device):
    by_id = {str(j.canonical_joint_id): j for j in skeleton.joints}
    ids = tuple(conditioning.joint_ids[0])
    if set(ids) != set(by_id):
        raise RuntimeError("A1_V4_JOINT_WORLD_ID_DRIFT")
    world = np.asarray([[by_id[jid].position for jid in ids]], np.float32)
    return (
        torch.as_tensor(world, device=device, dtype=torch.float32),
        torch.as_tensor(conditioning.parent_indices, device=device, dtype=torch.long),
    )


def preflight_equivariance_v4(model, ci, *, tol: float = 2.0e-5) -> dict:
    model.eval()
    with torch.no_grad():
        base = model(**ci).field_tokens.float()

        n = ci["surface_mask"].shape[1]
        p = torch.arange(n - 1, -1, -1, device=base.device)
        inv = torch.empty_like(p)
        inv[p] = torch.arange(n, device=base.device)
        cs = dict(ci)
        for name in (
            "surface_positions_normalized", "surface_normals", "surface_normal_valid",
            "surface_support_views", "surface_raster_xy", "surface_raster_valid",
            "surface_observed", "surface_completed", "surface_mask",
        ):
            cs[name] = ci[name][:, p]
        cs["edge_index"] = inv[ci["edge_index"]]
        cs["support_anchor_matrix"] = ci["support_anchor_matrix"][:, :, p]
        cs["pair_geometry"] = ci["pair_geometry"][:, p]
        cs["pair_mask"] = ci["pair_mask"][:, p]
        surface_delta = float((base - model(**cs).field_tokens.float()).abs().max().cpu())

        j = ci["joint_mask"].shape[1]
        q = torch.arange(j - 1, -1, -1, device=base.device)
        qinv = torch.empty_like(q)
        qinv[q] = torch.arange(j, device=base.device)
        cj = dict(ci)
        for name in (
            "joint_positions_normalized", "joint_mask", "root_mask",
            "deform_root_mask", "joint_depth_normalized",
        ):
            cj[name] = ci[name][:, q]
        old_parent = ci["parent_indices"][:, q]
        cj["parent_indices"] = torch.where(old_parent >= 0, qinv[old_parent.clamp_min(0)], old_parent)
        cj["support_anchor_matrix"] = ci["support_anchor_matrix"][:, q]
        cj["pair_geometry"] = ci["pair_geometry"][:, :, q]
        cj["pair_mask"] = ci["pair_mask"][:, :, q]
        joint_delta = float((base[:, q] - model(**cj).field_tokens.float()).abs().max().cpu())

        vp = torch.tensor([3, 0, 7, 1, 6, 2, 5, 4], device=base.device)
        cv = dict(ci)
        for name in ("surface_support_views", "surface_raster_xy", "surface_raster_valid"):
            cv[name] = ci[name][:, :, vp]
        cv["view_yaw_fourier"] = ci["view_yaw_fourier"][:, vp]
        view_delta = float((base - model(**cv).field_tokens.float()).abs().max().cpu())

    if max(surface_delta, joint_delta, view_delta) > tol:
        raise RuntimeError(
            f"A1_V4_EQUIVARIANCE_FAIL::{surface_delta}::{joint_delta}::{view_delta}"
        )
    return {
        "tolerance": float(tol),
        "surface_max_abs": surface_delta,
        "joint_max_abs": joint_delta,
        "view_binding_max_abs": view_delta,
        "support_anchor_remap_covered": True,
    }


def run(args) -> dict:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_REQUIRED")
    device = torch.device("cuda")
    gpu = torch.cuda.get_device_name(0)
    if "A100" not in gpu.upper() or not torch.cuda.is_bf16_supported():
        raise RuntimeError(f"A100_BF16_REQUIRED::{gpu}")
    if torch.cuda.mem_get_info()[0] < 18 * 1024**3:
        raise RuntimeError("A1_V4_REQUIRES_AT_LEAST_18_GIB_FREE_VRAM")

    prereg_path = Path(args.prereg)
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    validate_prereg(prereg)
    prereg_sha = sha_file(prereg_path)

    data_dir = Path(args.data_dir)
    cameras = load_cameras(data_dir)
    surface = build_surface(Path(args.zero_surface), cameras)
    skeleton = load_skeleton(Path(args.skeleton))
    bank = load_bank(Path(args.bank))
    codec, _ = load_frozen_codec(Path(args.a0_model), device)

    conditioning = ArachneRichConditioningAdapterV3(require_scene_first=True)([surface], [skeleton])
    if conditioning.source_surface_hashes != (EXPECTED_SURFACE_LINEAGE,):
        raise RuntimeError("A1_V4_SURFACE_LINEAGE_DRIFT")
    if conditioning.source_skeleton_hashes != (EXPECTED_SKELETON_LINEAGE,):
        raise RuntimeError("A1_V4_SKELETON_LINEAGE_DRIFT")
    if int(conditioning.surface_mask[0].sum()) != 950:
        raise RuntimeError("A1_V4_SURFACE_CARDINALITY_DRIFT")
    if int(conditioning.edge_mask[0].sum()) != 2813:
        raise RuntimeError("A1_V4_EDGE_CARDINALITY_DRIFT")
    if int(conditioning.joint_mask[0].sum()) != 22:
        raise RuntimeError("A1_V4_JOINT_CARDINALITY_DRIFT")
    if conditioning.relation_kind_vocab != ("SIGNED_ZERO_SURFACE_TOPOLOGY_NEIGHBOR",):
        raise RuntimeError("A1_V4_RELATION_KIND_CONTRACT_DRIFT")

    geom_delta = float(
        np.max(np.abs(conditioning.geometry7[0] - bank["gsa_geometry7"].astype(np.float32)))
    )
    if geom_delta > 5.0e-5:
        raise RuntimeError(f"A1_V4_CODEC_FRAME_PARITY_FAIL::{geom_delta}")
    if tuple(map(str, bank["surface_ids"])) != tuple(conditioning.surface_ids[0]):
        raise RuntimeError("A1_V4_SURFACE_ID_ORDER_DRIFT")
    if tuple(map(str, bank["joint_ids"])) != tuple(conditioning.joint_ids[0]):
        raise RuntimeError("A1_V4_JOINT_ID_ORDER_DRIFT")
    if not np.array_equal(
        bank["condition_query_indices"].astype(np.int64), conditioning.condition_query_indices[0]
    ):
        raise RuntimeError("A1_V4_CONDITION_QUERY_DRIFT")

    cfg = ArachneA1ConfigV4()
    lcfg = ArachneA1LossConfigV4()
    if cfg.config_hash != EXPECTED_A1_CONFIG_HASH:
        raise RuntimeError("A1_V4_CONFIG_HASH_DRIFT")
    if lcfg.config_hash != EXPECTED_LOSS_CONFIG_HASH:
        raise RuntimeError("A1_V4_LOSS_HASH_DRIFT")
    model = ArachneA1V4(cfg).to(device=device, dtype=torch.float32)
    if model.parameter_count != EXPECTED_A1_PARAMETER_COUNT:
        raise RuntimeError(f"A1_V4_PARAMETER_COUNT_DRIFT::{model.parameter_count}")

    signature = set(inspect.signature(model.forward).parameters)
    forbidden = {
        name for name in signature
        if any(term in name.lower() for term in (
            "teacher", "truth", "weight", "source_id", "character", "condition_token"
        ))
    }
    if forbidden:
        raise RuntimeError(f"PREDICTOR_INPUT_FIREWALL_FAIL::{sorted(forbidden)}")

    ci = conditioning_to_torch(conditioning, device)
    ci["view_yaw_fourier"] = camera_yaw_fourier(cameras, device)
    equivariance = preflight_equivariance_v4(model, ci)

    joint_world, parent_indices = joint_world_and_parent(skeleton, conditioning, device)
    articulated_transforms = build_articulated_probe_transforms(
        joint_world, parent_indices, ci["joint_mask"]
    )
    q = torch.arange(joint_world.shape[1] - 1, -1, -1, device=device)
    qinv = torch.empty_like(q)
    qinv[q] = torch.arange(len(q), device=device)
    old_parent = parent_indices[:, q]
    parent_perm = torch.where(old_parent >= 0, qinv[old_parent.clamp_min(0)], old_parent)
    articulated_perm = build_articulated_probe_transforms(
        joint_world[:, q], parent_perm, ci["joint_mask"][:, q]
    )
    probe_delta = float((articulated_transforms[:, :, q] - articulated_perm).abs().max().cpu())
    if probe_delta > 1.0e-6:
        raise RuntimeError(f"ARTICULATED_PROBE_JOINT_EQUIVARIANCE_FAIL::{probe_delta}")

    gsa_geom_t = torch.as_tensor(
        bank["gsa_geometry7"][None], device=device, dtype=torch.float32
    )
    qidx = torch.as_tensor(
        bank["condition_query_indices"][None], device=device, dtype=torch.long
    )
    with torch.no_grad(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16, enabled=True
    ):
        condition_tokens = codec.encode_condition(gsa_geom_t, qidx).detach()
        prepared_condition = codec.decoder.prepare_condition(condition_tokens).detach()
    prepared_condition = prepared_condition.clone()

    gsa_geom = np.asarray(bank["gsa_geometry7"], np.float32)
    gsa_truth = np.asarray(bank["gsa_teacher_weights"], np.float32)
    gsa_world = np.asarray(bank["gsa_world"], np.float32)
    gsa_mask = np.asarray(bank["gsa_supervision_mask"], bool)
    dense_geom = np.asarray(bank["dense8k_geometry7"], np.float32)
    dense_truth = np.asarray(bank["dense8k_teacher_weights"], np.float32)
    dense_world = np.asarray(bank["dense8k_world"], np.float32)
    hold_geom = np.asarray(bank["holdout_geometry7"], np.float32)
    hold_truth = np.asarray(bank["holdout_teacher_weights"], np.float32)
    hold_world = np.asarray(bank["holdout_world"], np.float32)
    gsa_pool = np.where(gsa_mask)[0].astype(np.int64)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_path = out_dir / "ARACHNE_A1_V4_FIT1_TRACE.json"
    model_path = out_dir / "ARACHNE_A1_V4_FIT1_MODEL_ONLY_FP32.pt"
    result_path = out_dir / "ARACHNE_A1_V4_FIT1_RESULT.json"
    if result_path.exists():
        raise RuntimeError(f"A1_V4_RESULT_ALREADY_EXISTS::{result_path}")

    trace = []
    best_score = None
    stable = 0
    closure_step = None

    def predict_all(geometry: np.ndarray) -> np.ndarray:
        model.eval()
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=True
        ):
            field_tokens = model(**ci).field_tokens
            logits = decode_logits(codec, prepared_condition, field_tokens, geometry)
            pred = normalize_joint_fields(
                torch.sigmoid(logits), ci["joint_mask"]
            ).squeeze(0)
        return pred.float().cpu().numpy()

    def articulated_metric(pred, truth, world, mask) -> dict:
        wp = torch.as_tensor(pred[None], device=device, dtype=torch.float32)
        wt = torch.as_tensor(truth[None], device=device, dtype=torch.float32)
        rest = torch.as_tensor(world[None], device=device, dtype=torch.float32)
        rm = torch.as_tensor(mask[None], device=device, dtype=torch.bool)
        with torch.no_grad():
            ratio, motion, err = articulated_deformation_ratio_loss(
                wp, wt, rest, articulated_transforms, rm
            )
        return {
            "ratio": float(ratio.cpu()),
            "teacher_motion_rms": float(motion.cpu()),
            "error_rms": float(err.cpu()),
        }

    def evaluate(step: int) -> dict:
        gsa_pred = predict_all(gsa_geom)
        hold_pred = predict_all(hold_geom)
        gsa_metrics = metrics(gsa_pred, gsa_truth, gsa_world, device, gsa_mask)
        hold_metrics = metrics(hold_pred, hold_truth, hold_world, device)
        gsa_art = articulated_metric(gsa_pred, gsa_truth, gsa_world, gsa_mask)
        hold_art = articulated_metric(
            hold_pred, hold_truth, hold_world, np.ones(len(hold_pred), bool)
        )

        proposal = make_skin_proposal_v4(gsa_pred, conditioning, cfg.config_hash)
        qskin = qualify_skin(
            surface,
            skeleton,
            proposal,
            max_simplex_repair_l1=1.0e-5,
            max_total_correction_l1=COMPILER_TOTAL_CORR_MAX,
        )
        report = qskin.qualification_report
        compiler = {
            "status": "PASS",
            "row_count": len(qskin.rows),
            "skin_lineage_hash": qskin.skin_lineage_hash,
            "qualification_report": report,
        }
        passed = (
            gsa_metrics["row_l1_p95"] <= FIT1_P95_MAX
            and gsa_metrics["deformation_error_ratio"] <= FIT1_LEGACY_DEFORM_MAX
            and gsa_art["ratio"] <= FIT1_ARTICULATED_DEFORM_MAX
            and len(qskin.rows) == 950
            and float(report["total_correction_l1"]) <= COMPILER_TOTAL_CORR_MAX
        )
        return {
            "step": int(step),
            "gsa950": gsa_metrics,
            "gsa950_articulated": gsa_art,
            "disjoint_holdout": hold_metrics,
            "holdout_articulated": hold_art,
            "compiler_skin": compiler,
            "fit1_gate": bool(passed),
        }

    # All source/data/authority/equivariance/probe gates occur before optimizer creation.
    row0 = evaluate(0)
    if row0["gsa950_articulated"]["teacher_motion_rms"] <= MIN_ARTICULATED_TEACHER_MOTION_RMS:
        raise RuntimeError(
            "A1_V4_ARTICULATED_PROBE_DEGENERATE::"
            + str(row0["gsa950_articulated"]["teacher_motion_rms"])
        )
    row0["stable_streak"] = 0
    trace.append(row0)
    write_json(
        trace_path,
        {"schema": SCHEMA + ".Trace.v1", "prereg_sha256": prereg_sha, "rows": trace},
    )
    print("A1V4_CHECK=" + json.dumps(row0, sort_keys=True), flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_factor)

    for step in range(1, MAX_STEPS + 1):
        model.train()
        rng = np.random.default_rng(SEED + step * 1009)
        gi = rng.choice(gsa_pool, TRAIN_GSA_ROWS, replace=False)
        di = rng.choice(len(dense_geom), TRAIN_DENSE_ROWS, replace=False)
        query_geom = np.concatenate([gsa_geom[gi], dense_geom[di]], axis=0)
        truth_np = np.concatenate([gsa_truth[gi], dense_truth[di]], axis=0)
        query_world = np.concatenate([gsa_world[gi], dense_world[di]], axis=0)
        order = rng.permutation(len(query_geom))
        query_geom = query_geom[order]
        truth_np = truth_np[order]
        query_world = query_world[order]

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            field_tokens = model(**ci).field_tokens
            logits = decode_logits(
                codec,
                prepared_condition,
                field_tokens,
                query_geom,
                chunk=len(query_geom),
            )
        truth = torch.as_tensor(truth_np[None], device=device, dtype=torch.float32)
        rest = torch.as_tensor(query_world[None], device=device, dtype=torch.float32)
        row_mask = torch.ones((1, len(query_geom)), device=device, dtype=torch.bool)
        losses = arachne_a1_behavior_loss_v4(
            logits,
            truth,
            row_mask,
            joint_mask=ci["joint_mask"],
            rest_world=rest,
            articulated_transforms=articulated_transforms,
            config=lcfg,
        )
        losses["total"].backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP))
        if not math.isfinite(grad_norm):
            raise FloatingPointError("NONFINITE_A1_V4_GRADIENT")
        optimizer.step()
        scheduler.step()

        if step % CHECK_EVERY:
            continue

        row = evaluate(step)
        row["losses"] = {k: float(v.detach().cpu()) for k, v in losses.items()}
        row["grad_norm"] = grad_norm
        row["lr"] = float(optimizer.param_groups[0]["lr"])
        stable = stable + 1 if row["fit1_gate"] and step >= MIN_CLOSURE_STEP else 0
        row["stable_streak"] = stable
        trace.append(row)
        print("A1V4_CHECK=" + json.dumps(row, sort_keys=True), flush=True)

        p95 = float(row["gsa950"]["row_l1_p95"])
        legacy_def = float(row["gsa950"]["deformation_error_ratio"])
        art_def = float(row["gsa950_articulated"]["ratio"])
        cvar = float(row["gsa950"]["cvar10"])
        score = (max(p95, legacy_def, art_def), p95, art_def, legacy_def, cvar)
        if best_score is None or score < best_score:
            payload = {
                "schema": SCHEMA + ".ModelOnly.v1",
                "prereg_sha256": prereg_sha,
                "a0_model_sha256": EXPECTED_A0_MODEL_SHA,
                "a1_supervision_bank_sha256": EXPECTED_BANK_SHA,
                "a1_config_hash": cfg.config_hash,
                "loss_config_hash": lcfg.config_hash,
                "parameter_count": model.parameter_count,
                "step": int(step),
                "score": list(score),
                "model": model.state_dict(),
                "product_evaluation_performed": False,
                "unseen_generalization_claimed": False,
            }
            atomic_torch_save(payload, model_path)
            best_score = score

        write_json(
            trace_path,
            {"schema": SCHEMA + ".Trace.v1", "prereg_sha256": prereg_sha, "rows": trace},
        )
        if stable >= REQUIRED_STABLE:
            closure_step = step
            break

    if not model_path.exists():
        raise RuntimeError("A1_V4_MODEL_CHECKPOINT_MISSING")
    try:
        payload = torch.load(model_path, map_location="cpu", weights_only=False, mmap=True)
    except Exception:
        payload = torch.load(model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"], strict=True)
    final = evaluate(int(payload["step"]))
    status = "PASS" if closure_step is not None and final["fit1_gate"] else "FAIL"

    result = {
        "schema": SCHEMA + ".Result.v1",
        "status": status,
        "prereg_sha256": prereg_sha,
        "seed": SEED,
        "a0": {
            "model_sha256": EXPECTED_A0_MODEL_SHA,
            "config_hash": EXPECTED_A0_CONFIG_HASH,
            "supervision_bank_sha256": EXPECTED_BANK_SHA,
            "field_tokens": 4,
            "latent_channels": 512,
            "ordered_latent_alignment_used": False,
        },
        "a1": {
            "architecture_id": cfg.architecture_id,
            "config_hash": cfg.config_hash,
            "parameter_count": model.parameter_count,
            "loss_config_hash": lcfg.config_hash,
            "model_sha256": sha_file(model_path),
            "best_step": int(payload["step"]),
            "closure_step": closure_step,
            "equivariance_preflight": equivariance,
            "articulated_probe_joint_permutation_delta": probe_delta,
        },
        "fit1_gate": {
            "gsa950_row_l1_p95_max": FIT1_P95_MAX,
            "gsa950_legacy_deformation_ratio_max": FIT1_LEGACY_DEFORM_MAX,
            "gsa950_articulated_deformation_ratio_max": FIT1_ARTICULATED_DEFORM_MAX,
            "stable_observations": REQUIRED_STABLE,
            "minimum_closure_step": MIN_CLOSURE_STEP,
            "compiler_row_count_required": 950,
            "compiler_total_correction_l1_max": COMPILER_TOTAL_CORR_MAX,
        },
        "final_best_checkpoint_evaluation": final,
        "teacher_boundary": "OBJECTIVE_AND_EVALUATION_ONLY__NEVER_PREDICTOR_INPUT",
        "holdout_role": "DIAGNOSTIC_ONLY__SAME_CHARACTER_DISJOINT_SURFACE__NOT_UNSEEN",
        "product_evaluation_performed": False,
        "product_pass_claimed": False,
        "unseen_generalization_claimed": False,
        "trace_path": str(trace_path),
        "model_path": str(model_path),
    }
    write_json(result_path, result)
    print("A1V4_RESULT=" + json.dumps(result, sort_keys=True), flush=True)
    if status != "PASS":
        raise AssertionError("A1_V4_FIT1_DID_NOT_CLOSE")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--zero-surface", required=True)
    parser.add_argument("--skeleton", required=True)
    parser.add_argument("--a0-model", required=True)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--prereg", required=True)
    parser.add_argument("--output-dir", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
