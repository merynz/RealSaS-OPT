from __future__ import annotations

"""Fresh Geppetto refit on the corrected Mage FIT2 pipeline substrate.

This runner intentionally bypasses the historical FIT1 1024/950-node substrate authority.
It reuses only the frozen target/loss/evaluation apparatus. The active authority is the
corrected H1 observable product surface -> exact-observation GSA8192 lineage.

Historical checkpoints are never loaded. Training starts from a fresh model init.
"""

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
import math
from pathlib import Path
import subprocess

import numpy as np
import torch
from PIL import Image, ImageDraw

from experiments.geppetto_reference_strength_fullstack_v1.geppetto_reference_strength_loss_v1 import (
    GeppettoReferenceStrengthLossConfigV1,
    reference_strength_loss_v1,
)
from experiments.geppetto_reference_strength_fullstack_v1.run_geppetto_reference_strength_fit1_v1 import (
    CHECK_EVERY,
    DIFFUSION_SEEDS,
    GRAD_CLIP,
    LR,
    MAX_STEPS,
    RESOURCE_STEP_LIMIT,
    SEED,
    TARGET_HASH,
    TERMINAL_CHECKS,
    WEIGHT_DECAY,
    _build_target,
    _check,
)
from experiments.mage_full_subject_reclosure_v1.run_frozen_geppetto_replay_v2 import (
    ACTIVE_GSA_EDGE_COUNT,
    ACTIVE_GSA_LINEAGE,
    ACTIVE_GSA_NODE_COUNT,
    ACTIVE_ZERO_SURFACE_SHA256,
    CAMERA_SHA256,
    NORMALIZED_CORPUS_SHA256,
    OBSERVATION_SHA256,
    _build_active_surface,
    _load_cameras,
    _load_observation_masks,
    _require_sha,
)
from models.geppetto.reference_strength_v1.geppetto_reference_strength_no_learned_slot_v1 import (
    GeppettoReferenceStrengthNoLearnedSlotV1,
    assert_no_learned_view_slot_identity_v1,
)

SCHEMA = "RealSaS.MageFIT2.GeppettoCorrectedSubstrateRefit.v1"
AUTHORITY_ID = "MAGE_FIT2_PIPELINE_REFIT_AUTHORITY_V1"
H1_SOURCE_RUN_ID = "20260912T074348Z"
H1_CHECKPOINT_SHA256 = "76fc68a8c6f2bed80ae8a678649006c875bc96b586065da8914e7f61528c8ce5"
EXPECTED_TENSORIZATION_HASH = "fe351362e195164805cebb0f63b74d1861ef123458b20ac56607016e00a6c67e"
EXPECTED_OBSERVED_NODES = 7391
EXPECTED_COMPLETED_NODES = 780
EXPECTED_OBSERVATION_SUPPORT_COUNTS = (2537, 2765, 2194, 3021, 2772, 2871, 2183, 2803)
HISTORICAL_GEPPETTO_CHECKPOINT_SHA256 = "b75f991564b64cfcec9b50b006544380ee482362a8439775bb505002349cbc30"


def _sha(path: Path) -> str:
    h = sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _repo_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "UNKNOWN"


def _load_inputs(args):
    normalized = Path(args.normalized_corpus)
    _require_sha(normalized, NORMALIZED_CORPUS_SHA256, "NORMALIZED_CORPUS")
    camera_paths = tuple(Path(x) for x in args.cameras)
    observation_paths = tuple(Path(x) for x in args.observations)
    for view, path in enumerate(camera_paths):
        _require_sha(path, CAMERA_SHA256[view], f"CAMERA_V{view}")
    cameras = _load_cameras(list(camera_paths))
    observation_masks = _load_observation_masks(observation_paths)

    class SurfaceArgs:
        zero_surface = args.zero_surface
        iris_run_id = H1_SOURCE_RUN_ID
        iris_checkpoint_sha256 = H1_CHECKPOINT_SHA256

    surface, tensor = _build_active_surface(SurfaceArgs, cameras, observation_masks)
    target, prepared = _build_target(normalized, tensor)

    if surface.geometry_lineage_hash != ACTIVE_GSA_LINEAGE:
        raise RuntimeError("FIT2_ACTIVE_GSA_LINEAGE_DRIFT")
    if tensor.node_count != ACTIVE_GSA_NODE_COUNT or tensor.edge_count != ACTIVE_GSA_EDGE_COUNT:
        raise RuntimeError("FIT2_ACTIVE_GSA_CARDINALITY_DRIFT")
    if tensor.tensorization_hash != EXPECTED_TENSORIZATION_HASH:
        raise RuntimeError(
            f"FIT2_TENSORIZATION_HASH_DRIFT::{tensor.tensorization_hash}::{EXPECTED_TENSORIZATION_HASH}"
        )
    if int(tensor.observed.sum()) != EXPECTED_OBSERVED_NODES:
        raise RuntimeError("FIT2_OBSERVED_NODE_COUNT_DRIFT")
    if int(tensor.completed.sum()) != EXPECTED_COMPLETED_NODES:
        raise RuntimeError("FIT2_COMPLETED_NODE_COUNT_DRIFT")
    support_counts = tuple(int(x) for x in tensor.support.sum(axis=0).tolist())
    if support_counts != EXPECTED_OBSERVATION_SUPPORT_COUNTS:
        raise RuntimeError(f"FIT2_OBSERVATION_SUPPORT_COUNT_DRIFT::{support_counts}")
    return normalized, camera_paths, observation_paths, cameras, surface, tensor, target, prepared


def _preflight_payload(args, tensor, target) -> dict:
    model = GeppettoReferenceStrengthNoLearnedSlotV1()
    assert_no_learned_view_slot_identity_v1(model)
    return {
        "schema": SCHEMA,
        "status": "PASS__FIT2_CORRECTED_SUBSTRATE__FRESH_GEPPETTO_REFIT_READY",
        "authority_id": AUTHORITY_ID,
        "repo_head": _repo_head(),
        "training_mode": "FRESH_FROM_SCRATCH__NO_HISTORICAL_GEPPETTO_CHECKPOINT",
        "historical_checkpoint_sha256": HISTORICAL_GEPPETTO_CHECKPOINT_SHA256,
        "historical_checkpoint_loaded": False,
        "thresholds_changed": False,
        "zero_surface_sha256": ACTIVE_ZERO_SURFACE_SHA256,
        "h1_source_run_id": H1_SOURCE_RUN_ID,
        "h1_checkpoint_sha256": H1_CHECKPOINT_SHA256,
        "gsa_lineage_hash": ACTIVE_GSA_LINEAGE,
        "surface_node_count": tensor.node_count,
        "surface_edge_count": tensor.edge_count,
        "tensorization_hash": tensor.tensorization_hash,
        "observed_nodes": int(tensor.observed.sum()),
        "completed_nodes": int(tensor.completed.sum()),
        "observation_support_counts": [int(x) for x in tensor.support.sum(axis=0).tolist()],
        "target_content_sha256": TARGET_HASH,
        "target_count": int(target.count),
        "model_architecture_id": model.config.architecture_id,
        "model_config_hash": model.config.config_hash,
        "max_steps": MAX_STEPS,
        "check_every": CHECK_EVERY,
        "terminal_checks": TERMINAL_CHECKS,
        "diffusion_eval_seeds": list(DIFFUSION_SEEDS),
        "teacher_supervision_during_training": True,
        "teacher_feedback_during_free_eval": False,
        "arachne_refit_authorized": False,
    }


def _project_world(points: np.ndarray, camera: dict) -> np.ndarray:
    p = np.asarray(points, dtype=np.float64)
    origin = np.asarray(camera["origin"], dtype=np.float64)
    right = np.asarray(camera["right"], dtype=np.float64)
    up = np.asarray(camera["screen_up"], dtype=np.float64)
    half = float(camera["half_extent"])
    res = int(camera["resolution"])
    d = p - origin[None, :]
    gx = (d @ right) / half
    gy = -(d @ up) / half
    return np.stack(((gx + 1.0) * 0.5 * res - 0.5, (gy + 1.0) * 0.5 * res - 0.5), axis=1)


def _render_skeleton_8view(final_check: dict, cameras, observations, outdir: Path) -> list[str]:
    qualified = None
    for row in final_check.get("diffusion_seed_reports", []):
        if row.get("pass") and row.get("qualified"):
            qualified = row["qualified"]
            break
    if qualified is None:
        return []
    joints = qualified["joints"]
    positions = np.asarray([j["position"] for j in joints], dtype=np.float64)
    id_to_index = {str(j["canonical_joint_id"]): i for i, j in enumerate(joints)}
    files = []
    panels = []
    for v, (camera, obs_path) in enumerate(zip(cameras, observations)):
        with Image.open(obs_path) as im:
            canvas = im.convert("RGBA")
        draw = ImageDraw.Draw(canvas)
        xy = _project_world(positions, camera)
        for i, j in enumerate(joints):
            parent = j.get("parent_canonical_id")
            if parent is not None and str(parent) in id_to_index:
                k = id_to_index[str(parent)]
                draw.line((float(xy[i,0]), float(xy[i,1]), float(xy[k,0]), float(xy[k,1])), fill=(255,255,255,230), width=4)
        for x, y in xy:
            r = 5
            draw.ellipse((float(x-r), float(y-r), float(x+r), float(y+r)), fill=(255,80,30,255), outline=(0,0,0,255), width=1)
        draw.rectangle((0,0,220,34), fill=(0,0,0,180))
        draw.text((10,8), f"FIT2 Geppetto V{v} / {camera['yaw_deg']} deg", fill=(255,255,255,255))
        path = outdir / f"GEPPETTO_FIT2_V{v}_SKELETON_OVERLAY.png"
        canvas.save(path)
        files.append(path.name)
        panels.append(canvas.convert("RGB"))
    sheet = Image.new("RGB", (2048, 4096))
    for idx, panel in enumerate(panels):
        x = (idx % 2) * 1024
        y = (idx // 2) * 1024
        sheet.paste(panel, (x, y))
    sheet_path = outdir / "GEPPETTO_FIT2_8VIEW_SKELETON_CONTACT_SHEET.png"
    sheet.save(sheet_path)
    files.append(sheet_path.name)
    return files


def run(args) -> dict:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    normalized, camera_paths, observation_paths, cameras, surface, tensor, target, prepared = _load_inputs(args)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    preflight = _preflight_payload(args, tensor, target)
    preflight_path = outdir / "GEPPETTO_FIT2_CORRECTED_SUBSTRATE_PREFLIGHT.json"
    preflight_path.write_text(json.dumps(preflight, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(preflight["status"], flush=True)
    if args.preflight_only:
        return preflight

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_REQUIRED_FOR_GEPPETTO_FIT2_REFIT")
    device = torch.device("cuda")
    model = GeppettoReferenceStrengthNoLearnedSlotV1().to(device)
    assert_no_learned_view_slot_identity_v1(model)
    loss_cfg = GeppettoReferenceStrengthLossConfigV1()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    train_gen = torch.Generator(device=device)
    train_gen.manual_seed(SEED)
    teacher_pos = torch.as_tensor(prepared.positions_normalized, device=device, dtype=torch.float32)

    trace = []
    terminal_streak = 0
    closure_step = None
    for step in range(1, MAX_STEPS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        out = model.forward_surface(
            tensor,
            decode_steps=prepared.count,
            teacher_positions_normalized=teacher_pos,
            generator=train_gen,
        )
        losses = reference_strength_loss_v1(out, prepared, config=loss_cfg)
        losses["total"].backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP))
        if not math.isfinite(grad_norm):
            raise FloatingPointError("NON_FINITE_GRADIENT_NORM")
        optimizer.step()

        if step != 1 and step % CHECK_EVERY:
            continue
        check = _check(model, surface, tensor, target, step)
        check["losses"] = {k: float(v.detach().cpu()) for k, v in losses.items()}
        check["grad_norm"] = grad_norm
        terminal_streak = terminal_streak + 1 if check["pass"] else 0
        check["terminal_streak"] = terminal_streak
        trace.append(check)
        print("GEPPETTO_FIT2_CHECK=" + json.dumps(check, sort_keys=True), flush=True)
        if terminal_streak >= TERMINAL_CHECKS:
            closure_step = step
            break

    final_step = closure_step or MAX_STEPS
    final_check = _check(model, surface, tensor, target, final_step)
    status = "FIT2_GEPPETTO_TERMINAL_PASS" if closure_step is not None else "FIT2_GEPPETTO_NO_TERMINAL_CLOSURE"
    checkpoint_path = outdir / "GEPPETTO_FIT2_CORRECTED_SUBSTRATE_CHECKPOINT.pt"
    torch.save(
        {
            "schema": SCHEMA,
            "authority_id": AUTHORITY_ID,
            "model": model.state_dict(),
            "config": asdict(model.config),
            "config_hash": model.config.config_hash,
            "repo_head": _repo_head(),
            "target_hash": TARGET_HASH,
            "gsa_lineage_hash": ACTIVE_GSA_LINEAGE,
            "tensorization_hash": tensor.tensorization_hash,
            "step": final_step,
            "historical_checkpoint_loaded": False,
        },
        checkpoint_path,
    )
    checkpoint_sha = _sha(checkpoint_path)
    render_files = _render_skeleton_8view(final_check, cameras, observation_paths, outdir) if closure_step else []
    result = {
        "schema": SCHEMA,
        "status": status,
        "authority_id": AUTHORITY_ID,
        "repo_head": _repo_head(),
        "training_mode": "FRESH_FROM_SCRATCH__NO_HISTORICAL_GEPPETTO_CHECKPOINT",
        "historical_checkpoint_sha256": HISTORICAL_GEPPETTO_CHECKPOINT_SHA256,
        "historical_checkpoint_loaded": False,
        "seed": SEED,
        "architecture_id": model.config.architecture_id,
        "config_hash": model.config.config_hash,
        "loss_config": asdict(loss_cfg),
        "h1_source_run_id": H1_SOURCE_RUN_ID,
        "h1_checkpoint_sha256": H1_CHECKPOINT_SHA256,
        "zero_surface_sha256": ACTIVE_ZERO_SURFACE_SHA256,
        "normalized_corpus_sha256": NORMALIZED_CORPUS_SHA256,
        "target_content_sha256": TARGET_HASH,
        "gsa_lineage_hash": ACTIVE_GSA_LINEAGE,
        "surface_node_count": tensor.node_count,
        "surface_edge_count": tensor.edge_count,
        "tensorization_hash": tensor.tensorization_hash,
        "resource_step_limit_execution_only": RESOURCE_STEP_LIMIT,
        "thresholds_changed": False,
        "teacher_supervision_during_training": True,
        "teacher_feedback_during_free_eval": False,
        "closure_step": closure_step,
        "terminal_streak": terminal_streak,
        "required_terminal_checks": TERMINAL_CHECKS,
        "final_check": final_check,
        "trace": trace,
        "checkpoint_sha256": checkpoint_sha,
        "eight_view_skeleton_artifacts": render_files,
        "arachne_refit_authorized": bool(closure_step is not None and final_check.get("pass")),
        "generalization_claim": False,
        "product_pass_claimed": False,
        "promotion_authorized": False,
    }
    result_path = outdir / "GEPPETTO_FIT2_CORRECTED_SUBSTRATE_RESULT.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("GEPPETTO_FIT2_RESULT=" + json.dumps(result, sort_keys=True), flush=True)
    if status != "FIT2_GEPPETTO_TERMINAL_PASS":
        raise AssertionError(status)
    return result


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--zero-surface", required=True)
    p.add_argument("--normalized-corpus", required=True)
    p.add_argument("--cameras", nargs=8, required=True)
    p.add_argument("--observations", nargs=8, required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--preflight-only", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
