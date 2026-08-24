from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset_pv5_r256_onecell import PV5R256OneCellDataset
from model_pv5_r256 import IRISSinglePoseV2PV5R256
from pv5_depth_objective import p_only_objective, sample_p_field
from train_pv5_r256_onecell_v1 import configure_p_only

SEED = 20260825
PARENT_SELECTED_STEP = 2048
PARENT_SELECTED_P95 = 0.005682396539486942
PARENT_CHECKPOINT_SHA256 = "cad4421ae728848bbf9181c87e0d4912c38ec6f94d61c86cd114ad0a251c79ee"
P_THRESHOLD = 0.005
BASELINE_TOL = 1e-7
STEPS = 512
EVAL_STEPS = (0, 64, 128, 256, 512)
ARMS = (
    ("A_LR3E4_CONTROL", 3e-4),
    ("B_LR1E4", 1e-4),
    ("C_LR3E5", 3e-5),
)
BETAS = (0.9, 0.95)
WEIGHT_DECAY = 0.0


def sha256_file(path: Path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def atomic_json(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def seed_all(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def q(a, x):
    a = np.asarray(a, dtype=np.float64)
    return float(np.quantile(a, x)) if a.size else float("nan")


def pack(batch, device):
    return {
        "images": batch["images"].to(device, non_blocking=True),
        "yaw_deg": batch["yaw_deg"].to(device, non_blocking=True),
        "sheet_half_extent": batch["sheet_half_extent"].to(device, non_blocking=True),
        "geom_xy": batch["geom_xy"].to(device, non_blocking=True),
        "geom_p": batch["geom_p"].to(device, non_blocking=True),
        "geom_mask": batch["geom_mask"].to(device, non_blocking=True),
    }


def evaluate_detailed(model, batch, device):
    model.eval()
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            out = model(batch["images"], batch["yaw_deg"], batch["sheet_half_extent"])
        if tuple(out["P"].shape) != (1, 8, 3, 256, 256):
            raise RuntimeError(f"R256 evaluator shape drift: {tuple(out['P'].shape)}")
        pred = sample_p_field(out["P"], batch["geom_xy"])
        truth = batch["geom_p"].float()
        mask = batch["geom_mask"].bool()
        err = torch.linalg.norm(pred.float() - truth, dim=-1)

    per_view = []
    all_err = []
    for v in range(8):
        ev = err[0, v][mask[0, v]].detach().cpu().numpy().astype(np.float64)
        all_err.append(ev)
        per_view.append({
            "view": v,
            "samples": int(ev.size),
            "P_p50": q(ev, 0.50),
            "P_p90": q(ev, 0.90),
            "P_p95": q(ev, 0.95),
            "P_p99": q(ev, 0.99),
            "P_max": float(np.max(ev)),
            "count_gt_0p005": int(np.sum(ev > P_THRESHOLD)),
            "fraction_gt_0p005": float(np.mean(ev > P_THRESHOLD)),
        })
    agg = np.concatenate(all_err)
    aggregate = {
        "samples": int(agg.size),
        "P_mean": float(np.mean(agg)),
        "P_p50": q(agg, 0.50),
        "P_p90": q(agg, 0.90),
        "P_p95": q(agg, 0.95),
        "P_p99": q(agg, 0.99),
        "P_max": float(np.max(agg)),
        "count_gt_0p005": int(np.sum(agg > P_THRESHOLD)),
        "fraction_gt_0p005": float(np.mean(agg > P_THRESHOLD)),
        "pass_0p005": bool(q(agg, 0.95) <= P_THRESHOLD),
    }
    return {"aggregate": aggregate, "per_view": per_view}


def load_parent_checkpoint(parent_run: Path):
    decision_path = parent_run / "P_V5_R256_ONE_CELL_DECISION.json"
    complete_path = parent_run / "RUN_COMPLETE_P_V5_R256_ONE_CELL_V1.json"
    checkpoint_path = parent_run / "BEST_CHECKPOINT.pt"
    for p in (decision_path, complete_path, checkpoint_path):
        if not p.is_file():
            raise FileNotFoundError(f"parent authority missing: {p}")

    decision = json.load(open(decision_path, encoding="utf-8"))
    complete = json.load(open(complete_path, encoding="utf-8"))
    required = {
        "status": "P_V5_R256_ONE_CELL_OPTIMIZATION_INSUFFICIENT",
        "selected_step": PARENT_SELECTED_STEP,
        "output_field_hw": 256,
        "tune_consumed": False,
        "sealed_splits_opened": False,
        "camera_json_consumed": False,
    }
    for k, v in required.items():
        if decision.get(k) != v:
            raise RuntimeError(f"parent decision drift {k}: {decision.get(k)!r} != {v!r}")
    if abs(float(decision["selected_P_p95"]) - PARENT_SELECTED_P95) > 1e-15:
        raise RuntimeError("parent selected P p95 drift")
    if complete.get("best_checkpoint_sha256") != PARENT_CHECKPOINT_SHA256:
        raise RuntimeError("parent run-complete checkpoint hash drift")
    got_sha = sha256_file(checkpoint_path)
    if got_sha != PARENT_CHECKPOINT_SHA256:
        raise RuntimeError(f"checkpoint SHA drift: {got_sha} != {PARENT_CHECKPOINT_SHA256}")

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if ckpt.get("schema") != "RealSaS.IRISSinglePoseV2.PV5R256OneCellCheckpoint.v1":
        raise RuntimeError(f"checkpoint schema drift: {ckpt.get('schema')}")
    if ckpt.get("optimizer_steps") != PARENT_SELECTED_STEP:
        raise RuntimeError(f"checkpoint step drift: {ckpt.get('optimizer_steps')}")
    return decision, complete, checkpoint_path, ckpt["model_state"]


def run_arm(name, lr, base_state, batch, device, out_dir: Path):
    seed_all(SEED)
    model = IRISSinglePoseV2PV5R256().to(device)
    model.load_state_dict(base_state, strict=True)
    trainable = configure_p_only(model)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(lr), betas=BETAS, weight_decay=WEIGHT_DECAY,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    arm_dir = out_dir / name
    arm_dir.mkdir(parents=True, exist_ok=True)
    eval_records = []
    train_records = []
    best = None

    baseline = evaluate_detailed(model, batch, device)["aggregate"]
    eval_records.append({"restart_step": 0, **baseline})
    atomic_json(arm_dir / "EVAL_0000.json", {"arm": name, "lr": lr, "restart_step": 0, **baseline})

    for step in range(1, STEPS + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            out = model(batch["images"], batch["yaw_deg"], batch["sheet_half_extent"])
        parts = p_only_objective(out, {
            "geom_xy": batch["geom_xy"],
            "geom_p": batch["geom_p"],
            "geom_mask": batch["geom_mask"],
            "yaw_deg": batch["yaw_deg"],
        })
        scaler.scale(parts["total"]).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        scaler.step(opt)
        scaler.update()

        if step == 1 or step % 16 == 0:
            train_records.append({
                "restart_step": step,
                "depth_smooth_l1": float(parts["depth_smooth_l1"].detach().cpu()),
                "train_P_mean": float(parts["p_mean"].detach().cpu()),
                "train_P_p95": float(parts["p_p95"].detach().cpu()),
            })

        if step in EVAL_STEPS[1:]:
            ev = evaluate_detailed(model, batch, device)["aggregate"]
            rec = {"restart_step": step, **ev}
            eval_records.append(rec)
            atomic_json(arm_dir / f"EVAL_{step:04d}.json", {"arm": name, "lr": lr, **rec})
            if best is None or (ev["P_p95"], step) < (best["P_p95"], best["restart_step"]):
                best = {"P_p95": ev["P_p95"], "restart_step": step, "pass_0p005": ev["pass_0p005"]}
                torch.save({
                    "schema": "RealSaS.IRISSinglePoseV2.PV5R256OptimizerLocalizationCheckpoint.v1",
                    "arm": name,
                    "lr": float(lr),
                    "parent_optimizer_steps": PARENT_SELECTED_STEP,
                    "restart_step": step,
                    "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                }, arm_dir / "BEST_CHECKPOINT.pt")
            print(f"[{name}] step={step}/{STEPS} p95={ev['P_p95']:.9f} pass={ev['pass_0p005']}", flush=True)

    atomic_json(arm_dir / "TRAIN_HISTORY.json", {"arm": name, "lr": lr, "records": train_records})
    pass_steps = [r["restart_step"] for r in eval_records if r["pass_0p005"]]
    summary = {
        "arm": name,
        "lr": float(lr),
        "fresh_optimizer_restart": True,
        "planned_restart_steps": STEPS,
        "eval_steps": list(EVAL_STEPS),
        "minimum_P_p95": float(min(r["P_p95"] for r in eval_records)),
        "minimum_eval_step": int(min(eval_records, key=lambda r: (r["P_p95"], r["restart_step"]))["restart_step"]),
        "first_pass_step": int(min(pass_steps)) if pass_steps else None,
        "pass_any": bool(pass_steps),
        "final_P_p95": float(eval_records[-1]["P_p95"]),
        "trainable_parameter_names": trainable,
    }
    atomic_json(arm_dir / "ARM_SUMMARY.json", summary)
    del model, opt, scaler
    torch.cuda.empty_cache()
    return summary


def main():
    ap = argparse.ArgumentParser(description="R256 one-cell optimizer restart localization V1")
    ap.add_argument("--root", required=True)
    ap.add_argument("--parent-run", required=True)
    ap.add_argument("--membership", required=True)
    ap.add_argument("--stage-script", required=True)
    ap.add_argument("--cache-script", required=True)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--persist-dir", required=True)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU REQUIRED for optimizer localization")

    root = Path(args.root)
    parent_run = Path(args.parent_run)
    work = Path(args.work_dir)
    persist = Path(args.persist_dir)
    membership = Path(args.membership)
    stage_script = Path(args.stage_script)
    cache_script = Path(args.cache_script)

    if persist.exists():
        raise RuntimeError(f"persist target already exists; refusing overwrite: {persist}")
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

    decision, complete, checkpoint_path, base_state = load_parent_checkpoint(parent_run)

    # Restage/cache exact same frozen cell. Subprocess boundaries keep the firewall identical to parent gate.
    stage = work / "stage"
    cache = work / "cache"
    import subprocess, sys
    subprocess.check_call([sys.executable, str(stage_script), "--root", str(root), "--membership", str(membership), "--out", str(stage)])
    stage_manifest = stage / "STAGE_MANIFEST.json"
    subprocess.check_call([sys.executable, str(cache_script), "--stage-manifest", str(stage_manifest), "--out", str(cache), "--samples-per-view", "4096"])
    cache_manifest = cache / "CACHE_MANIFEST.json"

    ds = PV5R256OneCellDataset(cache_manifest)
    if len(ds) != 1:
        raise RuntimeError(f"expected one frozen cell, got {len(ds)}")
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    cpu_batch = next(iter(dl))
    device = torch.device("cuda")
    batch = pack(cpu_batch, device)

    # Zero-step reproduction and per-view residual audit.
    seed_all(SEED)
    baseline_model = IRISSinglePoseV2PV5R256().to(device)
    baseline_model.load_state_dict(base_state, strict=True)
    configure_p_only(baseline_model)
    baseline_detail = evaluate_detailed(baseline_model, batch, device)
    baseline_p95 = baseline_detail["aggregate"]["P_p95"]
    baseline_diff = abs(baseline_p95 - PARENT_SELECTED_P95)
    zero = {
        "schema": "RealSaS.IRISSinglePoseV2.PV5R256OptimizerLocalizationZeroStep.v1",
        "parent_status": decision["status"],
        "parent_selected_step": PARENT_SELECTED_STEP,
        "parent_selected_P_p95": PARENT_SELECTED_P95,
        "reproduced_P_p95": baseline_p95,
        "absolute_difference": baseline_diff,
        "tolerance": BASELINE_TOL,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "aggregate": baseline_detail["aggregate"],
        "per_view": baseline_detail["per_view"],
        "scientific_optimizer_steps": 0,
        "status": "PASS" if baseline_diff <= BASELINE_TOL else "APPARATUS_OR_CHECKPOINT_DRIFT",
    }
    atomic_json(work / "ZERO_STEP_RESIDUAL_AUDIT.json", zero)
    del baseline_model
    torch.cuda.empty_cache()
    if zero["status"] != "PASS":
        raise RuntimeError(f"zero-step checkpoint reproduction failed: diff={baseline_diff}")

    preopt = {
        "schema": "RealSaS.IRISSinglePoseV2.PV5R256OptimizerLocalizationPreOpt.v1",
        "status": "FROZEN__THREE_RESTART_ARMS_AUTHORIZED",
        "parent_decision_sha256": sha256_file(parent_run / "P_V5_R256_ONE_CELL_DECISION.json"),
        "parent_run_complete_sha256": sha256_file(parent_run / "RUN_COMPLETE_P_V5_R256_ONE_CELL_V1.json"),
        "parent_checkpoint_sha256": sha256_file(checkpoint_path),
        "zero_step_audit_sha256": sha256_file(work / "ZERO_STEP_RESIDUAL_AUDIT.json"),
        "arms": [{"name": n, "lr": lr, "restart_steps": STEP_STEPSIF_NEGER} for n, lr in ARMS],
        "eval_steps": list(EVAL_STEPS),
        "threshold_P_p95": P_THRESHOLD,
        "tune_consumed": False,
        "sealed_splits_opened": False,
        "camera_json_consumed": False,
        "scientific_optimizer_steps_at_freeze": 0,
    }
    atomic_json(work / "PREOPT_AUTHORITY.json", preopt)

    arm_root = work / "arms"
    arm_summaries = []
    for name, lr in ARMS:
        arm_summaries.append(run_arm(name, lr, base_state, batch, device, arm_root))

    any_pass = any(a["pass_any"] for a in arm_summaries)
    control = next(a for a in arm_summaries if a["arm"] == "A_LR3E4_CONTROL")
    reduced = [a for a in arm_summaries if a["arm"] != "A_LR3E4_CONTROL"]
    if control["pass_any"]:
        localization = "BUDGET_OR_FRESH_RESTART_SUFFICIENT_AT_ORIGINAL_LR"
    elif any(a["pass_any"] for a in reduced):
        localization = "LATE_STAGE_LR_FLOOR_SUPPORTED"
    else:
        localization = "SHORT_RESTART_LR_SWEEP_INSUFFICIENT"

    best_arm = min(arm_summaries, key=lambda a: (a["minimum_P_p95"], a["minimum_eval_step"], a["lr"]))
    status = "P_V5_R256_ONE_CELL_RECOVERY_PASS" if any_pass else "P_V5_R256_ONE_CELL_OPTIMIZER_RESTART_SWEEP_INSUFFICIENT"
    next_policy = (
        "preregister 1 asset x 2 styles R256 only; freeze repaired optimizer evidence from this localization"
        if any_pass else
        "remain at one-cell; inspect residual-tail structure, objective alignment and learner feature capacity before broader training"
    )
    result = {
        "schema": "RealSaS.IRISSinglePoseV2.PV5R256OptimizerLocalizationDecision.v1",
        "status": status,
        "localization_label": localization,
        "parent_selected_P_p95": PARENT_SELECTED_P95,
        "threshold_P_p95": P_THRESHOLD,
        "best_arm": best_arm,
        "arms": arm_summaries,
        "tune_consumed": False,
        "sealed_splits_opened": False,
        "camera_json_consumed": False,
        "next_policy": next_policy,
    }
    atomic_json(work / "P_V5_R256_OPTIMIZER_LOCALIZATION_DECISION.json", result)

    # Persist one global best checkpoint plus compact arm evidence.
    persist.parent.mkdir(parents=True, exist_ok=True)
    tmp = persist.with_name(persist.name + ".partial")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()
    for p in (work / "ZERO_STEP_RESIDUAL_AUDIT.json", work / "PREOPT_AUTHORITY.json", work / "P_V5_R256_OPTIMIZER_LOCALIZATION_DECISION.json"):
        shutil.copy2(p, tmp / p.name)
    shutil.copytree(arm_root, tmp / "arms", ignore=shutil.ignore_patterns("BEST_CHECKPOINT.pt"))
    best_cp = arm_root / best_arm["arm"] / "BEST_CHECKPOINT.pt"
    if best_cp.is_file():
        shutil.copy2(best_cp, tmp / "BEST_CHECKPOINT.pt")
    os.replace(tmp, persist)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
