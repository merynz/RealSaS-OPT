from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch

from compiler.realsas_compiler_core.geometry_artifact_codec_v2 import rigging_surface_from_dict
from compiler.realsas_compiler_core.preproduct_authority_v1 import normalization_domain_from_dict
from models.tessa.v1 import (
    TESSAConfigV1,
    TESSAV1,
    build_teacher_asset_sequence_v1,
    build_truncated_teacher_windows_v1,
    build_tessa_conditioning_v1,
)


SCHEMA = "RealSaS.TESSASupervisedFitResult.v2"
CHECKPOINT_SCHEMA = "RealSaS.TESSASupervisedCheckpoint.v2"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_teacher_vertices(vertices: np.ndarray, *, center: tuple[float, float, float], scale: float) -> np.ndarray:
    """Map world-space teacher vertices into the Stage08-bound TESSA token frame."""
    v = np.asarray(vertices, dtype=np.float64)
    c = np.asarray(center, dtype=np.float64)
    if v.ndim != 2 or v.shape[1] != 3 or not np.isfinite(v).all():
        raise ValueError("TESSA_TEACHER_VERTICES_INVALID")
    out = (v - c[None, :]) / float(scale)
    if np.any(out < -0.5000001) or np.any(out > 0.5000001):
        lo = out.min(axis=0).tolist()
        hi = out.max(axis=0).tolist()
        raise ValueError(f"TESSA_TEACHER_OUTSIDE_CANONICAL_NORMALIZATION:{lo}:{hi}")
    return out


def load_teacher(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(npz_path, allow_pickle=False) as z:
        vertex_key = "vertices_source" if "vertices_source" in z.files else "vertices"
        face_key = "faces" if "faces" in z.files else "faces_source"
        if vertex_key not in z.files or face_key not in z.files:
            raise ValueError(f"TESSA_TEACHER_NPZ_KEYS_MISSING:{z.files}")
        vertices = np.asarray(z[vertex_key], dtype=np.float64)
        faces = np.asarray(z[face_key], dtype=np.int64)
    return vertices, faces


def parameter_count(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(lr)


def scheduled_lr(*, step: int, max_steps: int, base_lr: float, min_lr: float, warmup_steps: int) -> float:
    if step < 1 or max_steps < 1:
        raise ValueError("TESSA_LR_STEP_INVALID")
    if warmup_steps > 0 and step <= warmup_steps:
        return float(base_lr) * float(step) / float(warmup_steps)
    if max_steps <= warmup_steps:
        return float(min_lr)
    progress = min(1.0, max(0.0, (step - warmup_steps) / float(max_steps - warmup_steps)))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(min_lr) + (float(base_lr) - float(min_lr)) * cosine


def cuda_memory_record(device: torch.device) -> dict[str, float | int]:
    if device.type != "cuda":
        return {"cuda_allocated_bytes": 0, "cuda_reserved_bytes": 0, "cuda_peak_bytes": 0}
    return {
        "cuda_allocated_bytes": int(torch.cuda.memory_allocated()),
        "cuda_reserved_bytes": int(torch.cuda.memory_reserved()),
        "cuda_peak_bytes": int(torch.cuda.max_memory_allocated()),
    }


def save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    pass_index: int,
    cfg: TESSAConfigV1,
    metadata: dict,
) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "schema": CHECKPOINT_SCHEMA,
            "step": int(step),
            "pass_index": int(pass_index),
            "config": asdict(cfg),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "metadata": metadata,
        },
        tmp,
    )
    tmp.replace(path)


def make_window_stream(window_count: int, seed: int, *, start_pass: int = 0):
    if window_count < 1:
        raise ValueError("TESSA_WINDOW_STREAM_EMPTY")
    pass_index = int(start_pass)
    while True:
        order = list(range(window_count))
        random.Random(seed + pass_index).shuffle(order)
        for window_index in order:
            yield pass_index, window_index
        pass_index += 1


@torch.no_grad()
def evaluate_teacher_forced(
    *,
    model: TESSAV1,
    surface_features: torch.Tensor,
    windows,
    cfg: TESSAConfigV1,
    use_bf16: bool,
    amp_dtype: torch.dtype,
    device: torch.device,
    step: int,
) -> dict:
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    correct1 = 0
    correct5 = 0
    t0 = time.time()
    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_bf16):
        memory = model.encode_surface(surface_features)
        for window in windows:
            ids = torch.tensor(window.input_ids, dtype=torch.long, device=device).unsqueeze(0)
            labels = torch.tensor(window.labels, dtype=torch.long, device=device).unsqueeze(0)
            out = model.decode_tokens(
                memory=memory,
                input_ids=ids,
                labels=labels,
                sequence_position_offset=window.sequence_position_offset,
            )
            if out.loss is None:
                raise RuntimeError("TESSA_EVAL_LOSS_MISSING")
            mask = labels != cfg.PAD
            scored = int(mask.sum().item())
            if scored < 1:
                raise RuntimeError("TESSA_EVAL_WINDOW_EMPTY")
            logits = out.logits[mask]
            target = labels[mask]
            total_nll += float(out.loss.detach().float().cpu()) * scored
            total_tokens += scored
            top = torch.topk(logits, k=min(5, logits.shape[-1]), dim=-1).indices
            correct1 += int((top[:, 0] == target).sum().item())
            correct5 += int((top == target[:, None]).any(dim=-1).sum().item())
    ce = total_nll / max(total_tokens, 1)
    elapsed = max(time.time() - t0, 1e-9)
    record = {
        "step": int(step),
        "teacher_forced_ce": float(ce),
        "teacher_forced_perplexity": float(math.exp(min(ce, 20.0))),
        "teacher_forced_top1": float(correct1 / max(total_tokens, 1)),
        "teacher_forced_top5": float(correct5 / max(total_tokens, 1)),
        "scored_tokens": int(total_tokens),
        "eval_seconds": float(elapsed),
        "eval_tokens_per_second": float(total_tokens / elapsed),
    }
    record.update(cuda_memory_record(device))
    return record


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--surface-json", required=True)
    ap.add_argument("--normalization-json", required=True)
    ap.add_argument("--teacher-npz", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--max-steps", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--min-lr", type=float, default=1e-5)
    ap.add_argument("--warmup-steps", type=int, default=64)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--target-tokens", type=int, default=4096)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--log-every", type=int, default=16)
    ap.add_argument("--eval-every", type=int, default=64)
    ap.add_argument("--checkpoint-every", type=int, default=256)
    ap.add_argument("--fit-ce-threshold", type=float, default=0.50)
    ap.add_argument("--fit-top1-threshold", type=float, default=0.90)
    ap.add_argument("--fit-top5-threshold", type=float, default=0.98)
    ap.add_argument("--fit-relative-improvement-threshold", type=float, default=0.90)
    ap.add_argument("--fit-final-regression-fraction", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--small-smoke", action="store_true")
    args = ap.parse_args()

    integer_args = (
        args.max_steps,
        args.grad_accum,
        args.target_tokens,
        args.log_every,
        args.eval_every,
        args.checkpoint_every,
    )
    if any(int(x) < 1 for x in integer_args) or args.target_tokens < 32:
        raise ValueError("TESSA_TRAIN_ARGUMENT_INVALID")
    if not (0.0 < args.min_lr <= args.lr):
        raise ValueError("TESSA_LR_RANGE_INVALID")
    if args.warmup_steps < 0 or args.warmup_steps >= args.max_steps:
        raise ValueError("TESSA_WARMUP_INVALID")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    surface_path = Path(args.surface_json)
    normalization_path = Path(args.normalization_json)
    teacher_path = Path(args.teacher_npz)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("TESSA_PHASE", json.dumps({"phase": "LOAD_AUTHORITY_INPUTS"}), flush=True)
    surface_payload = json.loads(surface_path.read_text(encoding="utf-8"))
    normalization_payload = json.loads(normalization_path.read_text(encoding="utf-8"))
    surface = rigging_surface_from_dict(surface_payload)
    normalization = normalization_domain_from_dict(normalization_payload)
    conditioning = build_tessa_conditioning_v1(surface, normalization=normalization)
    teacher_vertices, teacher_faces = load_teacher(teacher_path)
    normalized_teacher = normalize_teacher_vertices(
        teacher_vertices,
        center=conditioning.center,
        scale=conditioning.scale,
    )

    cfg = TESSAConfigV1()
    if args.small_smoke:
        cfg = TESSAConfigV1(
            d_model=128,
            n_heads=8,
            surface_layers=2,
            decoder_layers=2,
            mlp_ratio=2,
            surface_latent_count=96,
            local_attention_window=256,
            query_chunk_size=64,
            max_faces=32768,
            max_vertices=32768,
        )

    print("TESSA_PHASE", json.dumps({"phase": "SERIALIZE_GLOBAL_TEACHER_TOPOLOGY"}), flush=True)
    teacher_sequence = build_teacher_asset_sequence_v1(normalized_teacher, teacher_faces, cfg=cfg)
    windows = build_truncated_teacher_windows_v1(
        teacher_sequence,
        cfg=cfg,
        target_tokens=int(args.target_tokens),
    )
    if not windows:
        raise RuntimeError("TESSA_NO_TRAINING_WINDOWS")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("TESSA_PHASE", json.dumps({"phase": "BUILD_MODEL", "device": str(device)}), flush=True)
    model = TESSAV1(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    latest = output_dir / "TESSA_V1_LATEST.pt"
    global_step = 0
    pass_index = 0

    manifest = {
        "surface_sha256": sha256_file(surface_path),
        "normalization_sha256": sha256_file(normalization_path),
        "normalization_hash": conditioning.normalization_hash,
        "coordinate_frame": conditioning.coordinate_frame,
        "tessa_world_scale": float(conditioning.scale),
        "teacher_sha256": sha256_file(teacher_path),
        "surface_geometry_lineage_hash": conditioning.source_geometry_lineage_hash,
        "teacher_vertex_count": int(len(teacher_vertices)),
        "teacher_face_count": int(len(teacher_faces)),
        "teacher_component_count": int(teacher_sequence.component_count),
        "teacher_restart_count": int(teacher_sequence.restart_count),
        "global_sequence_tokens": int(len(teacher_sequence.token_ids)),
        "training_window_count": int(len(windows)),
        "target_tokens_per_window": int(args.target_tokens),
        "causal_context_tokens": int(cfg.local_attention_window),
        "artificial_topology_chart_split": False,
        "teacher_refit_or_clip_performed": False,
        "model_parameter_count": int(parameter_count(model)),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "bf16": bool(device.type == "cuda" and torch.cuda.is_bf16_supported()),
        "seed": int(args.seed),
        "step_authority": True,
        "target_optimizer_steps": int(args.max_steps),
        "micro_windows_per_optimizer_step": int(args.grad_accum),
        "log_every_steps": int(args.log_every),
        "eval_every_steps": int(args.eval_every),
        "checkpoint_every_steps": int(args.checkpoint_every),
        "lr_schedule": {
            "kind": "LINEAR_WARMUP_COSINE_DECAY_V1",
            "base_lr": float(args.lr),
            "min_lr": float(args.min_lr),
            "warmup_steps": int(args.warmup_steps),
        },
        "fit_gate": {
            "teacher_forced_ce_max": float(args.fit_ce_threshold),
            "teacher_forced_top1_min": float(args.fit_top1_threshold),
            "teacher_forced_top5_min": float(args.fit_top5_threshold),
            "relative_ce_improvement_min": float(args.fit_relative_improvement_threshold),
            "final_regression_from_best_fraction_max": float(args.fit_final_regression_fraction),
            "gate_scope": "FIT1_TEACHER_FORCED_RECONSTRUCTION_ONLY",
            "autoregressive_generation_gate_required_next": True,
        },
        "product_authority_claimed": False,
        "generalization_claimed": False,
    }

    if args.resume and latest.is_file():
        print("TESSA_PHASE", json.dumps({"phase": "RESUME_CHECKPOINT"}), flush=True)
        state = torch.load(latest, map_location="cpu")
        if state.get("schema") != CHECKPOINT_SCHEMA:
            raise ValueError("TESSA_RESUME_SCHEMA_MISMATCH")
        if dict(state.get("config") or {}) != asdict(cfg):
            raise ValueError("TESSA_RESUME_CONFIG_MISMATCH")
        old_meta = dict(state.get("metadata") or {})
        for key in (
            "surface_sha256",
            "normalization_sha256",
            "normalization_hash",
            "coordinate_frame",
            "teacher_sha256",
            "global_sequence_tokens",
            "target_tokens_per_window",
            "causal_context_tokens",
            "target_optimizer_steps",
            "micro_windows_per_optimizer_step",
        ):
            if old_meta.get(key) != manifest.get(key):
                raise ValueError(f"TESSA_RESUME_INPUT_CONTRACT_MISMATCH:{key}")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        global_step = int(state["step"])
        pass_index = int(state.get("pass_index", 0))
        print("TESSA_RESUME", json.dumps({"step": global_step, "pass_index": pass_index}), flush=True)

    if global_step >= args.max_steps:
        raise ValueError(f"TESSA_RESUME_ALREADY_AT_OR_BEYOND_TARGET:{global_step}>={args.max_steps}")

    surface_features = conditioning.features.unsqueeze(0).to(device)
    use_bf16 = bool(manifest["bf16"])
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float32
    (output_dir / "TESSA_V1_FIT_INPUT_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("TESSA_PREFLIGHT", json.dumps(manifest, sort_keys=True), flush=True)

    print("TESSA_PHASE", json.dumps({"phase": "INITIAL_EVALUATION"}), flush=True)
    initial_eval = evaluate_teacher_forced(
        model=model,
        surface_features=surface_features,
        windows=windows,
        cfg=cfg,
        use_bf16=use_bf16,
        amp_dtype=amp_dtype,
        device=device,
        step=global_step,
    )
    print("TESSA_EVAL", json.dumps(initial_eval, sort_keys=True), flush=True)

    evaluations = [initial_eval]
    best_eval = dict(initial_eval)
    window_stream = make_window_stream(len(windows), args.seed, start_pass=pass_index)
    session_start_step = global_step
    session_t0 = time.time()
    rolling_loss_sum = 0.0
    rolling_tokens = 0
    rolling_micro_windows = 0
    nonfinite_seen = False

    print("TESSA_PHASE", json.dumps({"phase": "TRAIN", "target_step": args.max_steps}), flush=True)
    while global_step < args.max_steps:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        step_loss_sum = 0.0
        step_tokens = 0
        step_pass_index = pass_index
        for _ in range(args.grad_accum):
            step_pass_index, window_index = next(window_stream)
            pass_index = step_pass_index
            window = windows[window_index]
            ids = torch.tensor(window.input_ids, dtype=torch.long, device=device).unsqueeze(0)
            labels = torch.tensor(window.labels, dtype=torch.long, device=device).unsqueeze(0)
            scored = int((labels != cfg.PAD).sum().item())
            if scored < 1:
                raise RuntimeError("TESSA_WINDOW_HAS_NO_SCORED_TOKENS")
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_bf16):
                out = model(
                    surface_features=surface_features,
                    input_ids=ids,
                    labels=labels,
                    sequence_position_offset=window.sequence_position_offset,
                )
                if out.loss is None:
                    raise RuntimeError("TESSA_LOSS_MISSING")
                if not torch.isfinite(out.loss):
                    nonfinite_seen = True
                    raise RuntimeError("TESSA_NONFINITE_LOSS")
                loss = out.loss / float(args.grad_accum)
            loss.backward()
            raw_loss = float(out.loss.detach().float().cpu())
            step_loss_sum += raw_loss * scored
            step_tokens += scored
            rolling_loss_sum += raw_loss * scored
            rolling_tokens += scored
            rolling_micro_windows += 1

        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).detach().float().cpu())
        next_step = global_step + 1
        lr = scheduled_lr(
            step=next_step,
            max_steps=args.max_steps,
            base_lr=args.lr,
            min_lr=args.min_lr,
            warmup_steps=args.warmup_steps,
        )
        set_optimizer_lr(optimizer, lr)
        optimizer.step()
        global_step = next_step

        if global_step % args.log_every == 0 or global_step == args.max_steps:
            elapsed = max(time.time() - session_t0, 1e-9)
            completed = max(global_step - session_start_step, 1)
            steps_per_second = completed / elapsed
            remaining = max(args.max_steps - global_step, 0)
            eta_seconds = remaining / max(steps_per_second, 1e-12)
            progress = {
                "step": int(global_step),
                "target_step": int(args.max_steps),
                "progress_fraction": float(global_step / args.max_steps),
                "pass_index": int(pass_index),
                "lr": float(lr),
                "rolling_mean_token_loss": float(rolling_loss_sum / max(rolling_tokens, 1)),
                "last_step_mean_token_loss": float(step_loss_sum / max(step_tokens, 1)),
                "grad_norm_preclip": float(grad_norm),
                "micro_windows_since_log": int(rolling_micro_windows),
                "scored_tokens_since_log": int(rolling_tokens),
                "elapsed_seconds": float(elapsed),
                "eta_seconds": float(eta_seconds),
                "optimizer_steps_per_second": float(steps_per_second),
                "scored_tokens_per_second_since_log": float(rolling_tokens / max(elapsed / completed * args.log_every, 1e-9)),
            }
            progress.update(cuda_memory_record(device))
            print("TESSA_PROGRESS", json.dumps(progress, sort_keys=True), flush=True)
            rolling_loss_sum = 0.0
            rolling_tokens = 0
            rolling_micro_windows = 0

        should_eval = global_step % args.eval_every == 0 or global_step == args.max_steps
        if should_eval:
            record = evaluate_teacher_forced(
                model=model,
                surface_features=surface_features,
                windows=windows,
                cfg=cfg,
                use_bf16=use_bf16,
                amp_dtype=amp_dtype,
                device=device,
                step=global_step,
            )
            evaluations.append(record)
            if record["teacher_forced_ce"] < best_eval["teacher_forced_ce"]:
                best_eval = dict(record)
            print("TESSA_EVAL", json.dumps(record, sort_keys=True), flush=True)

        should_checkpoint = global_step % args.checkpoint_every == 0 or global_step == args.max_steps
        if should_checkpoint:
            print("TESSA_PHASE", json.dumps({"phase": "CHECKPOINT", "step": global_step}), flush=True)
            save_checkpoint(
                latest,
                model=model,
                optimizer=optimizer,
                step=global_step,
                pass_index=pass_index,
                cfg=cfg,
                metadata=manifest,
            )
            print(
                "TESSA_CHECKPOINT",
                json.dumps({"step": global_step, "path": latest.name, "bytes": latest.stat().st_size}, sort_keys=True),
                flush=True,
            )

    final_eval = evaluations[-1]
    initial_ce = float(initial_eval["teacher_forced_ce"])
    final_ce = float(final_eval["teacher_forced_ce"])
    best_ce = float(best_eval["teacher_forced_ce"])
    relative_improvement = (initial_ce - final_ce) / max(initial_ce, 1e-12)
    final_regression_fraction = max(0.0, (final_ce - best_ce) / max(best_ce, 1e-12))

    fit_gate = {
        "target_optimizer_steps_reached": bool(global_step == args.max_steps),
        "teacher_forced_ce_pass": bool(final_ce <= args.fit_ce_threshold),
        "teacher_forced_top1_pass": bool(final_eval["teacher_forced_top1"] >= args.fit_top1_threshold),
        "teacher_forced_top5_pass": bool(final_eval["teacher_forced_top5"] >= args.fit_top5_threshold),
        "relative_ce_improvement_pass": bool(relative_improvement >= args.fit_relative_improvement_threshold),
        "final_regression_pass": bool(final_regression_fraction <= args.fit_final_regression_fraction),
        "nonfinite_loss_absent": bool(not nonfinite_seen),
        "initial_ce": initial_ce,
        "final_ce": final_ce,
        "best_ce": best_ce,
        "relative_ce_improvement": float(relative_improvement),
        "final_regression_fraction": float(final_regression_fraction),
        "final_top1": float(final_eval["teacher_forced_top1"]),
        "final_top5": float(final_eval["teacher_forced_top5"]),
        "gate_scope": "FIT1_TEACHER_FORCED_RECONSTRUCTION_ONLY",
        "autoregressive_generation_gate_run": False,
    }
    gate_pass = all(
        bool(v)
        for k, v in fit_gate.items()
        if k.endswith("_pass") or k in {"target_optimizer_steps_reached", "nonfinite_loss_absent"}
    )
    if args.small_smoke:
        gate_pass = True
        fit_gate["smoke_override"] = True

    result = {
        "schema": SCHEMA,
        "status": "PASS" if gate_pass else "FAIL",
        "training_class": "T1_SUPERVISED_TOPOLOGY_FIT",
        "fit1_teacher_forced_gate": fit_gate,
        "input_manifest": manifest,
        "evaluations": evaluations,
        "latest_checkpoint": latest.name,
        "latest_checkpoint_sha256": sha256_file(latest),
        "final_optimizer_step": int(global_step),
        "product_authority_claimed": False,
        "generalization_claimed": False,
        "mechanical_consequence_training_included": False,
        "next_required_rung": "T1B_AUTOREGRESSIVE_RECONSTRUCTION" if gate_pass and not args.small_smoke else "T1_FIT1_REMEDIATION",
    }
    (output_dir / "TESSA_V1_FIT_RESULT.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("TESSA_FIT1_GATE", json.dumps(fit_gate, sort_keys=True), flush=True)
    print("TESSA_SUPERVISED_FIT_DONE", json.dumps({"status": result["status"], "step": global_step, "next": result["next_required_rung"]}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
