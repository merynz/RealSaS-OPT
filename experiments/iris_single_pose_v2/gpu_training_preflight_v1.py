from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path

import torch

from losses import total_loss
from model import IRISSinglePoseV2, IRISV2Config, count_parameters


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def environment_snapshot():
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "cudnn_available": bool(torch.backends.cudnn.is_available()),
        "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
    }


def synthetic_batch(device, resolution=256, geom_samples=1024, tracks=128):
    g = torch.Generator(device="cpu")
    g.manual_seed(20260824)
    images = torch.rand((1, 8, 4, resolution, resolution), generator=g).to(device)
    images[:, :, 3] = 1.0
    yaw = (torch.arange(8, dtype=torch.float32)[None] * 45.0).to(device)
    geom_xy = (torch.rand((1, 8, geom_samples, 2), generator=g) * 1.8 - 0.9).to(device)
    geom_p = (torch.rand((1, 8, geom_samples, 3), generator=g) - 0.5).to(device)
    geom_n = torch.nn.functional.normalize(
        torch.randn((1, 8, geom_samples, 3), generator=g).to(device), dim=-1
    )
    geom_mask = torch.ones((1, 8, geom_samples), dtype=torch.bool, device=device)
    track_p = (torch.rand((1, tracks, 3), generator=g) - 0.5).to(device)
    track_p[:, 1::8] = track_p[:, 0::8][:, : track_p[:, 1::8].shape[1]] + 0.001
    track_xy = (torch.rand((1, tracks, 8, 2), generator=g) * 1.8 - 0.9).to(device)
    vis = torch.ones((1, tracks, 8), dtype=torch.bool, device=device)
    vis[:, ::5, 3] = False
    vis[:, 1::7, 6] = False
    return {
        "images": images,
        "yaw_deg": yaw,
        "geom_xy": geom_xy,
        "geom_p": geom_p,
        "geom_n": geom_n,
        "geom_mask": geom_mask,
        "track_p": track_p,
        "track_xy": track_xy,
        "track_visible": vis,
    }


def base_report(resolution, mode):
    return {
        "schema": "RealSaS.IRISSinglePoseV2.MiniGPUCapacityPreflight.v1",
        "status": "NOT_RUN",
        "mode": mode,
        "diagnostic_stage": "init",
        "input_resolution": int(resolution),
        "microbatch": 1,
        "grad_accum_prereg": 4,
        "geom_samples": 1024,
        "track_samples": 128,
        "mixed_precision_policy": "AMP FP16 model forward only; total_loss outside autocast in explicit FP32 numerics",
        "full_loss_enabled": False,
        "forward_backward_executed": False,
        "adamw_moment_memory_accounted_without_step": False,
        "adamw_moment_bytes": 0,
        "scientific_optimizer_steps": 0,
        "parameters": None,
        "gpu": None,
        "gpu_capability": None,
        "gpu_total_memory_GB": None,
        "gpu_free_memory_before_GB": None,
        "peak_allocated_GB": None,
        "peak_reserved_GB": None,
        "elapsed_sec": None,
        "loss": None,
        "shapes": None,
        "output_dtypes": None,
        "loss_dtype": None,
        "finite_gradients": None,
        "error_type": None,
        "error": None,
        "traceback": None,
        "environment": environment_snapshot(),
    }


def cpu_semantic_smoke(report):
    report["diagnostic_stage"] = "cpu_model_init"
    model = IRISSinglePoseV2(IRISV2Config()).to("cpu").train()
    report["parameters"] = count_parameters(model)
    report["diagnostic_stage"] = "cpu_synthetic_batch"
    batch = synthetic_batch(torch.device("cpu"), report["input_resolution"])
    report["diagnostic_stage"] = "cpu_model_forward"
    out = model(batch["images"], batch["yaw_deg"])
    report["diagnostic_stage"] = "cpu_fp32_full_loss"
    parts = total_loss(out, batch, epoch=3, warmup_epochs=3)
    loss = parts["total"]
    report["full_loss_enabled"] = True
    report["diagnostic_stage"] = "cpu_backward"
    loss.backward()
    finite = all(p.grad is None or torch.isfinite(p.grad).all().item() for p in model.parameters())
    if not finite or not torch.isfinite(loss).item():
        raise RuntimeError("non-finite CPU semantic full-loss/backward smoke")
    report["forward_backward_executed"] = True
    report["finite_gradients"] = True
    report["loss"] = float(loss.detach().float().cpu())
    report["loss_dtype"] = str(loss.dtype)
    report["shapes"] = {k: list(v.shape) for k, v in out.items()}
    report["output_dtypes"] = {k: str(v.dtype) for k, v in out.items()}
    report["status"] = "CPU_SEMANTIC_PASS"
    report["diagnostic_stage"] = "complete"


def cuda_capacity_probe(report):
    if not torch.cuda.is_available():
        report["status"] = "NO_CUDA"
        report["diagnostic_stage"] = "cuda_availability"
        raise RuntimeError(
            "CUDA is not available in this runtime. This mini requires a GPU Colab runtime before optimizer step 1."
        )

    dev = torch.device("cuda")
    report["gpu"] = torch.cuda.get_device_name(dev)
    report["gpu_capability"] = list(torch.cuda.get_device_capability(dev))
    free_bytes, total_bytes = torch.cuda.mem_get_info(dev)
    report["gpu_total_memory_GB"] = total_bytes / 1e9
    report["gpu_free_memory_before_GB"] = free_bytes / 1e9

    report["diagnostic_stage"] = "cuda_reset_memory_stats"
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(dev)

    report["diagnostic_stage"] = "cuda_model_init"
    model = IRISSinglePoseV2(IRISV2Config()).to(dev).train()
    report["parameters"] = count_parameters(model)

    report["diagnostic_stage"] = "cuda_synthetic_batch"
    batch = synthetic_batch(dev, report["input_resolution"])

    report["diagnostic_stage"] = "cuda_autocast_model_forward"
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        out = model(batch["images"], batch["yaw_deg"])
    report["output_dtypes"] = {k: str(v.dtype) for k, v in out.items()}

    report["diagnostic_stage"] = "cuda_fp32_full_loss"
    parts = total_loss(out, batch, epoch=3, warmup_epochs=3)
    loss = parts["total"]
    report["full_loss_enabled"] = True
    report["loss_dtype"] = str(loss.dtype)
    if loss.dtype != torch.float32:
        raise RuntimeError(f"full loss must be FP32 outside autocast, got {loss.dtype}")

    report["diagnostic_stage"] = "cuda_backward"
    loss.backward()

    report["diagnostic_stage"] = "cuda_adamw_moment_memory"
    buffers = []
    state_bytes = 0
    for p in model.parameters():
        if p.requires_grad:
            buffers.extend(
                [
                    torch.zeros_like(p, dtype=torch.float32, device=dev),
                    torch.zeros_like(p, dtype=torch.float32, device=dev),
                ]
            )
            state_bytes += 2 * p.numel() * 4
    report["adamw_moment_bytes"] = int(state_bytes)
    _ = sum(x.numel() for x in buffers)

    report["diagnostic_stage"] = "cuda_finite_check"
    torch.cuda.synchronize(dev)
    finite = all(p.grad is None or torch.isfinite(p.grad).all().item() for p in model.parameters())
    if not finite or not torch.isfinite(loss).item():
        raise RuntimeError("non-finite full-loss/backward capacity probe")

    report["forward_backward_executed"] = True
    report["adamw_moment_memory_accounted_without_step"] = True
    report["finite_gradients"] = True
    report["loss"] = float(loss.detach().float().cpu())
    report["shapes"] = {k: list(v.shape) for k, v in out.items()}
    report["peak_allocated_GB"] = torch.cuda.max_memory_allocated(dev) / 1e9
    report["peak_reserved_GB"] = torch.cuda.max_memory_reserved(dev) / 1e9
    report["status"] = "PASS"
    report["diagnostic_stage"] = "complete"


def main():
    ap = argparse.ArgumentParser(description="No-step production-width mini training capacity probe")
    ap.add_argument("--out", required=True)
    ap.add_argument("--resolution", type=int, default=256)
    ap.add_argument(
        "--cpu-semantic-smoke",
        action="store_true",
        help="CI-only: execute the exact synthetic full-loss/backward graph on CPU; not a capacity PASS.",
    )
    a = ap.parse_args()

    mode = "CPU_SEMANTIC_SMOKE" if a.cpu_semantic_smoke else "CUDA_CAPACITY"
    report = base_report(a.resolution, mode)
    t0 = time.time()
    exit_code = 0
    try:
        if a.resolution != 256:
            report["status"] = "CONTRACT_ERROR"
            report["diagnostic_stage"] = "resolution_contract"
            raise RuntimeError("mini prereg freezes capacity probe at 256")
        if a.cpu_semantic_smoke:
            cpu_semantic_smoke(report)
        else:
            cuda_capacity_probe(report)
    except torch.cuda.OutOfMemoryError as exc:
        report["status"] = "OOM"
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)
        report["traceback"] = traceback.format_exc()
        exit_code = 2
    except Exception as exc:
        if report["status"] == "NOT_RUN":
            report["status"] = "ERROR"
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)
        report["traceback"] = traceback.format_exc()
        exit_code = 2
    finally:
        if torch.cuda.is_available():
            try:
                report["peak_allocated_GB"] = torch.cuda.max_memory_allocated() / 1e9
                report["peak_reserved_GB"] = torch.cuda.max_memory_reserved() / 1e9
            except Exception:
                pass
        report["elapsed_sec"] = time.time() - t0
        atomic_json(a.out, report)
        print(json.dumps(report, indent=2), flush=True)

    if a.cpu_semantic_smoke:
        if report["status"] != "CPU_SEMANTIC_PASS":
            raise SystemExit(exit_code or 2)
    elif report["status"] != "PASS":
        raise SystemExit(exit_code or 2)


if __name__ == "__main__":
    main()
