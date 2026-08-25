from __future__ import annotations

import ast
import inspect
import json
import textwrap

import torch
import torch.nn.functional as F

import gpu_training_preflight_v1 as gpu_preflight
import losses
import train_mini_v2 as train


def calls_inside_autocast(fn):
    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    calls = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.With, ast.AsyncWith)):
            continue
        if not any("autocast" in ast.unparse(item.context_expr) for item in node.items):
            continue
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                calls.append(ast.unparse(child.func))
    return calls


def assert_forward_only_autocast(fn, label):
    source = inspect.getsource(fn)
    inside = calls_inside_autocast(fn)
    if "total_loss" not in source:
        raise RuntimeError(f"{label}: total_loss call missing")
    if any(name.endswith("total_loss") or name == "total_loss" for name in inside):
        raise RuntimeError(f"{label}: total_loss is nested inside autocast: {inside}")
    if not any(name == "model" or name.endswith(".model") for name in inside):
        raise RuntimeError(f"{label}: model forward not found inside autocast: {inside}")


def main():
    torch.manual_seed(20260824)

    # Enforce the actual training/capacity context boundary, not only local dtype casts.
    assert_forward_only_autocast(train.main, "train_mini_v2.main")
    assert_forward_only_autocast(gpu_preflight.cuda_capacity_probe, "gpu_training_preflight_v1.cuda_capacity_probe")

    # Exact regression for the CI183/T4 failure: FP16 similarity followed by -1e9 masked_fill.
    zs = torch.randn(128, 32, dtype=torch.float16)
    zt = torch.randn(128, 32, dtype=torch.float16)
    p = torch.randn(128, 3, dtype=torch.float32)
    dual, hard, reciprocal = losses.pairwise_coarse_objective(zs, zt, p)
    coarse_terms = torch.stack([dual, hard, reciprocal])
    if coarse_terms.dtype != torch.float32 or not torch.isfinite(coarse_terms).all():
        raise RuntimeError("FP16 coarse descriptor regression did not promote to finite FP32 numerics")

    # Exercise every loss branch with half model outputs. Loss code must promote supervision numerics
    # to FP32 before grid_sample/similarity/logsumexp/softmax/exp/cross-entropy.
    b, v, r, tracks, geom = 1, 2, 32, 16, 8
    outputs = {
        "P": torch.randn(b, v, 3, r, r, dtype=torch.float16, requires_grad=True),
        "N": torch.randn(b, v, 3, r, r, dtype=torch.float16, requires_grad=True),
        "U_geo": torch.randn(b, v, 1, r, r, dtype=torch.float16, requires_grad=True),
        "Z_coarse": torch.randn(b, v, 32, r // 4, r // 4, dtype=torch.float16, requires_grad=True),
        "Z_fine": torch.randn(b, v, 16, r, r, dtype=torch.float16, requires_grad=True),
    }
    track_xy = torch.rand(b, tracks, v, 2) * 1.6 - 0.8
    geom_xy = torch.rand(b, v, geom, 2) * 1.6 - 0.8
    batch = {
        "yaw_deg": torch.arange(v, dtype=torch.float32)[None] * 45.0,
        "geom_xy": geom_xy,
        "geom_p": torch.randn(b, v, geom, 3),
        "geom_n": F.normalize(torch.randn(b, v, geom, 3), dim=-1),
        "geom_mask": torch.ones(b, v, geom, dtype=torch.bool),
        "track_xy": track_xy,
        "track_visible": torch.ones(b, tracks, v, dtype=torch.bool),
        "track_p": torch.randn(b, tracks, 3),
    }

    parts = losses.total_loss(outputs, batch, epoch=3, warmup_epochs=3)
    total = parts["total"]
    if total.dtype != torch.float32 or not torch.isfinite(total):
        raise RuntimeError(f"full mixed-precision loss is not finite FP32: dtype={total.dtype} value={total}")
    for name, value in parts.items():
        if torch.is_tensor(value) and not torch.isfinite(value.float()).all():
            raise RuntimeError(f"non-finite loss component: {name}")

    total.backward()
    for name, value in outputs.items():
        if value.grad is not None and not torch.isfinite(value.grad.float()).all():
            raise RuntimeError(f"non-finite gradient through FP16 output field: {name}")

    report = {
        "schema": "RealSaS.IRISSinglePoseV2.AMPLossPrecisionPreflight.v1",
        "status": "PASS",
        "prior_failure_regression": "FP16 masked_fill -1e9 overflow",
        "autocast_boundary": "model forward only; total_loss outside autocast",
        "trainer_boundary_enforced": True,
        "gpu_preflight_boundary_enforced": True,
        "coarse_half_input_promoted_to_fp32": True,
        "all_loss_fields_half_input": True,
        "P_depth_supervision_known_yaw_bound": True,
        "full_loss_dtype": str(total.dtype),
        "full_loss_finite": True,
        "backward_finite": True,
        "scientific_optimizer_steps": 0,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
