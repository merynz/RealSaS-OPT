from __future__ import annotations

import json
import torch
import torch.nn.functional as F

import losses


def main():
    torch.manual_seed(20260824)

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
        "coarse_half_input_promoted_to_fp32": True,
        "all_loss_fields_half_input": True,
        "full_loss_dtype": str(total.dtype),
        "full_loss_finite": True,
        "backward_finite": True,
        "scientific_optimizer_steps": 0,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
