from __future__ import annotations
import torch


def depth_metrics_v2(predicted_depth: torch.Tensor, teacher_depth: torch.Tensor, support: torch.Tensor) -> dict[str, float]:
    mask = support.bool()
    if not mask.any():
        return {"count": 0, "mae": float("nan"), "rmse": float("nan"), "p95_abs": float("nan")}
    err = (predicted_depth - teacher_depth)[mask].detach().float().abs().cpu()
    return {
        "count": int(err.numel()),
        "mae": float(err.mean()),
        "rmse": float(torch.sqrt((err.square()).mean())),
        "p95_abs": float(torch.quantile(err, 0.95)),
    }
