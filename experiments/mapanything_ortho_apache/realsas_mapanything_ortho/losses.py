from __future__ import annotations
from typing import Dict, Tuple
import torch
import torch.nn.functional as F
from .config import OrthoConfig
from .camera import forward_depth_from_points


def sample_field(field: torch.Tensor, xy01: torch.Tensor) -> torch.Tensor:
    B, V, C, H, W = field.shape
    if V != 8 or xy01.shape[:2] != (B, 8): raise ValueError((field.shape, xy01.shape))
    N = xy01.shape[2]
    grid = xy01.to(field.dtype) * 2.0 - 1.0
    y = F.grid_sample(field.reshape(B * 8, C, H, W), grid.reshape(B * 8, N, 1, 2), mode="bilinear", padding_mode="border", align_corners=False)
    return y[..., 0].permute(0, 2, 1).reshape(B, 8, N, C)


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.to(x.dtype)
    while m.ndim < x.ndim: m = m.unsqueeze(-1)
    denom = m.expand_as(x).sum().clamp_min(1.0)
    return (x * m).sum() / denom


def multiview_consistency(sampled_P: torch.Tensor, visible: torch.Tensor) -> torch.Tensor:
    m = visible.to(sampled_P.dtype).unsqueeze(-1)
    count = m.sum(dim=1, keepdim=True).clamp_min(1.0)
    center = (sampled_P * m).sum(dim=1, keepdim=True) / count
    valid = (visible.sum(dim=1, keepdim=True) >= 2).expand_as(visible)
    return masked_mean(F.smooth_l1_loss(sampled_P, center.expand_as(sampled_P), reduction="none", beta=0.02), visible & valid)


def geometry_loss(outputs: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor], cfg: OrthoConfig) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    xy = target["XY01"].to(outputs["point"].device); P_true = target["P"].to(outputs["point"].device); N_true = target["N"].to(outputs["point"].device); V = target["V"].to(outputs["point"].device).bool()
    tP = P_true[:, None].expand(-1, 8, -1, -1); tN = N_true[:, None].expand(-1, 8, -1, -1)
    sP = sample_field(outputs["point"], xy); sN = sample_field(outputs["normal"], xy); sd = sample_field(outputs["depth"], xy)[..., 0]; sls = sample_field(outputs["log_sigma"], xy)[..., 0]
    d_true = forward_depth_from_points(P_true, xy)
    point_err = F.smooth_l1_loss(sP, tP, reduction="none", beta=cfg.huber_beta).sum(-1)
    Lp = masked_mean(point_err, V)
    Ld = masked_mean(F.smooth_l1_loss(sd, d_true, reduction="none", beta=cfg.huber_beta), V)
    Ln = masked_mean(1.0 - (sN * tN).sum(-1).clamp(-1.0, 1.0), V)
    Lc = multiview_consistency(sP, V)
    e = torch.linalg.norm(sP - tP, dim=-1).detach()
    Lu = masked_mean(torch.exp(-sls) * e + 0.05 * sls, V)
    total = cfg.w_point * Lp + cfg.w_depth * Ld + cfg.w_normal * Ln + cfg.w_consistency * Lc + cfg.w_risk * Lu
    return total, {"P": Lp, "depth": Ld, "normal": Ln, "consistency": Lc, "risk": Lu, "total": total}


@torch.no_grad()
def geometry_metrics(outputs: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor]) -> Dict[str, float]:
    xy = target["XY01"].to(outputs["point"].device); P_true = target["P"].to(outputs["point"].device); N_true = target["N"].to(outputs["point"].device); V = target["V"].to(outputs["point"].device).bool()
    tP = P_true[:, None].expand(-1, 8, -1, -1); tN = N_true[:, None].expand(-1, 8, -1, -1)
    sP = sample_field(outputs["point"], xy); sN = sample_field(outputs["normal"], xy)
    e = torch.linalg.norm(sP - tP, dim=-1)[V]
    ang = torch.rad2deg(torch.acos((sN * tN).sum(-1).clamp(-1.0, 1.0)))[V]
    def q(x, p): return float(torch.quantile(x.float(), p).item()) if x.numel() else float("nan")
    return {"P_mean": float(e.mean().item()) if e.numel() else float("nan"), "P50": q(e,.50), "P90": q(e,.90), "P95": q(e,.95), "N_deg_median": q(ang,.50), "N_deg_p95": q(ang,.95), "visible_count": int(V.sum().item())}
