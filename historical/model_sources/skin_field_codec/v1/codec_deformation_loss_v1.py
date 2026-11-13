from __future__ import annotations

import torch


def torch_verified_lbs_v1(rest_points: torch.Tensor, weights: torch.Tensor, transforms: torch.Tensor) -> torch.Tensor:
    """Differentiable evaluation-only LBS used for deformation-sensitive codec loss."""
    if rest_points.ndim != 3 or rest_points.shape[-1] != 3:
        raise ValueError("rest_points must be [B,N,3]")
    if weights.ndim != 3 or weights.shape[:2] != rest_points.shape[:2]:
        raise ValueError("weights must be [B,N,J]")
    if transforms.ndim != 5 or transforms.shape[0] != rest_points.shape[0] or transforms.shape[2] != weights.shape[2] or transforms.shape[-2:] != (4,4):
        raise ValueError("transforms must be [B,P,J,4,4]")
    if not torch.isfinite(rest_points).all() or not torch.isfinite(weights).all() or not torch.isfinite(transforms).all():
        raise ValueError("deformation inputs must be finite")
    ones = torch.ones((*rest_points.shape[:2], 1), dtype=rest_points.dtype, device=rest_points.device)
    hom = torch.cat([rest_points, ones], dim=-1)
    # transforms [B,P,J,A,C] x hom [B,N,C] -> [B,P,J,N,A]
    transformed = torch.einsum("bpjac,bnc->bpjna", transforms, hom)[..., :3]
    # weights [B,N,J] x transformed [B,P,J,N,A] -> [B,P,N,A]
    return torch.einsum("bnj,bpjna->bpna", weights, transformed)


def codec_deformation_loss_v1(
    decoded_weights: torch.Tensor,
    teacher_weights: torch.Tensor,
    rest_points: torch.Tensor,
    probe_transforms: torch.Tensor,
    surface_mask: torch.Tensor,
    joint_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if decoded_weights.shape != teacher_weights.shape:
        raise ValueError("decoded/teacher weight shape mismatch")
    if rest_points.shape[:2] != decoded_weights.shape[:2] or rest_points.shape[-1] != 3:
        raise ValueError("rest point shape mismatch")
    if surface_mask.shape != decoded_weights.shape[:2] or joint_mask.shape != (decoded_weights.shape[0], decoded_weights.shape[2]):
        raise ValueError("mask shape mismatch")
    pair = surface_mask[:, :, None].bool() & joint_mask[:, None, :].bool()
    pred_w = decoded_weights * pair.to(decoded_weights.dtype)
    truth_w = teacher_weights * pair.to(teacher_weights.dtype)
    pred_sum = pred_w.sum(dim=-1, keepdim=True)
    truth_sum = truth_w.sum(dim=-1, keepdim=True)
    valid_rows = surface_mask.bool()
    if (pred_sum[valid_rows] <= 1e-8).any() or (truth_sum[valid_rows] <= 1e-8).any():
        raise ValueError("deformation loss received zero-mass valid row")
    pred_w = pred_w / pred_sum.clamp_min(1e-8)
    truth_w = truth_w / truth_sum.clamp_min(1e-8)
    pred = torch_verified_lbs_v1(rest_points, pred_w, probe_transforms)
    truth = torch_verified_lbs_v1(rest_points, truth_w, probe_transforms)
    valid = surface_mask[:, None, :, None].to(pred.dtype)
    delta = (pred - truth) * valid
    denom = valid.sum().clamp_min(1.0) * pred.shape[1] * pred.shape[-1]
    mse = delta.square().sum() / denom
    rms = torch.sqrt(mse + 1e-12)
    return {"deformation_mse": mse, "deformation_rms": rms, "predicted_deformation": pred, "teacher_deformation": truth}
