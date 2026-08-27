from __future__ import annotations
from typing import List, Dict, Tuple
import torch
import torch.nn.functional as F
from .camera import orthographic_basis


def normalize_dinov2(rgb: torch.Tensor) -> torch.Tensor:
    try:
        from uniception.models.encoders.image_normalizations import IMAGE_NORMALIZATION_DICT
    except Exception as exc:
        raise RuntimeError("uniception==0.1.7 is required for exact MapAnything preprocessing") from exc
    norm = IMAGE_NORMALIZATION_DICT["dinov2"]
    mean = torch.as_tensor(norm.mean, device=rgb.device, dtype=rgb.dtype).view(1, 1, 3, 1, 1)
    std = torch.as_tensor(norm.std, device=rgb.device, dtype=rgb.dtype).view(1, 1, 3, 1, 1)
    return (rgb - mean) / std


def prepare_global_views(native_rgba: torch.Tensor, global_size: int = 518) -> Tuple[List[Dict], torch.Tensor]:
    """RGBA [B,8,4,H,W] -> MapAnything RGB views + deterministic known-camera rotation."""
    if native_rgba.ndim != 5 or native_rgba.shape[1:3] != (8, 4):
        raise ValueError(f"expected [B,8,4,H,W], got {tuple(native_rgba.shape)}")
    B = native_rgba.shape[0]
    rgb = native_rgba[:, :, :3]
    alpha = native_rgba[:, :, 3:4]
    x = rgb.reshape(B * 8, 3, *rgb.shape[-2:])
    a = alpha.reshape(B * 8, 1, *alpha.shape[-2:])
    x = F.interpolate(x, size=(global_size, global_size), mode="bilinear", align_corners=False, antialias=True)
    a = F.interpolate(a, size=(global_size, global_size), mode="bilinear", align_corners=False, antialias=True)
    x = x.view(B, 8, 3, global_size, global_size)
    a = a.view(B, 8, 1, global_size, global_size)
    x = normalize_dinov2(x)
    try:
        from mapanything.utils.geometry import rotation_matrix_to_quaternion
    except Exception as exc:
        raise RuntimeError("pinned MapAnything geometry utilities are required") from exc
    right, forward, up = orthographic_basis(x.device, torch.float32)
    R = torch.stack((right, -up, forward), dim=-1)
    quats = rotation_matrix_to_quaternion(R).to(x.device)
    views: List[Dict] = []
    for v in range(8):
        views.append({
            "img": x[:, v],
            "data_norm_type": ["dinov2"] * B,
            "idx": [v] * B,
            "instance": [f"V{v}"] * B,
            "camera_pose_quats": quats[v].view(1, 4).expand(B, -1).to(x.dtype),
            "camera_pose_trans": torch.zeros(B, 3, device=x.device, dtype=x.dtype),
            "is_metric_scale": torch.zeros(B, device=x.device, dtype=torch.bool),
        })
    return views, a


def masked_standardize(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-4) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    m = mask.to(x.dtype)
    denom = m.sum(dim=(-2, -1), keepdim=True).clamp_min(16.0)
    mean = (x * m).sum(dim=(-2, -1), keepdim=True) / denom
    var = (((x - mean) ** 2) * m).sum(dim=(-2, -1), keepdim=True) / denom
    std = var.clamp_min(eps * eps).sqrt()
    z = ((x - mean) / std).clamp(-8.0, 8.0)
    return z, mean, std
