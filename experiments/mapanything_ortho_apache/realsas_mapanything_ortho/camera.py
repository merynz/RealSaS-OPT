from __future__ import annotations
import math
from typing import Tuple
import torch
import torch.nn.functional as F


VIEW_NAMES = ("S", "SE", "E", "NE", "N", "NW", "W", "SW")


def orthographic_basis(device=None, dtype=torch.float32) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return RealSaS canonical (right, forward, up) for V0..V7."""
    rows_r, rows_f, rows_u = [], [], []
    for v in range(8):
        t = math.radians(v * 45.0)
        rows_r.append((math.cos(t), -math.sin(t), 0.0))
        rows_f.append((math.sin(t),  math.cos(t), 0.0))
        rows_u.append((0.0, 0.0, 1.0))
    return (
        torch.tensor(rows_r, device=device, dtype=dtype),
        torch.tensor(rows_f, device=device, dtype=dtype),
        torch.tensor(rows_u, device=device, dtype=dtype),
    )


def xy01_grid(height: int, width: int, device=None, dtype=torch.float32) -> torch.Tensor:
    y = (torch.arange(height, device=device, dtype=dtype) + 0.5) / float(height)
    x = (torch.arange(width, device=device, dtype=dtype) + 0.5) / float(width)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((xx, yy), dim=-1)


def ray_origins_from_xy01(xy01: torch.Tensor, view_index: torch.Tensor | int) -> torch.Tensor:
    if xy01.shape[-1] != 2:
        raise ValueError(f"xy01 must end in 2, got {tuple(xy01.shape)}")
    r, _, u = orthographic_basis(xy01.device, xy01.dtype)
    vi = torch.as_tensor(view_index, device=xy01.device, dtype=torch.long)
    right = r[vi]
    up = u[vi]
    x = xy01[..., 0] - 0.5
    y_up = 0.5 - xy01[..., 1]
    while right.ndim < xy01.ndim:
        right = right.unsqueeze(-2)
        up = up.unsqueeze(-2)
    return x.unsqueeze(-1) * right + y_up.unsqueeze(-1) * up


def dense_ray_field(batch: int, height: int, width: int, device, dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    xy = xy01_grid(height, width, device, dtype)
    r, f, u = orthographic_basis(device, dtype)
    x = xy[..., 0] - 0.5
    y_up = 0.5 - xy[..., 1]
    O = x[None, None, :, :, None] * r[:, None, None, :] + y_up[None, None, :, :, None] * u[:, None, None, :]
    Fw = f[:, None, None, :].expand(8, height, width, 3)[None]
    O = O.expand(batch, -1, -1, -1, -1).permute(0, 1, 4, 2, 3).contiguous()
    Fw = Fw.expand(batch, -1, -1, -1, -1).permute(0, 1, 4, 2, 3).contiguous()
    return O, Fw


def point_from_depth(depth: torch.Tensor) -> torch.Tensor:
    if depth.ndim != 5 or depth.shape[1] != 8 or depth.shape[2] != 1:
        raise ValueError(f"expected depth [B,8,1,H,W], got {tuple(depth.shape)}")
    O, Fw = dense_ray_field(depth.shape[0], depth.shape[-2], depth.shape[-1], depth.device, depth.dtype)
    return O + depth * Fw


def forward_depth_from_points(points: torch.Tensor, xy01: torch.Tensor) -> torch.Tensor:
    squeeze = points.ndim == 2
    if squeeze:
        points = points.unsqueeze(0)
        xy01 = xy01.unsqueeze(0)
    if points.ndim != 3 or xy01.ndim != 4 or xy01.shape[1] != 8:
        raise ValueError((tuple(points.shape), tuple(xy01.shape)))
    B, _, N, _ = xy01.shape
    if points.shape[:2] != (B, N):
        raise ValueError((tuple(points.shape), tuple(xy01.shape)))
    _, f, _ = orthographic_basis(points.device, points.dtype)
    O = torch.stack([ray_origins_from_xy01(xy01[:, v], v) for v in range(8)], dim=1)
    d = ((points[:, None] - O) * f[None, :, None]).sum(-1)
    return d[0] if squeeze else d


def normals_from_point_field(P: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if P.ndim != 5 or P.shape[1:3] != (8, 3):
        raise ValueError(f"expected [B,8,3,H,W], got {tuple(P.shape)}")
    x = P.reshape(-1, 3, P.shape[-2], P.shape[-1])
    dx = F.pad(x[..., 2:] - x[..., :-2], (1, 1, 0, 0), mode="replicate")
    dy = F.pad(x[..., 2:, :] - x[..., :-2, :], (0, 0, 1, 1), mode="replicate")
    n = F.normalize(torch.cross(dy, dx, dim=1), dim=1, eps=eps)
    return n.view(P.shape[0], 8, 3, P.shape[-2], P.shape[-1])
