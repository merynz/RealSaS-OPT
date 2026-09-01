from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class OrthoCameraBatch:
    """Exact RealSaS Mode-G orthographic camera authority.

    Shapes:
      forward/right/screen_up: [B,V,3]
      half_extent: [B,V] or [B,1]
      yaw_deg: optional [B,V]

    Object-frame origin is the orthographic image-plane origin used by the frozen
    RealSaS turnaround contract.  Projection is therefore exactly the Gate-0
    algebra: gx=<P,R>/h, gy=-<P,U>/h, d=<P,F>.
    """

    forward: torch.Tensor
    right: torch.Tensor
    screen_up: torch.Tensor
    half_extent: torch.Tensor
    yaw_deg: torch.Tensor | None = None

    def validate(self, *, expected_views: int = 8, atol: float = 1e-5) -> None:
        f, r, u = self.forward, self.right, self.screen_up
        if f.ndim != 3 or f.shape[-1] != 3:
            raise ValueError(f"CAMERA_FORWARD_SHAPE:{tuple(f.shape)}")
        if r.shape != f.shape or u.shape != f.shape:
            raise ValueError("CAMERA_BASIS_SHAPE_MISMATCH")
        b, v, _ = f.shape
        if v != expected_views:
            raise ValueError(f"CAMERA_VIEW_COUNT:{v}!={expected_views}")
        h = self.half_extent
        if h.ndim == 1:
            h = h[:, None]
        if h.ndim != 2 or h.shape[0] != b or h.shape[1] not in (1, v):
            raise ValueError(f"CAMERA_HALF_EXTENT_SHAPE:{tuple(self.half_extent.shape)}")
        if not torch.isfinite(f).all() or not torch.isfinite(r).all() or not torch.isfinite(u).all():
            raise ValueError("CAMERA_NONFINITE_BASIS")
        if not torch.isfinite(h).all() or not torch.all(h > 0):
            raise ValueError("CAMERA_HALF_EXTENT_INVALID")

        def _unit(x: torch.Tensor):
            return torch.linalg.vector_norm(x.float(), dim=-1)

        for name, x in (("FORWARD", f), ("RIGHT", r), ("SCREEN_UP", u)):
            if torch.max(torch.abs(_unit(x) - 1.0)).item() > atol:
                raise ValueError(f"CAMERA_{name}_NOT_UNIT")
        # Pairwise orthogonality is enough here and avoids constructing BxVx3x3.
        if torch.max(torch.abs((f * r).sum(-1))).item() > atol:
            raise ValueError("CAMERA_FORWARD_RIGHT_NOT_ORTHOGONAL")
        if torch.max(torch.abs((f * u).sum(-1))).item() > atol:
            raise ValueError("CAMERA_FORWARD_UP_NOT_ORTHOGONAL")
        if torch.max(torch.abs((r * u).sum(-1))).item() > atol:
            raise ValueError("CAMERA_RIGHT_UP_NOT_ORTHOGONAL")
        if self.yaw_deg is not None:
            y = self.yaw_deg
            if y.ndim == 1:
                y = y[None].expand(b, -1)
            if tuple(y.shape) != (b, v) or not torch.isfinite(y).all():
                raise ValueError("CAMERA_YAW_INVALID")

    def expanded_half_extent(self) -> torch.Tensor:
        b, v, _ = self.forward.shape
        h = self.half_extent
        if h.ndim == 1:
            h = h[:, None]
        if h.shape[1] == 1:
            h = h.expand(b, v)
        return h

    def to(self, device=None, dtype=None) -> "OrthoCameraBatch":
        def cv(x):
            if x is None:
                return None
            if dtype is None or not x.is_floating_point():
                return x.to(device=device)
            return x.to(device=device, dtype=dtype)

        return OrthoCameraBatch(
            forward=cv(self.forward),
            right=cv(self.right),
            screen_up=cv(self.screen_up),
            half_extent=cv(self.half_extent),
            yaw_deg=cv(self.yaw_deg),
        )


def canonical_orbit_cameras(
    *,
    batch_size: int = 1,
    half_extent: float = 0.54,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> OrthoCameraBatch:
    """Construct the frozen S,SE,E,NE,N,NW,W,SW Z-up camera family.

    This is a convenience for synthetic tests. Final product inference must load
    exact camera authority supplied with the observations rather than silently
    replacing it with this constructor.
    """
    yaw = torch.arange(0, 360, 45, device=device, dtype=dtype)
    t = torch.deg2rad(yaw)
    z = torch.zeros_like(t)
    f = torch.stack([torch.sin(t), torch.cos(t), z], dim=-1)
    r = torch.stack([torch.cos(t), -torch.sin(t), z], dim=-1)
    u = torch.stack([z, z, torch.ones_like(t)], dim=-1)
    f = f[None].expand(batch_size, -1, -1).contiguous()
    r = r[None].expand(batch_size, -1, -1).contiguous()
    u = u[None].expand(batch_size, -1, -1).contiguous()
    h = torch.full((batch_size, 8), float(half_extent), device=device, dtype=dtype)
    y = yaw[None].expand(batch_size, -1).contiguous()
    c = OrthoCameraBatch(f, r, u, h, y)
    c.validate()
    return c


def project_world(points: torch.Tensor, cameras: OrthoCameraBatch) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact world -> normalized-grid + forward-depth projection.

    points: [B,Q,3] or [B,V,Q,3]
    returns:
      grid_xy: [B,V,Q,2], align_corners=False normalized image coordinates
      depth:   [B,V,Q]
    """
    cameras.validate(expected_views=cameras.forward.shape[1])
    if points.ndim == 3:
        if points.shape[-1] != 3 or points.shape[0] != cameras.forward.shape[0]:
            raise ValueError(f"POINT_SHAPE:{tuple(points.shape)}")
        p = points[:, None, :, :]
    elif points.ndim == 4:
        if points.shape[-1] != 3 or points.shape[:2] != cameras.forward.shape[:2]:
            raise ValueError(f"POINT_VIEW_SHAPE:{tuple(points.shape)}")
        p = points
    else:
        raise ValueError(f"POINT_RANK:{points.ndim}")
    f = cameras.forward[:, :, None, :].to(p.dtype)
    r = cameras.right[:, :, None, :].to(p.dtype)
    u = cameras.screen_up[:, :, None, :].to(p.dtype)
    h = cameras.expanded_half_extent()[:, :, None].to(p.dtype)
    gx = (p * r).sum(-1) / h
    gy = -(p * u).sum(-1) / h
    depth = (p * f).sum(-1)
    return torch.stack([gx, gy], dim=-1), depth


def ray_origin_from_grid(grid_xy: torch.Tensor, cameras: OrthoCameraBatch) -> torch.Tensor:
    """Return object-frame orthographic ray origins on the d=0 image plane.

    grid_xy: [B,V,Q,2] -> [B,V,Q,3]
    """
    if grid_xy.ndim != 4 or grid_xy.shape[-1] != 2:
        raise ValueError(f"GRID_SHAPE:{tuple(grid_xy.shape)}")
    if grid_xy.shape[:2] != cameras.forward.shape[:2]:
        raise ValueError("GRID_CAMERA_BATCH_VIEW_MISMATCH")
    h = cameras.expanded_half_extent()[:, :, None, None].to(grid_xy.dtype)
    x = grid_xy[..., 0:1] * h
    up = -grid_xy[..., 1:2] * h
    r = cameras.right[:, :, None, :].to(grid_xy.dtype)
    u = cameras.screen_up[:, :, None, :].to(grid_xy.dtype)
    return x * r + up * u


def backproject_grid_depth(grid_xy: torch.Tensor, depth: torch.Tensor, cameras: OrthoCameraBatch) -> torch.Tensor:
    """Exact analytic P = O + dF reconstruction.

    grid_xy: [B,V,Q,2]
    depth:   [B,V,Q]
    returns: [B,V,Q,3]
    """
    if depth.shape != grid_xy.shape[:-1]:
        raise ValueError("BACKPROJECT_DEPTH_SHAPE_MISMATCH")
    o = ray_origin_from_grid(grid_xy, cameras)
    f = cameras.forward[:, :, None, :].to(depth.dtype)
    return o + depth[..., None] * f


def normalized_grid_inside(grid_xy: torch.Tensor, *, margin: float = 0.0) -> torch.Tensor:
    lim = 1.0 - float(margin)
    return (grid_xy[..., 0].abs() <= lim) & (grid_xy[..., 1].abs() <= lim)


def normalized_grid_to_pixel(grid_xy: torch.Tensor, resolution: int) -> torch.Tensor:
    """Gate-0-compatible normalized-grid -> integer pixel bins.

    The mapping matches align_corners=False pixel-cell boundaries and the original
    deterministic Gate-0 implementation: floor((g+1)/2 * resolution).
    """
    r = int(resolution)
    if r <= 0:
        raise ValueError("RESOLUTION_NONPOSITIVE")
    pix = torch.floor((grid_xy + 1.0) * (0.5 * r)).to(torch.long)
    return pix
