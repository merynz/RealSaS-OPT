from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .camera import OrthoCameraBatch, normalized_grid_inside, project_world


@dataclass(frozen=True)
class QLattice:
    """Canonical isotropic object-frame candidate lattice.

    points_world is flattened in Z,Y,X-major meshgrid order. volume_shape is
    (Z,Y,X), so a flattened q feature tensor can be scattered/reshaped without
    inventing another spatial convention.
    """

    points_world: torch.Tensor      # [Q,3] XYZ
    axis: torch.Tensor              # [L]
    spacing: float
    half_extent: float
    volume_shape: tuple[int, int, int]

    @property
    def candidate_count(self) -> int:
        return int(self.points_world.shape[0])


def build_canonical_q_lattice(
    *,
    half_extent: float,
    spacing: float,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> QLattice:
    h = float(half_extent)
    s = float(spacing)
    if not (h > 0.0 and s > 0.0):
        raise ValueError("Q_LATTICE_NONPOSITIVE_H_OR_SPACING")
    # Exact Gate-0 center convention: arange(-h+s/2, h, s).
    axis = torch.arange(-h + 0.5 * s, h, s, device=device, dtype=dtype)
    if axis.numel() < 2:
        raise ValueError("Q_LATTICE_TOO_COARSE")
    zz, yy, xx = torch.meshgrid(axis, axis, axis, indexing="ij")
    p = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3).contiguous()
    n = int(axis.numel())
    return QLattice(p, axis, s, h, (n, n, n))


def pad_binary_support(alpha_support: torch.Tensor, padding_px: int) -> torch.Tensor:
    """Conservative binary silhouette dilation with zero outside image.

    alpha_support: [B,V,1,H,W] bool/0-1
    """
    if alpha_support.ndim != 5 or alpha_support.shape[2] != 1:
        raise ValueError(f"ALPHA_SUPPORT_SHAPE:{tuple(alpha_support.shape)}")
    p = int(padding_px)
    if p < 0:
        raise ValueError("NEGATIVE_HULL_PADDING")
    x = alpha_support.float()
    if p == 0:
        return x > 0.5
    b, v, _, h, w = x.shape
    y = F.max_pool2d(x.reshape(b * v, 1, h, w), kernel_size=2 * p + 1, stride=1, padding=p)
    return y.reshape(b, v, 1, h, w) > 0.5


def visual_hull_active_mask(
    lattice: QLattice,
    cameras: OrthoCameraBatch,
    alpha_support: torch.Tensor,
    *,
    padding_px: int,
    chunk_q: int = 131072,
) -> torch.Tensor:
    """Return conservative 8-view visual-hull membership [B,Q].

    This is a deterministic candidate-domain constraint, never learned occupancy
    truth. A q survives only when its exact projection lies in every padded
    authoritative foreground silhouette.
    """
    cameras.validate(expected_views=alpha_support.shape[1])
    if alpha_support.shape[0] != cameras.forward.shape[0]:
        raise ValueError("HULL_BATCH_MISMATCH")
    if alpha_support.shape[1] != cameras.forward.shape[1]:
        raise ValueError("HULL_VIEW_MISMATCH")
    masks = pad_binary_support(alpha_support, padding_px)
    b, v, _, h, w = masks.shape
    q = lattice.points_world.to(device=cameras.forward.device, dtype=cameras.forward.dtype)
    out = torch.empty((b, q.shape[0]), dtype=torch.bool, device=q.device)
    cq = max(1, int(chunk_q))
    for start in range(0, q.shape[0], cq):
        stop = min(q.shape[0], start + cq)
        p = q[start:stop][None].expand(b, -1, -1)
        grid, _ = project_world(p, cameras)
        inside = normalized_grid_inside(grid)
        # grid_sample expects [N,Hout,Wout,2]. We query Q points as Hout=Q,Wout=1.
        g = grid.reshape(b * v, stop - start, 1, 2)
        m = masks.float().reshape(b * v, 1, h, w)
        sampled = F.grid_sample(
            m,
            g,
            mode="nearest",
            padding_mode="zeros",
            align_corners=False,
        ).reshape(b, v, stop - start)
        keep = inside & (sampled > 0.5)
        out[:, start:stop] = keep.all(dim=1)
    return out


def active_q_points(lattice: QLattice, active_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-batch convenience used by the fitted demo and synthetic gates.

    Returns active flattened lattice indices and active XYZ points.  Full generic
    batching remains represented by visual_hull_active_mask; later batching may
    pack ragged active sets without changing the product contract.
    """
    if active_mask.ndim != 2 or active_mask.shape[0] != 1:
        raise ValueError("ACTIVE_Q_POINTS_CURRENTLY_REQUIRES_BATCH1_RAGGED_PACKING")
    idx = torch.nonzero(active_mask[0], as_tuple=False).squeeze(-1)
    if idx.numel() == 0:
        raise RuntimeError("VISUAL_HULL_EMPTY")
    return idx, lattice.points_world.to(active_mask.device)[idx]


def scatter_active_to_volume(
    lattice: QLattice,
    active_indices: torch.Tensor,
    active_features: torch.Tensor,
    *,
    fill_value: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scatter sparse active q features to dense isotropic Z,Y,X volume.

    active_features: [Q_active,C]
    returns volume [1,C,Z,Y,X], active volume mask [1,1,Z,Y,X].
    """
    if active_features.ndim != 2 or active_features.shape[0] != active_indices.numel():
        raise ValueError("ACTIVE_FEATURE_SHAPE_MISMATCH")
    q_all = lattice.candidate_count
    c = active_features.shape[1]
    dense = active_features.new_full((q_all, c), float(fill_value))
    dense[active_indices] = active_features
    mask = torch.zeros(q_all, dtype=torch.bool, device=active_features.device)
    mask[active_indices] = True
    z, y, x = lattice.volume_shape
    vol = dense.reshape(z, y, x, c).permute(3, 0, 1, 2).unsqueeze(0).contiguous()
    m = mask.reshape(z, y, x).unsqueeze(0).unsqueeze(0)
    return vol, m
