from __future__ import annotations

from dataclasses import dataclass, replace
import torch

from .observation_contract_v2 import ObservationContractV2


PRODUCTION_Q_DOMAIN_AUTHORITY_V2 = "RGB_CAMERA_FULL_FRAME_LATTICE_V1"
MASK_DIAGNOSTIC_Q_DOMAIN_AUTHORITY_V2 = "MASK_CONSTRAINED_DIAGNOSTIC_V1"
UNSEALED_Q_DOMAIN_AUTHORITY_V2 = "CALLER_ANCHORS_UNSEALED_V1"


@dataclass(frozen=True)
class QCandidatePolicyV2:
    minimum_foreground_support_views: int
    foreground_padding_px: int = 0
    schema_version: str = "RealSaS.QCandidatePolicy.v2"

    def __post_init__(self) -> None:
        if not (1 <= int(self.minimum_foreground_support_views) <= 8):
            raise ValueError("minimum_foreground_support_views must be 1..8")
        if int(self.foreground_padding_px) < 0:
            raise ValueError("foreground_padding_px must be nonnegative")


@dataclass(frozen=True)
class RayHypothesisDomainV2:
    anchor_view: torch.Tensor
    anchor_grid: torch.Tensor
    depth_values: torch.Tensor
    q_points: torch.Tensor
    projected_grid: torch.Tensor
    projected_depth: torch.Tensor
    in_frame: torch.Tensor
    foreground_support: torch.Tensor
    candidate_valid: torch.Tensor
    contract_hashes: tuple[str, ...]
    candidate_policy: QCandidatePolicyV2 | None = None
    construction_authority: str = "UNSEALED_SOURCE_TEST"
    anchor_stride_px: int | None = None

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(self.depth_values.shape)


def _camera_tensors(contracts: tuple[ObservationContractV2, ...], device, dtype):
    origin = torch.tensor([[c.origin for c in x.cameras] for x in contracts], device=device, dtype=dtype)
    right = torch.tensor([[c.right for c in x.cameras] for x in contracts], device=device, dtype=dtype)
    up = torch.tensor([[c.up for c in x.cameras] for x in contracts], device=device, dtype=dtype)
    forward = torch.tensor([[c.forward for c in x.cameras] for x in contracts], device=device, dtype=dtype)
    half = torch.tensor([[c.half_extent for c in x.cameras] for x in contracts], device=device, dtype=dtype)
    return origin, right, up, forward, half


def _pad_masks(mask: torch.Tensor, pad: int) -> torch.Tensor:
    if pad == 0:
        return mask.bool()
    import torch.nn.functional as F
    x = F.max_pool2d(mask.float().reshape(-1, 1, mask.shape[-2], mask.shape[-1]), kernel_size=2 * pad + 1, stride=1, padding=pad)
    return x.reshape(*mask.shape).bool()


def _sample_foreground(masks: torch.Tensor, projected_grid: torch.Tensor) -> torch.Tensor:
    B, Q, D, V, _ = projected_grid.shape
    if masks.ndim != 4 or masks.shape[:2] != (B, V):
        raise ValueError("foreground masks must be [B,8,H,W]")
    H, W = masks.shape[-2:]
    gx = projected_grid[..., 0]
    gy = projected_grid[..., 1]
    x = torch.floor((gx + 1.0) * 0.5 * W).long()
    y = torch.floor((gy + 1.0) * 0.5 * H).long()
    inside = (x >= 0) & (x < W) & (y >= 0) & (y < H)
    safe_x = x.clamp(0, W - 1)
    safe_y = y.clamp(0, H - 1)
    b = torch.arange(B, device=masks.device)[:, None, None, None].expand(B, Q, D, V)
    v = torch.arange(V, device=masks.device)[None, None, None, :].expand(B, Q, D, V)
    value = masks[b, v, safe_y, safe_x]
    return inside & value.bool()


def build_q_domain_v2(
    contracts: tuple[ObservationContractV2, ...],
    anchor_view: torch.Tensor,
    anchor_grid: torch.Tensor,
    depth_values: torch.Tensor,
    *,
    foreground_masks: torch.Tensor | None = None,
    candidate_policy: QCandidatePolicyV2 | None = None,
) -> RayHypothesisDomainV2:
    """Build an analytic q domain from caller-supplied anchors.

    This generic constructor is suitable for source tests and deterministic Gate-0
    measurements. It is deliberately *not* production-authorized because caller
    anchors have no provenance proving that renderer masks/truth did not choose them.
    Production learner execution must use `build_production_observation_ray_lattice_v2`.
    """
    if not contracts:
        raise ValueError("contracts required")
    if anchor_view.ndim != 2 or anchor_grid.shape != (*anchor_view.shape, 2):
        raise ValueError("anchor view/grid shape mismatch")
    if depth_values.ndim == 1:
        depth_values = depth_values[None, None, :].expand(anchor_view.shape[0], anchor_view.shape[1], -1)
    elif depth_values.ndim == 2:
        depth_values = depth_values[:, None, :].expand(-1, anchor_view.shape[1], -1)
    if depth_values.ndim != 3 or depth_values.shape[:2] != anchor_view.shape:
        raise ValueError("depth_values must broadcast to [B,Q,D]")
    B, Q = anchor_view.shape
    if len(contracts) != B or ((anchor_view < 0) | (anchor_view > 7)).any():
        raise ValueError("batch/camera mismatch or invalid anchor view")
    if not torch.isfinite(depth_values).all() or not torch.isfinite(anchor_grid).all():
        raise ValueError("q domain coordinates must be finite")
    dtype = depth_values.dtype
    device = depth_values.device
    anchor_grid = anchor_grid.to(device=device, dtype=dtype)
    anchor_view = anchor_view.to(device=device, dtype=torch.long)
    origin, right, up, forward, half = _camera_tensors(contracts, device, dtype)
    bidx = torch.arange(B, device=device)[:, None].expand(B, Q)
    o = origin[bidx, anchor_view]
    r = right[bidx, anchor_view]
    u = up[bidx, anchor_view]
    f = forward[bidx, anchor_view]
    h = half[bidx, anchor_view]
    ray_origin = o + anchor_grid[..., 0, None] * h[..., None] * r - anchor_grid[..., 1, None] * h[..., None] * u
    q_points = ray_origin[:, :, None, :] + depth_values[..., None] * f[:, :, None, :]
    diff = q_points[:, :, :, None, :] - origin[:, None, None, :, :]
    gx = (diff * right[:, None, None, :, :]).sum(-1) / half[:, None, None, :]
    gy = -(diff * up[:, None, None, :, :]).sum(-1) / half[:, None, None, :]
    projected_grid = torch.stack([gx, gy], dim=-1)
    projected_depth = (diff * forward[:, None, None, :, :]).sum(-1)
    in_frame = (projected_grid.abs() <= 1.0).all(dim=-1)
    if foreground_masks is None:
        foreground_support = in_frame.clone()
        candidate_valid = in_frame.any(dim=-1)
        if candidate_policy is not None:
            raise ValueError("candidate policy requires foreground masks")
        authority = UNSEALED_Q_DOMAIN_AUTHORITY_V2
    else:
        if candidate_policy is None:
            raise ValueError("foreground-constrained q domain requires explicit candidate policy")
        masks = _pad_masks(foreground_masks.to(device=device).bool(), int(candidate_policy.foreground_padding_px))
        foreground_support = _sample_foreground(masks, projected_grid) & in_frame
        candidate_valid = foreground_support.sum(dim=-1) >= int(candidate_policy.minimum_foreground_support_views)
        authority = MASK_DIAGNOSTIC_Q_DOMAIN_AUTHORITY_V2
    return RayHypothesisDomainV2(
        anchor_view,
        anchor_grid,
        depth_values,
        q_points,
        projected_grid,
        projected_depth,
        in_frame,
        foreground_support,
        candidate_valid,
        tuple(c.contract_hash for c in contracts),
        candidate_policy,
        authority,
        None,
    )


def pixel_center_grid_v2(x: torch.Tensor, y: torch.Tensor, resolution: int = 1024) -> torch.Tensor:
    gx = 2.0 * (x.to(torch.float64) + 0.5) / float(resolution) - 1.0
    gy = 2.0 * (y.to(torch.float64) + 0.5) / float(resolution) - 1.0
    return torch.stack([gx, gy], dim=-1)


def _full_frame_anchor_grid_v2(*, stride_px: int, device, dtype) -> torch.Tensor:
    if int(stride_px) <= 0:
        raise ValueError("anchor_stride_px must be positive")
    resolution = 1024
    ys = torch.arange(int(stride_px) // 2, resolution, int(stride_px), device=device)
    xs = torch.arange(int(stride_px) // 2, resolution, int(stride_px), device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    if xx.numel() == 0:
        raise ValueError("anchor stride produced empty full-frame lattice")
    return pixel_center_grid_v2(xx.reshape(-1), yy.reshape(-1), resolution).to(device=device, dtype=dtype)


def build_production_observation_ray_lattice_v2(
    contracts: tuple[ObservationContractV2, ...],
    *,
    anchor_view_index: int,
    anchor_stride_px: int,
    depth_values: torch.Tensor,
) -> RayHypothesisDomainV2:
    """Build the only Q-domain authorized for learned production execution.

    Anchors cover the full native frame on a deterministic camera-only pixel lattice.
    No alpha, foreground mask, raster authority, teacher geometry or image-derived
    segmentation may create/delete rays or candidates. Candidate validity is purely
    analytic in-frame validity under exact cameras.
    """
    if not contracts or anchor_view_index not in range(8):
        raise ValueError("production q domain requires contracts and anchor view 0..7")
    B = len(contracts)
    if depth_values.ndim not in (1, 2, 3) or not torch.isfinite(depth_values).all():
        raise ValueError("finite depth_values required")
    grid = _full_frame_anchor_grid_v2(stride_px=int(anchor_stride_px), device=depth_values.device, dtype=depth_values.dtype)
    anchor_grid = grid[None].expand(B, -1, -1).clone()
    anchor_view = torch.full((B, grid.shape[0]), int(anchor_view_index), dtype=torch.long, device=depth_values.device)
    domain = build_q_domain_v2(contracts, anchor_view, anchor_grid, depth_values)
    domain = replace(
        domain,
        construction_authority=PRODUCTION_Q_DOMAIN_AUTHORITY_V2,
        anchor_stride_px=int(anchor_stride_px),
    )
    validate_production_observation_domain_v2(domain)
    return domain


def validate_production_observation_domain_v2(domain: RayHypothesisDomainV2) -> None:
    """Fail closed if a mask/externally-selected anchor domain reaches the learner."""
    if domain.construction_authority != PRODUCTION_Q_DOMAIN_AUTHORITY_V2:
        raise ValueError(f"IRIS production learner forbids unsealed/mask Q domain:{domain.construction_authority}")
    if domain.candidate_policy is not None:
        raise ValueError("IRIS production learner forbids foreground candidate policy")
    if domain.anchor_stride_px is None or int(domain.anchor_stride_px) <= 0:
        raise ValueError("IRIS production Q domain missing full-frame anchor stride")
    if not torch.equal(domain.foreground_support.bool(), domain.in_frame.bool()):
        raise ValueError("IRIS production Q domain foreground support must equal analytic in-frame support")
    if not torch.equal(domain.candidate_valid.bool(), domain.in_frame.any(dim=-1).bool()):
        raise ValueError("IRIS production candidate validity must be camera-only in-frame validity")
    if domain.anchor_view.ndim != 2 or domain.anchor_grid.shape != (*domain.anchor_view.shape, 2):
        raise ValueError("IRIS production anchor shape drift")
    if ((domain.anchor_view < 0) | (domain.anchor_view > 7)).any():
        raise ValueError("IRIS production anchor view drift")
    first_view = domain.anchor_view[:, :1]
    if not torch.equal(domain.anchor_view, first_view.expand_as(domain.anchor_view)):
        raise ValueError("IRIS production lattice requires one fixed analytic anchor view")
    expected = _full_frame_anchor_grid_v2(
        stride_px=int(domain.anchor_stride_px),
        device=domain.anchor_grid.device,
        dtype=domain.anchor_grid.dtype,
    )
    if domain.anchor_grid.shape[1] != expected.shape[0]:
        raise ValueError("IRIS production anchor count is not full-frame lattice")
    expected_batched = expected[None].expand(domain.anchor_grid.shape[0], -1, -1)
    if not torch.equal(domain.anchor_grid, expected_batched):
        raise ValueError("IRIS production anchors were not generated by full-frame camera lattice")


def build_canonical_ray_lattice_v2(
    contracts: tuple[ObservationContractV2, ...],
    foreground_masks: torch.Tensor,
    *,
    anchor_view_index: int,
    anchor_stride_px: int,
    depth_values: torch.Tensor,
    candidate_policy: QCandidatePolicyV2,
) -> RayHypothesisDomainV2:
    """Legacy/Gate-0 mask-constrained lattice. Diagnostic only; never learner input."""
    if anchor_view_index not in range(8) or anchor_stride_px <= 0:
        raise ValueError("invalid canonical ray lattice configuration")
    B, V, H, W = foreground_masks.shape
    if V != 8 or H != 1024 or W != 1024 or len(contracts) != B:
        raise ValueError("canonical ray lattice requires [B,8,1024,1024] masks")
    all_grids = []
    for b in range(B):
        mask = foreground_masks[b, anchor_view_index].bool()
        ys = torch.arange(anchor_stride_px // 2, H, anchor_stride_px, device=mask.device)
        xs = torch.arange(anchor_stride_px // 2, W, anchor_stride_px, device=mask.device)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        keep = mask[yy, xx]
        if not keep.any():
            raise ValueError(f"anchor foreground produced no q rays:batch={b}")
        all_grids.append(pixel_center_grid_v2(xx[keep], yy[keep], W).to(torch.float32))
    q = max(len(g) for g in all_grids)
    if any(len(g) != q for g in all_grids):
        raise ValueError("batched canonical ray lattice requires equal anchor count; batch assets separately")
    anchor_grid = torch.stack(all_grids, dim=0).to(depth_values.device)
    anchor_view = torch.full((B, q), int(anchor_view_index), dtype=torch.long, device=depth_values.device)
    domain = build_q_domain_v2(
        contracts,
        anchor_view,
        anchor_grid,
        depth_values,
        foreground_masks=foreground_masks.to(depth_values.device),
        candidate_policy=candidate_policy,
    )
    return replace(domain, anchor_stride_px=int(anchor_stride_px))
