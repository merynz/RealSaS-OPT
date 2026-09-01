from __future__ import annotations

from dataclasses import dataclass

import torch

from realsas_compiler_core.types import RiggingSurfaceIR


@dataclass(frozen=True)
class GeppettoConditioningConfig:
    local_k: int = 16
    raster_reference_resolution: float = 1024.0


@dataclass
class GeppettoConditioningBatch:
    points_world: torch.Tensor          # [1,N,3]
    points_norm: torch.Tensor           # [1,N,3]
    features: torch.Tensor              # [1,N,F]
    center: torch.Tensor                # [1,1,3]
    scale: torch.Tensor                 # [1,1,1]
    surface_ids: tuple[str, ...]
    knn_index: torch.Tensor             # [1,N,K]


def _normalize(points: torch.Tensor):
    pmin = points.amin(dim=1, keepdim=True)
    pmax = points.amax(dim=1, keepdim=True)
    center = 0.5 * (pmin + pmax)
    scale = (pmax - pmin).amax(dim=-1, keepdim=True).clamp_min(1e-6)
    return (points - center) / scale, center, scale


def _knn(points_norm: torch.Tensor, k: int) -> torch.Tensor:
    b, n, _ = points_norm.shape
    if n < 2:
        return torch.zeros((b, n, 1), dtype=torch.long, device=points_norm.device)
    kk = min(max(1, int(k)), n - 1)
    d = torch.cdist(points_norm, points_norm)
    eye = torch.eye(n, dtype=torch.bool, device=points_norm.device).unsqueeze(0)
    d = d.masked_fill(eye, float("inf"))
    return torch.topk(d, k=kk, largest=False, dim=-1).indices


def _local_sign_invariant_geometry(points_norm: torch.Tensor, knn: torch.Tensor):
    """Deterministic local scale/shape descriptors without inventing normal authority.

    Eigenvalues of the local covariance and neighbor radius are sign/orientation
    invariant and are legal deterministic functions of already-admitted P.  The
    PCA eigenvector is deliberately *not* returned: an unoriented PCA normal is
    not promoted as qualified N_d by this adapter.
    """
    b, n, _ = points_norm.shape
    k = knn.shape[-1]
    batch = torch.arange(b, device=points_norm.device)[:, None, None].expand(b, n, k)
    neigh = points_norm[batch, knn]
    rel = neigh - points_norm[:, :, None, :]
    radius = torch.linalg.norm(rel, dim=-1).mean(dim=-1, keepdim=True)
    cov = torch.einsum("bnki,bnkj->bnij", rel, rel) / float(max(1, k))
    evals = torch.linalg.eigvalsh(cov).clamp_min(0.0)
    spectrum = evals / evals.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return spectrum, radius


def _qualified_normal_features(surface: RiggingSurfaceIR, device: torch.device, dtype: torch.dtype):
    rows = []
    for node in surface.surface_nodes:
        n = node.derived_normal
        if n is None:
            rows.append([0.0, 0.0, 0.0, 0.0])
        else:
            t = torch.tensor(n, dtype=torch.float64)
            norm = float(torch.linalg.norm(t).item())
            if not torch.isfinite(t).all() or norm <= 1e-12:
                raise ValueError(f"GEPPETTO_INVALID_QUALIFIED_NORMAL:{node.surface_id}")
            rows.append([float(t[0] / norm), float(t[1] / norm), float(t[2] / norm), 1.0])
    return torch.tensor(rows, dtype=dtype, device=device)[None]


def _support_features(surface: RiggingSurfaceIR, device: torch.device, dtype: torch.dtype, rr: float):
    rows = []
    rr = float(rr)
    if rr <= 0:
        raise ValueError("GEPPETTO_BAD_RASTER_REFERENCE_RESOLUTION")
    for node in surface.surface_nodes:
        support = [0.0] * 8
        for v in node.support_views:
            if 0 <= int(v) < 8:
                support[int(v)] = 1.0
        bindings = list(node.raster_bindings)
        if bindings:
            mx = sum(float(xy[0]) for _, xy in bindings) / len(bindings)
            my = sum(float(xy[1]) for _, xy in bindings) / len(bindings)
            sx = sum((float(xy[0]) - mx) ** 2 for _, xy in bindings) / len(bindings)
            sy = sum((float(xy[1]) - my) ** 2 for _, xy in bindings) / len(bindings)
            raster = [mx / rr, my / rr, sx ** 0.5 / rr, sy ** 0.5 / rr]
        else:
            raster = [0.0, 0.0, 0.0, 0.0]
        rows.append(support + [sum(support) / 8.0] + raster)
    return torch.tensor(rows, dtype=dtype, device=device)[None]


def build_geppetto_conditioning(
    surface: RiggingSurfaceIR,
    *,
    device: torch.device | str = "cpu",
    cfg: GeppettoConditioningConfig = GeppettoConditioningConfig(),
) -> GeppettoConditioningBatch:
    if not surface.surface_nodes:
        raise ValueError("GEPPETTO_EMPTY_SURFACE")
    ids = tuple(n.surface_id for n in surface.surface_nodes)
    if len(set(ids)) != len(ids):
        raise ValueError("GEPPETTO_DUPLICATE_SURFACE_ID")
    device = torch.device(device)
    p = torch.tensor([n.P for n in surface.surface_nodes], dtype=torch.float32, device=device)[None]
    if not torch.isfinite(p).all():
        raise ValueError("GEPPETTO_NONFINITE_SURFACE")
    pn, center, scale = _normalize(p)
    knn = _knn(pn, cfg.local_k)
    spectrum, radius = _local_sign_invariant_geometry(pn, knn)
    qualified_normal = _qualified_normal_features(surface, device, p.dtype)
    support = _support_features(surface, device, p.dtype, cfg.raster_reference_resolution)
    # Product-legal deterministic feature contract:
    # P_norm(3), local covariance spectrum(3), radius(1),
    # qualified derived normal xyz + valid(4), support mask(8),
    # support fraction(1), raster mean/std(4) = 24D.
    feat = torch.cat([pn, spectrum, radius, qualified_normal, support], dim=-1)
    if feat.shape[-1] != 24:
        raise AssertionError(feat.shape)
    if not torch.isfinite(feat).all():
        raise ValueError("GEPPETTO_NONFINITE_CONDITIONING")
    return GeppettoConditioningBatch(p, pn, feat, center, scale, ids, knn)
