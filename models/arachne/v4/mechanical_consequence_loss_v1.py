from __future__ import annotations

"""Differentiable product-space mechanical consequence loss for Arachne.

A fixed surface->candidate transfer matrix maps Arachne surface weights onto the
exact compiler candidate. The loss then evaluates the same physical classes as
G3 (edge stretch, area change, local condition) under deterministic probe LBS.

Compiler G3 remains the hard authority. This module is only training pressure.
"""

import torch


def _lbs(rest_vertices: torch.Tensor, weights: torch.Tensor, transforms: torch.Tensor) -> torch.Tensor:
    # rest [B,V,3], weights [B,V,J], transforms [B,P,J,4,4]
    ones = torch.ones((*rest_vertices.shape[:2], 1), device=rest_vertices.device, dtype=rest_vertices.dtype)
    hom = torch.cat([rest_vertices, ones], dim=-1)
    moved = torch.einsum("bpjac,bvc->bpjva", transforms, hom)[..., :3]
    return torch.einsum("bvj,bpjva->bpva", weights, moved)


def _triangle_metrics(rest: torch.Tensor, posed: torch.Tensor, faces: torch.Tensor):
    # rest [B,V,3], posed [B,P,V,3], faces [F,3]
    f = faces.long()
    r = rest[:, f]                    # [B,F,3,3]
    p = posed[:, :, f]                # [B,P,F,3,3]

    re01 = torch.linalg.norm(r[:, :, 1] - r[:, :, 0], dim=-1)
    re12 = torch.linalg.norm(r[:, :, 2] - r[:, :, 1], dim=-1)
    re20 = torch.linalg.norm(r[:, :, 0] - r[:, :, 2], dim=-1)
    pe01 = torch.linalg.norm(p[:, :, :, 1] - p[:, :, :, 0], dim=-1)
    pe12 = torch.linalg.norm(p[:, :, :, 2] - p[:, :, :, 1], dim=-1)
    pe20 = torch.linalg.norm(p[:, :, :, 0] - p[:, :, :, 2], dim=-1)
    edge = torch.stack(
        [
            pe01 / re01[:, None].clamp_min(1e-12),
            pe12 / re12[:, None].clamp_min(1e-12),
            pe20 / re20[:, None].clamp_min(1e-12),
        ],
        dim=-1,
    )

    r1 = r[:, :, 1] - r[:, :, 0]
    r2 = r[:, :, 2] - r[:, :, 0]
    l1 = torch.linalg.norm(r1, dim=-1).clamp_min(1e-12)
    u = r1 / l1[..., None]
    x2 = (r2 * u).sum(-1)
    perp = r2 - x2[..., None] * u
    y2 = torch.linalg.norm(perp, dim=-1).clamp_min(1e-12)
    inv = torch.zeros((*l1.shape, 2, 2), device=rest.device, dtype=rest.dtype)
    inv[..., 0, 0] = 1.0 / l1
    inv[..., 0, 1] = -x2 / (l1 * y2)
    inv[..., 1, 1] = 1.0 / y2

    pedges = torch.stack(
        [p[:, :, :, 1] - p[:, :, :, 0], p[:, :, :, 2] - p[:, :, :, 0]],
        dim=-1,
    )                                 # [B,P,F,3,2]
    Fm = torch.einsum("bpfci,bfij->bpfcj", pedges, inv)
    s = torch.linalg.svdvals(Fm)
    smax = s[..., 0]
    smin = s[..., 1].clamp_min(1e-12)
    area = smax * smin
    condition = smax / smin
    return edge, area, condition


def mechanical_consequence_loss_v1(
    surface_weights: torch.Tensor,
    *,
    surface_to_candidate: torch.Tensor,
    candidate_rest_vertices: torch.Tensor,
    candidate_faces: torch.Tensor,
    probe_transforms: torch.Tensor,
    base_surface_weights: torch.Tensor | None = None,
    teacher_surface_weights: torch.Tensor | None = None,
    teacher_valid_mask: torch.Tensor | None = None,
    max_edge_ratio: float = 4.0,
    min_area_ratio: float = 0.05,
    max_area_ratio: float = 20.0,
    max_condition_number: float = 16.0,
    trust_weight: float = 0.25,
    teacher_weight: float = 1.0,
    mechanical_weight: float = 1.0,
):
    # transfer [B,V,N] is deterministic compiler support transfer and rows sum to 1.
    vw = torch.einsum("bvn,bnj->bvj", surface_to_candidate.float(), surface_weights.float())
    vw = vw / vw.sum(-1, keepdim=True).clamp_min(1e-12)
    posed = _lbs(candidate_rest_vertices.float(), vw, probe_transforms.float())
    edge, area, cond = _triangle_metrics(candidate_rest_vertices.float(), posed, candidate_faces)

    edge_pen = torch.relu(torch.log(edge.clamp_min(1e-12) / float(max_edge_ratio))).square().mean()
    area_hi = torch.relu(torch.log(area.clamp_min(1e-12) / float(max_area_ratio))).square().mean()
    area_lo = torch.relu(torch.log(float(min_area_ratio) / area.clamp_min(1e-12))).square().mean()
    cond_pen = torch.relu(torch.log(cond.clamp_min(1e-12) / float(max_condition_number))).square().mean()
    mechanical = edge_pen + area_hi + area_lo + cond_pen

    trust = surface_weights.new_zeros(())
    if base_surface_weights is not None:
        trust = (surface_weights - base_surface_weights.to(surface_weights)).abs().sum(-1).mean()

    supervised = surface_weights.new_zeros(())
    if teacher_surface_weights is not None:
        row = (surface_weights - teacher_surface_weights.to(surface_weights)).abs().sum(-1)
        if teacher_valid_mask is not None:
            m = teacher_valid_mask.bool()
            supervised = row[m].mean() if bool(m.any()) else row.new_zeros(())
        else:
            supervised = row.mean()

    total = mechanical_weight * mechanical + trust_weight * trust + teacher_weight * supervised
    return {
        "total": total,
        "mechanical": mechanical,
        "edge": edge_pen,
        "area_hi": area_hi,
        "area_lo": area_lo,
        "condition": cond_pen,
        "trust_row_l1": trust,
        "teacher_row_l1": supervised,
        "maximum_edge_ratio": edge.detach().max(),
        "maximum_area_ratio": area.detach().max(),
        "minimum_area_ratio": area.detach().min(),
        "maximum_condition_number": cond.detach().max(),
    }


__all__ = ["mechanical_consequence_loss_v1"]
