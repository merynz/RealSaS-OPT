from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .contracts_v1 import TESSAMechanicalRewardPolicyV1


@dataclass
class TESSATriangleMetricsV1:
    jacobian: torch.Tensor          # [B,F,3,2]
    singular_values: torch.Tensor   # [B,F,2]
    area_ratio: torch.Tensor        # [B,F]
    condition_number: torch.Tensor  # [B,F]
    max_edge_ratio: torch.Tensor    # [B,F]
    normal: torch.Tensor            # [B,F,3]


def _batched_vertices(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 2:
        x = x.unsqueeze(0)
    if x.ndim != 3 or x.shape[-1] != 3:
        raise ValueError("TESSA_MECHANICAL_VERTEX_SHAPE_INVALID")
    return x


def triangle_deformation_metrics_v1(
    rest_vertices: torch.Tensor,
    posed_vertices: torch.Tensor,
    faces: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> TESSATriangleMetricsV1:
    """Differentiable per-triangle deformation diagnostics.

    This mirrors the consequence class used by Compiler G3/G3B but is only an
    optimization surrogate.  Hard Compiler proof remains product authority.
    """
    rest = _batched_vertices(rest_vertices)
    posed = _batched_vertices(posed_vertices)
    if rest.shape != posed.shape:
        raise ValueError("TESSA_MECHANICAL_REST_POSED_SHAPE_MISMATCH")
    if faces.ndim != 2 or faces.shape[-1] != 3:
        raise ValueError("TESSA_MECHANICAL_FACE_SHAPE_INVALID")
    faces = faces.long()
    if faces.numel() == 0:
        raise ValueError("TESSA_MECHANICAL_FACE_SET_EMPTY")
    if int(faces.min()) < 0 or int(faces.max()) >= rest.shape[1]:
        raise ValueError("TESSA_MECHANICAL_FACE_INDEX_INVALID")

    r = rest[:, faces]   # B,F,3,3
    p = posed[:, faces]
    r1 = r[:, :, 1] - r[:, :, 0]
    r2 = r[:, :, 2] - r[:, :, 0]
    p1 = p[:, :, 1] - p[:, :, 0]
    p2 = p[:, :, 2] - p[:, :, 0]

    l1 = torch.linalg.vector_norm(r1, dim=-1).clamp_min(eps)
    u = r1 / l1[..., None]
    x2 = (r2 * u).sum(dim=-1)
    perp = r2 - x2[..., None] * u
    y2 = torch.linalg.vector_norm(perp, dim=-1).clamp_min(eps)

    inv = torch.zeros((*l1.shape, 2, 2), dtype=rest.dtype, device=rest.device)
    inv[..., 0, 0] = 1.0 / l1
    inv[..., 0, 1] = -x2 / (l1 * y2)
    inv[..., 1, 1] = 1.0 / y2
    posed_edges = torch.stack((p1, p2), dim=-1)  # B,F,3,2
    jacobian = posed_edges @ inv

    singular = torch.linalg.svdvals(jacobian)
    smax = singular[..., 0]
    smin = singular[..., 1].clamp_min(eps)
    area_ratio = smax * smin
    condition = smax / smin

    rest_edges = torch.stack(
        (
            torch.linalg.vector_norm(r[:, :, 1] - r[:, :, 0], dim=-1),
            torch.linalg.vector_norm(r[:, :, 2] - r[:, :, 1], dim=-1),
            torch.linalg.vector_norm(r[:, :, 0] - r[:, :, 2], dim=-1),
        ),
        dim=-1,
    ).clamp_min(eps)
    posed_edges_len = torch.stack(
        (
            torch.linalg.vector_norm(p[:, :, 1] - p[:, :, 0], dim=-1),
            torch.linalg.vector_norm(p[:, :, 2] - p[:, :, 1], dim=-1),
            torch.linalg.vector_norm(p[:, :, 0] - p[:, :, 2], dim=-1),
        ),
        dim=-1,
    )
    max_edge_ratio = (posed_edges_len / rest_edges).amax(dim=-1)

    normal = torch.cross(p1, p2, dim=-1)
    normal = F.normalize(normal, dim=-1, eps=eps)
    return TESSATriangleMetricsV1(
        jacobian=jacobian,
        singular_values=singular,
        area_ratio=area_ratio,
        condition_number=condition,
        max_edge_ratio=max_edge_ratio,
        normal=normal,
    )


def mechanical_consequence_loss_v1(
    metrics: TESSATriangleMetricsV1,
    *,
    reference_jacobian: torch.Tensor | None = None,
    reference_normal: torch.Tensor | None = None,
    max_edge_ratio: float = 4.0,
    max_condition: float = 16.0,
    min_area_ratio: float = 0.25,
    max_area_ratio: float = 4.0,
    weights: TESSAMechanicalRewardPolicyV1 | None = None,
) -> dict[str, torch.Tensor]:
    weights = weights or TESSAMechanicalRewardPolicyV1()
    weights.validate()

    stretch = F.relu(metrics.max_edge_ratio - float(max_edge_ratio)).mean()
    condition = F.relu(metrics.condition_number - float(max_condition)).mean()
    area_low = F.relu(float(min_area_ratio) - metrics.area_ratio)
    area_high = F.relu(metrics.area_ratio - float(max_area_ratio))
    area = (area_low + area_high).mean()

    jacobian = metrics.jacobian.new_zeros(())
    if reference_jacobian is not None:
        if reference_jacobian.shape != metrics.jacobian.shape:
            raise ValueError("TESSA_REFERENCE_JACOBIAN_SHAPE_MISMATCH")
        jacobian = F.smooth_l1_loss(metrics.jacobian, reference_jacobian)

    orientation = metrics.jacobian.new_zeros(())
    if reference_normal is not None:
        if reference_normal.shape != metrics.normal.shape:
            raise ValueError("TESSA_REFERENCE_NORMAL_SHAPE_MISMATCH")
        ref = F.normalize(reference_normal, dim=-1)
        orientation = (1.0 - (metrics.normal * ref).sum(dim=-1).clamp(-1.0, 1.0)).mean()

    # `flip` weight applies to orientation disagreement with the oracle/source
    # deformation field.  It is deliberately not a rest-normal test because a
    # legitimate articulated surface may rotate beyond 90 degrees.
    total = (
        weights.stretch * stretch
        + weights.condition * condition
        + weights.area * area
        + weights.jacobian * jacobian
        + weights.flip * orientation
    )
    return {
        "total": total,
        "stretch": stretch,
        "condition": condition,
        "area": area,
        "jacobian": jacobian,
        "orientation": orientation,
    }
