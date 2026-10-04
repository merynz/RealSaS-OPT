from __future__ import annotations

"""Differentiable fixed-topology mechanical compatibility surrogate.

Research-only V1. The hard Compiler G3/G3B courts remain fail-closed authority.
This module exists to make the same deformation consequences visible to learned
systems before product qualification.

The topology is an explicit input. Changing only face connectivity can therefore
change the loss even when rest vertices, skin weights, rig and posed vertices are
otherwise identical.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class JointMechanicalLossConfigV1:
    max_condition: float = 16.0
    min_area_ratio: float = 0.05
    max_area_ratio: float = 20.0
    max_edge_ratio: float = 4.0
    condition_weight: float = 1.0
    area_weight: float = 1.0
    edge_weight: float = 1.0
    eps: float = 1e-8


def lbs_points_v1(
    rest_points: torch.Tensor,
    weights: torch.Tensor,
    skin_matrices: torch.Tensor,
) -> torch.Tensor:
    """Apply differentiable linear-blend skinning.

    rest_points: [V,3]
    weights: [V,J], simplex-valued
    skin_matrices: [Q,J,4,4] or [J,4,4]
    returns: [Q,V,3]
    """
    if rest_points.ndim != 2 or rest_points.shape[-1] != 3:
        raise ValueError("JOINT_MECH_REST_SHAPE")
    if weights.ndim != 2 or weights.shape[0] != rest_points.shape[0]:
        raise ValueError("JOINT_MECH_WEIGHT_SHAPE")
    if skin_matrices.ndim == 3:
        skin_matrices = skin_matrices.unsqueeze(0)
    if (
        skin_matrices.ndim != 4
        or skin_matrices.shape[-2:] != (4, 4)
        or skin_matrices.shape[1] != weights.shape[1]
    ):
        raise ValueError("JOINT_MECH_MATRIX_SHAPE")

    hom = torch.cat(
        [rest_points, torch.ones_like(rest_points[:, :1])], dim=-1
    )
    per_joint = torch.einsum(
        "vc,qjkc->qvjk", hom, skin_matrices
    )[..., :3]
    return torch.einsum("vj,qvjk->qvk", weights, per_joint)


def triangle_mechanical_metrics_v1(
    rest_points: torch.Tensor,
    posed_points: torch.Tensor,
    faces: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> dict[str, torch.Tensor]:
    """Compute G3-like differentiable triangle metrics.

    posed_points: [Q,V,3] or [V,3]
    faces: [F,3]
    """
    if posed_points.ndim == 2:
        posed_points = posed_points.unsqueeze(0)
    if posed_points.ndim != 3 or posed_points.shape[-1] != 3:
        raise ValueError("JOINT_MECH_POSED_SHAPE")
    if faces.ndim != 2 or faces.shape[-1] != 3:
        raise ValueError("JOINT_MECH_FACE_SHAPE")

    faces = faces.to(dtype=torch.long, device=rest_points.device)
    r = rest_points[faces]                         # [F,3,3]
    p = posed_points[:, faces]                     # [Q,F,3,3]

    r1 = r[:, 1] - r[:, 0]
    r2 = r[:, 2] - r[:, 0]
    l1 = torch.linalg.vector_norm(r1, dim=-1)
    u = r1 / l1.clamp_min(eps).unsqueeze(-1)
    x2 = torch.sum(r2 * u, dim=-1)
    perp = r2 - x2.unsqueeze(-1) * u
    y2 = torch.linalg.vector_norm(perp, dim=-1)

    if bool(torch.any(l1 <= eps).item()) or bool(torch.any(y2 <= eps).item()):
        raise ValueError("JOINT_MECH_REST_TRIANGLE_DEGENERATE")

    inv = torch.zeros(
        (faces.shape[0], 2, 2),
        dtype=rest_points.dtype,
        device=rest_points.device,
    )
    inv[:, 0, 0] = 1.0 / l1
    inv[:, 0, 1] = -x2 / (l1 * y2)
    inv[:, 1, 1] = 1.0 / y2

    pe = torch.stack(
        [p[:, :, 1] - p[:, :, 0], p[:, :, 2] - p[:, :, 0]], dim=-1
    )                                               # [Q,F,3,2]
    jac = torch.einsum("qfij,fjk->qfik", pe, inv)  # [Q,F,3,2]
    singular = torch.linalg.svdvals(jac)
    smax = singular[..., 0]
    smin = singular[..., 1]
    area_ratio = smax * smin
    condition = smax / smin.clamp_min(eps)

    rest_edges = torch.stack(
        [
            torch.linalg.vector_norm(r[:, 1] - r[:, 0], dim=-1),
            torch.linalg.vector_norm(r[:, 2] - r[:, 1], dim=-1),
            torch.linalg.vector_norm(r[:, 0] - r[:, 2], dim=-1),
        ],
        dim=-1,
    )
    posed_edges = torch.stack(
        [
            torch.linalg.vector_norm(p[:, :, 1] - p[:, :, 0], dim=-1),
            torch.linalg.vector_norm(p[:, :, 2] - p[:, :, 1], dim=-1),
            torch.linalg.vector_norm(p[:, :, 0] - p[:, :, 2], dim=-1),
        ],
        dim=-1,
    )
    edge_ratio = posed_edges / rest_edges.clamp_min(eps).unsqueeze(0)

    return {
        "condition": condition,
        "area_ratio": area_ratio,
        "edge_ratio": edge_ratio,
        "jacobian": jac,
    }


def joint_mechanical_loss_v1(
    rest_points: torch.Tensor,
    faces: torch.Tensor,
    weights: torch.Tensor,
    skin_matrices: torch.Tensor,
    *,
    config: JointMechanicalLossConfigV1 | None = None,
) -> dict[str, torch.Tensor]:
    cfg = config or JointMechanicalLossConfigV1()
    posed = lbs_points_v1(rest_points, weights, skin_matrices)
    metrics = triangle_mechanical_metrics_v1(
        rest_points, posed, faces, eps=cfg.eps
    )

    condition = metrics["condition"]
    area = metrics["area_ratio"]
    edge = metrics["edge_ratio"]

    # Log-space condition penalty is scale-stable and mirrors a hard threshold.
    cond_excess = F.relu(
        torch.log(condition.clamp_min(cfg.eps))
        - torch.log(
            torch.as_tensor(
                cfg.max_condition,
                dtype=condition.dtype,
                device=condition.device,
            )
        )
    )
    cond_loss = (cond_excess * cond_excess).mean()

    area_low = F.relu(
        torch.as_tensor(
            cfg.min_area_ratio, dtype=area.dtype, device=area.device
        ) - area
    )
    area_high = F.relu(
        area - torch.as_tensor(
            cfg.max_area_ratio, dtype=area.dtype, device=area.device
        )
    )
    area_loss = (area_low.square() + area_high.square()).mean()

    edge_excess = F.relu(
        edge
        - torch.as_tensor(
            cfg.max_edge_ratio, dtype=edge.dtype, device=edge.device
        )
    )
    edge_loss = edge_excess.square().mean()

    total = (
        cfg.condition_weight * cond_loss
        + cfg.area_weight * area_loss
        + cfg.edge_weight * edge_loss
    )
    return {
        "loss": total,
        "condition_loss": cond_loss,
        "area_loss": area_loss,
        "edge_loss": edge_loss,
        "max_condition": condition.max(),
        "min_area_ratio": area.min(),
        "max_area_ratio": area.max(),
        "max_edge_ratio": edge.max(),
        "posed_points": posed,
    }


__all__ = [
    "JointMechanicalLossConfigV1",
    "lbs_points_v1",
    "triangle_mechanical_metrics_v1",
    "joint_mechanical_loss_v1",
]
