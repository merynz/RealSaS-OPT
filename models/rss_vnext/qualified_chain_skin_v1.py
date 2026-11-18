from __future__ import annotations

"""Exact native rig/skin -> mechanical-chain supervision on ONE TESSA M*.

The input joint->chain map is an independently qualified *source* partition.
No bone-name heuristic, 2D part label, semantic role or anonymous region slot
can mint that map. A missing joint owner is UNKNOWN, not another chain.
This objective guides MIRA's existing skin weights; it does not make a second
skinner, rebind vertices or introduce a physical solver.
"""

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class QualifiedChainSkinResultV1:
    total: Tensor
    chain_mass: Tensor
    within_chain: Tensor
    eligible_vertex_count: int
    eligible_conditional_count: int
    qualified_chain_count: int


def qualified_source_chain_skin_loss_v1(
    *,
    predicted_skin: Tensor,
    source_skin: Tensor,
    joint_chain_ids: Tensor,
    qualified_vertex_mask: Tensor,
    unknown_mass_tolerance: float = 1.e-5,
) -> QualifiedChainSkinResultV1:
    """Supervise chain mass and conditional weights WITHOUT synthetic labels.

    Both skin tensors have shape [1,V,J] in the SAME native-matched M*/AXIS
    joint order. joint_chain_ids is [J] int64, with -1 meaning not established.
    The mask is an independently admitted [1,V] source qualification. If a
    vertex carries material weight on UNKNOWN joints, it cannot be treated
    as a fully classified negative example and is excluded.

    P(chain|vertex) is the sum of source weights owned by that chain;
    P(joint|chain,vertex) is the source skin normalized within its chain.
    This decomposition is an identity for qualified skin/owner maps and
    supplies no invented Wing, Cloth, attachment or physics ground truth.
    """
    if (
        predicted_skin.ndim != 3 or predicted_skin.shape[0] != 1
        or source_skin.shape != predicted_skin.shape
        or predicted_skin.shape[1] < 1 or predicted_skin.shape[2] < 1
        or qualified_vertex_mask.shape != predicted_skin.shape[:2]
        or qualified_vertex_mask.dtype != torch.bool
        or joint_chain_ids.shape != (predicted_skin.shape[2],)
        or joint_chain_ids.dtype != torch.long
        or any(t.device != predicted_skin.device for t in
               (source_skin, joint_chain_ids, qualified_vertex_mask))
        or not 0 <= unknown_mass_tolerance <= 1.e-3
        or not bool(torch.isfinite(predicted_skin).all())
        or not bool(torch.isfinite(source_skin).all())
        or bool((source_skin < -1.e-7).any())
        or bool((predicted_skin < -1.e-7).any())
        or bool((joint_chain_ids < -1).any())
    ):
        raise ValueError("RSS_VNEXT_CHAIN_SKIN_SOURCE_DOMAIN_INVALID")
    if not bool((qualified_vertex_mask).any()):
        raise ValueError("RSS_VNEXT_CHAIN_SKIN_NO_QUALIFIED_VERTICES")
    if not bool(torch.allclose(
        source_skin.sum(dim=-1), torch.ones_like(source_skin[..., 0]),
        atol=1.e-4, rtol=0.,
    )):
        raise ValueError("RSS_VNEXT_CHAIN_SKIN_SOURCE_NOT_NORMALIZED")
    chain_ids = torch.unique(joint_chain_ids[joint_chain_ids >= 0], sorted=True)
    if chain_ids.numel() < 1:
        raise ValueError("RSS_VNEXT_CHAIN_SKIN_NO_SOURCE_CHAINS")
    # The semantic names of chain IDs never enter this arithmetic.
    # Sorted unique IDs compact arbitrary source-native chain namespaces.
    _, compact = torch.unique(
        joint_chain_ids[joint_chain_ids >= 0], sorted=True, return_inverse=True,
    )
    owner = joint_chain_ids >= 0
    j = predicted_skin.shape[-1]
    v = predicted_skin.shape[1]
    k = int(chain_ids.numel())
    mapped = torch.zeros(j, dtype=torch.long, device=predicted_skin.device)
    mapped[owner] = compact
    select = mapped.view(1, 1, j).expand(1, v, j)
    src_mass = torch.zeros(1, v, k, device=source_skin.device,
                           dtype=source_skin.dtype).scatter_add(
        -1, select, source_skin * owner.to(source_skin.dtype),
    )
    pred_mass = torch.zeros(1, v, k, device=predicted_skin.device,
                            dtype=predicted_skin.dtype).scatter_add(
        -1, select, predicted_skin * owner.to(predicted_skin.dtype),
    )
    unknown_mass = source_skin[..., ~owner].sum(dim=-1)
    eligible = qualified_vertex_mask & (unknown_mass <= unknown_mass_tolerance)
    if not bool(eligible.any()):
        raise ValueError("RSS_VNEXT_CHAIN_SKIN_ALL_VERTICES_AMBIGUOUS")
    chain_loss = F.smooth_l1_loss(pred_mass[eligible], src_mass[eligible])
    # A conditional bone target exists only if its SOURCE chain carries mass.
    # No penalties for a source-inactive chain's arbitrary within-chain ratios.
    src_per_joint = torch.gather(src_mass, -1, select).clamp_min(1.e-8)
    pred_per_joint = torch.gather(pred_mass, -1, select).clamp_min(1.e-8)
    active = (src_per_joint > 1.e-6) & owner.view(1, 1, j) & eligible.unsqueeze(-1)
    if bool(active.any()):
        target_cond = (source_skin / src_per_joint)[active]
        student_cond = (predicted_skin / pred_per_joint)[active]
        within_loss = F.smooth_l1_loss(student_cond, target_cond)
        active_count = int(active.sum().item())
    else:
        within_loss = predicted_skin.sum() * 0.
        active_count = 0
    total = chain_loss + within_loss
    if not bool(torch.isfinite(total)):
        raise ValueError("RSS_VNEXT_CHAIN_SKIN_LOSS_NONFINITE")
    return QualifiedChainSkinResultV1(
        total=total,
        chain_mass=chain_loss,
        within_chain=within_loss,
        eligible_vertex_count=int(eligible.sum().item()),
        eligible_conditional_count=active_count,
        qualified_chain_count=k,
    )


__all__ = ["QualifiedChainSkinResultV1", "qualified_source_chain_skin_loss_v1"]
