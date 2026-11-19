from __future__ import annotations

"""Single sparse skin support policy for MIRA training and Compiler replay.

This is a deterministic selection operator on the already learned MIRA
vertex-to-joint weights. It does NOT synthesize joint identities, skin truth,
bone groups or a new model head. The identical hard forward is applied to
teacher-forced, free-running and exact-FSQ replay paths.

The straight-through backward supplies surrogate gradients to all dense
candidate joints, including currently unselected source-positive joints.
Ablation of the support budget and hard-gradient estimator is still needed
on independently admitted, held-out real rigs.
"""

import torch
from torch import Tensor


def sparse_skin_support_v1(
    dense: Tensor,
    *,
    max_influences: int,
    straight_through: bool = False,
) -> Tensor:
    """Exact-zero top-K support with stable low-joint-index tie-break.

    [B,V,J] or [V,J] source-simplex weights are required. The returned
    forward values are identical regardless of straight_through; only the
    backward changes. When J<=K, the original simplex is preserved.
    """
    if (
        not isinstance(dense, Tensor)
        or dense.ndim not in (2, 3)
        or dense.numel() < 1
        or not dense.is_floating_point()
        or type(max_influences) is not int
        or not 1 <= max_influences <= 41
        or type(straight_through) is not bool
        or not bool(torch.isfinite(dense).all())
        or bool((dense < 0).any())
        or not bool(torch.allclose(
            dense.sum(dim=-1), torch.ones_like(dense[..., 0]),
            atol=1.e-5, rtol=0.,
        ))
    ):
        raise ValueError("RSS_VNEXT_SKIN_SPARSE_SUPPORT_DOMAIN_INVALID")
    joints = dense.shape[-1]
    if joints <= max_influences:
        return dense
    # Stable argsort orders equal scores by native causal AXIS joint index.
    # No nondeterministic topk ties and no source-bone-name sorting.
    indices = torch.argsort(dense.detach(), dim=-1, descending=True, stable=True)
    selected = indices[..., :max_influences]
    support = torch.zeros_like(dense, dtype=torch.bool).scatter_(
        dim=-1, index=selected, value=True,
    )
    selected_weights = dense.masked_fill(~support, 0.)
    denominator = selected_weights.sum(dim=-1, keepdim=True)
    if not bool((denominator > 0).all()):
        raise ValueError("RSS_VNEXT_SKIN_SPARSE_SUPPORT_ZERO_ROW")
    sparse = selected_weights / denominator
    if straight_through:
        # Exactly sparse forward; dense surrogate derivative reaches joints
        # that are currently outside the selected K (important for teachers
        # whose truly weighted joints were initially ranked too low).
        return sparse + (dense - dense.detach())
    return sparse


__all__ = ["sparse_skin_support_v1"]
