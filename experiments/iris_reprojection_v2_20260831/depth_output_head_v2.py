from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .ray_modes_v2 import RayModesV2


@dataclass
class DepthOutputV2:
    support_logits: torch.Tensor
    log_sigma: torch.Tensor
    support_probability: torch.Tensor
    conditional_support_probability: torch.Tensor | None = None
    mode_probability: torch.Tensor | None = None
    view_support_logits: torch.Tensor | None = None
    view_support_probability: torch.Tensor | None = None


class DepthSupportUncertaintyHeadV2(nn.Module):
    """Mode existence/uncertainty plus conditional per-view observational support.

    `support_probability` remains the joint probability that a selected ray mode is
    a supported surface hypothesis. `view_support_probability[b,q,k,v]` answers a
    different question: conditional on that selected mode, does observation view v
    actually support/see the locus? It must never be inferred merely from in-frame
    reprojection.
    """

    def __init__(self, hidden_dim: int = 192):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 2),
        )
        self.view_head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        hidden: torch.Tensor,
        modes: RayModesV2,
        *,
        view_tokens: torch.Tensor | None = None,
        in_frame: torch.Tensor | None = None,
    ) -> DepthOutputV2:
        idx = modes.mode_indices.clamp_min(0)
        H = hidden.shape[-1]
        gathered = torch.gather(hidden, 2, idx[..., None].expand(-1, -1, -1, H))
        raw = self.head(gathered)
        sl = raw[..., 0]
        ls = raw[..., 1].clamp(-8.0, 3.0)
        conditional = torch.sigmoid(sl)
        mode_probability = modes.mode_probabilities
        sp = conditional * mode_probability

        sl = torch.where(modes.mode_valid, sl, torch.full_like(sl, -1e4))
        ls = torch.where(modes.mode_valid, ls, torch.zeros_like(ls))
        conditional = torch.where(modes.mode_valid, conditional, torch.zeros_like(conditional))
        mode_probability = torch.where(modes.mode_valid, mode_probability, torch.zeros_like(mode_probability))
        sp = torch.where(modes.mode_valid, sp, torch.zeros_like(sp))

        view_logits = None
        view_probability = None
        if view_tokens is not None:
            if view_tokens.ndim != 5 or view_tokens.shape[:3] != hidden.shape[:3] or view_tokens.shape[3] != 8 or view_tokens.shape[4] != H:
                raise ValueError("per-view support head requires [B,Q,D,8,H] view tokens")
            B, Q, K = idx.shape
            gather_idx = idx[..., None, None].expand(B, Q, K, 8, H)
            selected_views = torch.gather(view_tokens, 2, gather_idx)
            global_hidden = gathered[..., None, :].expand(B, Q, K, 8, H)
            view_logits = self.view_head(torch.cat([global_hidden, selected_views], dim=-1)).squeeze(-1)
            valid = modes.mode_valid[..., None].expand(B, Q, K, 8)
            if in_frame is not None:
                if in_frame.ndim != 4 or in_frame.shape[:3] != hidden.shape[:3] or in_frame.shape[3] != 8:
                    raise ValueError("per-view support in_frame must be [B,Q,D,8]")
                selected_in_frame = torch.gather(in_frame.bool(), 2, idx[..., None].expand(B, Q, K, 8))
                valid = valid & selected_in_frame
            view_logits = torch.where(valid, view_logits, torch.full_like(view_logits, -1e4))
            view_probability = torch.sigmoid(view_logits) * valid.to(view_logits.dtype)
        elif in_frame is not None:
            raise ValueError("in_frame supplied without view_tokens")

        return DepthOutputV2(
            sl,
            ls,
            sp,
            conditional,
            mode_probability,
            view_logits,
            view_probability,
        )
