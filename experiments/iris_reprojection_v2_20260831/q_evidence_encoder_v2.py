from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import nn

from .q_domain_v2 import RayHypothesisDomainV2
from .q_descriptor_sampler_v2 import SampledQDescriptorsV2


@dataclass
class QEvidenceEncodingV2:
    view_tokens: torch.Tensor
    pooled: torch.Tensor
    valid_views: torch.Tensor
    view_weights: torch.Tensor


class QEvidenceEncoderV2(nn.Module):
    """Permutation-equivariant multi-view evidence aggregation.

    View slots have no learned absolute identity. Camera/candidate relation enters
    only through analytic per-view reprojection quantities: projected grid and
    camera-forward projected depth. Re-enumerating the eight observations together
    with those analytic quantities must therefore only re-enumerate view tokens and
    weights; pooled evidence is invariant.
    """
    def __init__(self, descriptor_dim: int, hidden_dim: int = 192, heads: int = 6, layers: int = 2):
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden_dim must divide heads")
        # descriptor + projected grid (2) + projected camera-forward depth (1)
        # + validity (1). No absolute view-index scalar is admitted.
        self.input = nn.Sequential(nn.Linear(descriptor_dim + 4, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        layer = nn.TransformerEncoderLayer(hidden_dim, heads, hidden_dim * 3, dropout=0.0, batch_first=True, norm_first=True, activation="gelu")
        self.view_reasoner = nn.TransformerEncoder(layer, layers, norm=nn.LayerNorm(hidden_dim))
        self.robust_score = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))
        self.hidden_dim = hidden_dim

    def forward(self, sampled: SampledQDescriptorsV2, domain: RayHypothesisDomainV2) -> QEvidenceEncodingV2:
        x = sampled.descriptors
        B, Q, D, V, C = x.shape
        if V != 8 or sampled.valid_views.shape != (B, Q, D, V):
            raise ValueError("view cardinality/mask mismatch")
        if domain.projected_grid.shape != (B, Q, D, V, 2) or domain.projected_depth.shape != (B, Q, D, V):
            raise ValueError("analytic reprojection geometry shape mismatch")
        inp = torch.cat([
            x,
            domain.projected_grid.to(x.dtype),
            domain.projected_depth[..., None].to(x.dtype),
            sampled.valid_views[..., None].to(x.dtype),
        ], dim=-1)
        h = self.input(inp).reshape(B * Q * D, V, self.hidden_dim)
        mask = ~sampled.valid_views.reshape(B * Q * D, V).bool()
        safe = mask.clone()
        safe[mask.all(dim=1), 0] = False
        h = self.view_reasoner(h, src_key_padding_mask=safe).reshape(B, Q, D, V, self.hidden_dim)
        h = h * sampled.valid_views[..., None].to(h.dtype)
        logits = self.robust_score(h).squeeze(-1).masked_fill(~sampled.valid_views.bool(), -1e4)
        w = torch.softmax(logits, dim=-1) * sampled.valid_views.to(x.dtype)
        w = w / w.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        pooled = (w[..., None] * h).sum(dim=-2) * sampled.valid_views.any(dim=-1, keepdim=True).to(h.dtype)
        return QEvidenceEncodingV2(h, pooled, sampled.valid_views, w)
