from __future__ import annotations

from dataclasses import dataclass
import math
import torch
from torch import nn

from .q_domain_v2 import RayHypothesisDomainV2
from .q_evidence_encoder_v2 import QEvidenceEncodingV2
from .q_spatial_graph_v2 import build_anchor_neighbor_graph_v2


@dataclass
class EvidenceFieldOutputV2:
    score_logits: torch.Tensor
    hidden: torch.Tensor
    support_fraction: torch.Tensor
    neighbor_indices: torch.Tensor
    neighbor_mask: torch.Tensor


class EvidenceFieldV2(nn.Module):
    """Sparse learned C(Z,X,Y) evidence field.

    The old source implementation scored every q/depth candidate independently.
    V2 generic-strength keeps exact q candidates but adds two bounded context
    mechanisms: causal reasoning along each ray/depth sequence and sparse local
    message passing across neighboring anchor rays. The XY graph is recovered
    from native raster coordinates without QxQ materialization and every message
    is parameterized by exact world-space relative geometry.
    """

    def __init__(self, hidden_dim: int = 192, *, heads: int = 6, ray_layers: int = 2, spatial_neighbors: int = 8):
        super().__init__()
        if hidden_dim <= 0 or hidden_dim % heads:
            raise ValueError("hidden_dim/head mismatch")
        if ray_layers < 1 or spatial_neighbors < 1:
            raise ValueError("invalid evidence-field capacity")
        self.hidden_dim = int(hidden_dim)
        self.spatial_neighbors = int(spatial_neighbors)
        self.local_fuse = nn.Sequential(
            nn.Linear(hidden_dim + 4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        ray_layer = nn.TransformerEncoderLayer(
            hidden_dim,
            heads,
            hidden_dim * 3,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.ray_reasoner = nn.TransformerEncoder(ray_layer, ray_layers, norm=nn.LayerNorm(hidden_dim))
        self.spatial_message = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.spatial_score = nn.Sequential(nn.Linear(hidden_dim * 2 + 4, hidden_dim // 2), nn.GELU(), nn.Linear(hidden_dim // 2, 1))
        self.spatial_norm = nn.LayerNorm(hidden_dim)
        self.output_ff = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 2), nn.GELU(), nn.Linear(hidden_dim * 2, hidden_dim))
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.score = nn.Linear(hidden_dim, 1)

    @staticmethod
    def _gather_q(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        b = torch.arange(x.shape[0], device=x.device)[:, None, None]
        return x[b, idx]

    def _reason_along_rays(self, h: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        B, Q, D, H = h.shape
        flat = h.reshape(B * Q, D, H)
        mask = ~valid.reshape(B * Q, D).bool()
        safe = mask.clone()
        all_invalid = mask.all(dim=1)
        safe[all_invalid, 0] = False
        out = self.ray_reasoner(flat, src_key_padding_mask=safe).reshape(B, Q, D, H)
        return out * valid[..., None].to(out.dtype)

    def _reason_across_rays(
        self,
        h: torch.Tensor,
        q_points: torch.Tensor,
        valid: torch.Tensor,
        neighbor_indices: torch.Tensor,
        neighbor_mask: torch.Tensor,
    ) -> torch.Tensor:
        B, Q, D, H = h.shape
        hn = self._gather_q(h, neighbor_indices)  # [B,Q,K,D,H]
        qn = self._gather_q(q_points, neighbor_indices)  # [B,Q,K,D,3]
        hi = h[:, :, None, :, :].expand_as(hn)
        delta = qn - q_points[:, :, None, :, :]
        dist = torch.linalg.norm(delta, dim=-1, keepdim=True)
        geometry = torch.cat([delta, dist], dim=-1)
        pair = torch.cat([hi, hn, geometry], dim=-1)
        message = self.spatial_message(pair)
        logits = self.spatial_score(pair).squeeze(-1)
        vn = self._gather_q(valid, neighbor_indices)
        edge_valid = neighbor_mask[..., None] & valid[:, :, None, :] & vn
        logits = logits.masked_fill(~edge_valid, -1e4)
        weights = torch.softmax(logits, dim=2) * edge_valid.to(logits.dtype)
        weights = weights / weights.sum(dim=2, keepdim=True).clamp_min(1e-8)
        agg = (weights[..., None] * message).sum(dim=2)
        out = self.spatial_norm(h + agg)
        out = self.output_norm(out + self.output_ff(out))
        return out * valid[..., None].to(out.dtype)

    def forward(self, evidence: QEvidenceEncodingV2, domain: RayHypothesisDomainV2) -> EvidenceFieldOutputV2:
        sf = evidence.valid_views.float().mean(dim=-1)
        valid = evidence.valid_views.any(dim=-1) & domain.candidate_valid.bool()
        q = domain.q_points.to(evidence.pooled.dtype)
        radius = torch.linalg.norm(q, dim=-1, keepdim=True)
        h = self.local_fuse(torch.cat([evidence.pooled, q, radius], dim=-1))
        h = self._reason_along_rays(h, valid)
        valid_q = valid.any(dim=-1)
        neighbor_indices, neighbor_mask = build_anchor_neighbor_graph_v2(
            domain.anchor_grid,
            valid_q,
            max_neighbors=self.spatial_neighbors,
        )
        h = self._reason_across_rays(h, q, valid, neighbor_indices, neighbor_mask)
        logits = self.score(h).squeeze(-1).masked_fill(~valid, -1e4)
        return EvidenceFieldOutputV2(logits, h, sf, neighbor_indices, neighbor_mask)
