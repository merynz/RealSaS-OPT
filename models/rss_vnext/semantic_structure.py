from __future__ import annotations

"""RSS.vNext generic semantic/structure inference, NOT a named-bodypart atlas.

Identity-bearing region slots read the SAME live 3D H(x) as topology, control,
and graphics. The slots are exchangeable candidate regions and may be empty.
No material name, material group, mesh object name or human skeleton template
is used as structural truth. Pair links are *evidence proposals*, never
Compiler-issued attachment or drawable authority.
"""

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn


@dataclass
class RSSvNextRegionStructureOutput:
    region_slots: Tensor          # [B,R,C]
    membership_logits: Tensor     # [B,N,R+1], last = UNKNOWN
    relation_logits: Tensor       # [B,R,R,3]
    relation_valid_mask: Tensor   # [B,R,R], excludes self-pairs
    authority: str = "UNQUALIFIED_REGION_AND_ATTACHMENT_PROPOSALS"


class RSSvNextGenericRegionStructure(nn.Module):
    """Perceiver-style slot readout plus variable-source slot relation graph.

    Slot indices are permutation equivariant under a corresponding permutation
    of the learned queries. Data-level matching/persistent identity is NOT
    solved merely by slot embeddings; qualification must track correspondences.
    """

    relation_keys = (
        "same_drawable_affinity",
        "cut_affinity",
        "attachment_affinity",
    )

    def __init__(self, carrier_dim: int, region_slots: int, *, heads: int = 8, layers: int = 2):
        super().__init__()
        if (min(carrier_dim, region_slots, heads, layers) <= 0
                or carrier_dim % heads):
            raise ValueError("RSS_VNEXT_REGION_CONFIG_INVALID")
        self.region_slots_count = region_slots
        self.learned_region_queries = nn.Parameter(
            torch.randn(region_slots, carrier_dim) / math.sqrt(carrier_dim),
        )
        self.cross_attention = nn.MultiheadAttention(
            carrier_dim, heads, batch_first=True,
        )
        self.norm = nn.LayerNorm(carrier_dim)
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(carrier_dim),
                nn.Linear(carrier_dim, 4 * carrier_dim),
                nn.GELU(),
                nn.Linear(4 * carrier_dim, carrier_dim),
            ) for _ in range(layers)
        ])
        self.point = nn.Sequential(
            nn.LayerNorm(carrier_dim), nn.Linear(carrier_dim, carrier_dim),
        )
        self.slot = nn.Sequential(
            nn.LayerNorm(carrier_dim), nn.Linear(carrier_dim, carrier_dim),
        )
        self.unknown = nn.Sequential(
            nn.LayerNorm(carrier_dim), nn.Linear(carrier_dim, 1),
        )
        self.pair = nn.Sequential(
            nn.LayerNorm(4 * carrier_dim),
            nn.Linear(4 * carrier_dim, 2 * carrier_dim),
            nn.GELU(),
            nn.Linear(2 * carrier_dim, 3),
        )

    def assign(self, point_h: Tensor, slots: Tensor) -> Tensor:
        if point_h.ndim != 3 or slots.ndim != 3 or point_h.shape[0] != slots.shape[0]:
            raise ValueError("RSS_VNEXT_REGION_ASSIGN_BATCH")
        if point_h.shape[2] != slots.shape[2] or slots.shape[1] != self.region_slots_count:
            raise ValueError("RSS_VNEXT_REGION_ASSIGN_WIDTH")
        membership = torch.matmul(
            self.point(point_h), self.slot(slots).transpose(1, 2),
        ) / math.sqrt(point_h.shape[-1])
        return torch.cat((membership, self.unknown(point_h)), dim=-1)

    def forward(self, point_h: Tensor, *, valid_mask: Tensor | None = None) -> RSSvNextRegionStructureOutput:
        if point_h.ndim != 3 or not point_h.shape[1] or not torch.isfinite(point_h).all():
            raise ValueError("RSS_VNEXT_REGION_SOURCE_FIELD_INVALID")
        b, n, d = point_h.shape
        if valid_mask is not None:
            if valid_mask.shape != (b, n) or valid_mask.dtype != torch.bool:
                raise ValueError("RSS_VNEXT_REGION_VALID_MASK_INVALID")
            if not bool(valid_mask.any(dim=1).all()):
                raise ValueError("RSS_VNEXT_REGION_ALL_UNOBSERVED")
        q = self.learned_region_queries[None].expand(b, -1, -1)
        context, _ = self.cross_attention(
            q, point_h, point_h,
            key_padding_mask=None if valid_mask is None else ~valid_mask,
            need_weights=False,
        )
        slots = self.norm(q + context)
        for layer in self.layers:
            slots = slots + layer(slots)
        a = slots[:, :, None, :].expand(-1, -1, self.region_slots_count, -1)
        bslot = slots[:, None, :, :].expand(-1, self.region_slots_count, -1, -1)
        pair = self.pair(torch.cat((a, bslot, a - bslot, a * bslot), dim=-1))
        # A continuous assignment graph can represent symmetric affinities
        # and a directed attachment likelihood without inventing parentage.
        symmetric = .5 * (pair[..., :2] + pair.transpose(1, 2)[..., :2])
        pair = torch.cat((symmetric, pair[..., 2:3]), dim=-1)
        diagonal = torch.eye(
            self.region_slots_count, device=point_h.device, dtype=torch.bool,
        )[None].expand(point_h.shape[0], -1, -1)
        return RSSvNextRegionStructureOutput(
            region_slots=slots,
            membership_logits=self.assign(point_h, slots),
            relation_logits=pair,
            relation_valid_mask=~diagonal,
        )


__all__ = ["RSSvNextRegionStructureOutput", "RSSvNextGenericRegionStructure"]
