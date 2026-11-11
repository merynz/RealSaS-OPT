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
from torch.nn import functional as F


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


def qualified_source_region_pair_loss(
    membership_logits: Tensor,
    pair_indices: Tensor,
    pair_same_target: Tensor,
    pair_qualified_mask: Tensor,
) -> Tensor:
    """Permutation-invariant co-region supervision, NOT a drawable truth loss.

    Trusted source pairs describe whether two exact scene-query positions lie
    in the same declared source REGION (not necessarily a bone, component,
    mechanical domain or drawable). Unknown pair labels are masked. The
    model's anonymous region slots are exchangeable, so neither teacher nor
    compiler may impose arbitrary fixed slot IDs.
    """
    if membership_logits.ndim != 3 or membership_logits.shape[-1] < 2:
        raise ValueError("RSS_VNEXT_REGION_PAIR_MEMBERSHIP_SHAPE")
    b, n, classes = membership_logits.shape
    if (
        pair_indices.ndim != 3 or pair_indices.shape[0] != b
        or pair_indices.shape[-1] != 2 or pair_indices.dtype != torch.long
    ):
        raise ValueError("RSS_VNEXT_REGION_PAIR_INDEX_SHAPE")
    count = pair_indices.shape[1]
    if (
        not count or pair_same_target.shape != (b, count)
        or pair_qualified_mask.shape != (b, count)
        or pair_qualified_mask.dtype != torch.bool
        or not torch.is_floating_point(pair_same_target)
    ):
        raise ValueError("RSS_VNEXT_REGION_PAIR_TARGET_SHAPE")
    if (
        membership_logits.device != pair_indices.device
        or membership_logits.device != pair_same_target.device
        or membership_logits.device != pair_qualified_mask.device
    ):
        raise ValueError("RSS_VNEXT_REGION_PAIR_DEVICE_DRIFT")
    if (
        not torch.isfinite(membership_logits).all()
        or pair_indices.numel() and (
            int(pair_indices.min()) < 0 or int(pair_indices.max()) >= n
        )
    ):
        raise ValueError("RSS_VNEXT_REGION_PAIR_INPUT_INVALID")
    if bool((pair_indices[..., 0] == pair_indices[..., 1]).any()):
        raise ValueError("RSS_VNEXT_REGION_PAIR_SELF_PAIR_FORBIDDEN")
    if not bool(pair_qualified_mask.any()):
        raise ValueError("RSS_VNEXT_REGION_PAIR_NO_QUALIFIED_LABELS")
    admitted_targets = pair_same_target[pair_qualified_mask]
    if (
        not torch.isfinite(admitted_targets).all()
        or not bool(((admitted_targets == 0) | (admitted_targets == 1)).all())
    ):
        raise ValueError("RSS_VNEXT_REGION_PAIR_SOURCE_BINARY_INVALID")

    # UNKNOWN is NOT a shared region: compare conditional probabilities only
    # among learned region slots. Compute known-mass in log space, so even
    # highly confident UNKNOWN predictions receive a useful gradient rather
    # than underflowing/clamping to a zero-gradient vacuous solution.
    known_prob = F.softmax(membership_logits[..., :-1], dim=-1)
    left = torch.gather(
        known_prob, 1, pair_indices[..., 0:1].expand(-1, -1, classes - 1),
    )
    right = torch.gather(
        known_prob, 1, pair_indices[..., 1:2].expand(-1, -1, classes - 1),
    )
    conditional = (left * right).sum(dim=-1).clamp(1e-6, 1 - 1e-6)
    log_known = torch.logsumexp(
        F.log_softmax(membership_logits, dim=-1)[..., :-1],
        dim=-1,
    )
    left_coverage = torch.gather(log_known, 1, pair_indices[..., 0])
    right_coverage = torch.gather(log_known, 1, pair_indices[..., 1])
    coverage = -(left_coverage + right_coverage)
    return (
        F.binary_cross_entropy(conditional[pair_qualified_mask], admitted_targets)
        + 0.25 * coverage[pair_qualified_mask].mean()
    )


__all__ = [
    "RSSvNextRegionStructureOutput", "RSSvNextGenericRegionStructure",
    "qualified_source_region_pair_loss",
]
