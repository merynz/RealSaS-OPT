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


def source_region_pairs_from_qualified_labels(
    region_ids: Tensor,
    source_valid_mask: Tensor,
    pair_indices: Tensor,
) -> tuple[Tensor, Tensor]:
    """Materialize pair targets from an externally qualified source region map.

    This builder does NOT qualify its inputs. It cannot infer regions, repair
    missing source labels, or assert a drawable/control relationship. The
    trusted foundry must establish these exact 3D query labels, source rights,
    and receipt hashes independently before the trainer accepts its result.
    """
    if region_ids.ndim != 2 or region_ids.dtype != torch.long:
        raise ValueError("RSS_VNEXT_SOURCE_REGION_IDS_SHAPE")
    b, n = region_ids.shape
    if source_valid_mask.shape != (b, n) or source_valid_mask.dtype != torch.bool:
        raise ValueError("RSS_VNEXT_SOURCE_REGION_MASK_SHAPE")
    if (
        pair_indices.ndim != 3 or pair_indices.shape[:1] != (b,)
        or pair_indices.shape[-1] != 2 or pair_indices.dtype != torch.long
        or pair_indices.shape[1] < 1
    ):
        raise ValueError("RSS_VNEXT_SOURCE_REGION_PAIRS_SHAPE")
    if not (region_ids.device == source_valid_mask.device == pair_indices.device):
        raise ValueError("RSS_VNEXT_SOURCE_REGION_DEVICE_DRIFT")
    if (
        int(pair_indices.min()) < 0 or int(pair_indices.max()) >= n
        or bool((pair_indices[..., 0] == pair_indices[..., 1]).any())
    ):
        raise ValueError("RSS_VNEXT_SOURCE_REGION_PAIR_ADDRESSES_INVALID")
    if bool((region_ids[source_valid_mask] < 0).any()):
        raise ValueError("RSS_VNEXT_SOURCE_REGION_ADMITTED_LABEL_INVALID")
    left = region_ids.gather(1, pair_indices[..., 0])
    right = region_ids.gather(1, pair_indices[..., 1])
    known_left = source_valid_mask.gather(1, pair_indices[..., 0])
    known_right = source_valid_mask.gather(1, pair_indices[..., 1])
    pair_mask = known_left & known_right
    # An unobserved point is never treated as a negative pair.
    pair_same = torch.where(
        pair_mask, (left == right).to(torch.float32),
        torch.zeros_like(left, dtype=torch.float32),
    )
    return pair_same, pair_mask


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


# Fixed channel semantics of the current RSS.vNext region-pair expert.
# Source part IDs do NOT supervise all three channels. In particular,
# directed attachment requires independently observed contact/intervention.
TYPED_REGION_RELATION_CHANNELS_V1 = (
    "same_drawable_affinity", "cut_affinity", "directed_attachment_affinity",
)


def qualified_source_typed_relation_loss(
    membership_logits: Tensor,
    relation_logits: Tensor,
    relation_valid_mask: Tensor,
    pair_indices: Tensor,
    target: Tensor,
    qualified_channel_mask: Tensor,
) -> tuple[Tensor, tuple[int, int, int]]:
    """Learn typed relations from *query pairs*, not fixed anonymous slot IDs.

    The model's R slots are exchangeable. Source-part UUIDs or material names
    must NEVER be copied into slot indices. Instead, each admitted source pair
    (x_i, x_j) marginalizes R×R directed region hypotheses using the same
    inferred membership probabilities. The first two pair channels are
    symmetric; the attachment channel is explicitly directed.

    Each channel has its OWN qualified source mask. Unobserved contact,
    disconnected geometry, drawable/edit consequence or material segmentation
    cannot be turned into negative training labels. Teacher evidence is only
    supplied by the separately trusted launcher's exact source receipt.
    """
    if membership_logits.ndim != 3 or membership_logits.shape[-1] < 3:
        raise ValueError("RSS_VNEXT_TYPED_RELATION_MEMBERSHIP_SHAPE")
    b, n, classes = membership_logits.shape
    r = classes - 1  # last channel is UNKNOWN, never a source part identity
    if (
        relation_logits.shape != (b, r, r, 3)
        or relation_valid_mask.shape != (b, r, r)
        or relation_valid_mask.dtype != torch.bool
    ):
        raise ValueError("RSS_VNEXT_TYPED_RELATION_SLOT_GRAPH_SHAPE")
    if (
        pair_indices.ndim != 3 or pair_indices.shape[0] != b
        or pair_indices.shape[-1] != 2 or pair_indices.dtype != torch.long
        or pair_indices.shape[1] < 1
        or target.shape != (b, pair_indices.shape[1], 3)
        or qualified_channel_mask.shape != target.shape
        or qualified_channel_mask.dtype != torch.bool
        or not torch.is_floating_point(target)
    ):
        raise ValueError("RSS_VNEXT_TYPED_RELATION_TEACHER_SHAPE")
    if len({
        membership_logits.device, relation_logits.device,
        relation_valid_mask.device, pair_indices.device, target.device,
        qualified_channel_mask.device,
    }) != 1:
        raise ValueError("RSS_VNEXT_TYPED_RELATION_DEVICE_DRIFT")
    if (
        not bool(torch.isfinite(membership_logits).all())
        or not bool(torch.isfinite(relation_logits).all())
        or int(pair_indices.min()) < 0 or int(pair_indices.max()) >= n
        or bool((pair_indices[..., 0] == pair_indices[..., 1]).any())
    ):
        raise ValueError("RSS_VNEXT_TYPED_RELATION_INPUT_INVALID")
    expected_mask = ~torch.eye(r, dtype=torch.bool, device=relation_logits.device)[None]
    if not torch.equal(relation_valid_mask, expected_mask.expand(b, -1, -1)):
        raise ValueError("RSS_VNEXT_TYPED_RELATION_SELF_PAIR_MASK_INVALID")
    if not torch.allclose(
        relation_logits[..., :2], relation_logits.transpose(1, 2)[..., :2],
        rtol=0.0, atol=1.e-5,
    ):
        raise ValueError("RSS_VNEXT_TYPED_RELATION_SYMMETRY_INVALID")
    if not bool(qualified_channel_mask.any()):
        raise ValueError("RSS_VNEXT_TYPED_RELATION_NO_QUALIFIED_CHANNEL")
    admitted = target[qualified_channel_mask]
    if (
        not bool(torch.isfinite(admitted).all())
        or not bool(((admitted == 0) | (admitted == 1)).all())
    ):
        raise ValueError("RSS_VNEXT_TYPED_RELATION_SOURCE_BINARY_INVALID")

    # Conditional known-slot probabilities preserve permutation invariance:
    # permuting both slots and relation matrix leaves every prediction equal.
    # UNKNOWN is not a negative class or a relation node.
    known = F.softmax(membership_logits[..., :-1].float(), dim=-1)
    left = known.gather(
        1, pair_indices[..., :1].expand(-1, -1, r),
    )
    right = known.gather(
        1, pair_indices[..., 1:].expand(-1, -1, r),
    )
    joint = (left.unsqueeze(-1) * right.unsqueeze(-2)
             * relation_valid_mask[:, None].float())
    norm = joint.sum(dim=(2, 3)).clamp_min(1.e-10)
    scores = torch.sigmoid(relation_logits.float())[:, None]
    predicted = (
        (joint[..., None] * scores).sum(dim=(2, 3)) / norm[..., None]
    ).clamp(1.e-6, 1. - 1.e-6)

    # A source-qualified pair proves that its two points are observed; don't
    # let UNKNOWN absorb all membership to trivially avoid typed supervision.
    all_log_prob = F.log_softmax(membership_logits.float(), dim=-1)
    log_known = torch.logsumexp(all_log_prob[..., :-1], dim=-1)
    coverage = -(
        log_known.gather(1, pair_indices[..., 0])
        + log_known.gather(1, pair_indices[..., 1])
    )
    eligible_pairs = qualified_channel_mask.any(dim=-1)
    objective = (
        F.binary_cross_entropy(
            predicted[qualified_channel_mask], admitted.float(),
        )
        + .25 * coverage[eligible_pairs].mean()
    )
    counts = tuple(int(qualified_channel_mask[..., channel].sum().item())
                   for channel in range(3))
    return objective, counts


__all__ = [
    "RSSvNextRegionStructureOutput", "RSSvNextGenericRegionStructure",
    "qualified_source_region_pair_loss",
    "source_region_pairs_from_qualified_labels",
    "TYPED_REGION_RELATION_CHANNELS_V1",
    "qualified_source_typed_relation_loss",
]
