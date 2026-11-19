from __future__ import annotations

"""Learned, source-qualified AXIS chain mixture for ONE MIRA skin field.

An AXIS CHAIN is a real contiguous causal joint path, not an anatomical label.
Only joints uniquely owned by one declared chain receive that chain's group
ID; shared roots, branch junctions and unclaimed joints remain in the explicit
UNKNOWN joint bucket. UNKNOWN is a valid learnable bucket, never a fabricated
"body" semantic class.

The distribution is an exact mixture:
  P(j|v)=P(g(j)|v)*P(j|v,g(j)).
The group gate is independently trainable, not post-hoc summation of flat
skin weights. The base per-joint logits and learned gate share the original
source-qualified M* and AXIS causal joint order.

When no CHAIN is declared all joints belong to UNKNOWN and the mixture
reduces exactly to the existing flat softmax (no invented grouping).
"""

from collections.abc import Mapping, Sequence
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from models.axis.rig_program_model_v6 import (
    AXISSequenceV6, RigProgramEventV6, RigTokenKindV6,
)
from .axis_mechanical_chain_partition_v1 import decode_axis_joint_chain_partition_v1


def axis_joint_group_ids_v1(
    sequence: AXISSequenceV6, *, batch_index: int = 0,
) -> Tensor:
    partition = decode_axis_joint_chain_partition_v1(
        sequence, batch_index=batch_index,
    )
    # -1 is reserved for UNKNOWN; chain IDs are causal event ordinal, not
    # mechanical morphology or semantic labels.
    return partition.joint_chain_ids


def axis_joint_group_ids_from_cas_events_v1(
    events: Sequence[Mapping[str, object]],
) -> Tensor:
    """Exact same partition in the independent product Compiler replay."""
    if not isinstance(events, (tuple, list)) or not events:
        raise ValueError("RSS_VNEXT_MIRA_GROUP_CAUSAL_EVENTS_REQUIRED")
    converted: list[RigProgramEventV6] = []
    for row in events:
        if not isinstance(row, Mapping):
            raise ValueError("RSS_VNEXT_MIRA_GROUP_EVENT_DOMAIN_INVALID")
        try:
            raw_kind = row["kind"]
            if type(raw_kind) is not int:
                raise ValueError()
            kind = RigTokenKindV6(raw_kind)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("RSS_VNEXT_MIRA_GROUP_EVENT_KIND_INVALID") from exc
        pa = row.get("pointer_a", -1)
        pb = row.get("pointer_b", -1)
        if type(pa) is not int or type(pb) is not int:
            raise ValueError("RSS_VNEXT_MIRA_GROUP_POINTER_TYPE_INVALID")
        converted.append(RigProgramEventV6(
            kind=kind, pointer_a=pa, pointer_b=pb,
        ))
    sequence = AXISSequenceV6.from_events(
        converted, device=torch.device("cpu"),
    )
    return axis_joint_group_ids_v1(sequence)


class RSSvNextLearnedChainMixtureV1(nn.Module):
    """Trainable group gate over hard AXIS CHAIN topology; no second skin head."""

    def __init__(
        self, *, vertex_feature_dim: int, joint_feature_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        if min(vertex_feature_dim, joint_feature_dim, hidden_dim) < 1:
            raise ValueError("RSS_VNEXT_MIRA_GROUP_DIM_INVALID")
        self.vertex_projection = nn.Sequential(
            nn.LayerNorm(vertex_feature_dim),
            nn.Linear(vertex_feature_dim, hidden_dim),
            nn.GELU(),
        )
        self.joint_projection = nn.Sequential(
            nn.LayerNorm(joint_feature_dim),
            nn.Linear(joint_feature_dim, hidden_dim),
            nn.GELU(),
        )
        self.group_pair = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, *, joint_logits: Tensor,
        vertex_features: Tensor, joint_features: Tensor,
        joint_group_ids: Tensor,
    ) -> Tensor:
        if (
            not isinstance(joint_logits, Tensor)
            or joint_logits.ndim != 2
            or joint_logits.numel() == 0
            or vertex_features.ndim != 2
            or joint_features.ndim != 2
            or vertex_features.shape[0] != joint_logits.shape[0]
            or joint_features.shape[0] != joint_logits.shape[1]
            or vertex_features.shape[1]
                != self.vertex_projection[0].normalized_shape[0]
            or joint_features.shape[1]
                != self.joint_projection[0].normalized_shape[0]
            or joint_group_ids.shape != (joint_logits.shape[1],)
            or joint_group_ids.dtype != torch.long
            or any(x.device != joint_logits.device for x in (
                vertex_features, joint_features, joint_group_ids,
            ))
            or not torch.is_floating_point(joint_logits)
            or not bool(torch.isfinite(joint_logits).all())
            or not bool(torch.isfinite(vertex_features).all())
            or not bool(torch.isfinite(joint_features).all())
            or bool((joint_group_ids < -1).any())
        ):
            raise ValueError("RSS_VNEXT_MIRA_GROUP_ADDRESS_INVALID")
        _, inverse = torch.unique(
            joint_group_ids, sorted=True, return_inverse=True,
        )
        group_count = int(inverse.max().item()) + 1
        if group_count == 1:
            # An unqualified rig has no semantic or chain mass truth.
            return F.softmax(joint_logits, dim=-1)
        vertex_h = self.vertex_projection(vertex_features)
        joint_h = self.joint_projection(joint_features)
        joint_indices=[]
        group_context=[]
        group_log_mass=[]
        conditional=[]
        for group_index in range(group_count):
            indices=torch.where(inverse == group_index)[0]
            joint_indices.append(indices)
            group_context.append(joint_h.index_select(0,indices).mean(dim=0))
            chunk=joint_logits.index_select(1,indices)
            group_log_mass.append(torch.logsumexp(chunk,dim=-1))
            conditional.append(torch.softmax(chunk,dim=-1))
        gh=torch.stack(group_context,dim=0)
        n=vertex_h.shape[0]
        k=gh.shape[0]
        pair=torch.cat((
            vertex_h[:,None,:].expand(n,k,-1),
            gh[None,:,:].expand(n,k,-1),
        ),dim=-1)
        learned_group_logit=self.group_pair(pair).squeeze(-1)
        # The logsumexp baseline is important: with zero learned correction
        # this exactly recovers the flat joint softmax; the group-specific
        # network must LEARN deviations rather than obtaining a group prior
        # solely from arbitrary chain sizes.
        gate=F.softmax(
            torch.stack(group_log_mass,dim=-1)+learned_group_logit,
            dim=-1,
        )
        out=torch.zeros_like(joint_logits)
        for group_index,(indices,distribution) in enumerate(
            zip(joint_indices,conditional)
        ):
            out[:,indices]=gate[:,group_index,None]*distribution
        if (
            not bool(torch.isfinite(out).all())
            or not bool(torch.allclose(
                out.sum(dim=-1),torch.ones_like(out[:,0]),
                atol=1.e-5,rtol=0.,
            ))
        ):
            raise RuntimeError("RSS_VNEXT_MIRA_GROUP_MIXTURE_SIMPLEX_DRIFT")
        return out


__all__ = [
    "axis_joint_group_ids_v1",
    "axis_joint_group_ids_from_cas_events_v1",
    "RSSvNextLearnedChainMixtureV1",
]
