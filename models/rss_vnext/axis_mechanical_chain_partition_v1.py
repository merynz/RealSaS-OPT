from __future__ import annotations

"""AXIS V6 causal chain program -> deterministic joint ownership proposal.

This is *not* semantic classification and never invents a chain from an image,
a source bone name or a mesh component. An AXIS CHAIN event explicitly names a
start/end path in its own causal JOINT parent graph. A joint covered by more
than one declared chain stays UNKNOWN (-1), as does a joint with no chain.
The returned indices refer to causal joint order, shared by the MIRA codec.

Source-qualified teachers may supervise these exact mechanical identities;
a malformed or underdefined program must not silently assign labels.
"""

from dataclasses import dataclass
import torch
from torch import Tensor

from models.axis.rig_program_model_v6 import AXISSequenceV6, RigTokenKindV6


@dataclass(frozen=True)
class AXISJointChainPartitionV1:
    joint_chain_ids: Tensor    # [J] int64, -1 = no unique ownership
    parent_joint_indices: tuple[int, ...]
    causal_chain_paths: tuple[tuple[int, ...], ...]
    uncertain_joint_count: int
    authority: str = "DETERMINISTIC_AXIS_DECLARED_CHAIN_GRAPH_NOT_SEMANTIC_TRUTH"


def decode_axis_joint_chain_partition_v1(
    seq: AXISSequenceV6, *, batch_index: int = 0,
) -> AXISJointChainPartitionV1:
    if (
        not isinstance(seq, AXISSequenceV6)
        or seq.kind_ids.ndim != 2
        or not 0 <= batch_index < seq.kind_ids.shape[0]
        or any(x.shape != seq.kind_ids.shape for x in (
            seq.pointer_a, seq.pointer_b, seq.attention_mask,
        ))
    ):
        raise ValueError("RSS_VNEXT_CHAIN_SEQUENCE_DOMAIN_INVALID")
    kind = seq.kind_ids[batch_index]
    valid = seq.attention_mask[batch_index].bool()
    joint_tokens = torch.where(valid & (kind == int(RigTokenKindV6.JOINT)))[0].tolist()
    chain_tokens = torch.where(valid & (kind == int(RigTokenKindV6.CHAIN)))[0].tolist()
    if not joint_tokens:
        raise ValueError("RSS_VNEXT_CHAIN_NO_AXIS_JOINTS")
    parents: list[int] = []
    for token in joint_tokens:
        p = int(seq.pointer_a[batch_index, token])
        if p < -1 or p >= len(parents):
            raise ValueError("RSS_VNEXT_CHAIN_JOINT_PARENT_NOT_CAUSAL")
        parents.append(p)
    paths: list[tuple[int, ...]] = []
    for token in chain_tokens:
        a = int(seq.pointer_a[batch_index, token])
        b = int(seq.pointer_b[batch_index, token])
        if not 0 <= a < len(joint_tokens) or not 0 <= b < len(joint_tokens):
            raise ValueError("RSS_VNEXT_CHAIN_PATH_JOINT_POINTER_INVALID")
        # Traverse from end up through the same causal tree; forbid invented
        # links across disconnected or sibling chains.
        backwards = [b]
        curr = b
        while curr != a:
            curr = parents[curr]
            if curr < 0:
                raise ValueError("RSS_VNEXT_CHAIN_NONANCESTRAL_PATH")
            backwards.append(curr)
        paths.append(tuple(reversed(backwards)))
    membership: list[list[int]] = [[] for _ in joint_tokens]
    for chain_id, path in enumerate(paths):
        for joint in path:
            membership[joint].append(chain_id)
    # A shared root may be a legitimate attachment for many chains but its
    # mechanical *ownership* is ambiguous. Never silently choose first path.
    ids = [links[0] if len(links) == 1 else -1 for links in membership]
    return AXISJointChainPartitionV1(
        joint_chain_ids=torch.tensor(ids,dtype=torch.long,device=seq.kind_ids.device),
        parent_joint_indices=tuple(parents),
        causal_chain_paths=tuple(paths),
        uncertain_joint_count=sum(v < 0 for v in ids),
    )


def assert_qualified_chain_partition_matches_axis_v1(
    seq: AXISSequenceV6,
    source_joint_chain_ids: Tensor,
) -> AXISJointChainPartitionV1:
    """Compare partitions, not arbitrary source chain *names/IDs*.

    A source may call two chains 13 and 91; AXIS emits them in causal order.
    Both must agree on joint grouping and exactly which joints are UNKNOWN.
    """
    derived = decode_axis_joint_chain_partition_v1(seq)
    a = source_joint_chain_ids
    b = derived.joint_chain_ids
    if (
        not isinstance(a, Tensor) or a.shape != b.shape
        or a.dtype != torch.long or a.device != b.device
        or bool((a < -1).any())
    ):
        raise ValueError("RSS_VNEXT_CHAIN_SOURCE_PARTITION_DOMAIN_INVALID")
    known_a = a >= 0
    known_b = b >= 0
    if not torch.equal(known_a, known_b):
        raise ValueError("RSS_VNEXT_CHAIN_SOURCE_VS_AXIS_UNKNOWN_OWNERSHIP")
    if bool(known_a.any()):
        # Vectorized equivalence of the partition relation: joint labels are
        # source-defined and must never become global morphology classes.
        x = a[known_a]
        y = b[known_b]
        if not torch.equal(x[:, None] == x[None, :], y[:, None] == y[None, :]):
            raise ValueError("RSS_VNEXT_CHAIN_SOURCE_VS_AXIS_GROUPING_DRIFT")
    return derived


__all__ = [
    "AXISJointChainPartitionV1",
    "decode_axis_joint_chain_partition_v1",
    "assert_qualified_chain_partition_matches_axis_v1",
]
