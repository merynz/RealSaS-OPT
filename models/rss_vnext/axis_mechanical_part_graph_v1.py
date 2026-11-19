from __future__ import annotations

"""Source-qualified causal AXIS mechanical PART connectivity over CHAIN paths.

The existing AXIS V6 CHAIN event has a *previously unused* pointer_c field.
It is now a typed MECHANICAL_PART address:
  -1 = UNKNOWN (not a negative part label)
   0 = create the first part, when no earlier part exists
   p = reference an existing part, or create next p exactly once

An existing part can only gain a chain if its joint path overlaps a member
chain. This permits tree-branching groups (e.g. wing spars, skirt pleats)
without calling a branched part an ancestor->end CHAIN and without inventing
new attachment edges. Semantics (WING/SKIRT/HAIR) are NOT inferred or supplied
by this grammar. A part ordinal is local to one emitted rig program.

The group map fed to MIRA is deterministic from exactly those learned events.
Shared joints belonging to two DIFFERENT parts remain UNKNOWN, not reparented.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from models.axis.rig_program_model_v6 import (
    AXISSequenceV6, RigTokenKindV6,
)
from .axis_mechanical_chain_partition_v1 import (
    decode_axis_joint_chain_partition_v1,
)


@dataclass(frozen=True)
class AXISMechanicalPartGraphV1:
    part_by_chain: tuple[int, ...]
    chain_paths: tuple[tuple[int, ...], ...]
    joint_group_ids: Tensor
    declared_part_count: int
    unknown_joint_count: int
    authority: str = "AXIS_CAUSAL_PART_GRAPH_PROPOSAL_NOT_SEMANTIC_TRUTH"


def decode_axis_mechanical_part_graph_v1(
    seq: AXISSequenceV6, *, batch_index: int = 0,
) -> AXISMechanicalPartGraphV1:
    partition=decode_axis_joint_chain_partition_v1(
        seq,batch_index=batch_index,
    )
    kinds=seq.kind_ids[batch_index]
    valid=seq.attention_mask[batch_index].bool()
    chain_tokens=torch.where(valid & (kinds==int(RigTokenKindV6.CHAIN)))[0].tolist()
    if not torch.equal(
        torch.tensor(len(chain_tokens)),torch.tensor(len(partition.causal_chain_paths))
    ):
        raise ValueError("AXIS_V6_PART_CHAIN_COUNT_DRIFT")
    raw_group_ids=[]
    part_count=0
    member_paths:dict[int,set[int]]={}
    for chain_ordinal,(token,path) in enumerate(
        zip(chain_tokens,partition.causal_chain_paths)
    ):
        part=int(seq.pointer_c[batch_index,token])
        if part < -1:
            raise ValueError("AXIS_V6_PART_POINTER_DOMAIN_INVALID")
        if part>=0:
            if part>part_count:
                raise ValueError("AXIS_V6_PART_NONCAUSAL_GROUP_ADDRESS")
            if part==part_count:
                part_count+=1
                member_paths[part]=set(path)
            else:
                if not (set(path) & member_paths[part]):
                    raise ValueError("AXIS_V6_PART_GROUP_DISCONNECTED_CHAIN")
                member_paths[part].update(path)
        raw_group_ids.append(part)
    # Chain ordinals not assigned to a part are explicit UNKNOWN ownership,
    # but still become separate local mechanical groups for skin decoding.
    # Shared joints are owned only if all their memberships resolve to the
    # SAME part (the intended multi-branch wing or cloth component).
    membership:list[set[tuple[str,int]]]=[
        set() for _ in partition.parent_joint_indices
    ]
    for chain_index,(part,path) in enumerate(
        zip(raw_group_ids,partition.causal_chain_paths)
    ):
        tag=("PART",part) if part>=0 else ("UNKNOWN_CHAIN",chain_index)
        for joint in path:
            membership[joint].add(tag)
    # Stable local ordinal order: earlier observed group wins its integer ID
    # only in the naming namespace; ambiguity still remains UNKNOWN (-1).
    names:dict[tuple[str,int],int]={}
    for chain_index,part in enumerate(raw_group_ids):
        tag=("PART",part) if part>=0 else ("UNKNOWN_CHAIN",chain_index)
        names.setdefault(tag,len(names))
    group_ids=[
        names[next(iter(members))] if len(members)==1 else -1
        for members in membership
    ]
    out=torch.tensor(
        group_ids,dtype=torch.long,device=seq.kind_ids.device,
    )
    return AXISMechanicalPartGraphV1(
        part_by_chain=tuple(raw_group_ids),
        chain_paths=partition.causal_chain_paths,
        joint_group_ids=out,
        declared_part_count=part_count,
        unknown_joint_count=int((out<0).sum().item()),
    )


def candidate_axis_part_pointer_choices_v1(
    *, parents:Sequence[int],
    existing_chains:Sequence[tuple[int,int,int]],
    start_joint:int,end_joint:int,
) -> tuple[int,...]:
    """Possible pointer_c values for a NEW CHAIN, including -1 UNKNOWN.

    Runtime never repairs an impossible disconnected group; it masks
    unsupported part references before sampling from the existing
    AXIS pointer_c head.
    """
    if (
        not parents or type(start_joint) is not int
        or type(end_joint) is not int
        or not 0<=start_joint<len(parents)
        or not 0<=end_joint<len(parents)
    ):
        raise ValueError("AXIS_V6_PART_CANDIDATE_JOINT_DOMAIN_INVALID")
    def path(start:int,end:int)->set[int]:
        seen=set()
        cur=end
        while cur>=0 and cur not in seen:
            seen.add(cur)
            if cur==start:
                return seen
            cur=parents[cur]
        raise ValueError("AXIS_V6_PART_NONANCESTRAL_CHAIN")
    current_path=path(start_joint,end_joint)
    parts:dict[int,set[int]]={}
    count=0
    for a,b,p in existing_chains:
        prior_path=path(a,b)
        if p>=0:
            if p>count:
                raise ValueError("AXIS_V6_PART_NONCAUSAL_GROUP_ADDRESS")
            if p==count:
                parts[p]=set(prior_path)
                count+=1
            else:
                if not (parts[p] & prior_path):
                    raise ValueError("AXIS_V6_PART_GROUP_DISCONNECTED_CHAIN")
                parts[p].update(prior_path)
    return tuple([-1]+[
        p for p in range(count) if parts[p]&current_path
    ]+[count])


__all__=[
    "AXISMechanicalPartGraphV1",
    "decode_axis_mechanical_part_graph_v1",
    "candidate_axis_part_pointer_choices_v1",
]
