from __future__ import annotations

"""Deterministic component-preserving surface supervision sampling.

Full indexed topology is never downsampled or reordered in the student.
Only the SOURCE-QUALIFIED LOSS QUERY SET is bounded. Sampling varies with
checkpointed global_step; each component keeps explicit representation.
Morton spatial order avoids the common failure of slicing contiguous source
vertices (which biases glTF authoring/part order). Discrete sampling choices
never feed teacher geometry into the neural forward.
"""

from dataclasses import dataclass, asdict
from hashlib import sha256
import json
import math

import torch
from torch import Tensor


SCHEMA="RealSaS.RSSvNext.SeededSurfaceSampling.v1"


@dataclass(frozen=True)
class SeededSurfaceSamplingPolicyV1:
    max_student_vertices: int = 8192
    max_source_points: int = 8192
    max_quality_faces: int = 16384
    min_samples_per_component: int = 3

    def validate(self) -> None:
        if (
            any(type(v) is not int for v in asdict(self).values())
            or not 256 <= self.max_student_vertices <= 65536
            or not 256 <= self.max_source_points <= 65536
            or not 256 <= self.max_quality_faces <= 131072
            or not 1 <= self.min_samples_per_component <= 8
        ):
            raise ValueError("RSS_VNEXT_SEEDED_SAMPLING_POLICY_INVALID")

    def digest(self) -> str:
        self.validate()
        return sha256(json.dumps(
            {"schema":SCHEMA,**asdict(self)},
            sort_keys=True,separators=(",",":"),
        ).encode()).hexdigest()


def _morton_3d_v1(xyz: Tensor) -> Tensor:
    """30-bit integer Morton key from detached positions, no source labels."""
    with torch.no_grad():
        finite=bool(torch.isfinite(xyz).all())
        if not finite:
            raise ValueError("RSS_VNEXT_SEEDED_SAMPLE_NONFINITE")
        p=xyz.detach().float()
        lower=p.amin(dim=0)
        span=(p.amax(dim=0)-lower).clamp_min(1.e-8)
        q=((p-lower)/span * 1023.0).clamp(0,1023).long()
        key=torch.zeros(len(q),device=q.device,dtype=torch.long)
        for bit in range(10):
            mask=1<<bit
            key = key | ((q[:,0]&mask)<<(2*bit))
            key = key | ((q[:,1]&mask)<<(2*bit+1))
            key = key | ((q[:,2]&mask)<<(2*bit+2))
        return key


def _component_quotas(counts: list[int], *, budget: int, minimum: int) -> list[int]:
    minimums=[min(n,minimum) for n in counts]
    if sum(minimums)>budget:
        raise ValueError("RSS_VNEXT_SEEDED_COMPONENT_COUNT_EXCEEDS_LOSS_BUDGET")
    quotas=minimums[:]
    outstanding=budget-sum(quotas)
    # Weighted by sqrt(component size): a large body does not starve hair or
    # a small cape, while larger components retain denser coverage.
    while outstanding:
        capacity=[n-k for n,k in zip(counts,quotas)]
        active=[i for i,n in enumerate(capacity) if n>0]
        if not active:
            break
        weights=[math.sqrt(capacity[i]) for i in active]
        total=sum(weights)
        desired=[outstanding*w/total for w in weights]
        gained=0
        for i,extra in zip(active,desired):
            number=min(capacity[i],int(extra))
            quotas[i]+=number
            gained+=number
        outstanding-=gained
        if outstanding==0:
            break
        # Largest fractional remainders; deterministic tie break by component ID.
        # A saturated component does not consume the spare budget.
        ranking=sorted(
            zip(active,desired), key=lambda t:(-(t[1]-int(t[1])),t[0]),
        )
        for i,_ in ranking:
            if outstanding==0:
                break
            if quotas[i]<counts[i]:
                quotas[i]+=1
                outstanding-=1
    return quotas


@torch.no_grad()
def component_spatial_sample_indices_v1(
    xyz: Tensor,
    component_ids: Tensor,
    *,
    budget: int,
    step: int,
    minimum: int = 3,
) -> Tensor:
    """An indexed, coordinate-deterministic subset with full component recall.

    The entire rest mesh is still used for AXIS, MIRA and Graphics; this
    sampling only limits supervision queries and keeps original vertex IDs.
    Deterministic phase shifts rotate across neighboring Morton addresses.
    """
    if (
        not isinstance(xyz,Tensor) or xyz.ndim!=2 or xyz.shape[-1]!=3
        or not 1<=len(xyz)<=2_000_000
        or not isinstance(component_ids,Tensor)
        or component_ids.shape!=(len(xyz),)
        or component_ids.dtype!=torch.long
        or xyz.device!=component_ids.device
        or bool((component_ids<0).any())
        or type(budget) is not int or not 1<=budget<=131072
        or type(step) is not int or step<0
        or type(minimum) is not int or not 1<=minimum<=8
    ):
        raise ValueError("RSS_VNEXT_SEEDED_SAMPLE_INPUT_DOMAIN")
    if len(xyz)<=budget:
        return torch.arange(len(xyz),device=xyz.device,dtype=torch.long)
    components,inverse,counts=torch.unique(
        component_ids,sorted=True,return_inverse=True,return_counts=True,
    )
    if len(components)>budget:
        raise ValueError("RSS_VNEXT_SEEDED_COMPONENT_COUNT_EXCEEDS_LOSS_BUDGET")
    counts_list=counts.tolist()
    quotas=_component_quotas(
        counts_list,budget=budget,minimum=minimum,
    )
    # One global sort: never O(C * V) boolean filtering for many components.
    morton=_morton_3d_v1(xyz)
    key=inverse.long()*(1<<30)+morton
    order=torch.argsort(key,stable=True)
    chosen=[]
    begin=0
    for count,quota in zip(counts_list,quotas):
        # Stratify uniformly in spatial order, rotate each step so previously
        # unsampled detail receives gradient over successive checkpoints.
        base=torch.div(
            (torch.arange(quota,device=xyz.device,dtype=torch.long)*2+1)
            *count, 2*quota,rounding_mode="floor",
        )
        phase=(step % count)
        local=torch.remainder(base+phase,count)
        chosen.append(order[begin+local])
        begin+=count
    result=torch.cat(chosen)
    if result.numel()!=min(len(xyz),budget) or int(torch.unique(result).numel())!=result.numel():
        raise RuntimeError("RSS_VNEXT_SEEDED_SAMPLE_VERTEX_DUPLICATION_OR_BUDGET")
    return result


__all__=[
    "SCHEMA","SeededSurfaceSamplingPolicyV1",
    "component_spatial_sample_indices_v1",
]
