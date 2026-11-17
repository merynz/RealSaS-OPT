from __future__ import annotations

"""Per-component open/closed UDF field queries from ONE IRIS H(x).

The existing global unsigned_distance field measures distance to the nearest
surface anywhere in the character and CANNOT uniquely isolate a cloak.
This head is conditioned on learned exchangeable region slots from the SAME
H(x), not an independent model or external semantic/corpus label.

Slot identity is NOT source truth. Teacher matching is permutation-invariant,
discrete and source-qualified. All output is evidence proposals; topology,
open/closed admission, M0 and final mesh authority remain with Compiler.
"""

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from typing import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .native_face_layer_truth_v1 import tensor_sha256_v1

SCHEMA = "RealSaS.RSSvNext.ComponentConditionedUDF.v1"
TRAIN_SCHEMA = "RealSaS.RSSvNext.SourceNativeComponentUDFAssignment.v1"
KIND_CLOSED = 0
KIND_OPEN = 1
KIND_UNKNOWN = 2


class RSSvNextComponentConditionedUDFV1(nn.Module):
    """Distance to each learned candidate region, sharing scene H(x).

    Region slots come from RSSvNextGenericRegionStructure; slot index is an
    exchangeable hypothesis and not automatically a persistent part ID.
    """

    def __init__(self, width: int, *, max_query_slot_pairs: int = 32768):
        super().__init__()
        if (
            type(width) is not int or width<8
            or type(max_query_slot_pairs) is not int
            or not 128<=max_query_slot_pairs<=262144
        ):
            raise ValueError("RSS_VNEXT_COMPONENT_UDF_CONFIG")
        self.width=width
        self.max_query_slot_pairs=max_query_slot_pairs
        self.h_norm=nn.LayerNorm(width)
        self.slot_norm=nn.LayerNorm(width)
        self.distance=nn.Sequential(
            nn.Linear(3*width,width),nn.GELU(),
            nn.Linear(width,width//2),nn.GELU(),
            nn.Linear(width//2,1),
        )
        self.kind=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,3))

    def _validate(self, point_h:Tensor, region_slots:Tensor)->tuple[int,int,int,int]:
        if not isinstance(point_h,Tensor) or not isinstance(region_slots,Tensor):
            raise ValueError("RSS_VNEXT_COMPONENT_UDF_FIELD_TENSOR")
        if (
            point_h.ndim!=3 or region_slots.ndim!=3
            or point_h.shape[0]!=region_slots.shape[0]
            or point_h.shape[2]!=self.width
            or region_slots.shape[2]!=self.width
            or not point_h.shape[1]
            or not region_slots.shape[1]
            or region_slots.shape[1]>256
            or point_h.device!=region_slots.device
            or point_h.dtype!=region_slots.dtype
            or not bool(torch.isfinite(point_h).all())
            or not bool(torch.isfinite(region_slots).all())
        ):
            raise ValueError("RSS_VNEXT_COMPONENT_UDF_SHARED_H_DOMAIN")
        return point_h.shape[0],point_h.shape[1],region_slots.shape[1],self.width

    def topology_kind_logits(self,region_slots:Tensor)->Tensor:
        if (region_slots.ndim!=3 or region_slots.shape[-1]!=self.width
            or not bool(torch.isfinite(region_slots).all())):
            raise ValueError("RSS_VNEXT_COMPONENT_UDF_KIND_DOMAIN")
        return self.kind(region_slots)

    def query_slot(
        self,point_h:Tensor,region_slots:Tensor,*,slot_index:int,
    ) -> Tensor:
        """[B,Q] nonnegative from one selected region; gradients to H+slot."""
        b,q,r,h=self._validate(point_h,region_slots)
        if type(slot_index) is not int or not 0<=slot_index<r:
            raise ValueError("RSS_VNEXT_COMPONENT_UDF_SLOT_INDEX")
        slot=self.slot_norm(region_slots[:,slot_index])
        pieces=[]
        # Query chunk bound applies to B*Q*R in all-slot mode and B*Q
        # here; never evaluate all high-res points in one activation block.
        step=max(1,self.max_query_slot_pairs//b)
        for start in range(0,q,step):
            hn=self.h_norm(point_h[:,start:start+step])
            s=slot[:,None].expand(-1,hn.shape[1],-1)
            values=self.distance(torch.cat((hn,s,hn*s),dim=-1))
            pieces.append(F.softplus(values).squeeze(-1))
        return torch.cat(pieces,dim=1)

    def query_all(self,point_h:Tensor,region_slots:Tensor)->Tensor:
        """[B,Q,R]. Chunked per slot to bound memory; no semantic argmax."""
        _,_,r,_=self._validate(point_h,region_slots)
        return torch.stack([
            self.query_slot(point_h,region_slots,slot_index=i)
            for i in range(r)
        ],dim=-1)


@dataclass(frozen=True)
class SourceNativeComponentUDFFitV1:
    source_asset_sha256: str
    source_qualification_key: str
    original_query_xyz: Tensor
    source_component_udf: Tensor           # [C,Q] native per-component UDF
    source_component_mask: Tensor          # [C,Q] independently qualified
    source_component_open_kind: Tensor     # [C] int CLOSED / OPEN (no unknown)
    native_component_inventory_sha256: str
    independently_complete_inventory: bool = False

    def validate(self) -> None:
        x,d,m,k=self.original_query_xyz,self.source_component_udf,self.source_component_mask,self.source_component_open_kind
        if (
            not all(isinstance(v,Tensor) for v in (x,d,m,k))
            or x.ndim!=2 or x.shape[-1]!=3 or not x.shape[0]
            or d.ndim!=2 or not 1<=d.shape[0]<=64
            or d.shape[1]!=len(x)
            or m.shape!=d.shape or m.dtype!=torch.bool
            or k.shape==(0,) or k.shape!=(len(d),)
            or k.dtype!=torch.long
            or not bool(m.any(dim=1).all())
            or bool(((k!=KIND_CLOSED)&(k!=KIND_OPEN)).any())
            or any(v.device!=x.device for v in (d,m,k))
            or d.dtype!=x.dtype
            or not bool(torch.isfinite(x).all())
            or not bool(torch.isfinite(d).all())
            or bool((d<0).any())
            or any(type(h) is not str or len(h)!=64
                   or any(ch not in "0123456789abcdef" for ch in h)
                   for h in (self.source_asset_sha256,self.native_component_inventory_sha256))
            or type(self.source_qualification_key) is not str
            or not self.source_qualification_key
            or type(self.independently_complete_inventory) is not bool
        ):
            raise ValueError("RSS_VNEXT_COMPONENT_UDF_NATIVE_TEACHER_DOMAIN")


def _hungarian_min_injective_v1(cost:Tensor)->tuple[int,...]:
    """Deterministic rectangular minimum-cost matching (C <= R).

    Hungarian shortest augmenting path on detached CPU cost only. The loss
    itself is evaluated from the ORIGINAL differentiable torch cost entries.
    No source component ID is ever assigned to a fixed predicted slot.
    """
    if cost.ndim!=2 or not 1<=cost.shape[0]<=cost.shape[1]<=256:
        raise ValueError("RSS_VNEXT_COMPONENT_UDF_MATCH_SHAPE")
    if not bool(torch.isfinite(cost).all()):
        raise ValueError("RSS_VNEXT_COMPONENT_UDF_MATCH_NONFINITE")
    a=cost.detach().double().cpu().tolist()
    n,m=len(a),len(a[0])
    u=[0.]*(n+1);v=[0.]*(m+1)
    p=[0]*(m+1);way=[0]*(m+1)
    for i in range(1,n+1):
        p[0]=i
        j0=0
        minv=[float("inf")]*(m+1)
        used=[False]*(m+1)
        while True:
            used[j0]=True
            i0=p[j0];delta=float("inf");j1=-1
            for j in range(1,m+1):
                if used[j]:
                    continue
                cur=a[i0-1][j-1]-u[i0]-v[j]
                if cur<minv[j]:
                    minv[j]=cur;way[j]=j0
                if minv[j]<delta:
                    delta=minv[j];j1=j
            if not math.isfinite(delta) or j1<0:
                raise ValueError("RSS_VNEXT_COMPONENT_UDF_MATCH_NO_ASSIGNMENT")
            for j in range(m+1):
                if used[j]:
                    u[p[j]]+=delta;v[j]-=delta
                else:
                    minv[j]-=delta
            j0=j1
            if p[j0]==0:
                break
        while True:
            j1=way[j0]
            p[j0]=p[j1]
            j0=j1
            if j0==0:
                break
    assignment=[-1]*n
    for j in range(1,m+1):
        if p[j]!=0:
            assignment[p[j]-1]=j-1
    if any(x<0 for x in assignment) or len(set(assignment))!=n:
        raise ValueError("RSS_VNEXT_COMPONENT_UDF_ASSIGNMENT_INVALID")
    return tuple(assignment)


def source_native_component_udf_objective_v1(
    *, predicted_distances:Tensor, predicted_kind_logits:Tensor,
    teacher:SourceNativeComponentUDFFitV1,
    trusted_source_registry:Mapping[str,Mapping[str,object]],
    source_query_xyz:Tensor,
    kind_cost_weight:float=.15,
    unmatched_unknown_weight:float=.05,
) -> dict[str,Tensor|tuple[int,...]]:
    """Permutation-invariant source-qualified component distance + kind loss.

    A teacher defines exact source components, NEVER predicted slot IDs.
    The slot matching is minimum cost on detached predictions and is
    recomputed per step; gradients go to matched predicted fields and slots.
    Unknown slots are penalized only when native component inventory is
    independently COMPLETE, never when data coverage is partial.
    """
    teacher.validate()
    if (
        predicted_distances.ndim!=3 or predicted_distances.shape[0]!=1
        or predicted_kind_logits.ndim!=3 or predicted_kind_logits.shape[0]!=1
        or predicted_kind_logits.shape[-1]!=3
        or predicted_distances.shape[1]!=len(teacher.original_query_xyz)
        or predicted_kind_logits.shape[1]!=predicted_distances.shape[2]
        or not teacher.source_component_udf.shape[0]<=predicted_distances.shape[2]
        or predicted_distances.device!=teacher.original_query_xyz.device
        or predicted_kind_logits.device!=teacher.original_query_xyz.device
        or predicted_distances.dtype!=teacher.original_query_xyz.dtype
        or predicted_kind_logits.dtype!=teacher.original_query_xyz.dtype
        or source_query_xyz.shape!=teacher.original_query_xyz.shape
        or not bool(torch.equal(source_query_xyz,teacher.original_query_xyz))
        or any(not bool(torch.isfinite(x).all())
               for x in (predicted_distances,predicted_kind_logits))
        or bool((predicted_distances<0).any())
        or type(kind_cost_weight) not in (float,int)
        or not 0<=kind_cost_weight<=1
        or type(unmatched_unknown_weight) not in (float,int)
        or not 0<=unmatched_unknown_weight<=1
    ):
        raise ValueError("RSS_VNEXT_COMPONENT_UDF_TRAIN_DOMAIN")
    receipt=trusted_source_registry.get(teacher.source_qualification_key)
    qualified={
        "schema":TRAIN_SCHEMA,
        "admission_status":"QUALIFIED_NATIVE_COMPONENT_UDF_AND_OPENNESS",
        "source_asset_sha256":teacher.source_asset_sha256,
        "native_component_inventory_sha256":teacher.native_component_inventory_sha256,
        "original_query_xyz_sha256":tensor_sha256_v1(teacher.original_query_xyz),
        "source_component_udf_sha256":tensor_sha256_v1(teacher.source_component_udf),
        "source_component_mask_sha256":tensor_sha256_v1(teacher.source_component_mask),
        "source_component_open_kind_sha256":tensor_sha256_v1(teacher.source_component_open_kind),
        "independently_complete_inventory":teacher.independently_complete_inventory,
        "source_rights_independently_verified":True,
        "generated_views_as_geometry_teacher":False,
        "student_slot_ids_as_source_truth":False,
    }
    if not isinstance(receipt,Mapping) or any(
        receipt.get(k)!=val for k,val in qualified.items()
    ):
        raise ValueError("RSS_VNEXT_COMPONENT_UDF_TRUSTED_SOURCE_RECEIPT_REQUIRED")
    c,q=teacher.source_component_udf.shape
    r=predicted_distances.shape[-1]
    # Cost matrix [C,R] from source matched query locations. Even when
    # two components overlap in XYZ their independently measured UDF targets
    # remain distinct; no nearest-body label leaks into cape UDF.
    delta=F.smooth_l1_loss(
        predicted_distances[0].transpose(0,1).unsqueeze(0),
        teacher.source_component_udf.unsqueeze(1),
        reduction="none",
    )  # [C,R,Q]
    valid=teacher.source_component_mask[:,None].to(delta.dtype)
    geometric=(delta*valid).sum(-1)/valid.sum(-1).clamp_min(1)
    log_probs=F.log_softmax(predicted_kind_logits[0],dim=-1)
    classification= -log_probs[
        None,torch.arange(r,device=log_probs.device)[None,:],
        teacher.source_component_open_kind[:,None],
    ]
    # Broadcasted advanced index -> [C,R]
    pair=geometric+kind_cost_weight*classification
    slots=_hungarian_min_injective_v1(pair)
    ids=torch.tensor(slots,device=pair.device,dtype=torch.long)
    selected=pair[torch.arange(c,device=pair.device),ids]
    target_unknown=predicted_kind_logits.new_zeros(())
    if teacher.independently_complete_inventory:
        unused=torch.tensor(
            [j for j in range(r) if j not in slots],
            device=pair.device,dtype=torch.long,
        )
        if len(unused):
            labels=torch.full(
                (len(unused),),KIND_UNKNOWN,
                device=pair.device,dtype=torch.long,
            )
            target_unknown=F.cross_entropy(
                predicted_kind_logits[0,unused],labels,
            )
    total=selected.mean()+unmatched_unknown_weight*target_unknown
    if not bool(torch.isfinite(total).all()):
        raise ValueError("RSS_VNEXT_COMPONENT_UDF_NONFINITE_LOSS")
    return {
        "total":total,
        "field":geometric[torch.arange(c,device=pair.device),ids].mean(),
        "kind":classification[torch.arange(c,device=pair.device),ids].mean(),
        "unmatched_unknown":target_unknown,
        "matched_predicted_slots":slots,
        "qualified_source_component_count":predicted_distances.new_tensor(c),
    }


__all__=[
    "SCHEMA","TRAIN_SCHEMA","KIND_CLOSED","KIND_OPEN","KIND_UNKNOWN",
    "RSSvNextComponentConditionedUDFV1",
    "SourceNativeComponentUDFFitV1",
    "source_native_component_udf_objective_v1",
]
