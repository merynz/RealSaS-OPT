from __future__ import annotations

"""Deterministic subject-free adequacy selection for RiggingSurfaceIR.

The dense signed zero-surface is geometry authority. This module chooses a bounded
mechanical carrier density; it never changes dense geometry or invents semantics.
"""

from dataclasses import asdict
import math
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from ..hashing import content_sha256
from ..types import QualificationError
from .scene_first_signed import (
    _self_zbuffer_support,
    rigging_surface_from_scene_first_zero_mesh_v1,
)

Json=dict[str,Any]


def substrate_adequacy_report_hash_v1(report:dict)->str:
    payload=dict(report)
    payload.pop("adequacy_report_hash",None)
    return content_sha256(payload)


def _dense_component_labels(vertex_count:int,faces:np.ndarray)->np.ndarray:
    parent=np.arange(int(vertex_count),dtype=np.int64)
    rank=np.zeros(int(vertex_count),dtype=np.int8)

    def find(x:int)->int:
        while parent[x]!=x:
            parent[x]=parent[parent[x]]
            x=int(parent[x])
        return x

    def union(a:int,b:int)->None:
        ra,rb=find(a),find(b)
        if ra==rb: return
        if rank[ra]<rank[rb]:
            ra,rb=rb,ra
        parent[rb]=ra
        if rank[ra]==rank[rb]:
            rank[ra]+=1

    for a,b,c in np.asarray(faces,dtype=np.int64):
        union(int(a),int(b)); union(int(b),int(c)); union(int(c),int(a))
    roots=np.asarray([find(i) for i in range(int(vertex_count))],dtype=np.int64)
    _,labels=np.unique(roots,return_inverse=True)
    return labels.astype(np.int64)


def _candidate_caps(policy:dict)->tuple[int,...]:
    lo=int(policy["min_candidate_nodes"])
    hi=int(policy["max_candidate_nodes"])
    growth=float(policy["candidate_growth_factor"])
    if lo<64 or hi<lo or not math.isfinite(growth) or growth<=1.0:
        raise QualificationError("SUBSTRATE_ADEQUACY_SEARCH_BOUNDS_INVALID")
    caps=[lo]
    while caps[-1]<hi:
        nxt=max(caps[-1]+1,int(math.ceil(caps[-1]*growth)))
        caps.append(min(hi,nxt))
        if caps[-1]==hi: break
    return tuple(caps)


def _policy(policy:dict)->dict:
    required=(
        "min_candidate_nodes","max_candidate_nodes","candidate_growth_factor","refinement_rounds",
        "max_dense_to_surface_p95_norm","max_dense_to_surface_max_norm",
        "max_normal_p95_deg","max_projected_p95_px","max_projected_max_px",
        "component_min_dense_fraction","min_nodes_per_component","max_component_alias_nodes",
    )
    missing=[k for k in required if k not in policy]
    if missing:
        raise QualificationError("SUBSTRATE_ADEQUACY_POLICY_INCOMPLETE:"+",".join(missing))
    p={k:policy[k] for k in required}
    p["visible_component_always_eligible"]=bool(policy.get("visible_component_always_eligible",False))
    p["component_aware_voxel_compaction"]=bool(policy.get("component_aware_voxel_compaction",False))
    p["mechanical_probe_enabled"]=bool(policy.get("mechanical_probe_enabled",False))
    if p["mechanical_probe_enabled"]:
        for k in ("max_mechanical_probe_p95_norm","max_mechanical_probe_max_norm"):
            if k not in policy:
                raise QualificationError("SUBSTRATE_ADEQUACY_MECHANICAL_POLICY_INCOMPLETE:"+k)
            p[k]=float(policy[k])
            if not math.isfinite(p[k]) or p[k]<0:
                raise QualificationError("SUBSTRATE_ADEQUACY_MECHANICAL_THRESHOLD_INVALID:"+k)
    for k in (
        "max_dense_to_surface_p95_norm","max_dense_to_surface_max_norm",
        "max_normal_p95_deg","max_projected_p95_px","max_projected_max_px",
        "component_min_dense_fraction",
    ):
        p[k]=float(p[k])
        if not math.isfinite(p[k]) or p[k]<0:
            raise QualificationError("SUBSTRATE_ADEQUACY_THRESHOLD_INVALID:"+k)
    p["candidate_growth_factor"]=float(p["candidate_growth_factor"])
    p["min_candidate_nodes"]=int(p["min_candidate_nodes"])
    p["max_candidate_nodes"]=int(p["max_candidate_nodes"])
    p["refinement_rounds"]=int(p["refinement_rounds"])
    p["min_nodes_per_component"]=int(p["min_nodes_per_component"])
    p["max_component_alias_nodes"]=int(p["max_component_alias_nodes"])
    if p["component_min_dense_fraction"]>1.0 or p["refinement_rounds"]<0 or p["refinement_rounds"]>16:
        raise QualificationError("SUBSTRATE_ADEQUACY_POLICY_INVALID")
    if p["min_nodes_per_component"]<1 or p["max_component_alias_nodes"]<0:
        raise QualificationError("SUBSTRATE_ADEQUACY_COMPONENT_POLICY_INVALID")
    _candidate_caps(p)
    return p


def _eligible_dense_components(
    dense_labels:np.ndarray,
    dense_support:np.ndarray,
    *,
    component_min_dense_fraction:float,
    visible_component_always_eligible:bool,
)->tuple[set[int],set[int],set[int]]:
    labels=np.asarray(dense_labels,dtype=np.int64)
    support=np.asarray(dense_support,dtype=bool)
    component_count=int(labels.max(initial=-1)+1)
    counts=np.bincount(labels,minlength=component_count)
    fraction_eligible={
        i for i,c in enumerate(counts)
        if float(c)/float(max(len(labels),1))+1e-15 >= float(component_min_dense_fraction)
    }
    visible_components={
        int(labels[i])
        for i in np.flatnonzero(support.any(axis=1))
    }
    eligible=set(fraction_eligible)
    if bool(visible_component_always_eligible):
        eligible.update(visible_components)
    return eligible,fraction_eligible,visible_components


def _mechanical_probe_displacement(points:np.ndarray,kind:str)->np.ndarray:
    p=np.asarray(points,dtype=np.float64)
    if kind=="TWIST_Z":
        angle=np.deg2rad(22.0)*p[:,2]
        ca,sa=np.cos(angle),np.sin(angle)
        out=p.copy()
        out[:,0]=ca*p[:,0]-sa*p[:,1]
        out[:,1]=sa*p[:,0]+ca*p[:,1]
        return out-p
    if kind=="BEND_Y":
        angle=np.deg2rad(18.0)*p[:,2]
        ca,sa=np.cos(angle),np.sin(angle)
        out=p.copy()
        out[:,0]=ca*p[:,0]+sa*p[:,2]
        out[:,2]=-sa*p[:,0]+ca*p[:,2]
        return out-p
    if kind=="HINGE_Z":
        pivot=np.asarray([0.15,0.0,0.0],dtype=np.float64)
        q=p-pivot[None,:]
        weight=np.clip((p[:,0]-0.05)/0.20,0.0,1.0)
        angle=np.deg2rad(25.0)*weight
        ca,sa=np.cos(angle),np.sin(angle)
        out=q.copy()
        out[:,0]=ca*q[:,0]-sa*q[:,1]
        out[:,1]=sa*q[:,0]+ca*q[:,1]
        out+=pivot[None,:]
        return out-p
    raise QualificationError("SUBSTRATE_ADEQUACY_MECHANICAL_PROBE_KIND_INVALID:"+str(kind))


def _mechanical_probe_metrics(dense_normalized:np.ndarray,compact_normalized:np.ndarray)->dict:
    dense=np.asarray(dense_normalized,dtype=np.float64)
    compact=np.asarray(compact_normalized,dtype=np.float64)
    k=min(4,len(compact))
    if k<1:
        raise QualificationError("SUBSTRATE_ADEQUACY_MECHANICAL_PROBE_EMPTY")
    distance,index=cKDTree(compact).query(dense,k=k,workers=-1)
    distance=np.asarray(distance,dtype=np.float64)
    index=np.asarray(index,dtype=np.int64)
    if k==1:
        distance=distance[:,None]
        index=index[:,None]
    weights=1.0/np.maximum(distance,1e-8)
    weights/=weights.sum(axis=1,keepdims=True)

    basis={}
    aggregate_p95=0.0
    aggregate_max=0.0
    for kind in ("TWIST_Z","BEND_Y","HINGE_Z"):
        dense_delta=_mechanical_probe_displacement(dense,kind)
        compact_delta=_mechanical_probe_displacement(compact,kind)
        interpolated=np.einsum("nk,nkd->nd",weights,compact_delta[index])
        error=np.linalg.norm(interpolated-dense_delta,axis=1)
        p95=float(np.quantile(error,0.95))
        maximum=float(error.max(initial=0.0))
        basis[kind]={"p95_norm":p95,"max_norm":maximum}
        aggregate_p95=max(aggregate_p95,p95)
        aggregate_max=max(aggregate_max,maximum)
    return {
        "contract":"KNN4_IDW__TWIST22_BEND18_HINGE25__NORMALIZED_CANONICAL_SPACE_V1",
        "basis":basis,
        "aggregate_p95_norm":float(aggregate_p95),
        "aggregate_max_norm":float(aggregate_max),
    }


def _metrics(
    *,
    surface,
    dense_world:np.ndarray,
    dense_normals:np.ndarray,
    dense_labels:np.ndarray,
    dense_support:np.ndarray,
    dense_raster:np.ndarray,
    normalization_center:np.ndarray,
    normalization_half_extent:float,
    policy:dict,
)->dict:
    compact=np.asarray([n.P for n in surface.surface_nodes],dtype=np.float64)
    compact_normals=np.asarray([n.derived_normal for n in surface.surface_nodes],dtype=np.float64)
    if len(compact)<1 or not np.isfinite(compact).all() or not np.isfinite(compact_normals).all():
        raise QualificationError("SUBSTRATE_ADEQUACY_COMPACT_INVALID")
    tree=cKDTree(compact)
    dist,nearest=tree.query(dense_world,k=1,workers=-1)
    dist_norm=np.asarray(dist,dtype=np.float64)/float(normalization_half_extent)
    dn=np.asarray(dense_normals,dtype=np.float64)
    dn/=np.linalg.norm(dn,axis=1,keepdims=True).clip(min=1e-12)
    cn=compact_normals[np.asarray(nearest,dtype=np.int64)]
    cn/=np.linalg.norm(cn,axis=1,keepdims=True).clip(min=1e-12)
    dots=np.clip(np.sum(dn*cn,axis=1),-1.0,1.0)
    normal_deg=np.degrees(np.arccos(dots))

    component_count=int(dense_labels.max(initial=-1)+1)
    eligible,fraction_eligible,visible_components=_eligible_dense_components(
        dense_labels,dense_support,
        component_min_dense_fraction=float(policy["component_min_dense_fraction"]),
        visible_component_always_eligible=bool(policy.get("visible_component_always_eligible",False)),
    )
    assigned=[set() for _ in range(component_count)]
    owners={}
    for dense_i,node_i in enumerate(np.asarray(nearest,dtype=np.int64)):
        comp=int(dense_labels[dense_i])
        if comp in eligible:
            assigned[comp].add(int(node_i))
            owners.setdefault(int(node_i),set()).add(comp)
    alias_nodes=sum(1 for comps in owners.values() if len(comps)>1)
    min_component_nodes=min((len(assigned[i]) for i in eligible),default=len(compact))

    raster_by_view={}
    projected_distances=[]
    for view in range(8):
        compact_xy=[]
        for node in surface.surface_nodes:
            if view not in set(map(int,node.support_views)):
                continue
            rows=[xy for vi,xy in node.raster_bindings if int(vi)==view]
            if len(rows)!=1:
                raise QualificationError("SUBSTRATE_ADEQUACY_RASTER_BINDING_INVALID")
            compact_xy.append(tuple(map(float,rows[0])))
        visible=np.asarray(dense_support[:,view],dtype=bool)
        dense_xy=np.asarray(dense_raster[visible,view],dtype=np.float64)
        if len(dense_xy)==0:
            raster_by_view[str(view)]={"visible_dense_count":0,"compact_support_count":len(compact_xy),"p95_px":0.0,"max_px":0.0}
            continue
        if not compact_xy:
            p95=max_px=float("inf")
        else:
            dt,_=cKDTree(np.asarray(compact_xy,dtype=np.float64)).query(dense_xy,k=1,workers=-1)
            dt=np.asarray(dt,dtype=np.float64)
            p95=float(np.quantile(dt,0.95)); max_px=float(dt.max(initial=0.0))
            projected_distances.extend(dt.tolist())
        raster_by_view[str(view)]={
            "visible_dense_count":int(len(dense_xy)),
            "compact_support_count":int(len(compact_xy)),
            "p95_px":float(p95),"max_px":float(max_px),
        }
    pd=np.asarray(projected_distances,dtype=np.float64)
    projected_p95=float(np.quantile(pd,0.95)) if len(pd) else float("inf")
    projected_max=float(pd.max(initial=0.0)) if len(pd) else float("inf")

    mechanical=None
    if bool(policy.get("mechanical_probe_enabled",False)):
        center=np.asarray(normalization_center,dtype=np.float64)
        half=float(normalization_half_extent)
        dense_normalized=(np.asarray(dense_world,dtype=np.float64)-center[None,:])/half
        compact_normalized=(compact-center[None,:])/half
        mechanical=_mechanical_probe_metrics(dense_normalized,compact_normalized)

    values={
        "actual_node_count":int(len(compact)),
        "relation_count":int(len(surface.local_relations)),
        "dense_to_surface_p95_norm":float(np.quantile(dist_norm,0.95)),
        "dense_to_surface_max_norm":float(dist_norm.max(initial=0.0)),
        "normal_p95_deg":float(np.quantile(normal_deg,0.95)),
        "projected_p95_px":projected_p95,
        "projected_max_px":projected_max,
        "dense_component_count":component_count,
        "eligible_dense_component_count":int(len(eligible)),
        "visible_dense_component_count":int(len(visible_components)),
        "fraction_eligible_dense_component_count":int(len(fraction_eligible)),
        "visibility_forced_eligible_component_count":int(len(visible_components-fraction_eligible)),
        "visible_component_always_eligible":bool(policy.get("visible_component_always_eligible",False)),
        "minimum_nodes_per_eligible_component":int(min_component_nodes),
        "component_alias_node_count":int(alias_nodes),
        "projected_by_view":raster_by_view,
        "mechanical_probe":mechanical,
    }
    mechanical_passed=(
        True if mechanical is None else (
            mechanical["aggregate_p95_norm"]<=policy["max_mechanical_probe_p95_norm"] and
            mechanical["aggregate_max_norm"]<=policy["max_mechanical_probe_max_norm"]
        )
    )
    passed=(
        values["dense_to_surface_p95_norm"]<=policy["max_dense_to_surface_p95_norm"] and
        values["dense_to_surface_max_norm"]<=policy["max_dense_to_surface_max_norm"] and
        values["normal_p95_deg"]<=policy["max_normal_p95_deg"] and
        values["projected_p95_px"]<=policy["max_projected_p95_px"] and
        values["projected_max_px"]<=policy["max_projected_max_px"] and
        values["minimum_nodes_per_eligible_component"]>=policy["min_nodes_per_component"] and
        values["component_alias_node_count"]<=policy["max_component_alias_nodes"] and
        mechanical_passed
    )
    values["passed"]=bool(passed)
    return values


def select_adequate_rigging_surface_v1(
    vertices_normalized,
    faces,
    implicit_normals,
    cameras,
    *,
    normalization_center,
    normalization_half_extent:float,
    authority_label:str,
    source_run_id:str,
    source_checkpoint_sha256:str,
    source_zero_surface_sha256:str,
    normal_k:int,
    visibility_depth_tolerance_norm:float,
    adequacy_policy:dict,
    metadata:dict|None=None,
):
    policy=_policy(dict(adequacy_policy))
    vn=np.asarray(vertices_normalized,dtype=np.float64)
    f=np.asarray(faces,dtype=np.int64)
    hints=np.asarray(implicit_normals,dtype=np.float64)
    center=np.asarray(normalization_center,dtype=np.float64)
    half=float(normalization_half_extent)
    if vn.ndim!=2 or vn.shape[1]!=3 or len(vn)<4 or f.ndim!=2 or f.shape[1]!=3:
        raise QualificationError("SUBSTRATE_ADEQUACY_DENSE_INPUT_INVALID")
    dense_world=center[None,:]+vn*half
    # The public GSA builder recomputes robust normals per candidate. For adequacy
    # measurement we use the signed decoder normals as orientation-bearing dense
    # reference; selected S still uses the canonical robust PCA operator.
    dense_normals=hints/np.linalg.norm(hints,axis=1,keepdims=True).clip(min=1e-12)
    dense_labels=_dense_component_labels(len(vn),f)
    dense_support,dense_raster,_=_self_zbuffer_support(
        dense_world,dense_world,tuple(cameras),
        depth_tolerance=float(visibility_depth_tolerance_norm)*half,
    )

    evaluated={}
    surfaces={}
    def evaluate(cap:int):
        cap=int(cap)
        if cap in evaluated: return
        surface=rigging_surface_from_scene_first_zero_mesh_v1(
            vn,f,hints,cameras,
            normalization_center=center,
            normalization_half_extent=half,
            authority_label=authority_label,
            source_run_id=source_run_id,
            source_checkpoint_sha256=source_checkpoint_sha256,
            source_zero_surface_sha256=source_zero_surface_sha256,
            target_nodes=cap,
            normal_k=int(normal_k),
            visibility_depth_tolerance_norm=float(visibility_depth_tolerance_norm),
            component_aware_compaction=bool(policy.get("component_aware_voxel_compaction",False)),
            metadata={**dict(metadata or {}),"substrate_adequacy_candidate":True,"candidate_target_node_cap":cap},
        )
        metric=_metrics(
            surface=surface,dense_world=dense_world,dense_normals=dense_normals,dense_labels=dense_labels,
            dense_support=dense_support,dense_raster=dense_raster,
            normalization_center=center,normalization_half_extent=half,policy=policy,
        )
        metric["candidate_target_node_cap"]=cap
        metric["surface_lineage_hash"]=surface.geometry_lineage_hash
        evaluated[cap]=metric; surfaces[cap]=surface

    caps=_candidate_caps(policy)
    for cap in caps:
        evaluate(cap)
    passing=[cap for cap in evaluated if evaluated[cap]["passed"]]
    if passing:
        best=min(passing,key=lambda c:(evaluated[c]["actual_node_count"],c))
        lower=max([c for c in evaluated if c<best],default=policy["min_candidate_nodes"])
        high=best
        for _ in range(int(policy["refinement_rounds"])):
            if high-lower<=1: break
            mid=(lower+high)//2
            evaluate(mid)
            if evaluated[mid]["passed"]:
                high=mid
            else:
                lower=mid
        passing=[cap for cap in evaluated if evaluated[cap]["passed"]]
        best=min(passing,key=lambda c:(evaluated[c]["actual_node_count"],c))
    else:
        best=None

    report={
        "schema":"RealSaS.SubstrateAdequacyReport.v1",
        "status":"PASS" if best is not None else "FAIL",
        "source_zero_surface_sha256":str(source_zero_surface_sha256),
        "source_checkpoint_sha256":str(source_checkpoint_sha256),
        "policy":policy,
        "evaluated_candidates":tuple(evaluated[c] for c in sorted(evaluated)),
        "selected_target_node_cap":None if best is None else int(best),
        "selected_actual_node_count":None if best is None else int(evaluated[best]["actual_node_count"]),
        "selected_surface_lineage_hash":"" if best is None else str(evaluated[best]["surface_lineage_hash"]),
        "selection_rule":"MINIMUM_ACTUAL_NODE_COUNT_AMONG_PASSING_BOUNDED_SEARCH_CANDIDATES",
        "teacher_truth_used":False,
        "categorical_recognition_used":False,
        "adequacy_report_hash":"",
    }
    report["adequacy_report_hash"]=substrate_adequacy_report_hash_v1(report)
    if best is None:
        return None,report
    selected=surfaces[best]
    selected.metadata.update if False else None
    return selected,report
