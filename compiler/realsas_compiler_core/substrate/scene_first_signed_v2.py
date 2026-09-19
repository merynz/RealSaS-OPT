from __future__ import annotations

"""Scene-first signed geometric substrate v2.

V2 preserves two authority invariants that v1 did not enforce during compaction:
1) every compact node is an exact representative vertex from the admitted dense
   signed zero-surface (voxel centroids may not move geometry off the zero set);
2) disconnected dense components are compacted independently and can never be
   merged into one compact node.

This module intentionally leaves the v1 implementation intact for historical
lineage. Promotion to mainline requires separate subject-free calibration.
"""

from dataclasses import replace
import math

import numpy as np

from .hashing import content_sha256
from .types import QualificationError, RiggingSurfaceIR, SurfaceNode, SurfaceRelation
from .scene_first_signed import (
    ZERO_SURFACE_NORMAL_OPERATOR_ID,
    ZERO_SURFACE_VISIBILITY_ID,
    _normalize_rows,
    _self_zbuffer_support,
    robust_zero_surface_normals_v1,
    zero_surface_normal_operator_hash_v1,
    zero_surface_normal_operator_identity_v1,
)

ZERO_SURFACE_COMPACTOR_V2_ID="RealSaS.GSA.ZeroSurfaceComponentRepresentativeVoxel.v2"


def _dense_component_labels_v2(vertex_count:int,faces:np.ndarray)->np.ndarray:
    parent=np.arange(int(vertex_count),dtype=np.int64)
    rank=np.zeros(int(vertex_count),dtype=np.int8)

    def find(x:int)->int:
        while parent[x]!=x:
            parent[x]=parent[parent[x]]
            x=int(parent[x])
        return x

    def union(a:int,b:int)->None:
        ra,rb=find(a),find(b)
        if ra==rb:
            return
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


def _allocate_component_quotas(
    component_sizes:np.ndarray,
    *,
    target_nodes:int,
    min_nodes_per_component:int,
)->np.ndarray:
    sizes=np.asarray(component_sizes,dtype=np.int64)
    if sizes.ndim!=1 or len(sizes)<1 or np.any(sizes<=0):
        raise QualificationError("GSA_V2_COMPONENT_SIZE_INVALID")
    minimum=int(min_nodes_per_component)
    target=int(target_nodes)
    if minimum<1:
        raise QualificationError("GSA_V2_COMPONENT_MINIMUM_INVALID")
    base=np.minimum(sizes,minimum).astype(np.int64)
    if int(base.sum())>target:
        raise QualificationError("GSA_V2_TARGET_BELOW_COMPONENT_MINIMUM_BUDGET")
    quota=base.copy()
    remaining=target-int(quota.sum())
    capacity=sizes-quota
    while remaining>0 and int(capacity.sum())>0:
        total=float(capacity.sum())
        ideal=remaining*capacity.astype(np.float64)/total
        add=np.minimum(capacity,np.floor(ideal).astype(np.int64))
        if int(add.sum())==0:
            frac=ideal-np.floor(ideal)
            candidates=np.where(capacity>0)[0]
            order=sorted(candidates.tolist(),key=lambda i:(-float(frac[i]),int(i)))
            take=min(remaining,len(order))
            add[np.asarray(order[:take],dtype=np.int64)]+=1
        quota+=add
        used=int(add.sum())
        remaining-=used
        capacity=sizes-quota
        if used<=0:
            break
    if remaining>0:
        candidates=np.where(sizes>quota)[0]
        for i in candidates.tolist():
            if remaining<=0:
                break
            take=min(int(sizes[i]-quota[i]),remaining)
            quota[i]+=take
            remaining-=take
    if remaining!=0 or np.any(quota>sizes) or int(quota.sum())>target:
        raise QualificationError("GSA_V2_COMPONENT_QUOTA_ALLOCATION_FAILED")
    return quota.astype(np.int64)


def _compact_one_component_exact(
    p:np.ndarray,
    n:np.ndarray,
    global_indices:np.ndarray,
    *,
    target_nodes:int,
):
    if len(p)<=int(target_nodes):
        inverse=np.arange(len(p),dtype=np.int64)
        divisions=0
    else:
        lo=p.min(axis=0)
        span=np.maximum(p.max(axis=0)-lo,1e-12)

        def labels_for(divisions:int):
            keys=np.floor((p-lo)/span*divisions).astype(np.int64)
            keys=np.clip(keys,0,divisions-1)
            return np.unique(keys,axis=0,return_inverse=True)

        low,high=1,1024
        best=None
        while low<=high:
            mid=(low+high)//2
            unique,inv=labels_for(mid)
            if len(unique)<=int(target_nodes):
                best=(mid,inv)
                low=mid+1
            else:
                high=mid-1
        if best is None:
            raise QualificationError("GSA_V2_COMPONENT_COMPACTION_FAILED")
        divisions,inverse=best

    count=int(inverse.max(initial=-1)+1)
    counts=np.bincount(inverse,minlength=count).astype(np.float64)
    centroid=np.zeros((count,3),dtype=np.float64)
    np.add.at(centroid,inverse,p)
    centroid/=counts[:,None]

    dist2=np.sum((p-centroid[inverse])**2,axis=1)
    # Deterministic representative: nearest dense zero-surface vertex to the
    # voxel centroid, lexical tie-break by original dense vertex index.
    order=np.lexsort((global_indices,dist2,inverse))
    sorted_labels=inverse[order]
    first=np.ones(len(order),dtype=bool)
    if len(order)>1:
        first[1:]=sorted_labels[1:]!=sorted_labels[:-1]
    chosen_local=order[first]
    chosen_labels=inverse[chosen_local]
    rep_by_label=np.empty(count,dtype=np.int64)
    rep_by_label[chosen_labels]=chosen_local
    representatives=global_indices[rep_by_label]

    cp=p[rep_by_label].copy()
    cn=_normalize_rows(n[rep_by_label]).copy()
    return cp,cn,inverse.astype(np.int64),representatives.astype(np.int64),int(divisions)


def adaptive_zero_surface_compact_v2(
    points,
    faces,
    dense_normals,
    *,
    target_nodes:int,
    min_nodes_per_component:int=8,
):
    p=np.asarray(points,dtype=np.float64)
    f=np.asarray(faces,dtype=np.int64)
    n=np.asarray(dense_normals,dtype=np.float64)
    if p.ndim!=2 or p.shape[1]!=3 or n.shape!=p.shape:
        raise QualificationError("GSA_V2_COMPACTOR_INPUT_INVALID")
    if f.ndim!=2 or f.shape[1]!=3 or np.any(f<0) or np.any(f>=len(p)):
        raise QualificationError("GSA_V2_COMPACTOR_FACE_INVALID")
    if int(target_nodes)<64:
        raise QualificationError("GSA_V2_TARGET_NODES_TOO_SMALL")

    labels=_dense_component_labels_v2(len(p),f)
    component_count=int(labels.max(initial=-1)+1)
    sizes=np.bincount(labels,minlength=component_count)
    quotas=_allocate_component_quotas(
        sizes,target_nodes=int(target_nodes),
        min_nodes_per_component=int(min_nodes_per_component),
    )

    inverse=np.empty(len(p),dtype=np.int64)
    compact_points=[]
    compact_normals=[]
    representative_indices=[]
    compact_component_ids=[]
    divisions_by_component=[]
    offset=0
    for component_id in range(component_count):
        idx=np.flatnonzero(labels==component_id).astype(np.int64)
        cp,cn,local_inv,reps,divisions=_compact_one_component_exact(
            p[idx],n[idx],idx,target_nodes=int(quotas[component_id])
        )
        compact_points.append(cp)
        compact_normals.append(cn)
        representative_indices.append(reps)
        compact_component_ids.extend([int(component_id)]*len(cp))
        divisions_by_component.append(int(divisions))
        inverse[idx]=local_inv+offset
        offset+=len(cp)

    cp=np.concatenate(compact_points,axis=0)
    cn=np.concatenate(compact_normals,axis=0)
    reps=np.concatenate(representative_indices,axis=0)
    compact_component_ids=np.asarray(compact_component_ids,dtype=np.int64)

    if len(cp)>int(target_nodes):
        raise QualificationError("GSA_V2_COMPACTOR_EXCEEDS_TARGET")
    if not np.array_equal(cp,p[reps]):
        raise QualificationError("GSA_V2_REPRESENTATIVE_ZERO_SET_DRIFT")
    if np.any(labels[reps]!=compact_component_ids):
        raise QualificationError("GSA_V2_REPRESENTATIVE_COMPONENT_DRIFT")

    mapped=inverse[f]
    edges=np.concatenate([mapped[:,[0,1]],mapped[:,[1,2]],mapped[:,[2,0]]],axis=0)
    edges=np.sort(edges,axis=1)
    edges=np.unique(edges[edges[:,0]!=edges[:,1]],axis=0)
    if len(edges)==0:
        raise QualificationError("GSA_V2_COMPACTION_REMOVED_ALL_TOPOLOGY")
    # No edge may bridge disconnected dense components.
    if np.any(compact_component_ids[edges[:,0]]!=compact_component_ids[edges[:,1]]):
        raise QualificationError("GSA_V2_CROSS_COMPONENT_EDGE_FORBIDDEN")

    return {
        "points":cp,
        "normals":cn,
        "edges":edges.astype(np.int64),
        "inverse":inverse,
        "representative_dense_indices":reps,
        "dense_component_labels":labels,
        "compact_component_ids":compact_component_ids,
        "component_sizes":sizes.astype(np.int64),
        "component_quotas":quotas.astype(np.int64),
        "component_node_counts":np.bincount(compact_component_ids,minlength=component_count).astype(np.int64),
        "divisions_by_component":tuple(divisions_by_component),
    }


def rigging_surface_from_scene_first_zero_mesh_v2(
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
    target_nodes:int=1024,
    normal_k:int=64,
    visibility_depth_tolerance_norm:float=0.02,
    min_nodes_per_component:int=8,
    metadata:dict|None=None,
)->RiggingSurfaceIR:
    vn=np.asarray(vertices_normalized,dtype=np.float64)
    f=np.asarray(faces,dtype=np.int64)
    hint=np.asarray(implicit_normals,dtype=np.float64)
    center=np.asarray(normalization_center,dtype=np.float64)
    half=float(normalization_half_extent)
    if vn.ndim!=2 or vn.shape[1]!=3 or len(vn)<4 or not np.isfinite(vn).all():
        raise QualificationError("GSA_V2_ZERO_SURFACE_VERTICES_INVALID")
    if f.ndim!=2 or f.shape[1]!=3 or np.any(f<0) or np.any(f>=len(vn)):
        raise QualificationError("GSA_V2_ZERO_SURFACE_FACES_INVALID")
    if hint.shape!=vn.shape or not np.isfinite(hint).all():
        raise QualificationError("GSA_V2_IMPLICIT_NORMALS_INVALID")
    if center.shape!=(3,) or not np.isfinite(center).all() or not math.isfinite(half) or half<=0:
        raise QualificationError("GSA_V2_NORMALIZATION_INVALID")
    if visibility_depth_tolerance_norm<=0:
        raise QualificationError("GSA_V2_VISIBILITY_TOLERANCE_INVALID")

    world=center[None,:]+vn*half
    dense_normals=robust_zero_surface_normals_v1(world,hint,k=normal_k)
    compact=adaptive_zero_surface_compact_v2(
        world,f,dense_normals,
        target_nodes=int(target_nodes),
        min_nodes_per_component=int(min_nodes_per_component),
    )
    points=compact["points"]; normals=compact["normals"]; edges=compact["edges"]
    support,raster,visible_counts=_self_zbuffer_support(
        world,points,tuple(cameras),
        depth_tolerance=float(visibility_depth_tolerance_norm)*half,
    )

    op_hash=zero_surface_normal_operator_hash_v1(k=normal_k)
    nodes=[]
    reps=compact["representative_dense_indices"]
    compids=compact["compact_component_ids"]
    for i,(p,n) in enumerate(zip(points,normals)):
        views=tuple(int(v) for v in range(8) if bool(support[i,v]))
        binds=tuple((v,(float(raster[i,v,0]),float(raster[i,v,1]))) for v in views)
        flags=("OBSERVED_SIGNED_ZERO_SURFACE",) if views else ("MODEL_COMPLETED_SIGNED_ZERO_SURFACE",)
        sid="SFS2:"+content_sha256({
            "dense_vertex_index":int(reps[i]),
            "P":p.tolist(),
            "views":views,
            "source":source_zero_surface_sha256,
            "dense_component_id":int(compids[i]),
        })[:20]
        nodes.append(SurfaceNode(
            surface_id=sid,
            P=tuple(map(float,p)),
            support_views=views,
            provenance_refs=(str(authority_label),),
            source_observation_ids=(),
            raster_bindings=binds,
            persistence_group_id=f"SCENE_FIRST_SIGNED_ZERO_V2:{i:05d}",
            derived_normal=tuple(map(float,n)),
            validity_flags=flags,
            metadata={
                "normal_operator":ZERO_SURFACE_NORMAL_OPERATOR_ID,
                "normal_operator_hash":op_hash,
                "normal_implicit_hint_only":True,
                "teacher_truth_used":False,
                "compactor_id":ZERO_SURFACE_COMPACTOR_V2_ID,
                "exact_dense_zero_surface_representative":True,
                "source_dense_vertex_index":int(reps[i]),
                "dense_component_id":int(compids[i]),
            },
        ))

    ids=tuple(n.surface_id for n in nodes)
    rel=[]
    for a,b in edges.tolist():
        dist=float(np.linalg.norm(points[a]-points[b]))
        rel.append(SurfaceRelation(
            relation_id="SFS2REL:"+content_sha256({"a":ids[a],"b":ids[b],"d":dist})[:20],
            a_surface_id=ids[a],b_surface_id=ids[b],
            relation_kind="SIGNED_ZERO_SURFACE_TOPOLOGY_NEIGHBOR",
            score=1.0,
            metadata={
                "world_distance":dist,
                "crosses_unknown":False,
                "unknown_bridge":False,
                "teacher_truth_used":False,
                "dense_component_id":int(compids[a]),
            },
        ))

    lineage=content_sha256({
        "schema":"RealSaS.SceneFirstSignedToRiggingSurface.v2",
        "source_run_id":str(source_run_id),
        "source_checkpoint_sha256":str(source_checkpoint_sha256),
        "source_zero_surface_sha256":str(source_zero_surface_sha256),
        "normal_operator":zero_surface_normal_operator_identity_v1(k=normal_k),
        "compactor":{
            "id":ZERO_SURFACE_COMPACTOR_V2_ID,
            "target_nodes":int(target_nodes),
            "min_nodes_per_component":int(min_nodes_per_component),
            "component_sizes":compact["component_sizes"].tolist(),
            "component_quotas":compact["component_quotas"].tolist(),
            "component_node_counts":compact["component_node_counts"].tolist(),
            "divisions_by_component":list(compact["divisions_by_component"]),
            "exact_dense_representatives":True,
        },
        "visibility":{
            "id":ZERO_SURFACE_VISIBILITY_ID,
            "depth_tolerance_norm":float(visibility_depth_tolerance_norm),
        },
        "nodes":[n.to_dict() for n in nodes],
        "relations":[r.to_dict() for r in rel],
    })
    meta={
        **dict(metadata or {}),
        "scene_first_signed_geometry":True,
        "scene_first_decoder":"SIGNED_FIELD_ZERO_LEVEL_SURFACE",
        "scene_first_compactor_version":"v2",
        "source_dense_vertex_count":int(len(world)),
        "compact_surface_node_count":int(len(nodes)),
        "compact_target_nodes":int(target_nodes),
        "dense_component_count":int(len(compact["component_sizes"])),
        "dense_component_sizes":tuple(map(int,compact["component_sizes"])),
        "component_quotas":tuple(map(int,compact["component_quotas"])),
        "component_node_counts":tuple(map(int,compact["component_node_counts"])),
        "compact_divisions_by_component":tuple(map(int,compact["divisions_by_component"])),
        "observed_node_count":int(np.any(support,axis=1).sum()),
        "completed_node_count":int((~np.any(support,axis=1)).sum()),
        "visibility_support_counts_by_view":visible_counts,
        "raster_coordinate_system":"PIXEL_CENTER_XY",
        "resolution":int(cameras[0]["resolution"]),
        "Nd_operator":ZERO_SURFACE_NORMAL_OPERATOR_ID,
        "Nd_operator_sha256":op_hash,
        "normal_implicit_hint_only":True,
        "source_run_id":str(source_run_id),
        "source_checkpoint_sha256":str(source_checkpoint_sha256),
        "source_zero_surface_sha256":str(source_zero_surface_sha256),
        "teacher_truth_used":False,
        "character_gen_runtime_used":False,
        "full_hidden_mesh_completeness_hard_gate":False,
        "full_3d_intermediate_allowed":True,
        "exact_dense_zero_surface_representatives":True,
        "disconnected_component_merge_forbidden":True,
    }
    return RiggingSurfaceIR(
        surface_nodes=tuple(nodes),
        local_relations=tuple(sorted(rel,key=lambda r:r.relation_id)),
        geometry_lineage_hash=lineage,
        builder_id="RealSaS.GeometricSubstrateAssembler.SceneFirstSigned.v2",
        schema_version="RealSaS.RiggingSurfaceIR.v1",
        metadata=meta,
    )


__all__=[
    "ZERO_SURFACE_COMPACTOR_V2_ID",
    "adaptive_zero_surface_compact_v2",
    "rigging_surface_from_scene_first_zero_mesh_v2",
]
