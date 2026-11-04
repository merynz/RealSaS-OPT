#!/usr/bin/env python3
"""Morphology-agnostic anonymous graphics-layer candidate compiler.

Input: independently source/canonical-bound, source-triangle-addressed mesh,
authored poses and FULL/REMOVE/recolor witness face-pairs from the pre-existing
private 3-subject court. Teacher faces are identifiers for *matching*, not
semantic or edit ownership truth.

Output: immutable deterministic candidate partitions, including: single
layer, manifold components, geometry seams, observed-motion seams, combined
seams, ray-seeded graph Voronoi and per-face atomic baseline. The held-out
view is NEVER used to construct a candidate. There is no trained model.

This module does not choose a model/training target or mint drawable owner
labels. Evaluation is a separate independent bounded edit-consumer court.
"""
from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
from pathlib import Path

import numpy as np

from tools.corpus.topology_partition_court_v1 import (
    canonicalize_partition, edge_connected_face_partition,
)

SCHEMA="RealSaS.Corpus370AnonymousLayerProposal.v1"
SOURCE_PAIR_SCHEMA="RealSaS.Corpus370GenericPairKinematicDiagnosticBatch.v1"
CANDIDATE_NAMES=(
    "SINGLE_LAYER","MANIFOLD_COMPONENTS","GEOMETRY_SEAMS",
    "MOTION_SEAMS","GEOMETRY_OR_MOTION_SEAMS","TRAIN_VIEW_RAY_SEEDED",
    "ATOMIC_PER_FACE",
)


def require(ok,code):
    if not ok:raise ValueError(code)


def digest(path:Path)->str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle,"sha256").hexdigest()


def require_sha(sha:str):
    require(isinstance(sha,str) and len(sha)==64
            and all(c in "0123456789abcdef" for c in sha),
            "ANON_LAYER_SHA_REQUIRED")


def _graph(rest:np.ndarray,faces:np.ndarray,poses:np.ndarray):
    """Only two-face manifold edges; point contact never connects a layer."""
    normals=np.cross(rest[faces[:,1]]-rest[faces[:,0]],
                     rest[faces[:,2]]-rest[faces[:,0]])
    sizes=np.linalg.norm(normals,axis=1)
    require(np.isfinite(sizes).all(),"ANON_LAYER_NONFINITE_NORMAL")
    # Degenerate source faces are not silently merged across an undefined seam.
    valid=sizes>1e-12
    safe=np.zeros_like(normals)
    safe[valid]=normals[valid]/sizes[valid,None]
    edges={}
    for i,(a,b,c) in enumerate(faces):
        for va,vb in ((a,b),(b,c),(c,a)):
            key=(int(min(va,vb)),int(max(va,vb)))
            edges.setdefault(key,[]).append(i)
    scale=float(np.linalg.norm(np.ptp(rest,axis=0)))
    require(scale>1e-10,"ANON_LAYER_ZERO_SOURCE_EXTENT")
    # All motions are observational proposals; no separate causal label.
    rest_center=np.mean(rest[faces],axis=1)
    deltas=[]
    for pose in poses:
        center=np.mean(pose[faces],axis=1)
        deltas.append(center-rest_center)
    # Keep at most nposes x nfaces source-world deltas.
    deltas=np.stack(deltas,axis=0)
    graph=[[] for _ in range(len(faces))]
    manifold=[]
    for incident in edges.values():
        if len(incident)!=2:continue
        a,b=incident
        if not(valid[a] and valid[b]):continue
        angle=float(math.acos(float(np.clip(np.dot(safe[a],safe[b]),-1,1))))
        # Translation is removed; differential displacement is invariant
        # under a shared translation. This is a proposal score, NOT owner truth.
        motion=float(np.linalg.norm(deltas[:,a]-deltas[:,b],axis=1).max()/scale)
        weight=1.0+angle/math.pi*5.0+min(motion,1.0)*20.0
        manifold.append((a,b,angle,motion,weight))
        graph[a].append((b,weight))
        graph[b].append((a,weight))
    for neighbors in graph:
        neighbors.sort(key=lambda p:p[0])
    return manifold,graph


def _components(faces_count:int,edges):
    parent=list(range(faces_count))
    def find(a):
        while parent[a]!=a:
            parent[a]=parent[parent[a]]
            a=parent[a]
        return a
    for a,b in edges:
        a=find(a);b=find(b)
        if a!=b:
            if a>b:a,b=b,a
            parent[b]=a
    return canonicalize_partition([find(i) for i in range(faces_count)])


def _seeded_graph_partition(graph,manifold_components,train_faces):
    """Train-view A/B are unique anonymous anchors. Not author partitions."""
    n=len(graph)
    seeds=sorted(set(int(x) for x in train_faces))
    require(seeds and len(seeds)<=2*n,"ANON_LAYER_TRAIN_SOURCE_HITS")
    shortest=[math.inf]*n
    assigned=[-1]*n
    heap=[]
    for i,face in enumerate(seeds):
        shortest[face]=0.
        assigned[face]=i
        heapq.heappush(heap,(0.,i,face))
    while heap:
        dist,owner,face=heapq.heappop(heap)
        if dist>shortest[face]+1e-10 or assigned[face]!=owner:
            continue
        for nxt,cost in graph[face]:
            proposed=dist+cost
            if (proposed<shortest[nxt]-1e-10
                or (abs(proposed-shortest[nxt])<=1e-10 and owner<assigned[nxt])):
                shortest[nxt]=proposed
                assigned[nxt]=owner
                heapq.heappush(heap,(proposed,owner,nxt))
    groups={}
    next_id=len(seeds)
    for idx,component in enumerate(manifold_components.tolist()):
        if assigned[idx]<0:
            if component not in groups:
                groups[component]=next_id
                next_id+=1
            assigned[idx]=groups[component]
    return canonicalize_partition(assigned)


def compile_candidates(rest_vertices,faces,posed_vertices,source_pair_rows,*,parents):
    for k in ("source_sha256","canonical_arrays_sha256","source_pair_receipt_sha256"):
        require_sha(parents.get(k))
    rest=np.asarray(rest_vertices,dtype=np.float64)
    triangles=np.asarray(faces)
    poses=np.asarray(posed_vertices,dtype=np.float64)
    require(rest.ndim==2 and rest.shape[1:]==(3,) and len(rest)>0
            and np.isfinite(rest).all(),"ANON_LAYER_REST")
    require(triangles.ndim==2 and triangles.shape[1:]==(3,)
            and len(triangles)>1 and triangles.dtype.kind in "iu"
            and (triangles>=0).all() and (triangles<len(rest)).all()
            and np.all(triangles[:,0]!=triangles[:,1])
            and np.all(triangles[:,1]!=triangles[:,2])
            and np.all(triangles[:,2]!=triangles[:,0]),
            "ANON_LAYER_FACES")
    require(poses.ndim==3 and poses.shape[1:]==rest.shape
            and len(poses)>=2 and np.isfinite(poses).all(),
            "ANON_LAYER_NATIVE_POSES")
    require(isinstance(source_pair_rows,list) and len(source_pair_rows)>=2,
            "ANON_LAYER_REAL_SOURCE_RAYS_REQUIRED")
    n=len(triangles)
    train_faces=[]
    views=set()
    for row in source_pair_rows:
        require(isinstance(row,dict),"ANON_LAYER_PAIR_SCHEMA")
        a=row.get("source_face_a_teacher_only")
        b=row.get("source_face_b_teacher_only")
        view=row.get("view_index")
        require(type(a) is int and type(b) is int and 0<=a<n
                and 0<=b<n and a!=b and type(view) is int
                and view in (0,4),"ANON_LAYER_FACE_OR_VIEW")
        views.add(view)
        if view==0:train_faces.extend((a,b))
    require(views=={0,4} and len(train_faces)>=2,
            "ANON_LAYER_0_TRAIN_4_HELDOUT_REQUIRED")
    edges,graph=_graph(rest,triangles,poses)
    partition=lambda filt:_components(n,((a,b) for a,b,angle,motion,_ in edges
                                          if filt(angle,motion)))
    manifold=partition(lambda angle,motion:True)
    geometry=partition(lambda angle,motion:angle<0.85)
    motion=partition(lambda angle,motion:motion<0.018)
    both=partition(lambda angle,motion:angle<0.85 and motion<0.018)
    layouts={
        "SINGLE_LAYER":np.zeros(n,np.int32),
        "MANIFOLD_COMPONENTS":manifold,
        "GEOMETRY_SEAMS":geometry,
        "MOTION_SEAMS":motion,
        "GEOMETRY_OR_MOTION_SEAMS":both,
        "TRAIN_VIEW_RAY_SEEDED":_seeded_graph_partition(
            graph,manifold,train_faces),
        "ATOMIC_PER_FACE":np.arange(n,dtype=np.int32),
    }
    require(tuple(layouts)==CANDIDATE_NAMES,"ANON_LAYER_CANDIDATE_CENSUS")
    models=[]
    for name,labels in layouts.items():
        labels=canonicalize_partition(labels)
        require(labels.shape==(n,) and np.min(labels)==0
                and len(np.unique(labels))==int(np.max(labels))+1,
                "ANON_LAYER_PARTITION_COVERAGE")
        models.append({
            "candidate":name,
            "slot_count":int(labels.max())+1,
            "face_slot_teacher_only":labels.astype(int).tolist(),
            "source_import_appearance_retained":True,
            "render_heldout_replay_executed":False,
            "operational_drawable_authority_minted":False,
        })
    return {
        "schema":SCHEMA,
        "status":"ANONYMOUS_CANDIDATE_PARTITIONS__BOUNDED_CONSUMER_REQUIRED",
        "parents":dict(parents),
        "face_count":n,
        "authored_pose_count":len(poses),
        "independently_sampled_source_ray_count":len(source_pair_rows),
        "train_view":0,"heldout_view":4,
        "candidate_count":len(models),"candidates":models,
        "source_semantic_or_material_names_used":False,
        "teacher_source_faces_exposed_to_model":False,
        "graphics_owner_minted":False,
        "mechanical_owner_minted":False,
        "training_authorized":False,
    }


def main():
    cli=argparse.ArgumentParser()
    cli.add_argument("--canonical-arrays",type=Path,required=True)
    cli.add_argument("--expected-canonical-sha256",required=True)
    cli.add_argument("--pair-receipt",type=Path,required=True)
    cli.add_argument("--expected-pair-sha256",required=True)
    cli.add_argument("--source-sha256",required=True)
    cli.add_argument("--output",type=Path,required=True)
    a=cli.parse_args()
    require(a.canonical_arrays.is_file()
            and digest(a.canonical_arrays)==a.expected_canonical_sha256
            and a.pair_receipt.is_file()
            and digest(a.pair_receipt)==a.expected_pair_sha256,
            "ANON_LAYER_PARENT_BYTES_SHA")
    pairs=json.loads(a.pair_receipt.read_text())
    require(pairs.get("schema")==SOURCE_PAIR_SCHEMA
            and pairs.get("parents",{}).get("source_sha256")==a.source_sha256
            and pairs.get("training_authorized") is False
            and pairs.get("qualified_same_drawable_positives")==0
            and pairs.get("qualified_different_drawable_negatives")==0,
            "ANON_LAYER_PAIR_RECEIPT_AUTHORITY_DRIFT")
    with np.load(a.canonical_arrays,allow_pickle=False) as arrays:
        require({"rest_vertices_canonical","posed_vertices_canonical","faces"}
                <=set(arrays.files),"ANON_LAYER_CANONICAL_ARRAY_SCOPE")
        proposal=compile_candidates(
            arrays["rest_vertices_canonical"],arrays["faces"],
            arrays["posed_vertices_canonical"],pairs["rows"],
            parents={
                "source_sha256":a.source_sha256,
                "canonical_arrays_sha256":a.expected_canonical_sha256,
                "source_pair_receipt_sha256":a.expected_pair_sha256,
            },
        )
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(proposal,sort_keys=True,
                                   separators=(",",":"),allow_nan=False)+"\n")
    print(json.dumps({k:v for k,v in proposal.items() if k!="candidates"},
                     sort_keys=True),flush=True)


if __name__=="__main__":
    main()
