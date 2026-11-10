#!/usr/bin/env python3
"""Anonymous hierarchical *group* proposals, preserving existing source atoms.

The existing 7-way anonymous source-face proposal is immutable. This module
takes its TRAIN-selected partition (view 0 only) as atomic editor patches
and agglomerates them using spatial + rest/pose motion compatibility.
One train-view independently observed A/B reveal constraint is a
'cannot-link' for an *operational edit*, NOT author ownership truth.

The opposite view 4 is never read for merge order, cannot-links, group
selection or weighting. This is not a renderer, and source faces are
privileged addresses never supplied to a model.
"""
from __future__ import annotations

import heapq
import math
from typing import Any

import numpy as np

from tools.corpus.topology_partition_court_v1 import canonicalize_partition

SCHEMA = "RealSaS.Corpus370HierarchicalAnonymousLayerGroups.v1"
PARENT_SCHEMA = "RealSaS.Corpus370AnonymousLayerProposal.v1"
PROXY_SCHEMA = "RealSaS.Corpus370AnonymousLayerEditProxyCourt.v1"
TARGETS = (8,16,32,64)
EPS=1e-12


def require(ok:bool,code:str)->None:
    if not ok:raise ValueError(code)


def _array_faces(rest,faces,poses):
    v=np.asarray(rest,dtype=np.float64)
    tri=np.asarray(faces)
    p=np.asarray(poses,dtype=np.float64)
    require(v.ndim==2 and v.shape[1:]==(3,) and len(v)>3
            and np.isfinite(v).all(),"HIERARCHY_REST_INVALID")
    require(tri.ndim==2 and tri.shape[1:]==(3,)
            and tri.dtype.kind in "iu" and len(tri)>1
            and (tri>=0).all() and (tri<len(v)).all()
            and all(np.all(tri[:,i]!=tri[:,(i+1)%3]) for i in range(3)),
            "HIERARCHY_FACES_INVALID")
    require(p.ndim==3 and p.shape[1:]==v.shape and len(p)>=2
            and np.isfinite(p).all(),"HIERARCHY_AUTHORED_POSES_INVALID")
    scale=float(np.linalg.norm(np.ptp(v,axis=0)))
    require(math.isfinite(scale) and scale>1e-9,"HIERARCHY_SOURCE_EXTENT")
    return v,tri,p,scale


def _attributes(vertices,faces,posed):
    tri=vertices[faces]
    area=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],
                                tri[:,2]-tri[:,0]),axis=1)/2.
    area=np.maximum(area,1e-10)
    rest_centres=tri.mean(axis=1)
    posed_centres=posed[:,faces].mean(axis=2)
    return area,rest_centres,posed_centres-rest_centres[None]


def _cluster_feature(face_ids,faces,vertices,area,centres,deltas):
    ids=np.asarray(face_ids,dtype=np.int64)
    weights=area[ids]
    weight=float(weights.sum())
    normal=weights/weight
    return {
        "members":tuple(sorted(int(i) for i in ids)),
        "mass":weight,
        "rest_center":np.sum(centres[ids]*normal[:,None],axis=0),
        "min":vertices[faces[ids]].min(axis=(0,1)),
        "max":vertices[faces[ids]].max(axis=(0,1)),
        "motion":np.sum(deltas[:,ids,:]*normal[None,:,None],axis=1),
        "face_count":len(ids),
    }


def _combine(a,b):
    total=a["mass"]+b["mass"]
    wa=a["mass"]/total;wb=b["mass"]/total
    return {
        "members":tuple(sorted(a["members"]+b["members"])),
        "mass":total,
        "rest_center":wa*a["rest_center"]+wb*b["rest_center"],
        "min":np.minimum(a["min"],b["min"]),
        "max":np.maximum(a["max"],b["max"]),
        "motion":wa*a["motion"]+wb*b["motion"],
        "face_count":a["face_count"]+b["face_count"],
    }


def _distance(a,b,scale):
    # Bounding-box gap is zero for nearby/interleaved source shells.
    separation=np.maximum(np.maximum(a["min"]-b["max"],
                                     b["min"]-a["max"]),0.)
    boxgap=float(np.linalg.norm(separation))/scale
    centergap=float(np.linalg.norm(
        a["rest_center"]-b["rest_center"]))/scale
    motiongap=float(np.linalg.norm(
        a["motion"]-b["motion"],axis=1).max())/scale
    # Fixed proposal policy. Model quality needs heldout full render court.
    return 3.*boxgap+0.35*centergap+12.*motiongap


def _forbidden(groups,cannot_links):
    # A merge is forbidden if unioning these two groups would violate
    # any independently observed TRAIN-view removal/reveal distinction.
    x=set(groups[0]["members"])
    y=set(groups[1]["members"])
    return any((a in x and b in y) or (b in x and a in y)
               for a,b in cannot_links)


def coalesce(rest,faces,poses,initial_labels,train_cannot_links,
             *,target_counts=TARGETS):
    v,f,p,scale=_array_faces(rest,faces,poses)
    initial=canonicalize_partition(initial_labels)
    require(initial.shape==(len(f),) and initial.dtype.kind in "iu",
            "HIERARCHY_INITIAL_PARTITION")
    constraints=[]
    for item in train_cannot_links:
        require(isinstance(item,(tuple,list)) and len(item)==2
                and all(type(i) is int and 0<=i<len(f) for i in item)
                and item[0]!=item[1],"HIERARCHY_TRAIN_EDIT_FACE_ADDRESS")
        a,b=map(int,item)
        require(initial[a]!=initial[b],
                "HIERARCHY_PARENT_CONTRADICTS_TRAIN_INDEPENDENT_EDIT")
        constraints.append((a,b))
    require(constraints and len(set(constraints))==len(constraints),
            "HIERARCHY_TRAIN_EDIT_CONSTRAINTS_REQUIRED")
    req=tuple(sorted(set(target_counts),reverse=True))
    require(all(type(k) is int and 2<=k<=128 for k in req)
            and len(req)<=8,"HIERARCHY_BOUND_TARGET_SET")
    area,centres,deltas=_attributes(v,f,p)
    groups={}
    for label in np.unique(initial):
        members=np.flatnonzero(initial==label)
        groups[int(label)]=_cluster_feature(
            members,f,v,area,centres,deltas)
    start_count=len(groups)
    require(start_count>=2,"HIERARCHY_NO_INITIAL_PATCHES")
    # Stable heap with versioned representatives. Keeps all source faces in
    # distinct *atomic* provenance; joining groups never welds triangles.
    versions={i:0 for i in groups}
    heap=[]
    for i in sorted(groups):
        for j in sorted(groups):
            if j<=i:continue
            if _forbidden((groups[i],groups[j]),constraints):continue
            heapq.heappush(heap,(_distance(groups[i],groups[j],scale),
                                 i,j,versions[i],versions[j]))
    snapshots={}
    history=[]
    targets=set(k for k in req if k<start_count)
    if start_count in req:
        snapshots[start_count]=initial.copy()
    while len(groups)>2 and heap:
        cost,i,j,vi,vj=heapq.heappop(heap)
        if i not in groups or j not in groups or vi!=versions[i] or vj!=versions[j]:
            continue
        if _forbidden((groups[i],groups[j]),constraints):
            continue
        groups[i]=_combine(groups[i],groups[j])
        versions[i]+=1
        del groups[j]
        del versions[j]
        history.append({
            "remaining_groups":len(groups),
            "heuristic_merge_cost":float(cost),
            "source_atoms_not_welded":True,
        })
        for other in sorted(groups):
            if other==i:continue
            if _forbidden((groups[i],groups[other]),constraints):continue
            left,right=sorted((i,other))
            heapq.heappush(heap,(
                _distance(groups[left],groups[right],scale),
                left,right,versions[left],versions[right]))
        if len(groups) in targets:
            labels=np.empty(len(f),dtype=np.int32)
            for idx,feat in groups.items():
                labels[np.asarray(feat["members"],np.int64)]=idx
            snapshots[len(groups)]=canonicalize_partition(labels)
    for target,labels in snapshots.items():
        require(len(np.unique(labels))==target,"HIERARCHY_GROUP_CENSUS")
        require(all(labels[a]!=labels[b] for a,b in constraints),
                "HIERARCHY_TRAIN_EDIT_NO_CROSSMERGE")
        require(np.array_equal(
            canonicalize_partition(labels),labels),
            "HIERARCHY_GROUP_CANONICAL")
    return {
        "initial_patch_count":start_count,
        "training_view_cannot_link_count":len(constraints),
        "snapshots":snapshots,
        "history":history,
        "lowest_reached_group_count":len(groups),
        "source_faces_unchanged":True,
        "geometry_vertices_unchanged":True,
        "no_vertex_welding":True,
        "source_semantic_names_used":False,
        "test_view_used_for_merge":False,
    }


def build(rest,faces,poses,parent_proposal,parent_edit_court,
          train_view_pairs,*,target_counts=TARGETS):
    require(parent_proposal.get("schema")==PARENT_SCHEMA
            and parent_proposal.get("training_authorized") is False
            and parent_proposal.get("graphics_owner_minted") is False
            and parent_proposal.get("source_semantic_or_material_names_used") is False,
            "HIERARCHY_PARENT_PROPOSAL_SCOPE")
    require(parent_edit_court.get("schema")==PROXY_SCHEMA
            and parent_edit_court.get("training_authorized") is False
            and parent_edit_court.get("graphics_family_authorized") is False
            and parent_edit_court.get("qualified_same_drawable_positives")==0
            and parent_edit_court.get("qualified_different_drawable_negatives")==0
            and parent_edit_court.get("train_view")==0
            and parent_edit_court.get("heldout_view")==4,
            "HIERARCHY_PARENT_EDIT_COURT_SCOPE")
    selected=parent_edit_court.get("selected_candidate_train_only")
    candidate=next((x for x in parent_proposal.get("candidates") or []
                    if x["candidate"]==selected),None)
    require(candidate is not None and
            candidate["slot_count"]==parent_edit_court["selected_slot_count"],
            "HIERARCHY_SELECTED_PARTITION_NOT_BOUND")
    v,f,p,_=_array_faces(rest,faces,poses)
    require(parent_proposal.get("face_count")==len(f)
            and parent_proposal.get("authored_pose_count")==len(p),
            "HIERARCHY_PARENT_GEOMETRY_CENSUS")
    cannot=[]
    for row in train_view_pairs:
        require(row.get("view_index")==0
                and type(row.get("source_face_a_teacher_only")) is int
                and type(row.get("source_face_b_teacher_only")) is int
                and row.get("same_drawable")=="UNKNOWN"
                and row.get("different_drawable")=="UNKNOWN"
                and row.get("training_authorized") is False,
                "HIERARCHY_TRAIN_WITNESS_STATUS")
        cannot.append((row["source_face_a_teacher_only"],
                       row["source_face_b_teacher_only"]))
    measured=coalesce(v,f,p,candidate["face_slot_teacher_only"],cannot,
                     target_counts=target_counts)
    candidates=[]
    for target,labels in sorted(measured["snapshots"].items()):
        candidates.append({
            "candidate":"HIERARCHY_GROUPS_"+str(target),
            "slot_count":int(len(np.unique(labels))),
            "face_slot_teacher_only":labels.tolist(),
            "source_import_appearance_retained":True,
            "render_full_image_heldout_executed":False,
            "source_face_to_layer_is_operational_hypothesis":True,
            "training_authorized":False,
        })
    return {
        "schema":SCHEMA,
        "status":"HIERARCHICAL_ANONYMOUS_GROUP_CANDIDATES__FULL_RENDER_COURT_REQUIRED",
        "parents":dict(parent_proposal["parents"]),
        "base_candidate":selected,
        "base_atomic_patch_count":measured["initial_patch_count"],
        "face_count":len(f),
        "authored_pose_count":len(p),
        "training_view":0,"heldout_view":4,
        "cannot_link_source_pairs_from_train_only":len(cannot),
        "heldout_pairs_used_for_grouping":False,
        "source_names_and_materials_used":False,
        "potential_target_counts":list(target_counts),
        "actual_candidate_count":len(candidates),
        "candidates":candidates,
        "measured_merges":measured["history"],
        "source_faces_unchanged":True,
        "no_vertex_welding":True,
        "author_drawable_truth_minted":False,
        "operational_layer_training_authorized":False,
        "training_authorized":False,
    }
