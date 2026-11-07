#!/usr/bin/env python3
"""8-view source geometry hypotheses -> anonymous operational layer groups.

Geometric ray hits (not rendered/alpha-qualified) are PROPOSAL constraints
only. Use views 0,2,6, never view 4, to avoid fitting test-view edit
consequences. A/B along an intact geometric ray may belong to the SAME
author drawable; do NOT mint owner positives or negatives.

The prior source-anchored anonymous edit partition is retained. Where it
already combines geometry-hit A/B, split only a minimal addressed atomic face
before grouping. This is an *editable hypothesis*, not an artist-authoring
identity assertion.
"""
from __future__ import annotations

import numpy as np

from tools.corpus.compile_corpus370_hierarchical_layer_groups_v1 import (
    coalesce,require,
)
from tools.corpus.topology_partition_court_v1 import canonicalize_partition

SCHEMA="RealSaS.Corpus370MultiviewGeometricAnonymousGroups.v2"
RAY_SCHEMA="RealSaS.Corpus370V5SourceLocusGeometricRayCandidate.v1"
TRAIN_VIEWS=(0,2,6)
HELDOUT_VIEW=4
TARGETS=(8,16,32,64,128)


def geometry_constraints(geometry,*,source_sha,face_count):
    require(geometry.get("schema")==RAY_SCHEMA
            and geometry.get("status")==
                "SOURCE_GEOMETRIC_RAY_CANDIDATES_MEASURED__RENDERED_ALPHA_AND_OWNER_COURTS_REQUIRED"
            and geometry.get("source_sha256")==source_sha
            and geometry.get("authority",{}).get("training_authorized") is False
            and geometry.get("counts",{}).get("proposed_pixels")==len(geometry.get("rays") or [])
            and geometry.get("counts",{}).get("proposed_pixels")==64,
            "MULTIVIEW_GEOMETRIC_SOURCE_RAY_SCHEMA")
    constraints=set()
    valid={v:0 for v in TRAIN_VIEWS}
    views=set()
    for ray in geometry["rays"]:
        view=ray.get("view_index")
        require(type(view) is int and 0<=view<8,
                "MULTIVIEW_GEOMETRIC_VIEW_BOUNDS")
        views.add(view)
        if view not in TRAIN_VIEWS:continue
        if ray.get("status")!="VALID_GEOMETRIC_FULL_STACK_CANDIDATE":
            continue
        require(ray.get("complete") is True
                and ray.get("has_depth_tie") is False
                and ray.get("drawable_ownership_label_minted") is False
                and ray.get("source_alpha_compositing_qualified") is False,
                "MULTIVIEW_GEOMETRIC_FALSE_SOURCE_HIT_AUTHORITY")
        hits=ray.get("hits") or []
        if len(hits)<2:continue
        a=hits[0].get("face_index_teacher_only")
        b=hits[1].get("face_index_teacher_only")
        require(type(a) is int and type(b) is int
                and 0<=a<face_count and 0<=b<face_count
                and a!=b,
                "MULTIVIEW_GEOMETRIC_TWO_SOURCE_FACE_ADDRESSES")
        constraints.add((min(a,b),max(a,b)))
        valid[view]+=1
    require(views==set(range(8)) and
            all(0<valid[v]<=8 for v in TRAIN_VIEWS),
            "MULTIVIEW_GEOMETRIC_3_TRAIN_VIEW_CENSUS")
    require(constraints,"MULTIVIEW_GEOMETRIC_NO_PAIR_HYPOTHESES")
    return sorted(constraints),valid


def refine_only_conflicts(labels,cannot_links):
    base=canonicalize_partition(labels)
    refined=base.astype(np.int64).copy()
    fresh=int(refined.max())+1
    breaks=[]
    for a,b in sorted(cannot_links):
        if refined[a]==refined[b]:
            # B is the deterministic seed detached from this particular
            # old patch; a split changes operational granularity, never mesh.
            refined[b]=fresh
            fresh+=1
            breaks.append((a,b))
    refined=canonicalize_partition(refined)
    require(all(refined[a]!=refined[b] for a,b in cannot_links),
            "MULTIVIEW_GEOMETRIC_MINIMAL_REFINEMENT_CONTRADICTION")
    return refined,breaks


def build(rest,faces,poses,old_proposal,old_court,source_ray_doc):
    require(old_proposal.get("schema")==
            "RealSaS.Corpus370AnonymousLayerProposal.v1"
            and old_proposal.get("training_authorized") is False
            and old_proposal.get("graphics_owner_minted") is False
            and old_court.get("schema")==
            "RealSaS.Corpus370AnonymousLayerEditProxyCourt.v1"
            and old_court.get("training_authorized") is False
            and old_court.get("graphics_family_authorized") is False,
            "MULTIVIEW_GEOMETRIC_OLD_ANON_AUTHORITY")
    original=next((c for c in old_proposal.get("candidates") or []
                  if c["candidate"]==old_court["selected_candidate_train_only"]),None)
    require(original is not None and
            original["slot_count"]==old_court["selected_slot_count"],
            "MULTIVIEW_GEOMETRIC_PREVIOUS_SELECTED_PARTITION")
    f=np.asarray(faces)
    require(f.shape==(old_proposal["face_count"],3),
            "MULTIVIEW_GEOMETRIC_SOURCE_FACE_CENSUS")
    pairlinks,views=geometry_constraints(
        source_ray_doc,source_sha=old_proposal["parents"]["source_sha256"],
        face_count=len(f))
    old=np.asarray(original["face_slot_teacher_only"])
    refined,broken=refine_only_conflicts(old,pairlinks)
    grouped=coalesce(rest,f,poses,refined,pairlinks,target_counts=TARGETS)
    suggestions=[]
    for slots,labels in sorted(grouped["snapshots"].items()):
        suggestions.append({
            "candidate":"MULTIVIEW_GEOMETRY_GROUPS_"+str(slots),
            "slot_count":int(slots),
            "face_slot_teacher_only":labels.astype(int).tolist(),
            "geometry_only_ray_constraints_not_owner_truth":True,
            "full_frame_source_alpha_edit_replay":False,
            "training_authorized":False,
        })
    require(suggestions,"MULTIVIEW_GEOMETRIC_NO_QUALIFIED_GROUP_LEVEL")
    return {
        "schema":SCHEMA,
        "status":"ALL_VIEW_SOURCE_GEOMETRY_PROPOSAL__HELDOUT_RENDER_EFFECTS_UNKNOWN",
        "parents":dict(old_proposal["parents"]),
        "source_ray_schema":RAY_SCHEMA,
        "source_sha256":old_proposal["parents"]["source_sha256"],
        "old_selected_candidate":old_court["selected_candidate_train_only"],
        "old_source_atomic_group_count":original["slot_count"],
        "refined_patch_count":grouped["initial_patch_count"],
        "minimal_face_refinements_due_to_geometry_count":len(broken),
        "geometry_candidate_constraint_count":len(pairlinks),
        "geometry_train_view_candidate_counts":{str(k):v for k,v in views.items()},
        "train_geometry_views":list(TRAIN_VIEWS),
        "heldout_render_edit_view":HELDOUT_VIEW,
        "heldout_render_outcomes_used_for_selection":False,
        "heldout_geometry_view_used_for_selection":False,
        "face_count":len(f),
        "authored_pose_count":len(poses),
        "candidate_count":len(suggestions),
        "candidates":suggestions,
        "author_drawable_truth_minted":False,
        "geometry_heuristic_positives_as_labels":False,
        "model_training_authorized":False,
        "product_compilation_authorized":False,
        "training_authorized":False,
    }
