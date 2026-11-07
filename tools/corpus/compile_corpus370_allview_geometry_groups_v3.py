#!/usr/bin/env python3
"""All 8 source-view geometry; *rendered edit response* is withheld.

This is an intentionally different evaluation from view-holdout v2. A compile
product receives all original 8-view geometry at input; view4 geometry is
available. The real Blender edit consequence for view4 is NOT read by this
proposal. Do not claim unseen-view generalization or author drawable truth.
"""
from __future__ import annotations

import numpy as np

from tools.corpus.compile_corpus370_hierarchical_layer_groups_v1 import (
    coalesce,require,
)
from tools.corpus.compile_corpus370_multiview_geometry_groups_v2 import (
    RAY_SCHEMA,refine_only_conflicts,
)

SCHEMA="RealSaS.Corpus370AllViewGeometryOperationalGroups.v3"
INPUT_VIEWS=tuple(range(8))
TARGETS=(8,16,32,64,128)


def allview_geometry_links(doc,*,source_sha,face_count):
    require(doc.get("schema")==RAY_SCHEMA
            and doc.get("status")==
                "SOURCE_GEOMETRIC_RAY_CANDIDATES_MEASURED__RENDERED_ALPHA_AND_OWNER_COURTS_REQUIRED"
            and doc.get("source_sha256")==source_sha
            and doc.get("authority",{}).get("training_authorized") is False
            and doc.get("authority",{}).get("graphics_family_minted") is False
            and doc.get("counts",{}).get("proposed_pixels")==64
            and len(doc.get("rays") or [])==64,
            "ALLVIEW_SOURCE_RAY_SHA_CENSUS")
    views={i:0 for i in INPUT_VIEWS}
    constraints=set()
    for row in doc["rays"]:
        v=row.get("view_index")
        require(type(v) is int and v in INPUT_VIEWS,
                "ALLVIEW_GEOMETRIC_VIEW")
        if row.get("status")!="VALID_GEOMETRIC_FULL_STACK_CANDIDATE":
            continue
        require(row.get("complete") is True
                and row.get("has_depth_tie") is False
                and row.get("drawable_ownership_label_minted") is False
                and row.get("source_alpha_compositing_qualified") is False,
                "ALLVIEW_GEOMETRIC_FAKED_AUTHORITY")
        hits=row.get("hits") or []
        if len(hits)<2:continue
        a=hits[0].get("face_index_teacher_only")
        b=hits[1].get("face_index_teacher_only")
        require(type(a) is int and type(b) is int
                and 0<=a<face_count and 0<=b<face_count and a!=b,
                "ALLVIEW_GEOMETRIC_INVALID_FACE")
        constraints.add((min(a,b),max(a,b)))
        views[v]+=1
    require(all(v>=1 for v in views.values())
            and all(v<=8 for v in views.values())
            and constraints,
            "ALLVIEW_GEOMETRIC_EIGHT_CAMERA_COVERAGE")
    return sorted(constraints),views


def build(rest,faces,poses,old_proposal,old_court,ray_doc):
    require(old_proposal.get("schema")==
                "RealSaS.Corpus370AnonymousLayerProposal.v1"
            and old_proposal.get("training_authorized") is False
            and old_proposal.get("graphics_owner_minted") is False
            and old_proposal.get("source_semantic_or_material_names_used") is False
            and old_court.get("schema")==
                "RealSaS.Corpus370AnonymousLayerEditProxyCourt.v1"
            and old_court.get("training_authorized") is False
            and old_court.get("graphics_family_authorized") is False,
            "ALLVIEW_OLD_SOURCE_AUTHORITY")
    selected=old_court["selected_candidate_train_only"]
    base=next((x for x in old_proposal["candidates"]
               if x["candidate"]==selected),None)
    require(base is not None
            and base["slot_count"]==old_court["selected_slot_count"],
            "ALLVIEW_OLD_SOURCE_SELECTED")
    tri=np.asarray(faces)
    require(tri.shape==(old_proposal["face_count"],3),
            "ALLVIEW_SOURCE_TRIANGLE_CENSUS")
    links,view_counts=allview_geometry_links(
        ray_doc,source_sha=old_proposal["parents"]["source_sha256"],
        face_count=len(tri))
    initial=np.asarray(base["face_slot_teacher_only"])
    refined,breaks=refine_only_conflicts(initial,links)
    grouped=coalesce(rest,tri,poses,refined,links,target_counts=TARGETS)
    suggestions=[{
        "candidate":"ALLVIEW_GEOMETRY_GROUPS_"+str(k),
        "slot_count":int(k),
        "face_slot_teacher_only":labels.astype(int).tolist(),
        "all_eight_camera_geometry_is_compiler_input":True,
        "view4_rendered_edit_response_exposed_to_compiler":False,
        "full_image_edit_render_courted":False,
        "source_faces_changed":False,
        "training_authorized":False,
    } for k,labels in sorted(grouped["snapshots"].items())]
    require(suggestions and grouped["source_faces_unchanged"] is True,
            "ALLVIEW_NO_GROUP_HYPOTHESIS")
    return {
        "schema":SCHEMA,
        "status":"ALL_8_INPUT_VIEW_GEOMETRY_LAYER_HYPOTHESES__RENDER_OUTCOME_HELDOUT",
        "parents":dict(old_proposal["parents"]),
        "source_sha256":old_proposal["parents"]["source_sha256"],
        "prior_anonymous_candidate":selected,
        "prior_layer_count":base["slot_count"],
        "refined_source_patch_count":grouped["initial_patch_count"],
        "geometry_candidate_link_count":len(links),
        "source_face_refinement_count":len(breaks),
        "input_geometry_view_counts":{str(k):v for k,v in view_counts.items()},
        "source_geometry_views_used_for_compilation":list(INPUT_VIEWS),
        "view4_geometry_seen_by_compiler":True,
        "view4_rendered_edit_consequences_used_by_compiler":False,
        "evaluation_design":"RENDER_INTERVENTION_HOLDOUT_NOT_CAMERA_HOLDOUT",
        "source_face_count":len(tri),
        "source_native_pose_count":len(poses),
        "candidate_count":len(suggestions),
        "candidates":suggestions,
        "graphics_owner_truth_minted":False,
        "mechanical_response_authorized":False,
        "training_authorized":False,
        "product_compilation_authorized":False,
    }
