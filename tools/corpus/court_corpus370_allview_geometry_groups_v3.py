#!/usr/bin/env python3
"""Independent source-face and withheld Blender *edit* outcome consumer, v3.

All 8 geometry views are observed by v3 compiler (including 4). It is only
view4 renderer *intervention effects* that are withheld until this court.
Do NOT mislabel this as unseen-view generalization, original drawable labels,
or a complete full-frame alpha/CAA/product qualification.
"""
from __future__ import annotations
import math
import numpy as np

from tools.corpus.compile_corpus370_allview_geometry_groups_v3 import (
    allview_geometry_links,
)
from tools.corpus.topology_partition_court_v1 import canonicalize_partition

SCHEMA="RealSaS.Corpus370AllViewGeometryEditorProxyCourt.v3"
PROPOSAL="RealSaS.Corpus370AllViewGeometryOperationalGroups.v3"
RENDER_PAIRS="RealSaS.Corpus370GenericPairKinematicDiagnosticBatch.v1"


def require(value,code):
    if not value:raise ValueError(code)


def court(proposal,ray_doc,real_render_pairs,old_anon):
    require(proposal.get("schema")==PROPOSAL
            and proposal.get("status")==
                "ALL_8_INPUT_VIEW_GEOMETRY_LAYER_HYPOTHESES__RENDER_OUTCOME_HELDOUT"
            and proposal.get("view4_geometry_seen_by_compiler") is True
            and proposal.get("view4_rendered_edit_consequences_used_by_compiler") is False
            and proposal.get("source_geometry_views_used_for_compilation")==list(range(8))
            and proposal.get("evaluation_design")==
                "RENDER_INTERVENTION_HOLDOUT_NOT_CAMERA_HOLDOUT"
            and proposal.get("training_authorized") is False
            and proposal.get("product_compilation_authorized") is False
            and proposal.get("graphics_owner_truth_minted") is False,
            "ALLVIEW_COURT_SOURCE_PROPOSAL_SCOPE")
    require(old_anon.get("schema")==
                "RealSaS.Corpus370AnonymousLayerProposal.v1"
            and old_anon.get("training_authorized") is False
            and proposal.get("parents")==old_anon.get("parents")
            and real_render_pairs.get("schema")==RENDER_PAIRS
            and real_render_pairs.get("training_authorized") is False
            and real_render_pairs.get("qualified_same_drawable_positives")==0
            and real_render_pairs.get("qualified_different_drawable_negatives")==0
            and proposal.get("source_sha256")==
                real_render_pairs.get("parents",{}).get("source_sha256"),
            "ALLVIEW_COURT_PARENTS_AND_SOURCE")
    n=proposal.get("source_face_count")
    require(type(n) is int and n>1,
            "ALLVIEW_COURT_SOURCE_FACE_COUNT")
    links,view_counts=allview_geometry_links(
        ray_doc,source_sha=proposal["source_sha256"],face_count=n)
    require(len(links)==proposal["geometry_candidate_link_count"]
            and {str(k):v for k,v in view_counts.items()}==
                proposal["input_geometry_view_counts"],
            "ALLVIEW_COURT_INDEPENDENT_LINK_REPLAY")
    rows=real_render_pairs.get("rows") or []
    require(len(rows)==2
            and {x.get("view_index") for x in rows}=={0,4}
            and all(x.get("source_hit_visibility_owner")=="SOURCE_LOCUS_ONLY"
                    and x.get("same_drawable")=="UNKNOWN"
                    and x.get("different_drawable")=="UNKNOWN"
                    and x.get("training_authorized") is False
                    for x in rows),
            "ALLVIEW_COURT_RENDER_RAY_SCOPE")
    names=set()
    scores=[]
    for item in proposal.get("candidates") or []:
        labels=np.asarray(item.get("face_slot_teacher_only"))
        count=item.get("slot_count")
        require(item.get("candidate")=="ALLVIEW_GEOMETRY_GROUPS_"+str(count)
                and type(count) is int and 2<=count<=128
                and item.get("all_eight_camera_geometry_is_compiler_input") is True
                and item.get("view4_rendered_edit_response_exposed_to_compiler") is False
                and item.get("full_image_edit_render_courted") is False
                and item.get("source_faces_changed") is False
                and item.get("training_authorized") is False
                and labels.shape==(n,)
                and labels.dtype.kind in "iu"
                and np.array_equal(labels,canonicalize_partition(labels))
                and int(labels.max())+1==count
                and all(labels[a]!=labels[b] for a,b in links),
                "ALLVIEW_COURT_GROUP_PROPOSAL_INTEGRITY")
        require(item["candidate"] not in names,"ALLVIEW_COURT_DUPLICATE_GROUPS")
        names.add(item["candidate"])
        train=next(x for x in rows if x["view_index"]==0)
        held=next(x for x in rows if x["view_index"]==4)
        success_train=bool(labels[train["source_face_a_teacher_only"]]!=
                           labels[train["source_face_b_teacher_only"]])
        success_held=bool(labels[held["source_face_a_teacher_only"]]!=
                          labels[held["source_face_b_teacher_only"]])
        train_loss=(0 if success_train else 1) + 0.2*(count-1)/(n-1)
        scores.append({
            "candidate":item["candidate"],
            "slot_count":count,
            "training_view0_edit_expressible":success_train,
            "render_intervention_heldout_view4_edit_expressible":success_held,
            "training_only_proxy_loss":float(train_loss),
            "full_frame_render_replay_executed":False,
        })
    require(len(scores)==proposal.get("candidate_count") and scores,
            "ALLVIEW_COURT_CANDIDATE_CENSUS")
    choice=min(scores,key=lambda x:(x["training_only_proxy_loss"],
                                   x["slot_count"],x["candidate"]))
    faces_seen={f for r in rows for f in
                (r["source_face_a_teacher_only"],r["source_face_b_teacher_only"])}
    return {
        "schema":SCHEMA,
        "status":"ALLVIEW_GEOMETRY_WITH_INDEPENDENT_RENDER_EDIT_PROXY__NO_TRAINING",
        "selected_candidate_train_only":choice["candidate"],
        "selected_group_count":choice["slot_count"],
        "source_rest_face_count":n,
        "geometry_candidate_constraint_count":len(links),
        "independent_render_rays_measured":len(rows),
        "independent_render_face_coverage":len(faces_seen)/n,
        "heldout_render_edit_expressible":choice[
            "render_intervention_heldout_view4_edit_expressible"],
        "geometry_view4_seen_for_compilation":True,
        "render_intervention_view4_seen_for_compilation":False,
        "all_geometry_views_are_compiler_inputs":True,
        "view4_not_unseen_geometry":True,
        "not_view_holdout_generalization":True,
        "full_fov_rgba_render_replay_required":True,
        "full_native_resolution_caa_qualified":False,
        "independent_mechanical_control_court_qualified":False,
        "source_author_drawable_identity_qualified":False,
        "different_drawable_negative_labels":0,
        "same_drawable_positive_labels":0,
        "graphics_owner_family_authorized":False,
        "training_authorized":False,
        "candidate_scores":scores,
    }
