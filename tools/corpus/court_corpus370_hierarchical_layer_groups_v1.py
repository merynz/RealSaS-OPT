#!/usr/bin/env python3
"""Independent bounded editor-group consumer, never author layer truth.

This checks exact mesh face coverage, that grouping only coarsens the prior
atomic source-face partition, that all TRAIN-view A/B witnesses remain
separately toggleable and that HELDOUT view 4 is used only as an untouched
diagnostic. It cannot qualify render/appearance/CAA/attachment/owner family.

An operational grouping of disjoint source islands is permitted; *welding*
geometry is never permitted. A future full-image alpha/occlusion court is
mandatory before any student target or product layer is released.
"""
from __future__ import annotations

import math
import numpy as np

from tools.corpus.topology_partition_court_v1 import (
    canonicalize_partition,partition_refines,
)

SCHEMA="RealSaS.Corpus370HierarchicalLayerIndependentEditCourt.v1"
PROPOSAL_SCHEMA="RealSaS.Corpus370HierarchicalAnonymousLayerGroups.v1"
SOURCE_PAIRS="RealSaS.Corpus370GenericPairKinematicDiagnosticBatch.v1"
TRAIN=0
HELDOUT=4
ALPHA=0.2


def require(ok,code):
    if not ok:raise ValueError(code)


def verify(proposal,source_pairs,base_proposal,base_court):
    require(proposal.get("schema")==PROPOSAL_SCHEMA
            and proposal.get("status")==
                 "HIERARCHICAL_ANONYMOUS_GROUP_CANDIDATES__FULL_RENDER_COURT_REQUIRED"
            and proposal.get("training_authorized") is False
            and proposal.get("author_drawable_truth_minted") is False
            and proposal.get("operational_layer_training_authorized") is False
            and proposal.get("source_names_and_materials_used") is False
            and proposal.get("heldout_pairs_used_for_grouping") is False
            and proposal.get("source_faces_unchanged") is True
            and proposal.get("no_vertex_welding") is True
            and proposal.get("training_view")==TRAIN and proposal.get("heldout_view")==HELDOUT,
            "HIER_COURT_PARENT_SCOPE")
    require(source_pairs.get("schema")==SOURCE_PAIRS
            and source_pairs.get("training_authorized") is False
            and source_pairs.get("qualified_same_drawable_positives")==0
            and source_pairs.get("qualified_different_drawable_negatives")==0
            and source_pairs.get("pair_count")==len(source_pairs.get("rows") or []),
            "HIER_COURT_SOURCE_PAIR_SCOPE")
    require(proposal.get("parents")==base_proposal.get("parents")
            and proposal["parents"].get("source_sha256")==
                source_pairs.get("parents",{}).get("source_sha256")
            and base_court["selected_candidate_train_only"]==
                proposal["base_candidate"],
            "HIER_COURT_PARENT_BINDING")
    previous=next((x for x in base_proposal.get("candidates") or []
                   if x.get("candidate")==proposal["base_candidate"]),None)
    require(previous is not None and
            previous.get("slot_count")==proposal["base_atomic_patch_count"],
            "HIER_COURT_BASE_PATCH_PROVENANCE")
    n=proposal.get("face_count")
    initial=np.asarray(previous.get("face_slot_teacher_only"))
    require(type(n) is int and n>=2 and initial.shape==(n,)
            and initial.dtype.kind in "iu"
            and np.array_equal(canonicalize_partition(initial),initial),
            "HIER_COURT_FACE_PARTITION_INITIAL")
    rows=source_pairs["rows"]
    require(all(x.get("view_index") in (TRAIN,HELDOUT)
                and x.get("source_hit_visibility_owner")=="SOURCE_LOCUS_ONLY"
                and x.get("same_drawable")=="UNKNOWN"
                and x.get("different_drawable")=="UNKNOWN"
                and x.get("training_authorized") is False
                for x in rows),
            "HIER_COURT_SOURCE_PAIR_IDENTITY")
    train=[x for x in rows if x["view_index"]==TRAIN]
    heldout=[x for x in rows if x["view_index"]==HELDOUT]
    require(len(train)>0 and len(heldout)>0
            and len(train)==proposal.get("cannot_link_source_pairs_from_train_only"),
            "HIER_COURT_VIEW_CENSUS")
    all_faces={z for x in rows
               for z in (x["source_face_a_teacher_only"],
                         x["source_face_b_teacher_only"])}
    require(all(type(x) is int and 0<=x<n for x in all_faces),
            "HIER_COURT_FACE_ADDRESS")
    require(all(initial[x["source_face_a_teacher_only"]]!=
                initial[x["source_face_b_teacher_only"]]
                for x in train),"HIER_COURT_BASE_TRAIN_CONTRADICTION")
    candidates=proposal.get("candidates")
    require(isinstance(candidates,list)
            and len(candidates)==proposal.get("actual_candidate_count")
            and len(candidates)>0
            and len(candidates)==len(set(x["candidate"] for x in candidates)),
            "HIER_COURT_CANDIDATE_COUNT")
    scored=[]
    for candidate in candidates:
        labels=np.asarray(candidate.get("face_slot_teacher_only"))
        count=candidate.get("slot_count")
        require(labels.shape==(n,)
                and labels.dtype.kind in "iu"
                and np.array_equal(canonicalize_partition(labels),labels)
                and int(np.max(labels))+1==count
                and count<=proposal["base_atomic_patch_count"]
                and partition_refines(initial,labels)
                and candidate.get("source_import_appearance_retained") is True
                and candidate.get("source_face_to_layer_is_operational_hypothesis") is True
                and candidate.get("render_full_image_heldout_executed") is False
                and candidate.get("training_authorized") is False,
                "HIER_COURT_COARSEN_ONLY_PARTITION")
        require(candidate["candidate"]=="HIERARCHY_GROUPS_"+str(count),
                "HIER_COURT_GROUP_COUNT_NAME")
        tr=sum(labels[r["source_face_a_teacher_only"]]!=
               labels[r["source_face_b_teacher_only"]] for r in train)
        ho=sum(labels[r["source_face_a_teacher_only"]]!=
               labels[r["source_face_b_teacher_only"]] for r in heldout)
        require(tr==len(train),"HIER_COURT_TRAIN_CANNOT_LINK_VIOLATED")
        cost=ALPHA*(count-1)/(n-1)
        loss=(len(train)-tr)/len(train)+cost
        scored.append({
            "candidate":candidate["candidate"],
            "slot_count":count,
            "train_reveal_expressible":int(tr),
            "train_ray_count":len(train),
            "heldout_reveal_expressible":int(ho),
            "heldout_ray_count":len(heldout),
            "train_only_proxy_loss":loss,
            "heldout_reveal_fraction_diagnostic_only":ho/len(heldout),
            "source_face_preservation":True,
            "operational_layer_compilation_authorized":False,
        })
    chosen=min(scored,key=lambda x:(
        x["train_only_proxy_loss"],x["slot_count"],x["candidate"]))
    prior=int(proposal["base_atomic_patch_count"])
    require(prior>=chosen["slot_count"],"HIER_COURT_FALSE_COMPRESSION")
    coverage=len(all_faces)/n
    # Diagnostics cannot turn an unsupported 2-ray sample into true owners.
    return {
        "schema":SCHEMA,
        "status":"TRAIN_VIEW_REVEAL_CONSTRAINED_HIERARCHICAL_GROUPING__FULL_IMAGE_REPLAY_REQUIRED",
        "parents":dict(proposal["parents"]),
        "base_patch_count":prior,
        "selected_candidate_train_only":chosen["candidate"],
        "selected_slot_count":chosen["slot_count"],
        "group_compression_ratio":prior/chosen["slot_count"],
        "source_face_count":n,
        "independent_ray_face_coverage_fraction":coverage,
        "train_ray_count":len(train),
        "heldout_ray_count":len(heldout),
        "selected_heldout_edit_expressibility":chosen[
            "heldout_reveal_fraction_diagnostic_only"],
        "insufficient_spatial_evidence":len(rows)<max(8,math.ceil(n/200)),
        "candidate_scores":scored,
        "selection_uses_heldout":False,
        "render_appearance_parity_courted":False,
        "full_image_alpha_depth_replay_executed":False,
        "functional_product_quality":"ABSTAIN_FULL_IMAGE_AND_SPATIAL_EVIDENCE_REQUIRED",
        "true_author_drawable_partition_qualified":False,
        "same_drawable_positive_pairs":0,
        "different_drawable_negative_pairs":0,
        "graphics_family_authorized":False,
        "mechanical_response_authorized":False,
        "tessa_teacher_authorized":False,
        "student_targets_minted":False,
        "training_authorized":False,
    }
