#!/usr/bin/env python3
"""Independent original-ray cross-view layer GROUP court; source names prohibited.

Train-only geometry visibility *hypotheses* are checked against the actual
352-subject 8-view source ray stack. Heldout view4 independently measures
source-hidden edit *expressibility*, NOT image/alpha recovery.
The court does NOT convert geometry occlusion into same/different drawable.
"""
from __future__ import annotations

import math
import numpy as np

from tools.corpus.compile_corpus370_multiview_geometry_groups_v2 import (
    geometry_constraints,
)
from tools.corpus.topology_partition_court_v1 import (
    canonicalize_partition,
)

SCHEMA="RealSaS.Corpus370MultiviewGeometryAnonymousGroupCourt.v2"
SOURCE_PAIRS="RealSaS.Corpus370GenericPairKinematicDiagnosticBatch.v1"
PROPOSAL="RealSaS.Corpus370MultiviewGeometricAnonymousGroups.v2"


def require(ok,code):
    if not ok:raise ValueError(code)


def qualify(proposal,source_rays,actual_graphics_pairs,prior_anon):
    require(proposal.get("schema")==PROPOSAL
            and proposal.get("status")==
                "ALL_VIEW_SOURCE_GEOMETRY_PROPOSAL__HELDOUT_RENDER_EFFECTS_UNKNOWN"
            and proposal.get("model_training_authorized") is False
            and proposal.get("product_compilation_authorized") is False
            and proposal.get("author_drawable_truth_minted") is False
            and proposal.get("geometry_heuristic_positives_as_labels") is False
            and proposal.get("heldout_render_outcomes_used_for_selection") is False
            and proposal.get("heldout_geometry_view_used_for_selection") is False
            and proposal.get("train_geometry_views")==[0,2,6]
            and proposal.get("heldout_render_edit_view")==4,
            "GEOMETRIC_MULTIVIEW_COURT_NOT_QUALIFIED")
    require(actual_graphics_pairs.get("schema")==SOURCE_PAIRS
            and actual_graphics_pairs.get("training_authorized") is False
            and actual_graphics_pairs.get("qualified_same_drawable_positives")==0
            and actual_graphics_pairs.get("qualified_different_drawable_negatives")==0
            and proposal.get("source_sha256")==
                actual_graphics_pairs.get("parents",{}).get("source_sha256")
            and prior_anon.get("schema")==
                "RealSaS.Corpus370AnonymousLayerProposal.v1"
            and proposal.get("parents")==prior_anon.get("parents"),
            "GEOMETRIC_MULTIVIEW_COURT_SOURCE_JOIN")
    face_count=proposal.get("face_count")
    require(type(face_count) is int and face_count>=2,
            "GEOMETRIC_MULTIVIEW_COURT_FACE_CENSUS")
    constraints,views=geometry_constraints(source_rays,
        source_sha=proposal["source_sha256"],face_count=face_count)
    require(len(constraints)==proposal["geometry_candidate_constraint_count"]
            and {str(k):v for k,v in views.items()}==
                proposal["geometry_train_view_candidate_counts"],
            "GEOMETRIC_MULTIVIEW_COURT_CONSTRAINT_CENSUS")
    rows=actual_graphics_pairs.get("rows") or []
    train=[x for x in rows if x.get("view_index")==0]
    heldout=[x for x in rows if x.get("view_index")==4]
    require(train and heldout and
            all(x.get("same_drawable")=="UNKNOWN"
                and x.get("different_drawable")=="UNKNOWN"
                and x.get("training_authorized") is False for x in rows),
            "GEOMETRIC_MULTIVIEW_COURT_RENDER_PAIR_AUTHORITY")
    candidates=proposal.get("candidates") or []
    require(len(candidates)==proposal["candidate_count"] and
            len(set(x.get("candidate") for x in candidates))==len(candidates),
            "GEOMETRIC_MULTIVIEW_COURT_CANDIDATE_CENSUS")
    scores=[]
    for c in candidates:
        labels=np.asarray(c.get("face_slot_teacher_only"))
        count=c.get("slot_count")
        require(labels.shape==(face_count,)
                and labels.dtype.kind in "iu"
                and np.array_equal(labels,canonicalize_partition(labels))
                and int(np.max(labels))+1==count
                and c.get("candidate")=="MULTIVIEW_GEOMETRY_GROUPS_"+str(count)
                and c.get("training_authorized") is False
                and c.get("geometry_only_ray_constraints_not_owner_truth") is True
                and c.get("full_frame_source_alpha_edit_replay") is False,
                "GEOMETRIC_MULTIVIEW_COURT_PARTITION_FORGERY")
        require(all(labels[a]!=labels[b] for a,b in constraints),
                "GEOMETRIC_MULTIVIEW_COURT_SOURCE_GEOMETRY_VIOLATED")
        train_ok=sum(int(labels[x["source_face_a_teacher_only"]]!=
                         labels[x["source_face_b_teacher_only"]]) for x in train)
        held_ok=sum(int(labels[x["source_face_a_teacher_only"]]!=
                        labels[x["source_face_b_teacher_only"]]) for x in heldout)
        # Training-only renderer evidence is separate from geometric
        # proposal constraints; view4 is never used for the winner.
        complexity=0.2*(count-1)/(face_count-1)
        loss=1.-train_ok/len(train)+complexity
        scores.append({
            "candidate":c["candidate"],
            "slot_count":count,
            "training_view_render_edit_expressible":train_ok,
            "training_view_render_edit_count":len(train),
            "heldout_view_render_edit_expressible":held_ok,
            "heldout_view_render_edit_count":len(heldout),
            "train_only_proxy_loss":loss,
            "geometry_heuristic_not_ground_truth":True,
        })
    winner=min(scores,key=lambda x:(x["train_only_proxy_loss"],
                                    x["slot_count"],x["candidate"]))
    base=proposal["old_source_atomic_group_count"]
    touched={x["source_face_a_teacher_only"] for x in rows} | {
        x["source_face_b_teacher_only"] for x in rows}
    return {
        "schema":SCHEMA,
        "status":"SOURCE_GEOMETRY_CONSTRAINED_EDIT_PROXY_ONLY__FULL_CAA_REPLAY_REQUIRED",
        "selected_candidate_train_only":winner["candidate"],
        "selected_group_count":winner["slot_count"],
        "baseline_group_count":base,
        "baseline_group_count_less_than_selected":base<winner["slot_count"],
        "heldout_expressibility_diagnostic_only":(
            winner["heldout_view_render_edit_expressible"]/
            winner["heldout_view_render_edit_count"]),
        "geometry_source_ray_constraint_count":len(constraints),
        "source_face_count":face_count,
        "independent_render_edit_ray_count":len(rows),
        "spatial_edit_evidence_undersampled":len(rows)<max(
            8,math.ceil(face_count/200)),
        "source_locus_spatial_coverage_fraction":len(touched)/face_count,
        "candidate_scores":scores,
        "geometry_occlusion_is_operational_heuristic_only":True,
        "heldout_view_training_leakage":False,
        "real_full_frame_render_edit_qualified":False,
        "author_drawable_truth_qualified":False,
        "graphics_owner_family_authorized":False,
        "tessa_teacher_authorized":False,
        "training_authorized":False,
    }
