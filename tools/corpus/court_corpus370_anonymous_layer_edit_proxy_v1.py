#!/usr/bin/env python3
"""Independent bounded layer-toggle editability consumer (not full renderer).

A source-hit A is front visible and B is independently proven occluded and
recolorable after A removal. The consumer asks whether a proposed atomic
DRAWABLE SLOT toggle could suppress A without also suppressing B.

Candidate source-face slots are a teacher-side representation with *no
validated author identity*. A/B in different slots is necessary for this
specific atomic-slot edit, not sufficient for a full graphics/CAA release.
The evaluator has no access to semantic category names, UV material names,
source object names, rig names, or true editable author partitions.

Winner selection uses TRAIN VIEW 0 only. Opposing VIEW 4 is a untouched
diagnostic evaluation, never a hyperparameter/partition selection target.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from tools.corpus.topology_partition_court_v1 import canonicalize_partition

SCHEMA="RealSaS.Corpus370AnonymousLayerEditProxyCourt.v1"
PROPOSAL_SCHEMA="RealSaS.Corpus370AnonymousLayerProposal.v1"
PAIR_SCHEMA="RealSaS.Corpus370GenericPairKinematicDiagnosticBatch.v1"
TRAIN_VIEW=0
HELDOUT_VIEW=4
LAMBDA_COMPLEXITY=0.2


def require(ok,code):
    if not ok:raise ValueError(code)


def sha(path:Path)->str:
    with path.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def _pair_rows(doc):
    require(doc.get("schema")==PAIR_SCHEMA
            and doc.get("training_authorized") is False
            and doc.get("qualified_same_drawable_positives")==0
            and doc.get("qualified_different_drawable_negatives")==0
            and doc.get("pair_count")==len(doc.get("rows",[])),
            "ANON_EDIT_PAIR_EVIDENCE_OR_AUTHORITY")
    seen=set()
    output=[]
    for row in doc["rows"]:
        require(row.get("view_index") in (TRAIN_VIEW,HELDOUT_VIEW)
                and row.get("source_hit_visibility_owner")=="SOURCE_LOCUS_ONLY"
                and row.get("same_drawable")=="UNKNOWN"
                and row.get("different_drawable")=="UNKNOWN"
                and row.get("training_authorized") is False,
                "ANON_EDIT_SAMPLED_RAY_NOT_SOURCE_HIT")
        face_a=row.get("source_face_a_teacher_only")
        face_b=row.get("source_face_b_teacher_only")
        require(type(face_a) is int and type(face_b) is int
                and face_a!=face_b,"ANON_EDIT_SOURCE_FACES")
        key=(row["view_index"],row["pixel_x"],row["pixel_y"])
        require(key not in seen,"ANON_EDIT_DUPLICATE_PIXEL")
        seen.add(key)
        output.append((row["view_index"],face_a,face_b))
    require(any(r[0]==TRAIN_VIEW for r in output)
            and any(r[0]==HELDOUT_VIEW for r in output),
            "ANON_EDIT_TRAIN_HELDOUT_RAYS_REQUIRED")
    return output


def court(proposal,independent_pair_doc):
    require(proposal.get("schema")==PROPOSAL_SCHEMA
            and proposal.get("status")==
                "ANONYMOUS_CANDIDATE_PARTITIONS__BOUNDED_CONSUMER_REQUIRED"
            and proposal.get("train_view")==TRAIN_VIEW
            and proposal.get("heldout_view")==HELDOUT_VIEW
            and proposal.get("training_authorized") is False
            and proposal.get("graphics_owner_minted") is False
            and proposal.get("mechanical_owner_minted") is False
            and proposal.get("teacher_source_faces_exposed_to_model") is False
            and proposal.get("source_semantic_or_material_names_used") is False,
            "ANON_EDIT_PROPOSAL_SCOPE_OR_AUTHORITY")
    parents=proposal.get("parents") or {}
    require(parents.get("source_sha256")==independent_pair_doc.get("parents",{}).get("source_sha256")
            and parents.get("source_pair_receipt_sha256")
                and isinstance(parents.get("canonical_arrays_sha256"),str),
            "ANON_EDIT_SOURCE_PARENT_DRIFT")
    n=proposal.get("face_count")
    require(type(n) is int and n>1
            and type(proposal.get("authored_pose_count")) is int
            and proposal["authored_pose_count"]>=2,
            "ANON_EDIT_FACE_OR_MOTION_CENSUS")
    pair_rows=_pair_rows(independent_pair_doc)
    require(proposal.get("independently_sampled_source_ray_count")==len(pair_rows)
            and proposal.get("candidate_count")==len(proposal.get("candidates",[])),
            "ANON_EDIT_RAY_OR_CANDIDATE_CENSUS")
    names=[x.get("candidate") for x in proposal["candidates"]]
    require(len(names)==len(set(names))==7 and
            set(names)=={
                "SINGLE_LAYER","MANIFOLD_COMPONENTS","GEOMETRY_SEAMS",
                "MOTION_SEAMS","GEOMETRY_OR_MOTION_SEAMS",
                "TRAIN_VIEW_RAY_SEEDED","ATOMIC_PER_FACE",
            },"ANON_EDIT_EXACT_BASELINE_SET")
    results=[]
    for candidate in proposal["candidates"]:
        name=candidate["candidate"]
        labels=np.asarray(candidate.get("face_slot_teacher_only"))
        require(labels.shape==(n,) and labels.dtype.kind in "iu"
                and (labels>=0).all()
                and len(np.unique(labels))==candidate["slot_count"]
                and np.array_equal(canonicalize_partition(labels),labels)
                and candidate.get("source_import_appearance_retained") is True
                and candidate.get("render_heldout_replay_executed") is False
                and candidate.get("operational_drawable_authority_minted") is False,
                "ANON_EDIT_CANDIDATE_PARTITION_INTEGRITY")
        if name=="SINGLE_LAYER":
            require(candidate["slot_count"]==1,"ANON_EDIT_SINGLE_BASELINE")
        if name=="ATOMIC_PER_FACE":
            require(candidate["slot_count"]==n,"ANON_EDIT_ATOMIC_BASELINE")
        hit={TRAIN_VIEW:[0,0],HELDOUT_VIEW:[0,0]}
        for view,face_a,face_b in pair_rows:
            require(0<=face_a<n and 0<=face_b<n,
                    "ANON_EDIT_REAL_SOURCE_FACE_RANGE")
            # The independently measured render conditional effect of B
            # after removing A is a necessity witness, not author identity.
            # Atomic per-slot edit cannot remove A while preserving B if
            # both belong to the same toggled slot.
            recovered=bool(labels[face_a]!=labels[face_b])
            hit[view][0]+=int(recovered)
            hit[view][1]+=1
        train_ok,train_total=hit[TRAIN_VIEW]
        heldout_ok,heldout_total=hit[HELDOUT_VIEW]
        complexity=LAMBDA_COMPLEXITY*(candidate["slot_count"]-1)/(n-1)
        train_proxy_loss=(train_total-train_ok)/train_total+complexity
        results.append({
            "candidate":name,
            "slot_count":candidate["slot_count"],
            "train_ray_atomic_layer_reveal_hits":train_ok,
            "train_ray_count":train_total,
            "heldout_ray_atomic_layer_reveal_hits":heldout_ok,
            "heldout_ray_count":heldout_total,
            "complexity_penalty":complexity,
            "train_proxy_loss":train_proxy_loss,
            "heldout_fraction":heldout_ok/heldout_total,
            "selected_by_heldout":False,
            "full_render_edit_replay_executed":False,
            "same_drawable":"UNKNOWN",
            "different_drawable":"UNKNOWN",
            "model_training_authorized":False,
        })
    results.sort(key=lambda row:row["candidate"])
    selected=min(results,key=lambda row:(row["train_proxy_loss"],
                                         row["slot_count"],row["candidate"]))
    atomic=next(x for x in results if x["candidate"]=="ATOMIC_PER_FACE")
    require(atomic["train_ray_atomic_layer_reveal_hits"]==atomic["train_ray_count"]
            and atomic["heldout_ray_atomic_layer_reveal_hits"]==atomic["heldout_ray_count"],
            "ANON_EDIT_ATOMIC_BASELINE_CONTRADICTION")
    single=next(x for x in results if x["candidate"]=="SINGLE_LAYER")
    require(single["train_ray_atomic_layer_reveal_hits"]==0
            and single["heldout_ray_atomic_layer_reveal_hits"]==0,
            "ANON_EDIT_SINGLE_BASELINE_CONTRADICTION")
    touched_faces=len({face for _,a,b in pair_rows for face in (a,b)})
    coverage=touched_faces/n
    # Research *quality warnings*, never assertions of author boundaries.
    # A small hand-picked number of source hits is insufficient evidence
    # to promote a thousands-face graphics partition into a Studio product.
    overfragmented=(selected["slot_count"]>32
                    and selected["slot_count"]/n>0.20)
    undersampled=(len(pair_rows)<max(8,math.ceil(n/200)))
    return {
        "schema":SCHEMA,
        "status":"BOUNDED_SOURCE_HIT_ATOMIC_LAYER_EDIT_PROXY_ONLY__NOT_DRAWABLE_AUTHORITY",
        "parents":dict(parents),
        "train_view":TRAIN_VIEW,
        "heldout_view":HELDOUT_VIEW,
        "selection_by":"TRAIN_ONLY_REVEAL_SURVIVAL_PLUS_FIXED_SLOT_COMPLEXITY",
        "complexity_penalty_coefficient":LAMBDA_COMPLEXITY,
        "selected_candidate_train_only":selected["candidate"],
        "selected_slot_count":selected["slot_count"],
        "selected_heldout_reveal_fraction_diagnostic_only":selected["heldout_fraction"],
        "source_face_coverage_fraction":coverage,
        "source_faces_touched_by_independent_ray_courts":touched_faces,
        "selected_layer_to_face_ratio":selected["slot_count"]/n,
        "overfragmentation_review_required":overfragmented,
        "insufficient_spatial_edit_evidence_review_required":undersampled,
        "functional_layer_compilation_status":(
            "ABSTAIN_OVERFRAGMENTED_SELECTED_CANDIDATE"
            if overfragmented else
            "ABSTAIN_UNDERSAMPLED_EDIT_CONSEQUENCES"
            if undersampled else
            "RESEARCH_PROXY_ONLY__FULL_RENDER_PENDING"
        ),
        "candidate_scores":results,
        "full_render_edit_replay_required":True,
        "cross_view_seam_and_appearance_replay_required":True,
        "isolated_mechanical_consumer_required":True,
        "author_drawable_partition_identified":False,
        "qualified_same_drawable_positives":0,
        "qualified_different_drawable_negatives":0,
        "operational_layer_student_targets_minted":False,
        "graphics_family_authorized":False,
        "tessa_teacher_authorized":False,
        "training_authorized":False,
    }


def main():
    cli=argparse.ArgumentParser()
    cli.add_argument("--proposal",type=Path,required=True)
    cli.add_argument("--expected-proposal-sha256",required=True)
    cli.add_argument("--pair-receipt",type=Path,required=True)
    cli.add_argument("--expected-pair-sha256",required=True)
    cli.add_argument("--output",type=Path,required=True)
    a=cli.parse_args()
    require(a.proposal.is_file() and sha(a.proposal)==a.expected_proposal_sha256
            and a.pair_receipt.is_file() and sha(a.pair_receipt)==a.expected_pair_sha256,
            "ANON_EDIT_PARENT_BYTES_SHA")
    proposal=json.loads(a.proposal.read_text())
    pairs=json.loads(a.pair_receipt.read_text())
    require(proposal.get("parents",{}).get("source_pair_receipt_sha256")
            ==a.expected_pair_sha256,
            "ANON_EDIT_PAIR_SHA_BINDING")
    result=court(proposal,pairs)
    result["parents"]["proposal_sha256"]=a.expected_proposal_sha256
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,sort_keys=True,indent=2,
                                   allow_nan=False)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k!="candidate_scores"},
                     sort_keys=True),flush=True)


if __name__=="__main__":
    main()
