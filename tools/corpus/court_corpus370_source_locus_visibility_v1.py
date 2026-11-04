#!/usr/bin/env python3
"""Qualify bounded per-source-hit visibility at REAL rendered pixels.

This court independently replays BOTH existing renderer/alpha and V2
reciprocal causal courts. It can qualify source A as visible-positive and a
provably opaque-occluded B as visibility-negative under known pixel tolerance.
A *negative visibility* is NEVER a negative same-drawable/owner/attachment.

The source face+barycentric is TEACHER ADDRESS ONLY; this is a small candidate
court, NOT a V3.2 student materialization or family/training authorization.
"""
from __future__ import annotations

import json
from pathlib import Path

from tools.corpus.generic_source_locus_reciprocal_palette_v2 import (
    COURT as V2_COURT, TOL, court_document as v2_court,
)
from tools.corpus.validate_corpus370_v5_palette_intervention_v1 import (
    court as v1_court,
)

SCHEMA="RealSaS.Corpus370BoundedSourceHitVisibilityCandidateCourt.v1"
OPAQUE_EPS=1e-6

def require(ok,code):
    if not ok: raise ValueError(code)

def qualify(document):
    v1=v1_court(document)
    require(v1["training_authorized"] is False
            and v1["student_drawable_ownership_qualified"] is False,
            "VIS_COURT_V1_FAKE_AUTHORITY")
    v2=v2_court(document)
    require(v2["training_authorized"] is False and
            v2["graphics_family_authority_minted"] is False,
            "VIS_COURT_V2_FAKE_AUTHORITY")
    require(len(v2["rows"])==len(document["rows"]), "VIS_COURT_ROW_CENSUS")
    results=[]
    for original,measured in zip(document["rows"],v2["rows"]):
        if measured.get("schema")!=V2_COURT:
            results.append({
                "status":"ABSTAIN_SOURCE_RENDER_INVALID",
                "targets":[],
            })
            continue
        pixels=original.get("source_barycentric_teacher_only") or ()
        faces=original.get("source_faces_teacher_only") or ()
        require(len(pixels)==len(faces)==2, "VIS_COURT_SOURCE_ADDRESS")
        # Verified pixel replay and reconstruction come from independent V1.
        require(original.get("status")=="PASS_SOURCE_RENDER_INTERVENTION_CANDIDATE__DRAWABLE_OWNER_SEPARATE"
                and original.get("source_reimport_vs_full_soup_max_error",1)>-1
                and original["source_reimport_vs_full_soup_max_error"]<=TOL
                and original.get("front_face_reconstruction_error",1)<=TOL,
                "VIS_COURT_RENDER_REPLAY_NOT_ADMITTED")
        a=measured["a_effects"]
        b=measured["b_effects"]
        # A directly recolored in FULL. If it changes the original output,
        # A is a measured visible source at this bounded pixel.
        a_direct=a["recolor_direct"]>TOL
        # B may be physically present yet fully hidden behind A. Require
        # B's positive counterfactual effect when A is removed (sensitivity
        # control), and EXACT standalone opacity of A before a negative.
        front_alpha=float(original["standalone_rgba"][3])
        a_solid=front_alpha>=1.0-OPAQUE_EPS
        b_direct=b["recolor_direct"]>TOL
        b_after_a=b["recolor_given_a_absent"]>TOL
        # Fully opaque A at the source locus and a material-edit response
        # from B *through* it is a real renderer/order contradiction, not
        # proof of B's original visibility. Never promote such a sample.
        require(not (a_direct and a_solid and b_direct),
                "VIS_COURT_OPAQUE_FRONT_BACK_RESPONSE_CONTRADICTION")
        if b_direct:
            b_label="POSITIVE_VISIBLE_CONTRIBUTOR"
            b_basis="DIRECT_RECOLOR_FULL"
        elif a_direct and a_solid and b_after_a and b["recolor_direct"]<=TOL:
            b_label="NEGATIVE_OPAQUE_OCCLUDED_CONTRIBUTOR"
            b_basis="DIRECT_NO_EFFECT_PLUS_REVEALED_AFTER_A_REMOVAL_AND_OPAQUE_A"
        else:
            b_label="ABSTAIN_VISIBILITY_UNPROVEN"
            b_basis="INSUFFICIENT_NEGATIVE_VISIBILITY_CONTROL"
        a_label=("POSITIVE_VISIBLE_CONTRIBUTOR" if a_direct
                 else "ABSTAIN_VISIBILITY_UNPROVEN")
        row={
            "status":"MEASURED_SOURCE_HIT_VISIBILITY_CANDIDATE_ONLY",
            "view_index":measured["view_index"],
            "pixel_x":measured["pixel_x"],"pixel_y":measured["pixel_y"],
            "source_sha256":measured["source_sha256"],
            "camera_sha256":measured["camera_sha256"],
            "rest_geometry_sha256":measured["rest_geometry_sha256"],
            "targets":[
                {"source_face_teacher_only":faces[0],
                 "barycentric_teacher_only":pixels[0],
                 "visibility_label":a_label,
                 "qualification":("DIRECT_RECOLOR_FULL" if a_direct else "ABSTAIN")},
                {"source_face_teacher_only":faces[1],
                 "barycentric_teacher_only":pixels[1],
                 "visibility_label":b_label,
                 "qualification":b_basis},
            ],
            "alpha_standalone_a":front_alpha,
            "valid_positive_source_visibility":int(a_direct)+int(b_direct),
            "valid_negative_source_visibility":int(b_label=="NEGATIVE_OPAQUE_OCCLUDED_CONTRIBUTOR"),
            "same_drawable":"UNKNOWN","cut":"UNKNOWN","attachment":"UNKNOWN",
            "mechanical_owner":"UNKNOWN","graphics_owner":"UNKNOWN",
            "training_authorized":False,
        }
        results.append(row)
    positives=sum(x.get("valid_positive_source_visibility",0) for x in results)
    negatives=sum(x.get("valid_negative_source_visibility",0) for x in results)
    abstain=sum(2-len([v for v in row.get("targets",[])
                        if v["visibility_label"]!="ABSTAIN_VISIBILITY_UNPROVEN"])
                for row in results)
    return {
        "schema":SCHEMA,
        "status":"REAL_BOUNDED_SOURCE_HIT_VISIBILITY_CANDIDATES__NO_OWNER_AUTHORITY",
        "subject_id":document["subject_id"],
        "source_sha256":document["source_sha256"],
        "pixel_count":len(results),
        "qualified_visible_hit_positives":positives,
        "qualified_occluded_hit_negatives":negatives,
        "abstain_hit_count":abstain,
        "rows":results,
        "qualified_same_drawable_positives":0,
        "qualified_different_drawable_negatives":0,
        "graphics_family_authority_minted":False,
        "training_authorized":False,
    }

def main():
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--source-manifest",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    result=qualify(json.loads(a.source_manifest.read_text()))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k!="rows"},sort_keys=True))

if __name__=="__main__":main()
