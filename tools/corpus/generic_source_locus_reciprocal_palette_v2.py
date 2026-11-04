#!/usr/bin/env python3
"""Independent, morphology-agnostic V2 causal visibility court.

A painter can edit source faces arbitrarily. This court measures which exact
source locus contributes to a pixel under paired full/removal/recolor controls;
it CANNOT infer a stable graphics drawable, a cut or a mechanical owner.
No Blender/bpy, material/object/part/semantic name, model-head or source
privileged ID is used as a student label. All training authority stays FALSE.

V2 is a *new evidence version* and does not reclassify or modify sealed V1
independent court receipts.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from tools.corpus.generic_source_locus_counterfactual_v1 import _address, _rgba, require

SCHEMA="RealSaS.Corpus370GenericLocusReciprocalPalette.v2"
COURT="RealSaS.Corpus370GenericLocusReciprocalPaletteCourt.v2"
DOCUMENT_COURT="RealSaS.Corpus370GenericLocusReciprocalDocumentCourt.v2"
ARM_NAMES={
    "full","noop","remove_a","remove_b","remove_both",
    "recolor_a","recolor_b",
    "recolor_a_given_remove_b","recolor_b_given_remove_a",
    "recolor_a_noop","recolor_b_noop",
}
TOL=0.025

def distance(a,b,mode):
    return max(abs(x-y) for x,y in zip(_rgba(a,mode),_rgba(b,mode)))

def _observations(x,threshold=TOL):
    return [name for name,v in x.items() if v>threshold]

def evaluate(row):
    require(isinstance(row,dict) and row.get("schema")==SCHEMA,"RECIPROCAL_V2_SCHEMA")
    for key in ("source_sha256","camera_sha256","rest_geometry_sha256"):
        s=row.get(key)
        require(isinstance(s,str) and len(s)==64 and set(s)<set("0123456789abcdef"),
                "RECIPROCAL_V2_SOURCE_SHA:"+key)
    for key in ("training_authorized","owner_identity_minted",
                "teacher_labels_to_student","semantic_or_material_name_as_label"):
        require(row.get(key) is False,"RECIPROCAL_V2_AUTHORITY_LEAK:"+key)
    require(row.get("camera_ray_complete") is True and row.get("depth_tie") is False,
            "RECIPROCAL_V2_INVALID_RAY")
    require(row.get("interventions_preserve_geometry_camera_pose_alpha") is True,
            "RECIPROCAL_V2_UNCONTROLLED_INTERVENTION")
    require(type(row.get("view_index")) is int and 0<=row["view_index"]<8
            and all(type(row.get(k)) is int and row[k]>=0 for k in ("pixel_x","pixel_y")),
            "RECIPROCAL_V2_PIXEL")
    require(_address(row.get("locus_a"))!=_address(row.get("locus_b")),
            "RECIPROCAL_V2_SAME_FACE_LOCUS")
    mode=row.get("carrier_mode")
    rgba=row.get("rgba")
    require(isinstance(rgba,dict) and set(rgba)==ARM_NAMES,
            "RECIPROCAL_V2_MISSING_OR_EXTRA_ARM")
    for v in rgba.values():
        _rgba(v,mode)
    for arm in ("noop","recolor_a_noop","recolor_b_noop"):
        require(distance(rgba["full"],rgba[arm],mode)<=TOL,
                "RECIPROCAL_V2_NOOP_SOURCE_DRIFT:"+arm)
    # Blender may use straight or premultiplied EXR. Recolor means same alpha,
    # not same RGB. In a conditional arm its baseline is "remove-other".
    for new,baseline in (
        ("recolor_a","full"),("recolor_b","full"),
        ("recolor_a_given_remove_b","remove_b"),
        ("recolor_b_given_remove_a","remove_a"),
    ):
        require(abs(float(rgba[new][3])-float(rgba[baseline][3]))<=TOL,
                "RECIPROCAL_V2_RECOLOR_CHANGED_ALPHA:"+new)
    a={
        "remove_direct":distance(rgba["full"],rgba["remove_a"],mode),
        "recolor_direct":distance(rgba["full"],rgba["recolor_a"],mode),
        "remove_given_b_absent":distance(rgba["remove_b"],rgba["remove_both"],mode),
        "recolor_given_b_absent":distance(rgba["remove_b"],rgba["recolor_a_given_remove_b"],mode),
    }
    b={
        "remove_direct":distance(rgba["full"],rgba["remove_b"],mode),
        "recolor_direct":distance(rgba["full"],rgba["recolor_b"],mode),
        "remove_given_a_absent":distance(rgba["remove_a"],rgba["remove_both"],mode),
        "recolor_given_a_absent":distance(rgba["remove_a"],rgba["recolor_b_given_remove_a"],mode),
    }
    visible_a=_observations(a)
    visible_b=_observations(b)
    status=(
        "BOTH_LOCI_CAUSALLY_OBSERVED__OWNER_UNKNOWN" if visible_a and visible_b
        else "ONE_LOCUS_CAUSALLY_OBSERVED__OTHER_ABSTAINS" if visible_a or visible_b
        else "INSUFFICIENT_PIXEL_CONSEQUENCE__BOTH_ABSTAIN"
    )
    # One-sided palette differences do not prove source partition membership,
    # and color identity is not a training label. This is visibility ONLY.
    return {
        "schema":COURT,"status":status,
        "source_sha256":row["source_sha256"],
        "camera_sha256":row["camera_sha256"],
        "rest_geometry_sha256":row["rest_geometry_sha256"],
        "view_index":row["view_index"],"pixel_x":row["pixel_x"],"pixel_y":row["pixel_y"],
        "a_causal":bool(visible_a),"b_causal":bool(visible_b),
        "a_evidence_modalities":visible_a,"b_evidence_modalities":visible_b,
        "a_effects":a,"b_effects":b,
        "same_drawable":"UNKNOWN","different_drawable":"UNKNOWN",
        "cut":"UNKNOWN","attachment":"UNKNOWN",
        "mechanical_owner":"UNKNOWN","graphics_owner":"UNKNOWN",
        "visibility_owner":"SOURCE_LOCUS_ONLY",
        "teacher_ids_in_student_input":False,
        "qualified_same_owner_positives":0,"qualified_different_owner_negatives":0,
        "graphics_family_authority_minted":False,"training_authorized":False,
    }

def court_document(document):
    require(isinstance(document,dict) and
            document.get("schema")=="RealSaS.Corpus370SourceLocusPaletteIntervention.v1"
            and document.get("status")=="MEASURED_SOURCE_PALETTE_PILOT__NO_GRAPHICS_FAMILY_AUTHORITY"
            and document.get("authority",{}).get("graphics_family_minted") is False
            and document.get("authority",{}).get("training_authorized") is False,
            "RECIPROCAL_V2_RENDER_DOCUMENT_SCOPE")
    job=document.get("parents") or {}
    require(job.get("training_authorized") is False
            and job.get("schema")=="RealSaS.Corpus370SourceLocusPaletteJob.v1",
            "RECIPROCAL_V2_RENDER_PARENT_SCOPE")
    results=[]
    for row in document.get("rows") or ():
        if row.get("status")!="PASS_SOURCE_RENDER_INTERVENTION_CANDIDATE__DRAWABLE_OWNER_SEPARATE":
            results.append({"status":"ABSTAIN_UNQUALIFIED_PALETTE_V1_RENDER",
                            "view_index":row.get("view_index"),
                            "pixel_x":row.get("pixel_x"),"pixel_y":row.get("pixel_y")})
            continue
        v2=row.get("generic_counterfactual_v2")
        require(isinstance(v2,dict),"RECIPROCAL_V2_REAL_RENDER_ARM_MISSING")
        require(v2.get("source_sha256")==document.get("source_sha256")
                ==job.get("source",{}).get("sha256")
                and v2.get("camera_sha256")==job.get("camera",{}).get("sha256")
                and v2.get("rest_geometry_sha256")==job.get("geometry",{}).get("sha256")
                and v2.get("view_index")==row.get("view_index")
                and v2.get("pixel_x")==row.get("pixel_x")
                and v2.get("pixel_y")==row.get("pixel_y")
                and v2.get("locus_a",{}).get("face_index_teacher_only")
                    ==row.get("source_faces_teacher_only",[None,None])[0]
                and v2.get("locus_b",{}).get("face_index_teacher_only")
                    ==row.get("source_faces_teacher_only",[None,None])[1],
                "RECIPROCAL_V2_PARENT_SOURCE_CAMERA_LOCUS_DRIFT")
        require(distance(v2["rgba"]["full"],row["full_soup_rgba"],v2["carrier_mode"])<1e-5
                and distance(v2["rgba"]["remove_a"],row["removal_rgba"],v2["carrier_mode"])<1e-5,
                "RECIPROCAL_V2_V1_RENDER_BASELINE_DRIFT")
        results.append(evaluate(v2))
    require(len(results)>0,"RECIPROCAL_V2_EMPTY_DOCUMENT")
    strong=[r for r in results if r.get("schema")==COURT]
    two=sum(r["a_causal"] and r["b_causal"] for r in strong)
    one=sum(r["a_causal"]!=r["b_causal"] for r in strong)
    return {
        "schema":DOCUMENT_COURT,
        "status":"REAL_SOURCE_LOCUS_RECIPROCAL_EVIDENCE__DRAWABLE_OWNER_NOT_ADMITTED",
        "subject_id":document["subject_id"],
        "source_sha256":document["source_sha256"],
        "row_count":len(results),"two_locus_count":two,
        "one_locus_count":one,"abstain_count":len(results)-two-one,
        "rows":results,"qualified_same_owner_positives":0,
        "qualified_different_owner_negatives":0,
        "graphics_family_authority_minted":False,"training_authorized":False,
    }

def main():
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--source-manifest",type=Path,required=True)
    p.add_argument("--court-output",type=Path,required=True)
    args=p.parse_args()
    out=court_document(json.loads(args.source_manifest.read_text()))
    args.court_output.parent.mkdir(parents=True,exist_ok=True)
    args.court_output.write_text(json.dumps(out,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps({k:v for k,v in out.items() if k!="rows"},sort_keys=True))

if __name__=="__main__":
    main()
