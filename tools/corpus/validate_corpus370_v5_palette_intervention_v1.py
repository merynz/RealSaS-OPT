#!/usr/bin/env python3
"""Independent numeric/provenance court of rendered palette interventions.

This module intentionally does not import the Blender producer, its RGB carrier
conversion, its source mesh routines or teacher semantic heuristics. Court PASS
means bounded measured reconstruction on ray samples, NOT source drawable
ownership, mechanical ownership, or model target/training eligibility.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from tools.corpus.generic_source_locus_counterfactual_v1 import evaluate_row

SCHEMA="RealSaS.Corpus370SourceLocusPaletteIntervention.v1"
STATUS="MEASURED_SOURCE_PALETTE_PILOT__NO_GRAPHICS_FAMILY_AUTHORITY"
PASS="PASS_SOURCE_RENDER_INTERVENTION_CANDIDATE__DRAWABLE_OWNER_SEPARATE"
ABSTAINS={
    "ABSTAIN_RENDER_SOURCE_PARITY",
    "ABSTAIN_ALPHA_COMPOSITING_RECONSTRUCTION",
    "ABSTAIN_ALPHA_CHANGED_BY_RECOLOR",
    "ABSTAIN_PALETTE_NO_OBSERVABLE_RGB_DELTA",
}
TOL=0.025


def require(ok,code):
    if not ok:
        raise ValueError(code)


def file_sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def rgba(value,code):
    arr=np.asarray(value,dtype=np.float64)
    require(arr.shape==(4,) and bool(np.isfinite(arr).all())
            and -1e-5<=arr[3]<=1+1e-5,code)
    return arr


def premul(value,mode):
    a=rgba(value,"PALETTE_COURT_RGBA")
    require(mode in ("PREMULTIPLIED_LINEAR_RGB","STRAIGHT_LINEAR_RGB"),
            "PALETTE_COURT_CARRIER_MODE")
    return float(a[3]),(a[:3] if mode=="PREMULTIPLIED_LINEAR_RGB" else a[:3]*a[3])


def max_consequence(a,b):
    return float(max(abs(a[0]-b[0]),np.max(np.abs(a[1]-b[1]))))


def court(document):
    require(document.get("schema")==SCHEMA and document.get("status")==STATUS,
            "PALETTE_COURT_INPUT_STATUS")
    job=document.get("parents") or {}
    require(job.get("schema")=="RealSaS.Corpus370SourceLocusPaletteJob.v1"
            and job.get("training_authorized") is False,
            "PALETTE_COURT_JOB")
    require(document.get("source_sha256")==job.get("source",{}).get("sha256")
            and document.get("subject_id")==job.get("subject_id"),
            "PALETTE_COURT_SUBJECT_SOURCE_ID")
    a=document.get("authority") or {}
    for key in ("student_semantic_owner_labels_minted","graphics_family_minted","training_authorized"):
        require(a.get(key) is False,"PALETTE_COURT_FORGED_AUTHORITY:"+key)
    require(a.get("material_and_owner_names_teacher_only") is True
            and a.get("source_material_alpha_unchanged") is True,
            "PALETTE_COURT_PRIVILEGED_BOUNDARY")
    require(document.get("renderer")=="BLENDER_CYCLES_EXR_PREMULTIPLIED_CENTRE_RAY",
            "PALETTE_COURT_RENDERER")
    rows=document.get("rows")
    require(isinstance(rows,list) and 1<=len(rows)<=8,
            "PALETTE_COURT_ROWS")
    # The exporter calibrates its framebuffer carrier independently; v1
    # declares the known conversion in the payload and the court rebuilds it.
    mode=document.get("carrier_mode")
    require(mode in ("PREMULTIPLIED_LINEAR_RGB","STRAIGHT_LINEAR_RGB"),
            "PALETTE_COURT_CARRIER_MISSING")
    passed=0;abstained=0;paired=0
    generic_evidence=[]
    for row in rows:
        require(row.get("status") in {PASS,*ABSTAINS},"PALETTE_COURT_ROW_STATUS")
        require(row.get("material_semantic_name_used_as_label") is False
                and row.get("same_drawable_label_minted") is False,
                "PALETTE_COURT_OWNER_LEAKAGE")
        require(type(row.get("view_index")) is int and 0<=row["view_index"]<=7
                and type(row.get("pixel_x")) is int and type(row.get("pixel_y")) is int,
                "PALETTE_COURT_VIEW_PIXEL")
        faces=row.get("source_faces_teacher_only")
        bary=row.get("source_barycentric_teacher_only")
        require(isinstance(faces,list) and len(faces)==2
                and all(type(f) is int and f>=0 for f in faces)
                and isinstance(bary,list) and len(bary)==2,
                "PALETTE_COURT_SOURCE_ADDRESS")
        for xyz in bary:
            xyz=np.asarray(xyz,dtype=np.float64)
            require(xyz.shape==(3,) and np.isfinite(xyz).all()
                    and np.all(xyz>=-1e-6) and abs(float(xyz.sum())-1)<1e-5,
                    "PALETTE_COURT_BARYCENTRIC")
        original=premul(row["original_rgba"],mode)
        full=premul(row["full_soup_rgba"],mode)
        source_err=max_consequence(original,full)
        require(abs(float(row["source_reimport_vs_full_soup_max_error"])-source_err)<1e-6,
                "PALETTE_COURT_FULL_PARITY_METRIC_DRIFT")
        if row["status"]=="ABSTAIN_RENDER_SOURCE_PARITY":
            require(source_err>TOL-1e-6,"PALETTE_COURT_PARITY_ABSTAIN_WITHOUT_ERROR")
            abstained+=1
            continue
        require(source_err<=TOL,"PALETTE_COURT_FALSE_FULL_PARITY")
        standalone=premul(row["standalone_rgba"],mode)
        removal=premul(row["removal_rgba"],mode)
        measured=(standalone[0]+(1-standalone[0])*removal[0],
                  standalone[1]+(1-standalone[0])*removal[1])
        rec_error=max_consequence(full,measured)
        require(abs(float(row["front_face_reconstruction_error"])-rec_error)<1e-6,
                "PALETTE_COURT_RECONSTRUCTION_METRIC_DRIFT")
        if row["status"]=="ABSTAIN_ALPHA_COMPOSITING_RECONSTRUCTION":
            require(rec_error>TOL-1e-6,
                    "PALETTE_COURT_FALSE_RECONSTRUCTION_ABSTAIN")
            abstained+=1
            continue
        require(rec_error<=TOL,"PALETTE_COURT_RECONSTRUCTION_NOT_BOUNDED")
        split=premul(row["palette_distinct_rgba"],mode)
        collision=premul(row["palette_collision_rgba"],mode)
        max_alpha=max(abs(split[0]-full[0]),abs(collision[0]-full[0]))
        require(abs(float(row["palette_alpha_max_delta"])-max_alpha)<1e-6,
                "PALETTE_COURT_ALPHA_METRIC_DRIFT")
        delta=float(np.linalg.norm(split[1]-collision[1]))
        require(abs(float(row["palette_rgb_difference_norm"])-delta)<1e-6,
                "PALETTE_COURT_RGB_METRIC_DRIFT")
        require(row.get("split_and_collision_are_color_only") is True,
                "PALETTE_COURT_SPLIT_PALETTE_CONTROL")
        # An independent pure-Python court verifies the additional
        # reciprocal source-locus counterfactual controls. A SOURCE face
        # address remains privileged; any result must keep ownership UNKNOWN.
        generic=row.get("generic_counterfactual")
        if generic is not None:
            require(generic.get("source_sha256")==document.get("source_sha256")
                    and generic.get("camera_sha256")==job.get("camera",{}).get("sha256")
                    and generic.get("rest_geometry_sha256")==job.get("geometry",{}).get("sha256")
                    and generic.get("view_index")==row["view_index"]
                    and generic.get("pixel_x")==row["pixel_x"]
                    and generic.get("pixel_y")==row["pixel_y"]
                    and generic.get("locus_a",{}).get("face_index_teacher_only")==faces[0]
                    and generic.get("locus_b",{}).get("face_index_teacher_only")==faces[1],
                    "PALETTE_COURT_GENERIC_LOCUS_PARENT_DRIFT")
            measured=evaluate_row(generic)
            require(measured.get("same_drawable")=="UNKNOWN"
                    and measured.get("graphics_owner")=="UNKNOWN"
                    and measured.get("training_authorized") is False,
                    "PALETTE_COURT_GENERIC_OWNER_FORGERY")
            generic_evidence.append(measured)
        if row["status"]=="ABSTAIN_ALPHA_CHANGED_BY_RECOLOR":
            require(max_alpha>TOL-1e-6,"PALETTE_COURT_FALSE_ALPHA_ABSTAIN")
            abstained+=1
            continue
        require(max_alpha<=TOL,"PALETTE_COURT_RECOLOR_DID_NOT_PRESERVE_ALPHA")
        if row["status"]=="ABSTAIN_PALETTE_NO_OBSERVABLE_RGB_DELTA":
            require(delta<=0.005+1e-6,"PALETTE_COURT_FALSE_RGB_ABSTAIN")
            abstained+=1
            continue
        require(delta>0.005,"PALETTE_COURT_INVISIBLE_PALETTE_FALSE_PASS")
        passed+=1
        paired+=int(bool(row.get("different_source_domains")))
    require(passed==document.get("qualified_intervention_rows")
            and abstained==document.get("abstain_intervention_rows"),
            "PALETTE_COURT_ROLLUP_DRIFT")
    return {
        "schema":"RealSaS.Corpus370SourceLocusPaletteIndependentCourt.v1",
        "status":"PASS_SOURCE_PALETTE_INTERVENTION_REPLAY_COURT__GRAPHICS_MODEL_ADMISSION_SEPARATE",
        "subject_id":document["subject_id"],
        "source_sha256":document["source_sha256"],
        "measured_valid_palette_rows":passed,
        "measured_source_domain_pair_controls":paired,
        "abstained_rows":abstained,
        "generic_counterfactual_status":(
            "MEASURED_SOURCE_LOCI_ONLY__OWNER_UNKNOWN" if generic_evidence
            else "NOT_MEASURED__GENERIC_COUNTERFACTUAL_ARMS_MISSING"),
        "generic_two_causal_loci_count":sum(
            x["measured_a_causal"] and x["measured_b_causal"]
            for x in generic_evidence),
        "generic_counterfactuals":generic_evidence,
        "student_drawable_ownership_qualified":False,
        "family_authority_minted":False,
        "training_authorized":False,
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    value=json.loads(a.manifest.read_text())
    result=court(value)
    result["parent"]={"path":str(a.manifest.resolve()),"sha256":file_sha(a.manifest)}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print(json.dumps(result,sort_keys=True),flush=True)


if __name__=="__main__":main()
