#!/usr/bin/env python3
"""Fail-closed, morphology-agnostic *teacher diagnostic* for two source loci.

A causal pixel contribution is NOT same-drawable, cut, attachment, occlusion
owner or mechanical owner ground truth. Do not promote these measurements to
IRIS/TESSA training targets without independent ownership/response courts.
No semantic/material/object names are used for inference or class labels.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

SCHEMA = "RealSaS.Corpus370GenericLocusCounterfactual.v1"
COURT_SCHEMA = "RealSaS.Corpus370GenericLocusCounterfactualCourt.v1"
MODES = {"PREMULTIPLIED_LINEAR_RGB", "STRAIGHT_LINEAR_RGB"}
PARITY_TOL = 0.025
EFFECT_TOL = 0.025


def require(ok, code):
    if not ok:
        raise ValueError(code)


def _sha(s):
    return isinstance(s, str) and len(s) == 64 and all(c in "0123456789abcdef" for c in s)


def _rgba(value, mode):
    require(mode in MODES, "LOCUS_RGB_CARRIER_UNKNOWN")
    require(isinstance(value, (list, tuple)) and len(value) == 4,
            "LOCUS_RGBA_SHAPE")
    require(all(type(x) in (float, int) and math.isfinite(x) and 0 <= x <= 1
                for x in value), "LOCUS_RGBA_RANGE_OR_NONFINITE")
    r, g, b, a = (float(x) for x in value)
    return (r, g, b, a) if mode == "PREMULTIPLIED_LINEAR_RGB" else (
        r * a, g * a, b * a, a)


def _distance(a, b, mode):
    return max(abs(x-y) for x,y in zip(_rgba(a,mode), _rgba(b,mode)))


def _address(value):
    require(isinstance(value, dict), "LOCUS_SOURCE_ADDRESS_REQUIRED")
    f = value.get("face_index_teacher_only")
    w = value.get("barycentric_teacher_only")
    require(type(f) is int and f >= 0, "LOCUS_FACE_INVALID")
    require(isinstance(w,(list,tuple)) and len(w)==3
            and all(type(x) in (float,int) and math.isfinite(x)
                    and -1e-6 <= x <= 1+1e-6 for x in w)
            and abs(sum(w)-1) <= 1e-5, "LOCUS_BARYCENTRIC_INVALID")
    return f,tuple(float(x) for x in w)


def evaluate_row(row):
    require(isinstance(row,dict) and row.get("schema")==SCHEMA,
            "LOCUS_SCHEMA")
    for key in ("source_sha256","camera_sha256","rest_geometry_sha256"):
        require(_sha(row.get(key)), "LOCUS_SOURCE_SHA_REQUIRED:"+key)
    for key in ("training_authorized","owner_identity_minted","teacher_labels_to_student",
                "semantic_or_material_name_as_label"):
        require(row.get(key) is False, "LOCUS_FORGED_AUTHORITY_OR_LABEL:"+key)
    require(row.get("camera_ray_complete") is True
            and row.get("depth_tie") is False, "LOCUS_INCOMPLETE_OR_TIED_RAY")
    require(type(row.get("view_index")) is int and 0 <= row["view_index"] < 8
            and all(type(row.get(k)) is int and row[k]>=0
                    for k in ("pixel_x","pixel_y")), "LOCUS_PIXEL_INVALID")
    require(_address(row.get("locus_a")) != _address(row.get("locus_b")),
            "LOCUS_SELF_COMPARISON")
    require(row.get("interventions_preserve_geometry_camera_pose_alpha") is True,
            "LOCUS_COUNTERFACTUAL_UNCONTROLLED")
    mode=row.get("carrier_mode")
    require(mode in MODES,"LOCUS_RGB_CARRIER_UNKNOWN")
    rgba=row.get("rgba")
    expected={"full","noop","remove_a","remove_b","remove_both",
              "recolor_a","recolor_a_noop"}
    require(isinstance(rgba,dict) and set(rgba)==expected,
            "LOCUS_MISSING_COUNTERFACTUAL_ARM")
    for v in rgba.values():
        _rgba(v,mode)
    require(_distance(rgba["full"],rgba["noop"],mode) <= PARITY_TOL
            and _distance(rgba["full"],rgba["recolor_a_noop"],mode) <= PARITY_TOL,
            "LOCUS_NOOP_PARITY_FAILED")
    require(abs(rgba["full"][3]-rgba["recolor_a"][3]) <= PARITY_TOL,
            "LOCUS_RECOLOR_ALPHA_CHANGED")
    a_full=_distance(rgba["full"],rgba["remove_a"],mode)
    b_full=_distance(rgba["full"],rgba["remove_b"],mode)
    a_cond=_distance(rgba["remove_b"],rgba["remove_both"],mode)
    b_cond=_distance(rgba["remove_a"],rgba["remove_both"],mode)
    a_observed=max(a_full,a_cond)>EFFECT_TOL
    b_observed=max(b_full,b_cond)>EFFECT_TOL
    status=("TWO_LOCI_OBSERVABLE__OWNERSHIP_UNKNOWN" if a_observed and b_observed
            else "ONE_LOCUS_OBSERVABLE__OTHER_ABSTAIN" if a_observed or b_observed
            else "BOTH_LOCI_ABSTAIN")
    return {
        "schema":COURT_SCHEMA,"status":status,
        "source_sha256":row["source_sha256"],"camera_sha256":row["camera_sha256"],
        "rest_geometry_sha256":row["rest_geometry_sha256"],
        "view_index":row["view_index"],"pixel_x":row["pixel_x"],
        "pixel_y":row["pixel_y"],
        "measured_a_causal":a_observed,"measured_b_causal":b_observed,
        "effects":{"a_full":a_full,"b_full":b_full,
                   "a_conditioned_on_b_removal":a_cond,
                   "b_conditioned_on_a_removal":b_cond,
                   "a_recolor":_distance(rgba["full"],rgba["recolor_a"],mode),
                   "b_revealed_by_removing_a":b_cond>EFFECT_TOL and b_full<=EFFECT_TOL},
        "same_drawable":"UNKNOWN","cut":"UNKNOWN",
        "attachment":"UNKNOWN","mechanical_owner":"UNKNOWN",
        "graphics_owner":"UNKNOWN",
        "student_teacher_ids_exposed":False,
        "family_authority_minted":False,"training_authorized":False,
    }


def court_manifest(doc):
    require(isinstance(doc,dict) and
            doc.get("schema")=="RealSaS.Corpus370GenericLocusCounterfactualBatch.v1"
            and doc.get("training_authorized") is False,
            "LOCUS_BATCH_SCHEMA_OR_AUTHORITY")
    rows=doc.get("rows")
    require(isinstance(rows,list) and bool(rows),"LOCUS_EMPTY_BATCH")
    results=[evaluate_row(row) for row in rows]
    keys={(r["source_sha256"],r["camera_sha256"],r["view_index"],
           r["pixel_x"],r["pixel_y"]) for r in results}
    require(len(keys)==len(results),"LOCUS_DUPLICATE_PIXEL_ROW")
    return {
        "schema":"RealSaS.Corpus370GenericLocusCounterfactualBatchCourt.v1",
        "status":"CAUSAL_SOURCE_LOCI_ONLY__GENERIC_OWNER_NOT_CERTIFIED",
        "rows":results,
        "two_causal_loci_count":sum(x["measured_a_causal"] and x["measured_b_causal"]
                                   for x in results),
        "single_causal_locus_count":sum(x["measured_a_causal"] != x["measured_b_causal"]
                                      for x in results),
        "abstain_count":sum(not(x["measured_a_causal"] or x["measured_b_causal"])
                            for x in results),
        "family_authority_minted":False,"training_authorized":False,
    }


def main():
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--input",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    args=p.parse_args()
    result=court_manifest(json.loads(args.input.read_text()))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k!="rows"},sort_keys=True))


if __name__=="__main__":
    main()
