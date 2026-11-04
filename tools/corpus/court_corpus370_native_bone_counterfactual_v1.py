#!/usr/bin/env python3
"""Independent numeric court for actual source-authored rig-control experiments.

Zero parity, source-rest face-order equivalence, both interventions, inverse
symmetry and common-root sham are read from NPZ, not believed from producer
summary. The distinct command effects DO NOT qualify editable drawable owner,
response stiffness, independently replayed mechanics, or training.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from typing import Any
import numpy as np

SCHEMA="RealSaS.Corpus370NativeBoneInterventionIndependentCourt.v1"
PRODUCER_SCHEMA="RealSaS.Corpus370NativeBoneCounterfactual.v1"
STATES=("ZERO_CONTROL","ZERO_RESET","A_POS","A_NEG",
        "B_POS","B_NEG","AB_POS","COMMON_ROOT_SHAM")


def require(ok:bool,code:str)->None:
    if not ok:raise ValueError(code)


def sha(path:Path)->str:
    with path.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def measure(rest,faces,actual,source_rest,source_faces,*,root_sham_available):
    r=np.asarray(rest,dtype=np.float64)
    f=np.asarray(faces)
    p=np.asarray(actual,dtype=np.float64)
    sr=np.asarray(source_rest,dtype=np.float64)
    sf=np.asarray(source_faces)
    require(r.ndim==2 and r.shape[1]==3 and len(r)>0
            and f.ndim==2 and f.shape[1]==3 and len(f)>0
            and f.dtype.kind in "iu" and (f>=0).all() and (f<len(r)).all()
            and np.isfinite(r).all(),"INTERVENTION_REST_INVALID")
    require(sr.shape==r.shape and sf.shape==f.shape
            and np.array_equal(sf,f) and np.isfinite(sr).all(),
            "INTERVENTION_SOURCE_FACE_ORDER_DRIFT")
    scale=float(np.linalg.norm(np.ptp(r,axis=0)))
    require(np.isfinite(scale) and scale>1e-9,"INTERVENTION_SOURCE_SCALE")
    require(float(np.max(np.abs(sr-r)))<=scale*1e-6,
            "INTERVENTION_INDEPENDENT_REST_GEOMETRY_DRIFT")
    require(p.shape==(len(STATES),len(r),3) and np.isfinite(p).all(),
            "INTERVENTION_POSED_ARRAY_INVALID")
    zero=p[0]
    reset_error=float(np.max(np.linalg.norm(p[1]-zero,axis=1)))/scale
    baseline_error=float(np.max(np.linalg.norm(zero-r,axis=1)))/scale
    require(baseline_error<=1e-6 and reset_error<=1e-7,
            "INTERVENTION_ZERO_PARITY_FAIL")
    arms={}
    for i,state in enumerate(STATES):
        delta=p[i]-zero
        max_norm=float(np.linalg.norm(delta,axis=1).max())/scale
        active=int(np.sum(np.linalg.norm(delta,axis=1)>scale*1e-6))
        arms[state]={"max_displacement_over_scale":max_norm,
                     "affected_source_vertices":active,
                     "any_measured_effect":max_norm>1e-6}
    inverse={}
    for name,up,down in (("A","A_POS","A_NEG"),("B","B_POS","B_NEG")):
        pos=(p[STATES.index(up)]-zero).reshape(-1)
        neg=(p[STATES.index(down)]-zero).reshape(-1)
        n0=float(np.linalg.norm(pos));n1=float(np.linalg.norm(neg))
        if min(n0,n1)<=scale*1e-6:
            inverse[name]={"status":"ABSTAIN_INSUFFICIENT_MEASURABLE_REVERSAL",
                           "opposition_cosine":None}
        else:
            cosine=float(np.clip(np.dot(pos,neg)/(n0*n1),-1,1))
            inverse[name]={"status":("PASS_OPPOSITE_DIRECTION"
                if cosine<=-0.8 else "ABSTAIN_NON_REVERSIBLE_OR_NONLINEAR_SOURCE_CONTROL"),
                "opposition_cosine":cosine}
    command_qualified=(
        arms["A_POS"]["any_measured_effect"] and
        arms["B_POS"]["any_measured_effect"] and
        inverse["A"]["status"]=="PASS_OPPOSITE_DIRECTION" and
        inverse["B"]["status"]=="PASS_OPPOSITE_DIRECTION")
    require(type(root_sham_available) is bool,
            "INTERVENTION_SHAM_DECLARATION")
    return {
        "schema":SCHEMA,
        "status":("PASS_TWO_INDEPENDENT_COMMAND_EFFECTS_OBSERVED__IDENTITY_UNKNOWN"
                  if command_qualified else "MASKED_INSUFFICIENT_TWO_COMMAND_CONSEQUENCES"),
        "source_scale":scale,
        "rest_vs_zero_max_error_over_scale":baseline_error,
        "no_op_reset_max_error_over_scale":reset_error,
        "controls":arms,
        "inverse_controls":inverse,
        "common_root_sham":{
            "status":("OBSERVED_SOURCED_COMMON_ANCESTOR_PERTURBATION"
                      if root_sham_available else "MASKED_NO_DISTINCT_COMMON_ANCESTOR"),
            "measured_effect":(arms["COMMON_ROOT_SHAM"]["any_measured_effect"]
                               if root_sham_available else None),
        },
        "two_distinct_commands_executed":True,
        "two_measurable_control_effects":bool(command_qualified),
        "independent_blender_replay_required":True,
        "independent_render_edit_consumer_required":True,
        "same_drawable":"UNKNOWN",
        "different_drawable":"UNKNOWN",
        "mechanical_owner":"UNKNOWN",
        "response_stiffness":"UNKNOWN",
        "causal_response_family_authorized":False,
        "graphics_owner_family_authorized":False,
        "tessa_teacher_authorized":False,
        "training_authorized":False,
    }


def court(manifest:dict[str,Any], *, source_sha256:str,
          source_rest_sha256:str, npz_path:Path|None=None,
          source_rest_path:Path|None=None)->dict[str,Any]:
    require(manifest.get("schema")==PRODUCER_SCHEMA
            and manifest.get("source_sha256")==source_sha256
            and manifest.get("teacher_only") is True
            and manifest.get("control_selection_uses_render_or_motion_outcome") is False
            and manifest.get("labels_from_source_names") is False
            and manifest.get("independent_replay_courted") is False
            and manifest.get("same_drawable")=="UNKNOWN"
            and manifest.get("different_drawable")=="UNKNOWN"
            and manifest.get("mechanical_owner")=="UNKNOWN"
            and manifest.get("training_authorized") is False
            and manifest.get("causal_response_authorized") is False,
            "INTERVENTION_PRODUCER_SOURCE_OR_AUTHORITY")
    if manifest.get("status")=="MASKED_NO_TWO_WEIGHTED_PARENT_CHILD_COMMANDS":
        require("output" not in manifest
                and npz_path is None
                and source_rest_path is None,
                "INTERVENTION_FALSE_MASKED_WITH_ARRAYS")
        return {"schema":SCHEMA,
                "status":"MASKED_NO_SOURCE_RIG_CONTROL_PAIR__NOT_ZERO",
                "source_sha256":source_sha256,
                "same_drawable":"UNKNOWN",
                "different_drawable":"UNKNOWN",
                "training_authorized":False}
    require(manifest.get("status")==
            "MEASURED_ISOLATED_RIG_CONTROL_CANDIDATE__DRAWABLE_AND_CAUSAL_COURTS_SEPARATE"
            and tuple(manifest.get("control_states") or ())==STATES
            and manifest.get("pose_count")==len(STATES)
            and manifest.get("root_sham_available") in (True,False)
            and manifest.get("rest_face_triangulation_frozen") is True
            and manifest.get("source_polygon_connectivity_stable") is True
            and isinstance(manifest.get("amplitude_radians"),float)
            and 0.03<manifest["amplitude_radians"]<0.2,
            "INTERVENTION_EXACT_COMMAND_PROTOCOL")
    require(npz_path is not None and source_rest_path is not None
            and npz_path.is_file() and source_rest_path.is_file()
            and manifest.get("output",{}).get("sha256")==sha(npz_path)
            and sha(source_rest_path)==source_rest_sha256,
            "INTERVENTION_PARENT_ARRAY_BYTES_SHA")
    with np.load(npz_path,allow_pickle=False) as arrays, np.load(
        source_rest_path,allow_pickle=False) as source:
        require(set(arrays.files)=={
            "rest_vertices_source_world","faces","face_owner",
            "face_material","intervention_positions_source_world"
        } and {"rest_vertices_source_world","faces"}<=set(source.files),
            "INTERVENTION_TEACHER_ARRAY_SCOPE")
        rest=arrays["rest_vertices_source_world"];faces=arrays["faces"]
        require(arrays["face_owner"].shape==(len(faces),)
                and arrays["face_material"].shape==(len(faces),),
                "INTERVENTION_FACE_OWNERS_MATERIALS_ARE_ADDRESSES_ONLY")
        output=measure(rest,faces,arrays["intervention_positions_source_world"],
                       source["rest_vertices_source_world"],source["faces"],
                       root_sham_available=manifest["root_sham_available"])
    output["source_sha256"]=source_sha256
    output["measured_source_rest_sha256"]=source_rest_sha256
    output["intervention_array_sha256"]=sha(npz_path)
    return output


def main():
    a=argparse.ArgumentParser()
    a.add_argument("--source-sha256",required=True)
    a.add_argument("--manifest",type=Path,required=True)
    a.add_argument("--expected-manifest-sha256",required=True)
    a.add_argument("--rest-geometry",type=Path,required=True)
    a.add_argument("--rest-geometry-sha256",required=True)
    a.add_argument("--output",type=Path,required=True)
    p=a.parse_args()
    require(sha(p.manifest)==p.expected_manifest_sha256,
            "INTERVENTION_MANIFEST_SHA")
    m=json.loads(p.manifest.read_text())
    if m.get("status")=="MASKED_NO_TWO_WEIGHTED_PARENT_CHILD_COMMANDS":
        result=court(m,source_sha256=p.source_sha256,
                     source_rest_sha256=p.rest_geometry_sha256)
    else:
        npz=Path(m["output"]["path"])
        result=court(m,source_sha256=p.source_sha256,
                     source_rest_sha256=p.rest_geometry_sha256,
                     npz_path=npz,source_rest_path=p.rest_geometry)
    p.output.parent.mkdir(parents=True,exist_ok=True)
    p.output.write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:v for k,v in result.items()
                      if k not in ("controls","inverse_controls")},
                     sort_keys=True))


if __name__=="__main__":main()
