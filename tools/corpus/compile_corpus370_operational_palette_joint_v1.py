#!/usr/bin/env python3
"""FIT native actuator plus independently courted original shader palette tasks.

Policy-conditioned edit algebra; neither original artist drawable owners nor
student model training labels are minted. Unsupported shader faces are UNKNOWN.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

from tools.corpus.compile_corpus370_operational_edit_algebra_v1 import (
    compile_evidence, canonical_bytes, digest, INPUT_SCHEMA)
from tools.corpus.court_corpus370_operational_edit_algebra_v1 import court as algebra_court
from tools.corpus.court_corpus370_operational_native_support_v1 import baseline_mismatch

SCHEMA="RealSaS.Corpus370JointOriginalPaletteNativeEditWorkloadPilot.v1"
POLICY="FIT_NATIVE_ACTUATOR_BOOLEAN_PLUS_ORIGINAL_SHADER_EXACT_CONSTANT_PALETTE_V1"
DENIED=("graphics_owner_authorized","iris_training_authorized",
        "tessa_training_authorized","materialization_authorized","training_authorized")

def require(value,code):
    if not value:raise ValueError(code)

def sha(path):
    with Path(path).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()

def read(path):return json.loads(Path(path).read_text(encoding="utf-8"))

def compile_joint(native,palette,court,palette_sha,original_native_sha):
    require(native.get("schema")==INPUT_SCHEMA,"OD_JOINT_NATIVE_SCHEMA")
    require(palette.get("schema")=="RealSaS.Corpus370OriginalShaderConstantPaletteFacts.v1",
            "OD_JOINT_PALETTE_SCHEMA")
    require(court.get("schema")=="RealSaS.Corpus370IndependentOriginalShaderPaletteCourt.v1"
        and court.get("status")=="INDEPENDENT_SOURCE_SHADER_ADDRESS_VERIFIED__NO_OWNER_AUTHORITY"
        and court.get("palette_producer_sha256")==palette_sha,
        "OD_JOINT_PALETTE_INDEPENDENT_COURT_NOT_BOUND")
    require(native.get("source_original_sha256")==palette.get("original_source_sha256")
        ==court.get("original_source_sha256"),"OD_JOINT_SOURCE_SHA")
    require(native.get("source_face_order_sha256")==palette.get("source_face_order_sha256")
        and native.get("source_face_count")==palette.get("source_face_count")
        ==court.get("original_face_count"),"OD_JOINT_FACE_ADDRESS")
    require(all(palette.get(x) is False for x in DENIED)
        and court.get("graphics_owner_authorized") is False
        and court.get("training_authorized") is False,
        "OD_JOINT_NONPROMOTION")
    valid=palette.get("face_palette_validity")
    owner=palette.get("face_palette_color_key_index_or_null")
    colors=palette.get("constant_palette_color_keys_hex_rgba")
    n=native["source_face_count"]
    require(isinstance(valid,list) and isinstance(owner,list)
        and isinstance(colors,list) and len(valid)==len(owner)==n
        and all(type(v) is bool for v in valid)
        and len(colors)==palette["palette_operation_count"]
        and len(set(tuple(c) for c in colors))==len(colors)
        and all((v and type(k) is int and 0<=k<len(colors)) or
                (not v and k is None) for v,k in zip(valid,owner)),
        "OD_JOINT_PALETTE_DOMAIN")
    tasks=[dict(x) for x in native["operations"]]
    for idx,color in enumerate(colors):
        color_id=digest({"rgba_hex":color})
        operation_id="PALETTE_EXACT_RGBA_"+color_id[:24]
        require(not any(t["operation_id"]==operation_id for t in tasks),
                "OD_JOINT_PALETTE_TASK_DUPLICATE")
        tasks.append({"operation_id":operation_id,
            "support":[v and k==idx for v,k in zip(valid,owner)],
            "validity":list(valid),
            "evidence_sha256":digest({
               "palette_facts_sha256":palette_sha,
               "source_native_evidence_sha256":original_native_sha,
               "color_hex_rgba":color,
               "policy":POLICY})})
    joint={k:native[k] for k in
        ("schema","source_original_sha256","source_rest_sha256",
         "source_face_order_sha256","source_face_count")}
    joint["operations"]=tasks
    joint["workload_policy"]=POLICY
    decision=compile_evidence(joint)
    algebra=algebra_court(joint,decision)
    require(algebra["status"]=="PASS_ALGEBRA_ONLY__NO_REAL_SOURCE_OWNER_AUTHORITY",
            "OD_JOINT_ALGEBRA_FAILED")
    summary={"schema":SCHEMA,"original_source_sha256":native["source_original_sha256"],
        "native_task_count":len(native["operations"]),
        "palette_task_count":len(colors),"joint_task_count":len(tasks),
        "source_face_count":n,
        "source_shader_constant_qualified_faces":sum(valid),
        "source_shader_unsupported_faces":len(valid)-sum(valid),
        "native_source_face_task_unknowns":sum(sum(not z for z in t["validity"])
                                              for t in native["operations"]),
        "palette_unsupported_is_unknown":True,
        "mechanical_native_only_source_evidence_sha256":original_native_sha,
        "source_shader_palette_fact_sha256":palette_sha,
        "source_shader_independent_court_verified_producer_sha256":court["palette_producer_sha256"],
        "joint_evidence_sha256":digest(joint),
        "joint_candidate_atom_count":len(decision["atoms"]),
        "joint_unknown_placeholder_count":decision["audit"]["unknown_face_count"],
        "joint_algebra_court_status":algebra["status"],
        "original_artist_owner":"UNKNOWN",
        "actual_blender_source_edit_court":"NOT_EVALUATED",
        **{x:False for x in DENIED}}
    return joint,decision,summary

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("native-evidence","source-palette","palette-court",
                "v18a-npz","joint-evidence","joint-decision","summary"):
        p.add_argument("--"+key,required=True,type=Path)
    a=p.parse_args()
    n,palette,court=(read(x) for x in (a.native_evidence,a.source_palette,a.palette_court))
    joint,decision,summary=compile_joint(n,palette,court,sha(a.source_palette),
                                         sha(a.native_evidence))
    with np.load(a.v18a_npz,allow_pickle=False) as v:
        require("faces" in v.files and len(v["faces"])==n["source_face_count"],
                "OD_JOINT_V18A_FACE_COUNT")
        comparisons={}
        for threshold in ("0.045","0.1","0.18"):
            key="labels_"+threshold.replace(".","p")
            require(key in v.files,"OD_JOINT_V18A_BASELINE_MISSING")
            comparison=baseline_mismatch(np.asarray(v[key],dtype=np.int64),
                                         joint["operations"])
            comparisons["V18A_THRESHOLD_"+threshold]=comparison
    summary["source_shader_independent_court_receipt_sha256"]=sha(a.palette_court)
    summary["v18a_baseline_comparison"]=comparisons
    summary["native_only_atom_count"]=len(compile_evidence(n)["atoms"])
    summary["added_palette_tasks_change_quotient"]=(
        summary["native_only_atom_count"]!=summary["joint_candidate_atom_count"])
    for path,obj in ((a.joint_evidence,joint),(a.joint_decision,decision),
                     (a.summary,summary)):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(canonical_bytes(obj))
    print(json.dumps({"status":"ORIGINAL_SHADER_NATIVE_JOINT_WORKLOAD__NO_OWNER",
        "native_tasks":summary["native_task_count"],
        "palette_tasks":summary["palette_task_count"],
        "native_atoms":summary["native_only_atom_count"],
        "joint_atoms":summary["joint_candidate_atom_count"],
        "joint_unknown_faces":summary["joint_unknown_placeholder_count"],
        "training_authorized":False},sort_keys=True))

if __name__=="__main__":main()
