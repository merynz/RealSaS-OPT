#!/usr/bin/env python3
"""Admit independently measured SourceSurface UDF/normals into Corpus370 family matrix.

The independent 352 Stage50 component-geometry court proves source-surface UDF
and oriented normals (with row masks) for each exact V5 observation PASS. Its
component-local signed queries are NOT global solid SDF or occupancy. Its
connected components are NOT TESSA mechanical-topology teacher evidence.
No training, global holdout or product claims are granted by this adapter.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from tools.corpus.audit_corpus370_v5_head_target_readiness_v1 import (
    V5_SHA, SCHEMA as READBACK_SCHEMA, build as head_build,
    verify_actual_court_receipts,
)

SCHEMA="RealSaS.Corpus370FamilyEvidenceIndex.v1"
GEOMETRY_INDEX_SCHEMA="RealSaS.Corpus370V5ComponentGeometryIndependentSweep.v1"
CANDIDATE_COURT_SCHEMA="RealSaS.ComponentGeometryAuthorityCourt.v1"
FAMILIES=("SURFACE_GEOMETRY_UDF","SURFACE_NORMALS")


def require(ok,code):
    if not ok: raise ValueError(code)


def sha(path):
    with Path(path).open("rb") as f: return hashlib.file_digest(f,"sha256").hexdigest()


def link(path,schema):
    path=Path(path).resolve()
    require(path.is_file(),"V5_GEOMETRY_EVIDENCE_PARENT_MISSING")
    return {"schema":schema,"path":str(path),"sha256":sha(path)}


def claim(family,row,manifest_ref,court_ref):
    require(family in FAMILIES,"V5_GEOMETRY_EVIDENCE_FAMILY")
    name="SURFACE_GEOMETRY" if family=="SURFACE_GEOMETRY_UDF" else "ORIENTED_NORMALS"
    outcome=row["family_candidate_outcomes"][name]
    counts=row["counts"]
    require(outcome in ("QUALIFIED_CORE_CANDIDATE","QUALIFIED_MASKED_CANDIDATE"),
            "V5_GEOMETRY_EVIDENCE_UNQUALIFIED_FAMILY")
    require(counts["valid_face_count"]>0
            and counts["face_count"]>=counts["valid_face_count"]
            and counts["query_count"]==8192,
            "V5_GEOMETRY_EVIDENCE_NO_VALID_ROW")
    masked=outcome=="QUALIFIED_MASKED_CANDIDATE"
    return {
        "status":"PASS","reason_codes":[],
        "evidence_refs":[manifest_ref,court_ref],
        "court":{**court_ref,"status":"PASS"},
        "producer_refs":[manifest_ref],
        "metadata":{
            "source_domain":"SourceSurface",
            "family":family,
            "exact_current_court":"PASS_COMPONENT_GEOMETRY_AUTHORITY_INDEPENDENT_COURT__FINAL_ADMISSION_SEPARATE",
            "qualified_row_count":counts["valid_face_count"],
            "total_face_count":counts["face_count"],
            "row_level_validity_mask_required":masked,
            "original_source_geometry_unmodified":True,
            "global_solid_sign_proven":False,
            "tessa_mechanical_surface_teacher_proven":False,
            "target_is_component_local_sign":False,
            "training_authorized":False,
        }
    }


def build(v5_path,geometry_path,roster_path):
    v5_path=Path(v5_path).resolve();geometry_path=Path(geometry_path).resolve()
    roster_path=Path(roster_path).resolve()
    require(sha(v5_path)==V5_SHA,"V5_GEOMETRY_EVIDENCE_SOURCE_INDEX_SHA")
    v5=json.loads(v5_path.read_text())
    geom=json.loads(geometry_path.read_text())
    roster=json.loads(roster_path.read_text())
    require(roster.get("schema")=="RealSaS.Corpus370ExactSourceRoster.v1"
            and roster.get("subject_count")==370 and roster.get("training_authorized") is False,
            "V5_GEOMETRY_EVIDENCE_ROSTER")
    # Exact per-subject re-opening; any mismatch fails the WHOLE evidence index.
    verify_actual_court_receipts(geom)
    head=head_build(v5,geom,v5_sha=V5_SHA,geometry_sha=sha(geometry_path))
    require(head["independent_geometry_court_pass_count"]==352 and
            head["family_authority_minted"] is False and
            head["source_component_sign_is_global_sdf"] is False,
            "V5_GEOMETRY_EVIDENCE_FORGED_HEAD_READBACK")
    known={r["subject_id"]:r for r in roster["subjects"]}
    qualified={r["subject_id"]:r for r in v5["subjects"]}
    require(len(known)==370 and len(qualified)==352,"V5_GEOMETRY_EVIDENCE_COHORT")
    rows=[]
    for sid in sorted(qualified):
        row=geom["subject_results"][sid]
        source_sha=qualified[sid]["source_sha256"]
        require(known[sid]["source_sha256"]==source_sha
                and row["input"]["source_sha256"]==source_sha,
                "V5_GEOMETRY_EVIDENCE_SOURCE_IDENTITY:"+sid)
        manifest=link(row["manifest"]["path"],"RealSaS.ComponentGeometryAuthority.v1")
        court=link(row["court"]["path"],CANDIDATE_COURT_SCHEMA)
        require(manifest["sha256"]==row["manifest"]["sha256"]
                and court["sha256"]==row["court"]["sha256"],
                "V5_GEOMETRY_EVIDENCE_COURT_PARENT_SHA:"+sid)
        rows.append({"subject_id":sid,"source_sha256":source_sha,
                     "families":{family:claim(family,row,manifest,court)
                                 for family in FAMILIES}})
    return {
        "schema":SCHEMA,"status":"CURRENT_352_INDEPENDENT_SOURCE_GEOMETRY_FAMILY_EVIDENCE__FINAL_MATRIX_SPLIT_AND_MODEL_ADMISSION_SEPARATE",
        "subject_count":len(rows),"subjects":rows,
        "parents":{"v5_index":link(v5_path,v5["schema"]),
                   "component_index":link(geometry_path,GEOMETRY_INDEX_SCHEMA),
                   "exact_source_roster":link(roster_path,roster["schema"])},
        "rollup":{"SURFACE_GEOMETRY_UDF":{"PASS":352},"SURFACE_NORMALS":{"PASS":352}},
        "authority":{"current_independent_source_geometry_courts_passed":True,
                     "global_solid_sdf_minted":False,
                     "mechanical_topology_teacher_minted":False,
                     "model_training_admission_minted":False},
        "training_authorized":False,
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--v5-index",type=Path,required=True)
    p.add_argument("--component-index",type=Path,required=True)
    p.add_argument("--exact-source-roster",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    result=build(a.v5_index,a.component_index,a.exact_source_roster)
    a.output.resolve().parent.mkdir(parents=True,exist_ok=True)
    a.output.resolve().write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps({"status":result["status"],"subject_count":result["subject_count"],
                      "rollup":result["rollup"],"training_authorized":False},sort_keys=True),flush=True)


if __name__=="__main__":main()
