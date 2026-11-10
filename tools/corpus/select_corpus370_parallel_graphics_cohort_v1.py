#!/usr/bin/env python3
"""Deterministic pre-render, geometry-only selection for an 8-subject graphics lane.

All 352 source rays were measured before this experiment. We exclude the
previously measured 3 *by prior participation*, never by observed RGB or
owner outcome, and deterministically stratify exact-source format/source-pack
candidates with at least two geometric multihit rays. Each selected candidate
is retained even if Blender subsequently FAILS or ABSTAINS. No semantic names,
teacher labels or training authority are used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

V5_SHA = "8ab2ebf526bbf91d78117580b68ed3784e53bcbafdbd326f012e143c924c0ac4"
ROSTER_SHA = "925dac90ecbf949375aa3ef5d2ed2f40788ae7dcf5669a02b708ada8d9435355"
RAY_SCHEMA = "RealSaS.Corpus370V5GeometricRayPilotIndex.v1"
SCHEMA = "RealSaS.Corpus370ParallelGraphicsGeometricSelection.v1"
EXCLUDED_PRIOR_THREE = frozenset({
    "corpus_000ca1ae51f9582fde9e",
    "corpus_004e16e05b90505ca23c",
    "corpus_00a3688e42a29045056b",
})
SALT = "REALSAS_CORPUS370_PARALLEL_GRAPHICS_RESEARCH_V1"


def require(ok,code):
    if not ok:
        raise ValueError(code)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def score(sid,source_sha):
    return hashlib.sha256((SALT+"|"+sid+"|"+source_sha).encode()).hexdigest()


def choose(v5,roster,ray_index,count=8):
    require(type(count) is int and count==8,"PARALLEL_GRAPHICS_EXACT_EIGHT")
    require(v5.get("schema")=="RealSaS.Corpus370PreparedObservationPackets.v5"
            and v5.get("prepared_subject_count")==352
            and v5.get("failure_count")==18
            and v5.get("training_authorized") is False,
            "PARALLEL_GRAPHICS_V5_COHORT")
    require(roster.get("schema")=="RealSaS.Corpus370ExactSourceRoster.v1"
            and roster.get("subject_count")==370
            and roster.get("training_authorized") is False,
            "PARALLEL_GRAPHICS_ROSTER_COHORT")
    require(ray_index.get("schema")==RAY_SCHEMA
            and ray_index.get("status")=="PILOT_SUBJECTS_ACCOUNTED__GRAPHICS_AUTHORITY_SEPARATE"
            and ray_index.get("training_authorized") is False
            and ray_index.get("graphics_owner_truth_minted") is False
            and len(ray_index.get("subject_results") or {})==352,
            "PARALLEL_GRAPHICS_RAY_SCOPE")
    rows={r["subject_id"]:r for r in v5["subjects"]}
    sources={r["subject_id"]:r for r in roster["subjects"]}
    ray_rows=ray_index["subject_results"]
    require(len(rows)==352 and len(sources)==370 and set(rows)==set(ray_rows)
            and EXCLUDED_PRIOR_THREE <= set(rows), "PARALLEL_GRAPHICS_CENSUS")
    buckets=defaultdict(list)
    accounted={"eligible_geometry":0,"excluded_previous":0,
               "insufficient_multihit":0,"invalid_ray_status":0}
    for sid in sorted(rows):
        source=sources[sid]
        ray=ray_rows[sid]
        require(source["source_sha256"]==rows[sid]["source_sha256"]
                and ray["source_sha256"]==source["source_sha256"],
                "PARALLEL_GRAPHICS_SOURCE_SHA_DRIFT")
        if sid in EXCLUDED_PRIOR_THREE:
            accounted["excluded_previous"]+=1
            continue
        if ray["status"]!="MEASURED_GEOMETRIC_RAY_CANDIDATE":
            accounted["invalid_ray_status"]+=1
            continue
        if ray.get("counts",{}).get("multihit_geometric_rays",0)<2:
            accounted["insufficient_multihit"]+=1
            continue
        # Format and source pack are diversity strata, NEVER drawable labels.
        stratum=(source.get("source_format") or "UNDECLARED",
                 source.get("source_id") or "UNDECLARED")
        buckets[stratum].append((score(sid,source["source_sha256"]),sid))
        accounted["eligible_geometry"]+=1
    require(accounted["eligible_geometry"]>=count,
            "PARALLEL_GRAPHICS_INSUFFICIENT_GEOMETRIC_CANDIDATES")
    for group in buckets.values():
        group.sort()
    selected=[]
    while len(selected)<count:
        remaining=[(v[0][0],k) for k,v in buckets.items() if v]
        require(remaining,"PARALLEL_GRAPHICS_EXHAUSTED_CANDIDATES")
        # One per stratum in first round; reproducible across machines.
        for _,key in sorted(remaining):
            if len(selected)>=count:break
            _,sid=buckets[key].pop(0)
            selected.append(sid)
    require(len(selected)==len(set(selected))==count
            and not set(selected)&EXCLUDED_PRIOR_THREE,
            "PARALLEL_GRAPHICS_DUPLICATE_OR_PRIOR_SOURCE")
    chosen = {
        "schema":RAY_SCHEMA,
        "status":"PILOT_SUBJECTS_ACCOUNTED__GRAPHICS_AUTHORITY_SEPARATE",
        "subject_ids":selected,
        "subject_results":{sid:ray_rows[sid] for sid in selected},
        "training_authorized":False,
        "graphics_owner_truth_minted":False,
        "appearance_authority_minted":False,
    }
    report={
        "schema":SCHEMA,
        "status":"DETERMINISTIC_PRE_RENDER_GEOMETRIC_SELECTION__NO_GRAPHICS_OR_TRAINING_AUTHORITY",
        "selected_subject_ids":selected,
        "geometry_candidate_census":accounted,
        "strata_considered":len(buckets),
        "excluded_prior_subject_count":len(EXCLUDED_PRIOR_THREE),
        "subject_ids_teacher_only":True,
        "selection_uses_graphics_outcome":False,
        "selection_uses_semantics_or_material_names":False,
        "same_drawable_labels_minted":False,
        "training_authorized":False,
    }
    return chosen,report


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--v5-index",type=Path,required=True)
    p.add_argument("--exact-roster",type=Path,required=True)
    p.add_argument("--all352-ray-index",type=Path,required=True)
    p.add_argument("--subset-ray-index",type=Path,required=True)
    p.add_argument("--selection-receipt",type=Path,required=True)
    a=p.parse_args()
    require(sha(a.v5_index)==V5_SHA and sha(a.exact_roster)==ROSTER_SHA,
            "PARALLEL_GRAPHICS_INPUT_BYTES_SHA")
    index_sha=sha(a.all352_ray_index)
    selected,report=choose(json.loads(a.v5_index.read_text()),
                           json.loads(a.exact_roster.read_text()),
                           json.loads(a.all352_ray_index.read_text()))
    report["parents"]={"v5_sha256":V5_SHA,"roster_sha256":ROSTER_SHA,
                       "all352_source_ray_index_sha256":index_sha}
    for path,value in ((a.subset_ray_index,selected),(a.selection_receipt,report)):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value,sort_keys=True,indent=2)+"\n")
    print(json.dumps({"status":report["status"],"subject_count":len(selected["subject_ids"]),
                      "geometry_candidate_census":report["geometry_candidate_census"],
                      "training_authorized":False},sort_keys=True))


if __name__ == "__main__":
    main()
