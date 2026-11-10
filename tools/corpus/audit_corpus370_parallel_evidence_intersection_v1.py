#!/usr/bin/env python3
"""Live 370-subject evidence INTERSECTION and missing-head worklist, not training.

Do not infer unmeasured families from otherwise successful geometry/pose/ray
metrics. This audit deliberately distinguishes a measured NEGATIVE from absent
measurement (UNKNOWN), and a source-hit visibility negative from a
different-drawable negative. No trainer-ready roster or numeric tensor is minted.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

SCHEMA = "RealSaS.Corpus370ParallelEvidenceIntersection.v1"
SHA = {
    "v5":"8ab2ebf526bbf91d78117580b68ed3784e53bcbafdbd326f012e143c924c0ac4",
    "roster":"925dac90ecbf949375aa3ef5d2ed2f40788ae7dcf5669a02b708ada8d9435355",
    "rest":"384c92940e1ea8eec7bbb98cf319eeda5f89222c07b782b607998e2067f1d9c8",
    "rays":"0eeb24089322d0802567ba9736ada1de03e234bbf133bc91164132a80bec3c70",
    "native":"5aa3c4d9cc97ab669d2a7a951a4c9d18218ef5ea4d6f58f5b1ff1ac5b89cc272",
    "joint":"e72f3380c2f34cbf3e088218ebfcd4be0f4aa71c9138d4b3d1865a4f4074439a",
}


def require(ok,reason):
    if not ok:
        raise ValueError(reason)


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def build(v5,roster,rest,rays,native,joint):
    require(v5.get("schema")=="RealSaS.Corpus370PreparedObservationPackets.v5"
            and (v5.get("subject_count"),v5.get("prepared_subject_count"),
                 v5.get("failure_count"))==(370,352,18)
            and v5.get("training_authorized") is False,
            "PARALLEL_INTERSECTION_V5_UNIVERSE")
    require(roster.get("schema")=="RealSaS.Corpus370ExactSourceRoster.v1"
            and roster.get("subject_count")==370
            and roster.get("training_authorized") is False,
            "PARALLEL_INTERSECTION_ROSTER")
    require(rest.get("summary",{}).get("independent_source_rest_pass")==352
            and rest.get("training_authorized") is False,
            "PARALLEL_INTERSECTION_REST_COURTS")
    require(rays.get("schema")=="RealSaS.Corpus370V5GeometricRayPilotIndex.v1"
            and rays.get("status")=="PILOT_SUBJECTS_ACCOUNTED__GRAPHICS_AUTHORITY_SEPARATE"
            and rays.get("training_authorized") is False
            and rays.get("graphics_owner_truth_minted") is False,
            "PARALLEL_INTERSECTION_SOURCE_RAYS")
    require(native.get("schema")=="RealSaS.Corpus370V5AuthoredTemporalTeacherPilot.v1"
            and native.get("status")=="PILOT_ALL_SELECTED_ACCOUNTED__CAUSAL_AND_MODEL_ADMISSION_SEPARATE"
            and native.get("training_authorized") is False,
            "PARALLEL_INTERSECTION_NATIVE")
    require(joint.get("schema")=="RealSaS.Corpus370HostedNativeVisibilityPairJoin.v1"
            and joint.get("status")=="ALL_THREE_NATIVE_VISIBILITY_PAIRS_ACCOUNTED__OWNER_AUTHORITY_SEPARATE"
            and joint.get("training_authorized") is False
            and joint.get("graphic_owner_identity_minted") is False
            and joint.get("tessa_teacher_authorized") is False
            and joint.get("independently_courted_causal_controls")==0
            and joint.get("same_drawable_positives")==0
            and joint.get("different_drawable_negatives")==0
            and joint.get("failure_count")==0,
            "PARALLEL_INTERSECTION_NO_FALSE_OWNER_CLAIMS")
    require(joint.get("parents",{}).get("native_teacher_summary_sha256")==SHA["native"]
            and joint.get("parents",{}).get("source_rest_index_sha256")==SHA["rest"]
            and joint.get("parents",{}).get("v5_sha256")==SHA["v5"]
            and joint.get("parents",{}).get("roster_sha256")==SHA["roster"],
            "PARALLEL_INTERSECTION_SOURCE_PARENT_BINDING")
    observations={r["subject_id"]:r for r in v5["subjects"]}
    excluded={r["subject_id"]:r for r in v5["failures"]}
    rows={r["subject_id"]:r for r in roster["subjects"]}
    rests=rest.get("subjects") or {}
    ray_rows=rays.get("subject_results") or {}
    natives={r["subject_id"]:r for r in native["results"]}
    joined=joint["subjects"]
    require(len(rows)==370 and len(observations)==352 and len(excluded)==18
            and set(rows)==set(observations)|set(excluded)
            and not set(observations)&set(excluded)
            and set(rests)==set(ray_rows)==set(observations)
            and len(natives)==len(joined)==3
            and set(natives)==set(joined)
            and set(native.get("selected_subject_ids") or [])==set(natives),
            "PARALLEL_INTERSECTION_EXACT_370_352_3_CENSUS")
    result=[]
    counts=Counter()
    for sid in sorted(rows):
        row=rows[sid]
        if sid in excluded:
            require(excluded[sid]["source_sha256"]==row["source_sha256"],
                    "PARALLEL_INTERSECTION_EXCLUDED_SOURCE_SHA")
            result.append({
                "subject_id":sid,
                "source_sha256":row["source_sha256"],
                "observation":"FAIL_OBSERVATION_NOT_QUALIFIED",
                "source_rest":"NOT_ADMITTED",
                "geometric_rays":"NOT_ADMITTED",
                "native_pose":"UNKNOWN",
                "visibility_court":"UNKNOWN",
                "paired_kinematics":"UNKNOWN",
                "drawable_owner":"UNKNOWN",
                "causal_response":"UNKNOWN",
                "tessa_teacher":"UNKNOWN",
                "family_materialization":"NOT_AUTHORIZED",
                "reason_codes":["OBSERVATION_FAILED__REPAIR_REQUIRED"],
            })
            counts["observation_fail"]+=1
            continue
        require(observations[sid]["source_sha256"]==row["source_sha256"]
                and rests[sid]["input"]["source_sha256"]==row["source_sha256"]
                and rests[sid]["status"]=="PASS_SOURCE_REST_INDEPENDENT_COURT"
                and ray_rows[sid]["source_sha256"]==row["source_sha256"]
                and ray_rows[sid]["status"]=="MEASURED_GEOMETRIC_RAY_CANDIDATE"
                and ray_rows[sid]["training_authorized"] is False,
                "PARALLEL_INTERSECTION_INDEPENDENT_REST_RAY_SOURCE_DRIFT")
        pose="UNKNOWN_NOT_SAMPLED_IN_PILOT"
        vis="UNKNOWN_NOT_BOUND_PER_SUBJECT"
        pair="UNKNOWN_NOT_SAMPLED_IN_PILOT"
        if sid in natives:
            n=natives[sid]
            require(n["source_sha256"]==row["source_sha256"]
                    and n["status"]=="PASS_SOURCE_AUTHORED_TEMPORAL_GEOMETRY_CANDIDATE__CAUSAL_RESPONSE_SEPARATE"
                    and n["training_authorized"] is False,
                    "PARALLEL_INTERSECTION_NATIVE_SOURCE_DRIFT")
            require(joined[sid]["source_sha256"]==row["source_sha256"]
                    and joined[sid]["status"]=="PASS_PRIVATE_SOURCE_REST_AUTHORED_AND_VISIBILITY_JOIN__OWNER_UNKNOWN"
                    and joined[sid]["training_authorized"] is False
                    and joined[sid]["kinematic_pair_count"]==2,
                    "PARALLEL_INTERSECTION_CANONICAL_JOIN_DRIFT")
            pose="SOURCE_AUTHORED_POSE_OBSERVED__NO_CAUSAL_CONTROL"
            vis="SOURCE_HIT_VISIBILITY_REPLAY_ONLY__NO_DRAWABLE_ID"
            pair="SOURCE_BARYCENTRIC_MOTION_PAIR_OBSERVED__NO_OWNER_ID"
            counts["native_join_subjects"]+=1
            counts["native_join_pixels"]+=joined[sid]["kinematic_pair_count"]
        result.append({
            "subject_id":sid,"source_sha256":row["source_sha256"],
            "observation":"PASS_OBSERVATION_CANDIDATE",
            "source_rest":"PASS_EXACT_INDEPENDENT_REST_GEOMETRY__FAMILY_SEPARATE",
            "geometric_rays":"PASS_SOURCE_GEOMETRY_RAY_CANDIDATES__RENDER_SEPARATE",
            "native_pose":pose,"visibility_court":vis,
            "paired_kinematics":pair,
            "drawable_owner":"UNKNOWN",
            "causal_response":"UNKNOWN",
            "tessa_teacher":"UNKNOWN",
            "family_materialization":"NOT_AUTHORIZED",
            "reason_codes":[
                "NEEDS_INDEPENDENT_FAMILY_AUTHORITY_MATRIX",
                "NEEDS_LINEAGE_SAFE_GLOBAL_HOLDOUT",
                "OWNER_POSITIVE_AND_NEGATIVE_NOT_PROVEN",
                "CAUSAL_RESPONSE_NOT_PROVEN",
                "TESSA_TEACHER_NOT_COURTED",
            ],
        })
        counts["observation_pass"]+=1
        counts["independent_source_rest_pass"]+=1
        counts["source_geometric_rays_measured"]+=1
    require(len(result)==370 and counts["native_join_subjects"]==3
            and counts["native_join_pixels"]==6,
            "PARALLEL_INTERSECTION_ACCOUNTING")
    return {
        "schema":SCHEMA,
        "status":"FULL_370_KNOWN_EVIDENCE_INTERSECTION__FAMILY_ADMISSION_INCOMPLETE",
        "subject_count":370,
        "parents":{k:v for k,v in SHA.items()},
        "measured_candidate_counts":dict(sorted(counts.items())),
        "unproven_family_status":{
            "DRAWABLE_OWNERSHIP":"UNKNOWN",
            "CAUSAL_RESPONSE":"UNKNOWN",
            "TESSA_TEACHER":"UNKNOWN",
            "GLOBAL_SOLID_SDF":"UNKNOWN_OR_SUBJECT_MASKED_UNTIL_EXPLICIT_SOLID_COURT",
            "GRAPHICS_RELATIONS":"UNKNOWN_NOT_COURTED_AT_FAMILY_LEVEL",
            "APPEARANCE_CAA":"UNKNOWN_NOT_COURTED_AT_FAMILY_LEVEL",
        },
        "preregistered_graphics_expansion":"PARALLEL_8_FRESH_GEOMETRIC_CANDIDATES",
        "preregistered_native_expansion":"PARALLEL_8_AUTHORED_SOURCE_CANDIDATES",
        "lineage_safe_split":"NOT_SEALED",
        "materialization_authorized":False,
        "training_authorized":False,
        "subjects":result,
    }


def main():
    p=argparse.ArgumentParser()
    for key in SHA:
        p.add_argument("--"+key.replace("_","-"),type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    sources={}
    for key,expected in SHA.items():
        path=getattr(a,key)
        require(path.is_file() and digest(path)==expected,
                "PARALLEL_INTERSECTION_IMMUTABLE_INPUT_SHA:"+key)
        sources[key]=json.loads(path.read_text())
    report=build(**sources)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
    print(json.dumps({
        "status":report["status"],
        "subject_count":report["subject_count"],
        "measured_candidate_counts":report["measured_candidate_counts"],
        "unproven_family_status":report["unproven_family_status"],
        "training_authorized":False,
    },sort_keys=True))


if __name__=="__main__":
    main()
