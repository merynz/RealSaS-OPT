#!/usr/bin/env python3
"""Classify EXISTING source evidence; do not rerender, repair or mint truth.

SHA-bound join of previously validated Master1024 imported-scene observations
and independent quality triage. Output is a typed source evidence ledger and
six head-specific evidence ROUTES, explicitly not supervision admissions.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

SOURCE_SCHEMA="RealSaS.Corpus4008Parallel1024ImportedSceneMasterLedger.v1"
TRIAGE_SCHEMA="RealSaS.Corpus4008SourceCourtReviewTriage.v1"
SCHEMA="RealSaS.Corpus4008ExistingTruthClassification.v1"
HEADS=("IRIS_GEOMETRY","IRIS_SEMANTIC","IRIS_GRAPHICS","TESSA","AXIS","MIRA")
REQUIRED_COURT={
    "IRIS_GEOMETRY":"SOURCE_GEOMETRY_AND_NORMALIZED_FRAME_COURT",
    "IRIS_SEMANTIC":"SOURCE_BOUND_SEMANTIC_REGIONS_COURT",
    "IRIS_GRAPHICS":"SOURCE_BOUND_2D_GAME_ART_RGBA_DRAWABLE_COURT",
    "TESSA":"SOURCE_TOPOLOGY_CORRESPONDENCE_COURT",
    "AXIS":"JOINT_CONTROL_MOTION_BEHAVIOR_COURT",
    "MIRA":"BIND_SKIN_INFLUENCE_DEFORM_PARITY_COURT",
}


def require(ok,code):
    if not ok:
        raise ValueError("C4008_EXISTING_TRUTH_"+code)


def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda:f.read(4<<20),b""):
            h.update(block)
    return h.hexdigest()


def read_csv(p):
    with p.open(encoding="utf-8",newline="") as f:
        return list(csv.DictReader(f))


def count(row,k):
    try:
        n=int(row[k])
        require(n>=0,"NEGATIVE_COUNTER_"+k)
        return n
    except (ValueError,TypeError,KeyError):
        raise ValueError("C4008_EXISTING_TRUTH_COUNTER_"+k)


def evidence(src,qa):
    n=lambda name:count(src,name)
    geom=n("vertex_count") and n("mesh_object_count")
    skeleton=n("armature_count") and n("observed_bone_count")
    skin=n("skin_rows_sampled")
    motion=n("observed_action_count") and n("observed_action_curve_count")
    ik=n("observed_ik_constraint_count")
    material=n("material_count")
    img=n("image_count")
    e={
        "original_bytes":"SHA_VERIFIED_PRIOR_TO_IMPORT",
        "rights":"UNKNOWN_NOT_REVALIDATED",
        "character_vs_prop":"UNKNOWN",
        "morphology":"UNKNOWN",
        "anatomical_frame":"UNKNOWN",
        "geometry":"IMPORTED_GEOMETRY_OBSERVATION" if geom else "NOT_OBSERVED_IN_IMPORT",
        "topology":"IMPORTED_INDEX_INCIDENCE_OBSERVATION_NOT_SOLID_TRUTH" if geom else "NOT_OBSERVED_IN_IMPORT",
        "joint_forest":"IMPORTED_JOINT_FOREST" if skeleton else "NOT_OBSERVED_IN_IMPORT_SOURCE_UNKNOWN",
        "skin_influence":"IMPORTED_SPARSE_WEIGHT_HINT" if skin else "NOT_OBSERVED_IN_IMPORT_SOURCE_UNKNOWN",
        "bind_deform_parity":"UNKNOWN",
        "motion":"IMPORTED_ACTION_CURVE_INVENTORY" if motion else "NOT_OBSERVED_IN_IMPORT_SOURCE_UNKNOWN",
        "motion_semantics":"UNKNOWN",
        "native_IK":"IMPORTED_IK_CONSTRAINT_UNVALIDATED" if ik else "NOT_OBSERVED_IN_IMPORT_SOURCE_UNKNOWN",
        "attachment":"UNKNOWN",
        "source_material":"IMPORTED_MATERIAL_SLOT" if material else "NOT_OBSERVED_IN_IMPORT_SOURCE_UNKNOWN",
        "source_image":"IMPORTED_IMAGE_REFERENCES" if img else "NOT_OBSERVED_IN_IMPORT_SOURCE_UNKNOWN",
        "native_rgba_parity":"UNKNOWN",
        "graphics_2d_game_art":"UNREVIEWED_2D_STYLIZED_ART_REQUIRED",
        "semantic_regions":"UNKNOWN",
        "drawable_ownership":"UNKNOWN",
    }
    flags=[]
    if n("mesh_boundary_edges") or n("mesh_nonmanifold_edges"):
        flags.append("REVIEW_MODULAR_OR_OPEN_MESH_NOT_AUTO_REJECT")
    if n("skin_rows_bad_sum"):
        flags.append("REVIEW_IMPORTED_WEIGHT_SUM")
    if n("skin_rows_zero_weights"):
        flags.append("REVIEW_UNWEIGHTED_VERTICES")
    if n("image_dependency_missing_count"):
        flags.append("RELINK_EXTERNAL_TEXTURES")
    if not motion:
        flags.append("MOTION_NOT_OBSERVED_SOURCE_UNKNOWN")
    if not ik:
        flags.append("IK_NOT_OBSERVED_SOURCE_UNKNOWN_NOT_NEGATIVE_LABEL")
    flags.extend(("REVALIDATE_ORIGINAL_TRAINING_RIGHTS",
                  "CLASSIFY_CHARACTER_AND_MORPHOLOGY",
                  "COURT_SOURCE_BOUND_2D_ART_BEFORE_GRAPHICS"))
    return e,flags


def classify(source_summary,source_matrix,triage_summary,triage_matrix,out,
             *,expected_count=1024):
    summary=json.loads(source_summary.read_text())
    triage=json.loads(triage_summary.read_text())
    require(summary.get("schema")==SOURCE_SCHEMA
            and summary.get("unique_source_candidates_measured")==expected_count
            and summary.get("training_authorized") is False
            and summary.get("typed_source_matrix_sha256")==sha(source_matrix),
            "SOURCE_LINEAGE_SHA")
    require(triage.get("schema")==TRIAGE_SCHEMA
            and triage.get("source_count")==expected_count
            and triage.get("training_authorized") is False
            and triage.get("triage_matrix_sha256")==sha(triage_matrix)
            and triage.get("source_readback_summary_sha256")==sha(source_summary)
            and triage.get("source_readback_matrix_sha256")==sha(source_matrix),
            "QA_LINEAGE_SHA")
    sources=read_csv(source_matrix)
    qa_rows=read_csv(triage_matrix)
    require(len(sources)==len(qa_rows)==expected_count,"SOURCE_COUNTS")
    qa={r["canonical_asset_id"]:r for r in qa_rows}
    require(len(qa)==len(qa_rows),"DUPLICATE_QA_ID")
    source_ids=set()
    source_shas=set()
    rows=[]
    routes=[]
    stats=Counter()
    for s in sources:
        aid=s["canonical_asset_id"]
        h=s["source_sha256"]
        require(aid not in source_ids and h not in source_shas,
                "DUPLICATE_SOURCE_IDENTITY")
        source_ids.add(aid)
        source_shas.add(h)
        require(aid in qa,"MISSING_QA_ROW")
        q=qa[aid]
        require(
            s["split"]=="FIT"
            and s["source_import_observation_status"]=="IMPORTED_SOURCE_OBSERVATIONS_ONLY"
            and s["source_byte_verified_at_import"]=="True"
            and s["candidate_id"].endswith(h)
            and s["head_admission_count"]=="0"
            and s["training_authorized"]=="False"
            and q["original_source_sha256"]==h
            and q["source_import_status"]==s["source_import_observation_status"]
            and q["training_authorized"]=="False"
            and q["materialization_authorized"]=="False",
            "SOURCE_SPLIT_OR_AUTHORITY")
        for field in (
            "source_character_vs_prop","morphology",
            "original_license_commercial_ml_rights","functional_native_ik_fk",
            "source_skin_bind_deformation","source_semantic_regions",
            "graphics_source_bound_2d_art_qualified",
            "tessa_ready_topology_truth","axis_typed_controllable_rig_program_truth",
            "mira_weights_truth"):
            require(s[field]=="UNKNOWN","SOURCE_UNKNOWN_PROMOTED_"+field)
        # Ensure QA did not silently mutate/import different source observations.
        for a,b in (
            ("mesh_object_count","source_observed_mesh_objects"),
            ("observed_bone_count","source_observed_bones"),
            ("observed_action_count","source_observed_actions"),
            ("observed_ik_constraint_count","source_observed_IK_constraints"),
            ("skin_rows_sampled","source_observed_skin_rows"),
            ("skin_rows_bad_sum","source_observed_skin_bad_sum"),
            ("skin_rows_zero_weights","source_observed_skin_no_weights"),
            ("mesh_boundary_edges","source_observed_boundary_edges"),
            ("mesh_nonmanifold_edges","source_observed_nonmanifold_edges"),
            ("image_dependency_missing_count",
             "source_observed_unresolved_external_images")):
            require(count(s,a)==count(q,b),"SOURCE_QA_DRIFT_"+a)
        e,flags=evidence(s,q)
        for k,v in e.items():stats[k+"__"+v]+=1
        for f in flags:stats["flag__"+f]+=1
        stats["review_bucket__"+q["source_review_queue"]]+=1
        row={
            "canonical_asset_id":aid,"candidate_id":s["candidate_id"],
            "source_sha256":h,"source_shard":s["source_shard"],
            "split":s["split"],"source_format":s["source_format"],
            "quality_review_bucket":q["source_review_queue"],
            **{"evidence__"+k:v for k,v in e.items()},
            "review_flags":";".join(flags),
            "observed_bones":s["observed_bone_count"],
            "observed_skin_rows":s["skin_rows_sampled"],
            "observed_motion_actions":s["observed_action_count"],
            "head_supervision_admitted":"0",
            "materialization_authorized":"False",
            "training_authorized":"False",
        }
        rows.append(row)
        for head in HEADS:
            observed={
                "IRIS_GEOMETRY":("geometry","topology"),
                "IRIS_SEMANTIC":("semantic_regions",),
                "IRIS_GRAPHICS":("native_rgba_parity","graphics_2d_game_art"),
                "TESSA":("topology",),
                "AXIS":("joint_forest","motion","native_IK"),
                "MIRA":("skin_influence","bind_deform_parity"),
            }[head]
            routes.append({
                "canonical_asset_id":aid,
                "source_sha256":h,
                "head":head,
                "observed_import_evidence":";".join(
                    k for k in observed if e[k].startswith("IMPORTED_")),
                "required_unfinished_court":REQUIRED_COURT[head],
                "source_truth":"NOT_ADMITTED",
                "observation_is_not_source_truth":True,
                "training_authorized":False,
            })
    require(len(source_ids)==len(qa)==expected_count,"JOIN_UNIQUENESS")
    out.mkdir(parents=True,exist_ok=True)
    def csv_write(p,data):
        with p.open("w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,list(data[0]))
            w.writeheader()
            w.writerows(sorted(data,key=lambda x:
                (x["canonical_asset_id"],x.get("head",""))))
    matrix=out/"SOURCE_EXISTING_EVIDENCE_CLASSIFICATION_V1.csv"
    head_matrix=out/"SIX_HEAD_EXISTING_EVIDENCE_ROUTES_V1.csv"
    csv_write(matrix,rows)
    csv_write(head_matrix,routes)
    result={
        "schema":SCHEMA,"source_count":len(rows),"head_route_count":len(routes),
        "source_summary_sha256":sha(source_summary),
        "source_matrix_sha256":sha(source_matrix),
        "qa_summary_sha256":sha(triage_summary),
        "qa_matrix_sha256":sha(triage_matrix),
        "source_evidence_output_sha256":sha(matrix),
        "head_route_output_sha256":sha(head_matrix),
        "observation_and_cleanup_counts":dict(sorted(stats.items())),
        "training_admitted_by_head":{k:0 for k in HEADS},
        "source_truth_synthesized":False,
        "new_mechanical_experiment_executed":False,
        "graphics_stylized_2d_courted":False,
        "materialization_authorized":False,
        "training_authorized":False,
    }
    (out/"EXISTING_EVIDENCE_CLASSIFICATION_SUMMARY_V1.json").write_text(
        json.dumps(result,indent=2,sort_keys=True)+"\n")
    return result


def main():
    p=argparse.ArgumentParser()
    for field in ("source-summary","source-matrix","triage-summary",
                  "triage-matrix","output"):
        p.add_argument("--"+field,type=Path,required=True)
    a=p.parse_args()
    result=classify(a.source_summary,a.source_matrix,
                    a.triage_summary,a.triage_matrix,a.output)
    print(json.dumps({"sources":result["source_count"],
                      "head_routes":result["head_route_count"],
                      "training_admitted":0},sort_keys=True))


if __name__=="__main__":
    main()
