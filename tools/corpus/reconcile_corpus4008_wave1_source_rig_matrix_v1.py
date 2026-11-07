#!/usr/bin/env python3
"""Join 8 fully source-bound Wave1 Blender scene/rig receipts.

This validates SHA and model/candidate lineage and emits a per-asset typed
coverage matrix. Imported observations are NOT training authorities. Unknown
source semantics, source licenses, native animator controls and 2D art stay
UNKNOWN even when imported counts are nonzero.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

SCHEMA="RealSaS.Corpus4008Wave1SourceRigTypedMatrix.v1"
PLAN_SCHEMA="RealSaS.Corpus4008Wave1SourceImportPlan.v1"
RECEIPT_SCHEMA="RealSaS.Corpus4008Wave1SourceRigSceneObservation.v1"
BATCH_SCHEMA="RealSaS.Corpus4008Wave1ImportBatchSummary.v1"
EXPECTED_SHARDS=(0,4,8,12,16,20,24,28)
SUCCESS="IMPORTED_SOURCE_OBSERVATIONS_ONLY"
OTHER_STATUSES={"SOURCE_FETCH_ERROR","SOURCE_SHA_OR_SIZE_MISMATCH",
                "SOURCE_IMPORT_EXCEPTION"}
HEADS=("IRIS_GEOMETRY","IRIS_SEMANTIC","IRIS_GRAPHICS",
       "TESSA","AXIS","MIRA")


def require(ok,msg):
    if not ok:raise ValueError("C4008_WAVE1_MATRIX_"+msg)


def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(4<<20),b""):
            h.update(b)
    return h.hexdigest()


def json_file(path):
    require(path.is_file(),"MISSING_FILE_"+path.name)
    return json.loads(path.read_text())


def load_receipt(path):
    require(path.is_file() and path.stat().st_size>0
            and path.stat().st_size<30*1024*1024,
            "SOURCE_GZIP_RECEIPT_SIZE")
    with gzip.open(path,"rb") as f:
        raw=f.read(70*1024*1024+1)
    require(len(raw)<=70*1024*1024,"SOURCE_RECEIPT_DECOMPRESS_BUDGET")
    return json.loads(raw)


def reconcile(root:Path,out:Path,expected_shards=EXPECTED_SHARDS):
    ids=set()
    sources=set()
    rows=[]
    status_counts=Counter()
    head_observation_candidates=Counter()
    counts=Counter()
    for shard in expected_shards:
        base=root/("shard-%d"%shard)
        plan=json_file(base/"PLAN.json")
        status=json_file(base/"BATCH_STATUS.json")
        require(plan["schema"]==PLAN_SCHEMA
                and plan["shard_index"]==shard
                and plan["model_training_authorized"] is False,
                "PLAN_SCHEMA_OR_ADMISSION")
        require(status["schema"]==BATCH_SCHEMA
                and status["shard"]==shard
                and status["training_authorized"] is False
                and status["truth_admitted"]==0,
                "BATCH_AUTHORITY")
        result_file=base/"RESULTS.jsonl"
        require(result_file.is_file(),"NO_RESULTS_FILE")
        results=[json.loads(x) for x in result_file.read_text().splitlines()
                 if x.strip()]
        require(len(results)==len(plan["entries"])==plan["sample_size"]
                ==status["selected_source_count"],"BATCH_ROWS")
        lookup={r["asset_id"]:r for r in results}
        require(len(lookup)==len(results),"BATCH_DUPLICATE_ASSET")
        require(set(lookup)=={r["canonical_asset_id"] for r in plan["entries"]},
                "BATCH_SOURCE_IDENTITY")
        count=Counter(r["status"] for r in results)
        require(dict(sorted(count.items()))==status["status_counts"],
                "BATCH_STATUS_INCONSISTENT")
        for p in plan["entries"]:
            aid=p["canonical_asset_id"]
            require(aid not in ids and p["source_sha256"] not in sources,
                    "CROSS_SHARD_IDENTITY_OR_SOURCE_DUP")
            ids.add(aid);sources.add(p["source_sha256"])
            r=lookup[aid]
            require(r["source_sha256"]==p["source_sha256"]
                    and r["training_authorized"] is False
                    and r["truth_admitted"] is False,
                    "RESULT_SOURCE_OR_AUTHORITY")
            state=r["status"]
            require(state in OTHER_STATUSES|{SUCCESS},"UNKNOWN_STATUS")
            numeric={}
            if state==SUCCESS:
                summary=json_file(base/"receipts"/(aid+".summary.json"))
                compressed=base/"receipts"/(aid+".full.source.json.gz")
                require(sha(compressed)==summary["receipt_gz_sha256"],
                        "DETAILED_RECEIPT_SHA")
                detail=load_receipt(compressed)
                require(detail.get("schema")==RECEIPT_SCHEMA
                        and detail.get("status")==
                        "FULL_IMPORTED_SOURCE_SCENE_AND_CONTROL_CENSUS__NO_SOURCE_TRUTH_COURTS",
                        "DETAILED_SCHEMA")
                require(detail["source_sha256"]==p["source_sha256"]
                        and detail["asset_id"]==aid
                        and detail["candidate_id"]==p["candidate_id"]
                        and detail["source_format"]==p["source_ext"],
                        "DETAILED_LINEAGE")
                require(detail["source_character_vs_prop"]=="UNKNOWN"
                        and detail["functional_native_ik_fk"]=="UNKNOWN"
                        and detail["commercial_ml_rights_courted"] is False
                        and detail["training_authorized"] is False
                        and detail["trainable_model_family_count"]==0
                        and detail["2d_art_visual_quality_courted"] is False,
                        "FALSE_SOURCE_OR_HEAD_ADMISSION")
                numeric=detail["numeric_census"]
                require(numeric==summary["numeric_census"],
                        "SOURCE_SUMMARY_DRIFT")
                require(detail["target_family_observation_routes"]=={
                    "IRIS_GEOMETRY":"IMPORTED_MESH_AND_TOPOLOGY_OBSERVATION",
                    "IRIS_SEMANTIC":"SOURCE_GROUP_HINTS_ONLY_NOT_LABEL",
                    "IRIS_GRAPHICS":"MATERIAL_AND_TEXTURE_CENSUS_NOT_2D_ART",
                    "TESSA":"SOURCE_SURFACE_PLUS_RIG_CANDIDATE",
                    "AXIS":"IMPORTED_RIG_MOTION_CONTROL_HINTS_ONLY",
                    "MIRA":"VERTEX_GROUP_WEIGHT_CANDIDATE_NOT_VALID_BIND",
                },"ROUTING_CONTRACT")
                counts["SOURCE_IMPORT_SUCCESS"]+=1
                # Imported scene morphology doesn't establish character class.
                if numeric["mesh_object_count"]>0:
                    head_observation_candidates["IRIS_GEOMETRY"]+=1
                    head_observation_candidates["IRIS_SEMANTIC"]+=1
                    head_observation_candidates["IRIS_GRAPHICS"]+=1
                if numeric["armature_count"]>0:
                    head_observation_candidates["AXIS"]+=1
                    head_observation_candidates["TESSA"]+=1
                if (numeric["armature_count"]>0
                    and numeric["skin_rows_sampled"]>numeric["skin_rows_zero_weights"]):
                    # Merely enumerating vertices does not establish skin.
                    head_observation_candidates["MIRA"]+=1
                if numeric["observed_ik_constraint_count"]>0:
                    counts["OBSERVED_IMPORTED_IK_SOURCE_ROWS"]+=1
                if numeric["observed_action_count"]>0:
                    counts["OBSERVED_IMPORT_ACTION_ROWS"]+=1
                if numeric["image_dependency_missing_count"]>0:
                    counts["SOURCE_MISSING_EXTERNAL_IMAGES"]+=1
                if numeric["mesh_boundary_edges"]>0:
                    counts["SOURCE_NONCLOSED_MESH_ASSETS"]+=1
            status_counts[state]+=1
            row={
                "canonical_asset_id":aid,
                "candidate_id":p["candidate_id"],
                "source_sha256":p["source_sha256"],
                "source_format":p["source_ext"],
                "source_shard":shard,
                "split":"FIT",
                "source_import_observation_status":state,
                "source_byte_verified_at_import":state==SUCCESS,
            }
            fields=(
              "mesh_object_count","source_scene_node_count","armature_count",
              "vertex_count","polygon_count","mesh_boundary_edges",
              "mesh_nonmanifold_edges","mesh_disconnected_components",
              "skin_rows_sampled","skin_rows_zero_weights","skin_rows_bad_sum",
              "bone_parented_mesh_candidates",
              "image_dependency_missing_count","material_count","image_count",
              "observed_bone_count","observed_ik_constraint_count",
              "observed_driver_channel_count","observed_action_curve_count",
              "observed_action_count","observed_nla_strip_count",
              "observed_nondeform_bone_count",
            )
            for k in fields:row[k]=numeric.get(k)
            row.update({
                "anatomical_frame_camera":"UNKNOWN",
                "source_character_vs_prop":"UNKNOWN",
                "morphology":"UNKNOWN",
                "original_license_commercial_ml_rights":"UNKNOWN",
                "functional_native_ik_fk":"UNKNOWN",
                "source_skin_bind_deformation":"UNKNOWN",
                "source_semantic_regions":"UNKNOWN",
                "native_source_material_rgb_parity":"UNKNOWN",
                "graphics_style_image_target_domain":"STYLIZED_2D_GAME_ART_REQUIRED",
                "graphics_source_bound_2d_art_qualified":"UNKNOWN",
                "tessa_ready_topology_truth":"UNKNOWN",
                "axis_typed_controllable_rig_program_truth":"UNKNOWN",
                "mira_weights_truth":"UNKNOWN",
                "source_original_material_truth":"UNKNOWN",
                "head_admission_count":0,
                "training_authorized":False,
            })
            rows.append(row)
    require(len(rows)==len(ids)==len(sources),"RESULT_UNIQUE")
    require(rows,"EMPTY_IMPORT_MATRIX")
    out.mkdir(parents=True,exist_ok=True)
    with (out/"WAVE1_SOURCE_RIG_SCENE_TYPED_MATRIX_V1.csv").open(
        "w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    report={
        "schema":SCHEMA,
        "sampled_original_source_count":len(rows),
        "sampled_shards":list(expected_shards),
        "source_import_status":dict(sorted(status_counts.items())),
        "source_observation_counts":dict(sorted(counts.items())),
        "per_head_import_observation_candidates_not_qualified":{
            h:head_observation_candidates[h] for h in HEADS
        },
        "per_head_training_admitted":{h:0 for h in HEADS},
        "morphology_human_review_admitted":0,
        "native_control_ik_evaluated_courts":0,
        "graphics_2d_style_independent_courts":0,
        "training_authorized":False,
    }
    (out/"WAVE1_SOURCE_RIG_SCENE_SUMMARY_V1.json").write_text(
        json.dumps(report,sort_keys=True,indent=2)+"\n")
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(reconcile(a.root,a.output),sort_keys=True))


if __name__=="__main__":
    main()
