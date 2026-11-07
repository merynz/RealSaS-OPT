#!/usr/bin/env python3
"""Reconcile 1024 source-dense observations against immutable Master3993.

For successful source numerical studies, independently replay the SHA-bound
source-world bounding boxes to IRIS and TESSA, confirm every identity and
non-admission flag. The original source bytes were hash-checked during import;
this tool does NOT re-download or independently inspect original source truth.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from tools.corpus.build_corpus4008_normalization_plan_v1 import (
    plan as replay
)
from tools.corpus.run_corpus4008_dense_normalization_shard_v1 import SCHEMA as BATCH
from tools.corpus.plan_corpus4008_dense_normalization_shard_v1 import SCHEMA as PLAN

MASTER_SHA="af2436d2a25a6f715e2d81af14b731837c7b02b609f1a4fc206acb591beb61c9"
SCHEMA="RealSaS.Corpus4008DenseSourceNormalizationGlobalLedger.v1"
SUCCESS="IMPORTED_FULL_VERTEX_BONE_WEIGHT_NUMERIC_CONSISTENCY_ONLY"


def require(x,why):
    if not x:raise ValueError("C4008_DENSE_LEDGER_"+why)


def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(4<<20),b""):
            h.update(b)
    return h.hexdigest()


def read(path):
    require(path.is_file(),"FILE_ABSENT_"+path.name)
    return json.loads(path.read_text())


def reconcile(master_path:Path,shards_root:Path,out:Path,*,
              expected_master_sha=MASTER_SHA,expected_master_count=3993):
    require(sha(master_path)==expected_master_sha,"MASTER_SHA")
    master=read(master_path)
    require(len(master)==expected_master_count,"MASTER_COUNT")
    seen_ids=set()
    seen_shas=set()
    rows=[]
    counts=Counter()
    feat=Counter()
    max_v=0.
    max_b=0.
    max_o=0.
    max_bbox=0.
    total_vertex=0
    total_bones=0
    total_weights=0
    for shard in range(32):
        base=shards_root/f"shard-{shard}"
        ppath=base/"PLAN.json"
        plan=read(ppath)
        batch=read(base/"BATCH_SUMMARY.json")
        require(plan.get("schema")==PLAN and plan.get("shard")==shard
                and plan.get("source_count")==len(plan["source_candidates"])==32
                and plan.get("training_authorized") is False,
                "ORIGINAL_DENSE_SOURCE_PLAN")
        require(batch.get("schema")==BATCH
                and batch.get("shard")==shard
                and batch.get("plan_sha256")==sha(ppath)
                and batch.get("training_authorized") is False
                and batch.get("materialization_authorized") is False
                and batch.get("source_count")==32,
                "DENSE_BATCH_CERTIFICATE")
        statusfile=base/"RESULTS.jsonl"
        require(statusfile.is_file(),"DENSE_RESULTS_MISSING")
        statusrows=[json.loads(x) for x in statusfile.read_text().splitlines()
                    if x.strip()]
        require(len(statusrows)==32
                and len({x["canonical_asset_id"] for x in statusrows})==32,
                "DENSE_BATCH_RESULTS")
        lookup={x["canonical_asset_id"]:x for x in statusrows}
        require(set(lookup)=={x["canonical_asset_id"]
                              for x in plan["source_candidates"]},
                "DENSE_RESULTS_CANDIDATE_SET")
        this_status=Counter(x["status"] for x in statusrows)
        require(dict(sorted(this_status.items()))==batch["status_counts"],
                "DENSE_BATCH_STATUS_RECOUNT")
        for e in plan["source_candidates"]:
            aid=e["canonical_asset_id"]
            original=e["original_source_sha256"]
            require(aid in master and
                    master[aid]["candidate_id"]==e["candidate_id"] and
                    master[aid]["split"]=="FIT" and
                    e["split"]=="FIT" and
                    e["source_recovery_shard"]==shard,
                    "MASTER_ORIGINAL_SPLIT_LINEAGE")
            require(aid not in seen_ids and original not in seen_shas,
                    "SOURCE_DOUBLE_COUNT")
            seen_ids.add(aid);seen_shas.add(original)
            status=lookup[aid]
            require(status["original_source_sha256"]==original
                    and status["source_shard"]==shard
                    and status["training_authorized"] is False
                    and status["materialization_authorized"] is False,
                    "DENSE_STATUS_PROVENANCE")
            state=status["status"]
            counts[state]+=1
            numeric=None
            bounds_sha=None
            plan_sha=None
            dense_sha=None
            if state==SUCCESS:
                root=base/"receipts"/aid
                bounds_path=Path(str(root)+".BOUNDS.json")
                dense_path=Path(str(root)+".DENSE.json")
                normalized_path=Path(str(root)+".NORMALIZATION.json")
                b=read(bounds_path)
                d=read(dense_path)
                n=read(normalized_path)
                require(b.get("source_sha256")==original
                        and b.get("canonical_asset_id")==aid
                        and b.get("training_authorized") is False
                        and b.get("independent_source_truth_courted") is False,
                        "SOURCE_BOUNDS_IDENTITY")
                require(d.get("schema")==
                        "RealSaS.Corpus4008ImportedDenseNormalizationConsistency.v1"
                        and d.get("original_source_sha256")==original
                        and d.get("canonical_asset_id")==aid
                        and d.get("training_authorized") is False
                        and d.get("materialization_authorized") is False
                        and d.get("independent_source_truth_courted") is False
                        and d.get("skin_bind_to_deformation_consequence_courted") is False
                        and d.get("graphics_stylized_2d_art_courted") is False,
                        "DENSE_NOT_SOURCE_TRUTH")
                require(n==replay(bounds_path,expected_sha256=sha(bounds_path)),
                        "DENSE_NORMALIZATION_REPLAY_DRIFT")
                require(d["geometry_imported_vertex_count"]==
                        b["mesh_vertex_count"]
                        and d["imported_mesh_object_count"]==
                        b["mesh_object_count"],
                        "DENSE_VERTEX_COUNT_DRIFT")
                v=float(d["source_vertex_roundtrip_max_relative_error"])
                bw=float(d["source_bone_rest_world_roundtrip_max_relative_error"])
                ow=float(d["source_object_world_matrix_roundtrip_max_relative_error"])
                bb=float(n["max_source_bbox_corner_relative_error"])
                require(all(math.isfinite(x) and 0<=x<=1e-6
                            for x in (v,bw,ow,bb)),
                        "NUMERIC_TRANSFORM_CONSISTENCY_GATE")
                require(d["mesh_source_index_not_reordered"] is True
                        and d["sparse_weights_not_mutated"] is True
                        and len(d["source_sparse_weights_sha256"])==64,
                        "SPARSE_OR_SOURCE_INDEX_DRIFT")
                max_v=max(max_v,v)
                max_b=max(max_b,bw)
                max_o=max(max_o,ow)
                max_bbox=max(max_bbox,bb)
                numeric=d
                bounds_sha=sha(bounds_path)
                plan_sha=sha(normalized_path)
                dense_sha=sha(dense_path)
                total_vertex+=d["geometry_imported_vertex_count"]
                total_bones+=d["imported_bone_rest_matrix_count"]
                total_weights+=d["imported_sparse_influence_total_entries"]
                if d["imported_skin_candidate_vertices_bad_sum_1e_minus3"]>0:
                    feat["SOURCE_SKIN_GROUP_WEIGHT_SUM_ANOMALY"]+=1
                if d["imported_skin_candidate_vertices_without_groups"]>0:
                    feat["SOURCE_SKIN_CANDIDATE_HAS_UNWEIGHTED_VERTICES"]+=1
                if d["imported_negative_handed_mesh_objects"]>0:
                    feat["SOURCE_IMPORT_HAS_NEGATIVE_DETERMINANT"]+=1
                if d["imported_bone_rest_matrix_count"]>0:
                    feat["SOURCE_IMPORTED_BONES_MEASURED"]+=1
                feat["SOURCE_IMPORTED_DENSE_NUMERIC_CONSISTENCY"]+=1
            rows.append({
                "canonical_asset_id":aid,
                "original_sha256":original,
                "candidate_id":e["candidate_id"],
                "source_shard":shard,
                "status":state,
                "source_format":e["original_source_extension"],
                "imported_vertices":numeric["geometry_imported_vertex_count"] if numeric else None,
                "imported_bone_rest_matrices":numeric["imported_bone_rest_matrix_count"] if numeric else None,
                "sparse_influence_entries":numeric["imported_sparse_influence_total_entries"] if numeric else None,
                "skin_group_bad_sum_vertices":numeric["imported_skin_candidate_vertices_bad_sum_1e_minus3"] if numeric else None,
                "skin_candidate_unweighted_vertices":numeric["imported_skin_candidate_vertices_without_groups"] if numeric else None,
                "negative_determinant_meshes":numeric["imported_negative_handed_mesh_objects"] if numeric else None,
                "source_vertex_roundtrip_relative_error":numeric["source_vertex_roundtrip_max_relative_error"] if numeric else None,
                "source_bone_rest_roundtrip_relative_error":numeric["source_bone_rest_world_roundtrip_max_relative_error"] if numeric else None,
                "source_sparse_weight_digest":numeric["source_sparse_weights_sha256"] if numeric else None,
                "bounds_receipt_sha256":bounds_sha,
                "normalized_plan_sha256":plan_sha,
                "dense_receipt_sha256":dense_sha,
                "source_authoring_pose":"UNKNOWN",
                "original_native_control_truth":"UNKNOWN",
                "source_skin_deformation_consequence":"UNKNOWN",
                "graphics_2d_art":"UNKNOWN",
                "semantic_source_face_support":"UNKNOWN",
                "original_source_license":"UNKNOWN",
                "training_authorized":False,
            })
    require(len(rows)==len(seen_ids)==len(seen_shas)==1024,
            "GLOBAL1024_SOURCE_UNIQUENESS")
    out.mkdir(parents=True,exist_ok=True)
    matrix=out/"CORPUS4008_DENSE1024_NORMALIZATION_MATRIX_V1.csv"
    with matrix.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,list(rows[0]))
        w.writeheader()
        w.writerows(sorted(rows,key=lambda x:x["canonical_asset_id"]))
    summary={
        "schema":SCHEMA,
        "master_selection_sha256":sha(master_path),
        "source_count":len(rows),
        "imported_source_math_success_count":counts[SUCCESS],
        "source_failure_reasons":{k:v for k,v in sorted(counts.items()) if k!=SUCCESS},
        "imported_vertex_count_total":total_vertex,
        "imported_bone_rest_matrix_count_total":total_bones,
        "imported_sparse_influence_entries_total":total_weights,
        "source_risk_feature_counts":dict(sorted(feat.items())),
        "max_imported_vertex_coordinate_roundtrip_relative_error":max_v,
        "max_imported_bone_world_rest_roundtrip_relative_error":max_b,
        "max_imported_object_world_matrix_roundtrip_relative_error":max_o,
        "max_source_bbox_roundtrip_relative_error":max_bbox,
        "matrix_sha256":sha(matrix),
        "source_original_bytes_rehashed_in_this_stage":False,
        "native_source_rig_control_behavior_courted":False,
        "source_skin_deformation_consequence_courted":False,
        "source_semantics_courted":False,
        "graphics_2d_art_courted":False,
        "per_head_training_admitted_count":0,
        "materialization_authorized":False,
        "training_authorized":False,
    }
    (out/"CORPUS4008_DENSE1024_NORMALIZATION_SUMMARY_V1.json").write_text(
        json.dumps(summary,indent=2,sort_keys=True)+"\n")
    return summary


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--master",type=Path,required=True)
    p.add_argument("--shards",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    report=reconcile(a.master,a.shards,a.output)
    print(json.dumps({
        "source_count":report["source_count"],
        "numerical_consistency_success":report["imported_source_math_success_count"],
        "failure_reasons":report["source_failure_reasons"],
        "materialization_authorized":False,
    },sort_keys=True))


if __name__=="__main__":
    main()
