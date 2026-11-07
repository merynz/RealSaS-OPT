#!/usr/bin/env python3
"""Fail-closed readback/replay of 32 real source-bound normalization plans.

Checks every original source identity against immutable Master3993 and
RECOMPUTES the normalized plan from the SHA-verified imported bounds.
No full mesh, rig, skin, native source unit, anatomical or Graphics teacher
authority is granted merely by a reversible bbox transform.
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
    plan as replay_plan, SCHEMA as PLAN_SCHEMA
)

SCHEMA="RealSaS.Corpus4008ParallelNormalizationReadback.v1"
SELECT_SHA="af2436d2a25a6f715e2d81af14b731837c7b02b609f1a4fc206acb591beb61c9"
SUCCESS="SOURCE_BBOX_REVERSIBLE_PLAN_ONLY"
SELECTION_SCHEMA="RealSaS.Corpus4008ParallelNormalizationPilotSelection.v1"
BOUND_SCHEMA="RealSaS.Corpus4008MeasuredImportedSourceBounds.v1"


def require(x,code):
    if not x:raise ValueError("C4008_PARALLEL_NORMALIZE_READBACK_"+code)


def hash_file(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(4<<20),b""):
            h.update(chunk)
    return h.hexdigest()


def load(path):
    require(path.is_file(),"INPUT_MISSING_"+path.name)
    return json.loads(path.read_text())


def reconcile(root:Path,master_path:Path,out:Path,*,
              expected_master_count=3993,
              expected_master_sha=SELECT_SHA):
    require(master_path.is_file() and
            hash_file(master_path)==expected_master_sha,
            "MASTER_SELECTION_SHA")
    master=load(master_path)
    require(isinstance(master,dict) and len(master)==expected_master_count,
            "MASTER_SELECTION_COUNT")
    identities=set()
    original_shas=set()
    rows=[]
    statuses=Counter()
    evidence_sha={}
    max_err=0.
    for group in range(8):
        base=root/("group-%d"%group)
        statuses_path=base/"STATUS.jsonl"
        require(statuses_path.is_file(),"GROUP_STATUS_MISSING")
        outcome=[json.loads(x) for x in statuses_path.read_text().splitlines()
                 if x.strip()]
        require(len(outcome)==4,"GROUP_FOUR_SHARDS")
        for row in outcome:
            shard=row.get("shard")
            require(type(shard) is int and
                    group*4<=shard<group*4+4,"GROUP_SHARD_INCORRECT")
            status=row.get("status")
            require(status==SUCCESS,"SOURCE_NORMALIZATION_NOT_SUCCESS")
            require(row.get("materialization_authorized") is False
                    and row.get("training_authorized") is False
                    and row.get("source_mesh_vertex_roundtrip_courted") is False
                    and row.get("source_skin_bind_parity_courted") is False,
                    "NO_TEACHER_PRIVILEGE")
            aid=row["canonical_asset_id"]
            original_sha=row["original_sha256"]
            require(aid not in identities and original_sha not in original_shas,
                    "DUPLICATE_NORMALIZED_SOURCE")
            identities.add(aid)
            original_shas.add(original_sha)
            require(aid in master and
                    master[aid]["split"]=="FIT",
                    "NORMALIZED_SOURCE_NOT_FIT")
            selection_file=base/f"shard-{shard}.SELECTION.json"
            bounds_file=base/f"shard-{shard}.BOUNDS.json"
            plan_file=base/f"shard-{shard}.NORMALIZATION.json"
            selection=load(selection_file)
            bounds=load(bounds_file)
            normalization=load(plan_file)
            require(selection.get("schema")==SELECTION_SCHEMA and
                    selection.get("shard")==shard and
                    selection.get("materialization_authorized") is False and
                    selection.get("training_authorized") is False,
                    "PILOT_SELECTION_AUTHORITY")
            candidate=selection["candidate"]
            require(candidate["canonical_asset_id"]==aid and
                    candidate["candidate_id"]==master[aid]["candidate_id"] and
                    candidate["source_sha256"]==original_sha and
                    candidate["split"]=="FIT",
                    "SELECTION_MASTER_DRIFT")
            require(bounds.get("schema")==BOUND_SCHEMA and
                    bounds.get("canonical_asset_id")==aid and
                    bounds.get("source_sha256")==original_sha and
                    bounds.get("candidate_id")==candidate["candidate_id"] and
                    bounds.get("anatomical_front")=="UNKNOWN" and
                    bounds.get("anatomical_up")=="UNKNOWN" and
                    bounds.get("source_meter_unit")=="UNKNOWN" and
                    bounds.get("training_authorized") is False and
                    bounds.get("independent_source_truth_courted") is False,
                    "ORIGINAL_BOUNDS_PROVENANCE")
            require(normalization.get("schema")==PLAN_SCHEMA and
                    normalization.get("original_source_sha256")==original_sha and
                    normalization.get("canonical_asset_id")==aid and
                    normalization.get("materialization_authorized") is False and
                    normalization.get("training_authorized") is False and
                    normalization.get("per_view_recenter_permitted") is False and
                    normalization.get("per_view_renormalization_permitted") is False and
                    normalization.get("source_mesh_vertex_roundtrip_courted") is False and
                    normalization.get("source_skin_bind_and_weight_parity_courted") is False,
                    "NORMALIZATION_IS_NOT_SOURCE_TRUTH")
            expected=replay_plan(bounds_file,
                                 expected_sha256=hash_file(bounds_file))
            require(normalization==expected,"SOURCE_BOUND_REPLAY_DRIFT")
            err=normalization["max_source_bbox_corner_relative_error"]
            require(math.isfinite(err) and err<=1e-6,
                    "SOURCE_BOUNDS_CORNERS_ROUNDTRIP")
            max_err=max(max_err,err)
            statuses[status]+=1
            evidence_sha[aid]={
                "selection":hash_file(selection_file),
                "bounds":hash_file(bounds_file),
                "normalization":hash_file(plan_file),
            }
            rows.append({
                "canonical_asset_id":aid,
                "original_source_sha256":original_sha,
                "candidate_id":candidate["candidate_id"],
                "historical_split":"FIT",
                "shard":shard,
                "group":group,
                "source_file_type":candidate["source_ext"],
                "imported_source_mesh_objects":bounds["mesh_object_count"],
                "imported_source_vertices":bounds["mesh_vertex_count"],
                "source_world_half_extent":normalization["global_source_half_extent"],
                "source_world_center_xyz":json.dumps(
                    normalization["global_source_center"]),
                "iristessa_coord_contract":"SOURCE_TO_IRIS_TO_TESSA_REPLAY_CHECKED",
                "bbox_corner_relative_error":err,
                "input_bounds_sha256":hash_file(bounds_file),
                "normalized_plan_sha256":hash_file(plan_file),
                "source_anatomical_front":"UNKNOWN",
                "source_anatomical_up":"UNKNOWN",
                "source_original_units":"UNKNOWN",
                "source_vertex_skin_bone_parity":"UNKNOWN",
                "functional_ik_fk":"UNKNOWN",
                "model_supervision_authority":"NONE",
                "materialization_authorized":False,
                "training_authorized":False,
            })
    require(len(rows)==len(identities)==len(original_shas)==32,
            "EXACT32_UNIQUE_NORMALIZED_ORIGINALS")
    out.mkdir(parents=True,exist_ok=True)
    matrix=out/"CORPUS4008_32SOURCE_NORMALIZATION_REPLAY_MATRIX_V1.csv"
    with matrix.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(sorted(rows,key=lambda x:x["canonical_asset_id"]))
    summary={
        "schema":SCHEMA,
        "status":"32_SOURCE_NORMALIZATION_PLANS_REPLAY_VERIFIED__NO_SOURCE_TRUTH",
        "original_master_selection_sha256":hash_file(master_path),
        "original_master_candidate_count":len(master),
        "normalized_unique_original_sources":len(rows),
        "per_source_normalization_status_counts":dict(statuses),
        "max_bbox_corner_relative_replay_error":max_err,
        "source_normalization_observation_receipts_sha256":evidence_sha,
        "matrix_sha256":hash_file(matrix),
        "source_to_iris_tessa_plan_recomputed":True,
        "source_native_rest_parity_courted":False,
        "source_mesh_and_bone_roundtrip_courted":False,
        "source_skin_bind_parity_courted":False,
        "anatomical_camera_courted":False,
        "graphics_2d_art_courted":False,
        "materialization_authorized":False,
        "training_authorized":False,
    }
    (out/"CORPUS4008_32SOURCE_NORMALIZATION_REPLAY_SUMMARY_V1.json").write_text(
        json.dumps(summary,indent=2,sort_keys=True)+"\n")
    return summary


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--master",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    report=reconcile(a.root,a.master,a.output)
    print(json.dumps({
        "source_count":report["normalized_unique_original_sources"],
        "max_relative_error":report["max_bbox_corner_relative_replay_error"],
        "materialization_authorized":False,
    },sort_keys=True))


if __name__=="__main__":
    main()
