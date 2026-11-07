#!/usr/bin/env python3
"""Select exactly one already-imported FIT source per recovery shard.

A genuine original SHA and source byte count are required; selection is a
disjoint pilot across 32 shards, not a quality ranking or model admission.
Never substitute a different source, never select a failed import as PASS.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

SCHEMA="RealSaS.Corpus4008ParallelNormalizationPilotSelection.v1"
PLAN_SCHEMA="RealSaS.Corpus4008Wave1SourceImportPlan.v1"
SUCCESS="IMPORTED_SOURCE_OBSERVATIONS_ONLY"


def require(v,why):
    if not v:raise ValueError("C4008_PARNORM_"+why)


def pick(plan_file:Path,results_file:Path,shard:int):
    require(0<=shard<32,"SHARD_INDEX")
    p=json.loads(plan_file.read_text())
    require(p.get("schema")==PLAN_SCHEMA
            and p.get("shard_index")==shard
            and p.get("model_training_authorized") is False
            and p.get("sample_size")==len(p.get("entries",[]))==8,
            "ORIGINAL_EIGHT_RECEIPT_PLAN")
    rows=[json.loads(x) for x in results_file.read_text().splitlines()
          if x.strip()]
    index={x["asset_id"]:x for x in rows}
    require(len(index)==len(rows)==8,"SOURCE_IDENTITY_ROWS")
    eligible=[]
    for e in p["entries"]:
        aid=e["canonical_asset_id"]
        require(aid in index,"RECOVERY_ASSET_NOT_IN_BATCH")
        outcome=index[aid]
        require(outcome["source_sha256"]==e["source_sha256"]
                and outcome["training_authorized"] is False
                and outcome["truth_admitted"] is False,
                "ORIGINAL_SHA_OR_TRAINER_PRIVILEGE")
        require(e["split"]=="FIT"
                and e["original_rights_courted"] is False
                and e["source_character_courted"] is False
                and e["training_authorized"] is False,
                "SOURCE_REVIEW_BYPASS")
        source=e["source_sha256"]
        require(len(source)==64
                and all(c in "0123456789abcdef" for c in source)
                and (e["candidate_id"].endswith(source) or
                     e["candidate_id"].endswith(source[:24])),
                "SOURCE_CANONICAL_SHA")
        require(e["source_ext"] in (".glb",".fbx",".gltf")
                and e["source_quarantine_basename"]==source+e["source_ext"]
                and isinstance(e["source_byte_count"],int)
                and 0<e["source_byte_count"]<=32*1024*1024,
                "UNSAFE_ORIGINAL_SOURCE_BYTES")
        if outcome["status"]==SUCCESS:
            eligible.append(e)
    require(eligible,"NO_VERIFIED_IMPORT_SUCCESS_IN_SHARD")
    chosen=sorted(eligible,key=lambda e:e["source_sha256"])[0]
    return {
        "schema":SCHEMA,
        "source_sample_kind":"EXACT_SHA_VERIFIED_FIT_IMPORT_OBSERVATION_ONLY",
        "shard":shard,
        "selection_policy":"MIN_SHA_OF_SUCCEEDED_SOURCE_IMPORTS",
        "previous_successful_import_candidates":len(eligible),
        "candidate":chosen,
        "source_anatomical_orientation":"UNKNOWN",
        "source_authoring_unit":"UNKNOWN",
        "source_appearance_parity":"UNKNOWN",
        "source_skin_bind_parity":"UNKNOWN",
        "source_mesh_geometry_parity":"UNKNOWN",
        "training_authorized":False,
        "materialization_authorized":False,
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--plan",type=Path,required=True)
    ap.add_argument("--results",type=Path,required=True)
    ap.add_argument("--shard",type=int,required=True)
    ap.add_argument("--output",type=Path,required=True)
    a=ap.parse_args()
    j=pick(a.plan,a.results,a.shard)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(j,sort_keys=True,indent=2)+"\n")
    print(json.dumps({
        "shard":a.shard,"candidate_count":1,
        "previous_import_success_candidates":j["previous_successful_import_candidates"],
        "materialization_authorized":False,
    },sort_keys=True))


if __name__=="__main__":
    main()
