#!/usr/bin/env python3
"""Strict merger for up to 1024 original-source import observations in 3 waves.

Inputs are evidence from 8+24+32 independent hosted-source batches with
immutable source identity and per-receipt SHA. Selection is SHA-pinned against
original Master3993. No import observation becomes commercial use rights,
character class, semantic, verified IK or MIRA/Graphics/IRIS training PASS.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter,defaultdict
from pathlib import Path

from tools.corpus.reconcile_corpus4008_wave1_source_rig_matrix_v1 import (
    EXPECTED_SHARDS as WAVE1_SHARDS,
    reconcile as reconcile_wave,
)
from tools.corpus.reconcile_corpus4008_recovery_into_source_matrix_v1 import (
    SELECT_SHA as MASTER_SELECTION_SHA,
)

WAVE2_SHARDS=tuple(i for i in range(32) if i not in WAVE1_SHARDS)
WAVE3_SHARDS=tuple(range(32))
SCHEMA="RealSaS.Corpus4008ParallelSourceRigObservationMerge.v1"
PER_HEAD=("IRIS_GEOMETRY","IRIS_SEMANTIC","IRIS_GRAPHICS",
          "TESSA","AXIS","MIRA")


def require(value, reason):
    if not value:raise ValueError("C4008_WAVE_MERGE_"+reason)


def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(4<<20),b""):
            h.update(b)
    return h.hexdigest()


def load_csv(path):
    with path.open(newline="",encoding="utf-8") as f:
        return list(csv.DictReader(f))


def numeric_or_none(value):
    if value in (None,"", "None"):
        return None
    try:return int(value)
    except ValueError:raise ValueError("C4008_WAVE_MERGE_NONINTEGER_MEASUREMENT")


def merge(master_path:Path,wave1:Path,wave2:Path,wave3:Path,
          output:Path,*,expected_count:int=3993,
          expected_master_sha:str=MASTER_SELECTION_SHA):
    require(sha(master_path)==expected_master_sha,"MASTER_SELECTION_SHA")
    master=json.loads(master_path.read_text())
    require(isinstance(master,dict) and len(master)==expected_count,
            "MASTER_SELECTION_COUNT")
    wave_configs=(
        (1,wave1,WAVE1_SHARDS),(2,wave2,WAVE2_SHARDS),
        (3,wave3,WAVE3_SHARDS),
    )
    output.mkdir(parents=True,exist_ok=True)
    observations=[]
    wave_summaries={}
    observed_count=Counter()
    for wave_idx,root,shards in wave_configs:
        # This rechecks every source shard, every original SHA, every receipt,
        # every status and SHA of compressed full source observation.
        result=reconcile_wave(root,output/("wave%d"%wave_idx),
                              expected_shards=shards)
        rows=load_csv(output/("wave%d"%wave_idx)/
                      "WAVE1_SOURCE_RIG_SCENE_TYPED_MATRIX_V1.csv")
        require(len(rows)==result["sampled_original_source_count"],
                "WAVE_CSV_ROWS")
        wave_summaries[str(wave_idx)]=result
        for x in rows:
            x["import_wave"]=wave_idx
            observations.append(x)
            observed_count[wave_idx]+=1
    indexed={}
    source_clusters={}
    for x in observations:
        aid=x["canonical_asset_id"]
        require(aid not in indexed,"REPEATED_ASSET_ACROSS_WAVES")
        source=x["source_sha256"]
        require(source not in source_clusters,
                "REPEATED_ORIGINAL_SHA_ACROSS_WAVES")
        source_clusters[source]=aid
        require(aid in master and
                x["candidate_id"]==master[aid]["candidate_id"] and
                master[aid]["split"]=="FIT",
                "MASTER_IDENTITY_SPLIT_DRIFT")
        # Full SHA for recovered Objaverse, never trust a canonical short hash
        token=x["candidate_id"].split(":",1)[-1]
        require(len(token) in (24,64) and
                source.startswith(token),"ORIGINAL_CANDIDATE_SHA")
        require(x["training_authorized"]=="False"
                and x["head_admission_count"]=="0"
                and x["functional_native_ik_fk"]=="UNKNOWN"
                and x["source_character_vs_prop"]=="UNKNOWN"
                and x["graphics_source_bound_2d_art_qualified"]=="UNKNOWN",
                "ROUTING_PRIVILEGE_ESCALATION")
        indexed[aid]=x
    # Same-source two-pass plan proof: every wave3 shard carries exactly
    # the original eight per-shard sources in its exclusion set.
    for shard in WAVE3_SHARDS:
        original_wave=(1 if shard in WAVE1_SHARDS else 2)
        original_root=wave1 if original_wave==1 else wave2
        prior=json.loads((original_root/("shard-%d"%shard)/
                          "PLAN.json").read_text())
        future=json.loads((wave3/("shard-%d"%shard)/
                           "PLAN.json").read_text())
        require(prior.get("sample_size")==len(prior.get("entries",[]))==8,
                "PRIOR_8_SOURCE_PLAN")
        require(future.get("sample_method")==
                "DISJOINT_AFTER_FIRST8_PER_SHARD__BALANCED_FORMAT_PLUS_SHA_QUANTILES",
                "DISJOINT_SELECT_POLICY")
        reserve=set(future.get("prior_wave_reserved_source_sha256",[]))
        expected={x["source_sha256"] for x in prior["entries"]}
        require(reserve==expected and len(reserve)==8
                and future.get("prior_wave_reserved_count")==8,
                "WAVE3_EXCLUSION_MISMATCH")
        require(not reserve.intersection(
            {x["source_sha256"] for x in future["entries"]}),
            "WAVE3_ORIGINAL_OVERLAP")
    counts=Counter()
    formats=Counter()
    observations_by_head=Counter()
    observed_distinct_originals_with_skin=0
    for x in observations:
        status=x["source_import_observation_status"]
        counts[status]+=1
        formats[x["source_format"]]+=1
        if status!="IMPORTED_SOURCE_OBSERVATIONS_ONLY":
            continue
        for key in PER_HEAD:
            # imported *observations*, NOT independent training truth:
            # geometry metadata, appearance hint and source group hints.
            if key in ("IRIS_GEOMETRY","IRIS_SEMANTIC","IRIS_GRAPHICS"):
                if (numeric_or_none(x["mesh_object_count"]) or 0)>0:
                    observations_by_head[key]+=1
            if key in ("AXIS","TESSA"):
                if (numeric_or_none(x["armature_count"]) or 0)>0:
                    observations_by_head[key]+=1
            if key=="MIRA":
                sampled=(numeric_or_none(x["skin_rows_sampled"]) or 0)
                empty=(numeric_or_none(x["skin_rows_zero_weights"]) or 0)
                if sampled>empty and (
                     numeric_or_none(x["armature_count"]) or 0)>0:
                    observations_by_head[key]+=1
                    observed_distinct_originals_with_skin+=1
    require(len(indexed)==sum(observed_count.values()),
            "FINAL_COUNT")
    columns=list(observations[0])
    with (output/"CORPUS4008_PARALLEL1024_SOURCE_RIG_OBSERVATION_V1.csv").open(
            "w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,columns)
        w.writeheader();w.writerows(sorted(observations,
                                           key=lambda x:x["canonical_asset_id"]))
    report={
        "schema":SCHEMA,
        "master_selected_source_sha256":sha(master_path),
        "master_source_candidate_count":len(master),
        "wave_source_sample_counts":dict(sorted(observed_count.items())),
        "total_distinct_source_candidates_examined":len(indexed),
        "per_source_import_status":dict(sorted(counts.items())),
        "input_format_distribution":dict(sorted(formats.items())),
        "per_head_import_observation_count_NOT_qualified":{
           h:observations_by_head[h] for h in PER_HEAD
        },
        "per_head_training_qualified_count":{h:0 for h in PER_HEAD},
        "original_source_license_independently_reviewed":False,
        "source_morphology_independently_reviewed":False,
        "native_IK_FK_behavior_independently_verified":False,
        "stylized_2D_art_human_court_approved":False,
        "fail_closed_if_any_shard_or_receipt_missing":True,
        "all_source_formats_and_split_identity_courted":True,
        "training_authorized":False,
        "note":"SOURCE IMPORT OBSERVATIONS. No per-head independent model supervision teacher is admitted.",
    }
    (output/"CORPUS4008_PARALLEL1024_OBSERVATION_SUMMARY_V1.json").write_text(
        json.dumps(report,sort_keys=True,indent=2)+"\n")
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--selection",type=Path,required=True)
    p.add_argument("--wave1",type=Path,required=True)
    p.add_argument("--wave2",type=Path,required=True)
    p.add_argument("--wave3",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    report=merge(a.selection,a.wave1,a.wave2,a.wave3,a.output)
    print(json.dumps(report,sort_keys=True))


if __name__=="__main__":
    main()
