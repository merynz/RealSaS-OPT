#!/usr/bin/env python3
"""32-source SHA-bounded per-shard Blender runner with explicit failure receipts.

Each source fetched once, opened once, SHA verified twice, and original bytes
deleted after observation. No source bytes or secret paths enter public GitHub
logs. Dynamic source IDs are validated against SHA-pinned readback plan.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from collections import Counter
from pathlib import Path

SCHEMA="RealSaS.Corpus4008DenseNormalizationOriginalBatch.v1"
MAX_SOURCE_BYTES=32*1024*1024


def require(x,why):
    if not x:raise ValueError("C4008_DENSE_RUNNER_"+why)


def hash_file(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda:f.read(4<<20),b""):
            h.update(block)
    return h.hexdigest()


def one(entry,base,source_dir,out,bin_path,script_path):
    aid=entry["canonical_asset_id"]
    digest=entry["original_source_sha256"]
    ext=entry["original_source_extension"]
    require(re.fullmatch(r"asset_[0-9a-f]{24}",aid) and
            re.fullmatch(r"[0-9a-f]{64}",digest) and
            ext in (".fbx",".glb",".gltf") and
            entry["candidate_id"].endswith(digest) and
            entry["training_authorized"] is False,
            "UNTRUSTED_SOURCE_PLAN")
    filename=digest+ext
    source=source_dir/filename
    stem=out/"receipts"/aid
    log=out/"logs"/(aid+".blender.log")
    fetch_log=out/"logs"/(aid+".fetch.log")
    status="ORIGINAL_FETCH_ERROR"
    try:
        with fetch_log.open("wb") as f:
            result=subprocess.run(["rclone","copyto",
                                   base.rstrip("/")+"/"+filename,str(source),
                                   "--retries","3"],stdout=f,stderr=f,
                                  timeout=180,check=False)
        if result.returncode!=0:
            return status
        status="ORIGINAL_SOURCE_SHA_OR_SIZE_FAILURE"
        if not source.is_file() or not 0<source.stat().st_size<=MAX_SOURCE_BYTES:
            return status
        if hash_file(source)!=digest:
            return status
        status="DENSE_BLENDER_SOURCE_IMPORT_EXCEPTION"
        bounds=Path(str(stem)+".BOUNDS.json")
        dense=Path(str(stem)+".DENSE.json")
        command=[str(bin_path),"--background","--factory-startup",
                 "--disable-autoexec","--python-exit-code","1",
                 "--python",str(script_path),"--",
                 "--source",str(source),
                 "--source-sha256",digest,
                 "--asset-id",aid,
                 "--candidate-id",entry["candidate_id"],
                 "--bounds-output",str(bounds),
                 "--dense-output",str(dense)]
        with log.open("wb") as f:
            rc=subprocess.run(command,stdout=f,stderr=f,timeout=180,check=False)
        if rc.returncode!=0 or not bounds.is_file() or not dense.is_file():
            return status
        d=json.loads(dense.read_text())
        b=json.loads(bounds.read_text())
        require(d.get("schema")==
                "RealSaS.Corpus4008ImportedDenseNormalizationConsistency.v1"
                and d.get("original_source_sha256")==digest
                and d.get("canonical_asset_id")==aid
                and d.get("training_authorized") is False
                and d.get("independent_source_truth_courted") is False
                and b.get("source_sha256")==digest,
                "SOURCE_DENSE_PROVENANCE")
        from tools.corpus.build_corpus4008_normalization_plan_v1 import plan
        normalized=plan(bounds,expected_sha256=hash_file(bounds))
        require(normalized["training_authorized"] is False
                and normalized["materialization_authorized"] is False
                and normalized["original_source_sha256"]==digest,
                "DENSE_NORMALIZATION_NO_TEACHER")
        np=Path(str(stem)+".NORMALIZATION.json")
        np.write_text(json.dumps(normalized,indent=2,sort_keys=True)+"\n")
        return "IMPORTED_FULL_VERTEX_BONE_WEIGHT_NUMERIC_CONSISTENCY_ONLY"
    except subprocess.TimeoutExpired:
        return "SOURCE_PROCESS_TIMEOUT"
    except Exception:
        return "SOURCE_EVIDENCE_VALIDATION_EXCEPTION"
    finally:
        source.unlink(missing_ok=True)


def run(root:Path,plan_file:Path,originals_root:str,
        blender:Path,script:Path,output:Path):
    require(blender.is_file() and script.is_file(),"PINNED_IMPORTER_ABSENT")
    data=json.loads(plan_file.read_text())
    require(data.get("schema")==
            "RealSaS.Corpus4008DenseNormalizationShardPlan.v1"
            and data.get("source_count")==32
            and data.get("training_authorized") is False
            and data.get("materialization_authorized") is False,
            "DENSE_ORIGINAL_PLAN")
    shard=data.get("shard")
    require(type(shard) is int and 0<=shard<32,"SHARD")
    require(originals_root.startswith("gdrive:RealSaS_MASTER_CORPUS_1024_V3/"),
            "SOURCE_PRIVATE_REMOTE")
    source=root/"source"
    source.mkdir(parents=True,exist_ok=True)
    for sub in ("receipts","logs"):
        (output/sub).mkdir(parents=True,exist_ok=True)
    rows=[]
    for e in data["source_candidates"]:
        require(e["source_recovery_shard"]==shard and
                e["split"]=="FIT" and
                e["source_rights_courted"] is False,
                "SOURCE_PLANNER_SPLIT")
        state=one(e,originals_root+f"/shard-{shard}/verified_quarantine_objects",
                  source,output,blender,script)
        rows.append({
            "canonical_asset_id":e["canonical_asset_id"],
            "original_source_sha256":e["original_source_sha256"],
            "source_shard":shard,
            "status":state,
            "source_dense_observation_truth":"IMPORTED_MATH_ONLY",
            "source_original_authoring_rig_truth":"UNKNOWN",
            "source_2d_game_art_truth":"UNKNOWN",
            "materialization_authorized":False,
            "training_authorized":False,
        })
    require(len(rows)==len({r["original_source_sha256"] for r in rows})==32,
            "RESULT_DUPLICATES")
    (output/"RESULTS.jsonl").write_text(
        "".join(json.dumps(x,sort_keys=True)+"\n" for x in rows))
    report={
        "schema":SCHEMA,
        "shard":shard,
        "plan_sha256":hash_file(plan_file),
        "source_count":len(rows),
        "status_counts":dict(sorted(Counter(x["status"] for x in rows).items())),
        "source_original_full_SHA_verified_prior_to_import":True,
        "source_index_parity_court":"IN_MEMORY_IMPORTED_SCENE_ONLY",
        "original_source_rig_parity_court":"UNKNOWN",
        "commercial_training_rights_courted":False,
        "graphics_style2d_courted":False,
        "training_authorized":False,
        "materialization_authorized":False,
    }
    (output/"BATCH_SUMMARY.json").write_text(
        json.dumps(report,indent=2,sort_keys=True)+"\n")
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--plan",type=Path,required=True)
    p.add_argument("--originals-root",required=True)
    p.add_argument("--blender",type=Path,required=True)
    p.add_argument("--script",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    result=run(a.root,a.plan,a.originals_root,a.blender,a.script,a.output)
    print(json.dumps({"shard":result["shard"],
                      "source_count":result["source_count"],
                      "status_counts":result["status_counts"],
                      "training_authorized":False},sort_keys=True))


if __name__=="__main__":
    main()
