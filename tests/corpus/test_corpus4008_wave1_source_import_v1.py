from __future__ import annotations

import json

import pytest

from tools.corpus.plan_corpus4008_wave1_import_v1 import (
    plan, SCHEMA, SOURCE_SCHEMA,
)


def case(tmp_path):
    entries=[]
    for i in range(16):
        h=f"{i:064x}"
        ext=[".fbx",".glb",".gltf"][i%3]
        entries.append({
            "canonical_asset_id":f"asset_{i:02d}",
            "candidate_id":"objaverse_animated_originals:"+h,
            "source_expected_sha256":h,
            "retrieved_source_sha256":h,
            "historical_split":"FIT",
            "file_type":ext,
            "private_staged_filename":h+ext,
            "source_bytes_recovered":True,
            "status":"EXACT_ORIGINAL_BYTES_STAGED__COURTS_PENDING",
            "byte_count":1000000,
            "training_authorized":False,
            "current_license_revalidated":False,
        })
    entries[0]["historical_split"]="EXTERNAL_HOLDOUT"
    entries[1]["source_bytes_recovered"]=False
    entries[1]["retrieved_source_sha256"]=None
    entries[1]["status"]="HTTP_STATUS_404"
    doc={
        "summary":{
            "schema":SOURCE_SCHEMA,
            "shard_index":0,"shard_count":32,
            "private_quarantine_only":True,
            "training_authorized":False,
            "deferred_count":0
        },"candidates_private":entries,
    }
    path=tmp_path/"SHARD_PRIVATE.json"
    path.write_text(json.dumps(doc))
    return path


def test_wave1_format_diversity_courts_and_train_firewall(tmp_path):
    p=case(tmp_path)
    r=plan(p,shard_index=0,limit=8)
    assert r["schema"]==SCHEMA
    assert r["sample_size"]==8
    assert len({x["canonical_asset_id"] for x in r["entries"]})==8
    assert {x["source_ext"] for x in r["entries"]}=={
        ".fbx",".glb",".gltf",
    }
    assert not any(x["canonical_asset_id"] in ("asset_00","asset_01")
                   for x in r["entries"])
    assert all(x["training_authorized"] is False for x in r["entries"])
    assert all(x["source_character_courted"] is False for x in r["entries"])
    assert r==plan(p,shard_index=0,limit=8)


def test_wave1_rejects_forged_source_receipt(tmp_path):
    p=case(tmp_path)
    j=json.loads(p.read_text())
    j["summary"]["training_authorized"]=True
    p.write_text(json.dumps(j))
    with pytest.raises(ValueError,match="SOURCE_SHARD_CERTIFICATE"):
        plan(p,shard_index=0)


def test_wave1_no_sample_from_failed_or_holdout_originals(tmp_path):
    p=case(tmp_path)
    j=json.loads(p.read_text())
    for row in j["candidates_private"]:
        if row["historical_split"]=="FIT":
            row["status"]="HTTP_STATUS_404"
            row["source_bytes_recovered"]=False
            row["retrieved_source_sha256"]=None
    p.write_text(json.dumps(j))
    with pytest.raises(ValueError,match="NO_SAFE_FIT_CANDIDATES"):
        plan(p,shard_index=0)


def test_wave1_does_not_equate_two_animations_to_two_sources(tmp_path):
    p=case(tmp_path)
    j=json.loads(p.read_text())
    # A repeated canonical source identity must never enter twice.
    x=dict(j["candidates_private"][3])
    j["candidates_private"].append(x)
    p.write_text(json.dumps(j))
    with pytest.raises(ValueError,match="REPEATED_SOURCE_IDENTITY"):
        plan(p,shard_index=0,limit=16)
