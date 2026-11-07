"""Wave3 cannot overlap wave1/2 sources, split holdouts or fake admission."""
from __future__ import annotations
import json
import pytest
from tools.corpus.plan_corpus4008_wave3_disjoint_v1 import next_plan
from tools.corpus.plan_corpus4008_wave1_import_v1 import plan as first_plan


def fixture(tmp):
    entries=[]
    for i in range(52):
        h=("%064x"%(i+1))
        ext=[".fbx",".glb",".gltf"][i%3]
        entries.append({
          "canonical_asset_id":f"asset_{i:024x}",
          "candidate_id":"objaverse_animated_originals:"+h,
          "source_expected_sha256":h,
          "retrieved_source_sha256":h,
          "historical_split":"FIT",
          "file_type":ext,
          "private_staged_filename":h+ext,
          "source_bytes_recovered":True,
          "status":"EXACT_ORIGINAL_BYTES_STAGED__COURTS_PENDING",
          "byte_count":1024,
          "training_authorized":False,
          "current_license_revalidated":False,
        })
    entries[0]["historical_split"]="EXTERNAL_HOLDOUT"
    entries[1]["source_bytes_recovered"]=False
    entries[1]["status"]="HTTP_STATUS_404"
    entries[1]["retrieved_source_sha256"]=None
    p=tmp/"SHARD_PRIVATE.json"
    p.write_text(json.dumps({
       "summary":{
          "schema":"RealSaS.Corpus4008PinnedGithubOriginalShard.v1",
          "shard_index":3,"shard_count":32,
          "private_quarantine_only":True,"training_authorized":False,
          "deferred_count":0,
       },"candidates_private":entries,
    }))
    return p


def test_24_additional_sources_are_disjoint_and_reproducible(tmp_path):
    p=fixture(tmp_path)
    prior=first_plan(p,shard_index=3,limit=8)
    new=next_plan(p,shard_index=3,limit=24)
    again=next_plan(p,shard_index=3,limit=24)
    assert new==again
    old={x["source_sha256"] for x in prior["entries"]}
    fresh={x["source_sha256"] for x in new["entries"]}
    assert old==set(new["prior_wave_reserved_source_sha256"])
    assert len(old)==8 and len(fresh)==24 and old.isdisjoint(fresh)
    assert all(x["split"]=="FIT" and x["training_authorized"] is False
               for x in new["entries"])
    assert {x["source_ext"] for x in new["entries"]} == {
        ".fbx",".glb",".gltf",
    }


def test_cross_shard_receipt_cannot_be_reused(tmp_path):
    p=fixture(tmp_path)
    with pytest.raises(ValueError,match="SOURCE_SHARD_CERTIFICATE"):
        next_plan(p,shard_index=4,limit=24)


def test_malicious_implicit_training_pass_is_blocked(tmp_path):
    p=fixture(tmp_path)
    j=json.loads(p.read_text())
    j["candidates_private"][10]["training_authorized"]=True
    p.write_text(json.dumps(j))
    with pytest.raises(ValueError,match="WAVE3_SOURCE_ADMISSION"):
        next_plan(p,shard_index=3,limit=24)


def test_no_source_more_than_32mb_and_no_holdout(tmp_path):
    p=fixture(tmp_path)
    j=json.loads(p.read_text())
    j["candidates_private"][6]["byte_count"]=70*1024*1024
    p.write_text(json.dumps(j))
    r=next_plan(p,shard_index=3,limit=24)
    assert all(x["source_byte_count"]<=32*1024*1024 for x in r["entries"])
    assert "asset_"+f"{0:024x}" not in {
        x["canonical_asset_id"] for x in r["entries"]}
