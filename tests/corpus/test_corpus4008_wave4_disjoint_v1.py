"""No source recovered in first 1024 is selected for Wave4 reimport."""
import json
import pytest

from tools.corpus.plan_corpus4008_wave1_import_v1 import plan as first
from tools.corpus.plan_corpus4008_wave3_disjoint_v1 import next_plan as third
from tools.corpus.plan_corpus4008_wave4_remaining_v1 import plan as fourth


def make(tmp):
    rows=[]
    for i in range(109):
        h=f"{i+1:064x}"
        ext=[".fbx",".glb",".gltf"][i%3]
        rows.append({
            "canonical_asset_id":f"asset_{i+1:024x}",
            "candidate_id":"objaverse_animated_originals:"+h,
            "source_expected_sha256":h,
            "retrieved_source_sha256":h,
            "historical_split":"FIT" if i<105 else "EXTERNAL_HOLDOUT",
            "file_type":ext,
            "private_staged_filename":h+ext,
            "source_bytes_recovered":True,
            "status":"EXACT_ORIGINAL_BYTES_STAGED__COURTS_PENDING",
            "byte_count":230000,
            "training_authorized":False,
            "current_license_revalidated":False,
        })
    p=tmp/"SHARD_PRIVATE.json"
    p.write_text(json.dumps({
        "summary":{
            "schema":"RealSaS.Corpus4008PinnedGithubOriginalShard.v1",
            "shard_index":11,"shard_count":32,
            "deferred_count":0,
            "private_quarantine_only":True,
            "training_authorized":False,
        },"candidates_private":rows,
    }))
    return p


def test_64_new_sources_are_disjoint_from_prior_8_and_24(tmp_path):
    p=make(tmp_path)
    a=first(p,shard_index=11,limit=8)
    b=third(p,shard_index=11,limit=24)
    c=fourth(p,shard=11,limit=64)
    s1={x["source_sha256"] for x in a["entries"]}
    s2={x["source_sha256"] for x in b["entries"]}
    s3={x["source_sha256"] for x in c["entries"]}
    assert len(s1)==8 and len(s2)==24 and len(s3)==64
    assert not (s1&s2 or s1&s3 or s2&s3)
    assert c["previous_wave_shard_reserved_count"]==32
    assert c["post_wave4_remaining_eligible_fit"]==9
    assert c==fourth(p,shard=11,limit=64)
    assert all(x["split"]=="FIT" and x["training_authorized"] is False
               for x in c["entries"])


def test_excluded_holdout_never_returned(tmp_path):
    p=make(tmp_path)
    j=json.loads(p.read_text())
    for i in range(50,109):
        j["candidates_private"][i]["historical_split"]="EXTERNAL_HOLDOUT"
    p.write_text(json.dumps(j))
    c=fourth(p,shard=11,limit=64)
    assert c["new_selected_count"]==18
    assert all(x["split"]=="FIT" for x in c["entries"])


def test_forged_training_rights_fail_closed(tmp_path):
    p=make(tmp_path)
    j=json.loads(p.read_text())
    j["candidates_private"][80]["training_authorized"]=True
    p.write_text(json.dumps(j))
    with pytest.raises(ValueError,match="WAVE4_UNVERIFIED_RIGHTS_CANNOT_ADMIT_TRAINING"):
        fourth(p,shard=11,limit=64)


def test_shard_swap_does_not_steal_sources(tmp_path):
    p=make(tmp_path)
    with pytest.raises(ValueError,match="SOURCE_SHARD_CERTIFICATE"):
        fourth(p,shard=12,limit=64)
