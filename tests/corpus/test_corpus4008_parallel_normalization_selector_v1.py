import json

import pytest

from tools.corpus.plan_corpus4008_parallel_normalization_v1 import (
    pick, SUCCESS, PLAN_SCHEMA,
)


def make(tmp_path, shard=5):
    entries=[]
    status=[]
    for i in range(8):
        digest=f"{shard*100+i+1:064x}"
        aid=f"asset_{shard*100+i+1:024x}"
        entries.append({
            "canonical_asset_id":aid,
            "candidate_id":"objaverse_animated_originals:"+digest,
            "source_sha256":digest,
            "source_ext":".glb",
            "source_quarantine_basename":digest+".glb",
            "source_byte_count":250000,
            "split":"FIT",
            "original_rights_courted":False,
            "source_character_courted":False,
            "training_authorized":False,
        })
        status.append({
            "asset_id":aid,"source_sha256":digest,
            "status":SUCCESS if i>0 else "SOURCE_IMPORT_EXCEPTION",
            "truth_admitted":False,"training_authorized":False,
        })
    p=tmp_path/"PLAN.json"
    r=tmp_path/"RESULTS.jsonl"
    p.write_text(json.dumps({
        "schema":PLAN_SCHEMA,
        "shard_index":shard,
        "model_training_authorized":False,
        "sample_size":8,"entries":entries,
    }))
    r.write_text("".join(json.dumps(x)+"\n" for x in status))
    return p,r


def test_select_recovered_import_success_not_failed_first_row(tmp_path):
    p,r=make(tmp_path)
    x=pick(p,r,5)
    assert x["shard"]==5
    assert x["previous_successful_import_candidates"]==7
    assert x["candidate"]["source_sha256"]==f"{502:064x}"
    assert x["materialization_authorized"] is False
    assert x["source_anatomical_orientation"]=="UNKNOWN"
    assert x==pick(p,r,5)


def test_split_lineage_not_inferred_from_file_extension(tmp_path):
    p,r=make(tmp_path)
    j=json.loads(p.read_text())
    j["entries"][2]["split"]="EXTERNAL_HOLDOUT"
    p.write_text(json.dumps(j))
    with pytest.raises(ValueError,match="SOURCE_REVIEW_BYPASS"):
        pick(p,r,5)


def test_corrupt_original_sha_is_fatal(tmp_path):
    p,r=make(tmp_path)
    j=json.loads(r.read_text().splitlines()[1])
    j["source_sha256"]="0"*64
    v=[json.loads(x) for x in r.read_text().splitlines()]
    v[1]=j
    r.write_text("\n".join(json.dumps(x) for x in v)+"\n")
    with pytest.raises(ValueError,match="ORIGINAL_SHA_OR_TRAINER_PRIVILEGE"):
        pick(p,r,5)


def test_shard_id_cannot_reuse_another_shards_sources(tmp_path):
    p,r=make(tmp_path)
    with pytest.raises(ValueError,match="ORIGINAL_EIGHT_RECEIPT_PLAN"):
        pick(p,r,7)


def test_no_success_cannot_make_normalization_teacher(tmp_path):
    p,r=make(tmp_path)
    v=[json.loads(x) for x in r.read_text().splitlines()]
    for x in v:x["status"]="SOURCE_IMPORT_EXCEPTION"
    r.write_text("\n".join(json.dumps(x) for x in v)+"\n")
    with pytest.raises(ValueError,match="NO_VERIFIED_IMPORT_SUCCESS_IN_SHARD"):
        pick(p,r,5)
