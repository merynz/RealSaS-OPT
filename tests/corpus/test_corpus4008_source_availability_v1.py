import hashlib
import json

import pytest

from tools.corpus.reconcile_corpus4008_recovery_into_source_matrix_v1 import (
    run,PLAN_SCHEMA,SHARD_SCHEMA,
)


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True))
    return path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def case(tmp):
    master={
        "a":{"canonical_asset_id":"a","candidate_id":"obs:"+"a"*64,
             "source_registry_id":"obs","split":"FIT","decision":"ADMIT"},
        "b":{"canonical_asset_id":"b","candidate_id":"obs:"+"b"*64,
             "source_registry_id":"obs","split":"FIT","decision":"ADMIT"},
        "h":{"canonical_asset_id":"h","candidate_id":"obs:"+"c"*64,
             "source_registry_id":"obs","split":"EXTERNAL_HOLDOUT","decision":"ADMIT"},
        "old":{"canonical_asset_id":"old","candidate_id":"quaternius:"+"d"*24,
             "source_registry_id":"quaternius","split":"DEV","decision":"ADMIT"},
    }
    m=dump(tmp/"selection.json",master)
    plan_rows=[
        {"canonical_asset_id":aid,
         "candidate_id":master[aid]["candidate_id"],
         "historical_split":master[aid]["split"],
         "source_expected_sha256":l*64}
        for aid,l in (("a","a"),("b","b"),("h","c"))
    ]
    plan=dump(tmp/"plan.json",{
        "schema":PLAN_SCHEMA,"missing_originals_private":plan_rows,
        "summary":{"missing_original_count":3},
    })
    shard_dir=tmp/"shards"
    success={
        "canonical_asset_id":"a","candidate_id":master["a"]["candidate_id"],
        "historical_split":"FIT","source_expected_sha256":"a"*64,
        "retrieved_source_sha256":"a"*64,"source_bytes_recovered":True,
        "status":"EXACT_ORIGINAL_BYTES_STAGED__COURTS_PENDING",
        "private_staged_filename":"a"*64+".fbx","byte_count":19,
        "training_authorized":False,"current_license_revalidated":False,
    }
    failure={
        "canonical_asset_id":"b","candidate_id":master["b"]["candidate_id"],
        "historical_split":"FIT","source_expected_sha256":"b"*64,
        "retrieved_source_sha256":None,"source_bytes_recovered":False,
        "status":"HTTP_STATUS_404","byte_count":0,
        "training_authorized":False,"current_license_revalidated":False,
    }
    for i,outcome in enumerate([success,failure]):
        dump(shard_dir/f"shard-{i}"/"SHARD_PRIVATE.json",{
            "summary":{
                "schema":SHARD_SCHEMA,"shard_count":2,
                "shard_index":i,"selected_count":1,
                "eligible_in_shard_count":1,"deferred_count":0,
                "private_quarantine_only":True,"training_authorized":False,
            },
            "candidates_private":[outcome],
        })
    return m,plan,shard_dir


def test_all_4_source_states_reconciled_without_a_single_truth_pass(tmp_path):
    master,plan,shards=case(tmp_path)
    report=run(master,plan,shards,tmp_path/"result",
               selection_count=4,expected_missing=3,shard_count=2,
               expected_plan_sha=sha(plan),expected_selection_sha=sha(master))
    assert report["reported_recovered_sha_receipt_count"]==1
    assert report["source_stage_counts"]=={
        "SHA_VERIFIED_RECOVERY_RECEIPT":1,
        "RECOVERY_EXCEPTION":1,
        "PROTECTED_HOLDOUT_ORIGINAL_NOT_TOUCHED":1,
        "LEGACY_PATH_MATCH_UNREHASHED":1,
    }
    import csv
    with (tmp_path/"result/SOURCE_AVAILABILITY_MATRIX_V1.csv").open() as fp:
        rows={x["canonical_asset_id"]:x for x in csv.DictReader(fp)}
    assert rows["a"]["graphics_image_target_domain"]=="STYLIZED_2D_GAME_ART_REQUIRED"
    assert rows["h"]["holdout_protected"]=="True"
    assert rows["b"]["recovery_exception"]=="HTTP_STATUS_404"
    assert all(x["training_authorized"]=="False" for x in rows.values())


def test_holdout_in_train_recovery_is_rejected(tmp_path):
    master,plan,shards=case(tmp_path)
    p=shards/"shard-1/SHARD_PRIVATE.json"
    j=json.loads(p.read_text())
    j["candidates_private"][0].update({
        "canonical_asset_id":"h","candidate_id":"obs:"+"c"*64,
        "historical_split":"EXTERNAL_HOLDOUT",
        "source_expected_sha256":"c"*64,
    })
    dump(p,j)
    with pytest.raises(ValueError,match="HOLDOUT_IN_RECOVERY"):
        run(master,plan,shards,tmp_path/"out",
            selection_count=4,expected_missing=3,shard_count=2,
            expected_plan_sha=sha(plan),expected_selection_sha=sha(master))


def test_bogus_sha_receipt_never_mints_recovered(tmp_path):
    master,plan,shards=case(tmp_path)
    p=shards/"shard-0/SHARD_PRIVATE.json"
    j=json.loads(p.read_text())
    j["candidates_private"][0]["retrieved_source_sha256"]="e"*64
    dump(p,j)
    with pytest.raises(ValueError,match="RECOVERY_SUCCESS_WITHOUT_RECEIPT"):
        run(master,plan,shards,tmp_path/"out",
            selection_count=4,expected_missing=3,shard_count=2,
            expected_plan_sha=sha(plan),expected_selection_sha=sha(master))
