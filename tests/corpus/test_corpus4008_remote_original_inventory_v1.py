"""Private remote original-source metadata count, identity and abstention tests."""
import hashlib
import json

import pytest

from tools.corpus.build_corpus4008_remote_original_inventory_v1 import (
    COUNTS, make_inventory,
)


def _digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def fixture(tmp_path):
    master, selected, remote = [], {}, []
    i = 0
    for provider,n in COUNTS.items():
        for j in range(n):
            token=hashlib.sha256((provider+str(j)).encode()).hexdigest()
            aid="asset_test_"+str(i).zfill(4)
            qa=provider=="khronos_gltf_qa"
            row={
                "canonical_asset_id":aid,
                "candidate_id":provider+":"+token if not qa else provider+":"+token[:24],
                "source_registry_id":provider,
                "split":"QA_ONLY" if qa else "EXTERNAL_HOLDOUT" if i<169 else "FIT",
                "decision":"ADMIT",
            }
            master.append(row)
            if not qa:selected[aid]=row
            if i<2:
                remote.append({"Path":provider+"/"+token+"_"+token[:16]+".fbx",
                               "Size":1024+j,"IsDir":False})
            if i==1:
                remote.append({"Path":provider+"/extra/"+token+"_copy.fbx",
                               "Size":1025,"IsDir":False})
            i+=1
    m=tmp_path/"master.jsonl"
    m.write_text("\n".join(json.dumps(x) for x in master)+"\n")
    s=tmp_path/"selection.json"
    s.write_text(json.dumps(selected))
    r=tmp_path/"rclone.json"
    r.write_text(json.dumps(remote))
    return m,s,r


def call(files):
    m,s,r=files
    return make_inventory(m,s,r,expected_master_sha=_digest(m),
                          expected_selection_sha=_digest(s))


def test_4008_identity_full_source_inventory_and_no_training(tmp_path):
    doc=call(fixture(tmp_path))
    assert len(doc["subjects_private"])==4008
    summary=doc["summary"]
    assert summary["statuses"]=={
        "AMBIGUOUS_REMOTE_ORIGINAL_VARIANTS":1,
        "REMOTE_ORIGINAL_METADATA_MATCH__BYTES_NOT_VERIFIED":1,
        "REMOTE_ORIGINAL_NOT_FOUND":4006}
    assert summary["qa_only_count"]==15
    assert summary["historical_external_holdout_count"]==169
    assert summary["training_authorized"] is False
    assert summary["source_bytes_verified_count"]==0
    assert all(x["source_character_class"]=="UNKNOWN" for x in doc["subjects_private"])


def test_fail_closed_when_original_master_hash_drifts(tmp_path):
    m,s,r=fixture(tmp_path)
    with pytest.raises(ValueError,match="MASTER_MANIFEST_SHA"):
        make_inventory(m,s,r,expected_master_sha="0"*64,
                       expected_selection_sha=_digest(s))


def test_untrusted_remote_metadata_never_creates_source_bytes_truth(tmp_path):
    m,s,r=fixture(tmp_path)
    remote=json.loads(r.read_text())
    remote[0]["Size"]=-1
    r.write_text(json.dumps(remote))
    with pytest.raises(ValueError,match="SOURCE_REMOTE_SIZE"):
        call((m,s,r))
