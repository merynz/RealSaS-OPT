"""Source cluster overlays accept only full original-source SHA byte proof."""
import csv,hashlib,json
from pathlib import Path
import pytest
from tools.corpus.build_verified_cross_corpus_source_clusters_v2 import run

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def write_csv(path,items):
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,list(items[0]));w.writeheader();w.writerows(items)

def case(tmp_path,prefix_only=False,wrong_digest=False):
    full=hashlib.sha256(b"synthetic-original-source").hexdigest()
    p=tmp_path/"pairs.csv";s=tmp_path/"summary.json"
    r=tmp_path/"roster.csv";q=tmp_path/"quarantine.csv"
    write_csv(p,[{
        "master_asset_id":"ma","core_subject_id":"co",
        "master_original_split":"EXTERNAL_HOLDOUT",
        "identity_evidence":"96_BIT_SHA_PREFIX_CANDIDATE_ONLY"
           if prefix_only else "FULL_SHA256_BYTE_VERIFIED",
        "core_source_sha256_historical":full,
        "real_source_sha256":"0"*64 if wrong_digest else full}])
    s.write_text(json.dumps({
        "schema":"RealSaS.CrossCorpusSourceIdentityCourt.v1",
        "training_authorized":False,
        "fully_byte_sha_verified_overlap_count":0 if prefix_only else 1,
        "prefix_overlap_requires_full_source_rehash_count":1 if prefix_only else 0,
        "pair_file_sha256":sha(p)}))
    write_csv(r,[
        {"global_record_key":"MASTER3993:ma","namespace":"MASTER3993",
         "split_state":"EXTERNAL_HOLDOUT"},
        {"global_record_key":"CORE370:co","namespace":"CORE370",
         "split_state":"CORE370_SPLIT_REVALIDATION_REQUIRED"}])
    write_csv(q,[{
        "global_record_key":key,"head":"IRIS_GEOMETRY","target":"surface",
        "training_authorized":"False","loss_mask_valid":"False"} for key
        in ("MASTER3993:ma","CORE370:co")])
    return p,s,r,q

def check(tmp_path,paths):
    return run(*paths,tmp_path/"out",expected_pairs=1,
               expected_routes_per_source=1,expected_roster_count=2)

def test_proven_same_source_keeps_two_records_and_hard_holdout(tmp_path):
    paths=case(tmp_path)
    result=check(tmp_path,paths)
    assert result["verified_source_sha256_clusters"]==1
    assert result["cross_corpus_source_members"]==2
    assert result["protected_identity_clusters"]==1
    assert result["blocked_head_target_routes"]==2
    assert result["unique_usable_characters_certified"] is False
    assert result["training_authorized"] is False
    with (tmp_path/"out"/"VERIFIED_CROSS_CORPUS_SOURCE_IDENTITY_CLUSTERS_V2.csv").open() as f:
        row=list(csv.DictReader(f))[0]
    assert row["source_group_policy"]=="EXTERNAL_HOLDOUT_HARD_BLOCK"

def test_prefix_only_disallowed(tmp_path):
    paths=case(tmp_path,prefix_only=True)
    with pytest.raises(ValueError,match="REQUIRED_FULL_BYTE_SHA_COURT"):
        check(tmp_path,paths)

def test_spoofed_full_sha_rejected(tmp_path):
    paths=case(tmp_path,wrong_digest=True)
    with pytest.raises(ValueError,match="UNVERIFIED_SOURCE_BYTES"):
        check(tmp_path,paths)

def test_missing_core_quarantine_forbidden(tmp_path):
    paths=case(tmp_path)
    q=paths[3]
    write_csv(q,[{"global_record_key":"MASTER3993:ma","head":"IRIS_GEOMETRY",
                  "target":"surface","training_authorized":"False",
                  "loss_mask_valid":"False"}])
    with pytest.raises(ValueError,match="QUARANTINE_ROUTE_CENSUS"):
        check(tmp_path,paths)

def test_false_training_mask_rejected(tmp_path):
    paths=case(tmp_path)
    q=paths[3]
    a=list(csv.DictReader(q.open()));a[0]["training_authorized"]="True"
    write_csv(q,a)
    with pytest.raises(ValueError,match="PROMOTED_QUARANTINE_TARGET"):
        check(tmp_path,paths)
