"""Synthetic-only quarantine tests; no private asset records in CI."""
import csv,hashlib,json
from pathlib import Path
import pytest
from tools.corpus.build_cross_corpus_identity_quarantine_v1 import run

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def csvw(path,items):
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,list(items[0]));w.writeheader();w.writerows(items)

def example(tmp_path,forgery=False):
    pairs=tmp_path/"pairs.csv"
    summary=tmp_path/"summary.json"
    routes=tmp_path/"routes.csv"
    csvw(pairs,[{
        "master_asset_id":"assetA","core_subject_id":"coreB",
        "identity_evidence":"96_BIT_SHA_PREFIX_CANDIDATE_ONLY",
        "priority":"P0_EVALUATION_HOLDOUT_SPLIT_RISK"}])
    summary.write_text(json.dumps({
        "schema":"RealSaS.CrossCorpusSourceIdentityCourt.v1",
        "training_authorized":False,"matched_sha_prefix_pairs":1,
        "pair_file_sha256":sha(pairs)}))
    rows=[{
        "global_record_key":key,"head":"IRIS_GEOMETRY","target":"surface",
        "training_authorized":"False","loss_mask_valid":"False",
        "negative_label_assigned":"False"
    } for key in ("MASTER3993:assetA","CORE370:coreB")]
    if forgery:rows[0]["loss_mask_valid"]="True"
    csvw(routes,rows)
    return pairs,summary,routes

def test_two_side_quarantine_never_mints_truth(tmp_path):
    pairs,summary,routes=example(tmp_path)
    res=run(pairs,summary,routes,tmp_path/"out",
            expected_pairs=1,expected_per_source=1,expected_route_rows=2)
    assert res["source_records_quarantined"]==2
    assert res["head_target_routes_quarantined"]==2
    assert res["protected_split_risk_target_routes"]==2
    assert res["training_authorized"] is False
    with (tmp_path/"out"/"CROSS_CORPUS_IDENTITY_BLOCKED_HEAD_TARGET_ROUTES_V1.csv").open() as f:
        output=list(csv.DictReader(f))
    assert all(r["training_authorized"]=="False"
        and r["loss_mask_valid"]=="False" for r in output)

def test_route_fake_pass_is_rejected(tmp_path):
    p,s,r=example(tmp_path,forgery=True)
    with pytest.raises(ValueError,match="ALREADY_PROMOTED_ROUTE"):
        run(p,s,r,tmp_path/"out",expected_pairs=1,expected_per_source=1,expected_route_rows=2)

def test_missing_related_core_half_rejected(tmp_path):
    p,s,r=example(tmp_path)
    csvw(r,[{
        "global_record_key":"MASTER3993:assetA","head":"IRIS_GEOMETRY",
        "target":"surface","training_authorized":"False",
        "loss_mask_valid":"False","negative_label_assigned":"False",
    }])
    with pytest.raises(ValueError,match="MISSING_OR_DUPLICATE_TARGET"):
        run(p,s,r,tmp_path/"out",expected_pairs=1,expected_per_source=1,expected_route_rows=1)

def test_forged_pair_receipt_rejected(tmp_path):
    p,s,r=example(tmp_path)
    p.write_text(p.read_text().replace("assetA","assetFAKE"))
    with pytest.raises(ValueError,match="PAIR_COURT_SHA"):
        run(p,s,r,tmp_path/"out",expected_pairs=1,expected_per_source=1,expected_route_rows=2)
