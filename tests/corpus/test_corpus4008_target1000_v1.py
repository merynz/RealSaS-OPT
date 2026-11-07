"""Strict quota integrity: clips, derivative styles and stale admissions never count."""
import json
from pathlib import Path

import pytest

from tools.corpus.audit_corpus4008_target1000_v1 import (
    HEAD_TARGETS, audit,
)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))
    return path


def fixture(tmp):
    manifest = write_json(tmp / "selection.json", {
        "asset_a": {
            "canonical_asset_id": "asset_a",
            "candidate_id": "objaverse_animated_originals:" + "a" * 64,
            "split": "FIT",
        },
        "asset_hold": {
            "canonical_asset_id": "asset_hold",
            "candidate_id": "objaverse_animated_originals:" + "b" * 64,
            "split": "EXTERNAL_HOLDOUT",
        },
    })
    rs = tmp / "receipts"
    es = tmp / "evidence"
    rs.mkdir()
    es.mkdir()
    return manifest, rs, es


def receipt(tmp, *, aid="asset_a", sha="a"*64, split="FIT",
            head="IRIS_GEOMETRY", cluster="cluster_unique_abcdef"):
    r = {
        "schema": "RealSaS.IndependentHeadAdmissionEvidence.v1",
        "canonical_asset_id": aid,
        "candidate_id": "objaverse_animated_originals:" + sha,
        "source_sha256": sha,
        "head": head,
        "split": split,
        "character_class": "CHARACTER",
        "commercial_ml_rights": "INDEPENDENTLY_VERIFIED",
        "source_scene_court": "PASS",
        "visual_court": "PASS",
        "head_independent_court": "PASS",
        "cross_split_lineage_court": "PASS",
        "source_identity_cluster": cluster,
        "morphology_stratum": "NONHUMANOID",
        "target_truth": {k: "PASS" for k in HEAD_TARGETS[head]},
        "independent_evidence_files": {},
        "training_authorized": False,
    }
    for kind in ("source", "rights", "split", "character", "visual", "head"):
        p = tmp / "evidence" / (aid + "_" + head + "_" + kind + ".json")
        write_json(p, {
            "schema": "RealSaS.IndependentSourceCourt.v1",
            "asset_id": aid, "source_sha256": sha,
            "court_kind": kind, "status": "PASS",
            "independent_review": True, "training_authorized": False,
        })
        import hashlib
        r["independent_evidence_files"][kind] = {
            "path": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        }
    return r


def test_no_independent_admission_is_not_a_pass(tmp_path):
    sel, rs, es = fixture(tmp_path)
    result = audit(sel, rs, es, tmp_path / "out.json",
                   quota=1000, expected_master_count=2)
    assert result["master_count"] == 2
    assert all(v["missing_to_quota"] == 1000
               and v["evidence_receipt_supported_distinct_fit_clusters"] == 0
               for v in result["per_head"].values())
    assert result["model_training_authorized"] is False


def test_one_verified_receipt_only_counts_one_head(tmp_path):
    sel, rs, es = fixture(tmp_path)
    write_json(rs / "one.json", receipt(tmp_path))
    result = audit(sel, rs, es, tmp_path / "out.json",
                   quota=1000, expected_master_count=2)
    assert result["per_head"]["IRIS_GEOMETRY"][
        "evidence_receipt_supported_distinct_fit_clusters"] == 1
    assert result["per_head"]["IRIS_GEOMETRY"]["missing_to_quota"] == 999
    assert result["per_head"]["TESSA"]["missing_to_quota"] == 1000


def test_protected_holdout_never_counts_in_fit_quota(tmp_path):
    sel, rs, es = fixture(tmp_path)
    write_json(rs / "hold.json", receipt(
        tmp_path, aid="asset_hold", sha="b"*64, split="EXTERNAL_HOLDOUT",
        cluster="cluster_holdout_abcdef",
    ))
    result = audit(sel, rs, es, tmp_path / "out.json",
                   quota=1, expected_master_count=2)
    assert all(v["evidence_receipt_supported_distinct_fit_clusters"] == 0
               for v in result["per_head"].values())


@pytest.mark.parametrize("mutation", [
    "commercial_ml_rights",
    "character_class",
    "split",
    "source_sha256",
    "target_truth",
    "source_identity_cluster",
    "training_authorized",
    "court_missing",
    "court_hash_tamper",
])
def test_wrong_or_missing_authority_never_counts(tmp_path, mutation):
    sel, rs, es = fixture(tmp_path)
    rec = receipt(tmp_path)
    if mutation == "commercial_ml_rights":
        rec["commercial_ml_rights"] = "LEGACY_PASS"
    elif mutation == "character_class":
        rec["character_class"] = "PROP"
    elif mutation == "split":
        rec["split"] = "TUNE"
    elif mutation == "source_sha256":
        rec["source_sha256"] = "c"*64
    elif mutation == "target_truth":
        rec["target_truth"].pop("CAMERA_OBSERVATION")
    elif mutation == "source_identity_cluster":
        rec["source_identity_cluster"] = None
    elif mutation == "training_authorized":
        rec["training_authorized"] = True
    elif mutation == "court_missing":
        rec["independent_evidence_files"].pop("rights")
    elif mutation == "court_hash_tamper":
        (es / "asset_a_IRIS_GEOMETRY_rights.json").write_text(
            '{"status":"PASS","tampered":true}')
    write_json(rs / "bad.json", rec)
    res = audit(sel, rs, es, tmp_path / "out.json", quota=1,
                expected_master_count=2)
    assert res["per_head"]["IRIS_GEOMETRY"][
        "evidence_receipt_supported_distinct_fit_clusters"] == 0
    assert res["rejected_receipt_reason_counts"]


def test_quota_uses_cluster_not_asset_row_or_clip_count(tmp_path):
    sel, rs, es = fixture(tmp_path)
    data = json.loads(sel.read_text())
    data["asset_b"] = {
        "canonical_asset_id": "asset_b",
        "candidate_id": "objaverse_animated_originals:" + "c"*64,
        "split": "FIT",
    }
    sel.write_text(json.dumps(data))
    write_json(rs/"one.json", receipt(tmp_path))
    write_json(rs/"two.json", receipt(
        tmp_path, aid="asset_b", sha="c"*64, cluster="cluster_unique_abcdef"))
    r = audit(sel, rs, es, tmp_path / "out.json", quota=2,
              expected_master_count=3)
    assert r["per_head"]["IRIS_GEOMETRY"]["missing_to_quota"] == 1
