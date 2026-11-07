import csv
import hashlib
import json

import pytest

from tools.corpus.triage_corpus4008_source_courts_v1 import triage


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def fixture(tmp):
    rows=[]
    for i in range(4):
        source=f"{i+1:064x}"
        rows.append({
            "canonical_asset_id":f"asset_{i+1:024x}",
            "source_sha256":source,
            "split":"FIT",
            "training_authorized":"False",
            "head_admission_count":"0",
            "functional_native_ik_fk":"UNKNOWN",
            "source_character_vs_prop":"UNKNOWN",
            "graphics_source_bound_2d_art_qualified":"UNKNOWN",
            "source_import_observation_status":"IMPORTED_SOURCE_OBSERVATIONS_ONLY",
            "mesh_object_count":"1",
            "armature_count":"1",
            "observed_bone_count":"42",
            "mesh_disconnected_components":"2",
            "mesh_boundary_edges":str(10 if i==1 else 0),
            "mesh_nonmanifold_edges":"0",
            "image_dependency_missing_count":str(2 if i==2 else 0),
            "skin_rows_bad_sum":str(7 if i==0 else 0),
            "skin_rows_zero_weights":"0",
            "skin_rows_sampled":"1200",
            "observed_ik_constraint_count":"0",
            "observed_driver_channel_count":"0",
            "observed_action_count":"2",
        })
    csv_path=tmp/"ORIGINAL_MATRIX.csv"
    with csv_path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,list(rows[0]))
        w.writeheader();w.writerows(rows)
    sp=tmp/"ORIGINAL_SUMMARY.json"
    sp.write_text(json.dumps({
        "schema":"RealSaS.Corpus4008Parallel1024ImportedSceneMasterLedger.v1",
        "training_authorized":False,
        "unique_source_candidates_measured":4,
        "typed_source_matrix_sha256":sha(csv_path),
    }))
    return sp,csv_path,rows


def test_risk_courts_prioritize_but_never_admit_truth(tmp_path):
    summary,source,_=fixture(tmp_path)
    r=triage(summary,source,tmp_path/"out",count=4)
    assert r["source_count"]==4
    assert r["review_warning_counts"]["IMPORTED_SKIN_WEIGHT_ROW_SUM_ANOMALY"]==1
    assert r["review_warning_counts"]["OPEN_OR_NONMANIFOLD_SOURCE_MESH"]==1
    assert r["review_warning_counts"]["EXTERNAL_TEXTURES_NOT_RESOLVED"]==1
    assert r["source_truth_rights_courted"]==0
    assert r["per_head_training_admitted"]==0
    assert not r["training_authorized"]


def test_original_summary_hash_tamper_rejected(tmp_path):
    summary,source,_=fixture(tmp_path)
    source.write_text(source.read_text().replace("1200","1201",1))
    with pytest.raises(ValueError,match="SOURCE_READBACK_SHA_OR_TRUTH"):
        triage(summary,source,tmp_path/"out",count=4)


def test_false_ik_truth_admission_rejected(tmp_path):
    summary,source,rows=fixture(tmp_path)
    rows[0]["functional_native_ik_fk"]="PASS"
    with source.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,list(rows[0]))
        w.writeheader();w.writerows(rows)
    p=json.loads(summary.read_text())
    p["typed_source_matrix_sha256"]=sha(source)
    summary.write_text(json.dumps(p))
    with pytest.raises(ValueError,match="FALSE_SOURCE_TRUTH_OR_HOLDOUT"):
        triage(summary,source,tmp_path/"out",count=4)
