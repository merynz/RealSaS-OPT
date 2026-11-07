from __future__ import annotations

import gzip
import hashlib
import json

import pytest

from tools.corpus.reconcile_corpus4008_wave1_source_rig_matrix_v1 import (
    reconcile, SUCCESS, PLAN_SCHEMA, BATCH_SCHEMA, RECEIPT_SCHEMA,
)


def make(tmp):
    root=tmp/"shard-0"
    receipts=root/"receipts"
    receipts.mkdir(parents=True)
    a="asset_000000000000000000000001"
    b="asset_000000000000000000000002"
    sha_a="a"*64
    sha_b="b"*64
    entries=[
        {"canonical_asset_id":aid,
         "candidate_id":"src:"+sha,
         "source_sha256":sha,
         "source_ext":".fbx",
         "source_byte_count":1024,
         "training_authorized":False}
        for aid,sha in ((a,sha_a),(b,sha_b))
    ]
    (root/"PLAN.json").write_text(json.dumps({
        "schema":PLAN_SCHEMA,"shard_index":0,
        "model_training_authorized":False,
        "sample_size":2,"entries":entries,
    }))
    statuses=[
        {"asset_id":a,"source_sha256":sha_a,"status":SUCCESS,
         "truth_admitted":False,"training_authorized":False},
        {"asset_id":b,"source_sha256":sha_b,"status":"SOURCE_IMPORT_EXCEPTION",
         "truth_admitted":False,"training_authorized":False},
    ]
    (root/"RESULTS.jsonl").write_text(
        "".join(json.dumps(x)+"\n" for x in statuses))
    (root/"BATCH_STATUS.json").write_text(json.dumps({
        "schema":BATCH_SCHEMA,"shard":0,"selected_source_count":2,
        "status_counts":{SUCCESS:1,"SOURCE_IMPORT_EXCEPTION":1},
        "truth_admitted":0,"training_authorized":False,
    }))
    n={
      "mesh_object_count":1,"source_scene_node_count":2,"armature_count":1,
      "vertex_count":123,"polygon_count":101,
      "mesh_boundary_edges":5,"mesh_nonmanifold_edges":0,
      "mesh_disconnected_components":2,"skin_rows_sampled":123,
      "skin_rows_zero_weights":0,"skin_rows_bad_sum":0,
      "bone_parented_mesh_candidates":1,
      "image_dependency_missing_count":0,"material_count":2,"image_count":1,
      "observed_bone_count":42,"observed_ik_constraint_count":3,
      "observed_driver_channel_count":1,"observed_action_curve_count":100,
      "observed_action_count":2,"observed_nla_strip_count":0,
      "observed_nondeform_bone_count":12,
    }
    detail={
      "schema":RECEIPT_SCHEMA,
      "status":"FULL_IMPORTED_SOURCE_SCENE_AND_CONTROL_CENSUS__NO_SOURCE_TRUTH_COURTS",
      "source_sha256":sha_a,"asset_id":a,
      "candidate_id":"src:"+sha_a,"source_format":".fbx",
      "numeric_census":n,
      "source_character_vs_prop":"UNKNOWN",
      "functional_native_ik_fk":"UNKNOWN",
      "commercial_ml_rights_courted":False,
      "training_authorized":False,
      "trainable_model_family_count":0,
      "2d_art_visual_quality_courted":False,
      "target_family_observation_routes":{
        "IRIS_GEOMETRY":"IMPORTED_MESH_AND_TOPOLOGY_OBSERVATION",
        "IRIS_SEMANTIC":"SOURCE_GROUP_HINTS_ONLY_NOT_LABEL",
        "IRIS_GRAPHICS":"MATERIAL_AND_TEXTURE_CENSUS_NOT_2D_ART",
        "TESSA":"SOURCE_SURFACE_PLUS_RIG_CANDIDATE",
        "AXIS":"IMPORTED_RIG_MOTION_CONTROL_HINTS_ONLY",
        "MIRA":"VERTEX_GROUP_WEIGHT_CANDIDATE_NOT_VALID_BIND",
      },
    }
    path=receipts/(a+".full.source.json.gz")
    path.write_bytes(gzip.compress(json.dumps(detail).encode(),mtime=0))
    (receipts/(a+".summary.json")).write_text(json.dumps({
        "numeric_census":n,
        "receipt_gz_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
    }))
    return root,detail


def test_reconcile_typed_coverage_never_claims_source_or_training_truth(tmp_path):
    root,_=make(tmp_path)
    r=reconcile(tmp_path,tmp_path/"out",expected_shards=(0,))
    assert r["sampled_original_source_count"]==2
    assert r["source_import_status"]=={
        SUCCESS:1,"SOURCE_IMPORT_EXCEPTION":1}
    assert r["per_head_import_observation_candidates_not_qualified"][
        "IRIS_GRAPHICS"]==1
    assert r["source_observation_counts"]["OBSERVED_IMPORTED_IK_SOURCE_ROWS"]==1
    assert r["per_head_training_admitted"]["AXIS"]==0
    assert r["training_authorized"] is False
    import csv
    with (tmp_path/"out/WAVE1_SOURCE_RIG_SCENE_TYPED_MATRIX_V1.csv").open() as f:
        rows=list(csv.DictReader(f))
    assert rows[0]["functional_native_ik_fk"]=="UNKNOWN"
    assert rows[0]["graphics_source_bound_2d_art_qualified"]=="UNKNOWN"
    assert rows[1]["source_import_observation_status"]=="SOURCE_IMPORT_EXCEPTION"


def test_forged_original_source_sha_causes_strict_rejection(tmp_path):
    root,detail=make(tmp_path)
    path=root/"receipts"/(detail["asset_id"]+".full.source.json.gz")
    detail["source_sha256"]="c"*64
    path.write_bytes(gzip.compress(json.dumps(detail).encode(),mtime=0))
    p=root/"receipts"/(detail["asset_id"]+".summary.json")
    data=json.loads(p.read_text())
    data["receipt_gz_sha256"]=hashlib.sha256(path.read_bytes()).hexdigest()
    p.write_text(json.dumps(data))
    with pytest.raises(ValueError,match="DETAILED_LINEAGE"):
        reconcile(tmp_path,tmp_path/"out",expected_shards=(0,))


def test_missing_gzip_receipt_is_not_import_success(tmp_path):
    root,detail=make(tmp_path)
    (root/"receipts"/(detail["asset_id"]+".full.source.json.gz")).unlink()
    with pytest.raises(ValueError,match="SOURCE_GZIP_RECEIPT_SIZE"):
        reconcile(tmp_path,tmp_path/"out",expected_shards=(0,))


def test_forged_training_permission_is_rejected(tmp_path):
    root,detail=make(tmp_path)
    path=root/"receipts"/(detail["asset_id"]+".full.source.json.gz")
    detail["training_authorized"]=True
    path.write_bytes(gzip.compress(json.dumps(detail).encode(),mtime=0))
    p=root/"receipts"/(detail["asset_id"]+".summary.json")
    data=json.loads(p.read_text())
    data["receipt_gz_sha256"]=hashlib.sha256(path.read_bytes()).hexdigest()
    p.write_text(json.dumps(data))
    with pytest.raises(ValueError,match="FALSE_SOURCE_OR_HEAD_ADMISSION"):
        reconcile(tmp_path,tmp_path/"out",expected_shards=(0,))
