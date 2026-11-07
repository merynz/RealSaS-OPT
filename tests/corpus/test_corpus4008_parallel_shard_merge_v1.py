from __future__ import annotations

import csv
import hashlib
import json

import pytest

from tools.corpus.merge_corpus4008_parallel_shards_v1 import merge


def sha(file):
    return hashlib.sha256(file.read_bytes()).hexdigest()


def make(tmp):
    root=tmp/"batches"
    canonical={}
    for shard in range(32):
        base=root/f"shard-{shard}"
        base.mkdir(parents=True)
        rows=[]
        for i in range(32):
            number=shard*32+i+1
            original=f"{number:064x}"
            aid=f"asset_{number:024x}"
            candidate="objaverse_animated_originals:"+original
            canonical[aid]={"candidate_id":candidate,"split":"FIT"}
            rows.append({
                "canonical_asset_id":aid,
                "candidate_id":candidate,
                "source_sha256":original,
                "source_format":".glb",
                "source_shard":str(shard),
                "split":"FIT",
                "training_authorized":"False",
                "head_admission_count":"0",
                "functional_native_ik_fk":"UNKNOWN",
                "source_character_vs_prop":"UNKNOWN",
                "graphics_source_bound_2d_art_qualified":"UNKNOWN",
                "import_wave":str(3 if i>=8 else (
                    1 if shard%4==0 else 2)),
                "source_import_observation_status":
                    "IMPORTED_SOURCE_OBSERVATIONS_ONLY",
                "armature_count":"1",
                "mesh_object_count":"2",
                "skin_rows_sampled":"100",
                "skin_rows_zero_weights":"20",
                "observed_action_count":"1",
                "observed_ik_constraint_count":"0",
                "image_dependency_missing_count":"0",
                "mesh_boundary_edges":"2",
            })
        csv_file=base/"SHARD_TYPED_OBSERVATIONS.csv"
        with csv_file.open("w",newline="") as f:
            writer=csv.DictWriter(f,list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        summary={
            "schema":"RealSaS.Corpus4008ParallelShardReadback.v1",
            "shard":shard,
            "prior_count":8,
            "additional_wave_count":24,
            "total_unique_source_count":32,
            "original_sha256_list":sorted(r["source_sha256"] for r in rows),
            "asset_id_list":sorted(r["canonical_asset_id"] for r in rows),
            "source_import_status_counts":{
                "IMPORTED_SOURCE_OBSERVATIONS_ONLY":32},
            "typed_observations_csv_sha256":sha(csv_file),
            "source_original_bytes_refetched":False,
            "full_import_receipts_sha_verified":True,
            "per_head_training_admitted_count":0,
            "training_authorized":False,
        }
        (base/"SHARD_READBACK_SUMMARY.json").write_text(json.dumps(summary))
    selection=tmp/"MASTER.json"
    selection.write_text(json.dumps(canonical))
    return selection,root


def test_full_1024_source_global_no_training_authority(tmp_path):
    selection,root=make(tmp_path)
    result=merge(selection,root,tmp_path/"out",expected_master_count=1024,
                 expected_master_sha=sha(selection))
    assert result["unique_source_candidates_measured"]==1024
    assert result["head_observation_candidate_counts_not_training"]["MIRA"]==1024
    assert result["head_training_admitted_counts"]["AXIS"]==0
    assert result["source_original_rights_courted"] is False
    assert result["materialization_authorized"] is False
    assert result["training_authorized"] is False


def test_cross_shard_duplicate_original_fails(tmp_path):
    selection,root=make(tmp_path)
    bad=root/"shard-1"/"SHARD_TYPED_OBSERVATIONS.csv"
    with bad.open() as f:
        rows=list(csv.DictReader(f))
    rows[0]["source_sha256"]=f"{1:064x}"
    with bad.open("w",newline="") as f:
        w=csv.DictWriter(f,list(rows[0]))
        w.writeheader();w.writerows(rows)
    summary=root/"shard-1"/"SHARD_READBACK_SUMMARY.json"
    d=json.loads(summary.read_text())
    d["typed_observations_csv_sha256"]=sha(bad)
    d["original_sha256_list"]=sorted(r["source_sha256"] for r in rows)
    summary.write_text(json.dumps(d))
    with pytest.raises(ValueError,match="CROSS_SHARD_SOURCE_IDENTITY_DUPLICATE"):
        merge(selection,root,tmp_path/"out",expected_master_count=1024,
              expected_master_sha=sha(selection))


def test_head_pass_promotion_is_fatal(tmp_path):
    selection,root=make(tmp_path)
    bad=root/"shard-31"/"SHARD_TYPED_OBSERVATIONS.csv"
    with bad.open() as f:rows=list(csv.DictReader(f))
    rows[4]["head_admission_count"]="1"
    with bad.open("w",newline="") as f:
        w=csv.DictWriter(f,list(rows[0]))
        w.writeheader();w.writerows(rows)
    summary=root/"shard-31"/"SHARD_READBACK_SUMMARY.json"
    d=json.loads(summary.read_text())
    d["typed_observations_csv_sha256"]=sha(bad)
    summary.write_text(json.dumps(d))
    with pytest.raises(ValueError,match="FALSE_TEACHER_PROMOTION"):
        merge(selection,root,tmp_path/"out",expected_master_count=1024,
              expected_master_sha=sha(selection))


def test_missing_shard_fails_instead_of_accepting_partial_batch(tmp_path):
    selection,root=make(tmp_path)
    (root/"shard-29"/"SHARD_READBACK_SUMMARY.json").unlink()
    with pytest.raises(ValueError,match="SHARD_NOT_READY_29"):
        merge(selection,root,tmp_path/"out",expected_master_count=1024,
              expected_master_sha=sha(selection))


def test_hash_invalidated_by_post_review_file_edit(tmp_path):
    selection,root=make(tmp_path)
    (root/"shard-4"/"SHARD_TYPED_OBSERVATIONS.csv").write_text("tampered\n")
    with pytest.raises(ValueError,match="SHARD_CSV_DRIFT"):
        merge(selection,root,tmp_path/"out",expected_master_count=1024,
              expected_master_sha=sha(selection))
