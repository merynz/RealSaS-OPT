from __future__ import annotations

import hashlib
import json

import pytest

from tools.corpus.build_corpus4008_normalization_plan_v1 import plan
from tools.corpus.reconcile_corpus4008_parallel_normalization_v1 import (
    reconcile,
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path,j):
    path.write_text(json.dumps(j,sort_keys=True))
    return path


def make(tmp_path):
    root=tmp_path/"eight"
    master={}
    for shard in range(32):
        group=shard//4
        folder=root/f"group-{group}"
        folder.mkdir(parents=True,exist_ok=True)
        value=shard+1
        source=f"{value:064x}"
        aid=f"asset_{value:024x}"
        candidate="objaverse_animated_originals:"+source
        master[aid]={"candidate_id":candidate,"split":"FIT"}
        bounds=dump(folder/f"shard-{shard}.BOUNDS.json",{
            "schema":"RealSaS.Corpus4008MeasuredImportedSourceBounds.v1",
            "status":"IMPORTED_SOURCE_BOUNDS_ONLY",
            "canonical_asset_id":aid,
            "candidate_id":candidate,
            "source_sha256":source,
            "source_format":".glb",
            "source_space":"IMPORTED_BLENDER_SOURCE_WORLD",
            "source_world_min":[-3.,-1.,-2.],
            "source_world_max":[6.,4.,5.],
            "mesh_object_count":1,
            "mesh_vertex_count":100,
            "source_meter_unit":"UNKNOWN",
            "anatomical_front":"UNKNOWN",
            "anatomical_up":"UNKNOWN",
            "independent_source_truth_courted":False,
            "training_authorized":False,
        })
        norm=plan(bounds)
        dump(folder/f"shard-{shard}.NORMALIZATION.json",norm)
        dump(folder/f"shard-{shard}.SELECTION.json",{
            "schema":"RealSaS.Corpus4008ParallelNormalizationPilotSelection.v1",
            "shard":shard,"materialization_authorized":False,
            "training_authorized":False,
            "candidate":{
                "canonical_asset_id":aid,"candidate_id":candidate,
                "source_sha256":source,"source_ext":".glb","split":"FIT",
            },
        })
        # Build status incrementally for all four source records per group.
        with (folder/"STATUS.jsonl").open("a") as f:
            f.write(json.dumps({
                "shard":shard,
                "canonical_asset_id":aid,
                "original_sha256":source,
                "status":"SOURCE_BBOX_REVERSIBLE_PLAN_ONLY",
                "source_mesh_vertex_roundtrip_courted":False,
                "source_skin_bind_parity_courted":False,
                "materialization_authorized":False,
                "training_authorized":False,
            })+"\n")
    selection=dump(tmp_path/"selection.json",master)
    return root,selection


def test_32_replays_are_immutable_and_nonadmitting(tmp_path):
    root,selection=make(tmp_path)
    result=reconcile(root,selection,tmp_path/"out",
                     expected_master_count=32,
                     expected_master_sha=sha(selection))
    assert result["normalized_unique_original_sources"]==32
    assert result["status"].startswith("32_SOURCE")
    assert result["max_bbox_corner_relative_replay_error"]<=1e-6
    assert result["training_authorized"] is False
    assert result["source_skin_bind_parity_courted"] is False


def test_authoring_orientation_cannot_be_fabricated(tmp_path):
    root,selection=make(tmp_path)
    p=root/"group-0/shard-1.BOUNDS.json"
    j=json.loads(p.read_text())
    j["anatomical_up"]="Z_UP"
    dump(p,j)
    with pytest.raises(ValueError,match="ORIGINAL_BOUNDS_PROVENANCE"):
        reconcile(root,selection,tmp_path/"out",
                  expected_master_count=32,expected_master_sha=sha(selection))


def test_prepared_plan_sha_cannot_be_derived_from_different_bounds(tmp_path):
    root,selection=make(tmp_path)
    p=root/"group-2/shard-8.BOUNDS.json"
    j=json.loads(p.read_text())
    j["source_world_max"][0]+=1.
    dump(p,j)
    with pytest.raises(ValueError,match="SOURCE_BOUND_REPLAY_DRIFT"):
        reconcile(root,selection,tmp_path/"out",
                  expected_master_count=32,expected_master_sha=sha(selection))


def test_repeat_original_sha_across_shards_is_forbidden(tmp_path):
    root,selection=make(tmp_path)
    path=root/"group-1/shard-5.STATUS.jsonl"
    # There is no per-shard STATUS file: duplicate is in common group list.
    groupfile=root/"group-1/STATUS.jsonl"
    rows=[json.loads(x) for x in groupfile.read_text().splitlines()]
    rows[1]["original_sha256"]=rows[0]["original_sha256"]
    groupfile.write_text("\n".join(json.dumps(x) for x in rows)+"\n")
    with pytest.raises(ValueError,match="DUPLICATE_NORMALIZED_SOURCE"):
        reconcile(root,selection,tmp_path/"out",
                  expected_master_count=32,expected_master_sha=sha(selection))


def test_missing_group_fails_closed(tmp_path):
    root,selection=make(tmp_path)
    (root/"group-7/STATUS.jsonl").unlink()
    with pytest.raises(ValueError,match="GROUP_STATUS_MISSING"):
        reconcile(root,selection,tmp_path/"out",
                  expected_master_count=32,expected_master_sha=sha(selection))
