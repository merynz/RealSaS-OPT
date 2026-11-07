"""C4008 normalization is reversible and never invents anatomical truth."""
from __future__ import annotations

import hashlib
import json
import pytest

from tools.corpus.build_corpus4008_normalization_plan_v1 import (
    plan,point,four_by_four,
)


def fixture(tmp, lo=(-10.0,-4.0,1.0), hi=(6.0,8.0,13.0)):
    sha="a"*64
    p=tmp/"source_bounds.json"
    p.write_text(json.dumps({
        "schema":"RealSaS.Corpus4008MeasuredImportedSourceBounds.v1",
        "status":"IMPORTED_SOURCE_BOUNDS_ONLY",
        "canonical_asset_id":"asset_example",
        "candidate_id":"objaverse_animated_originals:"+sha,
        "source_sha256":sha,
        "source_space":"IMPORTED_BLENDER_SOURCE_WORLD",
        "source_world_min":list(lo),
        "source_world_max":list(hi),
        "source_meter_unit":"UNKNOWN",
        "anatomical_front":"UNKNOWN","anatomical_up":"UNKNOWN",
        "independent_source_truth_courted":False,
        "training_authorized":False,
    }))
    return p


def test_global_similarity_roundtrip_iris_and_tessa(tmp_path):
    p=fixture(tmp_path)
    original_sha=hashlib.sha256(p.read_bytes()).hexdigest()
    result=plan(p,expected_sha256=original_sha)
    assert result["status"]=="REVERSIBLE_PLAN_NUMERICALLY_CHECKED__NOT_SOURCE_PARITY"
    assert result["common_global_scale_for_all_views_and_poses"] is True
    assert result["per_view_renormalization_permitted"] is False
    assert result["source_anatomical_front"]=="UNKNOWN"
    assert result["source_meter_unit"]=="UNKNOWN"
    assert result["materialization_authorized"] is False
    w=result["source_world_to_iris"]
    inv=result["iris_to_source_world"]
    for pt in ((-10,-4,1),(6,8,13),(0,0,0),(4.9,3.3,10.6)):
        p2=point(w,pt)
        p3=point(inv,p2)
        assert max(abs(a-b) for a,b in zip(pt,p3))<1e-12
        p4=point(result["iris_to_tessa"],p2)
        assert max(abs(z) for z in p4)<.5000001 if pt!= (0,0,0) else True
    assert result["global_source_center"]==[-2.0,2.0,7.0]
    assert result["global_source_half_extent"]==8.0


def test_narrow_nonhumanoid_does_not_get_swapped_axes(tmp_path):
    p=fixture(tmp_path,lo=(-.01,-.00001,-100.),
              hi=(.02,.00002,200.))
    r=plan(p)
    assert r["source_world_to_iris"][1]==0
    assert r["source_world_to_iris"][4]==0
    assert r["source_world_to_iris"][8]==0
    assert r["source_anatomical_up"]=="UNKNOWN"
    assert r["source_world_to_iris"][0]==r["source_world_to_iris"][5]==r["source_world_to_iris"][10]


def test_source_bounds_receipt_sha_mismatch_closed(tmp_path):
    p=fixture(tmp_path)
    with pytest.raises(ValueError,match="BOUNDS_RECEIPT_SHA_DRIFT"):
        plan(p,expected_sha256="b"*64)


@pytest.mark.parametrize("mod,error",[
    ({"source_world_min":[5,0,0],"source_world_max":[0,1,1]},"BOUNDS_REVERSED"),
    ({"source_world_min":[1,1,1],"source_world_max":[1,1,1]},"DEGENERATE_GLOBAL_SOURCE_EXTENT"),
    ({"anatomical_front":"FRONT"},"UNAUTHORIZED_ORIENTATION_OR_UNIT"),
    ({"source_meter_unit":"METERS"},"UNAUTHORIZED_ORIENTATION_OR_UNIT"),
    ({"independent_source_truth_courted":True},"UNQUALIFIED_BOUNDS_RECEIPT"),
    ({"training_authorized":True},"UNQUALIFIED_BOUNDS_RECEIPT"),
    ({"source_world_min":[float("nan"),0,0]},"BOUNDS_NONFINITE"),
])
def test_bad_measurement_does_not_promote_teacher(tmp_path,mod,error):
    p=fixture(tmp_path)
    j=json.loads(p.read_text())
    j.update(mod)
    p.write_text(json.dumps(j))
    with pytest.raises(ValueError,match=error):
        plan(p)


def test_source_iris_tessa_matrix_convention(tmp_path):
    p=fixture(tmp_path)
    x=plan(p)
    predicted=four_by_four(x["iris_to_tessa"],x["source_world_to_iris"])
    assert max(abs(a-b) for a,b in zip(predicted,x["source_world_to_tessa"]))<1e-12
    assert x["source_mesh_vertex_roundtrip_courted"] is False
    assert x["source_skin_bind_and_weight_parity_courted"] is False
