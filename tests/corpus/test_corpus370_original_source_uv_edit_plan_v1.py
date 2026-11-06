"""Source-only UV tile edit plan contract and adversarial tests."""
import numpy as np
import pytest
from tools.corpus.compile_corpus370_original_source_uv_edit_plan_v1 import (
    choose_uv_tiles,uv_centroid_tile_census,require
)


def test_original_raw_uv_tile_rank_is_deterministic_without_render_pixels():
    census={(6,6):292,(6,7):246,(7,6):63,(7,4):44,(0,5):10}
    r=choose_uv_tiles(census)
    assert [(x["uv_tile_u"],x["uv_tile_v"],
             x["source_triangle_uv_centroid_count"]) for x in r]==[
                 (6,6,292),(6,7,246),(7,6,63)]


def test_equal_counts_use_uv_y_then_x_not_anatomical_names():
    census={(3,4):18,(2,4):18,(5,2):18,(7,7):4}
    chosen=choose_uv_tiles(census)
    assert [(x["uv_tile_u"],x["uv_tile_v"]) for x in chosen]==[
        (5,2),(2,4),(3,4)]


def test_low_support_tiles_cannot_be_promoted():
    with pytest.raises(ValueError,match="SOURCE_UV_TILES_INSUFFICIENT"):
        choose_uv_tiles({(1,2):15,(3,4):100,(6,7):100})


def test_invalid_minimum_rejected():
    with pytest.raises(ValueError,match="SOURCE_TILE_SELECTION_POLICY"):
        choose_uv_tiles({(1,2):100},count=0)


def triangle_document():
    # Three fully valid original GLB source triangles UV binned to u=6,v=6,
    # with distinct vertices and decoded raw little-endian buffer views.
    pos=np.array([[0,0,0],[1,0,0],[0,1,0],
                  [0,0,1],[1,0,1],[0,1,1]],dtype="<f4")
    uv=np.array([[.80,.80],[.83,.82],[.85,.82],
                 [.81,.81],[.82,.84],[.84,.82]],dtype="<f4")
    indices=np.array([0,1,2,3,4,5],dtype="<u2")
    offsets=[0,len(pos.tobytes()),len(pos.tobytes())+len(uv.tobytes())]
    chunks=[pos.tobytes(),uv.tobytes(),indices.tobytes()]
    doc={
        "asset":{"version":"2.0"},
        "meshes":[{"primitives":[{"mode":4,"indices":2,
            "attributes":{"POSITION":0,"TEXCOORD_0":1}}]}],
        "accessors":[
            {"bufferView":0,"componentType":5126,"count":6,"type":"VEC3"},
            {"bufferView":1,"componentType":5126,"count":6,"type":"VEC2"},
            {"bufferView":2,"componentType":5123,"count":6,"type":"SCALAR"}],
        "bufferViews":[
            {"buffer":0,"byteOffset":offsets[0],"byteLength":len(chunks[0])},
            {"buffer":0,"byteOffset":offsets[1],"byteLength":len(chunks[1])},
            {"buffer":0,"byteOffset":offsets[2],"byteLength":len(chunks[2])}],
    }
    return doc,b"".join(chunks)


def test_raw_source_mesh_uv_census_without_semantic_or_image_input():
    doc,buffer=triangle_document()
    census,count=uv_centroid_tile_census(doc,memoryview(buffer))
    assert count==2 and census=={(6,6):2}


@pytest.mark.parametrize("bad",["missing_uv","not_triangles","invalid_uv","invalid_index"])
def test_malformed_source_triangles_abstain(bad):
    doc,data=triangle_document()
    if bad=="missing_uv":
        del doc["meshes"][0]["primitives"][0]["attributes"]["TEXCOORD_0"]
    if bad=="not_triangles":
        doc["meshes"][0]["primitives"][0]["mode"]=1
    if bad=="invalid_uv":
        doc["accessors"][1]["normalized"]=True
    if bad=="invalid_index":
        doc["accessors"][2]["byteOffset"]=4
    with pytest.raises((ValueError,IndexError)):
        uv_centroid_tile_census(doc,memoryview(data))


def test_no_renderer_image_or_artist_ownership_required_for_tile_selection():
    r=choose_uv_tiles({(6,6):292,(6,7):246,(7,6):63})
    assert set(r[0])=={
        "uv_tile_u","uv_tile_v","source_triangle_uv_centroid_count"}
    assert "source_artist_drawable_owner" not in r[0]
