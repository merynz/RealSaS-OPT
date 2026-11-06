"""Generic multi-primitive source UV inventory tests; no original teacher bytes."""
from collections import Counter
import numpy as np
import pytest
from tools.corpus import qualify_source_uv_inventory_v0 as inv

A = np.asarray([[0.02, 0.06], [0.20, 0.07], [0.04, 0.21]])
B = np.asarray([[0.60, 0.45], [0.82, 0.44], [0.72, 0.79]])


def test_uv_triangle_inventory_is_order_independent_and_multiset_sensitive():
    source = Counter({inv.uv_triangle_key(A): 1, inv.uv_triangle_key(B): 2})
    observed = np.stack([B[::-1], A[[1, 2, 0]], B[[1, 0, 2]]])
    report = inv.compare_uv_inventories(source, observed)
    assert report["matched"]
    assert report["original_triangle_count"] == 3
    assert report["imported_triangle_count"] == 3
    changed = observed.copy()
    changed[2, 2, 0] += 0.02
    report = inv.compare_uv_inventories(source, changed)
    assert not report["matched"]
    assert report["missing_gltf_triangle_uv_incidents"] == 1
    assert report["extra_imported_triangle_uv_incidents"] == 1


def test_scene_mesh_instances_expand_primitives_not_artist_groups(monkeypatch):
    doc = {
        "scenes": [{"nodes": [0, 3]}], "scene": 0,
        "nodes": [
            {"children": [1, 2]}, {"mesh": 0}, {"mesh": 1},
            {"mesh": 1}, {"mesh": 0}, # unused fifth node
        ],
        "meshes": [
            {"primitives": [
                {"attributes": {"POSITION": 1, "TEXCOORD_0": 0}, "indices": 2},
            ]},
            {"primitives": [
                {"attributes": {"POSITION": 4, "TEXCOORD_0": 3}, "indices": 5},
            ]},
        ],
    }
    def fake_accessor(_doc, _bin, index):
        return {
            0: A, 1: np.array([(0,0,0), (1,0,0), (0,1,0)]),
            2: np.asarray([[0], [1], [2]], dtype=np.uint16),
            3: B, 4: np.array([(0,0,0), (1,0,0), (0,1,0)]),
            5: np.asarray([[0], [1], [2]], dtype=np.uint16),
        }[index]
    monkeypatch.setattr(inv, "glb_document", lambda _path: (doc, b""))
    monkeypatch.setattr(inv, "accessor", fake_accessor)
    inventory, primitive_rows = inv.original_gltf_uv_inventory("fixture.glb")
    assert len(primitive_rows) == 2
    assert inventory[inv.uv_triangle_key(A)] == 1
    assert inventory[inv.uv_triangle_key(B)] == 2
    assert inv.compare_uv_inventories(inventory, [A, B, B])["matched"]


def test_invalid_shared_scene_node_cannot_inflate_observed_source_count():
    nodes = [{"children": [2]}, {"children": [2]}, {"mesh": 0}]
    with pytest.raises(ValueError, match="NODE_SHARED_PARENT"):
        inv._scene_mesh_instance_counts({
            "scenes": [{"nodes": [0, 1]}],
            "nodes": nodes,
            "meshes": [{"primitives": []}],
        })


def test_active_scene_suppresses_unseen_scene_meshes():
    assert inv._scene_mesh_instance_counts({
        "scene": 0,
        "scenes": [{"nodes": [0]}, {"nodes": [1]}],
        "nodes": [{"mesh": 0}, {"mesh": 1}],
        "meshes": [{"primitives": []}, {"primitives": []}],
    }) == Counter({0: 1})


@pytest.mark.parametrize("uv", [
    np.zeros((2, 2)),
    np.asarray([[0., 0.], [float("nan"), 0.], [1., 1.]]),
    np.asarray([[-0.02, 0.], [0., 0.], [1., 1.]]),
])
def test_bad_uv_triangle_rejected(uv):
    with pytest.raises(ValueError, match="TRIANGLE_UV"):
        inv.uv_triangle_key(uv)
