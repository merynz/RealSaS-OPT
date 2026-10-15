import base64
import json
from pathlib import Path
import struct

import numpy as np

from tools.corpus.fit8_gltf_static_carrier_v1 import build_gltf_static_carrier_v1


def _align(blob: bytearray, alignment: int = 4) -> None:
    while len(blob) % alignment:
        blob.append(0)


def _write_minimal_two_material_gltf(path: Path) -> None:
    positions_a = np.asarray(
        [[-1.0, 0.0, -1.0], [1.0, 0.0, -1.0], [1.0, 0.0, 1.0]],
        dtype="<f4",
    )
    positions_b = np.asarray(
        [[-1.0, 0.0, -1.0], [1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]],
        dtype="<f4",
    )
    indices = np.asarray([0, 1, 2], dtype="<u2")
    blob = bytearray()
    views = []

    def add(raw: bytes, target=None):
        _align(blob)
        offset = len(blob)
        blob.extend(raw)
        row = {"buffer": 0, "byteOffset": offset, "byteLength": len(raw)}
        if target is not None:
            row["target"] = target
        views.append(row)
        return len(views) - 1

    pa = add(positions_a.tobytes(), 34962)
    ia = add(indices.tobytes(), 34963)
    pb = add(positions_b.tobytes(), 34962)
    ib = add(indices.tobytes(), 34963)
    uri = "data:application/octet-stream;base64," + base64.b64encode(bytes(blob)).decode("ascii")
    doc = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "buffers": [{"byteLength": len(blob), "uri": uri}],
        "bufferViews": views,
        "accessors": [
            {"bufferView": pa, "componentType": 5126, "count": 3, "type": "VEC3"},
            {"bufferView": ia, "componentType": 5123, "count": 3, "type": "SCALAR"},
            {"bufferView": pb, "componentType": 5126, "count": 3, "type": "VEC3"},
            {"bufferView": ib, "componentType": 5123, "count": 3, "type": "SCALAR"},
        ],
        "materials": [{"name": "A"}, {"name": "B"}],
        "meshes": [{
            "primitives": [
                {"attributes": {"POSITION": 0}, "indices": 1, "material": 0},
                {"attributes": {"POSITION": 2}, "indices": 3, "material": 1},
            ]
        }],
    }
    path.write_text(json.dumps(doc), encoding="utf-8")


def test_material_seam_render_vertices_merge_only_geometrically(tmp_path: Path):
    source = tmp_path / "seam.gltf"
    _write_minimal_two_material_gltf(source)
    manifest, arrays = build_gltf_static_carrier_v1(source)
    assert manifest["raw_render_vertex_count"] == 6
    assert manifest["geometric_vertex_count"] == 4
    assert manifest["face_count"] == 2
    assert manifest["material_region_count"] == 2
    assert manifest["mechanically_conflicted_position_count"] == 0
    assert manifest["render_seam_group_count"] == 2
    assert manifest["claims"]["material_id_source_exact"] is True
    assert manifest["claims"]["semantic_owner_authority"] is False
    assert arrays["vertices"].shape == (4, 3)
    assert arrays["faces"].shape == (2, 3)
    assert set(arrays["face_material_id"].tolist()) == {0, 1}
    assert np.all(arrays["face_owner_id"] == -1)
