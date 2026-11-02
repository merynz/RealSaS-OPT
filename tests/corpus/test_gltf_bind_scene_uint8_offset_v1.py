from __future__ import annotations

import base64
import json
import struct

from tools.corpus.gltf_bind_scene_v1 import SourceGLTF


def _align4(data: bytearray) -> None:
    while len(data) % 4:
        data.append(0)


def test_uint8_primitive_indices_are_widened_before_scene_offset(tmp_path):
    """Accessor storage width must not cap assembled scene-wide face indices."""
    blob = bytearray()
    views = []
    accessors = []

    def add_positions(count: int) -> int:
        _align4(blob)
        start = len(blob)
        for i in range(count):
            blob.extend(struct.pack("<fff", float(i), 0.0, 0.0))
        views.append({"buffer": 0, "byteOffset": start, "byteLength": count * 12})
        accessors.append({
            "bufferView": len(views) - 1,
            "componentType": 5126,
            "count": count,
            "type": "VEC3",
        })
        return len(accessors) - 1

    def add_uint8_triangle() -> int:
        start = len(blob)
        blob.extend(bytes((0, 1, 2)))
        views.append({"buffer": 0, "byteOffset": start, "byteLength": 3})
        accessors.append({
            "bufferView": len(views) - 1,
            "componentType": 5121,
            "count": 3,
            "type": "SCALAR",
        })
        return len(accessors) - 1

    p0 = add_positions(300)
    i0 = add_uint8_triangle()
    p1 = add_positions(3)
    i1 = add_uint8_triangle()

    encoded = base64.b64encode(bytes(blob)).decode("ascii")
    document = {
        "asset": {"version": "2.0"},
        "buffers": [{"byteLength": len(blob), "uri": "data:application/octet-stream;base64," + encoded}],
        "bufferViews": views,
        "accessors": accessors,
        "meshes": [
            {"primitives": [{"attributes": {"POSITION": p0}, "indices": i0}]},
            {"primitives": [{"attributes": {"POSITION": p1}, "indices": i1}]},
        ],
        "nodes": [{"mesh": 0}, {"mesh": 1}],
    }
    source = tmp_path / "uint8-offset.gltf"
    source.write_text(json.dumps(document))

    geometry = SourceGLTF(source).geometry()

    assert geometry["faces"] == [[0, 1, 2], [300, 301, 302]]
    assert len(geometry["vertices"]) == 303
