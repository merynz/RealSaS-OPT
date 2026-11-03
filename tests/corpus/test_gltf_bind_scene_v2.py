"""Synthetic exact-bind glTF regression; no Blender, source or model oracle."""
import base64
import json
from pathlib import Path

import numpy as np
import pytest

from tools.corpus.gltf_bind_scene_v1 import SourceGLTF
from tools.corpus.gltf_bind_scene_v2 import SourceGLTFBindRestV2


def fixture(path: Path, *, joint_y=1.0, bind_y=2.0, conflicting_skin=False):
    binary = bytearray()
    views = []
    accessors = []

    def add(array, component_type, shape):
        nonlocal binary
        value = np.asarray(array)
        data = value.tobytes()
        while len(binary) % 4:
            binary += b"\\x00"
        views.append({"buffer": 0, "byteOffset": len(binary), "byteLength": len(data)})
        binary += data
        accessors.append({"bufferView": len(views)-1,
                          "componentType": component_type,
                          "count": len(value), "type": shape})
        return len(accessors)-1

    positions = add(np.array([[0,0,0],[1,0,0],[0,1,0]], np.float32), 5126, "VEC3")
    indices = add(np.array([[0],[1],[2]], np.uint16), 5123, "SCALAR")
    ibm = np.eye(4, dtype=np.float32)
    ibm[1,3] = -bind_y
    inverse = add(ibm.T.reshape(1,16), 5126, "MAT4")
    skins = [{"joints": [0], "inverseBindMatrices": inverse}]
    nodes = [
        {"name": "Joint", "translation": [0, joint_y, 0], "children": [1]},
        {"name": "Accessory", "mesh": 0, "translation": [0, .2, 0]},
        {"name": "Body", "mesh": 1, "skin": 0},
    ]
    if conflicting_skin:
        wrong = np.eye(4, dtype=np.float32)
        wrong[1,3] = -bind_y - .5
        second = add(wrong.T.reshape(1,16), 5126, "MAT4")
        skins.append({"joints": [0], "inverseBindMatrices": second})
        nodes.append({"name": "OtherBody", "mesh": 1, "skin": 1})
    doc = {
        "asset": {"version": "2.0"},
        "buffers": [{"byteLength": len(binary),
                     "uri": "data:application/octet-stream;base64," + base64.b64encode(binary).decode()}],
        "bufferViews": views, "accessors": accessors,
        "meshes": [{"primitives": [{"attributes": {"POSITION": positions}, "indices": indices}]},
                   {"primitives": [{"attributes": {"POSITION": positions}, "indices": indices}]}],
        "nodes": nodes, "skins": skins, "scenes": [{"nodes": [0, 2]}], "scene": 0,
    }
    path.write_text(json.dumps(doc))
    return path


def test_unskinned_joint_child_moves_from_default_to_inverse_bind_rest(tmp_path):
    src = fixture(tmp_path / "source.gltf", joint_y=1.0, bind_y=2.0)
    before = np.asarray(SourceGLTF(src).geometry()["vertices"])
    after_doc = SourceGLTFBindRestV2(src).geometry()
    after = np.asarray(after_doc["vertices"])
    # glTF Y becomes Blender Z; the attachment receives exactly +1 unit.
    np.testing.assert_allclose(after[:3] - before[:3], np.tile([0, 0, 1.], (3, 1)), atol=1e-7)
    np.testing.assert_array_equal(after[3:], before[3:])  # skinned body untouched
    assert len(after_doc["bind_rest_attachments"]) == 1
    assert after_doc["reader_schema"] == "RealSaS.SourceGLTFBindRestReader.v2"


@pytest.mark.parametrize("default_joint", [0.0, 1.0, -1.4])
def test_bind_rest_does_not_depend_on_default_joint_pose(tmp_path, default_joint):
    src = fixture(tmp_path / "source.gltf", joint_y=default_joint, bind_y=2.0)
    out = np.asarray(SourceGLTFBindRestV2(src).geometry()["vertices"])
    np.testing.assert_allclose(out[:3, 2], [2.2, 2.2, 3.2], atol=1e-7)


def test_conflicting_skin_bind_frames_are_not_silently_selected(tmp_path):
    src = fixture(tmp_path / "source.gltf", conflicting_skin=True)
    with pytest.raises(ValueError, match="AMBIGUOUS_SKIN_JOINT"):
        SourceGLTFBindRestV2(src).geometry()
