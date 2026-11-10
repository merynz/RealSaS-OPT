import json
import pytest

from tools.corpus.gltf_bind_scene_v2 import SourceGLTFBindRestV2
from test_gltf_bind_scene_v2 import fixture


def test_two_instances_of_one_skin_with_different_node_frames_fail_closed(tmp_path):
    source = fixture(tmp_path / "source.gltf")
    value = json.loads(source.read_text())
    value["nodes"].append({"name": "SecondSkinInstance", "mesh": 1, "skin": 0,
                           "translation": [0, .2, 0]})
    value["scenes"][0]["nodes"].append(len(value["nodes"]) - 1)
    source.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="RAW_GLTF_BIND_AMBIGUOUS_SKIN_JOINT"):
        SourceGLTFBindRestV2(source).geometry()
