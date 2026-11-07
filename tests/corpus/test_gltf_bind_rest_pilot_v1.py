import json
from pathlib import Path

import pytest

from tools.corpus.gltf_bind_scene_v2 import SourceGLTFBindRestV2
from tools.corpus.verify_gltf_bind_rest_pilot_v1 import inspect, sha256
from test_gltf_bind_scene_v2 import fixture


def imported_fixture(tmp_path):
    source = fixture(tmp_path / "source.gltf", joint_y=1.0, bind_y=2.0)
    value = SourceGLTFBindRestV2(source).geometry()
    imported = tmp_path / "imported.json"
    imported.write_text(json.dumps({
        "vertices": value["vertices"], "bone_count": value["bone_count"],
        "import": {"source_sha256": sha256(source),
                   "pose_policy": "SOURCE_BIND_REST__NO_ANIMATION"},
    }))
    return source, imported


def test_scoped_numeric_pilot_reports_fix_but_no_admission(tmp_path):
    src, imported = imported_fixture(tmp_path)
    result = inspect(src, imported, expected_source_sha256=sha256(src),
                     expected_import_sha256=sha256(imported), subject_id="s")
    assert result["historical_relative_vertex_error"] > .1
    assert result["corrected_relative_vertex_error"] < 1e-10
    assert result["corrected_primitive_count"] == 1
    assert result["observation_admitted"] is False


def test_pilot_rejects_tampered_original_import(tmp_path):
    src, imported = imported_fixture(tmp_path)
    expected = sha256(imported)
    imported.write_text(imported.read_text() + " ")
    with pytest.raises(ValueError, match="BIND_PILOT_PARENT_DRIFT"):
        inspect(src, imported, expected_source_sha256=sha256(src),
                expected_import_sha256=expected, subject_id="s")
