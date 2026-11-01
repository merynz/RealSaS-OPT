import hashlib
import json
from pathlib import Path
import pytest

from tools.corpus.compile_expanded_subject_candidates_v1 import authored_key, compile_candidates, resource_closure
from tools.corpus.run_expanded_teacher_source_inspection_v1 import reusable_report


def test_export_formats_and_equipped_variants_do_not_inflate_subject_identity():
    assert authored_key("Characters/Normal/glTF/Panda.gltf") == authored_key("Characters/With Knife and Pan/FBX/Panda.fbx")
    assert authored_key("Models/FBX format/Knight.fbx") == authored_key("Models/glTF format/Knight.gltf")
    assert authored_key("Characters/Normal/glTF/Panda.gltf") != authored_key("Characters/Normal/glTF/Rabbit.gltf")


def test_missing_texture_is_not_a_complete_gltf_bundle(tmp_path):
    p = tmp_path / "actor.gltf"
    p.write_text(json.dumps({"asset": {"version": "2.0"}, "images": [{"uri": "missing.png"}]}))
    assert resource_closure(p)["missing"] == ["missing.png"]


def test_embedded_resource_requires_no_invented_texture_repair(tmp_path):
    p = tmp_path / "actor.gltf"
    p.write_text(json.dumps({"asset": {"version": "2.0"}, "images": [{"uri": "data:image/png;base64,AA=="}]}))
    assert resource_closure(p)["status"] == "COMPLETE_DECLARED_GLTF_RESOURCES"


def test_changed_base_source_fails_before_candidate_creation(tmp_path):
    p = tmp_path / "base/model/actors/one.gltf"
    p.parent.mkdir(parents=True)
    p.write_text("changed")
    base = {"subject_count": 1, "subjects": [{"subject_id": "base_one", "corpus_relative_path": "base/model/actors/one.gltf", "expected_sha256": "0" * 64}]}
    with pytest.raises(ValueError, match="EXPANDED_BASE_SOURCE_DRIFT"):
        compile_candidates(base, tmp_path, {"subjects": []}, {"subjects": []}, tmp_path, {"files": []})


def test_cache_requires_both_payloads_and_current_source(tmp_path):
    source = tmp_path / "actor.gltf"
    source.write_text("original")
    row = {"subject_id": "actor", "source": {"path": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}}
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"subject_id": "actor", "source": row["source"], "runtime": {"blender": "4.0.2"}, "producer_source_sha256": {"missing.py": "0" * 64}, "outputs": {}}))
    assert not reusable_report(report, row)
    report.write_text("{interrupted")
    assert not reusable_report(report, row)


def test_no_report_is_a_cache_miss(tmp_path):
    assert not reusable_report(tmp_path / "absent.json", {"subject_id": "actor", "source": {}})
