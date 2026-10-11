from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from tools import realsas_architecture as arch


def fixture():
    stages = [{"id": sid, "ordinal": i, "depends_on": deps, "manifest_keys": keys}
              for i, (sid, deps, keys) in enumerate([
                  ("scene", [], ["images"]), ("geometry", ["scene"], []),
                  ("colour", ["scene"], ["appearance"]), ("rig", ["geometry"], [])], 1)]
    return {"plan": {"stages": stages}, "pipeline_plan_sha256": "f" * 64,
            "dependency_files": [
                {"path": "models/shared.py", "consumer_stages": ["scene", "geometry", "colour"]},
                {"path": "models/colour.py", "consumer_stages": ["colour"]}],
            "critical_files": ["AGENTS.md"]}


def test_head_edit_preserves_independent_branch_but_shared_source_widens_scope():
    d = fixture()
    head = arch.impact_query(d, ["models/colour.py"], [])
    assert head["affected_stage_ids"] == ["colour"]
    assert head["outside_declared_impact_stage_ids"] == ["scene", "geometry", "rig"]
    assert arch.impact_query(d, ["models/shared.py"], [])["affected_stage_ids"] == ["scene", "geometry", "colour", "rig"]
    assert arch.impact_query(d, [], ["appearance"])["affected_stage_ids"] == ["colour"]


def test_unknown_native_control_and_manifest_edits_never_claim_safe_reuse():
    d = fixture()
    for paths, keys in [(["runtime/player.cpp"], []), ([], ["typo"])]:
        value = arch.impact_query(d, paths, keys)
        assert value["unclassified_inputs"]
        assert "not reuse eligibility" in value["authority_note"]
    global_edit = arch.impact_query(d, ["AGENTS.md"], [])
    assert global_edit["paths"][0]["classification"].startswith("GLOBAL_CLOSURE_VALIDATION")
    assert global_edit["affected_stage_ids"] == []
    assert len(arch.impact_query(d, ["canonical/MAINLINE_EXECUTION_PLAN_V2.json"], [])["affected_stage_ids"]) == 4


def test_map_edges_exactly_match_plan_and_do_not_invent_payload_acceptance(monkeypatch):
    d = fixture()
    d.update(registry={"stages": [{"stage_id": s["id"], "owner_module_id": "engine.test"} for s in d["plan"]["stages"]], "domains": []},
             stage_closures={s["id"]: {"stage_id": s["id"]} for s in d["plan"]["stages"]})
    monkeypatch.setattr(arch, "source_snapshot", lambda _: {"head_sha": "a" * 40})
    monkeypatch.setattr(arch, "audit_summary", lambda _: {})
    value = arch.architecture_map(d)
    assert {(e["producer"], e["consumer"]) for e in value["edges"]} == {("scene", "geometry"), ("scene", "colour"), ("geometry", "rig")}
    assert value["nodes"][0]["direct_consumers"] == ["geometry", "colour"]
    assert value["nodes"][0]["input_roles"] == ["manifest:images"]
    assert "independent consumer measurements" in value["evidence_boundary"]


def test_html_embedded_data_cannot_break_out_of_script(tmp_path):
    output = tmp_path / "map.html"
    value = {"payload": "</script><script>alert(1)</script>"}
    arch.write_map_html(value, str(output))
    text = output.read_text()
    embedded = text.split('<script id="data" type="application/json">')[1].split('</script>')[0]
    assert json.loads(embedded) == value
    assert "</script>" not in embedded


def test_map_rejects_source_edit_during_discovery(tmp_path, monkeypatch):
    paths = ["models/source.py", "canonical/MAINLINE_EXECUTION_PLAN_V2.json",
             "tools/realsas_architecture.py", "tools/architecture_map.html"]
    for path in paths:
        file = tmp_path / path
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text('{}')
    source = tmp_path / paths[0]
    digest = arch.mainline.sha256_file(source)
    d = {"dependency_files": [{"path": paths[0]}], "critical_file_hashes": [],
         "stage_closures": {"stage": {"dependency_file_hashes": {paths[0]: digest}}},
         "pipeline_plan_sha256": arch.mainline.content_sha256({})}
    monkeypatch.setattr(arch, "ROOT", tmp_path)
    monkeypatch.setattr(arch.mainline, "PLAN_PATH", tmp_path / paths[1])
    monkeypatch.setattr(arch, "git_files", lambda: paths)
    monkeypatch.setattr(arch, "_run", lambda *args: SimpleNamespace(stdout="a" * 40))
    first = arch.source_snapshot(d)
    source.write_text('changed')
    with pytest.raises(RuntimeError, match="SOURCE_CHANGED_DURING_DISCOVERY"):
        arch.source_snapshot(d)
    assert len(first["source_fingerprint_sha256"]) == 64
