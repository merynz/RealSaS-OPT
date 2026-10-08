import hashlib
import json
from pathlib import Path

import pytest

from compiler.realsas_compiler_services.orchestrator import mainline
from compiler.realsas_compiler_services.platform_worker.stage_inputs import hydrate_stage_inputs


@pytest.fixture
def portable(tmp_path, monkeypatch):
    monkeypatch.setenv("REALSAS_AUTHORITY_ROOT", str(tmp_path))
    monkeypatch.setattr(mainline, "_adapter_impl_hash", lambda adapter: "a" * 64)
    plan = mainline.load_json(mainline.PLAN_PATH)
    ledger = mainline.build_fresh_run_ledger(plan, run_id="NEW_ATTEMPT", subject_id="SUBJECT_FREE_PORTABLE", execution_class="IMPLEMENTATION_AUDIT", architecture_scope="PORTABLE_TEST", manifest_ref={"path":str(tmp_path/"manifest.json"),"sha256":"d"*64})
    stage = plan["stages"][0]
    output = b'{"schema":"RealSaS.TestIR.v1","carrier":"exact-original-bytes"}\n'
    objects = {}
    def put(data):
        digest = hashlib.sha256(data).hexdigest()
        objects[digest] = data
        return {"storage_key": "cas/sha256/" + digest, "content_sha256": digest, "size_bytes": len(data)}
    sealed = put(output)
    cached = {"schema":"RealSaS.StageResultManifest.v1", "stage_id": stage["id"], "compiler_status": "PASS", "implementation_sha256": "a" * 64,
              "semantic_parameters": {"manifest": mainline._manifest_subset({},stage)},
              "policy_sha256": mainline.content_sha256(stage["policy"]), "expected_semantic_sha256": "b" * 64,
              "outputs": [{**sealed, "relative_path": "evidence.json", "payload_schema": "RealSaS.TestIR.v1", "authority_class": "TEST"}]}
    def ref():
        return {**put(json.dumps(cached).encode()), "artifact_type": "RealSaS.StageResultManifest", "schema_version": "v1", "semantic_sha256": "b" * 64, "id": "immutable-artifact"}
    def read(obj):
        data = objects[obj["content_sha256"]]
        if hashlib.sha256(data).hexdigest() != obj["content_sha256"] or len(data) != obj["size_bytes"]:
            raise RuntimeError("CAS_OBJECT_CONTENT_DRIFT")
        return data
    def hydrate():
        hydrate_stage_inputs(plan=plan, ledger=ledger, manifest={}, inputs=[{"stage_id": stage["id"], "artifact": ref()}], read_object=read, mode="RESEARCH")
    return cached, ledger, objects, output, hydrate


def test_new_attempt_hydrates_exact_bytes_without_executing_adapter(portable):
    cached, ledger, objects, output, hydrate = portable
    hydrate()
    row = ledger["stages"][0]
    assert row["attempts"] == 0
    assert row["status"] == "PASS"
    assert Path(row["outputs"][0]["path"]).read_bytes() == output
    assert row["platform_artifact_id"] == "immutable-artifact"
    hydrate()  # crash/retry reconstructs the same immutable input
    assert row["attempts"] == 0


@pytest.mark.parametrize("relative", ["../weight.json", "/tmp/weight.json", ""])
def test_cache_cannot_escape_stage_authority_root(portable, relative):
    cached, _, _, _, hydrate = portable
    cached["outputs"][0]["relative_path"] = relative
    with pytest.raises(RuntimeError, match="PORTABLE_OUTPUT_REQUIRED"):
        hydrate()


def test_changed_implementation_cannot_consume_stale_stage_pass(portable):
    cached, _, _, _, hydrate = portable
    cached["implementation_sha256"] = "c" * 64
    with pytest.raises(RuntimeError, match="IMPLEMENTATION_DRIFT"):
        hydrate()


def test_corrupt_immutable_payload_fails_closed(portable):
    cached, _, objects, _, hydrate = portable
    objects[cached["outputs"][0]["content_sha256"]] = b"changed-carrier"
    with pytest.raises(RuntimeError, match="CONTENT_DRIFT"):
        hydrate()
