"""Exercise the real PASS readback boundary without submitting execution."""
from argparse import Namespace
import hashlib
import io
import json

import pytest

from tools import platform_source_ray_appearance_admission_v1 as operator


ATTEMPT = "783674b2-8283-498d-ad76-9ed1f09dc330"
SOURCE = "8cf5ad252037c96a2c6a7fee9015be0bb6957f19"


@pytest.fixture
def readback(tmp_path, monkeypatch):
    bundle = {"schema": "sealed-fixture"}
    raw = operator._json_bytes(bundle)
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_bytes(raw)
    run_id = "SOURCE_RAY_APPEARANCE_ADMISSION_" + SOURCE[:12].upper() + "_" + operator._sha(raw)[:12].upper()
    manifest = tmp_path / "run_manifest.json"
    manifest.write_bytes(operator._json_bytes({"run_id": run_id, "subject_id": "subject",
                                               "execution_class": "WITNESS", operator.MANIFEST_KEY: bundle}))
    context = dict(attempt_id=ATTEMPT, subject_id="subject", subject_input_id="input",
                   engine_release_id="release", execution_goal_id="goal", bundle_artifact_id="bundle",
                   run_id=run_id, run_manifest_path=str(manifest))
    monkeypatch.setattr(operator, "_verify_checkout", lambda *args: None)
    monkeypatch.setattr(operator, "_existing_context", lambda *args: context)
    monkeypatch.setattr(operator.mainline, "run_manifest_path", lambda run: manifest)
    terminal = {"attempt_id": ATTEMPT, "state": "PASS", "events": [{
        "type": "CAPABILITY_GOAL_COMPLETED", "payload": {"execution_goal_id": "goal",
        "promotion_policy": "NEVER", "resolved_capability_ids": [operator.CAPABILITY_ID]}}]}
    payload = {"measurement_set_sha256": "m" * 64, "canonical_mesh_binding_minted": False,
               "training_authorized": False, "product_authority_changed": False}
    bytes_by_id = {"bundle": raw}
    metadata = {"bundle": {}}
    rows = []
    for artifact_id, artifact_type in [("qualification", operator.QUALIFICATION_TYPE), ("evidence", operator.EVIDENCE_TYPE)]:
        bytes_by_id[artifact_id] = operator._json_bytes({**payload, "schema": artifact_type + ".v1"})
        metadata[artifact_id] = {"producer_contract": operator.CAPABILITY_ID,
            "qualifications": [{"qualification_type": "CAPABILITY_EXECUTION_PASS", "result": "PASS"}],
            "inputs": [{"role": "subject:manifest:" + operator.MANIFEST_KEY, "artifact_id": "bundle"}]}
        rows.append({"artifact_id": artifact_id, "artifact_type": artifact_type, "schema_version": "v1",
                     "source_kind": "ATTEMPT", "source_id": ATTEMPT})
    for key, value in metadata.items():
        value.update(content_sha256=hashlib.sha256(bytes_by_id[key]).hexdigest(), size_bytes=len(bytes_by_id[key]))
    calls = []

    def api(root, method, path, *args, **kwargs):
        calls.append((method, path))
        assert method == "GET", "readback must not mutate Platform state"
        if path == "/v1/attempts/" + ATTEMPT:
            return terminal
        if path == "/v1/subjects/subject/artifacts?limit=500":
            return rows
        return metadata[path.rsplit("/", 1)[1]]

    def urlopen(request, **kwargs):
        assert request.method == "GET"
        key = request.full_url.split("/")[-2]
        response = io.BytesIO(bytes_by_id[key])
        response.headers = {"X-Content-SHA256": metadata[key]["content_sha256"]}
        return response

    monkeypatch.setattr(operator, "_api", api)
    monkeypatch.setattr(operator.urllib.request, "urlopen", urlopen)
    args = Namespace(root=tmp_path, expected_main_sha="a" * 40, bundle=bundle_path,
                     finalize_existing_attempt_id=ATTEMPT, admission_source_sha=SOURCE,
                     api_root="http://fixture", timeout=1, out=tmp_path / "export/receipt.json")
    return Namespace(args=args, terminal=terminal, metadata=metadata, bytes=bytes_by_id,
                     context=context, manifest=manifest, calls=calls)


def test_existing_pass_reads_exact_registry_bytes_without_a_new_attempt(readback):
    receipt = operator.execute(readback.args)
    assert receipt["status"] == "PASS_PLATFORM_REGISTRY_ADMISSION"
    assert receipt["attempt_id"] == ATTEMPT
    assert receipt["source_code_sha"] == SOURCE
    assert receipt["readback_source_code_sha"] == "a" * 40
    assert receipt["recovered_existing_attempt"] is True
    assert receipt["terminal_state"] == "PASS"
    assert receipt["training_authorized"] is False
    assert all(method == "GET" for method, _ in readback.calls)
    assert readback.args.out.with_name("registry-readback.json").is_file()


@pytest.mark.parametrize("state", ["COMPLETED", "FAIL", "FAILED", "ABSTAIN"])
def test_other_terminal_states_cannot_finalize_capability_success(readback, state):
    readback.terminal["state"] = state
    with pytest.raises(RuntimeError, match="ATTEMPT_FAILED"):
        operator.execute(readback.args)
    assert not readback.args.out.with_name("registry-readback.json").exists()


@pytest.mark.parametrize("change", ["goal", "promotion", "capability", "missing_event"])
def test_pass_with_wrong_goal_scope_is_rejected(readback, change):
    payload = readback.terminal["events"][0]["payload"]
    if change == "goal": payload["execution_goal_id"] = "other"
    if change == "promotion": payload["promotion_policy"] = "AUTO"
    if change == "capability": payload["resolved_capability_ids"] = ["other"]
    if change == "missing_event": readback.terminal["events"] = []
    with pytest.raises(RuntimeError, match="TERMINAL_SCOPE_DRIFT"):
        operator.execute(readback.args)


def test_registry_hash_drift_is_rejected(readback):
    readback.bytes["qualification"] += b" "
    with pytest.raises(RuntimeError, match="REGISTRY_BYTES_DRIFT"):
        operator.execute(readback.args)


def test_another_bundle_under_same_role_is_rejected(readback):
    readback.metadata["qualification"]["inputs"][0]["artifact_id"] = "other"
    with pytest.raises(RuntimeError, match="EXACT_INPUT_LINEAGE_MISSING"):
        operator.execute(readback.args)


def test_old_manifest_cannot_be_rewritten_for_recovery(readback):
    readback.manifest.write_text("{}")
    with pytest.raises(RuntimeError, match="EXISTING_MANIFEST_DRIFT"):
        operator.execute(readback.args)
    assert readback.calls == []


def test_existing_context_uses_only_select_and_exact_go_scope(tmp_path, monkeypatch):
    context = {"attempt_id": ATTEMPT, "promotion_policy": "NEVER", "resolved_capability_ids": [operator.CAPABILITY_ID]}
    queries = []
    def read(root, sql):
        queries.append(sql)
        assert sql.strip().startswith("SELECT")
        assert "WHERE a.id='" + ATTEMPT + "'" in sql
        return json.dumps(context)
    monkeypatch.setattr(operator, "registry_read", read)
    assert operator._existing_context(tmp_path, ATTEMPT) == context
    context["promotion_policy"] = "AUTO"
    with pytest.raises(RuntimeError, match="EXISTING_SCOPE_DRIFT"):
        operator._existing_context(tmp_path, ATTEMPT)
    with pytest.raises(ValueError):
        operator._existing_context(tmp_path, "not-a-uuid; SELECT")
    assert len(queries) == 2
