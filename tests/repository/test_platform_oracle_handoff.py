from copy import deepcopy
import json
from pathlib import Path

import pytest

from tools import platform_visual_oracle_v1 as oracle
from tools import platform_deployment_smoke as smoke


def setup(tmp_path, monkeypatch, terminal_state="FAILED"):
    export = tmp_path / "export"
    export.mkdir()
    scope = {"session_id": "session", "attempt_id": "attempt", "subject_id": "577fa30e-ae60-465d-93dd-0094f2747d4a",
             "subject_input_id": "input", "engine_release_id": "release", "canonical_code_sha": "a" * 40,
             "target_stage_id": "49_VISUAL_ORACLE_MEASURE"}
    state = {"attempt_id": "attempt", "subject_id": scope["subject_id"], "subject_input_id": "input", "release_id": "release",
             "code_sha": scope["canonical_code_sha"], "run_id": "original", "ticket": {"scope": scope, "scope_sha256": "b" * 64}}
    terminal = {"attempt_id": "attempt", "state": terminal_state, "events": [{"error": "actual failure"}]}
    partial = tmp_path / "partial.json"
    partial.write_text('{}')
    ledger = tmp_path / "ledger.json"
    ledger.write_text(json.dumps({"stages": [{"id": "47_AUTHORED_VISUAL_ORACLE", "outputs": [{"path": str(partial), "sha256": oracle.mainline.sha256_file(partial)}]}]}))
    monkeypatch.setattr(oracle.mainline, "run_ledger_path", lambda _: ledger)
    monkeypatch.setattr(oracle.mainline, "run_manifest_path", lambda _: tmp_path / "absent-manifest")
    monkeypatch.setattr(oracle, "registry_read", lambda *args: "0")
    calls = []
    def request(path, value=None, **kwargs):
        calls.append((path, deepcopy(value)))
        if path.startswith("/v1/agents/context/"):
            return {"canonical_code_sha": "c" * 40, "active_session": scope, "last_handoff": None}
        assert path == "/v1/agents/exit", "Finalization must never enter or submit execution"
        return {"handoff_sha256": "d" * 64}
    monkeypatch.setattr(oracle, "request", request)
    return export, export / "checkpoint.json", state, terminal, calls, partial


def test_failed_attempt_preserves_partial_bytes_and_exits_original_scope_after_main_changes(tmp_path, monkeypatch):
    export, checkpoint, state, terminal, calls, _ = setup(tmp_path, monkeypatch)
    with pytest.raises(RuntimeError, match="DIAGNOSTICS_AND_HANDOFF_PRESERVED"):
        oracle.finalize_terminal(tmp_path, export, checkpoint, state, terminal)
    saved = json.loads(checkpoint.read_text())
    assert saved["handoff"] and saved["terminal_state"] == terminal
    assert len(saved["export_receipt"]["verified_outputs"]) == 1
    assert json.loads((export / "terminal-attempt.json").read_text())["events"] == terminal["events"]
    assert calls[-1][1]["scope_sha256"] == state["ticket"]["scope_sha256"]
    assert not saved["export_receipt"]["scientific_pass_inferred"]


def test_output_drift_is_export_gap_not_green_and_does_not_trap_terminal_session(tmp_path, monkeypatch):
    export, checkpoint, state, terminal, calls, partial = setup(tmp_path, monkeypatch)
    partial.write_text('changed')
    with pytest.raises(RuntimeError, match="DIAGNOSTICS_AND_HANDOFF_PRESERVED"):
        oracle.finalize_terminal(tmp_path, export, checkpoint, state, terminal)
    assert state["export_receipt"]["export_errors"][0]["error"] == "ORACLE_EXPORT_OUTPUT_DRIFT"
    assert not state["export_receipt"]["verified_outputs"]
    assert calls[-1][0] == "/v1/agents/exit"


@pytest.mark.parametrize("terminal_state", ["OPEN", "UNRECOGNIZED"])
def test_running_or_unknown_attempt_cannot_finalize(tmp_path, monkeypatch, terminal_state):
    export, checkpoint, state, terminal, calls, _ = setup(tmp_path, monkeypatch, terminal_state)
    with pytest.raises(RuntimeError, match="ORIGINAL_TERMINAL"):
        oracle.finalize_terminal(tmp_path, export, checkpoint, state, terminal)
    assert not calls


def test_checkpoint_scope_drift_cannot_exit(tmp_path, monkeypatch):
    export, checkpoint, state, terminal, calls, _ = setup(tmp_path, monkeypatch)
    state["ticket"]["scope"]["subject_input_id"] = "other-input"
    with pytest.raises(RuntimeError, match="CHECKPOINT_SCOPE_DRIFT"):
        oracle.finalize_terminal(tmp_path, export, checkpoint, state, terminal)
    assert not calls


def test_lost_exit_reply_retries_exact_body_even_if_export_changes(tmp_path, monkeypatch):
    export, checkpoint, state, terminal, calls, partial = setup(tmp_path, monkeypatch)
    original = oracle.request
    def fail_reply(path, value=None, **kwargs):
        result = original(path, value, **kwargs)
        if path == "/v1/agents/exit":
            raise RuntimeError("LOST_REPLY")
        return result
    monkeypatch.setattr(oracle, "request", fail_reply)
    with pytest.raises(RuntimeError, match="LOST_REPLY"):
        oracle.finalize_terminal(tmp_path, export, checkpoint, state, terminal)
    first_body = calls[-1][1]
    partial.write_text('changed')
    def advanced_context(path, value=None, **kwargs):
        if path.startswith("/v1/agents/context/"):
            return {"active_session": {"session_id": "another-session"}, "last_handoff": {"scope": {"session_id": "another-scope"}}}
        return original(path, value, **kwargs)
    monkeypatch.setattr(oracle, "request", advanced_context)
    recovered = json.loads(checkpoint.read_text())
    with pytest.raises(RuntimeError, match="DIAGNOSTICS_AND_HANDOFF_PRESERVED"):
        oracle.finalize_terminal(tmp_path, export, checkpoint, recovered, terminal)
    assert calls[-1][1] == first_body


def test_terminal_observation_retains_failure_but_smoke_still_requires_success(monkeypatch):
    failed = {"state": "FAILED", "attempt_id": "original", "events": ["diagnostic"]}
    monkeypatch.setattr(smoke, "request", lambda _: failed)
    assert smoke.wait_attempt_terminal("original", 1) == failed
    with pytest.raises(RuntimeError, match="SMOKE_ATTEMPT_FAILED"):
        smoke.wait_attempt("original", 1)


def test_timeout_does_not_finalize_or_submit(monkeypatch):
    calls = []
    monkeypatch.setattr(smoke, "request", lambda path: calls.append(path))
    with pytest.raises(RuntimeError, match="TIMEOUT"):
        smoke.wait_attempt_terminal("original", 0)
    assert not calls


def test_explicit_recovery_never_checks_new_execution_checkout_or_builds_inputs(tmp_path, monkeypatch):
    export, checkpoint, state, terminal, calls, _ = setup(tmp_path, monkeypatch)
    monkeypatch.setattr(oracle.mainline, "load_json", lambda path:
        {"experiment_id": "fixture"} if Path(path).name == "AUTHORED_VISUAL_ORACLE_V1_20261010.json" else
        {"authority_root": str(tmp_path), "code_sha": "c" * 40} if Path(path).name == "deployment.json" else
        state if Path(path).name == "api-checkpoint.json" else json.loads(Path(path).read_text()))
    expected_checkpoint = tmp_path / "oracle-results/fixture/api-checkpoint.json"
    expected_checkpoint.parent.mkdir(parents=True)
    expected_checkpoint.write_text('{}')
    def forbidden(*args, **kwargs):
        raise AssertionError("Recovery must not prepare or execute new work")
    monkeypatch.setattr(oracle, "canonical_checkout", forbidden)
    monkeypatch.setattr(oracle, "source_inputs", forbidden)
    original = oracle.request
    monkeypatch.setattr(oracle, "request", lambda path, value=None, **kwargs:
        terminal if path == "/v1/attempts/attempt" else original(path, value, **kwargs))
    with pytest.raises(RuntimeError, match="DIAGNOSTICS_AND_HANDOFF_PRESERVED"):
        oracle.run(tmp_path, 1, finalize_existing=True)
    assert state["handoff"]
