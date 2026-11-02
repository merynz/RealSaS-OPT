from __future__ import annotations
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from tools.truth_foundry import platform_corpus370_truth_operator_v1 as operator
from tools.truth_foundry import observe_corpus370_truth_v1 as observer


def test_projection_preserves_all_candidate_outcomes():
    p = {"candidate_subject_count": 370, "cohort_subject_count": 294,
         "excluded_subject_count": 76, "excluded_subjects": [{}] * 76}
    assert operator.projected_member_count(p, True) == 294
    p["excluded_subject_count"] = 75
    with pytest.raises(ValueError, match="PROJECTION_ACCOUNTING"):
        operator.projected_member_count(p, True)


def test_observer_accepts_exact_projection_and_does_not_claim_370_attempts(monkeypatch, tmp_path):
    members = [{"attempt_id": "a", "subject_id": "s", "role": "COHORT"}]
    calls = []
    def api(root, method, path):
        calls.append((method, path))
        if path.startswith("/v1/research/campaigns/"):
            return {"members": members}
        return {"state": "COMPLETED", "events": []}
    monkeypatch.setattr(observer, "api", api)
    monkeypatch.setattr(observer, "export_stage", lambda *a, **k: None)
    monkeypatch.setattr(observer, "export_diagnostics", lambda *a: [])
    result = observer.observe({"schema": observer.PROJECTION_SCHEMA,
        "member_count": 1, "candidate_subject_count": 370, "excluded_subject_count": 369,
        "campaign_id": "c", "experiment_id": "e", "source_code_sha": "1" * 40},
        api_root="unused", artifact_root=tmp_path, output=tmp_path,
        maximum_seconds=0, interval_seconds=1)
    assert result["all_projected_attempts_terminal"] is True
    assert result["all_370_attempts_terminal"] is False
    assert result["excluded_subject_count"] == 369
    assert all(method == "GET" for method, _ in calls)


def test_enabled_v2_request_binds_real_bundle_and_finite_campaign():
    root = Path(__file__).resolve().parents[2]
    request = json.loads((root / "canonical/CORPUS370_TRUTH_OPERATOR_REQUEST_V2.json").read_text())
    assert request["schema"] == operator.PROJECTION_SCHEMA
    assert request["enabled"] is True and request["training_allowed"] is False
    assert request["cohort_bundle"]["size_bytes"] == 1639176870
    assert request["cohort_bundle"]["sha256"] == "4462b7f90fcff58a8d56b1a975fde4f7f91cbda51417cad5121cf55f25994c5c"
    campaign = json.loads((root / request["campaign_contract_repo_path"]).read_text())
    assert campaign["required_target_stage_id"] == "49_SUBJECT_TRUTH_FAMILIES"
    assert campaign["candidate_universe_subject_count"] == 370


def test_observation_window_does_not_count_open_attempts_as_failures(monkeypatch, tmp_path):
    members = [{"attempt_id": state, "subject_id": state, "role": "COHORT"}
               for state in ("OPEN", "FAILED", "COMPLETED")]
    def api(root, method, path):
        if path.startswith("/v1/research/campaigns/"):
            return {"members": members}
        return {"state": path.rsplit("/", 1)[-1], "events": []}
    monkeypatch.setattr(observer, "api", api)
    monkeypatch.setattr(observer, "export_stage", lambda *a, **k: None)
    monkeypatch.setattr(observer, "export_diagnostics", lambda *a: [])
    result = observer.observe({"schema": observer.PROJECTION_SCHEMA,
        "member_count": 3, "candidate_subject_count": 370, "excluded_subject_count": 367,
        "campaign_id": "c", "experiment_id": "e", "source_code_sha": "1" * 40},
        api_root="unused", artifact_root=tmp_path, output=tmp_path,
        maximum_seconds=0, interval_seconds=1, metadata_only=True)
    assert result["failed_attempt_count"] == 1
    assert result["pending_attempt_count"] == 1
    assert result["observation_timed_out"] is True
    assert result["execution_resubmitted"] is False
    assert result["payload_export_mode"] == "REGISTRY_REFERENCES_ONLY"
