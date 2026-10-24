from __future__ import annotations

import json

import pytest

from tools.ops import execute_current_skin_parent_capabilities_v1 as subject


SUBJECT_ID = "11111111-1111-4111-8111-111111111111"
INPUT_ID = "22222222-2222-4222-8222-222222222222"


def _loaded(roles: list[str]) -> dict:
    return {
        "subject_input_id": INPUT_ID,
        "manifest_sha256": "a" * 64,
        "root_inputs": [{"role": role} for role in roles],
        "artifact_ids": [f"artifact-{index}" for index in range(len(roles))],
    }


def test_exact_subject_input_pin_preserves_declared_absence(monkeypatch: pytest.MonkeyPatch) -> None:
    loaded = _loaded([
        "subject:manifest:observation",
        "subject:manifest:source_audit",
    ])
    payloads = {
        "artifact-0": {"camera_bundle": {"path": "/x", "sha256": "b" * 64}},
        "artifact-1": {"subject_key": "SUBJECT2_KNIGHT"},
    }

    def fake_api(api_root: str, method: str, path: str, payload=None, raw=None):
        if path.endswith(f"/inputs/{INPUT_ID}"):
            return loaded
        if path.startswith("/v1/artifacts/"):
            return {
                "artifact_type": "RealSaS.RunManifestSection",
                "schema_version": "v1",
                "semantic_sha256": "c" * 64,
                "content_sha256": "d" * 64,
                "size_bytes": 1,
            }
        raise AssertionError(path)

    monkeypatch.setattr(subject, "_api", fake_api)
    monkeypatch.setattr(
        subject,
        "_artifact_bytes",
        lambda api_root, artifact_id, metadata: json.dumps(payloads[artifact_id]).encode("utf-8"),
    )

    current, receipt = subject._load_subject_input_by_id(
        "http://platform",
        SUBJECT_ID,
        INPUT_ID,
        {"observation", "source_audit", "components", "carrier_policy"},
    )

    assert current["subject_input_id"] == INPUT_ID
    assert receipt["selection_mode"] == "EXACT_PIN"
    assert set(receipt["sections"]) == {"observation", "source_audit"}
    assert receipt["declared_manifest_keys_absent_from_subject_input"] == [
        "carrier_policy",
        "components",
    ]


def test_exact_subject_input_pin_rejects_undeclared_manifest_section(monkeypatch: pytest.MonkeyPatch) -> None:
    loaded = _loaded(["subject:manifest:observation", "subject:manifest:historical_extra"])

    def fake_api(api_root: str, method: str, path: str, payload=None, raw=None):
        if path.endswith(f"/inputs/{INPUT_ID}"):
            return loaded
        if path.startswith("/v1/artifacts/"):
            return {
                "artifact_type": "RealSaS.RunManifestSection",
                "schema_version": "v1",
                "semantic_sha256": "c" * 64,
                "content_sha256": "d" * 64,
                "size_bytes": 1,
            }
        raise AssertionError(path)

    monkeypatch.setattr(subject, "_api", fake_api)
    monkeypatch.setattr(subject, "_artifact_bytes", lambda *args, **kwargs: b"{}")

    with pytest.raises(RuntimeError, match="SKIN_PARENT_SUBJECT_INPUT_UNDECLARED_MANIFEST_SECTION:historical_extra"):
        subject._load_subject_input_by_id(
            "http://platform",
            SUBJECT_ID,
            INPUT_ID,
            {"observation"},
        )
