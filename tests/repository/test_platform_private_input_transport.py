import json

import pytest

from tools import platform_private_input_transport as transport


RUN_ID = "12345"
CODE_SHA = "a" * 40


def test_inline_payload_remains_backward_compatible():
    payload = {
        "run_id": RUN_ID,
        "code_sha": CODE_SHA,
        "transfers": [],
        "config": {"render_resolution": 256},
    }
    assert transport.resolve_payload(payload, run_id=RUN_ID, code_sha=CODE_SHA) is payload


def test_manifest_payload_resolves_only_exact_run_and_code_identity():
    manifest = {
        "run_id": RUN_ID,
        "code_sha": CODE_SHA,
        "transfers": [{
            "name": "input.json",
            "download_url": "https://example.oaiusercontent.com/input",
            "size_bytes": 2,
            "sha256": "0" * 64,
        }],
        "config": {"render_resolution": 256},
    }
    wrapper = {
        "run_id": RUN_ID,
        "code_sha": CODE_SHA,
        "manifest": {
            "download_url": "https://example.oaiusercontent.com/manifest",
            "size_bytes": 1,
            "sha256": "1" * 64,
        },
    }
    calls = []

    def downloader(ref, label, max_bytes):
        calls.append((ref, label, max_bytes))
        return json.dumps(manifest).encode()

    resolved = transport.resolve_payload(
        wrapper, run_id=RUN_ID, code_sha=CODE_SHA, downloader=downloader)
    assert resolved == manifest
    assert calls == [(wrapper["manifest"], "manifest", transport.MAX_MANIFEST_BYTES)]


def test_manifest_identity_drift_fails_closed():
    wrapper = {
        "run_id": RUN_ID,
        "code_sha": CODE_SHA,
        "manifest": {
            "download_url": "https://example.oaiusercontent.com/manifest",
            "size_bytes": 1,
            "sha256": "1" * 64,
        },
    }
    wrong = {"run_id": RUN_ID, "code_sha": "b" * 40, "transfers": [], "config": {}}
    with pytest.raises(RuntimeError, match="MANIFEST_IDENTITY_DRIFT"):
        transport.resolve_payload(
            wrapper, run_id=RUN_ID, code_sha=CODE_SHA,
            downloader=lambda *_: json.dumps(wrong).encode())


def test_manifest_and_inline_payload_cannot_be_mixed():
    payload = {
        "run_id": RUN_ID,
        "code_sha": CODE_SHA,
        "manifest": {},
        "transfers": [],
    }
    with pytest.raises(RuntimeError, match="MANIFEST_PAYLOAD_AMBIGUOUS"):
        transport.resolve_payload(payload, run_id=RUN_ID, code_sha=CODE_SHA,
                                  downloader=lambda *_: b"{}")
