import hashlib
import io
import json
import zipfile

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


def _zip_bytes(files):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    return stream.getvalue()


def test_hash_pinned_bundle_rehydrates_exact_inner_files(tmp_path):
    files = {"one.json": b"{}", "two.npz": b"npz-bytes"}
    raw = _zip_bytes(files)
    ref = {
        "download_url": "https://example.oaiusercontent.com/bundle",
        "size_bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "files": [
            {"name": name, "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            for name, data in files.items()
        ],
    }
    verified = transport._extract_bundle(ref, tmp_path, downloader=lambda *_: raw)
    assert dict(verified) == {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == files


def test_bundle_rejects_nested_or_unexpected_zip_entries(tmp_path):
    good = b"ok"
    raw = _zip_bytes({"one.json": good, "nested/two.json": b"no"})
    ref = {
        "download_url": "https://example.oaiusercontent.com/bundle",
        "size_bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "files": [{"name": "one.json", "size_bytes": len(good),
                   "sha256": hashlib.sha256(good).hexdigest()}],
    }
    with pytest.raises(RuntimeError, match="ZIP_ENTRY_INVALID"):
        transport._extract_bundle(ref, tmp_path, downloader=lambda *_: raw)
