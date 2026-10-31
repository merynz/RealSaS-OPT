from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools.corpus.post139_build_materialization_closure_v1 import deterministic_zip, sha256_file
from tools.corpus.post139_corpus_authority_v1 import digest, stable_evidence_hash, verify_raw_ledger


def test_stable_evidence_hash_ignores_runtime_paths():
    base = {
        "source_root": "/tmp/runner-a/raw",
        "source": {
            "path": "/tmp/runner-a/raw/Actor.gltf",
            "name": "Actor.gltf",
            "extension": ".gltf",
            "sha256": "abc",
            "size_bytes": 5,
        },
        "fingerprints": {"structural_subject": "same"},
    }
    other = json.loads(json.dumps(base).replace("runner-a", "runner-b"))
    assert stable_evidence_hash(base) == stable_evidence_hash(other)


def test_raw_ledger_is_recomputed_from_exact_bytes_and_detects_tamper(tmp_path: Path):
    raw = tmp_path / "raw"
    raw.mkdir()
    model = raw / "Actor.gltf"
    model.write_bytes(b"actor")
    sha = hashlib.sha256(b"actor").hexdigest()
    rows = [{"path": "Actor.gltf", "size_bytes": 5, "sha256": sha, "extension": ".gltf"}]
    ledger = {
        "schema": "RealSaS.Post139.RawFileLedger.v1",
        "source_id": "src",
        "file_count": 1,
        "total_bytes": 5,
        "extension_counts": {".gltf": 1},
        "files": rows,
        "ledger_sha256": digest(rows),
    }
    verified = verify_raw_ledger("src", raw, ledger)
    assert verified["raw_ledger_sha256"] == ledger["ledger_sha256"]
    model.write_bytes(b"evil!")
    with pytest.raises(ValueError, match="POST139_AUTHORITY_RAW_FILE_HASH_MISMATCH"):
        verify_raw_ledger("src", raw, ledger)


def test_materialization_zip_is_deterministic_across_mtime_changes(tmp_path: Path):
    root = tmp_path / "payload"
    root.mkdir()
    payload = root / "a.txt"
    payload.write_text("same bytes", encoding="utf-8")
    first = tmp_path / "one.zip"
    second = tmp_path / "two.zip"
    deterministic_zip(root, first)
    payload.touch()
    deterministic_zip(root, second)
    assert sha256_file(first) == sha256_file(second)
