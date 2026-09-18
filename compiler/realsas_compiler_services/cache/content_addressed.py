from __future__ import annotations

"""Fail-closed content-addressed artifact cache.

Cache entries are keyed only by canonical input identities and an explicit producer
fingerprint. Cached bytes are never trusted without an exact size + SHA-256 check.
Writes are atomic at both entry and requested-output boundaries.
"""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping
import json
import os
import shutil
import tempfile
import uuid

CACHE_SCHEMA = "RealSaS.ContentAddressedStageCache.v1"
_ARTIFACT_NAME = "artifact.bin"
_MANIFEST_NAME = "manifest.json"


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_file(path: str | Path) -> str:
    h = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stage_cache_key(
    *,
    stage: str,
    producer_fingerprint: str,
    inputs: Mapping[str, Any],
) -> str:
    if not str(stage).strip() or not str(producer_fingerprint).strip():
        raise ValueError("cache stage and producer_fingerprint must be non-empty")
    payload = {
        "cache_schema": CACHE_SCHEMA,
        "stage": str(stage),
        "producer_fingerprint": str(producer_fingerprint),
        "inputs": dict(inputs),
    }
    return sha256(_canonical_json(payload)).hexdigest()


@dataclass(frozen=True)
class CacheRestore:
    hit: bool
    key: str
    reason: str
    artifact_sha256: str | None = None
    artifact_bytes: int | None = None
    metadata: dict[str, Any] | None = None
    output_already_current: bool = False


@dataclass(frozen=True)
class CacheStore:
    key: str
    artifact_sha256: str
    artifact_bytes: int
    entry_path: str
    reused_existing_entry: bool


class ContentAddressedStageCache:
    def __init__(self, root: str | Path, *, stage: str, producer_fingerprint: str):
        self.root = Path(root).expanduser().resolve()
        self.stage = str(stage)
        self.producer_fingerprint = str(producer_fingerprint)
        if not self.stage.strip() or not self.producer_fingerprint.strip():
            raise ValueError("cache stage and producer_fingerprint must be non-empty")

    def key(self, inputs: Mapping[str, Any]) -> str:
        return stage_cache_key(
            stage=self.stage,
            producer_fingerprint=self.producer_fingerprint,
            inputs=inputs,
        )

    def entry_path(self, key: str) -> Path:
        if len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
            raise ValueError("cache key must be lowercase SHA-256")
        return self.root / self.stage / key[:2] / key

    @staticmethod
    def _load_manifest(entry: Path) -> dict[str, Any] | None:
        try:
            value = json.loads((entry / _MANIFEST_NAME).read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return None
        return value if isinstance(value, dict) else None

    def _valid_manifest(self, key: str, manifest: Mapping[str, Any]) -> bool:
        artifact = manifest.get("artifact")
        return (
            manifest.get("cache_schema") == CACHE_SCHEMA
            and manifest.get("stage") == self.stage
            and manifest.get("key") == key
            and manifest.get("producer_fingerprint") == self.producer_fingerprint
            and isinstance(artifact, dict)
            and artifact.get("path") == _ARTIFACT_NAME
            and isinstance(artifact.get("sha256"), str)
            and len(artifact["sha256"]) == 64
            and isinstance(artifact.get("bytes"), int)
            and artifact["bytes"] >= 0
            and isinstance(manifest.get("metadata", {}), dict)
        )

    @staticmethod
    def _matches(path: Path, *, expected_sha256: str, expected_bytes: int) -> bool:
        try:
            if not path.is_file() or path.stat().st_size != expected_bytes:
                return False
            return sha256_file(path) == expected_sha256
        except OSError:
            return False

    @staticmethod
    def _atomic_copy(source: Path, target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent))
        temp = Path(temp_name)
        try:
            with os.fdopen(fd, "wb") as dst, source.open("rb") as src:
                shutil.copyfileobj(src, dst, length=1 << 20)
                dst.flush()
                os.fsync(dst.fileno())
            os.replace(temp, target)
        finally:
            try:
                temp.unlink()
            except FileNotFoundError:
                pass

    def restore(self, key: str, target: str | Path) -> CacheRestore:
        entry = self.entry_path(key)
        manifest = self._load_manifest(entry)
        if manifest is None or not self._valid_manifest(key, manifest):
            return CacheRestore(False, key, "MISS_OR_INVALID_MANIFEST")

        artifact_meta = manifest["artifact"]
        expected_sha = str(artifact_meta["sha256"])
        expected_bytes = int(artifact_meta["bytes"])
        target_path = Path(target).resolve()

        # If the requested product output is already exact, no cache payload read/copy is needed.
        if self._matches(target_path, expected_sha256=expected_sha, expected_bytes=expected_bytes):
            return CacheRestore(
                True,
                key,
                "HIT_OUTPUT_ALREADY_CURRENT",
                expected_sha,
                expected_bytes,
                dict(manifest.get("metadata", {})),
                True,
            )

        cached_artifact = entry / _ARTIFACT_NAME
        if not self._matches(cached_artifact, expected_sha256=expected_sha, expected_bytes=expected_bytes):
            return CacheRestore(False, key, "MISS_CORRUPT_ARTIFACT")

        self._atomic_copy(cached_artifact, target_path)
        if not self._matches(target_path, expected_sha256=expected_sha, expected_bytes=expected_bytes):
            try:
                target_path.unlink()
            except FileNotFoundError:
                pass
            return CacheRestore(False, key, "MISS_RESTORE_VERIFY_FAILED")

        return CacheRestore(
            True,
            key,
            "HIT_RESTORED",
            expected_sha,
            expected_bytes,
            dict(manifest.get("metadata", {})),
            False,
        )

    def store(
        self,
        key: str,
        source: str | Path,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> CacheStore:
        source_path = Path(source).resolve()
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        artifact_bytes = source_path.stat().st_size
        artifact_sha = sha256_file(source_path)
        entry = self.entry_path(key)

        existing = self._load_manifest(entry)
        if existing is not None and self._valid_manifest(key, existing):
            meta = existing["artifact"]
            if (
                meta["sha256"] == artifact_sha
                and int(meta["bytes"]) == artifact_bytes
                and self._matches(entry / _ARTIFACT_NAME, expected_sha256=artifact_sha, expected_bytes=artifact_bytes)
            ):
                return CacheStore(key, artifact_sha, artifact_bytes, str(entry), True)

        stage_parent = entry.parent
        stage_parent.mkdir(parents=True, exist_ok=True)
        temp = stage_parent / f".{key}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
        temp.mkdir()
        try:
            cached_artifact = temp / _ARTIFACT_NAME
            self._atomic_copy(source_path, cached_artifact)
            manifest = {
                "cache_schema": CACHE_SCHEMA,
                "stage": self.stage,
                "key": key,
                "producer_fingerprint": self.producer_fingerprint,
                "artifact": {
                    "path": _ARTIFACT_NAME,
                    "sha256": artifact_sha,
                    "bytes": artifact_bytes,
                },
                "metadata": dict(metadata or {}),
            }
            manifest_path = temp / _MANIFEST_NAME
            manifest_path.write_bytes(_canonical_json(manifest) + b"\n")
            with manifest_path.open("rb") as handle:
                os.fsync(handle.fileno())

            if entry.exists():
                quarantine = entry.with_name(f".{entry.name}.corrupt-{uuid.uuid4().hex}")
                try:
                    os.replace(entry, quarantine)
                    shutil.rmtree(quarantine, ignore_errors=True)
                except OSError:
                    shutil.rmtree(entry, ignore_errors=True)
            os.replace(temp, entry)
        finally:
            shutil.rmtree(temp, ignore_errors=True)

        return CacheStore(key, artifact_sha, artifact_bytes, str(entry), False)
