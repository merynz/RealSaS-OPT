from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import uuid
from pathlib import Path
from typing import Any, Callable

from compiler.realsas_compiler_services.orchestrator import mainline


_FAMILY_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_LOWER_HEX_40_RE = re.compile(r"^[0-9a-f]{40}$")
_LOWER_HEX_64_RE = re.compile(r"^[0-9a-f]{64}$")


def _canonical_uuid(value: Any, code: str) -> str:
    raw = str(value or "")
    try:
        parsed = uuid.UUID(raw)
    except (ValueError, AttributeError, TypeError) as exc:
        raise RuntimeError(code) from exc
    if str(parsed) != raw:
        raise RuntimeError(code)
    return raw


def _behavior_identity_payload(model: dict[str, Any]) -> dict[str, str]:
    family = str(model.get("family") or "")
    if not _FAMILY_RE.fullmatch(family):
        raise RuntimeError("ENGINE_MODEL_FAMILY_INVALID")
    architecture_id = str(model.get("architecture_id") or "")
    if not architecture_id or architecture_id != architecture_id.strip():
        raise RuntimeError("ENGINE_MODEL_ARCHITECTURE_INVALID")
    source_code_sha = str(model.get("source_code_sha") or "")
    if not _LOWER_HEX_40_RE.fullmatch(source_code_sha):
        raise RuntimeError("ENGINE_MODEL_SOURCE_SHA_INVALID")
    checkpoint = dict(model.get("checkpoint") or {})
    checkpoint_sha = str(checkpoint.get("content_sha256") or "")
    if not _LOWER_HEX_64_RE.fullmatch(checkpoint_sha):
        raise RuntimeError("ENGINE_MODEL_CHECKPOINT_SHA_INVALID")
    return {
        "schema": "RealSaS.ModelReleaseBehaviorIdentity.v1",
        "family": family,
        "model_release_id": _canonical_uuid(model.get("model_release_id"), "ENGINE_MODEL_RELEASE_ID_INVALID"),
        "architecture_id": architecture_id,
        "model_engine_release_id": _canonical_uuid(
            model.get("model_engine_release_id"), "ENGINE_MODEL_ENGINE_RELEASE_ID_INVALID"
        ),
        "source_code_sha": source_code_sha,
        "checkpoint_artifact_id": _canonical_uuid(
            checkpoint.get("id"), "ENGINE_MODEL_CHECKPOINT_ARTIFACT_ID_INVALID"
        ),
        "checkpoint_content_sha256": checkpoint_sha,
    }


def behavior_identity_sha256(model: dict[str, Any]) -> str:
    """Mirror platform/internal/semantic CanonicalJSON exactly for this payload.

    Go marshals the struct, decodes it to a generic map, then json.Encoder emits
    map keys lexicographically and appends a final newline. The payload contains
    only strings, so sort_keys=True + compact separators + newline is the exact
    cross-language encoding contract.
    """
    payload = _behavior_identity_payload(model)
    raw = (
        json.dumps(
            payload,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n"
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _materialize_checkpoint(
    *,
    root: Path,
    family: str,
    model_release_id: str,
    checkpoint: dict[str, Any],
    data: bytes,
) -> Path:
    digest = str(checkpoint["content_sha256"])
    target_root = (root / family / model_release_id).resolve()
    target_root.mkdir(parents=True, exist_ok=True)
    target = (target_root / f"{digest}.checkpoint").resolve()
    if target_root not in target.parents:
        raise RuntimeError("ENGINE_MODEL_CHECKPOINT_PATH_ESCAPE")
    if target.exists():
        if target.read_bytes() != data:
            raise RuntimeError("ENGINE_MODEL_CHECKPOINT_EXISTING_BYTES_DRIFT")
        return target
    with tempfile.NamedTemporaryFile(dir=target_root, delete=False) as handle:
        handle.write(data)
        temporary = Path(handle.name)
    os.replace(temporary, target)
    return target


def materialize_model_releases(
    *,
    manifest: dict[str, Any],
    models: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    read_object: Callable[[dict[str, Any]], bytes],
) -> list[dict[str, Any]]:
    """Verify and materialize the immutable Attempt model snapshot.

    The sealed run-manifest on disk is never changed. Only this in-memory copy
    receives `_platform_model_releases` and `_platform_model_snapshot_sha256`.
    No model framework is imported and no inference/training is performed here.
    """
    if not models:
        manifest.pop("_platform_model_releases", None)
        manifest.pop("_platform_model_snapshot_sha256", None)
        return []
    run_id = str(manifest.get("run_id") or "")
    if not run_id:
        raise RuntimeError("ENGINE_MODEL_RUN_ID_REQUIRED")

    normalized: list[tuple[str, dict[str, Any], dict[str, str], str]] = []
    seen: set[str] = set()
    for raw_model in models:
        model = dict(raw_model)
        identity = _behavior_identity_payload(model)
        family = identity["family"]
        if family in seen:
            raise RuntimeError("ENGINE_MODEL_DUPLICATE_FAMILY:" + family)
        seen.add(family)
        checkpoint = dict(model.get("checkpoint") or {})
        if str(checkpoint.get("role") or "") != "model:" + family:
            raise RuntimeError("ENGINE_MODEL_CHECKPOINT_ROLE_DRIFT:" + family)
        if not str(checkpoint.get("artifact_type") or "") or not str(checkpoint.get("schema_version") or ""):
            raise RuntimeError("ENGINE_MODEL_CHECKPOINT_TYPE_REQUIRED:" + family)
        semantic_sha = str(checkpoint.get("semantic_sha256") or "")
        if not _LOWER_HEX_64_RE.fullmatch(semantic_sha):
            raise RuntimeError("ENGINE_MODEL_CHECKPOINT_SEMANTIC_SHA_INVALID:" + family)
        size_bytes = checkpoint.get("size_bytes")
        if isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes < 0:
            raise RuntimeError("ENGINE_MODEL_CHECKPOINT_SIZE_INVALID:" + family)
        digest = behavior_identity_sha256(model)
        normalized.append((family, model, identity, digest))

    normalized.sort(key=lambda row: row[0])
    root = (mainline.authority_root() / "runs" / run_id / "models").resolve()
    root.mkdir(parents=True, exist_ok=True)
    materialized: list[dict[str, Any]] = []
    snapshot_rows: list[dict[str, str]] = []
    for family, model, identity, behavior_sha in normalized:
        checkpoint = dict(model["checkpoint"])
        data = read_object(checkpoint)
        actual = hashlib.sha256(data).hexdigest()
        if actual != identity["checkpoint_content_sha256"]:
            raise RuntimeError("ENGINE_MODEL_CHECKPOINT_CAS_DRIFT:" + family)
        if len(data) != int(checkpoint["size_bytes"]):
            raise RuntimeError("ENGINE_MODEL_CHECKPOINT_SIZE_DRIFT:" + family)
        path = _materialize_checkpoint(
            root=root,
            family=family,
            model_release_id=identity["model_release_id"],
            checkpoint=checkpoint,
            data=data,
        )
        materialized.append(
            {
                **identity,
                "behavior_identity_sha256": behavior_sha,
                "checkpoint": {
                    "artifact_type": str(checkpoint["artifact_type"]),
                    "schema_version": str(checkpoint["schema_version"]),
                    "semantic_sha256": str(checkpoint["semantic_sha256"]),
                    "content_sha256": identity["checkpoint_content_sha256"],
                    "size_bytes": int(checkpoint["size_bytes"]),
                    "path": str(path),
                },
            }
        )
        snapshot_rows.append(
            {"family": family, "behavior_identity_sha256": behavior_sha}
        )

    snapshot_sha = mainline.content_sha256(
        {"schema": "RealSaS.ModelReleaseSnapshot.v1", "models": snapshot_rows}
    )
    manifest["_platform_model_releases"] = materialized
    manifest["_platform_model_snapshot_sha256"] = snapshot_sha
    return materialized
