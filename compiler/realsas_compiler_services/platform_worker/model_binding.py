from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from compiler.realsas_compiler_services.orchestrator import mainline


@dataclass(frozen=True)
class MaterializedModelReleaseRef:
    family: str
    architecture_id: str
    model_release_id: str
    behavior_identity_sha256: str
    checkpoint_content_sha256: str
    checkpoint_path: Path
    schema: str = "RealSaS.MaterializedModelReleaseRef.v1"


def _lower_sha256(value: Any, code: str) -> str:
    raw = str(value or "")
    if len(raw) != 64 or raw.lower() != raw or any(ch not in "0123456789abcdef" for ch in raw):
        raise RuntimeError(code)
    return raw


def require_materialized_model_release(
    manifest: dict[str, Any],
    *,
    family: str,
    architecture_id: str,
) -> MaterializedModelReleaseRef:
    """Resolve one already-materialized immutable Attempt model, fail closed.

    This function never reads a model channel, registry alias, branch, or `latest`
    pointer.  Its only authority is the in-memory snapshot written by
    `materialize_model_releases` for the accepted Attempt.
    """

    family = str(family)
    architecture_id = str(architecture_id)
    if not family or family != family.strip():
        raise RuntimeError("ENGINE_MODEL_BINDING_FAMILY_INVALID")
    if not architecture_id or architecture_id != architecture_id.strip():
        raise RuntimeError("ENGINE_MODEL_BINDING_ARCHITECTURE_INVALID")

    rows = manifest.get("_platform_model_releases")
    if not isinstance(rows, list):
        raise RuntimeError("ENGINE_MODEL_BINDING_SNAPSHOT_REQUIRED")
    matching = [row for row in rows if isinstance(row, dict) and str(row.get("family")) == family]
    if not matching:
        raise RuntimeError("ENGINE_MODEL_BINDING_FAMILY_MISSING:" + family)
    if len(matching) != 1:
        raise RuntimeError("ENGINE_MODEL_BINDING_FAMILY_AMBIGUOUS:" + family)
    row = matching[0]
    if str(row.get("architecture_id") or "") != architecture_id:
        raise RuntimeError("ENGINE_MODEL_BINDING_ARCHITECTURE_DRIFT:" + family)

    run_id = str(manifest.get("run_id") or "")
    model_release_id = str(row.get("model_release_id") or "")
    if not run_id or not model_release_id:
        raise RuntimeError("ENGINE_MODEL_BINDING_IDENTITY_MISSING:" + family)
    behavior_sha = _lower_sha256(
        row.get("behavior_identity_sha256"), "ENGINE_MODEL_BINDING_BEHAVIOR_SHA_INVALID"
    )
    checkpoint = row.get("checkpoint")
    if not isinstance(checkpoint, dict):
        raise RuntimeError("ENGINE_MODEL_BINDING_CHECKPOINT_REQUIRED:" + family)
    checkpoint_sha = _lower_sha256(
        checkpoint.get("content_sha256"), "ENGINE_MODEL_BINDING_CHECKPOINT_SHA_INVALID"
    )
    size_bytes = checkpoint.get("size_bytes")
    if isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes < 0:
        raise RuntimeError("ENGINE_MODEL_BINDING_CHECKPOINT_SIZE_INVALID:" + family)

    raw_path = str(checkpoint.get("path") or "")
    if not raw_path:
        raise RuntimeError("ENGINE_MODEL_BINDING_CHECKPOINT_PATH_REQUIRED:" + family)
    path = Path(raw_path).resolve()
    expected_root = (
        mainline.authority_root() / "runs" / run_id / "models" / family / model_release_id
    ).resolve()
    if expected_root not in path.parents:
        raise RuntimeError("ENGINE_MODEL_BINDING_CHECKPOINT_PATH_ESCAPE:" + family)
    if not path.is_file():
        raise RuntimeError("ENGINE_MODEL_BINDING_CHECKPOINT_MISSING:" + family)
    if path.stat().st_size != size_bytes:
        raise RuntimeError("ENGINE_MODEL_BINDING_CHECKPOINT_SIZE_DRIFT:" + family)
    if mainline.sha256_file(path) != checkpoint_sha:
        raise RuntimeError("ENGINE_MODEL_BINDING_CHECKPOINT_CONTENT_DRIFT:" + family)

    return MaterializedModelReleaseRef(
        family=family,
        architecture_id=architecture_id,
        model_release_id=model_release_id,
        behavior_identity_sha256=behavior_sha,
        checkpoint_content_sha256=checkpoint_sha,
        checkpoint_path=path,
    )
