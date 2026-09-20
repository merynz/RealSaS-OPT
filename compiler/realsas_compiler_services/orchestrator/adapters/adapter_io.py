from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from compiler.realsas_compiler_core.artifact_codec_v2 import write_ir_json
from compiler.realsas_compiler_core.types import QualificationError


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolved_path(raw: str) -> Path:
    value = os.path.expandvars(str(raw))
    if any(token in Path(value).parts for token in ("latest", "current", "newest")):
        raise QualificationError("ADAPTER_MOVING_ALIAS_FORBIDDEN")
    return Path(value).expanduser().resolve()


def load_file_ref(
    ref: dict,
    *,
    expected_schema: str | None = None,
    json_required: bool = True,
):
    path = resolved_path(str(ref.get("path", "")))
    expected = str(ref.get("sha256", ""))
    if not path.is_file() or len(expected) != 64:
        raise QualificationError("ADAPTER_FILE_REF_INCOMPLETE")
    actual = sha256_file(path)
    if actual != expected:
        raise QualificationError("ADAPTER_FILE_SHA_MISMATCH")
    if not json_required:
        return path
    payload = json.loads(path.read_text(encoding="utf-8"))
    if expected_schema is not None:
        actual_schema = str(
            payload.get("schema") or payload.get("schema_version") or ""
        )
        if actual_schema != expected_schema:
            raise QualificationError(
                f"ADAPTER_FILE_SCHEMA_MISMATCH:{actual_schema}!={expected_schema}"
            )
    return payload


def stage_output_payload(ctx: dict, stage_id: str, schema: str) -> dict:
    row = next(
        (row for row in ctx["ledger"]["stages"] if row["id"] == stage_id),
        None,
    )
    if row is None or row.get("status") not in {"PASS", "CACHE_HIT"}:
        raise QualificationError(f"ADAPTER_UPSTREAM_NOT_PASS:{stage_id}")
    matches = [
        output
        for output in row.get("outputs", ())
        if output.get("schema") == schema
    ]
    if len(matches) != 1:
        raise QualificationError(
            f"ADAPTER_UPSTREAM_SCHEMA_CARDINALITY:{stage_id}:{schema}:{len(matches)}"
        )
    output = matches[0]
    path = resolved_path(output["path"])
    if not path.is_file() or sha256_file(path) != output.get("sha256"):
        raise QualificationError(
            f"ADAPTER_UPSTREAM_OUTPUT_DRIFT:{stage_id}:{schema}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    actual_schema = str(
        payload.get("schema") or payload.get("schema_version") or ""
    )
    if actual_schema != schema:
        raise QualificationError(
            f"ADAPTER_UPSTREAM_EMBEDDED_SCHEMA_DRIFT:{stage_id}:{schema}"
        )
    return payload


def write_json(
    path: Path,
    payload: dict,
    *,
    authority_class: str,
    schema: str,
) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "authority_class": authority_class,
        "schema": schema,
    }


def write_ir(path: Path, value, *, authority_class: str) -> dict:
    write_ir_json(path, value)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "authority_class": authority_class,
        "schema": str(value.schema_version),
    }
