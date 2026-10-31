#!/usr/bin/env python3
from __future__ import annotations

"""Admit one exact product-model checkpoint as a DRAFT Platform ModelRelease.

This tool is deliberately narrower than a promotion operator. It validates the
family-specific runtime/inference checkpoint contract, binds the bytes to an
exact canonical-main EngineRelease/source SHA, imports the checkpoint into CAS,
and registers a DRAFT ModelRelease. It never creates evaluation PASS records and
never promotes RESEARCH/CANDIDATE/PRODUCTION channels.
"""

import argparse
from dataclasses import dataclass
import hashlib
import http.client
import json
import os
from pathlib import Path
from typing import Any, BinaryIO, Callable, Mapping
from urllib.parse import quote, urlsplit


@dataclass(frozen=True)
class FamilySpec:
    family: str
    architecture_id: str
    artifact_type: str
    artifact_schema_version: str
    payload_schema: str
    validator: Callable[[Path], Mapping[str, Any]]


def _validate_sha(value: str, length: int, code: str) -> str:
    raw = str(value or "").strip()
    if len(raw) != length or raw.lower() != raw or any(ch not in "0123456789abcdef" for ch in raw):
        raise RuntimeError(code)
    return raw


def _sha256_file(path: Path, chunk_size: int = 8 << 20) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
    if size <= 0:
        raise RuntimeError("PRODUCT_MODEL_ADMISSION_CHECKPOINT_EMPTY")
    return digest.hexdigest(), size


def _validate_iris(path: Path) -> Mapping[str, Any]:
    from models.iris.vnext.checkpoint_v1 import CHECKPOINT_SCHEMA_V1, load_checkpoint_v1
    from models.iris.vnext.model_v3 import IRISVNextEvidenceModelV3

    model = load_checkpoint_v1(path, device="cpu")
    try:
        if model.architecture_id != IRISVNextEvidenceModelV3.architecture_id:
            raise RuntimeError("PRODUCT_MODEL_ADMISSION_IRIS_ARCHITECTURE_DRIFT")
        return {
            "schema": CHECKPOINT_SCHEMA_V1,
            "architecture_id": model.architecture_id,
            "parameter_count": sum(int(parameter.numel()) for parameter in model.parameters()),
        }
    finally:
        del model


def _validate_tessa(path: Path) -> Mapping[str, Any]:
    from models.tessa.v3.checkpoint_v1 import CHECKPOINT_SCHEMA_V1, load_checkpoint_v1
    from models.tessa.v3.model_v3 import TESSAStructuralizerV3

    model = load_checkpoint_v1(path, device="cpu")
    try:
        if model.architecture_id != TESSAStructuralizerV3.architecture_id:
            raise RuntimeError("PRODUCT_MODEL_ADMISSION_TESSA_ARCHITECTURE_DRIFT")
        return {
            "schema": CHECKPOINT_SCHEMA_V1,
            "architecture_id": model.architecture_id,
            "parameter_count": sum(int(parameter.numel()) for parameter in model.parameters()),
        }
    finally:
        del model


def _validate_axis(path: Path) -> Mapping[str, Any]:
    from models.axis.rig_program_model_v6 import AXIS_V6_PRODUCT_ARCHITECTURE_ID
    from models.axis.runtime_checkpoint_v2 import (
        AXIS_RUNTIME_CHECKPOINT_SCHEMA_V2,
        load_axis_runtime_model_v2,
    )

    model, payload = load_axis_runtime_model_v2(path, device="cpu")
    try:
        if str(payload.get("architecture_id") or "") != AXIS_V6_PRODUCT_ARCHITECTURE_ID:
            raise RuntimeError("PRODUCT_MODEL_ADMISSION_AXIS_ARCHITECTURE_DRIFT")
        codec = payload.get("skin_codec_binding")
        if not isinstance(codec, Mapping):
            raise RuntimeError("PRODUCT_MODEL_ADMISSION_AXIS_SKIN_CODEC_BINDING_REQUIRED")
        return {
            "schema": AXIS_RUNTIME_CHECKPOINT_SCHEMA_V2,
            "architecture_id": AXIS_V6_PRODUCT_ARCHITECTURE_ID,
            "parameter_count": sum(int(parameter.numel()) for parameter in model.parameters()),
            "skin_codec_architecture_id": str(codec.get("architecture_id") or ""),
            "skin_codec_checkpoint_sha256": str(codec.get("checkpoint_sha256") or ""),
        }
    finally:
        del model


def _validate_mira(path: Path) -> Mapping[str, Any]:
    from models.mira.runtime_checkpoint_v2 import (
        MIRA_RUNTIME_CHECKPOINT_SCHEMA_V2,
        load_mira_runtime_model_v2,
    )
    from models.mira.skin_fsq_codec_v6 import MIRA_V6_CODEC_ID

    model, payload = load_mira_runtime_model_v2(path, device="cpu")
    try:
        if str(payload.get("architecture_id") or "") != MIRA_V6_CODEC_ID:
            raise RuntimeError("PRODUCT_MODEL_ADMISSION_MIRA_ARCHITECTURE_DRIFT")
        contract = payload.get("codec_contract")
        if not isinstance(contract, Mapping):
            raise RuntimeError("PRODUCT_MODEL_ADMISSION_MIRA_CODEC_CONTRACT_REQUIRED")
        return {
            "schema": MIRA_RUNTIME_CHECKPOINT_SCHEMA_V2,
            "architecture_id": MIRA_V6_CODEC_ID,
            "parameter_count": sum(int(parameter.numel()) for parameter in model.parameters()),
            "codebook_size": int(contract.get("codebook_size") or 0),
            "tokens_per_joint": int(contract.get("tokens_per_joint") or 0),
        }
    finally:
        del model


def family_spec(family: str) -> FamilySpec:
    normalized = str(family or "").strip().upper()
    if normalized == "IRIS":
        from models.iris.vnext.checkpoint_v1 import CHECKPOINT_SCHEMA_V1
        from models.iris.vnext.model_v3 import IRISVNextEvidenceModelV3
        return FamilySpec(
            family="IRIS",
            architecture_id=IRISVNextEvidenceModelV3.architecture_id,
            artifact_type="RealSaS.IRISVNextEvidenceCheckpoint",
            artifact_schema_version="v1",
            payload_schema=CHECKPOINT_SCHEMA_V1,
            validator=_validate_iris,
        )
    if normalized == "TESSA":
        from models.tessa.v3.checkpoint_v1 import CHECKPOINT_SCHEMA_V1
        from models.tessa.v3.model_v3 import TESSAStructuralizerV3
        return FamilySpec(
            family="TESSA",
            architecture_id=TESSAStructuralizerV3.architecture_id,
            artifact_type="RealSaS.TESSAStructuralizerCheckpoint",
            artifact_schema_version="v1",
            payload_schema=CHECKPOINT_SCHEMA_V1,
            validator=_validate_tessa,
        )
    if normalized == "AXIS":
        from models.axis.rig_program_model_v6 import AXIS_V6_PRODUCT_ARCHITECTURE_ID
        from models.axis.runtime_checkpoint_v2 import AXIS_RUNTIME_CHECKPOINT_SCHEMA_V2
        return FamilySpec(
            family="AXIS",
            architecture_id=AXIS_V6_PRODUCT_ARCHITECTURE_ID,
            artifact_type="RealSaS.AXISRuntimeCheckpoint",
            artifact_schema_version="v2",
            payload_schema=AXIS_RUNTIME_CHECKPOINT_SCHEMA_V2,
            validator=_validate_axis,
        )
    if normalized == "MIRA":
        from models.mira.runtime_checkpoint_v2 import MIRA_RUNTIME_CHECKPOINT_SCHEMA_V2
        from models.mira.skin_fsq_codec_v6 import MIRA_V6_CODEC_ID
        return FamilySpec(
            family="MIRA",
            architecture_id=MIRA_V6_CODEC_ID,
            artifact_type="RealSaS.MIRARuntimeCheckpoint",
            artifact_schema_version="v2",
            payload_schema=MIRA_RUNTIME_CHECKPOINT_SCHEMA_V2,
            validator=_validate_mira,
        )
    raise RuntimeError("PRODUCT_MODEL_ADMISSION_FAMILY_UNSUPPORTED:" + normalized)


def _connection(api_root: str, timeout: float) -> tuple[http.client.HTTPConnection, str]:
    parsed = urlsplit(api_root)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise RuntimeError("PRODUCT_MODEL_ADMISSION_API_ROOT_INVALID")
    cls = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
    return cls(parsed.hostname, parsed.port, timeout=timeout), parsed.path.rstrip("/")


def _headers(token: str | None, *, content_type: str | None = None, content_length: int | None = None) -> dict[str, str]:
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = "Bearer " + token
    if content_type:
        headers["Content-Type"] = content_type
    if content_length is not None:
        headers["Content-Length"] = str(content_length)
    return headers


def _decode_response(response: http.client.HTTPResponse) -> Any:
    raw = response.read()
    text = raw.decode("utf-8", errors="replace")
    if response.status >= 400:
        raise RuntimeError(f"PRODUCT_MODEL_ADMISSION_PLATFORM_HTTP_{response.status}:{text.strip()}")
    if not raw:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError("PRODUCT_MODEL_ADMISSION_PLATFORM_JSON_INVALID") from exc


def _request_json(
    api_root: str,
    method: str,
    path: str,
    *,
    token: str | None,
    payload: Any | None = None,
    timeout: float = 120.0,
) -> Any:
    conn, prefix = _connection(api_root, timeout)
    try:
        body = None if payload is None else json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        conn.request(
            method,
            prefix + path,
            body=body,
            headers=_headers(
                token,
                content_type="application/json" if body is not None else None,
                content_length=len(body) if body is not None else None,
            ),
        )
        return _decode_response(conn.getresponse())
    finally:
        conn.close()


def _put_stream(
    api_root: str,
    path: str,
    handle: BinaryIO,
    size: int,
    *,
    token: str | None,
) -> Any:
    conn, prefix = _connection(api_root, 1800.0)
    try:
        conn.putrequest("PUT", prefix + path)
        for key, value in _headers(token, content_type="application/octet-stream", content_length=size).items():
            conn.putheader(key, value)
        conn.endheaders()
        remaining = size
        while remaining:
            chunk = handle.read(min(8 << 20, remaining))
            if not chunk:
                raise RuntimeError("PRODUCT_MODEL_ADMISSION_UPLOAD_SOURCE_TRUNCATED")
            conn.send(chunk)
            remaining -= len(chunk)
        if handle.read(1):
            raise RuntimeError("PRODUCT_MODEL_ADMISSION_UPLOAD_SOURCE_GREW")
        return _decode_response(conn.getresponse())
    finally:
        conn.close()


def _put_file(api_root: str, path: Path, sha256: str, size: int, *, token: str | None) -> dict[str, Any]:
    with path.open("rb") as handle:
        obj = _put_stream(
            api_root,
            f"/v1/artifacts/bytes?sha256={quote(sha256)}",
            handle,
            size,
            token=token,
        )
    if not isinstance(obj, dict):
        raise RuntimeError("PRODUCT_MODEL_ADMISSION_CAS_RESPONSE_INVALID")
    if str(obj.get("content_sha256") or "") != sha256 or int(obj.get("size_bytes") or -1) != size:
        raise RuntimeError("PRODUCT_MODEL_ADMISSION_CAS_IDENTITY_DRIFT")
    return obj


def _require_canonical_source(
    api_root: str,
    engine_release_id: str,
    source_code_sha: str,
    *,
    token: str | None,
) -> None:
    rows = _request_json(
        api_root,
        "GET",
        f"/v1/releases/{quote(engine_release_id)}/sources",
        token=token,
    )
    if not isinstance(rows, list):
        raise RuntimeError("PRODUCT_MODEL_ADMISSION_RELEASE_SOURCES_INVALID")
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        if row.get("code_sha") == source_code_sha and row.get("authority") == "CANONICAL_MAIN":
            return
    raise RuntimeError("PRODUCT_MODEL_ADMISSION_CANONICAL_SOURCE_REQUIRED")


def registration_payload(
    spec: FamilySpec,
    *,
    name: str,
    checkpoint_artifact_id: str,
    checkpoint_sha256: str,
    engine_release_id: str,
    source_code_sha: str,
    actor: str,
    validation: Mapping[str, Any],
    originating_experiment_id: str | None = None,
    originating_attempt_id: str | None = None,
    supersedes_model_release_id: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "family": spec.family,
        "name": name,
        "architecture_id": spec.architecture_id,
        "checkpoint_artifact_id": checkpoint_artifact_id,
        "checkpoint_content_sha256": checkpoint_sha256,
        "engine_release_id": engine_release_id,
        "source_code_sha": source_code_sha,
        "initial_state": "DRAFT",
        "metadata": {
            "admission_contract": "RealSaS.ProductModelCheckpointAdmission.v1",
            "checkpoint_artifact_type": spec.artifact_type,
            "checkpoint_schema_version": spec.artifact_schema_version,
            "checkpoint_payload_schema": spec.payload_schema,
            "validated_by_family_runtime_loader": True,
            "runtime_validation": dict(validation),
            "evaluation_records_minted": False,
            "promotion_performed": False,
        },
        "created_by": actor,
    }
    if originating_experiment_id:
        payload["originating_experiment_id"] = originating_experiment_id
    if originating_attempt_id:
        payload["originating_attempt_id"] = originating_attempt_id
    if supersedes_model_release_id:
        payload["supersedes_model_release_id"] = supersedes_model_release_id
    return payload


def _register_or_verify(
    api_root: str,
    payload: dict[str, Any],
    *,
    token: str | None,
) -> dict[str, Any]:
    rows = _request_json(api_root, "GET", "/v1/model-releases", token=token)
    if not isinstance(rows, list):
        raise RuntimeError("PRODUCT_MODEL_ADMISSION_RELEASE_LIST_INVALID")
    for record in rows:
        if not isinstance(record, Mapping):
            continue
        if record.get("family") != payload["family"] or record.get("name") != payload["name"]:
            continue
        for key in (
            "architecture_id",
            "checkpoint_artifact_id",
            "checkpoint_content_sha256",
            "engine_release_id",
            "source_code_sha",
        ):
            if str(record.get(key)) != str(payload[key]):
                raise RuntimeError("PRODUCT_MODEL_ADMISSION_EXISTING_IDENTITY_DRIFT:" + key)
        if str(record.get("lifecycle_state") or "") not in {"DRAFT", "RESEARCH_QUALIFIED", "PRODUCT_CANDIDATE", "PRODUCTION"}:
            raise RuntimeError("PRODUCT_MODEL_ADMISSION_EXISTING_LIFECYCLE_INVALID")
        return dict(record)
    created = _request_json(api_root, "POST", "/v1/model-releases", token=token, payload=payload)
    if not isinstance(created, dict):
        raise RuntimeError("PRODUCT_MODEL_ADMISSION_REGISTER_RESPONSE_INVALID")
    return created


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-root", required=True, help="Platform API root, normally internal/backend-only")
    parser.add_argument("--family", required=True, choices=("IRIS", "TESSA", "AXIS", "MIRA"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--engine-release-id", required=True)
    parser.add_argument("--source-code-sha", required=True)
    parser.add_argument("--actor", required=True)
    parser.add_argument("--name")
    parser.add_argument("--originating-experiment-id")
    parser.add_argument("--originating-attempt-id")
    parser.add_argument("--supersedes-model-release-id")
    parser.add_argument(
        "--operator-token",
        default=os.environ.get("REALSAS_OPERATOR_TOKEN", "").strip(),
        help="Optional bearer token; defaults to REALSAS_OPERATOR_TOKEN",
    )
    parser.add_argument("--apply", action="store_true", help="Perform CAS import and DRAFT ModelRelease registration")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    path = Path(args.checkpoint).expanduser().resolve()
    if not path.is_file():
        raise RuntimeError("PRODUCT_MODEL_ADMISSION_CHECKPOINT_MISSING")
    source_sha = _validate_sha(args.source_code_sha, 40, "PRODUCT_MODEL_ADMISSION_SOURCE_SHA_INVALID")
    spec = family_spec(args.family)
    checkpoint_sha, checkpoint_size = _sha256_file(path)

    validation = dict(spec.validator(path))
    if validation.get("schema") != spec.payload_schema:
        raise RuntimeError("PRODUCT_MODEL_ADMISSION_PAYLOAD_SCHEMA_DRIFT")
    if validation.get("architecture_id") != spec.architecture_id:
        raise RuntimeError("PRODUCT_MODEL_ADMISSION_ARCHITECTURE_DRIFT")

    token = str(args.operator_token or "").strip() or None
    _require_canonical_source(
        args.api_root,
        args.engine_release_id,
        source_sha,
        token=token,
    )
    name = str(args.name or f"{spec.family.lower()}-product-{checkpoint_sha[:12]}").strip()
    if not name:
        raise RuntimeError("PRODUCT_MODEL_ADMISSION_NAME_REQUIRED")

    plan = {
        "schema": "RealSaS.ProductModelCheckpointAdmissionPlan.v1",
        "family": spec.family,
        "name": name,
        "architecture_id": spec.architecture_id,
        "checkpoint_path": str(path),
        "checkpoint_content_sha256": checkpoint_sha,
        "checkpoint_size_bytes": checkpoint_size,
        "checkpoint_artifact_type": spec.artifact_type,
        "checkpoint_schema_version": spec.artifact_schema_version,
        "checkpoint_payload_schema": spec.payload_schema,
        "engine_release_id": args.engine_release_id,
        "source_code_sha": source_sha,
        "runtime_validation": validation,
        "initial_lifecycle_state": "DRAFT",
        "evaluation_records_minted": False,
        "promotion_performed": False,
        "apply": bool(args.apply),
    }
    if not args.apply:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0

    obj = _put_file(
        args.api_root,
        path,
        checkpoint_sha,
        checkpoint_size,
        token=token,
    )
    imported = _request_json(
        args.api_root,
        "POST",
        "/v1/artifacts/import",
        token=token,
        payload={
            "artifact_type": spec.artifact_type,
            "schema_version": spec.artifact_schema_version,
            "object": obj,
            "source_uri": f"urn:realsas:model:{spec.family.lower()}:{checkpoint_sha}",
            "created_by": args.actor,
        },
    )
    if not isinstance(imported, Mapping) or not imported.get("id"):
        raise RuntimeError("PRODUCT_MODEL_ADMISSION_ARTIFACT_IMPORT_INVALID")

    payload = registration_payload(
        spec,
        name=name,
        checkpoint_artifact_id=str(imported["id"]),
        checkpoint_sha256=checkpoint_sha,
        engine_release_id=args.engine_release_id,
        source_code_sha=source_sha,
        actor=args.actor,
        validation=validation,
        originating_experiment_id=args.originating_experiment_id,
        originating_attempt_id=args.originating_attempt_id,
        supersedes_model_release_id=args.supersedes_model_release_id,
    )
    record = _register_or_verify(args.api_root, payload, token=token)
    result = {
        **plan,
        "apply": True,
        "checkpoint_artifact_id": str(imported["id"]),
        "model_release": record,
        "next_gate": "WRITE_REAL_EVALUATIONS_THEN_PROMOTE_BY_POLICY",
    }
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
