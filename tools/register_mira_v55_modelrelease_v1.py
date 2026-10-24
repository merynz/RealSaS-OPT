#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
from pathlib import Path
from typing import Any, BinaryIO
from urllib.parse import urlsplit


RECEIPT_SCHEMA = "RealSaS.MIRAV55RuntimeRecovery.v1"
RUNTIME_ARTIFACT_TYPE = "RealSaS.MIRARuntimeCheckpoint"
RUNTIME_SCHEMA_VERSION = "v1"
RECOVERY_EVIDENCE_ARTIFACT_TYPE = "RealSaS.MIRAV55RuntimeRecoveryEvidence"
RECOVERY_EVIDENCE_SCHEMA_VERSION = "v1"
EXPECTED_COMPUTE_CONTRACT = "CUDA_BF16_BACKBONE_FP32_READOUT_TF32_OFF"
FAMILY = "MIRA"


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
    return digest.hexdigest(), size


def _load_receipt(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    data = json.loads(raw)
    if data.get("schema") != RECEIPT_SCHEMA:
        raise RuntimeError("MIRA_MODELRELEASE_RECOVERY_RECEIPT_SCHEMA_DRIFT")
    authority = str(data.get("authority", ""))
    if "NO_NEW_SCIENTIFIC_PASS" not in authority:
        raise RuntimeError("MIRA_MODELRELEASE_RECOVERY_AUTHORITY_DRIFT")

    runtime = data.get("runtime_package") or {}
    historical = data.get("historical_components") or {}
    compute = data.get("historical_compute_contract") or {}
    parity = data.get("promoted_readout_parity") or {}

    if runtime.get("schema") != "RealSaS.MIRA.RuntimeCheckpoint.v1":
        raise RuntimeError("MIRA_MODELRELEASE_RUNTIME_SCHEMA_DRIFT")
    if runtime.get("architecture_id") != historical.get("architecture_id"):
        raise RuntimeError("MIRA_MODELRELEASE_ARCHITECTURE_ID_DRIFT")
    if len(str(runtime.get("content_sha256", ""))) != 64 or int(runtime.get("size_bytes", -1)) <= 0:
        raise RuntimeError("MIRA_MODELRELEASE_RUNTIME_IDENTITY_INVALID")
    if runtime.get("self_contained_load_verified") is not True:
        raise RuntimeError("MIRA_MODELRELEASE_RUNTIME_LOAD_NOT_VERIFIED")
    if runtime.get("fit_executed") is not False or runtime.get("training_executed") is not False:
        raise RuntimeError("MIRA_MODELRELEASE_RECOVERY_EXECUTED_FIT_OR_TRAINING")
    if runtime.get("scientific_pass_minted") is not False:
        raise RuntimeError("MIRA_MODELRELEASE_RECOVERY_MINTED_SCIENTIFIC_PASS")
    if compute.get("contract_id") != EXPECTED_COMPUTE_CONTRACT:
        raise RuntimeError("MIRA_MODELRELEASE_COMPUTE_CONTRACT_DRIFT")
    if compute.get("cross_device_parity_proven") is not False:
        raise RuntimeError("MIRA_MODELRELEASE_CROSS_DEVICE_PARITY_CLAIM_DRIFT")
    if compute.get("cpu_fp32_product_execution_admitted") is not False:
        raise RuntimeError("MIRA_MODELRELEASE_CPU_EXECUTION_CLAIM_DRIFT")
    if parity.get("torch_equal_logits") is not True or parity.get("torch_equal_weights") is not True:
        raise RuntimeError("MIRA_MODELRELEASE_READOUT_PARITY_NOT_EXACT")
    if float(parity.get("max_abs_logit_delta", 1.0)) != 0.0 or float(parity.get("max_abs_weight_delta", 1.0)) != 0.0:
        raise RuntimeError("MIRA_MODELRELEASE_READOUT_PARITY_DELTA_NONZERO")
    return data


def _connection(api_root: str, timeout: float = 120.0) -> tuple[http.client.HTTPConnection, str]:
    parsed = urlsplit(api_root)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise RuntimeError("api root must be an absolute http(s) URL")
    cls = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
    conn = cls(parsed.hostname, parsed.port, timeout=timeout)
    prefix = parsed.path.rstrip("/")
    return conn, prefix


def _decode_response(response: http.client.HTTPResponse) -> Any:
    raw = response.read()
    text = raw.decode("utf-8", errors="replace")
    if response.status >= 400:
        raise RuntimeError(f"PLATFORM_HTTP_{response.status}: {text.strip()}")
    if not raw:
        return None
    return json.loads(raw)


def _request_json(api_root: str, method: str, path: str, payload: Any | None = None) -> Any:
    conn, prefix = _connection(api_root)
    try:
        body = None if payload is None else json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        headers = {"Accept": "application/json"}
        if body is not None:
            headers["Content-Type"] = "application/json"
            headers["Content-Length"] = str(len(body))
        conn.request(method, prefix + path, body=body, headers=headers)
        return _decode_response(conn.getresponse())
    finally:
        conn.close()


def _put_stream(api_root: str, path: str, handle: BinaryIO, size: int) -> Any:
    conn, prefix = _connection(api_root, timeout=900.0)
    try:
        conn.putrequest("PUT", prefix + path)
        conn.putheader("Content-Type", "application/octet-stream")
        conn.putheader("Accept", "application/json")
        conn.putheader("Content-Length", str(size))
        conn.endheaders()
        remaining = size
        while remaining:
            chunk = handle.read(min(8 << 20, remaining))
            if not chunk:
                raise RuntimeError("MIRA_MODELRELEASE_UPLOAD_SOURCE_TRUNCATED")
            conn.send(chunk)
            remaining -= len(chunk)
        if handle.read(1):
            raise RuntimeError("MIRA_MODELRELEASE_UPLOAD_SOURCE_GREW")
        return _decode_response(conn.getresponse())
    finally:
        conn.close()


def _put_file(api_root: str, path: Path, sha256: str, size: int) -> dict[str, Any]:
    with path.open("rb") as handle:
        obj = _put_stream(api_root, f"/v1/artifacts/bytes?sha256={sha256}", handle, size)
    if obj.get("content_sha256") != sha256 or int(obj.get("size_bytes", -1)) != size:
        raise RuntimeError("MIRA_MODELRELEASE_PLATFORM_CAS_IDENTITY_DRIFT")
    return obj


def _import_artifact(
    api_root: str,
    *,
    artifact_type: str,
    schema_version: str,
    obj: dict[str, Any],
    source_uri: str,
    actor: str,
) -> dict[str, Any]:
    return _request_json(
        api_root,
        "POST",
        "/v1/artifacts/import",
        {
            "artifact_type": artifact_type,
            "schema_version": schema_version,
            "object": obj,
            "source_uri": source_uri,
            "created_by": actor,
        },
    )


def _require_canonical_source(api_root: str, engine_release_id: str, source_code_sha: str) -> None:
    bindings = _request_json(api_root, "GET", f"/v1/releases/{engine_release_id}/sources")
    for binding in bindings:
        if binding.get("code_sha") == source_code_sha and binding.get("authority") == "CANONICAL_MAIN":
            return
    raise RuntimeError("MIRA_MODELRELEASE_ENGINE_RELEASE_CANONICAL_SOURCE_REQUIRED")


def _model_name(runtime_sha: str) -> str:
    return f"mira-v5.5-axis541-{runtime_sha[:12]}"


def _registration_payload(
    receipt: dict[str, Any],
    *,
    checkpoint_artifact_id: str,
    engine_release_id: str,
    source_code_sha: str,
    actor: str,
    name: str,
    evidence_artifact_id: str,
) -> dict[str, Any]:
    runtime = receipt["runtime_package"]
    compute = receipt["historical_compute_contract"]
    return {
        "family": FAMILY,
        "name": name,
        "architecture_id": runtime["architecture_id"],
        "checkpoint_artifact_id": checkpoint_artifact_id,
        "checkpoint_content_sha256": runtime["content_sha256"],
        "engine_release_id": engine_release_id,
        "source_code_sha": source_code_sha,
        "initial_state": "DRAFT",
        "metadata": {
            "runtime_checkpoint_schema": runtime["schema"],
            "runtime_checkpoint_size_bytes": runtime["size_bytes"],
            "historical_mira_source_commit": receipt["historical_mira_source_commit"],
            "recovery_evidence_artifact_id": evidence_artifact_id,
            "compute_contract_id": compute["contract_id"],
            "cross_device_parity_proven": False,
            "cpu_fp32_product_execution_admitted": False,
            "fit_executed_during_recovery": False,
            "training_executed_during_recovery": False,
            "new_scientific_pass_minted_by_recovery": False,
            "generalization_claimed": False,
            "product_skin_authority_claimed": False,
        },
        "created_by": actor,
    }


def _register_or_verify(api_root: str, payload: dict[str, Any]) -> dict[str, Any]:
    releases = _request_json(api_root, "GET", "/v1/model-releases")
    for record in releases:
        if record.get("family") != payload["family"] or record.get("name") != payload["name"]:
            continue
        checks = {
            "architecture_id": payload["architecture_id"],
            "checkpoint_artifact_id": payload["checkpoint_artifact_id"],
            "checkpoint_content_sha256": payload["checkpoint_content_sha256"],
            "engine_release_id": payload["engine_release_id"],
            "source_code_sha": payload["source_code_sha"],
        }
        for key, expected in checks.items():
            if str(record.get(key)) != str(expected):
                raise RuntimeError(f"MIRA_MODELRELEASE_EXISTING_IDENTITY_DRIFT:{key}")
        if record.get("lifecycle_state") != "DRAFT":
            raise RuntimeError("MIRA_MODELRELEASE_EXISTING_RELEASE_NOT_DRAFT")
        return record
    return _request_json(api_root, "POST", "/v1/model-releases", payload)


def _set_evaluation(
    api_root: str,
    *,
    model_release_id: str,
    scope: str,
    result: str,
    actor: str,
    evidence_artifact_id: str | None = None,
    details: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "model_release_id": model_release_id,
        "scope": scope,
        "result": result,
        "actor": actor,
        "details": details or {},
    }
    if evidence_artifact_id is not None:
        payload["evidence_artifact_id"] = evidence_artifact_id
    _request_json(api_root, "POST", "/v1/model-releases/evaluations", payload)


def main() -> int:
    parser = argparse.ArgumentParser(description="Register exact recovered MIRA V5.5 runtime as a DRAFT Platform ModelRelease")
    parser.add_argument("--api-root", required=True)
    parser.add_argument("--runtime-checkpoint", required=True)
    parser.add_argument("--recovery-receipt", default="canonical/MIRA_V55_RUNTIME_RECOVERY_V1.json")
    parser.add_argument("--engine-release-id", required=True)
    parser.add_argument("--source-code-sha", required=True)
    parser.add_argument("--actor", required=True)
    parser.add_argument("--name")
    parser.add_argument("--apply", action="store_true", help="perform Platform mutations; default is validation/plan only")
    args = parser.parse_args()

    receipt_path = Path(args.recovery_receipt).resolve()
    runtime_path = Path(args.runtime_checkpoint).resolve()
    receipt = _load_receipt(receipt_path)
    runtime = receipt["runtime_package"]

    actual_sha, actual_size = _sha256_file(runtime_path)
    if actual_sha != runtime["content_sha256"] or actual_size != int(runtime["size_bytes"]):
        raise RuntimeError("MIRA_MODELRELEASE_RUNTIME_FILE_IDENTITY_DRIFT")
    if len(args.source_code_sha) != 40 or args.source_code_sha.lower() != args.source_code_sha:
        raise RuntimeError("MIRA_MODELRELEASE_SOURCE_SHA_INVALID")

    _require_canonical_source(args.api_root, args.engine_release_id, args.source_code_sha)
    name = args.name or _model_name(actual_sha)

    plan = {
        "schema": "RealSaS.MIRAV55ModelReleaseRegistrationPlan.v1",
        "family": FAMILY,
        "name": name,
        "architecture_id": runtime["architecture_id"],
        "runtime_content_sha256": actual_sha,
        "runtime_size_bytes": actual_size,
        "engine_release_id": args.engine_release_id,
        "source_code_sha": args.source_code_sha,
        "initial_state": "DRAFT",
        "promotion_performed": False,
        "apply": bool(args.apply),
    }
    if not args.apply:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0

    runtime_obj = _put_file(args.api_root, runtime_path, actual_sha, actual_size)
    runtime_import = _import_artifact(
        args.api_root,
        artifact_type=RUNTIME_ARTIFACT_TYPE,
        schema_version=RUNTIME_SCHEMA_VERSION,
        obj=runtime_obj,
        source_uri=f"urn:realsas:mira:v5.5:runtime:{actual_sha}",
        actor=args.actor,
    )

    receipt_sha, receipt_size = _sha256_file(receipt_path)
    receipt_obj = _put_file(args.api_root, receipt_path, receipt_sha, receipt_size)
    evidence_import = _import_artifact(
        args.api_root,
        artifact_type=RECOVERY_EVIDENCE_ARTIFACT_TYPE,
        schema_version=RECOVERY_EVIDENCE_SCHEMA_VERSION,
        obj=receipt_obj,
        source_uri=f"urn:realsas:mira:v5.5:runtime-recovery:{receipt_sha}",
        actor=args.actor,
    )

    registration = _registration_payload(
        receipt,
        checkpoint_artifact_id=runtime_import["artifact_id"],
        engine_release_id=args.engine_release_id,
        source_code_sha=args.source_code_sha,
        actor=args.actor,
        name=name,
        evidence_artifact_id=evidence_import["artifact_id"],
    )
    model = _register_or_verify(args.api_root, registration)
    model_id = model["id"]
    evidence_id = evidence_import["artifact_id"]

    _set_evaluation(
        args.api_root,
        model_release_id=model_id,
        scope="RUNTIME_PACKAGE_IDENTITY_V1",
        result="PASS",
        actor=args.actor,
        evidence_artifact_id=evidence_id,
        details={
            "content_sha256": actual_sha,
            "size_bytes": actual_size,
            "self_contained_load_verified": True,
        },
    )
    parity = receipt["promoted_readout_parity"]
    _set_evaluation(
        args.api_root,
        model_release_id=model_id,
        scope="HISTORICAL_READOUT_PARITY_V55",
        result="PASS",
        actor=args.actor,
        evidence_artifact_id=evidence_id,
        details={
            "max_abs_logit_delta": parity["max_abs_logit_delta"],
            "max_abs_weight_delta": parity["max_abs_weight_delta"],
            "torch_equal_logits": parity["torch_equal_logits"],
            "torch_equal_weights": parity["torch_equal_weights"],
        },
    )
    _set_evaluation(
        args.api_root,
        model_release_id=model_id,
        scope="HISTORICAL_COMPUTE_CONTRACT_V1",
        result="PASS",
        actor=args.actor,
        evidence_artifact_id=evidence_id,
        details={
            "contract_id": receipt["historical_compute_contract"]["contract_id"],
            "cross_device_parity_proven": False,
            "cpu_fp32_product_execution_admitted": False,
        },
    )
    for scope in ("FIT8", "LOFO", "UNSEEN_GENERALIZATION"):
        _set_evaluation(
            args.api_root,
            model_release_id=model_id,
            scope=scope,
            result="NOT_RUN",
            actor=args.actor,
        )

    final = _request_json(args.api_root, "GET", f"/v1/model-releases/{model_id}")
    if final.get("lifecycle_state") != "DRAFT":
        raise RuntimeError("MIRA_MODELRELEASE_DRAFT_STATE_DRIFT")
    plan.update(
        {
            "runtime_artifact_id": runtime_import["artifact_id"],
            "recovery_evidence_artifact_id": evidence_id,
            "model_release_id": model_id,
            "lifecycle_state": final["lifecycle_state"],
            "evaluations": final.get("evaluations", []),
        }
    )
    print(json.dumps(plan, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
