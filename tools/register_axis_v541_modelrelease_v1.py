#!/usr/bin/env python3
from __future__ import annotations

"""Register the exact AXIS V5.4.1 runtime as a DRAFT ModelRelease.

This translates already-sealed historical FIT1 evidence plus deterministic
runtime packaging evidence into Platform records. It does not train, fit,
reselect parents, mint a new scientific PASS, or promote a model channel.
"""

import argparse
import hashlib
import http.client
import json
from pathlib import Path
from typing import Any, BinaryIO, Mapping
from urllib.parse import urlsplit

from models.axis.runtime_checkpoint_v1 import (
    AXIS_RUNTIME_CHECKPOINT_SCHEMA_V1,
    AXIS_V541_ARCHITECTURE_ID,
    load_axis_runtime_model_v1,
)

FAMILY = "AXIS"
RUNTIME_ARTIFACT_TYPE = "RealSaS.AXISRuntimeCheckpoint"
RUNTIME_SCHEMA_VERSION = "v1"
PACKAGE_EVIDENCE_ARTIFACT_TYPE = "RealSaS.AXISRuntimePackageEvidence"
PACKAGE_EVIDENCE_SCHEMA_VERSION = "v1"
HISTORICAL_EVIDENCE_ARTIFACT_TYPE = "RealSaS.AXISV541HistoricalTerminalEvidence"
HISTORICAL_EVIDENCE_SCHEMA_VERSION = "v1"
TRANSPORT_EVIDENCE_ARTIFACT_TYPE = "RealSaS.AXISV541SourceTransportEvidence"
TRANSPORT_EVIDENCE_SCHEMA_VERSION = "v1"

BASE_SHA = "1c124f876858f6f5b5fd0d4639fc7f2b596a25e962d19de8799ecd64ae59b08f"
V53_SHA = "5f527a51c676bf9ef2d82691f305c139b365e921867623b7338e0848c7dea55a"
V541_SHA = "4d484c325c3ae2a9a91df422a820cc5ccdf01c804adef4259ff5398cf30e12fb"


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


def _connection(api_root: str, timeout: float = 120.0) -> tuple[http.client.HTTPConnection, str]:
    parsed = urlsplit(api_root)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise RuntimeError("api root must be an absolute http(s) URL")
    cls = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
    return cls(parsed.hostname, parsed.port, timeout=timeout), parsed.path.rstrip("/")


def _decode_response(response: http.client.HTTPResponse) -> Any:
    raw = response.read()
    text = raw.decode("utf-8", errors="replace")
    if response.status >= 400:
        raise RuntimeError(f"PLATFORM_HTTP_{response.status}: {text.strip()}")
    return None if not raw else json.loads(raw)


def _request_json(api_root: str, method: str, path: str, payload: Any | None = None) -> Any:
    conn, prefix = _connection(api_root)
    try:
        body = None if payload is None else json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        headers = {"Accept": "application/json"}
        if body is not None:
            headers.update({"Content-Type": "application/json", "Content-Length": str(len(body))})
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
                raise RuntimeError("AXIS_MODELRELEASE_UPLOAD_SOURCE_TRUNCATED")
            conn.send(chunk)
            remaining -= len(chunk)
        if handle.read(1):
            raise RuntimeError("AXIS_MODELRELEASE_UPLOAD_SOURCE_GREW")
        return _decode_response(conn.getresponse())
    finally:
        conn.close()


def _put_file(api_root: str, path: Path, sha256: str, size: int) -> dict[str, Any]:
    with path.open("rb") as handle:
        obj = _put_stream(api_root, f"/v1/artifacts/bytes?sha256={sha256}", handle, size)
    if obj.get("content_sha256") != sha256 or int(obj.get("size_bytes", -1)) != size:
        raise RuntimeError("AXIS_MODELRELEASE_PLATFORM_CAS_IDENTITY_DRIFT")
    return obj


def _import_artifact(api_root: str, *, artifact_type: str, schema_version: str,
                     obj: dict[str, Any], source_uri: str, actor: str) -> dict[str, Any]:
    return _request_json(api_root, "POST", "/v1/artifacts/import", {
        "artifact_type": artifact_type,
        "schema_version": schema_version,
        "object": obj,
        "source_uri": source_uri,
        "created_by": actor,
    })


def _require_canonical_source(api_root: str, engine_release_id: str, source_code_sha: str) -> None:
    for binding in _request_json(api_root, "GET", f"/v1/releases/{engine_release_id}/sources"):
        if binding.get("code_sha") == source_code_sha and binding.get("authority") == "CANONICAL_MAIN":
            return
    raise RuntimeError("AXIS_MODELRELEASE_ENGINE_RELEASE_CANONICAL_SOURCE_REQUIRED")


def _mapping(value: Any, code: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(code)
    return value


def _load_torch_mapping(path: Path) -> Mapping[str, Any]:
    import torch
    value = torch.load(path, map_location="cpu", weights_only=False)
    return _mapping(value, "AXIS_MODELRELEASE_TORCH_PAYLOAD_NOT_MAPPING")


def _terminal_pass(composite: Mapping[str, Any]) -> bool:
    terminal = composite.get("terminal_eval")
    if not isinstance(terminal, Mapping):
        return False
    if terminal.get("pass") is True:
        return True
    report = terminal.get("report")
    return isinstance(report, Mapping) and report.get("pass") is True


def _validate_inputs(runtime_path: Path, package_receipt_path: Path,
                     historical_path: Path, transport_receipt_path: Path) -> dict[str, Any]:
    package = json.loads(package_receipt_path.read_text(encoding="utf-8"))
    if package.get("schema") != "RealSaS.AXIS.RuntimePackageReceipt.v1":
        raise RuntimeError("AXIS_MODELRELEASE_PACKAGE_RECEIPT_SCHEMA_DRIFT")
    if package.get("architecture_id") != AXIS_V541_ARCHITECTURE_ID:
        raise RuntimeError("AXIS_MODELRELEASE_PACKAGE_ARCHITECTURE_DRIFT")
    expected_sources = {
        "base_checkpoint_sha256": BASE_SHA,
        "v53_progress_sha256": V53_SHA,
        "v541_composite_sha256": V541_SHA,
    }
    for key, expected in expected_sources.items():
        if package.get(key) != expected:
            raise RuntimeError(f"AXIS_MODELRELEASE_PACKAGE_SOURCE_DRIFT:{key}")

    runtime_sha, runtime_size = _sha256_file(runtime_path)
    if runtime_sha != package.get("content_sha256") or runtime_size <= 0:
        raise RuntimeError("AXIS_MODELRELEASE_RUNTIME_IDENTITY_DRIFT")
    model, runtime_payload = load_axis_runtime_model_v1(runtime_path, device="cpu")
    del model
    if runtime_payload.get("schema") != AXIS_RUNTIME_CHECKPOINT_SCHEMA_V1:
        raise RuntimeError("AXIS_MODELRELEASE_RUNTIME_SCHEMA_DRIFT")
    if runtime_payload.get("architecture_id") != AXIS_V541_ARCHITECTURE_ID:
        raise RuntimeError("AXIS_MODELRELEASE_RUNTIME_ARCHITECTURE_DRIFT")
    sources = _mapping(runtime_payload.get("source_components"), "AXIS_MODELRELEASE_RUNTIME_SOURCES_MISSING")
    for key, expected in expected_sources.items():
        if sources.get(key) != expected:
            raise RuntimeError(f"AXIS_MODELRELEASE_RUNTIME_SOURCE_DRIFT:{key}")
    rules = _mapping(runtime_payload.get("runtime_rules"), "AXIS_MODELRELEASE_RUNTIME_RULES_MISSING")
    required_rules = {
        "self_contained_inference_state": True,
        "hard_causal_parent_authority": True,
        "compiler_parent_reselection_allowed": False,
        "geometry_authority": "ORDERED_256_BIN_XYZ_ARGMAX",
        "coord_bins_per_axis": 256,
        "diffusion_runtime_authority": False,
        "legacy_diffusion_state_packaged": False,
        "teacher_mapping_runtime_authority": False,
        "source_rig_runtime_authority": False,
        "source_skin_runtime_authority": False,
    }
    for key, expected in required_rules.items():
        if rules.get(key) != expected:
            raise RuntimeError(f"AXIS_MODELRELEASE_RUNTIME_RULE_DRIFT:{key}")

    historical_sha, historical_size = _sha256_file(historical_path)
    if historical_sha != V541_SHA:
        raise RuntimeError("AXIS_MODELRELEASE_HISTORICAL_COMPOSITE_SHA_DRIFT")
    composite = _load_torch_mapping(historical_path)
    if composite.get("schema") != "RealSaS.AXIS41DiscreteXYZComposite.v5_4_1":
        raise RuntimeError("AXIS_MODELRELEASE_HISTORICAL_SCHEMA_DRIFT")
    if composite.get("architecture") != AXIS_V541_ARCHITECTURE_ID:
        raise RuntimeError("AXIS_MODELRELEASE_HISTORICAL_ARCHITECTURE_DRIFT")
    if not _terminal_pass(composite):
        raise RuntimeError("AXIS_MODELRELEASE_HISTORICAL_TERMINAL_NOT_PASS")
    if composite.get("hard_causal_parent_authority") is not True:
        raise RuntimeError("AXIS_MODELRELEASE_HISTORICAL_PARENT_AUTHORITY_DRIFT")
    if composite.get("compiler_tree_reselection") is not False:
        raise RuntimeError("AXIS_MODELRELEASE_HISTORICAL_COMPILER_RESELECTION_DRIFT")
    if composite.get("diffusion_runtime_authority") is not False or int(composite.get("diffusion_invocation_count") or 0) != 0:
        raise RuntimeError("AXIS_MODELRELEASE_HISTORICAL_DIFFUSION_AUTHORITY_DRIFT")

    transport = json.loads(transport_receipt_path.read_text(encoding="utf-8"))
    if transport.get("schema") != "RealSaS.AXISV541SourceTransportReceipt.v1" or transport.get("status") != "EXACT_SOURCE_BYTES_STAGED":
        raise RuntimeError("AXIS_MODELRELEASE_TRANSPORT_RECEIPT_DRIFT")
    if transport.get("training_executed") is not False or transport.get("scientific_pass_minted") is not False:
        raise RuntimeError("AXIS_MODELRELEASE_TRANSPORT_AUTHORITY_DRIFT")
    transport_hashes = {row.get("name"): row.get("sha256") for row in transport.get("source_files", [])}
    if transport_hashes.get("GEPPETTO_KNIGHT_CHECKPOINT.pt") != BASE_SHA:
        raise RuntimeError("AXIS_MODELRELEASE_TRANSPORT_BASE_DRIFT")
    if transport_hashes.get("AXIS41_TREE_GRAMMAR_PROGRESS_V53.pt") != V53_SHA:
        raise RuntimeError("AXIS_MODELRELEASE_TRANSPORT_V53_DRIFT")
    if transport_hashes.get("AXIS41_DISCRETE_XYZ_FROZEN_CAUSAL_COMPOSITE_V541.pt") != V541_SHA:
        raise RuntimeError("AXIS_MODELRELEASE_TRANSPORT_V541_DRIFT")

    return {
        "runtime_sha": runtime_sha,
        "runtime_size": runtime_size,
        "historical_size": historical_size,
        "rules": dict(rules),
    }


def _register_or_verify(api_root: str, payload: dict[str, Any]) -> dict[str, Any]:
    for record in _request_json(api_root, "GET", "/v1/model-releases"):
        if record.get("family") != payload["family"] or record.get("name") != payload["name"]:
            continue
        for key in ("architecture_id", "checkpoint_artifact_id", "checkpoint_content_sha256", "engine_release_id", "source_code_sha"):
            if str(record.get(key)) != str(payload[key]):
                raise RuntimeError(f"AXIS_MODELRELEASE_EXISTING_IDENTITY_DRIFT:{key}")
        if record.get("lifecycle_state") not in {"DRAFT", "RESEARCH_QUALIFIED"}:
            raise RuntimeError("AXIS_MODELRELEASE_EXISTING_LIFECYCLE_DRIFT")
        return record
    return _request_json(api_root, "POST", "/v1/model-releases", payload)


def _set_evaluation(api_root: str, *, model_release_id: str, scope: str, result: str,
                    actor: str, evidence_artifact_id: str | None = None,
                    details: dict[str, Any] | None = None) -> None:
    payload: dict[str, Any] = {
        "model_release_id": model_release_id,
        "scope": scope,
        "result": result,
        "actor": actor,
        "details": details or {},
    }
    if evidence_artifact_id:
        payload["evidence_artifact_id"] = evidence_artifact_id
    _request_json(api_root, "POST", "/v1/model-releases/evaluations", payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-root", required=True)
    parser.add_argument("--runtime-checkpoint", required=True)
    parser.add_argument("--package-receipt", required=True)
    parser.add_argument("--historical-composite", required=True)
    parser.add_argument("--source-transport-receipt", required=True)
    parser.add_argument("--engine-release-id", required=True)
    parser.add_argument("--source-code-sha", required=True)
    parser.add_argument("--actor", required=True)
    parser.add_argument("--name")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    runtime_path = Path(args.runtime_checkpoint).resolve()
    package_receipt_path = Path(args.package_receipt).resolve()
    historical_path = Path(args.historical_composite).resolve()
    transport_receipt_path = Path(args.source_transport_receipt).resolve()
    validated = _validate_inputs(runtime_path, package_receipt_path, historical_path, transport_receipt_path)
    runtime_sha = validated["runtime_sha"]
    runtime_size = validated["runtime_size"]
    if len(args.source_code_sha) != 40 or args.source_code_sha.lower() != args.source_code_sha:
        raise RuntimeError("AXIS_MODELRELEASE_SOURCE_SHA_INVALID")
    _require_canonical_source(args.api_root, args.engine_release_id, args.source_code_sha)
    name = args.name or f"axis-v5.4.1-{runtime_sha[:12]}"

    plan: dict[str, Any] = {
        "schema": "RealSaS.AXISV541ModelReleaseRegistrationPlan.v1",
        "family": FAMILY,
        "name": name,
        "architecture_id": AXIS_V541_ARCHITECTURE_ID,
        "runtime_content_sha256": runtime_sha,
        "runtime_size_bytes": runtime_size,
        "engine_release_id": args.engine_release_id,
        "source_code_sha": args.source_code_sha,
        "initial_state": "DRAFT",
        "apply": bool(args.apply),
        "promotion_performed": False,
        "new_scientific_pass_minted": False,
        "genericity_claimed": False,
    }
    if not args.apply:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0

    def import_path(path: Path, artifact_type: str, schema_version: str, source_uri: str) -> dict[str, Any]:
        sha, size = _sha256_file(path)
        obj = _put_file(args.api_root, path, sha, size)
        return _import_artifact(args.api_root, artifact_type=artifact_type, schema_version=schema_version,
                                obj=obj, source_uri=source_uri, actor=args.actor)

    runtime_import = import_path(runtime_path, RUNTIME_ARTIFACT_TYPE, RUNTIME_SCHEMA_VERSION,
                                 f"urn:realsas:axis:v5.4.1:runtime:{runtime_sha}")
    package_import = import_path(package_receipt_path, PACKAGE_EVIDENCE_ARTIFACT_TYPE,
                                 PACKAGE_EVIDENCE_SCHEMA_VERSION,
                                 f"urn:realsas:axis:v5.4.1:runtime-package:{runtime_sha}")
    historical_import = import_path(historical_path, HISTORICAL_EVIDENCE_ARTIFACT_TYPE,
                                    HISTORICAL_EVIDENCE_SCHEMA_VERSION,
                                    f"urn:realsas:axis:v5.4.1:historical-terminal:{V541_SHA}")
    transport_sha, _ = _sha256_file(transport_receipt_path)
    transport_import = import_path(transport_receipt_path, TRANSPORT_EVIDENCE_ARTIFACT_TYPE,
                                   TRANSPORT_EVIDENCE_SCHEMA_VERSION,
                                   f"urn:realsas:axis:v5.4.1:source-transport:{transport_sha}")

    registration = {
        "family": FAMILY,
        "name": name,
        "architecture_id": AXIS_V541_ARCHITECTURE_ID,
        "checkpoint_artifact_id": runtime_import["artifact_id"],
        "checkpoint_content_sha256": runtime_sha,
        "engine_release_id": args.engine_release_id,
        "source_code_sha": args.source_code_sha,
        "initial_state": "DRAFT",
        "metadata": {
            "runtime_checkpoint_schema": AXIS_RUNTIME_CHECKPOINT_SCHEMA_V1,
            "runtime_checkpoint_size_bytes": runtime_size,
            "base_checkpoint_sha256": BASE_SHA,
            "v53_progress_sha256": V53_SHA,
            "v541_composite_sha256": V541_SHA,
            "package_evidence_artifact_id": package_import["artifact_id"],
            "historical_terminal_evidence_artifact_id": historical_import["artifact_id"],
            "source_transport_evidence_artifact_id": transport_import["artifact_id"],
            "new_scientific_pass_minted": False,
            "fit8_qualified": False,
            "lofo_qualified": False,
            "unseen_generalization_qualified": False,
            "product_rig_authority_claimed": False,
            "compiler_remains_canonical_rig_authority": True,
        },
        "created_by": args.actor,
    }
    model = _register_or_verify(args.api_root, registration)
    model_id = model["id"]
    package_evidence_id = package_import["artifact_id"]
    historical_evidence_id = historical_import["artifact_id"]

    _set_evaluation(args.api_root, model_release_id=model_id, scope="RUNTIME_PACKAGE_IDENTITY_V1",
                    result="PASS", actor=args.actor, evidence_artifact_id=package_evidence_id,
                    details={"content_sha256": runtime_sha, "size_bytes": runtime_size,
                             "source_components_exact": True})
    _set_evaluation(args.api_root, model_release_id=model_id, scope="RUNTIME_SELF_CONTAINED_LOAD_V1",
                    result="PASS", actor=args.actor, evidence_artifact_id=package_evidence_id,
                    details={"strict_state_load": True, "self_contained_inference_state": True})
    _set_evaluation(args.api_root, model_release_id=model_id, scope="HISTORICAL_FIT1_TERMINAL_V541",
                    result="PASS", actor=args.actor, evidence_artifact_id=historical_evidence_id,
                    details={"historical_composite_sha256": V541_SHA, "terminal_eval_pass": True,
                             "scope_limit": "HISTORICAL_FIT1_ONLY"})
    _set_evaluation(args.api_root, model_release_id=model_id, scope="AXIS_RUNTIME_AUTHORITY_CONTRACT_V1",
                    result="PASS", actor=args.actor, evidence_artifact_id=package_evidence_id,
                    details={"hard_causal_parent_authority": True,
                             "compiler_parent_reselection_allowed": False,
                             "geometry_authority": "ORDERED_256_BIN_XYZ_ARGMAX",
                             "diffusion_runtime_authority": False,
                             "teacher_mapping_runtime_authority": False,
                             "compiler_remains_canonical_rig_authority": True})
    for scope in ("FIT8", "LOFO", "UNSEEN_GENERALIZATION"):
        _set_evaluation(args.api_root, model_release_id=model_id, scope=scope,
                        result="NOT_RUN", actor=args.actor)

    final = _request_json(args.api_root, "GET", f"/v1/model-releases/{model_id}")
    if final.get("lifecycle_state") not in {"DRAFT", "RESEARCH_QUALIFIED"}:
        raise RuntimeError("AXIS_MODELRELEASE_LIFECYCLE_DRIFT")
    plan.update({
        "runtime_artifact_id": runtime_import["artifact_id"],
        "package_evidence_artifact_id": package_evidence_id,
        "historical_evidence_artifact_id": historical_evidence_id,
        "source_transport_evidence_artifact_id": transport_import["artifact_id"],
        "model_release_id": model_id,
        "lifecycle_state": final["lifecycle_state"],
        "evaluations": final.get("evaluations", []),
    })
    print(json.dumps(plan, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
