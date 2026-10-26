"""Consume one Platform-bound Oracle-Z V2 report without inferring a scientific verdict."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _write_json(path: Path, value: dict) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    path.write_bytes(data)
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(data).hexdigest(),
        "schema": value["schema"],
        "authority_class": "RESEARCH_APPARATUS_EVIDENCE",
    }


def consume_oracle_z_report(ctx: dict) -> dict:
    manifest = ctx["run_manifest"]
    spec = manifest.get("oracle_z_spec")
    if not isinstance(spec, dict) or spec.get("schema") != "RealSaS.OracleZExecutionSpec.v2":
        raise RuntimeError("ORACLE_Z_EXECUTION_SPEC_REQUIRED")

    external = dict(manifest.get("_platform_external_inputs") or {})
    bound = external.get("oracle_z_report")
    if not isinstance(bound, dict):
        raise RuntimeError("ORACLE_Z_REPORT_EXTERNAL_BINDING_REQUIRED")
    path = Path(str(bound.get("path") or ""))
    if not path.is_file():
        raise RuntimeError("ORACLE_Z_REPORT_BYTES_MISSING")
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != str(bound.get("sha256") or ""):
        raise RuntimeError("ORACLE_Z_REPORT_HASH_DRIFT")
    try:
        report = json.loads(data)
    except json.JSONDecodeError as error:
        raise RuntimeError("ORACLE_Z_REPORT_JSON_INVALID") from error

    if report.get("schema") != "RealSaS.OracleZRepresentationCourt.v2":
        raise RuntimeError("ORACLE_Z_REPORT_SCHEMA_DRIFT")
    if report.get("mode") != "PLATFORM_SCIENTIFIC":
        raise RuntimeError("ORACLE_Z_REPORT_NOT_PLATFORM_SCIENTIFIC")
    platform = report.get("platform_context")
    if not isinstance(platform, dict):
        raise RuntimeError("ORACLE_Z_REPORT_PLATFORM_CONTEXT_REQUIRED")
    expected = {
        "REALSAS_EXPERIMENT_ID": str(spec.get("experiment_id") or ""),
        "REALSAS_CANONICAL_CODE_SHA": str(spec.get("canonical_code_sha") or ""),
    }
    for key, value in expected.items():
        if not value or str(platform.get(key) or "") != value:
            raise RuntimeError("ORACLE_Z_REPORT_PLATFORM_CONTEXT_DRIFT:" + key)
    for key in ("REALSAS_ATTEMPT_ID", "REALSAS_ENGINE_RELEASE_ID"):
        if not str(platform.get(key) or ""):
            raise RuntimeError("ORACLE_Z_REPORT_PLATFORM_CONTEXT_REQUIRED:" + key)

    # The external binding itself is Attempt-scoped by Platform. This adapter
    # deliberately does not reinterpret report.status as a scientific PASS.
    root = ctx["run_root"] / "artifacts" / "R01_ORACLE_Z_REPRESENTATION_V2"
    exact = root / "oracle_z_report.json"
    exact.parent.mkdir(parents=True, exist_ok=True)
    exact.write_bytes(data)
    exact_ref = {
        "path": str(exact.resolve()),
        "sha256": digest,
        "schema": "RealSaS.OracleZRepresentationCourt.v2",
        "authority_class": "RESEARCH_APPARATUS_EVIDENCE",
    }
    receipt = {
        "schema": "RealSaS.OracleZPlatformConsumptionReceipt.v2",
        "report_sha256": digest,
        "report_status": str(report.get("status") or ""),
        "experiment_id": expected["REALSAS_EXPERIMENT_ID"],
        "canonical_code_sha": expected["REALSAS_CANONICAL_CODE_SHA"],
        "attempt_id": str(platform["REALSAS_ATTEMPT_ID"]),
        "engine_release_id": str(platform["REALSAS_ENGINE_RELEASE_ID"]),
        "scientific_verdict_inferred": False,
        "product_authority_changed": False,
        "model_promotion_authorized": False,
    }
    receipt_ref = _write_json(root / "consumption_receipt.json", receipt)
    return {
        "status": "PASS",
        "outputs": [exact_ref, receipt_ref],
        "diagnostics": {
            "external_report_consumed": True,
            "report_status": receipt["report_status"],
            "scientific_verdict_inferred": False,
            "product_authority_changed": False,
        },
    }
