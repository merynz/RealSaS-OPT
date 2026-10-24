#!/usr/bin/env python3
from __future__ import annotations

"""Seal one exact current SubjectInput from an existing authority run manifest.

This operation does not execute models or compiler stages. It selects only the
run-manifest sections required by the current product.rig capability closure,
imports those exact JSON values as immutable RunManifestSection artifacts, and
seals them through the existing Platform SubjectInput API.
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from compiler.realsas_compiler_services.orchestrator import mainline
from tools.ops.execute_current_skin_parent_capabilities_v1 import (
    _api,
    _atomic_json,
    _json_bytes,
    _product_rig_stage_closure,
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_seed_manifest(source_run_id: str) -> tuple[Path, dict[str, Any], bytes]:
    path = mainline.run_manifest_path(source_run_id)
    if not path.is_file():
        raise RuntimeError("CURRENT_SUBJECT_INPUT_SEED_RUN_MANIFEST_MISSING:" + str(path))
    raw = path.read_bytes()
    try:
        manifest = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise RuntimeError("CURRENT_SUBJECT_INPUT_SEED_RUN_MANIFEST_JSON_INVALID") from exc
    if not isinstance(manifest, dict):
        raise RuntimeError("CURRENT_SUBJECT_INPUT_SEED_RUN_MANIFEST_OBJECT_REQUIRED")
    return path, manifest, raw


def _required_sections(manifest: dict[str, Any], required_manifest_keys: set[str]) -> dict[str, Any]:
    missing = sorted(required_manifest_keys - set(manifest))
    if missing:
        raise RuntimeError("CURRENT_SUBJECT_INPUT_SEED_MISSING_READ_SET:" + ",".join(missing))
    return {key: manifest[key] for key in sorted(required_manifest_keys)}


def _put_import_section(
    api_root: str,
    *,
    source_run_id: str,
    key: str,
    value: Any,
    actor: str,
) -> dict[str, Any]:
    raw = _json_bytes(value)
    digest = _sha256(raw)
    obj = _api(api_root, "PUT", f"/v1/artifacts/bytes?sha256={digest}", raw=raw)
    imported = _api(api_root, "POST", "/v1/artifacts/import", {
        "artifact_type": "RealSaS.RunManifestSection",
        "schema_version": "v1",
        "object": obj,
        "source_uri": f"urn:realsas:authority-run:{source_run_id}:manifest:{key}",
        "created_by": actor,
    })
    if not imported.get("artifact_id"):
        raise RuntimeError("CURRENT_SUBJECT_INPUT_SECTION_IMPORT_ID_MISSING:" + key)
    return imported


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    plan = mainline.load_json(mainline.PLAN_PATH)
    plan_sha = mainline.validate_plan(plan)
    _stage_closure, required_manifest_keys = _product_rig_stage_closure(plan)

    subject = _api(args.api_root, "GET", f"/v1/subjects/by-slug/{args.subject_slug}")
    if subject.get("schema") != "RealSaS.SubjectIdentity.v1":
        raise RuntimeError("CURRENT_SUBJECT_INPUT_SUBJECT_IDENTITY_SCHEMA_DRIFT")
    subject_id = str(subject.get("subject_id") or "")
    if not subject_id:
        raise RuntimeError("CURRENT_SUBJECT_INPUT_SUBJECT_ID_REQUIRED")

    seed_path, seed_manifest, seed_raw = _load_seed_manifest(args.source_run_id)
    source_audit = seed_manifest.get("source_audit")
    evidence_subject = ""
    if isinstance(source_audit, dict):
        evidence_subject = str(source_audit.get("subject_key") or "")
    if not evidence_subject:
        evidence_subject = str(seed_manifest.get("subject_id") or "")
    if evidence_subject != args.subject_slug:
        raise RuntimeError(
            "CURRENT_SUBJECT_INPUT_SEED_SUBJECT_MISMATCH:"
            + evidence_subject
            + ":"
            + args.subject_slug
        )

    sections = _required_sections(seed_manifest, required_manifest_keys)
    bindings = []
    section_receipts = []
    for key, value in sections.items():
        imported = _put_import_section(
            args.api_root,
            source_run_id=args.source_run_id,
            key=key,
            value=value,
            actor=args.actor,
        )
        bindings.append({"role": "manifest:" + key, "artifact_id": imported["artifact_id"]})
        section_receipts.append({
            "key": key,
            "artifact_id": str(imported["artifact_id"]),
            "reused": bool(imported.get("reused", False)),
            "content_sha256": _sha256(_json_bytes(value)),
        })

    sealed = _api(args.api_root, "POST", "/v1/subject-inputs", {
        "subject_id": subject_id,
        "bindings": bindings,
        "created_by": args.actor,
    })
    subject_input_id = str(sealed.get("subject_input_id") or "")
    manifest_sha = str(sealed.get("manifest_sha256") or "")
    if not subject_input_id or len(manifest_sha) != 64:
        raise RuntimeError("CURRENT_SUBJECT_INPUT_SEAL_RESPONSE_INVALID")

    loaded = _api(
        args.api_root,
        "GET",
        f"/v1/subjects/{subject_id}/inputs/{subject_input_id}",
    )
    roots = list(loaded.get("root_inputs") or [])
    roles = {str(row.get("role") or "") for row in roots}
    expected_roles = {"subject:manifest:" + key for key in required_manifest_keys}
    if not expected_roles.issubset(roles):
        raise RuntimeError("CURRENT_SUBJECT_INPUT_POST_SEAL_READ_SET_MISSING")

    return {
        "schema": "RealSaS.CurrentSubjectInputMaterialization.v1",
        "status": "PASS_CURRENT_SUBJECT_INPUT_SEALED",
        "source_code_sha": args.expected_main_sha,
        "pipeline_plan_sha256": plan_sha,
        "subject": {
            "slug": args.subject_slug,
            "platform_subject_id": subject_id,
            "evidence_subject_id": evidence_subject,
        },
        "source_run": {
            "run_id": args.source_run_id,
            "run_manifest_path": str(seed_path),
            "run_manifest_sha256": _sha256(seed_raw),
        },
        "required_manifest_keys": sorted(required_manifest_keys),
        "sections": section_receipts,
        "subject_input_id": subject_input_id,
        "manifest_sha256": manifest_sha,
        "subject_input_reused": bool(sealed.get("reused", False)),
        "model_inference_executed": False,
        "compiler_stage_executed": False,
        "qualification_minted": False,
        "product_authority_minted": False,
        "genericity_claimed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-root", default="http://127.0.0.1:8080")
    parser.add_argument("--subject-slug", required=True)
    parser.add_argument("--source-run-id", required=True)
    parser.add_argument("--expected-main-sha", required=True)
    parser.add_argument("--actor", default="current-subject-input-materializer-v1")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if len(args.expected_main_sha) != 40 or any(ch not in "0123456789abcdef" for ch in args.expected_main_sha):
        raise RuntimeError("CURRENT_SUBJECT_INPUT_EXPECTED_MAIN_SHA_INVALID")
    if not args.source_run_id or any(
        ch not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-"
        for ch in args.source_run_id
    ):
        raise RuntimeError("CURRENT_SUBJECT_INPUT_SOURCE_RUN_ID_INVALID")

    try:
        receipt = materialize(args)
    except Exception as exc:
        receipt = {
            "schema": "RealSaS.CurrentSubjectInputMaterializationFailure.v1",
            "status": "FAIL_CURRENT_SUBJECT_INPUT_MATERIALIZATION",
            "source_code_sha": args.expected_main_sha,
            "subject_slug": args.subject_slug,
            "source_run_id": args.source_run_id,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "model_inference_executed": False,
            "compiler_stage_executed": False,
            "qualification_minted": False,
            "product_authority_minted": False,
            "genericity_claimed": False,
        }
        _atomic_json(args.out, receipt)
        print(json.dumps(receipt, sort_keys=True), flush=True)
        raise
    _atomic_json(args.out, receipt)
    print(json.dumps(receipt, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
