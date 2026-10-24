#!/usr/bin/env python3
from __future__ import annotations

"""Seal one exact current SubjectInput from a witness-bound authority run.

This operation does not execute models or Compiler stages. The Platform subject
identity and legacy evidence run are bound by an explicit canonical witness
inventory; the operation then selects only the run-manifest sections required by
the current product.rig capability closure, imports those exact JSON values as
immutable RunManifestSection artifacts, and seals them through the existing
Platform SubjectInput API.
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

ROOT = Path(__file__).resolve().parents[2]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _nested(value: dict[str, Any], dotted: str) -> Any:
    current: Any = value
    for key in dotted.split("."):
        if not isinstance(current, dict) or key not in current:
            raise RuntimeError("CURRENT_SUBJECT_INPUT_WITNESS_POINTER_INVALID:" + dotted)
        current = current[key]
    return current


def _load_witness_binding(
    repo_root: Path,
    inventory_ref: str,
    manifest_source: str,
    subject_slug: str,
) -> tuple[Path, dict[str, Any], bytes, str]:
    root = repo_root.resolve()
    path = (root / inventory_ref).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise RuntimeError("CURRENT_SUBJECT_INPUT_WITNESS_PATH_ESCAPE") from exc
    if not path.is_file():
        raise RuntimeError("CURRENT_SUBJECT_INPUT_WITNESS_MISSING:" + str(path))
    raw = path.read_bytes()
    try:
        witness = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise RuntimeError("CURRENT_SUBJECT_INPUT_WITNESS_JSON_INVALID") from exc
    if not isinstance(witness, dict):
        raise RuntimeError("CURRENT_SUBJECT_INPUT_WITNESS_OBJECT_REQUIRED")
    witness_subject = str(witness.get("subject_key") or "")
    if witness_subject != subject_slug:
        raise RuntimeError(
            "CURRENT_SUBJECT_INPUT_WITNESS_SUBJECT_MISMATCH:"
            + witness_subject
            + ":"
            + subject_slug
        )
    source_run_id = str(_nested(witness, manifest_source) or "")
    if not source_run_id or any(
        ch not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-"
        for ch in source_run_id
    ):
        raise RuntimeError("CURRENT_SUBJECT_INPUT_WITNESS_RUN_ID_INVALID")
    return path, witness, raw, source_run_id


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

    witness_path, witness, witness_raw, source_run_id = _load_witness_binding(
        ROOT,
        args.identity_inventory,
        args.manifest_source,
        args.subject_slug,
    )
    seed_path, seed_manifest, seed_raw = _load_seed_manifest(source_run_id)

    source_audit = seed_manifest.get("source_audit")
    evidence_subject = ""
    if isinstance(source_audit, dict):
        evidence_subject = str(source_audit.get("subject_key") or "")
    if not evidence_subject:
        evidence_subject = str(seed_manifest.get("subject_id") or "")
    if not evidence_subject:
        raise RuntimeError("CURRENT_SUBJECT_INPUT_EVIDENCE_SUBJECT_ID_REQUIRED")

    sections = _required_sections(seed_manifest, required_manifest_keys)
    bindings = []
    section_receipts = []
    for key, value in sections.items():
        imported = _put_import_section(
            args.api_root,
            source_run_id=source_run_id,
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
            "witness_subject_key": str(witness["subject_key"]),
            "evidence_subject_id": evidence_subject,
        },
        "identity_binding": {
            "inventory_ref": args.identity_inventory,
            "inventory_path": str(witness_path),
            "inventory_sha256": _sha256(witness_raw),
            "manifest_source": args.manifest_source,
            "resolved_source_run_id": source_run_id,
        },
        "source_run": {
            "run_id": source_run_id,
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
    parser.add_argument("--identity-inventory", required=True)
    parser.add_argument("--manifest-source", required=True)
    parser.add_argument("--expected-main-sha", required=True)
    parser.add_argument("--actor", default="current-subject-input-materializer-v1")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if len(args.expected_main_sha) != 40 or any(ch not in "0123456789abcdef" for ch in args.expected_main_sha):
        raise RuntimeError("CURRENT_SUBJECT_INPUT_EXPECTED_MAIN_SHA_INVALID")
    if not args.identity_inventory or not args.manifest_source:
        raise RuntimeError("CURRENT_SUBJECT_INPUT_WITNESS_BINDING_REQUIRED")

    try:
        receipt = materialize(args)
    except Exception as exc:
        receipt = {
            "schema": "RealSaS.CurrentSubjectInputMaterializationFailure.v1",
            "status": "FAIL_CURRENT_SUBJECT_INPUT_MATERIALIZATION",
            "source_code_sha": args.expected_main_sha,
            "subject_slug": args.subject_slug,
            "identity_inventory": args.identity_inventory,
            "manifest_source": args.manifest_source,
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
