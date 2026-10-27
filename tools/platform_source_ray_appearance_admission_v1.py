#!/usr/bin/env python3
from __future__ import annotations

"""Admit the sealed #243 sparse appearance court through real Platform execution.

This operator is intentionally narrow: external court bytes enter as one
RunManifestSection SubjectInput; a one-stage RESEARCH EngineRelease executes the
source-ray appearance admission adapter through the capability DAG; Platform CAS
and Registry mint the two typed outputs. Promotion is NEVER. No CanonicalMesh
binding, student target, ModelRelease promotion or ProductRevision is created.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.error
import urllib.request
from typing import Any

from compiler.realsas_compiler_services.orchestrator import mainline
from tools.platform_release_snapshot import snapshot

ACTOR = "source-ray-appearance-platform-admission-v1"
STAGE_ID = "91_SOURCE_RAY_APPEARANCE_PLATFORM_ADMISSION"
CAPABILITY_ID = "compiler.stage/" + STAGE_ID
MANIFEST_KEY = "source_ray_appearance_court_bundle"
QUALIFICATION_TYPE = "RealSaS.SourceRayAppearanceQualification"
EVIDENCE_TYPE = "RealSaS.SourceRayAppearanceEvidence"


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _api(api_root: str, method: str, path: str, payload: Any = None, raw: bytes | None = None) -> Any:
    if payload is not None and raw is not None:
        raise RuntimeError("SOURCE_RAY_APPEARANCE_PLATFORM_API_BODY_AMBIGUOUS")
    body = raw if raw is not None else (_json_bytes(payload) if payload is not None else None)
    request = urllib.request.Request(
        api_root.rstrip("/") + path,
        data=body,
        method=method,
        headers={"Content-Type": "application/octet-stream" if raw is not None else "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            data = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        raise RuntimeError(f"SOURCE_RAY_APPEARANCE_PLATFORM_API_{exc.code}:{path}:{detail}") from exc
    return {} if not data else json.loads(data)


def _artifact_bytes(api_root: str, artifact_id: str, metadata: dict[str, Any]) -> bytes:
    request = urllib.request.Request(api_root.rstrip("/") + f"/v1/artifacts/{artifact_id}/bytes", method="GET")
    with urllib.request.urlopen(request, timeout=120) as response:
        data = response.read()
        header = str(response.headers.get("X-Content-SHA256") or "")
    expected = str(metadata.get("content_sha256") or "")
    if header != expected or _sha(data) != expected or len(data) != int(metadata.get("size_bytes", -1)):
        raise RuntimeError("SOURCE_RAY_APPEARANCE_PLATFORM_REGISTRY_BYTES_DRIFT:" + artifact_id)
    return data


def admission_plan() -> dict[str, Any]:
    base = mainline.load_json(mainline.PLAN_PATH)
    plan = dict(base)
    plan["stages"] = [{
        "ordinal": 1,
        "id": STAGE_ID,
        "title": "Sparse source-ray appearance Platform admission",
        "group": "truth_foundry",
        "depends_on": [],
        "adapter": "compiler.realsas_compiler_services.orchestrator.adapters.source_ray_appearance_platform_v1:admit_source_ray_appearance_v1",
        "manifest_keys": [MANIFEST_KEY],
        "policy": {
            "fail_closed": True,
            "cacheable": False,
            "output_hash_required": True,
            "product_pass_authority": False,
        },
    }]
    plan["stage_count"] = 1
    mainline.validate_plan(plan, require_product_pass_authority=False)
    return plan


def _field(value: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in value:
            return value[name]
    raise RuntimeError("SOURCE_RAY_APPEARANCE_PLATFORM_RESPONSE_FIELD_MISSING:" + ":".join(names))


def _wait(api_root: str, attempt_id: str, timeout: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    terminal = {"COMPLETED", "FAILED", "PASS", "FAIL", "ABSTAIN", "CANCELED", "CANCELLED", "ABORTED"}
    while time.monotonic() < deadline:
        state = _api(api_root, "GET", f"/v1/attempts/{attempt_id}")
        status = str(state.get("state") or "")
        if status in terminal:
            return state
        if status != "OPEN":
            raise RuntimeError("SOURCE_RAY_APPEARANCE_PLATFORM_UNKNOWN_ATTEMPT_STATE:" + status)
        time.sleep(2)
    raise RuntimeError("SOURCE_RAY_APPEARANCE_PLATFORM_ATTEMPT_TIMEOUT:" + attempt_id)


def _verify_checkout(root: Path, expected_sha: str) -> dict[str, Any]:
    deployment = json.loads((root / "deployment.json").read_text(encoding="utf-8"))
    if str(deployment.get("code_sha") or "") != expected_sha:
        raise RuntimeError("SOURCE_RAY_APPEARANCE_PLATFORM_DEPLOYED_MAIN_DRIFT")
    checkout = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if checkout != expected_sha:
        raise RuntimeError("SOURCE_RAY_APPEARANCE_PLATFORM_CHECKOUT_DRIFT")
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all"], text=True)
    if dirty:
        raise RuntimeError("SOURCE_RAY_APPEARANCE_PLATFORM_SOURCE_OVERLAY_FORBIDDEN")
    os.environ["REALSAS_AUTHORITY_ROOT"] = str(deployment["authority_root"])
    return deployment


def _import_bundle(api_root: str, raw: bytes, source_uri: str) -> dict[str, Any]:
    digest = _sha(raw)
    obj = _api(api_root, "PUT", f"/v1/artifacts/bytes?sha256={digest}", raw=raw)
    return _api(api_root, "POST", "/v1/artifacts/import", {
        "artifact_type": "RealSaS.RunManifestSection",
        "schema_version": "v1",
        "object": obj,
        "source_uri": source_uri,
        "created_by": ACTOR,
    })


def _attempt_outputs(api_root: str, subject_id: str, attempt_id: str) -> dict[str, dict[str, Any]]:
    rows = _api(api_root, "GET", f"/v1/subjects/{subject_id}/artifacts?limit=500")
    selected: dict[str, dict[str, Any]] = {}
    for artifact_type in (QUALIFICATION_TYPE, EVIDENCE_TYPE):
        matches = [
            row for row in rows
            if str(row.get("source_kind") or "") == "ATTEMPT"
            and str(row.get("source_id") or "") == attempt_id
            and str(row.get("artifact_type") or "") == artifact_type
            and str(row.get("schema_version") or "") == "v1"
        ]
        if len(matches) != 1:
            raise RuntimeError(f"SOURCE_RAY_APPEARANCE_PLATFORM_OUTPUT_CARDINALITY:{artifact_type}:{len(matches)}")
        selected[artifact_type] = matches[0]
    if any(str(row.get("source_kind") or "") == "PRODUCT_REVISION" for row in rows):
        raise RuntimeError("SOURCE_RAY_APPEARANCE_PLATFORM_PRODUCT_REVISION_MINTED")
    return selected


def _verify_registry_artifact(api_root: str, row: dict[str, Any], expected_schema: str) -> dict[str, Any]:
    artifact_id = str(row["artifact_id"])
    metadata = _api(api_root, "GET", f"/v1/artifacts/{artifact_id}")
    if str(metadata.get("producer_contract") or "") != CAPABILITY_ID:
        raise RuntimeError("SOURCE_RAY_APPEARANCE_PLATFORM_PRODUCER_CONTRACT_DRIFT")
    qualifications = metadata.get("qualifications") or []
    if not any(q.get("qualification_type") == "CAPABILITY_EXECUTION_PASS" and q.get("result") == "PASS" for q in qualifications):
        raise RuntimeError("SOURCE_RAY_APPEARANCE_PLATFORM_CAPABILITY_QUALIFICATION_MISSING")
    if not any(str(i.get("role") or "") == "subject:manifest:" + MANIFEST_KEY for i in metadata.get("inputs") or []):
        raise RuntimeError("SOURCE_RAY_APPEARANCE_PLATFORM_INPUT_LINEAGE_MISSING")
    payload = json.loads(_artifact_bytes(api_root, artifact_id, metadata))
    if str(payload.get("schema") or "") != expected_schema:
        raise RuntimeError("SOURCE_RAY_APPEARANCE_PLATFORM_OUTPUT_SCHEMA_DRIFT")
    if payload.get("canonical_mesh_binding_minted") is not False or payload.get("training_authorized") is not False or payload.get("product_authority_changed") is not False:
        raise RuntimeError("SOURCE_RAY_APPEARANCE_PLATFORM_OUTPUT_AUTHORITY_ESCALATION")
    return {"registry": metadata, "payload": payload}


def execute(args: argparse.Namespace) -> dict[str, Any]:
    _verify_checkout(args.root, args.expected_main_sha)
    bundle = json.loads(args.bundle.read_text(encoding="utf-8"))
    raw = _json_bytes(bundle)
    bundle_sha = _sha(raw)

    subject = _api(args.api_root, "POST", "/v1/subjects", {
        "slug": "source-ray-appearance-fit8-v1",
        "display_name": "FIT8 sparse source-ray appearance admission",
        "created_by": ACTOR,
    })
    subject_id = str(_field(subject, "subject_id", "SubjectID"))
    imported = _import_bundle(args.api_root, raw, "urn:realsas:source-ray-appearance:#243:" + bundle_sha)
    bundle_artifact_id = str(imported["artifact_id"])
    subject_input = _api(args.api_root, "POST", "/v1/subject-inputs", {
        "subject_id": subject_id,
        "bindings": [{"role": "manifest:" + MANIFEST_KEY, "artifact_id": bundle_artifact_id}],
        "created_by": ACTOR,
    })
    subject_input_id = str(subject_input["subject_input_id"])

    plan = admission_plan()
    plan_sha = mainline.validate_plan(plan, require_product_pass_authority=False)
    release_request = snapshot(plan, name="source-ray-appearance-platform-admission-v1-" + args.expected_main_sha[:12], purpose="RESEARCH", created_by=ACTOR)
    release = _api(args.api_root, "POST", "/v1/releases", release_request)
    release_id = str(_field(release, "ReleaseID", "release_id"))

    run_id = "SOURCE_RAY_APPEARANCE_ADMISSION_" + args.expected_main_sha[:12].upper() + "_" + bundle_sha[:12].upper()
    manifest = {
        "run_id": run_id,
        "subject_id": subject_id,
        "execution_class": "WITNESS",
        MANIFEST_KEY: bundle,
    }
    manifest_path = mainline.run_manifest_path(run_id)
    raw_manifest = _json_bytes(manifest)
    if manifest_path.exists() and manifest_path.read_bytes() != raw_manifest:
        raise RuntimeError("SOURCE_RAY_APPEARANCE_PLATFORM_RUN_MANIFEST_DRIFT")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_bytes(raw_manifest)

    capability = _api(args.api_root, "POST", "/v1/capabilities/run", {
        "engine_release_id": release_id,
        "subject_id": subject_id,
        "targets": [CAPABILITY_ID],
        "input_artifact_ids": [],
        "parameters": {},
        "compiler_bridge": {
            "subject_input_id": subject_input_id,
            "compiler_run_id": run_id,
            "run_manifest_path": str(manifest_path),
            "run_ledger_path": str(mainline.run_ledger_path(run_id)),
            "pipeline_plan_sha256": plan_sha,
        },
        "idempotency_key": "source-ray-appearance-platform-admission:" + args.expected_main_sha + ":" + bundle_sha,
        "requested_by": ACTOR,
    })
    attempt_id = str(_field(capability, "AttemptID", "attempt_id"))
    goal_id = str(_field(capability, "ExecutionGoalID", "execution_goal_id"))
    terminal = _wait(args.api_root, attempt_id, args.timeout)
    if str(terminal.get("state") or "") != "COMPLETED":
        raise RuntimeError("SOURCE_RAY_APPEARANCE_PLATFORM_ATTEMPT_FAILED:" + json.dumps(terminal, sort_keys=True))

    outputs = _attempt_outputs(args.api_root, subject_id, attempt_id)
    qualification = _verify_registry_artifact(args.api_root, outputs[QUALIFICATION_TYPE], "RealSaS.SourceRayAppearanceQualification.v1")
    evidence = _verify_registry_artifact(args.api_root, outputs[EVIDENCE_TYPE], "RealSaS.SourceRayAppearanceEvidence.v1")
    if qualification["payload"]["measurement_set_sha256"] != evidence["payload"]["measurement_set_sha256"]:
        raise RuntimeError("SOURCE_RAY_APPEARANCE_PLATFORM_OUTPUT_SET_DRIFT")

    return {
        "schema": "RealSaS.SourceRayAppearancePlatformAdmissionReceipt.v1",
        "status": "PASS_PLATFORM_REGISTRY_ADMISSION",
        "source_code_sha": args.expected_main_sha,
        "bundle_sha256": bundle_sha,
        "subject_id": subject_id,
        "subject_input_id": subject_input_id,
        "bundle_artifact_id": bundle_artifact_id,
        "engine_release_id": release_id,
        "attempt_id": attempt_id,
        "execution_goal_id": goal_id,
        "capability_id": CAPABILITY_ID,
        "qualification_artifact_id": str(outputs[QUALIFICATION_TYPE]["artifact_id"]),
        "evidence_artifact_id": str(outputs[EVIDENCE_TYPE]["artifact_id"]),
        "measurement_set_sha256": qualification["payload"]["measurement_set_sha256"],
        "canonical_mesh_binding_minted": False,
        "training_authorized": False,
        "model_promotion_performed": False,
        "product_revision_minted": False,
        "genericity_claimed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--api-root", default="http://127.0.0.1:8080")
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--expected-main-sha", required=True)
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if len(args.expected_main_sha) != 40 or any(ch not in "0123456789abcdef" for ch in args.expected_main_sha):
        raise RuntimeError("SOURCE_RAY_APPEARANCE_PLATFORM_EXPECTED_MAIN_SHA_INVALID")
    receipt = execute(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
