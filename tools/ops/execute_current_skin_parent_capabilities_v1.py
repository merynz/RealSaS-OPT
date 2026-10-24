#!/usr/bin/env python3
from __future__ import annotations

"""Execute the current generic Mesh/Rig capability closure for one subject.

The operation is deliberately limited to a DEVELOPER_RUN / PromotionNever goal.
It reconstructs a fresh implementation-audit run manifest from the newest exact
SubjectInput that satisfies the current product.rig Compiler read set,
materializes SourceObservation.v1 from the already hash-bound source truth,
imports that payload as EXTERNAL_INPUT, creates an exact current RESEARCH
EngineRelease, and targets product.rig through the Platform capability DAG.

It never runs MIRA, mints CanonicalSkinIR, promotes a model, activates product.skin,
or mints a ProductRevision/genericity claim.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from compiler.realsas_compiler_services.orchestrator import mainline
from compiler.realsas_compiler_services.platform_worker.source_observation_materializer_v1 import (
    materialize_source_observation_from_manifest_v1,
)
from tools.platform_capability_versions import capability_specs
from tools.platform_release_snapshot import snapshot

REQUIRED_PARENT_TYPES = {
    "mesh": ("RealSaS.CanonicalMeshIR", "v1"),
    "rig": ("RealSaS.CanonicalRigIR", "v1"),
    "surface": ("RealSaS.RiggingSurfaceIR", "v1"),
    "support": ("RealSaS.TESSAMaterialSupportFieldIR", "v1"),
}
TERMINAL = {"COMPLETED", "FAILED", "PASS", "FAIL", "ABSTAIN", "CANCELED", "CANCELLED", "ABORTED"}


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _api(api_root: str, method: str, path: str, payload: Any = None, raw: bytes | None = None) -> Any:
    if raw is not None and payload is not None:
        raise RuntimeError("SKIN_PARENT_API_BODY_AMBIGUOUS")
    body = raw if raw is not None else (_json_bytes(payload) if payload is not None else None)
    content_type = "application/octet-stream" if raw is not None else "application/json"
    request = Request(
        api_root.rstrip("/") + path,
        data=body,
        method=method,
        headers={"Content-Type": content_type},
    )
    try:
        with urlopen(request, timeout=120) as response:
            data = response.read()
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        raise RuntimeError(f"SKIN_PARENT_PLATFORM_API_{exc.code}:{path}:{detail}") from exc
    if not data:
        return {}
    return json.loads(data)


def _artifact_bytes(api_root: str, artifact_id: str, metadata: dict[str, Any]) -> bytes:
    request = Request(api_root.rstrip("/") + f"/v1/artifacts/{artifact_id}/bytes", method="GET")
    try:
        with urlopen(request, timeout=120) as response:
            data = response.read()
            header_sha = str(response.headers.get("X-Content-SHA256") or "")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        raise RuntimeError(f"SKIN_PARENT_ARTIFACT_HTTP_{exc.code}:{artifact_id}:{detail}") from exc
    expected = str(metadata.get("content_sha256") or "")
    if header_sha != expected or _sha256(data) != expected:
        raise RuntimeError(f"SKIN_PARENT_ARTIFACT_HASH_DRIFT:{artifact_id}")
    if len(data) != int(metadata.get("size_bytes", -1)):
        raise RuntimeError(f"SKIN_PARENT_ARTIFACT_SIZE_DRIFT:{artifact_id}")
    return data


def _field(value: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in value:
            return value[name]
    raise RuntimeError("SKIN_PARENT_RESPONSE_FIELD_MISSING:" + ":".join(names))


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = _json_bytes(value)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        handle.write(raw)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _product_rig_stage_closure(plan: dict[str, Any]) -> tuple[set[str], set[str]]:
    specs = capability_specs()
    stage_map = {str(row["id"]): row for row in plan.get("stages") or ()}
    capability_seen: set[str] = set()
    stage_targets: set[str] = set()

    def visit_capability(capability_id: str) -> None:
        if capability_id in capability_seen:
            return
        capability_seen.add(capability_id)
        if capability_id not in specs:
            raise RuntimeError("SKIN_PARENT_CAPABILITY_SOURCE_UNKNOWN:" + capability_id)
        dependencies = specs[capability_id]["policy"].get("dependencies") or []
        for dependency in dependencies:
            dependency = str(dependency)
            if dependency.startswith("product."):
                visit_capability(dependency)
            elif dependency.startswith("compiler.stage/"):
                stage_targets.add(dependency.removeprefix("compiler.stage/"))
            else:
                raise RuntimeError("SKIN_PARENT_CAPABILITY_DEPENDENCY_UNKNOWN:" + dependency)

    visit_capability("product.rig")
    stage_closure: set[str] = set()

    def visit_stage(stage_id: str) -> None:
        if stage_id in stage_closure:
            return
        stage = stage_map.get(stage_id)
        if stage is None:
            raise RuntimeError("SKIN_PARENT_STAGE_NOT_IN_PLAN:" + stage_id)
        for dependency in stage.get("depends_on") or []:
            visit_stage(str(dependency))
        stage_closure.add(stage_id)

    for stage_id in sorted(stage_targets):
        visit_stage(stage_id)
    manifest_keys = {
        str(key)
        for stage_id in stage_closure
        for key in (stage_map[stage_id].get("manifest_keys") or [])
    }
    return stage_closure, manifest_keys


def _subject_input_candidates(api_root: str, subject_id: str) -> list[dict[str, Any]]:
    query = urlencode({"artifact_type": "RealSaS.RunManifestSection", "schema_version": "v1", "limit": 500})
    rows = _api(api_root, "GET", f"/v1/subjects/{subject_id}/artifacts?{query}")
    if not isinstance(rows, list):
        raise RuntimeError("SKIN_PARENT_SUBJECT_ARTIFACT_LIST_INVALID")
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        if str(row.get("source_kind") or "") != "SUBJECT_INPUT":
            continue
        source_id = str(row.get("source_id") or "")
        if not source_id:
            continue
        item = grouped.setdefault(source_id, {"subject_input_id": source_id, "observed_at": "", "rows": []})
        item["rows"].append(dict(row))
        observed = str(row.get("observed_at") or "")
        if observed > str(item["observed_at"]):
            item["observed_at"] = observed
    return sorted(grouped.values(), key=lambda row: str(row["observed_at"]), reverse=True)


def _load_subject_input(
    api_root: str,
    subject_id: str,
    required_manifest_keys: set[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    for candidate in _subject_input_candidates(api_root, subject_id):
        input_id = str(candidate["subject_input_id"])
        loaded = _api(api_root, "GET", f"/v1/subjects/{subject_id}/inputs/{input_id}")
        roots = list(loaded.get("root_inputs") or [])
        ids = list(loaded.get("artifact_ids") or [])
        if len(roots) != len(ids) or not roots:
            diagnostics.append({"subject_input_id": input_id, "status": "INVALID_ROOT_BINDINGS"})
            continue
        roles = {str(row.get("role") or "") for row in roots}
        needed = {"subject:manifest:" + key for key in required_manifest_keys}
        missing = sorted(needed - roles)
        if missing:
            diagnostics.append({
                "subject_input_id": input_id,
                "status": "MISSING_CURRENT_READ_SET",
                "missing_roles": missing,
            })
            continue
        sections: dict[str, Any] = {}
        binding_receipt = []
        for root, artifact_id in zip(roots, ids):
            role = str(root.get("role") or "")
            if not role.startswith("subject:manifest:"):
                continue
            key = role[len("subject:manifest:"):]
            metadata = _api(api_root, "GET", f"/v1/artifacts/{artifact_id}")
            if metadata.get("artifact_type") != "RealSaS.RunManifestSection" or metadata.get("schema_version") != "v1":
                raise RuntimeError("SKIN_PARENT_MANIFEST_SECTION_TYPE_DRIFT:" + role)
            raw = _artifact_bytes(api_root, str(artifact_id), metadata)
            try:
                section = json.loads(raw.decode("utf-8"))
            except Exception as exc:
                raise RuntimeError("SKIN_PARENT_MANIFEST_SECTION_JSON_INVALID:" + role) from exc
            sections[key] = section
            binding_receipt.append({
                "role": role,
                "artifact_id": str(artifact_id),
                "semantic_sha256": str(metadata["semantic_sha256"]),
                "content_sha256": str(metadata["content_sha256"]),
            })
        missing_sections = sorted(required_manifest_keys - set(sections))
        if missing_sections:
            raise RuntimeError("SKIN_PARENT_REQUIRED_SECTION_DECODE_MISSING:" + ",".join(missing_sections))
        return loaded, {
            "sections": sections,
            "bindings": binding_receipt,
            "selection_diagnostics": diagnostics,
            "observed_at": candidate["observed_at"],
        }
    raise RuntimeError("SKIN_PARENT_NO_CURRENT_SUBJECT_INPUT_SATISFIES_PRODUCT_RIG_READ_SET")


def _put_import_source_observation(
    api_root: str,
    payload: bytes,
    *,
    subject_input_id: str,
    manifest_sha256: str,
    actor: str,
) -> dict[str, Any]:
    digest = _sha256(payload)
    obj = _api(api_root, "PUT", f"/v1/artifacts/bytes?sha256={digest}", raw=payload)
    return _api(api_root, "POST", "/v1/artifacts/import", {
        "artifact_type": "RealSaS.SourceObservation",
        "schema_version": "v1",
        "object": obj,
        "source_uri": (
            "urn:realsas:source-observation:subject-input:"
            + subject_input_id
            + ":manifest:"
            + manifest_sha256
        ),
        "created_by": actor,
    })


def _wait_attempt(api_root: str, attempt_id: str, timeout_seconds: int) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        state = _api(api_root, "GET", f"/v1/attempts/{attempt_id}")
        status = str(state.get("state") or "")
        if status in TERMINAL:
            return state
        if status != "OPEN":
            raise RuntimeError("SKIN_PARENT_UNKNOWN_ATTEMPT_STATE:" + status)
        time.sleep(2)
    raise RuntimeError("SKIN_PARENT_ATTEMPT_TIMEOUT:" + attempt_id)


def _attempt_parent_artifacts(api_root: str, subject_id: str, attempt_id: str) -> dict[str, Any]:
    rows = _api(api_root, "GET", f"/v1/subjects/{subject_id}/artifacts?limit=500")
    if not isinstance(rows, list):
        raise RuntimeError("SKIN_PARENT_ATTEMPT_ARTIFACT_LIST_INVALID")
    selected: dict[str, Any] = {}
    for role, (artifact_type, schema) in REQUIRED_PARENT_TYPES.items():
        matches = [
            row for row in rows
            if str(row.get("source_kind") or "") == "ATTEMPT"
            and str(row.get("source_id") or "") == attempt_id
            and str(row.get("artifact_type") or "") == artifact_type
            and str(row.get("schema_version") or "") == schema
        ]
        unique = {str(row.get("artifact_id") or ""): row for row in matches}
        if len(unique) != 1:
            raise RuntimeError(f"SKIN_PARENT_ATTEMPT_TYPED_ARTIFACT_CARDINALITY:{role}:{len(unique)}")
        row = next(iter(unique.values()))
        selected[role] = {
            key: row[key]
            for key in (
                "artifact_id",
                "artifact_type",
                "schema_version",
                "semantic_sha256",
                "content_sha256",
                "size_bytes",
                "producer_contract",
                "implementation_sha256",
                "policy_sha256",
                "role",
                "origin",
            )
            if key in row
        }
    return selected


def execute(args: argparse.Namespace) -> dict[str, Any]:
    plan = mainline.load_json(mainline.PLAN_PATH)
    plan_sha = mainline.validate_plan(plan)
    _stage_closure, required_manifest_keys = _product_rig_stage_closure(plan)

    subject = _api(args.api_root, "GET", f"/v1/subjects/by-slug/{args.subject_slug}")
    if subject.get("schema") != "RealSaS.SubjectIdentity.v1":
        raise RuntimeError("SKIN_PARENT_SUBJECT_IDENTITY_SCHEMA_DRIFT")
    subject_id = str(subject.get("subject_id") or "")
    if not subject_id:
        raise RuntimeError("SKIN_PARENT_SUBJECT_ID_REQUIRED")

    subject_input, source = _load_subject_input(args.api_root, subject_id, required_manifest_keys)
    subject_input_id = str(subject_input["subject_input_id"])
    manifest_sha = str(subject_input["manifest_sha256"])
    sections = dict(source["sections"])
    evidence_subject = str((sections.get("source_audit") or {}).get("subject_key") or args.subject_slug)
    manifest = {
        "run_id": args.compiler_run_id,
        "subject_id": evidence_subject,
        "execution_class": "IMPLEMENTATION_AUDIT",
        **sections,
    }
    manifest_path = mainline.run_manifest_path(args.compiler_run_id)
    expected_manifest_raw = _json_bytes(manifest)
    if manifest_path.exists():
        if manifest_path.read_bytes() != expected_manifest_raw:
            raise RuntimeError("SKIN_PARENT_RUN_MANIFEST_EXISTING_DRIFT")
    else:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_bytes(expected_manifest_raw)

    materialized = materialize_source_observation_from_manifest_v1(manifest)
    imported = _put_import_source_observation(
        args.api_root,
        materialized.payload,
        subject_input_id=subject_input_id,
        manifest_sha256=manifest_sha,
        actor=args.actor,
    )
    source_observation_id = str(imported["artifact_id"])

    release_request = snapshot(
        plan,
        name=f"{args.subject_slug}-current-skin-parents-{args.expected_main_sha[:12]}",
        purpose="RESEARCH",
        created_by=args.actor,
    )
    if str(release_request["manifest"]["pipeline_plan_sha256"]) != plan_sha:
        raise RuntimeError("SKIN_PARENT_RELEASE_PLAN_SHA_DRIFT")
    release = _api(args.api_root, "POST", "/v1/releases", release_request)
    release_id = str(_field(release, "ReleaseID", "release_id"))

    ledger_path = mainline.run_ledger_path(args.compiler_run_id)
    capability = _api(args.api_root, "POST", "/v1/capabilities/run", {
        "engine_release_id": release_id,
        "subject_id": subject_id,
        "targets": ["product.rig"],
        "input_artifact_ids": [source_observation_id],
        "parameters": {},
        "compiler_bridge": {
            "subject_input_id": subject_input_id,
            "compiler_run_id": args.compiler_run_id,
            "run_manifest_path": str(manifest_path),
            "run_ledger_path": str(ledger_path),
            "pipeline_plan_sha256": plan_sha,
        },
        "idempotency_key": (
            "current-skin-parents:"
            + args.subject_slug
            + ":"
            + args.expected_main_sha
            + ":"
            + manifest_sha
            + ":"
            + materialized.diagnostics["output_content_sha256"]
        ),
        "requested_by": args.actor,
    })
    attempt_id = str(_field(capability, "AttemptID", "attempt_id"))
    goal_id = str(_field(capability, "ExecutionGoalID", "execution_goal_id"))
    terminal = _wait_attempt(args.api_root, attempt_id, args.timeout_seconds)
    if str(terminal.get("state") or "") != "COMPLETED":
        raise RuntimeError("SKIN_PARENT_CAPABILITY_ATTEMPT_FAILED:" + json.dumps(terminal, sort_keys=True))
    parents = _attempt_parent_artifacts(args.api_root, subject_id, attempt_id)

    return {
        "schema": "RealSaS.CurrentSkinParentCapabilityExecution.v1",
        "status": "PASS_CURRENT_TYPED_SKIN_PARENTS_EXECUTED",
        "source_code_sha": args.expected_main_sha,
        "subject": {"slug": args.subject_slug, "platform_subject_id": subject_id, "evidence_subject_id": evidence_subject},
        "subject_input": {
            "subject_input_id": subject_input_id,
            "manifest_sha256": manifest_sha,
            "observed_at": source["observed_at"],
            "required_manifest_keys": sorted(required_manifest_keys),
            "bindings": source["bindings"],
            "selection_diagnostics": source["selection_diagnostics"],
        },
        "compiler": {
            "run_id": args.compiler_run_id,
            "run_manifest_path": str(manifest_path),
            "run_manifest_sha256": _sha256(expected_manifest_raw),
            "run_ledger_path": str(ledger_path),
            "pipeline_plan_sha256": plan_sha,
            "execution_class": "IMPLEMENTATION_AUDIT",
        },
        "source_observation": {
            "artifact_id": source_observation_id,
            "import_reused": bool(imported.get("reused", False)),
            **materialized.diagnostics,
        },
        "engine_release_id": release_id,
        "capability_target": "product.rig",
        "attempt_id": attempt_id,
        "execution_goal_id": goal_id,
        "attempt_state": str(terminal.get("state") or ""),
        "typed_parents": parents,
        "promotion_policy": "NEVER",
        "mira_inference_executed": False,
        "canonical_skin_minted": False,
        "model_promotion_performed": False,
        "product_skin_activated": False,
        "product_revision_minted": False,
        "genericity_claimed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-root", default="http://127.0.0.1:8080")
    parser.add_argument("--subject-slug", required=True)
    parser.add_argument("--expected-main-sha", required=True)
    parser.add_argument("--compiler-run-id", required=True)
    parser.add_argument("--actor", default="current-skin-parent-executor-v1")
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if len(args.expected_main_sha) != 40 or any(ch not in "0123456789abcdef" for ch in args.expected_main_sha):
        raise RuntimeError("SKIN_PARENT_EXPECTED_MAIN_SHA_INVALID")
    if not args.compiler_run_id or any(ch not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-" for ch in args.compiler_run_id):
        raise RuntimeError("SKIN_PARENT_COMPILER_RUN_ID_INVALID")

    try:
        receipt = execute(args)
    except Exception as exc:
        receipt = {
            "schema": "RealSaS.CurrentSkinParentCapabilityExecutionFailure.v1",
            "status": "FAIL_CURRENT_TYPED_SKIN_PARENT_EXECUTION",
            "source_code_sha": args.expected_main_sha,
            "subject_slug": args.subject_slug,
            "compiler_run_id": args.compiler_run_id,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "mira_inference_executed": False,
            "canonical_skin_minted": False,
            "model_promotion_performed": False,
            "product_skin_activated": False,
            "product_revision_minted": False,
            "genericity_claimed": False,
        }
        _atomic_json(args.out, receipt)
        print(json.dumps(receipt, sort_keys=True), flush=True)
        raise
    _atomic_json(args.out, receipt)
    print(json.dumps(receipt, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
