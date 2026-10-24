#!/usr/bin/env python3
"""Register the recovered MIRA V5.5 runtime as scoped Platform model identity.

Dry-run is the default. Mutation requires --apply. RESEARCH channel movement is
separately explicit through --promote-research. This operator never activates
product.skin, creates a ProductRevision, runs model inference, trains/fits, or
promotes MIRA beyond RESEARCH.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

REGISTRATION_SCHEMA = "RealSaS.MIRAV55ModelReleaseRegistration.v1"
RECOVERY_SCHEMA = "RealSaS.MIRAV55RuntimeRecovery.v1"
RECEIPT_SCHEMA = "RealSaS.MIRAV55ModelReleaseRegistrationReceipt.v1"
EXPECTED_FAMILY = "MIRA"
EXPECTED_NAME = "mira-v55-direct-carrier-axis541-recovered-v1"
EXPECTED_ARCHITECTURE = "RealSaS.MIRA.DirectCarrierSimplex.Axis541.v5_5"
EXPECTED_CHECKPOINT_TYPE = "RealSaS.MIRARuntimeCheckpoint"
EXPECTED_CHECKPOINT_SCHEMA = "v1"
EXPECTED_CHECKPOINT_SHA256 = "b5ad2d56a9006058cba30c0b123d4e111f3eea3b78ed42f3542cbd095983139a"
EXPECTED_CHECKPOINT_SIZE = 558_840_979
EXPECTED_COMPUTE_CONTRACT = "CUDA_BF16_BACKBONE_FP32_READOUT_TF32_OFF"
EXPECTED_RECOVERY_TYPE = "RealSaS.MIRAV55RuntimeRecovery"
EXPECTED_RECOVERY_SCHEMA_VERSION = "v1"
EXPECTED_PASS_SCOPES = (
    "HISTORICAL_READOUT_BITWISE_PARITY",
    "RUNTIME_PACKAGE_SELF_CONTAINED_LOAD",
    "HISTORICAL_KNIGHT_TERMINAL_PASS",
)
HEX = set("0123456789abcdef")


class RegistrationError(RuntimeError):
    pass


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RegistrationError(f"MIRA_REGISTRATION_JSON_READ_FAILED:{path}:{exc}") from exc
    if not isinstance(value, dict):
        raise RegistrationError(f"MIRA_REGISTRATION_JSON_OBJECT_REQUIRED:{path}")
    return value


def _require_hex(value: Any, length: int, code: str) -> str:
    if not isinstance(value, str) or len(value) != length or any(ch not in HEX for ch in value):
        raise RegistrationError(f"MIRA_REGISTRATION_{code}_INVALID")
    return value


def _require_uuid(value: Any, code: str) -> str:
    try:
        parsed = uuid.UUID(str(value))
    except (ValueError, TypeError) as exc:
        raise RegistrationError(f"MIRA_REGISTRATION_{code}_UUID_INVALID") from exc
    if parsed.int == 0:
        raise RegistrationError(f"MIRA_REGISTRATION_{code}_UUID_NIL")
    return str(parsed)


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 << 20), b""):
                digest.update(block)
                size += len(block)
    except OSError as exc:
        raise RegistrationError(f"MIRA_REGISTRATION_FILE_READ_FAILED:{path}:{exc}") from exc
    return digest.hexdigest(), size


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode:
        raise RegistrationError(f"MIRA_REGISTRATION_GIT_FAILED:{' '.join(args)}:{proc.stderr.strip()}")
    return proc.stdout.strip()


def _validate_descriptor(value: dict[str, Any]) -> dict[str, Any]:
    if value.get("schema") != REGISTRATION_SCHEMA:
        raise RegistrationError("MIRA_REGISTRATION_DESCRIPTOR_SCHEMA_DRIFT")
    if value.get("authority") != "OPERATIONAL_MODEL_IDENTITY_AND_RESEARCH_CHANNEL_POLICY_ONLY":
        raise RegistrationError("MIRA_REGISTRATION_DESCRIPTOR_AUTHORITY_DRIFT")
    model = value.get("model")
    checkpoint = value.get("checkpoint")
    evidence = value.get("recovery_evidence")
    compute = value.get("compute_contract")
    historical = value.get("historical_scope")
    promotion = value.get("promotion")
    activation = value.get("activation")
    forbidden = value.get("forbidden_claims")
    evaluations = value.get("evaluations")
    if not all(isinstance(x, dict) for x in (model, checkpoint, evidence, compute, historical, promotion, activation, forbidden)):
        raise RegistrationError("MIRA_REGISTRATION_DESCRIPTOR_SECTION_MISSING")
    if not isinstance(evaluations, list):
        raise RegistrationError("MIRA_REGISTRATION_EVALUATIONS_REQUIRED")
    if model != {
        "family": EXPECTED_FAMILY,
        "name": EXPECTED_NAME,
        "architecture_id": EXPECTED_ARCHITECTURE,
        "initial_state": "DRAFT",
    }:
        raise RegistrationError("MIRA_REGISTRATION_MODEL_IDENTITY_DRIFT")
    if checkpoint.get("artifact_type") != EXPECTED_CHECKPOINT_TYPE or checkpoint.get("schema_version") != EXPECTED_CHECKPOINT_SCHEMA:
        raise RegistrationError("MIRA_REGISTRATION_CHECKPOINT_TYPE_DRIFT")
    if checkpoint.get("content_sha256") != EXPECTED_CHECKPOINT_SHA256 or checkpoint.get("size_bytes") != EXPECTED_CHECKPOINT_SIZE:
        raise RegistrationError("MIRA_REGISTRATION_CHECKPOINT_IDENTITY_DRIFT")
    if not isinstance(checkpoint.get("source_uri"), str) or not checkpoint["source_uri"].strip():
        raise RegistrationError("MIRA_REGISTRATION_CHECKPOINT_SOURCE_URI_REQUIRED")
    if evidence.get("artifact_type") != EXPECTED_RECOVERY_TYPE or evidence.get("schema_version") != EXPECTED_RECOVERY_SCHEMA_VERSION:
        raise RegistrationError("MIRA_REGISTRATION_RECOVERY_EVIDENCE_TYPE_DRIFT")
    repo_path = evidence.get("repo_path")
    if not isinstance(repo_path, str) or not repo_path.startswith("canonical/") or ".." in Path(repo_path).parts:
        raise RegistrationError("MIRA_REGISTRATION_RECOVERY_EVIDENCE_PATH_INVALID")
    if compute != {
        "id": EXPECTED_COMPUTE_CONTRACT,
        "cuda_required": True,
        "cuda_bf16_required": True,
        "tf32_allowed": False,
        "cross_device_parity_proven": False,
    }:
        raise RegistrationError("MIRA_REGISTRATION_COMPUTE_CONTRACT_DRIFT")
    if historical.get("subject") != "KNIGHT" or historical.get("terminal_pass_observed") is not True:
        raise RegistrationError("MIRA_REGISTRATION_HISTORICAL_SCOPE_DRIFT")
    if historical.get("product_authority") is not False or historical.get("generalization_claimed") is not False:
        raise RegistrationError("MIRA_REGISTRATION_HISTORICAL_AUTHORITY_OVERCLAIM")
    if historical.get("current_platform_product_skin_smoke_observed") is not False:
        raise RegistrationError("MIRA_REGISTRATION_CURRENT_SMOKE_PREMATURE")
    if promotion.get("maximum_allowed_channel") != "RESEARCH" or promotion.get("target_lifecycle_state") != "RESEARCH_QUALIFIED":
        raise RegistrationError("MIRA_REGISTRATION_PROMOTION_BOUNDARY_DRIFT")
    if tuple(promotion.get("required_pass_scopes") or ()) != EXPECTED_PASS_SCOPES:
        raise RegistrationError("MIRA_REGISTRATION_REQUIRED_SCOPE_DRIFT")
    if promotion.get("candidate_promotion_allowed") is not False or promotion.get("production_promotion_allowed") is not False:
        raise RegistrationError("MIRA_REGISTRATION_PROMOTION_OVERCLAIM")
    if promotion.get("replace_existing_research_channel_requires_explicit_operator_flag") is not True:
        raise RegistrationError("MIRA_REGISTRATION_CHANNEL_REPLACEMENT_GUARD_MISSING")
    if activation != {
        "product_skin_activation_performed_by_registration": False,
        "current_platform_skin_smoke_required_before_activation": True,
        "separate_engine_release_required_for_activation": True,
    }:
        raise RegistrationError("MIRA_REGISTRATION_ACTIVATION_BOUNDARY_DRIFT")
    if any(v is not False for v in forbidden.values()) or set(forbidden) != {
        "new_scientific_pass",
        "unseen_generalization",
        "product_candidate",
        "production_ready",
        "product_revision_pass",
        "source_skin_runtime_authority",
        "gsa_skin_field_authority",
        "semantic_skin_transfer",
    }:
        raise RegistrationError("MIRA_REGISTRATION_FORBIDDEN_CLAIM_DRIFT")

    by_scope: dict[str, dict[str, Any]] = {}
    for row in evaluations:
        if not isinstance(row, dict):
            raise RegistrationError("MIRA_REGISTRATION_EVALUATION_OBJECT_REQUIRED")
        scope = row.get("scope")
        if not isinstance(scope, str) or not scope or scope in by_scope:
            raise RegistrationError("MIRA_REGISTRATION_EVALUATION_SCOPE_INVALID")
        result = row.get("result")
        if result not in {"PASS", "NOT_RUN"}:
            raise RegistrationError(f"MIRA_REGISTRATION_EVALUATION_RESULT_FORBIDDEN:{scope}")
        if result == "PASS" and row.get("evidence") != "RECOVERY_EVIDENCE":
            raise RegistrationError(f"MIRA_REGISTRATION_PASS_EVIDENCE_DRIFT:{scope}")
        if result == "NOT_RUN" and row.get("evidence") is not None:
            raise RegistrationError(f"MIRA_REGISTRATION_NOT_RUN_EVIDENCE_FORBIDDEN:{scope}")
        if not isinstance(row.get("details"), dict):
            raise RegistrationError(f"MIRA_REGISTRATION_EVALUATION_DETAILS_REQUIRED:{scope}")
        by_scope[scope] = row
    if set(EXPECTED_PASS_SCOPES) - set(by_scope):
        raise RegistrationError("MIRA_REGISTRATION_REQUIRED_PASS_SCOPE_MISSING")
    if by_scope.get("CURRENT_PLATFORM_SKIN_EXECUTOR_SMOKE", {}).get("result") != "NOT_RUN":
        raise RegistrationError("MIRA_REGISTRATION_CURRENT_SMOKE_MUST_START_NOT_RUN")
    if by_scope.get("UNSEEN_SUBJECT_GENERALIZATION", {}).get("result") != "NOT_RUN":
        raise RegistrationError("MIRA_REGISTRATION_UNSEEN_MUST_START_NOT_RUN")
    return value


def _validate_recovery(value: dict[str, Any], descriptor: dict[str, Any]) -> None:
    if value.get("schema") != RECOVERY_SCHEMA:
        raise RegistrationError("MIRA_REGISTRATION_RECOVERY_SCHEMA_DRIFT")
    runtime = value.get("runtime_package")
    compute = value.get("historical_compute_contract")
    parity = value.get("promoted_readout_parity")
    historical = value.get("historical_components")
    if not all(isinstance(x, dict) for x in (runtime, compute, parity, historical)):
        raise RegistrationError("MIRA_REGISTRATION_RECOVERY_SECTION_MISSING")
    checkpoint = descriptor["checkpoint"]
    if runtime.get("content_sha256") != checkpoint["content_sha256"] or runtime.get("size_bytes") != checkpoint["size_bytes"]:
        raise RegistrationError("MIRA_REGISTRATION_RECOVERY_RUNTIME_IDENTITY_DRIFT")
    if runtime.get("schema") != "RealSaS.MIRA.RuntimeCheckpoint.v1" or runtime.get("architecture_id") != EXPECTED_ARCHITECTURE:
        raise RegistrationError("MIRA_REGISTRATION_RECOVERY_RUNTIME_SCHEMA_DRIFT")
    if runtime.get("self_contained_load_verified") is not True:
        raise RegistrationError("MIRA_REGISTRATION_RECOVERY_SELF_LOAD_NOT_VERIFIED")
    if runtime.get("fit_executed") is not False or runtime.get("training_executed") is not False or runtime.get("scientific_pass_minted") is not False:
        raise RegistrationError("MIRA_REGISTRATION_RECOVERY_AUTHORITY_OVERCLAIM")
    if runtime.get("product_skin_activated") is not False or runtime.get("modelrelease_registered") is not False:
        raise RegistrationError("MIRA_REGISTRATION_RECOVERY_ALREADY_MUTATED_CLAIM_DRIFT")
    if compute.get("contract_id") != EXPECTED_COMPUTE_CONTRACT or compute.get("cross_device_parity_proven") is not False:
        raise RegistrationError("MIRA_REGISTRATION_RECOVERY_COMPUTE_DRIFT")
    if compute.get("cpu_fp32_product_execution_admitted") is not False or compute.get("product_executor_fail_closed_on_profile_drift") is not True:
        raise RegistrationError("MIRA_REGISTRATION_RECOVERY_COMPUTE_GUARD_DRIFT")
    if parity.get("torch_equal_logits") is not True or parity.get("torch_equal_weights") is not True:
        raise RegistrationError("MIRA_REGISTRATION_RECOVERY_PARITY_NOT_EXACT")
    if float(parity.get("max_abs_logit_delta", 1.0)) != 0.0 or float(parity.get("max_abs_weight_delta", 1.0)) != 0.0:
        raise RegistrationError("MIRA_REGISTRATION_RECOVERY_PARITY_DELTA")
    if historical.get("architecture_id") != EXPECTED_ARCHITECTURE:
        raise RegistrationError("MIRA_REGISTRATION_RECOVERY_ARCHITECTURE_DRIFT")


class API:
    def __init__(self, endpoint: str):
        self.endpoint = endpoint.rstrip("/")

    def request(self, method: str, path: str, payload: Any | None = None) -> Any:
        data = None if payload is None else json.dumps(payload, sort_keys=True).encode("utf-8")
        headers = {"Accept": "application/json"}
        if data is not None:
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(self.endpoint + path, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                raw = response.read(16 * 1024 * 1024 + 1)
        except urllib.error.HTTPError as exc:
            detail = exc.read(65536).decode("utf-8", "replace")
            raise RegistrationError(f"MIRA_REGISTRATION_API_HTTP_{exc.code}:{path}:{detail}") from exc
        except urllib.error.URLError as exc:
            raise RegistrationError(f"MIRA_REGISTRATION_API_UNREACHABLE:{path}:{exc}") from exc
        if len(raw) > 16 * 1024 * 1024:
            raise RegistrationError(f"MIRA_REGISTRATION_API_RESPONSE_TOO_LARGE:{path}")
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RegistrationError(f"MIRA_REGISTRATION_API_JSON_INVALID:{path}") from exc

    def get(self, path: str) -> Any:
        return self.request("GET", path)

    def post(self, path: str, payload: Any) -> Any:
        return self.request("POST", path, payload)


def _verify_git(repo: Path, descriptor_path: Path, recovery_path: Path, *, apply: bool) -> dict[str, Any]:
    head = _require_hex(_git(repo, "rev-parse", "HEAD"), 40, "SOURCE_CODE_SHA")
    dirty = bool(_git(repo, "status", "--porcelain"))
    if apply and dirty:
        raise RegistrationError("MIRA_REGISTRATION_DIRTY_WORKTREE_FORBIDDEN")
    for path in (descriptor_path, recovery_path):
        try:
            rel = path.resolve().relative_to(repo.resolve()).as_posix()
        except ValueError as exc:
            raise RegistrationError("MIRA_REGISTRATION_CANONICAL_FILE_OUTSIDE_REPO") from exc
        committed = subprocess.run(
            ["git", "-C", str(repo), "show", f"HEAD:{rel}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if committed.returncode:
            raise RegistrationError(f"MIRA_REGISTRATION_CANONICAL_FILE_NOT_COMMITTED:{rel}")
        if committed.stdout != path.read_bytes():
            raise RegistrationError(f"MIRA_REGISTRATION_CANONICAL_FILE_WORKTREE_DRIFT:{rel}")
    return {"source_code_sha": head, "worktree_dirty": dirty}


def _verify_engine_release(api: API, engine_release_id: str, source_code_sha: str) -> dict[str, Any]:
    sources = api.get(f"/v1/releases/{engine_release_id}/sources")
    if not isinstance(sources, list):
        raise RegistrationError("MIRA_REGISTRATION_ENGINE_RELEASE_SOURCES_INVALID")
    matching = [
        row
        for row in sources
        if isinstance(row, dict)
        and row.get("code_sha") == source_code_sha
        and row.get("authority") == "CANONICAL_MAIN"
    ]
    if len(matching) != 1:
        raise RegistrationError("MIRA_REGISTRATION_ENGINE_RELEASE_CURRENT_SOURCE_NOT_BOUND")
    capabilities = api.get(f"/v1/releases/{engine_release_id}/capabilities")
    if not isinstance(capabilities, dict) or capabilities.get("engine_release_id") != engine_release_id:
        raise RegistrationError("MIRA_REGISTRATION_ENGINE_RELEASE_CAPABILITY_SNAPSHOT_INVALID")
    descriptors = capabilities.get("capabilities")
    if not isinstance(descriptors, list):
        raise RegistrationError("MIRA_REGISTRATION_ENGINE_RELEASE_CAPABILITIES_INVALID")
    skin_active = any(isinstance(row, dict) and row.get("id") == "product.skin" for row in descriptors)
    return {
        "canonical_source_binding": matching[0],
        "capability_set_sha256": capabilities.get("capability_set_sha256"),
        "product_skin_active_in_supplied_release": skin_active,
    }


def _upload_with_realsasctl(
    executable: str,
    api_endpoint: str,
    path: Path,
    expected_sha256: str,
    expected_size: int,
) -> dict[str, Any]:
    actual_sha, actual_size = _sha256_file(path)
    if actual_sha != expected_sha256:
        raise RegistrationError(f"MIRA_REGISTRATION_LOCAL_FILE_SHA_DRIFT:{path}")
    if actual_size != expected_size:
        raise RegistrationError(f"MIRA_REGISTRATION_LOCAL_FILE_SIZE_DRIFT:{path}:{actual_size}!={expected_size}")
    proc = subprocess.run(
        [executable, "artifact-put", "--api", api_endpoint, "--file", str(path), "--sha256", expected_sha256],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode:
        raise RegistrationError(f"MIRA_REGISTRATION_ARTIFACT_UPLOAD_FAILED:{path}:{proc.stderr.strip()}")
    try:
        out = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RegistrationError(f"MIRA_REGISTRATION_ARTIFACT_UPLOAD_JSON_INVALID:{path}") from exc
    if not isinstance(out, dict):
        raise RegistrationError("MIRA_REGISTRATION_ARTIFACT_UPLOAD_OBJECT_REQUIRED")
    expected_key = f"cas/sha256/{expected_sha256[:2]}/{expected_sha256[2:4]}/{expected_sha256}"
    if out.get("content_sha256") != expected_sha256 or out.get("size_bytes") != expected_size or out.get("storage_key") != expected_key:
        raise RegistrationError(f"MIRA_REGISTRATION_ARTIFACT_UPLOAD_IDENTITY_DRIFT:{path}")
    return out


def _import_artifact(
    api: API,
    *,
    artifact_type: str,
    schema_version: str,
    obj: dict[str, Any],
    source_uri: str,
    actor: str,
) -> dict[str, Any]:
    out = api.post(
        "/v1/artifacts/import",
        {
            "artifact_type": artifact_type,
            "schema_version": schema_version,
            "object": obj,
            "source_uri": source_uri,
            "created_by": actor,
        },
    )
    if not isinstance(out, dict):
        raise RegistrationError("MIRA_REGISTRATION_ARTIFACT_IMPORT_RESPONSE_INVALID")
    _require_uuid(out.get("artifact_id"), "ARTIFACT_ID")
    _require_hex(out.get("semantic_sha256"), 64, "ARTIFACT_SEMANTIC_SHA256")
    return out


def _release_metadata(descriptor: dict[str, Any]) -> dict[str, Any]:
    historical = descriptor["historical_scope"]
    return {
        "registration_schema": REGISTRATION_SCHEMA,
        "qualification_scope": "HISTORICAL_RECOVERY_ONLY",
        "runtime_checkpoint_schema": "RealSaS.MIRA.RuntimeCheckpoint.v1",
        "runtime_checkpoint_size_bytes": EXPECTED_CHECKPOINT_SIZE,
        "inference_compute_contract": EXPECTED_COMPUTE_CONTRACT,
        "cuda_required": True,
        "cuda_bf16_required": True,
        "tf32_allowed": False,
        "cross_device_parity_proven": False,
        "historical_subject": historical["subject"],
        "historical_terminal_pass": historical["terminal_pass_observed"],
        "historical_product_authority": False,
        "historical_generalization_claimed": False,
        "current_platform_skin_smoke_observed": False,
        "maximum_registration_tool_channel": "RESEARCH",
        "product_skin_activation_performed": False,
        "candidate_or_production_promotion_justified": False,
    }


def _ensure_model_release(
    api: API,
    descriptor: dict[str, Any],
    *,
    checkpoint_artifact_id: str,
    engine_release_id: str,
    source_code_sha: str,
    actor: str,
) -> tuple[dict[str, Any], bool]:
    records = api.get("/v1/model-releases")
    if not isinstance(records, list):
        raise RegistrationError("MIRA_REGISTRATION_MODEL_RELEASE_LIST_INVALID")
    model = descriptor["model"]
    matches = [
        row
        for row in records
        if isinstance(row, dict) and row.get("family") == model["family"] and row.get("name") == model["name"]
    ]
    if len(matches) > 1:
        raise RegistrationError("MIRA_REGISTRATION_MODEL_RELEASE_NOT_UNIQUE")
    metadata = _release_metadata(descriptor)
    expected = {
        "architecture_id": model["architecture_id"],
        "checkpoint_artifact_id": checkpoint_artifact_id,
        "checkpoint_content_sha256": descriptor["checkpoint"]["content_sha256"],
        "engine_release_id": engine_release_id,
        "source_code_sha": source_code_sha,
        "metadata": metadata,
    }
    if matches:
        existing = matches[0]
        for key, value in expected.items():
            if existing.get(key) != value:
                raise RegistrationError(f"MIRA_REGISTRATION_EXISTING_MODEL_RELEASE_DRIFT:{key}")
        return existing, True
    created = api.post(
        "/v1/model-releases",
        {
            "family": model["family"],
            "name": model["name"],
            "architecture_id": model["architecture_id"],
            "checkpoint_artifact_id": checkpoint_artifact_id,
            "checkpoint_content_sha256": descriptor["checkpoint"]["content_sha256"],
            "engine_release_id": engine_release_id,
            "source_code_sha": source_code_sha,
            "initial_state": "DRAFT",
            "metadata": metadata,
            "created_by": actor,
        },
    )
    if not isinstance(created, dict) or created.get("lifecycle_state") != "DRAFT":
        raise RegistrationError("MIRA_REGISTRATION_MODEL_RELEASE_CREATE_RESPONSE_INVALID")
    _require_uuid(created.get("id"), "MODEL_RELEASE_ID")
    return created, False


def _evaluation_by_scope(release: dict[str, Any]) -> dict[str, dict[str, Any]]:
    evaluations = release.get("evaluations") or []
    if not isinstance(evaluations, list):
        raise RegistrationError("MIRA_REGISTRATION_MODEL_EVALUATIONS_INVALID")
    out: dict[str, dict[str, Any]] = {}
    for row in evaluations:
        if not isinstance(row, dict) or not isinstance(row.get("scope"), str) or row["scope"] in out:
            raise RegistrationError("MIRA_REGISTRATION_MODEL_EVALUATION_DUPLICATE")
        out[row["scope"]] = row
    return out


def _ensure_evaluations(
    api: API,
    descriptor: dict[str, Any],
    release: dict[str, Any],
    *,
    recovery_artifact_id: str,
    actor: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    release_id = _require_uuid(release.get("id"), "MODEL_RELEASE_ID")
    current = _evaluation_by_scope(release)
    receipts: list[dict[str, Any]] = []
    latest = release
    for spec in descriptor["evaluations"]:
        scope = spec["scope"]
        desired_result = spec["result"]
        existing = current.get(scope)
        if existing is not None:
            if desired_result == "NOT_RUN" and existing.get("result") in {"PASS", "FAIL", "ABSTAIN"}:
                receipts.append({"scope": scope, "action": "PRESERVE_LATER_OBSERVATION", "result": existing.get("result")})
                continue
            desired_evidence = recovery_artifact_id if desired_result == "PASS" else None
            if (
                existing.get("result") != desired_result
                or existing.get("evidence_artifact_id") != desired_evidence
                or existing.get("details") != spec["details"]
            ):
                raise RegistrationError(f"MIRA_REGISTRATION_EXISTING_EVALUATION_DRIFT:{scope}")
            receipts.append({"scope": scope, "action": "REUSE", "result": desired_result})
            continue
        payload: dict[str, Any] = {
            "model_release_id": release_id,
            "scope": scope,
            "result": desired_result,
            "details": spec["details"],
            "actor": actor,
        }
        if desired_result == "PASS":
            payload["evidence_artifact_id"] = recovery_artifact_id
        latest = api.post("/v1/model-releases/evaluations", payload)
        if not isinstance(latest, dict):
            raise RegistrationError(f"MIRA_REGISTRATION_EVALUATION_RESPONSE_INVALID:{scope}")
        current = _evaluation_by_scope(latest)
        observed = current.get(scope)
        if not isinstance(observed, dict) or observed.get("result") != desired_result:
            raise RegistrationError(f"MIRA_REGISTRATION_EVALUATION_NOT_VISIBLE:{scope}")
        receipts.append({"scope": scope, "action": "CREATE", "result": desired_result})
    return latest, receipts


def _research_channel(api: API) -> dict[str, Any] | None:
    channels = api.get("/v1/model-channels")
    if not isinstance(channels, list):
        raise RegistrationError("MIRA_REGISTRATION_MODEL_CHANNEL_LIST_INVALID")
    matches = [
        row
        for row in channels
        if isinstance(row, dict) and row.get("family") == EXPECTED_FAMILY and row.get("channel") == "RESEARCH"
    ]
    if len(matches) > 1:
        raise RegistrationError("MIRA_REGISTRATION_RESEARCH_CHANNEL_NOT_UNIQUE")
    return matches[0] if matches else None


def _promote_research(
    api: API,
    release: dict[str, Any],
    *,
    actor: str,
    replace_existing: bool,
) -> tuple[dict[str, Any], str]:
    release_id = _require_uuid(release.get("id"), "MODEL_RELEASE_ID")
    current = _research_channel(api)
    if current is not None and current.get("model_release_id") == release_id:
        return current, "REUSE"
    if current is not None and not replace_existing:
        raise RegistrationError("MIRA_REGISTRATION_RESEARCH_CHANNEL_OCCUPIED")
    lock_version = 0 if current is None else int(current.get("lock_version", -1))
    if lock_version < 0:
        raise RegistrationError("MIRA_REGISTRATION_RESEARCH_CHANNEL_LOCK_INVALID")
    promoted = api.post(
        "/v1/model-releases/promote",
        {
            "family": EXPECTED_FAMILY,
            "channel": "RESEARCH",
            "model_release_id": release_id,
            "required_evaluation_scopes": list(EXPECTED_PASS_SCOPES),
            "expected_lock_version": lock_version,
            "actor": actor,
        },
    )
    if not isinstance(promoted, dict) or promoted.get("family") != EXPECTED_FAMILY or promoted.get("channel") != "RESEARCH" or promoted.get("model_release_id") != release_id:
        raise RegistrationError("MIRA_REGISTRATION_RESEARCH_PROMOTION_RESPONSE_INVALID")
    return promoted, "CREATE" if current is None else "REPLACE_EXPLICIT"


def _dry_plan(
    descriptor: dict[str, Any],
    *,
    checkpoint_path: Path,
    recovery_path: Path,
    engine_release_id: str,
    source_code_sha: str,
    promote_research: bool,
    replace_research: bool,
    engine_release: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema": "RealSaS.MIRAV55ModelReleaseRegistrationPlan.v1",
        "mode": "DRY_RUN",
        "checkpoint": {
            "path": str(checkpoint_path),
            "content_sha256": EXPECTED_CHECKPOINT_SHA256,
            "size_bytes": EXPECTED_CHECKPOINT_SIZE,
            "action": "UPLOAD_AND_IMPORT_EXACT_ARTIFACT",
        },
        "recovery_evidence": {
            "path": str(recovery_path),
            "action": "UPLOAD_AND_IMPORT_RECOVERY_EVIDENCE",
        },
        "engine_release_id": engine_release_id,
        "source_code_sha": source_code_sha,
        "engine_release_verification": engine_release,
        "model_release": {
            "family": EXPECTED_FAMILY,
            "name": EXPECTED_NAME,
            "architecture_id": EXPECTED_ARCHITECTURE,
            "action": "ENSURE_DRAFT",
        },
        "evaluations": descriptor["evaluations"],
        "promotion": {
            "requested": promote_research,
            "maximum_allowed_channel": "RESEARCH",
            "replace_existing_research_channel": replace_research,
        },
        "product_skin_activation": "FORBIDDEN",
        "candidate_promotion": "FORBIDDEN",
        "production_promotion": "FORBIDDEN",
        "scientific_execution": "FORBIDDEN",
    }


def _apply(
    api: API,
    descriptor: dict[str, Any],
    *,
    api_endpoint: str,
    realsasctl: str,
    checkpoint_path: Path,
    recovery_path: Path,
    engine_release_id: str,
    source_code_sha: str,
    actor: str,
    promote_research: bool,
    replace_research: bool,
    engine_release: dict[str, Any],
) -> dict[str, Any]:
    checkpoint_obj = _upload_with_realsasctl(
        realsasctl,
        api_endpoint,
        checkpoint_path,
        EXPECTED_CHECKPOINT_SHA256,
        EXPECTED_CHECKPOINT_SIZE,
    )
    checkpoint_import = _import_artifact(
        api,
        artifact_type=EXPECTED_CHECKPOINT_TYPE,
        schema_version=EXPECTED_CHECKPOINT_SCHEMA,
        obj=checkpoint_obj,
        source_uri=descriptor["checkpoint"]["source_uri"],
        actor=actor,
    )
    evidence_sha, evidence_size = _sha256_file(recovery_path)
    evidence_obj = _upload_with_realsasctl(
        realsasctl,
        api_endpoint,
        recovery_path,
        evidence_sha,
        evidence_size,
    )
    evidence_import = _import_artifact(
        api,
        artifact_type=EXPECTED_RECOVERY_TYPE,
        schema_version=EXPECTED_RECOVERY_SCHEMA_VERSION,
        obj=evidence_obj,
        source_uri=descriptor["recovery_evidence"]["source_uri"],
        actor=actor,
    )
    release, reused = _ensure_model_release(
        api,
        descriptor,
        checkpoint_artifact_id=_require_uuid(checkpoint_import.get("artifact_id"), "CHECKPOINT_ARTIFACT_ID"),
        engine_release_id=engine_release_id,
        source_code_sha=source_code_sha,
        actor=actor,
    )
    release, evaluation_receipts = _ensure_evaluations(
        api,
        descriptor,
        release,
        recovery_artifact_id=_require_uuid(evidence_import.get("artifact_id"), "RECOVERY_ARTIFACT_ID"),
        actor=actor,
    )
    channel = None
    promotion_action = "NOT_REQUESTED"
    if promote_research:
        channel, promotion_action = _promote_research(
            api,
            release,
            actor=actor,
            replace_existing=replace_research,
        )
        release = api.get(f"/v1/model-releases/{_require_uuid(release.get('id'), 'MODEL_RELEASE_ID')}")
        if not isinstance(release, dict) or release.get("lifecycle_state") != "RESEARCH_QUALIFIED":
            raise RegistrationError("MIRA_REGISTRATION_RESEARCH_LIFECYCLE_NOT_VISIBLE")
    return {
        "schema": RECEIPT_SCHEMA,
        "mode": "APPLY",
        "checkpoint_object": checkpoint_obj,
        "checkpoint_artifact": checkpoint_import,
        "recovery_evidence_object": evidence_obj,
        "recovery_evidence_artifact": evidence_import,
        "engine_release_id": engine_release_id,
        "source_code_sha": source_code_sha,
        "engine_release_verification": engine_release,
        "model_release": release,
        "model_release_reused": reused,
        "evaluations": evaluation_receipts,
        "research_channel": channel,
        "research_promotion_action": promotion_action,
        "maximum_channel_allowed_by_tool": "RESEARCH",
        "product_skin_activation_performed": False,
        "candidate_or_production_promotion_performed": False,
        "scientific_execution_performed": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--descriptor",
        type=Path,
        default=Path("canonical/MIRA_V55_MODELRELEASE_REGISTRATION_V1.json"),
    )
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--engine-release-id", required=True)
    parser.add_argument("--api", default="http://127.0.0.1:8080")
    parser.add_argument("--realsasctl", default="realsasctl")
    parser.add_argument("--actor", default="p6c2-mira-v55-registration")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--promote-research", action="store_true")
    parser.add_argument("--replace-research-channel", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.promote_research and not args.apply:
            raise RegistrationError("MIRA_REGISTRATION_PROMOTION_REQUIRES_APPLY")
        if args.replace_research_channel and not args.promote_research:
            raise RegistrationError("MIRA_REGISTRATION_CHANNEL_REPLACEMENT_REQUIRES_PROMOTION")
        repo = args.repo.resolve()
        descriptor_path = (repo / args.descriptor).resolve() if not args.descriptor.is_absolute() else args.descriptor.resolve()
        descriptor = _validate_descriptor(_load_json(descriptor_path))
        recovery_path = (repo / descriptor["recovery_evidence"]["repo_path"]).resolve()
        recovery = _load_json(recovery_path)
        _validate_recovery(recovery, descriptor)
        checkpoint_path = args.checkpoint.resolve()
        checkpoint_sha, checkpoint_size = _sha256_file(checkpoint_path)
        if checkpoint_sha != EXPECTED_CHECKPOINT_SHA256 or checkpoint_size != EXPECTED_CHECKPOINT_SIZE:
            raise RegistrationError("MIRA_REGISTRATION_CHECKPOINT_LOCAL_IDENTITY_DRIFT")
        engine_release_id = _require_uuid(args.engine_release_id, "ENGINE_RELEASE_ID")
        provenance = _verify_git(repo, descriptor_path, recovery_path, apply=args.apply)
        api = API(args.api)
        engine_release = _verify_engine_release(api, engine_release_id, provenance["source_code_sha"])
        if args.apply:
            result = _apply(
                api,
                descriptor,
                api_endpoint=args.api,
                realsasctl=args.realsasctl,
                checkpoint_path=checkpoint_path,
                recovery_path=recovery_path,
                engine_release_id=engine_release_id,
                source_code_sha=provenance["source_code_sha"],
                actor=args.actor,
                promote_research=args.promote_research,
                replace_research=args.replace_research_channel,
                engine_release=engine_release,
            )
        else:
            result = _dry_plan(
                descriptor,
                checkpoint_path=checkpoint_path,
                recovery_path=recovery_path,
                engine_release_id=engine_release_id,
                source_code_sha=provenance["source_code_sha"],
                promote_research=args.promote_research,
                replace_research=args.replace_research_channel,
                engine_release=engine_release,
            )
        result["git_provenance"] = provenance
        encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(encoded, encoding="utf-8")
        sys.stdout.write(encoded)
        return 0
    except RegistrationError as exc:
        sys.stderr.write(str(exc) + "\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
