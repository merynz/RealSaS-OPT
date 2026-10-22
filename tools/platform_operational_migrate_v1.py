#!/usr/bin/env python3
"""Fail-closed P1 importer for legacy research continuation state.

Dry-run is the default. `--apply` is required for Platform mutation. The importer
creates/validates Experiment state first, links only explicitly recorded real
Attempt/Campaign UUIDs, and creates ModelRelease records only when exact Artifact
and EngineRelease identities are supplied through a separate bindings file.

It never executes training, courts, promotion, product construction or branch
mutation. Git lineage is verified against immutable commit/blob identities before
any apply request is sent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA = "RealSaS.OperationalResearchStateMigration.v1"
BINDINGS_SCHEMA = "RealSaS.LegacyModelPlatformBindings.v1"
HEX40 = set("0123456789abcdef")
HEX64 = HEX40


class MigrationError(RuntimeError):
    pass


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MigrationError(f"MIGRATION_JSON_READ_FAILED:{path}:{exc}") from exc
    if not isinstance(value, dict):
        raise MigrationError(f"MIGRATION_JSON_OBJECT_REQUIRED:{path}")
    return value


def _hex(value: Any, length: int, name: str) -> str:
    if not isinstance(value, str) or len(value) != length or any(ch not in HEX40 for ch in value):
        raise MigrationError(f"MIGRATION_{name}_INVALID")
    return value


def _uuid(value: Any, name: str) -> str:
    import uuid

    if not isinstance(value, str):
        raise MigrationError(f"MIGRATION_{name}_UUID_INVALID")
    try:
        parsed = uuid.UUID(value)
    except ValueError as exc:
        raise MigrationError(f"MIGRATION_{name}_UUID_INVALID") from exc
    if parsed.int == 0:
        raise MigrationError(f"MIGRATION_{name}_UUID_NIL")
    return str(parsed)


def validate_descriptor(value: dict[str, Any]) -> dict[str, Any]:
    if value.get("schema") != SCHEMA:
        raise MigrationError("MIGRATION_SCHEMA_INVALID")
    if not isinstance(value.get("migration_id"), str) or not value["migration_id"].strip():
        raise MigrationError("MIGRATION_ID_REQUIRED")

    source = value.get("source_authority")
    if not isinstance(source, dict):
        raise MigrationError("MIGRATION_SOURCE_AUTHORITY_REQUIRED")
    branch = source.get("legacy_branch")
    path = source.get("execution_binding_path")
    if not isinstance(branch, str) or not branch.startswith("research/"):
        raise MigrationError("MIGRATION_LEGACY_BRANCH_INVALID")
    if not isinstance(path, str) or not path.startswith("canonical/") or ".." in Path(path).parts:
        raise MigrationError("MIGRATION_EXECUTION_BINDING_PATH_INVALID")
    _hex(source.get("legacy_branch_head_sha"), 40, "LEGACY_HEAD_SHA")
    _hex(source.get("execution_binding_git_blob_sha"), 40, "EXECUTION_BINDING_BLOB_SHA")
    _hex(source.get("scientific_execution_source_sha"), 40, "EXECUTION_SOURCE_SHA")
    if source.get("scientific_execution_source_mutation_allowed") is not False:
        raise MigrationError("MIGRATION_EXECUTION_SOURCE_MUST_BE_IMMUTABLE")

    experiment = value.get("experiment")
    if not isinstance(experiment, dict):
        raise MigrationError("MIGRATION_EXPERIMENT_REQUIRED")
    required = ("slug", "name", "research_question", "hypothesis", "created_by")
    for key in required:
        if not isinstance(experiment.get(key), str) or not experiment[key].strip():
            raise MigrationError(f"MIGRATION_EXPERIMENT_{key.upper()}_REQUIRED")
    if experiment.get("initial_state") != "PREREGISTERED":
        raise MigrationError("MIGRATION_LEGACY_ACTIVE_EXPERIMENT_MUST_ENTER_PREREGISTERED")
    metadata = experiment.get("metadata")
    if not isinstance(metadata, dict):
        raise MigrationError("MIGRATION_EXPERIMENT_METADATA_REQUIRED")
    if metadata.get("scientific_training_executed_at_migration") is not False:
        raise MigrationError("MIGRATION_CANNOT_IMPORT_EXECUTED_TRAINING")
    if metadata.get("optimizer_steps_executed_at_migration") != 0:
        raise MigrationError("MIGRATION_OPTIMIZER_STEPS_MUST_BE_ZERO")
    if metadata.get("scientific_pass_or_fail_observed_at_migration") is not False:
        raise MigrationError("MIGRATION_CANNOT_INVENT_SCIENTIFIC_RESULT")

    links = value.get("links")
    if not isinstance(links, dict):
        raise MigrationError("MIGRATION_LINKS_REQUIRED")
    for key in ("attempt_ids", "campaign_ids"):
        ids = links.get(key)
        if not isinstance(ids, list):
            raise MigrationError(f"MIGRATION_{key.upper()}_LIST_REQUIRED")
        if len(ids) != len(set(ids)):
            raise MigrationError(f"MIGRATION_{key.upper()}_DUPLICATE")
        for index, item in enumerate(ids):
            _uuid(item, f"{key.upper()}_{index}")

    models = value.get("legacy_models")
    if not isinstance(models, list):
        raise MigrationError("MIGRATION_LEGACY_MODELS_REQUIRED")
    families: set[str] = set()
    for model in models:
        if not isinstance(model, dict):
            raise MigrationError("MIGRATION_LEGACY_MODEL_OBJECT_REQUIRED")
        family = model.get("family")
        if not isinstance(family, str) or not family.strip() or family in families:
            raise MigrationError("MIGRATION_LEGACY_MODEL_FAMILY_INVALID")
        families.add(family)
        _hex(model.get("source_code_sha"), 40, "MODEL_SOURCE_SHA")
        _hex(model.get("checkpoint_sha256"), 64, "MODEL_CHECKPOINT_SHA256")
        _hex(model.get("binding_sha256"), 64, "MODEL_BINDING_SHA256")
        if model.get("platform_checkpoint_artifact_id") is not None:
            raise MigrationError("MIGRATION_DESCRIPTOR_MUST_NOT_EMBED_MUTABLE_ARTIFACT_ID")
        if model.get("platform_engine_release_id") is not None:
            raise MigrationError("MIGRATION_DESCRIPTOR_MUST_NOT_EMBED_MUTABLE_ENGINE_RELEASE_ID")
        if model.get("migration_state") != "BLOCKED_IDENTITY_RESOLUTION":
            raise MigrationError("MIGRATION_MODEL_MUST_START_BLOCKED_IDENTITY_RESOLUTION")

    invariants = value.get("scientific_invariants")
    if not isinstance(invariants, dict):
        raise MigrationError("MIGRATION_SCIENTIFIC_INVARIANTS_REQUIRED")
    forbidden_true = (
        "change_thresholds",
        "change_architecture",
        "change_optimizer",
        "change_schedule",
        "execute_optimizer",
        "execute_scientific_courts",
        "claim_pass_or_fail",
        "claim_unseen_generalization",
        "claim_product_promotion",
        "teacher_truth_as_inference_conditioning",
        "named_semantic_taxonomy_as_structural_authority",
        "material_id_as_drawable_owner_truth",
    )
    for key in forbidden_true:
        if invariants.get(key) is not False:
            raise MigrationError(f"MIGRATION_SCIENTIFIC_INVARIANT_VIOLATION:{key}")

    retirement = value.get("branch_retirement")
    if not isinstance(retirement, dict):
        raise MigrationError("MIGRATION_BRANCH_RETIREMENT_REQUIRED")
    if retirement.get("wholesale_merge_legacy_branch_to_main") is not False:
        raise MigrationError("MIGRATION_WHOLESALE_LEGACY_MERGE_FORBIDDEN")
    if retirement.get("delete_legacy_branch_history") is not False:
        raise MigrationError("MIGRATION_LEGACY_HISTORY_DELETE_FORBIDDEN")
    return value


def validate_bindings(value: dict[str, Any] | None, descriptor: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if value is None:
        return {}
    if value.get("schema") != BINDINGS_SCHEMA:
        raise MigrationError("MIGRATION_BINDINGS_SCHEMA_INVALID")
    models = value.get("models")
    if not isinstance(models, dict):
        raise MigrationError("MIGRATION_BINDINGS_MODELS_REQUIRED")
    admitted = {item["family"]: item for item in descriptor["legacy_models"]}
    out: dict[str, dict[str, Any]] = {}
    for family, binding in models.items():
        if family not in admitted:
            raise MigrationError(f"MIGRATION_BINDING_UNKNOWN_FAMILY:{family}")
        if not isinstance(binding, dict):
            raise MigrationError(f"MIGRATION_BINDING_OBJECT_REQUIRED:{family}")
        checkpoint_artifact_id = _uuid(binding.get("checkpoint_artifact_id"), f"{family}_CHECKPOINT_ARTIFACT")
        engine_release_id = _uuid(binding.get("engine_release_id"), f"{family}_ENGINE_RELEASE")
        architecture_id = binding.get("architecture_id")
        release_name = binding.get("release_name")
        if not isinstance(architecture_id, str) or not architecture_id.strip():
            raise MigrationError(f"MIGRATION_BINDING_ARCHITECTURE_REQUIRED:{family}")
        if not isinstance(release_name, str) or not release_name.strip():
            raise MigrationError(f"MIGRATION_BINDING_RELEASE_NAME_REQUIRED:{family}")
        expected_sha = binding.get("expected_checkpoint_sha256")
        if expected_sha != admitted[family]["checkpoint_sha256"]:
            raise MigrationError(f"MIGRATION_BINDING_CHECKPOINT_SHA_MISMATCH:{family}")
        out[family] = {
            "checkpoint_artifact_id": checkpoint_artifact_id,
            "engine_release_id": engine_release_id,
            "architecture_id": architecture_id.strip(),
            "release_name": release_name.strip(),
            "expected_checkpoint_sha256": expected_sha,
        }
    return out


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise MigrationError(f"MIGRATION_GIT_FAILED:{' '.join(args)}:{proc.stderr.strip()}")
    return proc.stdout.strip()


def verify_git_provenance(repo: Path, descriptor: dict[str, Any]) -> dict[str, str]:
    source = descriptor["source_authority"]
    head = source["legacy_branch_head_sha"]
    path = source["execution_binding_path"]
    actual_blob = _git(repo, "rev-parse", f"{head}:{path}")
    if actual_blob != source["execution_binding_git_blob_sha"]:
        raise MigrationError("MIGRATION_EXECUTION_BINDING_BLOB_DRIFT")
    actual_type = _git(repo, "cat-file", "-t", source["scientific_execution_source_sha"])
    if actual_type != "commit":
        raise MigrationError("MIGRATION_EXECUTION_SOURCE_NOT_COMMIT")
    branch_head = _git(repo, "rev-parse", source["legacy_branch"])
    if branch_head != head:
        raise MigrationError("MIGRATION_LEGACY_BRANCH_HEAD_DRIFT")
    return {"legacy_branch_head_sha": branch_head, "execution_binding_git_blob_sha": actual_blob}


@dataclass
class API:
    endpoint: str

    def _request(self, method: str, path: str, payload: Any | None = None) -> Any:
        data = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            self.endpoint.rstrip("/") + path,
            data=data,
            method=method,
            headers=headers,
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                raw = response.read(8 * 1024 * 1024 + 1)
        except urllib.error.HTTPError as exc:
            detail = exc.read(64 * 1024).decode("utf-8", "replace")
            raise MigrationError(f"MIGRATION_API_HTTP_{exc.code}:{path}:{detail}") from exc
        except urllib.error.URLError as exc:
            raise MigrationError(f"MIGRATION_API_UNREACHABLE:{path}:{exc}") from exc
        if len(raw) > 8 * 1024 * 1024:
            raise MigrationError(f"MIGRATION_API_RESPONSE_TOO_LARGE:{path}")
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise MigrationError(f"MIGRATION_API_RESPONSE_JSON_INVALID:{path}") from exc

    def get(self, path: str) -> Any:
        return self._request("GET", path)

    def post(self, path: str, payload: Any) -> Any:
        return self._request("POST", path, payload)


def _existing_experiment(api: API, slug: str) -> dict[str, Any] | None:
    records = api.get("/v1/experiments")
    if not isinstance(records, list):
        raise MigrationError("MIGRATION_EXPERIMENT_LIST_INVALID")
    matches = [item for item in records if isinstance(item, dict) and item.get("slug") == slug]
    if len(matches) > 1:
        raise MigrationError("MIGRATION_EXPERIMENT_SLUG_NOT_UNIQUE")
    return matches[0] if matches else None


def _experiment_equivalent(existing: dict[str, Any], spec: dict[str, Any]) -> bool:
    return all(
        existing.get(key) == spec.get(key)
        for key in ("slug", "name", "research_question", "hypothesis")
    ) and existing.get("lifecycle_state") in {"PREREGISTERED", "RUNNING", "COMPLETED", "SUPERSEDED"}


def _ensure_experiment(api: API, descriptor: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    spec = descriptor["experiment"]
    existing = _existing_experiment(api, spec["slug"])
    if existing is not None:
        if not _experiment_equivalent(existing, spec):
            raise MigrationError("MIGRATION_EXISTING_EXPERIMENT_SPEC_DRIFT")
        return existing, True
    created = api.post("/v1/experiments", spec)
    if not isinstance(created, dict) or not isinstance(created.get("experiment"), dict):
        raise MigrationError("MIGRATION_EXPERIMENT_CREATE_RESPONSE_INVALID")
    return created["experiment"], bool(created.get("reused", False))


def _ensure_links(api: API, descriptor: dict[str, Any], experiment_id: str) -> list[dict[str, Any]]:
    receipts: list[dict[str, Any]] = []
    for kind, key in (("ATTEMPT", "attempt_ids"), ("CAMPAIGN", "campaign_ids")):
        for target_id in descriptor["links"][key]:
            receipt = api.post(
                "/v1/experiments/link",
                {
                    "experiment_id": experiment_id,
                    "target_kind": kind,
                    "target_id": target_id,
                    "actor": "p1-operational-migration",
                },
            )
            if not isinstance(receipt, dict):
                raise MigrationError("MIGRATION_LINK_RESPONSE_INVALID")
            receipts.append(receipt)
    return receipts


def _existing_models(api: API) -> list[dict[str, Any]]:
    records = api.get("/v1/model-releases")
    if not isinstance(records, list):
        raise MigrationError("MIGRATION_MODEL_LIST_INVALID")
    return [item for item in records if isinstance(item, dict)]


def _ensure_models(
    api: API,
    descriptor: dict[str, Any],
    experiment_id: str,
    bindings: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    existing = _existing_models(api)
    migrated: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    for legacy in descriptor["legacy_models"]:
        family = legacy["family"]
        binding = bindings.get(family)
        if binding is None:
            blocked.append(
                {
                    "family": family,
                    "legacy_name": legacy["legacy_name"],
                    "reason": "BLOCKED_IDENTITY_RESOLUTION",
                    "checkpoint_sha256": legacy["checkpoint_sha256"],
                }
            )
            continue
        candidates = [
            item
            for item in existing
            if item.get("family") == family and item.get("name") == binding["release_name"]
        ]
        if len(candidates) > 1:
            raise MigrationError(f"MIGRATION_EXISTING_MODEL_NOT_UNIQUE:{family}")
        if candidates:
            item = candidates[0]
            expected = {
                "architecture_id": binding["architecture_id"],
                "checkpoint_artifact_id": binding["checkpoint_artifact_id"],
                "engine_release_id": binding["engine_release_id"],
                "source_code_sha": legacy["source_code_sha"],
            }
            if any(item.get(key) != value for key, value in expected.items()):
                raise MigrationError(f"MIGRATION_EXISTING_MODEL_SPEC_DRIFT:{family}")
            migrated.append(item)
            continue
        item = api.post(
            "/v1/model-releases",
            {
                "family": family,
                "name": binding["release_name"],
                "architecture_id": binding["architecture_id"],
                "checkpoint_artifact_id": binding["checkpoint_artifact_id"],
                "engine_release_id": binding["engine_release_id"],
                "originating_experiment_id": experiment_id,
                "source_code_sha": legacy["source_code_sha"],
                "initial_state": "RESEARCH_QUALIFIED",
                "metadata": {
                    "migration_id": descriptor["migration_id"],
                    "legacy_name": legacy["legacy_name"],
                    "legacy_checkpoint_step": legacy["checkpoint_step"],
                    "legacy_checkpoint_sha256": legacy["checkpoint_sha256"],
                    "legacy_binding_sha256": legacy["binding_sha256"],
                    "identity_resolution": "EXACT_IDS_SUPPLIED_BY_SEPARATE_BINDINGS_FILE",
                    "promotion_performed_by_migration": false,
                },
                "created_by": "p1-operational-migration",
            },
        )
        if not isinstance(item, dict) or item.get("family") != family:
            raise MigrationError(f"MIGRATION_MODEL_CREATE_RESPONSE_INVALID:{family}")
        migrated.append(item)
    return migrated, blocked


def build_plan(descriptor: dict[str, Any], bindings: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema": "RealSaS.OperationalResearchStateMigrationPlan.v1",
        "migration_id": descriptor["migration_id"],
        "mode": "DRY_RUN",
        "experiment": {
            "action": "ENSURE_PREREGISTERED",
            "slug": descriptor["experiment"]["slug"],
        },
        "links": {
            "attempt_ids": list(descriptor["links"]["attempt_ids"]),
            "campaign_ids": list(descriptor["links"]["campaign_ids"]),
        },
        "models": [
            {
                "family": model["family"],
                "action": "ENSURE_MODEL_RELEASE" if model["family"] in bindings else "BLOCKED_IDENTITY_RESOLUTION",
                "checkpoint_sha256": model["checkpoint_sha256"],
            }
            for model in descriptor["legacy_models"]
        ],
        "promotion": "FORBIDDEN",
        "scientific_execution": "FORBIDDEN",
        "legacy_branch_mutation": "FORBIDDEN",
    }


def apply(api: API, descriptor: dict[str, Any], bindings: dict[str, dict[str, Any]]) -> dict[str, Any]:
    experiment, reused = _ensure_experiment(api, descriptor)
    experiment_id = _uuid(experiment.get("id"), "EXPERIMENT_ID")
    links = _ensure_links(api, descriptor, experiment_id)
    models, blocked = _ensure_models(api, descriptor, experiment_id, bindings)
    status = api.get("/v1/ops/status")
    if not isinstance(status, dict) or status.get("schema") != "RealSaS.OperationalStatus.v1":
        raise MigrationError("MIGRATION_OPERATIONAL_STATUS_INVALID")
    visible = any(
        isinstance(item, dict) and item.get("id") == experiment_id
        for item in status.get("active_experiments", [])
    )
    if not visible:
        raise MigrationError("MIGRATION_EXPERIMENT_NOT_VISIBLE_IN_OPERATIONAL_STATUS")
    return {
        "schema": "RealSaS.OperationalResearchStateMigrationReceipt.v1",
        "migration_id": descriptor["migration_id"],
        "experiment": experiment,
        "experiment_reused": reused,
        "link_receipts": links,
        "model_releases": models,
        "blocked_models": blocked,
        "promotion_performed": false,
        "scientific_execution_performed": false,
        "operational_status_verified": true,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--descriptor", required=True, type=Path)
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--api", default="http://127.0.0.1:8080")
    parser.add_argument("--bindings", type=Path)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    try:
        descriptor = validate_descriptor(_load_json(args.descriptor))
        binding_doc = _load_json(args.bindings) if args.bindings else None
        bindings = validate_bindings(binding_doc, descriptor)
        git_receipt = verify_git_provenance(args.repo.resolve(), descriptor)
        if args.apply:
            result = apply(API(args.api), descriptor, bindings)
            result["mode"] = "APPLY"
        else:
            result = build_plan(descriptor, bindings)
        result["git_provenance"] = git_receipt
        encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(encoded, encoding="utf-8")
        sys.stdout.write(encoded)
        return 0
    except MigrationError as exc:
        sys.stderr.write(str(exc) + "\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
