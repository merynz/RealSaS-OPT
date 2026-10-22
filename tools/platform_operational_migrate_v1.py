#!/usr/bin/env python3
"""Import legacy research continuation into the P0 operational spine.

Dry-run is the default. `--apply` is required for Platform mutation. This tool
never trains, runs courts, promotes a model, creates ProductRevision state, or
mutates/deletes the legacy branch.

A legacy experiment may already have completed training. In that case the
historical state is imported as RUNNING while the importer itself still performs
zero optimizer steps and zero scientific court execution.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

SCHEMA = "RealSaS.OperationalResearchStateMigration.v1"
BINDINGS_SCHEMA = "RealSaS.LegacyModelPlatformBindings.v1"
HEX = set("0123456789abcdef")


class MigrationError(RuntimeError):
    pass


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MigrationError(f"MIGRATION_JSON_READ_FAILED:{path}:{exc}") from exc
    if not isinstance(value, dict):
        raise MigrationError(f"MIGRATION_JSON_OBJECT_REQUIRED:{path}")
    return value


def require_hex(value: Any, length: int, name: str) -> str:
    if not isinstance(value, str) or len(value) != length or any(ch not in HEX for ch in value):
        raise MigrationError(f"MIGRATION_{name}_INVALID")
    return value


def require_uuid(value: Any, name: str) -> str:
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
    experiment = value.get("experiment")
    links = value.get("links")
    models = value.get("legacy_models")
    invariants = value.get("scientific_invariants")
    retirement = value.get("branch_retirement")
    if not all(isinstance(x, dict) for x in (source, experiment, links, invariants, retirement)):
        raise MigrationError("MIGRATION_REQUIRED_OBJECT_MISSING")
    if not isinstance(models, list):
        raise MigrationError("MIGRATION_LEGACY_MODELS_REQUIRED")

    branch = source.get("legacy_branch")
    binding_path = source.get("execution_binding_path")
    if not isinstance(branch, str) or not branch.startswith("research/"):
        raise MigrationError("MIGRATION_LEGACY_BRANCH_INVALID")
    if not isinstance(binding_path, str) or not binding_path.startswith("canonical/") or ".." in Path(binding_path).parts:
        raise MigrationError("MIGRATION_BINDING_PATH_INVALID")
    require_hex(source.get("legacy_branch_head_sha"), 40, "LEGACY_HEAD_SHA")
    require_hex(source.get("execution_binding_git_blob_sha"), 40, "BINDING_BLOB_SHA")
    require_hex(source.get("scientific_execution_source_sha"), 40, "EXECUTION_SOURCE_SHA")
    if source.get("scientific_execution_source_mutation_allowed") is not False:
        raise MigrationError("MIGRATION_EXECUTION_SOURCE_MUST_BE_IMMUTABLE")

    for key in ("slug", "name", "research_question", "hypothesis", "created_by"):
        if not isinstance(experiment.get(key), str) or not experiment[key].strip():
            raise MigrationError(f"MIGRATION_EXPERIMENT_{key.upper()}_REQUIRED")
    target_state = experiment.get("initial_state")
    if target_state not in {"PREREGISTERED", "RUNNING"}:
        raise MigrationError("MIGRATION_ACTIVE_LEGACY_EXPERIMENT_STATE_INVALID")

    metadata = experiment.get("metadata")
    if not isinstance(metadata, dict):
        raise MigrationError("MIGRATION_EXPERIMENT_METADATA_REQUIRED")
    training_executed = metadata.get("scientific_training_executed_at_migration")
    if not isinstance(training_executed, bool):
        raise MigrationError("MIGRATION_TRAINING_HISTORY_BOOLEAN_REQUIRED")
    if target_state == "PREREGISTERED" and training_executed:
        raise MigrationError("MIGRATION_PREREGISTERED_CANNOT_HAVE_EXECUTED_TRAINING")
    if target_state == "RUNNING":
        if not training_executed:
            raise MigrationError("MIGRATION_RUNNING_REQUIRES_EXECUTED_TRAINING_HISTORY")
        training_state = metadata.get("training_execution_state")
        if not isinstance(training_state, str) or not training_state.strip():
            raise MigrationError("MIGRATION_RUNNING_TRAINING_EXECUTION_STATE_REQUIRED")
    if metadata.get("scientific_free_running_result_observed_at_migration") is not False:
        raise MigrationError("MIGRATION_CANNOT_IMPORT_FINAL_FREE_RUNNING_RESULT")
    if metadata.get("scientific_causal_result_observed_at_migration") is not False:
        raise MigrationError("MIGRATION_CANNOT_IMPORT_FINAL_CAUSAL_RESULT")

    for key in ("attempt_ids", "campaign_ids"):
        ids = links.get(key)
        if not isinstance(ids, list) or len(ids) != len(set(ids)):
            raise MigrationError(f"MIGRATION_{key.upper()}_INVALID")
        for index, item in enumerate(ids):
            require_uuid(item, f"{key.upper()}_{index}")

    seen: set[str] = set()
    for model in models:
        if not isinstance(model, dict):
            raise MigrationError("MIGRATION_MODEL_OBJECT_REQUIRED")
        family = model.get("family")
        if not isinstance(family, str) or not family or family in seen:
            raise MigrationError("MIGRATION_MODEL_FAMILY_INVALID")
        seen.add(family)
        require_hex(model.get("source_code_sha"), 40, "MODEL_SOURCE_SHA")
        require_hex(model.get("checkpoint_sha256"), 64, "MODEL_CHECKPOINT_SHA256")
        require_hex(model.get("binding_sha256"), 64, "MODEL_BINDING_SHA256")
        if model.get("platform_checkpoint_artifact_id") is not None or model.get("platform_engine_release_id") is not None:
            raise MigrationError("MIGRATION_DESCRIPTOR_MUST_NOT_EMBED_PLATFORM_IDS")
        if model.get("migration_state") != "BLOCKED_IDENTITY_RESOLUTION":
            raise MigrationError("MIGRATION_MODEL_MUST_START_BLOCKED")

    forbidden = (
        "change_thresholds",
        "change_architecture",
        "change_optimizer",
        "change_schedule",
        "execute_optimizer",
        "repeat_training",
        "mutate_frozen_checkpoints",
        "execute_scientific_courts",
        "claim_pass_or_fail",
        "claim_unseen_generalization",
        "claim_product_promotion",
        "teacher_truth_as_inference_conditioning",
        "named_semantic_taxonomy_as_structural_authority",
        "material_id_as_drawable_owner_truth",
    )
    for key in forbidden:
        if invariants.get(key) is not False:
            raise MigrationError(f"MIGRATION_SCIENTIFIC_INVARIANT_VIOLATION:{key}")
    if retirement.get("wholesale_merge_legacy_branch_to_main") is not False:
        raise MigrationError("MIGRATION_WHOLESALE_BRANCH_MERGE_FORBIDDEN")
    if retirement.get("delete_legacy_branch_history") is not False:
        raise MigrationError("MIGRATION_BRANCH_HISTORY_DELETE_FORBIDDEN")
    return value


def validate_bindings(value: dict[str, Any] | None, descriptor: dict[str, Any]) -> dict[str, dict[str, str]]:
    if value is None:
        return {}
    if value.get("schema") != BINDINGS_SCHEMA or not isinstance(value.get("models"), dict):
        raise MigrationError("MIGRATION_BINDINGS_SCHEMA_INVALID")
    legacy = {item["family"]: item for item in descriptor["legacy_models"]}
    out: dict[str, dict[str, str]] = {}
    for family, raw in value["models"].items():
        if family not in legacy or not isinstance(raw, dict):
            raise MigrationError(f"MIGRATION_BINDING_UNKNOWN_FAMILY:{family}")
        expected = raw.get("expected_checkpoint_sha256")
        if expected != legacy[family]["checkpoint_sha256"]:
            raise MigrationError(f"MIGRATION_BINDING_CHECKPOINT_SHA_MISMATCH:{family}")
        architecture = raw.get("architecture_id")
        name = raw.get("release_name")
        if not isinstance(architecture, str) or not architecture.strip() or not isinstance(name, str) or not name.strip():
            raise MigrationError(f"MIGRATION_BINDING_MODEL_IDENTITY_REQUIRED:{family}")
        out[family] = {
            "checkpoint_artifact_id": require_uuid(raw.get("checkpoint_artifact_id"), family + "_ARTIFACT"),
            "engine_release_id": require_uuid(raw.get("engine_release_id"), family + "_ENGINE_RELEASE"),
            "architecture_id": architecture.strip(),
            "release_name": name.strip(),
            "expected_checkpoint_sha256": expected,
        }
    return out


def git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode:
        raise MigrationError(f"MIGRATION_GIT_FAILED:{' '.join(args)}:{proc.stderr.strip()}")
    return proc.stdout.strip()


def verify_git(repo: Path, descriptor: dict[str, Any]) -> dict[str, str]:
    source = descriptor["source_authority"]
    branch_head = git(repo, "rev-parse", source["legacy_branch"])
    if branch_head != source["legacy_branch_head_sha"]:
        raise MigrationError("MIGRATION_LEGACY_BRANCH_HEAD_DRIFT")
    blob = git(repo, "rev-parse", f"{branch_head}:{source['execution_binding_path']}")
    if blob != source["execution_binding_git_blob_sha"]:
        raise MigrationError("MIGRATION_EXECUTION_BINDING_BLOB_DRIFT")
    if git(repo, "cat-file", "-t", source["scientific_execution_source_sha"]) != "commit":
        raise MigrationError("MIGRATION_EXECUTION_SOURCE_NOT_COMMIT")
    return {"legacy_branch_head_sha": branch_head, "execution_binding_git_blob_sha": blob}


class API:
    def __init__(self, endpoint: str):
        self.endpoint = endpoint.rstrip("/")

    def request(self, method: str, path: str, payload: Any | None = None) -> Any:
        data = None if payload is None else json.dumps(payload, sort_keys=True).encode()
        headers = {"Accept": "application/json"}
        if data is not None:
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(self.endpoint + path, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                raw = response.read(8 * 1024 * 1024 + 1)
        except urllib.error.HTTPError as exc:
            detail = exc.read(65536).decode("utf-8", "replace")
            raise MigrationError(f"MIGRATION_API_HTTP_{exc.code}:{path}:{detail}") from exc
        except urllib.error.URLError as exc:
            raise MigrationError(f"MIGRATION_API_UNREACHABLE:{path}:{exc}") from exc
        if len(raw) > 8 * 1024 * 1024:
            raise MigrationError(f"MIGRATION_API_RESPONSE_TOO_LARGE:{path}")
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise MigrationError(f"MIGRATION_API_JSON_INVALID:{path}") from exc

    def get(self, path: str) -> Any:
        return self.request("GET", path)

    def post(self, path: str, payload: Any) -> Any:
        return self.request("POST", path, payload)


def ensure_experiment(api: API, spec: dict[str, Any]) -> dict[str, Any]:
    records = api.get("/v1/experiments")
    if not isinstance(records, list):
        raise MigrationError("MIGRATION_EXPERIMENT_LIST_INVALID")
    matches = [x for x in records if isinstance(x, dict) and x.get("slug") == spec["slug"]]
    if len(matches) > 1:
        raise MigrationError("MIGRATION_EXPERIMENT_NOT_UNIQUE")

    target_state = spec["initial_state"]
    if matches:
        experiment = matches[0]
        for key in ("slug", "name", "research_question", "hypothesis"):
            if experiment.get(key) != spec.get(key):
                raise MigrationError("MIGRATION_EXISTING_EXPERIMENT_SPEC_DRIFT")
        if experiment.get("metadata") != spec.get("metadata"):
            raise MigrationError("MIGRATION_EXISTING_EXPERIMENT_METADATA_DRIFT")
    else:
        create_spec = dict(spec)
        create_spec["initial_state"] = "PREREGISTERED"
        created = api.post("/v1/experiments", create_spec)
        if not isinstance(created, dict) or not isinstance(created.get("experiment"), dict):
            raise MigrationError("MIGRATION_EXPERIMENT_CREATE_RESPONSE_INVALID")
        experiment = created["experiment"]

    current_state = experiment.get("lifecycle_state")
    if target_state == "RUNNING" and current_state == "PREREGISTERED":
        transitioned = api.post(
            "/v1/experiments/transition",
            {
                "experiment_id": experiment.get("id"),
                "to_state": "RUNNING",
                "actor": "p1-operational-migration",
            },
        )
        if not isinstance(transitioned, dict) or transitioned.get("lifecycle_state") != "RUNNING":
            raise MigrationError("MIGRATION_EXPERIMENT_RUNNING_TRANSITION_FAILED")
        experiment = transitioned
    elif current_state != target_state:
        raise MigrationError(f"MIGRATION_EXISTING_EXPERIMENT_STATE_DRIFT:{current_state}->{target_state}")
    return experiment


def ensure_links(api: API, descriptor: dict[str, Any], experiment_id: str) -> list[dict[str, Any]]:
    receipts: list[dict[str, Any]] = []
    for kind, key in (("ATTEMPT", "attempt_ids"), ("CAMPAIGN", "campaign_ids")):
        for target in descriptor["links"][key]:
            result = api.post(
                "/v1/experiments/link",
                {
                    "experiment_id": experiment_id,
                    "target_kind": kind,
                    "target_id": target,
                    "actor": "p1-operational-migration",
                },
            )
            if not isinstance(result, dict):
                raise MigrationError("MIGRATION_LINK_RESPONSE_INVALID")
            receipts.append(result)
    return receipts


def ensure_models(
    api: API,
    descriptor: dict[str, Any],
    experiment_id: str,
    bindings: dict[str, dict[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records = api.get("/v1/model-releases")
    if not isinstance(records, list):
        raise MigrationError("MIGRATION_MODEL_LIST_INVALID")
    migrated: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    for legacy in descriptor["legacy_models"]:
        family = legacy["family"]
        binding = bindings.get(family)
        if binding is None:
            blocked.append(
                {
                    "family": family,
                    "checkpoint_sha256": legacy["checkpoint_sha256"],
                    "reason": "BLOCKED_IDENTITY_RESOLUTION",
                }
            )
            continue
        matches = [
            x
            for x in records
            if isinstance(x, dict)
            and x.get("family") == family
            and x.get("name") == binding["release_name"]
        ]
        if len(matches) > 1:
            raise MigrationError(f"MIGRATION_EXISTING_MODEL_NOT_UNIQUE:{family}")
        if matches:
            existing = matches[0]
            expected = {
                "architecture_id": binding["architecture_id"],
                "checkpoint_artifact_id": binding["checkpoint_artifact_id"],
                "checkpoint_content_sha256": legacy["checkpoint_sha256"],
                "engine_release_id": binding["engine_release_id"],
                "source_code_sha": legacy["source_code_sha"],
            }
            if any(existing.get(k) != v for k, v in expected.items()):
                raise MigrationError(f"MIGRATION_EXISTING_MODEL_SPEC_DRIFT:{family}")
            migrated.append(existing)
            continue
        created = api.post(
            "/v1/model-releases",
            {
                "family": family,
                "name": binding["release_name"],
                "architecture_id": binding["architecture_id"],
                "checkpoint_artifact_id": binding["checkpoint_artifact_id"],
                "checkpoint_content_sha256": legacy["checkpoint_sha256"],
                "engine_release_id": binding["engine_release_id"],
                "originating_experiment_id": experiment_id,
                "source_code_sha": legacy["source_code_sha"],
                "initial_state": "DRAFT",
                "metadata": {
                    "migration_id": descriptor["migration_id"],
                    "legacy_name": legacy["legacy_name"],
                    "legacy_checkpoint_step": legacy["checkpoint_step"],
                    "legacy_binding_sha256": legacy["binding_sha256"],
                    "promotion_performed_by_migration": False,
                },
                "created_by": "p1-operational-migration",
            },
        )
        if not isinstance(created, dict) or created.get("lifecycle_state") != "DRAFT":
            raise MigrationError(f"MIGRATION_MODEL_CREATE_RESPONSE_INVALID:{family}")
        migrated.append(created)
    return migrated, blocked


def dry_plan(descriptor: dict[str, Any], bindings: dict[str, dict[str, str]]) -> dict[str, Any]:
    experiment = descriptor["experiment"]
    return {
        "schema": "RealSaS.OperationalResearchStateMigrationPlan.v1",
        "migration_id": descriptor["migration_id"],
        "mode": "DRY_RUN",
        "experiment": {
            "slug": experiment["slug"],
            "action": "ENSURE_" + experiment["initial_state"],
            "historical_training_executed": experiment["metadata"]["scientific_training_executed_at_migration"],
        },
        "models": [
            {
                "family": model["family"],
                "checkpoint_sha256": model["checkpoint_sha256"],
                "action": "ENSURE_DRAFT_MODEL_RELEASE"
                if model["family"] in bindings
                else "BLOCKED_IDENTITY_RESOLUTION",
            }
            for model in descriptor["legacy_models"]
        ],
        "promotion": "FORBIDDEN",
        "scientific_execution": "FORBIDDEN",
        "branch_mutation": "FORBIDDEN",
    }


def apply(api: API, descriptor: dict[str, Any], bindings: dict[str, dict[str, str]]) -> dict[str, Any]:
    experiment = ensure_experiment(api, descriptor["experiment"])
    experiment_id = require_uuid(experiment.get("id"), "EXPERIMENT_ID")
    links = ensure_links(api, descriptor, experiment_id)
    models, blocked = ensure_models(api, descriptor, experiment_id, bindings)
    status = api.get("/v1/ops/status")
    if not isinstance(status, dict) or status.get("schema") != "RealSaS.OperationalStatus.v1":
        raise MigrationError("MIGRATION_OPERATIONAL_STATUS_INVALID")
    if not any(
        isinstance(x, dict) and x.get("id") == experiment_id
        for x in status.get("active_experiments", [])
    ):
        raise MigrationError("MIGRATION_EXPERIMENT_NOT_VISIBLE_IN_STATUS")
    return {
        "schema": "RealSaS.OperationalResearchStateMigrationReceipt.v1",
        "migration_id": descriptor["migration_id"],
        "experiment": experiment,
        "links": links,
        "model_releases": models,
        "blocked_models": blocked,
        "promotion_performed": False,
        "scientific_execution_performed": False,
        "historical_training_executed_at_migration": descriptor["experiment"]["metadata"][
            "scientific_training_executed_at_migration"
        ],
        "operational_status_verified": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--descriptor", required=True, type=Path)
    parser.add_argument("--repo", default=Path("."), type=Path)
    parser.add_argument("--api", default="http://127.0.0.1:8080")
    parser.add_argument("--bindings", type=Path)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    try:
        descriptor = validate_descriptor(load_json(args.descriptor))
        bindings = validate_bindings(load_json(args.bindings) if args.bindings else None, descriptor)
        provenance = verify_git(args.repo.resolve(), descriptor)
        result = apply(API(args.api), descriptor, bindings) if args.apply else dry_plan(descriptor, bindings)
        result["mode"] = "APPLY" if args.apply else "DRY_RUN"
        result["git_provenance"] = provenance
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
