from __future__ import annotations

import importlib.util
import sys
import uuid
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "platform_operational_migrate_v1",
    ROOT / "tools" / "platform_operational_migrate_v1.py",
)
assert SPEC and SPEC.loader
migration = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = migration
SPEC.loader.exec_module(migration)


def descriptor():
    return {
        "schema": migration.SCHEMA,
        "migration_id": "test",
        "source_authority": {
            "legacy_branch": "research/example",
            "legacy_branch_head_sha": "1" * 40,
            "execution_binding_path": "canonical/BINDING.json",
            "execution_binding_git_blob_sha": "2" * 40,
            "scientific_execution_source_sha": "3" * 40,
            "scientific_execution_source_mutation_allowed": False,
        },
        "experiment": {
            "slug": "structural-evidence-a-b-v2",
            "name": "Structural Evidence A/B V2",
            "research_question": "question",
            "hypothesis": "hypothesis",
            "initial_state": "RUNNING",
            "created_by": "test",
            "metadata": {
                "training_execution_state": "COMPLETED_FROZEN_ARM_CHECKPOINTS",
                "scientific_training_executed_at_migration": True,
                "scientific_free_running_result_observed_at_migration": False,
                "scientific_causal_result_observed_at_migration": False,
            },
        },
        "links": {"attempt_ids": [], "campaign_ids": []},
        "legacy_models": [
            {
                "family": "IRIS",
                "legacy_name": "iris-v8",
                "source_code_sha": "4" * 40,
                "checkpoint_step": 12000,
                "checkpoint_sha256": "5" * 64,
                "binding_sha256": "6" * 64,
                "platform_checkpoint_artifact_id": None,
                "platform_engine_release_id": None,
                "migration_state": "BLOCKED_IDENTITY_RESOLUTION",
            }
        ],
        "scientific_invariants": {
            key: False
            for key in (
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
        },
        "branch_retirement": {
            "wholesale_merge_legacy_branch_to_main": False,
            "delete_legacy_branch_history": False,
        },
    }


def test_running_descriptor_preserves_historical_training_without_claiming_result():
    value = descriptor()
    assert migration.validate_descriptor(value) is value
    plan = migration.dry_plan(value, {})
    assert plan["experiment"] == {
        "slug": "structural-evidence-a-b-v2",
        "action": "ENSURE_RUNNING",
        "historical_training_executed": True,
    }
    assert plan["scientific_execution"] == "FORBIDDEN"
    assert plan["promotion"] == "FORBIDDEN"


def test_running_requires_historical_training_evidence():
    value = descriptor()
    value["experiment"]["metadata"]["scientific_training_executed_at_migration"] = False
    with pytest.raises(migration.MigrationError, match="RUNNING_REQUIRES_EXECUTED_TRAINING_HISTORY"):
        migration.validate_descriptor(value)


def test_preregistered_state_cannot_claim_training_already_happened():
    value = descriptor()
    value["experiment"]["initial_state"] = "PREREGISTERED"
    with pytest.raises(migration.MigrationError, match="PREREGISTERED_CANNOT_HAVE_EXECUTED_TRAINING"):
        migration.validate_descriptor(value)


def test_final_causal_result_cannot_be_silently_imported():
    value = descriptor()
    value["experiment"]["metadata"]["scientific_causal_result_observed_at_migration"] = True
    with pytest.raises(migration.MigrationError, match="CANNOT_IMPORT_FINAL_CAUSAL_RESULT"):
        migration.validate_descriptor(value)


def test_scientific_claim_is_rejected():
    value = descriptor()
    value["scientific_invariants"]["claim_product_promotion"] = True
    with pytest.raises(migration.MigrationError, match="SCIENTIFIC_INVARIANT_VIOLATION"):
        migration.validate_descriptor(value)


def test_missing_platform_model_identity_stays_blocked():
    value = migration.validate_descriptor(descriptor())
    plan = migration.dry_plan(value, {})
    assert plan["models"][0]["action"] == "BLOCKED_IDENTITY_RESOLUTION"
    assert plan["promotion"] == "FORBIDDEN"


def test_binding_must_match_legacy_checkpoint_hash():
    value = migration.validate_descriptor(descriptor())
    binding = {
        "schema": migration.BINDINGS_SCHEMA,
        "models": {
            "IRIS": {
                "checkpoint_artifact_id": str(uuid.uuid4()),
                "engine_release_id": str(uuid.uuid4()),
                "architecture_id": "RealSaS.IRIS.V8",
                "release_name": "iris-v8-migrated",
                "expected_checkpoint_sha256": "7" * 64,
            }
        },
    }
    with pytest.raises(migration.MigrationError, match="CHECKPOINT_SHA_MISMATCH"):
        migration.validate_bindings(binding, value)


def test_exact_binding_only_authorizes_draft_model_plan():
    value = migration.validate_descriptor(descriptor())
    binding = migration.validate_bindings(
        {
            "schema": migration.BINDINGS_SCHEMA,
            "models": {
                "IRIS": {
                    "checkpoint_artifact_id": str(uuid.uuid4()),
                    "engine_release_id": str(uuid.uuid4()),
                    "architecture_id": "RealSaS.IRIS.V8",
                    "release_name": "iris-v8-migrated",
                    "expected_checkpoint_sha256": "5" * 64,
                }
            },
        },
        value,
    )
    plan = migration.dry_plan(value, binding)
    assert plan["models"][0]["action"] == "ENSURE_DRAFT_MODEL_RELEASE"
    assert plan["promotion"] == "FORBIDDEN"


class FakeAPI:
    def __init__(self):
        self.experiments = []
        self.posts = []

    def get(self, path):
        assert path == "/v1/experiments"
        return self.experiments

    def post(self, path, payload):
        self.posts.append((path, payload))
        if path == "/v1/experiments":
            assert payload["initial_state"] == "PREREGISTERED"
            record = {
                "id": str(uuid.uuid4()),
                **payload,
                "lifecycle_state": "PREREGISTERED",
            }
            record.pop("initial_state", None)
            self.experiments = [record]
            return {"experiment": record, "reused": False}
        if path == "/v1/experiments/transition":
            record = dict(self.experiments[0])
            record["lifecycle_state"] = payload["to_state"]
            self.experiments = [record]
            return record
        raise AssertionError(path)


def test_running_import_is_create_preregistered_then_transition_running():
    value = migration.validate_descriptor(descriptor())
    api = FakeAPI()
    record = migration.ensure_experiment(api, value["experiment"])
    assert record["lifecycle_state"] == "RUNNING"
    assert [path for path, _ in api.posts] == [
        "/v1/experiments",
        "/v1/experiments/transition",
    ]


def test_existing_same_slug_with_stale_metadata_fails_closed():
    value = migration.validate_descriptor(descriptor())
    api = FakeAPI()
    api.experiments = [
        {
            "id": str(uuid.uuid4()),
            "slug": value["experiment"]["slug"],
            "name": value["experiment"]["name"],
            "research_question": value["experiment"]["research_question"],
            "hypothesis": value["experiment"]["hypothesis"],
            "metadata": {"scientific_training_executed_at_migration": False},
            "lifecycle_state": "PREREGISTERED",
        }
    ]
    with pytest.raises(migration.MigrationError, match="METADATA_DRIFT"):
        migration.ensure_experiment(api, value["experiment"])
