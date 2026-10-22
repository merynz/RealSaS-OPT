from __future__ import annotations

import importlib.util
import sys
import uuid
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("platform_operational_migrate_v1", ROOT / "tools" / "platform_operational_migrate_v1.py")
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
            "initial_state": "PREREGISTERED",
            "created_by": "test",
            "metadata": {
                "optimizer_steps_executed_at_migration": 0,
                "scientific_training_executed_at_migration": False,
                "scientific_pass_or_fail_observed_at_migration": False,
            },
        },
        "links": {"attempt_ids": [], "campaign_ids": []},
        "legacy_models": [{
            "family": "IRIS",
            "legacy_name": "iris-v8",
            "source_code_sha": "4" * 40,
            "checkpoint_step": 12000,
            "checkpoint_sha256": "5" * 64,
            "binding_sha256": "6" * 64,
            "platform_checkpoint_artifact_id": None,
            "platform_engine_release_id": None,
            "migration_state": "BLOCKED_IDENTITY_RESOLUTION",
        }],
        "scientific_invariants": {key: False for key in (
            "change_thresholds", "change_architecture", "change_optimizer", "change_schedule",
            "execute_optimizer", "execute_scientific_courts", "claim_pass_or_fail",
            "claim_unseen_generalization", "claim_product_promotion",
            "teacher_truth_as_inference_conditioning",
            "named_semantic_taxonomy_as_structural_authority",
            "material_id_as_drawable_owner_truth",
        )},
        "branch_retirement": {
            "wholesale_merge_legacy_branch_to_main": False,
            "delete_legacy_branch_history": False,
        },
    }


def test_descriptor_is_preregistered_and_non_scientific():
    value = descriptor()
    assert migration.validate_descriptor(value) is value
    value["experiment"]["initial_state"] = "RUNNING"
    with pytest.raises(migration.MigrationError, match="MUST_ENTER_PREREGISTERED"):
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
    assert plan["scientific_execution"] == "FORBIDDEN"


def test_binding_must_match_legacy_checkpoint_hash():
    value = migration.validate_descriptor(descriptor())
    binding = {
        "schema": migration.BINDINGS_SCHEMA,
        "models": {"IRIS": {
            "checkpoint_artifact_id": str(uuid.uuid4()),
            "engine_release_id": str(uuid.uuid4()),
            "architecture_id": "RealSaS.IRIS.V8",
            "release_name": "iris-v8-migrated",
            "expected_checkpoint_sha256": "7" * 64,
        }},
    }
    with pytest.raises(migration.MigrationError, match="CHECKPOINT_SHA_MISMATCH"):
        migration.validate_bindings(binding, value)


def test_exact_binding_only_authorizes_draft_model_plan():
    value = migration.validate_descriptor(descriptor())
    binding = migration.validate_bindings({
        "schema": migration.BINDINGS_SCHEMA,
        "models": {"IRIS": {
            "checkpoint_artifact_id": str(uuid.uuid4()),
            "engine_release_id": str(uuid.uuid4()),
            "architecture_id": "RealSaS.IRIS.V8",
            "release_name": "iris-v8-migrated",
            "expected_checkpoint_sha256": "5" * 64,
        }},
    }, value)
    plan = migration.dry_plan(value, binding)
    assert plan["models"][0]["action"] == "ENSURE_DRAFT_MODEL_RELEASE"
    assert plan["promotion"] == "FORBIDDEN"
