from __future__ import annotations

import importlib.util
import json
import sys
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
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


def descriptor() -> dict:
    return {
        "schema": migration.SCHEMA,
        "migration_id": "test-migration",
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
            "created_by": "migration-test",
            "metadata": {
                "scientific_training_executed_at_migration": False,
                "optimizer_steps_executed_at_migration": 0,
                "scientific_pass_or_fail_observed_at_migration": False,
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


def test_descriptor_is_strictly_preregistered_and_non_scientific():
    value = descriptor()
    assert migration.validate_descriptor(value) is value
    value["experiment"]["initial_state"] = "RUNNING"
    with pytest.raises(migration.MigrationError, match="MUST_ENTER_PREREGISTERED"):
        migration.validate_descriptor(value)


def test_descriptor_rejects_scientific_or_product_claims():
    value = descriptor()
    value["scientific_invariants"]["claim_product_promotion"] = True
    with pytest.raises(migration.MigrationError, match="SCIENTIFIC_INVARIANT_VIOLATION"):
        migration.validate_descriptor(value)


def test_model_release_stays_blocked_without_exact_platform_ids():
    value = migration.validate_descriptor(descriptor())
    plan = migration.build_plan(value, {})
    assert plan["models"] == [
        {
            "family": "IRIS",
            "action": "BLOCKED_IDENTITY_RESOLUTION",
            "checkpoint_sha256": "5" * 64,
        }
    ]
    assert plan["promotion"] == "FORBIDDEN"
    assert plan["scientific_execution"] == "FORBIDDEN"


def test_bindings_require_exact_checkpoint_sha():
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


class State:
    def __init__(self):
        self.experiment_id = str(uuid.uuid4())
        self.experiments: list[dict] = []
        self.models: list[dict] = []
        self.links: list[dict] = []


class Handler(BaseHTTPRequestHandler):
    state: State

    def log_message(self, _format, *_args):
        return

    def _json(self, status: int, value):
        raw = json.dumps(value).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _body(self):
        return json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))

    def do_GET(self):
        if self.path == "/v1/experiments":
            return self._json(200, self.state.experiments)
        if self.path == "/v1/model-releases":
            return self._json(200, self.state.models)
        if self.path == "/v1/ops/status":
            return self._json(
                200,
                {
                    "schema": "RealSaS.OperationalStatus.v1",
                    "active_experiments": self.state.experiments,
                    "open_attempts": [],
                    "active_campaigns": [],
                    "current_models": [],
                    "current_products": [],
                },
            )
        return self._json(404, {"error": "not found"})

    def do_POST(self):
        body = self._body()
        if self.path == "/v1/experiments":
            experiment = {
                "id": self.state.experiment_id,
                **body,
                "lifecycle_state": body["initial_state"],
            }
            experiment.pop("initial_state", None)
            self.state.experiments.append(experiment)
            return self._json(200, {"experiment": experiment, "reused": False})
        if self.path == "/v1/experiments/link":
            self.state.links.append(body)
            return self._json(200, {**body, "reused": False})
        if self.path == "/v1/model-releases":
            model = {"id": str(uuid.uuid4()), **body, "lifecycle_state": body["initial_state"]}
            model.pop("initial_state", None)
            self.state.models.append(model)
            return self._json(200, model)
        return self._json(404, {"error": "not found"})


def server():
    state = State()
    handler = type("MigrationHandler", (Handler,), {"state": state})
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return state, httpd, thread


def test_apply_creates_experiment_but_not_model_without_identity_binding():
    value = migration.validate_descriptor(descriptor())
    state, httpd, thread = server()
    try:
        result = migration.apply(migration.API(f"http://127.0.0.1:{httpd.server_port}"), value, {})
    finally:
        httpd.shutdown()
        thread.join(timeout=5)
    assert result["experiment"]["slug"] == "structural-evidence-a-b-v2"
    assert result["blocked_models"][0]["reason"] == "BLOCKED_IDENTITY_RESOLUTION"
    assert state.models == []
    assert result["promotion_performed"] is False
    assert result["scientific_execution_performed"] is False


def test_apply_with_exact_binding_creates_draft_unpromoted_model_release():
    value = migration.validate_descriptor(descriptor())
    binding_doc = {
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
    }
    bindings = migration.validate_bindings(binding_doc, value)
    state, httpd, thread = server()
    try:
        result = migration.apply(
            migration.API(f"http://127.0.0.1:{httpd.server_port}"), value, bindings
        )
    finally:
        httpd.shutdown()
        thread.join(timeout=5)
    assert result["blocked_models"] == []
    assert len(state.models) == 1
    assert state.models[0]["family"] == "IRIS"
    assert state.models[0]["originating_experiment_id"] == state.experiment_id
    assert state.models[0]["checkpoint_content_sha256"] == "5" * 64
    assert state.models[0]["lifecycle_state"] == "DRAFT"
    assert result["promotion_performed"] is False
