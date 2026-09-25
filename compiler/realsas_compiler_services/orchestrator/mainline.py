from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import importlib.util
import json
import os
import re
from pathlib import Path
from time import perf_counter
from typing import Any

from compiler.realsas_compiler_services.orchestrator.status_semantics import (
    DEMO_ONLY_STATUS,
    FAIL_STATUSES,
    PASS_STATUSES,
    assert_demo_only_scope,
    dependency_status_admissible,
    normalize_success_status,
)

ROOT = Path(__file__).resolve().parents[3]
PLAN_PATH = ROOT / "canonical" / "MAINLINE_EXECUTION_PLAN_V2.json"
LEDGER_PATH = ROOT / "canonical" / "ACTIVE_RUN_V2.json"
READINESS_PATH = ROOT / "canonical" / "V2_IMPLEMENTATION_READINESS.json"

_STAGE_RE = re.compile(r"^\d{2}_[A-Z0-9_]+$")


def _canon(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _canon(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_canon(item) for item in value]
    return value


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        _canon(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def content_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _stage_map(plan: dict) -> dict[str, dict]:
    return {str(stage["id"]): stage for stage in plan["stages"]}


def _ledger_map(ledger: dict) -> dict[str, dict]:
    return {str(row["id"]): row for row in ledger["stages"]}


def topological_stage_ids(plan: dict) -> tuple[str, ...]:
    stages = _stage_map(plan)
    indegree = {stage_id: 0 for stage_id in stages}
    children = {stage_id: set() for stage_id in stages}
    for stage_id, stage in stages.items():
        for dependency in stage.get("depends_on", ()):
            dependency = str(dependency)
            if dependency not in stages:
                raise RuntimeError(
                    f"MAINLINE_PLAN_DEPENDENCY_UNKNOWN:{stage_id}:{dependency}"
                )
            indegree[stage_id] += 1
            children[dependency].add(stage_id)

    ordinal = {str(stage["id"]): int(stage["ordinal"]) for stage in plan["stages"]}
    ready = sorted(
        (stage_id for stage_id, degree in indegree.items() if degree == 0),
        key=lambda stage_id: (ordinal[stage_id], stage_id),
    )
    ordered: list[str] = []
    while ready:
        stage_id = ready.pop(0)
        ordered.append(stage_id)
        for child in sorted(children[stage_id], key=lambda x: (ordinal[x], x)):
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
                ready.sort(key=lambda x: (ordinal[x], x))
    if len(ordered) != len(stages):
        cyclic = sorted(stage_id for stage_id, degree in indegree.items() if degree > 0)
        raise RuntimeError("MAINLINE_PLAN_CYCLE:" + ",".join(cyclic))
    return tuple(ordered)


def validate_plan(plan: dict) -> str:
    if plan.get("schema") != "RealSaS.MainlineExecutionPlan.v2":
        raise RuntimeError("MAINLINE_V2_PLAN_SCHEMA_DRIFT")
    stages = list(plan.get("stages") or ())
    if int(plan.get("stage_count", -1)) != 46 or len(stages) != 46:
        raise RuntimeError("MAINLINE_V2_REQUIRES_EXACT_46_STAGES")
    if plan.get("canonical_branch") != "main":
        raise RuntimeError("MAINLINE_V2_BRANCH_DRIFT")
    if plan.get("subject_specific_code_forbidden") is not True:
        raise RuntimeError("MAINLINE_V2_GENERICITY_DRIFT")

    ids: list[str] = []
    ordinals: list[int] = []
    for stage in stages:
        stage_id = str(stage.get("id", ""))
        ordinal = int(stage.get("ordinal", -1))
        ids.append(stage_id)
        ordinals.append(ordinal)
        if not _STAGE_RE.fullmatch(stage_id):
            raise RuntimeError(f"MAINLINE_V2_STAGE_ID_INVALID:{stage_id}")
        adapter = str(stage.get("adapter", "")).strip()
        if not adapter:
            raise RuntimeError(f"MAINLINE_V2_ADAPTER_MISSING:{stage_id}")
        keys = stage.get("manifest_keys")
        if not isinstance(keys, list) or not all(
            isinstance(item, str) and item for item in keys
        ):
            raise RuntimeError(f"MAINLINE_V2_MANIFEST_SCOPE_INVALID:{stage_id}")
        policy = dict(stage.get("policy") or {})
        if (
            policy.get("fail_closed") is not True
            or policy.get("output_hash_required") is not True
        ):
            raise RuntimeError(f"MAINLINE_V2_FAIL_CLOSED_POLICY_DRIFT:{stage_id}")

    if len(ids) != len(set(ids)):
        raise RuntimeError("MAINLINE_V2_DUPLICATE_STAGE_ID")
    if sorted(ordinals) != list(range(1, 47)):
        raise RuntimeError("MAINLINE_V2_ORDINAL_SET_DRIFT")
    if len(ordinals) != len(set(ordinals)):
        raise RuntimeError("MAINLINE_V2_DUPLICATE_ORDINAL")
    topological_stage_ids(plan)
    return content_sha256(plan)


CURRENT_V2_FORBIDDEN_IMPORT_MODULES = {
    "compiler.realsas_compiler_core.product_artifact_codec_v1",
    "compiler.realsas_compiler_core.presentation_graph_v1",
    "compiler.realsas_compiler_core.product_appearance_v1",
    "compiler.realsas_compiler_core.product_composition_v1",
    "compiler.realsas_compiler_core.rest_render_v1",
    "compiler.realsas_compiler_core.rest_preservation_v1",
    "compiler.realsas_compiler_core.motion_compile_v1",
    "compiler.realsas_compiler_core.motion_dynamic_proof_v1",
    "compiler.realsas_compiler_core.motion_presentation_v1",
    "compiler.realsas_compiler_core.runtime_projection_v1",
    "compiler.realsas_compiler_core.playback_appearance_authority_v2",
    "compiler.realsas_compiler_core.playback_runtime_v3",
    "compiler.realsas_compiler_core.playback_runtime_v4",
    "compiler.realsas_compiler_core.playback_full_surface_v4",
    "compiler.realsas_compiler_services.export.current_v4_directional_runtime_v4",
    "compiler.realsas_compiler_services.export.current_v4_runtime_v2",
    "compiler.realsas_compiler_services.export.runtime_v4",
}

IMPLEMENTATION_CLOSURE_STATIC_PATHS = (
    "runtime/realsas_cpp/src/runtime_v2_caa_reference.cpp",
    ".github/workflows/current_mainline_self_hosted_ci.yml",
    ".github/workflows/current_runtime_self_hosted_ci.yml",
    ".github/workflows/model_mainline_source_gate.yml",
    ".github/workflows/native_runtime_source_gate.yml",
    ".github/workflows/proof_service_promotion_gate.yml",
    ".github/workflows/vf23_production_policy_e2e_bank.yml",
    ".github/workflows/v2_witness_orchestration_subject_free_dry_run.yml",
    ".github/workflows/subject2_knight_observation_preflight.yml",
    "canonical/REALSAS_CANONICAL_ARCHITECTURE_V2_20260920.json",
    "canonical/COMPLETE_APPEARANCE_AUTHORITY_V1_20260920.json",
    "canonical/CAA_V2_SUBJECT_FREE_NUMERICAL_POLICY_20260920.json",
    "canonical/PRESENTATION_PARTITION_POLICY_V1_20260921.json",
    "canonical/DYNAMIC_APPEARANCE_CONDITIONING_CALIBRATION_V1_20260921.json",
    "canonical/V1_TO_V2_ARCHITECTURE_TRANSITION_20260920.md",
    "requirements/mainline-ci.txt",
    "requirements/torch-cpu.txt",
    "tools/build_knowledge_artifact_catalog.py",
    "tools/render_authority_map.py",
    "tools/render_rehydration_packet.py",
    "tools/audit_context_coverage.py",
    "tools/verify_native_runtime_source_seal_v2.py",
    ".github/workflows/live_authority_map.yml",
    "compiler/realsas_compiler_services/orchestrator/mainline.py",
    "AGENTS.md",
    "SYSTEM_INDEX.md",
    "REPOSITORY_MAP.md",
    "canonical/README.md",
    "README.md",
)

IMPLEMENTATION_CLOSURE_DYNAMIC_GLOBS = (
    "canonical/COMPILER_RUNTIME_PROMOTION_SOURCE_SEAL_V*.json",
    "canonical/COMPILER_RUNTIME_SOURCE_EXTENSION_SEAL_V*.json",
)


def _implementation_closure_dynamic_files() -> tuple[str, ...]:
    rows: set[str] = set()
    for pattern in IMPLEMENTATION_CLOSURE_DYNAMIC_GLOBS:
        matches = sorted(ROOT.glob(pattern))
        if not matches:
            raise RuntimeError(
                "V2_IMPLEMENTATION_DYNAMIC_CLOSURE_GLOB_EMPTY:" + pattern
            )
        for path in matches:
            if not path.is_file():
                continue
            rows.add(str(path.relative_to(ROOT).as_posix()))
    return tuple(sorted(rows))


IMPLEMENTATION_CLOSURE_TEST_ROOTS = (
    "tests/repository",
    "tests/compiler",
    "tests/models",
    "tests/iris",
)


def _implementation_closure_test_files() -> tuple[str, ...]:
    rows: list[str] = []
    for root_rel in IMPLEMENTATION_CLOSURE_TEST_ROOTS:
        root = ROOT / root_rel
        if not root.is_dir():
            raise RuntimeError(f"V2_IMPLEMENTATION_TEST_ROOT_MISSING:{root_rel}")
        for path in sorted(root.rglob("*.py")):
            rows.append(str(path.relative_to(ROOT).as_posix()))
    return tuple(rows)


def implementation_closure_manifest(plan: dict | None = None) -> dict:
    plan = plan or load_json(PLAN_PATH)
    plan_hash = validate_plan(plan)
    adapter_rows = []
    imported_modules: set[str] = set()
    for stage in sorted(plan["stages"], key=lambda row: int(row["ordinal"])):
        adapter = str(stage["adapter"])
        module_name, separator, function_name = adapter.partition(":")
        if not separator or not module_name or not function_name:
            raise RuntimeError(f"MAINLINE_V2_ADAPTER_ID_INVALID:{adapter}")
        closure = _local_import_closure(module_name)
        imported_modules.update(name for name, _digest in closure)
        adapter_rows.append(
            {
                "stage_id": str(stage["id"]),
                "adapter": adapter,
                "implementation_hash": _adapter_impl_hash(adapter),
                "local_python_import_closure": [
                    {"module": name, "sha256": digest}
                    for name, digest in closure
                ],
            }
        )

    forbidden = sorted(
        module
        for module in imported_modules
        if module in CURRENT_V2_FORBIDDEN_IMPORT_MODULES
    )
    if forbidden:
        raise RuntimeError(
            "V2_CURRENT_CLOSURE_IMPORTS_DONOR_ERA_MODULE:"
            + ",".join(forbidden)
        )

    file_rows = []
    closure_files = (
        tuple(IMPLEMENTATION_CLOSURE_STATIC_PATHS)
        + _implementation_closure_dynamic_files()
        + _implementation_closure_test_files()
    )
    if len(closure_files) != len(set(closure_files)):
        raise RuntimeError("V2_IMPLEMENTATION_CLOSURE_DUPLICATE_FILE")
    for rel in closure_files:
        path = ROOT / rel
        if not path.is_file():
            raise RuntimeError(f"V2_IMPLEMENTATION_CLOSURE_FILE_MISSING:{rel}")
        file_rows.append({"path": rel, "sha256": sha256_file(path)})

    return {
        "schema": "RealSaS.V2ImplementationClosure.v1",
        "pipeline_plan_sha256": plan_hash,
        "adapter_implementation_closures": adapter_rows,
        "imported_module_count": len(imported_modules),
        "forbidden_import_modules": sorted(CURRENT_V2_FORBIDDEN_IMPORT_MODULES),
        "dynamic_governance_files": list(_implementation_closure_dynamic_files()),
        "critical_files": file_rows,
    }


def implementation_closure_sha256(plan: dict | None = None) -> str:
    return content_sha256(implementation_closure_manifest(plan))


def validate_readiness(plan: dict | None = None) -> str:
    plan = plan or load_json(PLAN_PATH)
    plan_hash = validate_plan(plan)
    if not READINESS_PATH.is_file():
        raise RuntimeError("V2_IMPLEMENTATION_READINESS_MISSING")
    readiness = load_json(READINESS_PATH)
    if readiness.get("schema") != "RealSaS.V2ImplementationReadiness.v1":
        raise RuntimeError("V2_IMPLEMENTATION_READINESS_SCHEMA_DRIFT")
    if readiness.get("status") != "READY_FOR_WITNESS_EXECUTION":
        raise RuntimeError(
            "V2_IMPLEMENTATION_NOT_READY:" + str(readiness.get("status", "UNKNOWN"))
        )
    if readiness.get("pipeline_plan_sha256") != plan_hash:
        raise RuntimeError("V2_IMPLEMENTATION_READINESS_PLAN_HASH_DRIFT")
    expected_closure = implementation_closure_sha256(plan)
    if readiness.get("implementation_closure_sha256") != expected_closure:
        raise RuntimeError(
            "V2_IMPLEMENTATION_READINESS_CLOSURE_HASH_DRIFT:"
            f"{readiness.get('implementation_closure_sha256','')}!={expected_closure}"
        )
    unbound = [
        str(stage["id"])
        for stage in plan["stages"]
        if str(stage.get("adapter", "")).strip() == "UNBOUND"
    ]
    if unbound:
        raise RuntimeError(
            "V2_IMPLEMENTATION_UNBOUND_STAGES:" + ",".join(unbound)
        )
    required = tuple(map(str, readiness.get("required_proofs") or ()))
    if not required:
        raise RuntimeError("V2_IMPLEMENTATION_READINESS_PROOFS_EMPTY")
    proof_rows = {
        str(row.get("proof_id")): row for row in readiness.get("proofs") or ()
    }
    missing = [
        proof_id
        for proof_id in required
        if proof_id not in proof_rows
        or str(proof_rows[proof_id].get("status")) != "PASS"
        or len(str(proof_rows[proof_id].get("sha256", ""))) != 64
    ]
    if missing:
        raise RuntimeError(
            "V2_IMPLEMENTATION_READINESS_PROOF_MISSING:" + ",".join(missing)
        )
    return content_sha256(readiness)


def validate_witness_authorization(plan: dict | None = None) -> str:
    """Require technical readiness plus an explicit, separately recorded user approval."""
    plan = plan or load_json(PLAN_PATH)
    readiness_digest = validate_readiness(plan)
    readiness = load_json(READINESS_PATH)
    seal = dict(readiness.get("readiness_seal") or {})
    if str(seal.get("status") or "") != "PASS":
        raise RuntimeError(
            "V2_WITNESS_AUTHORIZATION_REQUIRES_PASS_READINESS_SEAL"
        )
    if seal.get("witness_execution_allowed") is not True:
        raise RuntimeError(
            "V2_WITNESS_EXECUTION_NOT_TECHNICALLY_ALLOWED"
        )
    if seal.get("user_approval_required") is not True:
        raise RuntimeError(
            "V2_WITNESS_USER_APPROVAL_REQUIREMENT_DRIFT"
        )
    if seal.get("witness_execution_authorized_now") is not True:
        raise RuntimeError(
            "V2_WITNESS_EXPLICIT_USER_APPROVAL_REQUIRED"
        )
    if str(seal.get("user_approval_state") or "") != (
        "APPROVED_EXPLICITLY_BY_USER"
    ):
        raise RuntimeError(
            "V2_WITNESS_USER_APPROVAL_STATE_INVALID:"
            + str(seal.get("user_approval_state") or "UNKNOWN")
        )
    # Run progress is owned by the run-local ActiveRunLedger, not by the
    # durable approval predicate. Once explicit approval is granted it must
    # remain valid for subsequent execute/resume calls.
    return readiness_digest



def validate_demo_witness_authorization(manifest: dict, *, subject_id: str) -> str:
    demo=dict(manifest.get("demo_execution") or {})
    if str(demo.get("schema") or "")!="RealSaS.DemoExecutionAuthorization.v1":
        raise RuntimeError("DEMO_WITNESS_AUTHORIZATION_SCHEMA_INVALID")
    if str(demo.get("mode") or "")!="DEMO_ONLY":
        raise RuntimeError("DEMO_WITNESS_MODE_INVALID")
    if demo.get("explicit_user_approval") is not True:
        raise RuntimeError("DEMO_WITNESS_EXPLICIT_USER_APPROVAL_REQUIRED")
    if demo.get("product_authority_claimed") is not False:
        raise RuntimeError("DEMO_WITNESS_PRODUCT_AUTHORITY_FORBIDDEN")
    if demo.get("stage13_scientific_pass") is not False:
        raise RuntimeError("DEMO_WITNESS_STAGE13_SCIENTIFIC_STATE_DRIFT")
    if demo.get("allow_stage13_scientific_fail_for_demo") is not True:
        raise RuntimeError("DEMO_WITNESS_STAGE13_DEMO_ADMISSION_REQUIRED")
    if str(manifest.get("subject_id") or "")!=str(subject_id):
        raise RuntimeError("DEMO_WITNESS_SUBJECT_ID_DRIFT")
    authority=dict(demo.get("authority") or {})
    rel=str(authority.get("path") or "")
    expected=str(authority.get("sha256") or "")
    path=(ROOT/rel).resolve()
    if not rel or len(expected)!=64 or ROOT not in path.parents or not path.is_file():
        raise RuntimeError("DEMO_WITNESS_AUTHORITY_REF_INVALID")
    if sha256_file(path)!=expected:
        raise RuntimeError("DEMO_WITNESS_AUTHORITY_SHA_DRIFT")
    payload=load_json(path)
    if str(payload.get("schema") or "")!="RealSaS.DemoExecutionAuthority.v1":
        raise RuntimeError("DEMO_WITNESS_AUTHORITY_SCHEMA_DRIFT")
    if str(payload.get("status") or "")!="APPROVED_DEMO_ONLY":
        raise RuntimeError("DEMO_WITNESS_AUTHORITY_NOT_APPROVED")
    if payload.get("product_authority_claimed") is not False or payload.get("stage13_scientific_pass") is not False:
        raise RuntimeError("DEMO_WITNESS_AUTHORITY_SCOPE_DRIFT")
    return content_sha256(payload)


def validate_ledger(plan: dict, ledger: dict) -> None:
    plan_hash = validate_plan(plan)
    if ledger.get("schema") != "RealSaS.ActiveRunLedger.v2":
        raise RuntimeError("ACTIVE_RUN_V2_LEDGER_SCHEMA_DRIFT")
    if ledger.get("canonical_branch") != "main":
        raise RuntimeError("ACTIVE_RUN_V2_LEDGER_BRANCH_DRIFT")
    execution_class = str(ledger.get("execution_class") or "WITNESS")
    if execution_class not in {"WITNESS", "IMPLEMENTATION_AUDIT", "DEMO_WITNESS"}:
        raise RuntimeError("ACTIVE_RUN_V2_EXECUTION_CLASS_INVALID")
    assert_demo_only_scope(ledger)
    if (
        execution_class == "IMPLEMENTATION_AUDIT"
        and not str(ledger.get("subject_id") or "").startswith("SUBJECT_FREE_")
    ):
        raise RuntimeError("ACTIVE_RUN_V2_AUDIT_SUBJECT_ID_INVALID")
    if ledger.get("pipeline_plan_sha256") != plan_hash:
        raise RuntimeError("ACTIVE_RUN_V2_LEDGER_PLAN_HASH_DRIFT")

    rows = list(ledger.get("stages") or ())
    if len(rows) != 46:
        raise RuntimeError("ACTIVE_RUN_V2_LEDGER_STAGE_COUNT_DRIFT")
    if {str(row.get("id")) for row in rows} != {
        str(stage["id"]) for stage in plan["stages"]
    }:
        raise RuntimeError("ACTIVE_RUN_V2_LEDGER_STAGE_ID_SET_DRIFT")

    by_id = _ledger_map(ledger)
    for stage in plan["stages"]:
        row = by_id[stage["id"]]
        if int(row.get("ordinal", -1)) != int(stage["ordinal"]):
            raise RuntimeError(f"ACTIVE_RUN_V2_LEDGER_ORDINAL_DRIFT:{stage['id']}")
        if dependency_status_admissible(ledger, str(row.get("status") or "")):
            for dependency in stage.get("depends_on", ()):
                if not dependency_status_admissible(
                    ledger, str(by_id[str(dependency)].get("status") or "")
                ):
                    raise RuntimeError(
                        "ACTIVE_RUN_V2_PASS_WITH_UNPASSED_DEPENDENCY:"
                        f"{stage['id']}:{dependency}"
                    )

    completed = sum(
        dependency_status_admissible(ledger, str(row.get("status") or ""))
        for row in rows
    )
    if int(ledger.get("completed_count", -1)) != completed:
        raise RuntimeError("ACTIVE_RUN_V2_LEDGER_PROGRESS_DRIFT")
    expected_ready = list(ready_stage_ids(plan, ledger))
    if list(ledger.get("ready_stage_ids") or ()) != expected_ready:
        raise RuntimeError("ACTIVE_RUN_V2_LEDGER_READY_SET_DRIFT")


def authority_root() -> Path:
    raw = os.environ.get("REALSAS_AUTHORITY_ROOT", "").strip()
    return (
        Path(raw).expanduser().resolve()
        if raw
        else (Path.home() / "realsas_authority").resolve()
    )


def run_manifest_path(run_id: str) -> Path:
    return authority_root() / "runs" / run_id / "run_manifest.json"


def run_ledger_path(run_id: str) -> Path:
    return authority_root() / "runs" / run_id / "ACTIVE_RUN_V2.json"


def build_fresh_run_ledger(
    plan: dict,
    *,
    run_id: str,
    subject_id: str,
    manifest_ref: str,
    architecture_scope: str = "REALSAS_V2_FRESH_WITNESS",
    execution_class: str = "WITNESS",
) -> dict:
    plan_hash = validate_plan(plan)
    rows = [
        {
            "ordinal": int(stage["ordinal"]),
            "id": str(stage["id"]),
            "status": "PENDING",
            "attempts": 0,
            "input_fingerprint": "",
            "implementation_hash": "",
            "policy_hash": "",
            "outputs": [],
            "diagnostics_hash": "",
            "blockers": [],
            "wall_seconds": 0.0,
            "performance": {},
        }
        for stage in plan["stages"]
    ]
    execution_class = str(execution_class).upper()
    if execution_class not in {"WITNESS", "IMPLEMENTATION_AUDIT", "DEMO_WITNESS"}:
        raise RuntimeError(f"RUN_EXECUTION_CLASS_INVALID:{execution_class}")
    if execution_class == "IMPLEMENTATION_AUDIT" and not str(subject_id).startswith("SUBJECT_FREE_"):
        raise RuntimeError("IMPLEMENTATION_AUDIT_REQUIRES_SUBJECT_FREE_SUBJECT_ID")
    ledger = {
        "schema": "RealSaS.ActiveRunLedger.v2",
        "run_id": str(run_id),
        "subject_id": str(subject_id),
        "canonical_branch": "main",
        "architecture_scope": str(architecture_scope),
        "execution_class": execution_class,
        "pipeline_plan": "canonical/MAINLINE_EXECUTION_PLAN_V2.json",
        "pipeline_plan_sha256": plan_hash,
        "status": "ACTIVE",
        "execution_enabled": True,
        "completed_count": 0,
        "failed_count": 0,
        "total_count": len(rows),
        "ready_stage_ids": [],
        "failed_stage_ids": [],
        "blocked_by_failed_dependency": {},
        "run_manifest_ref": str(manifest_ref),
        "stages": rows,
        "history": [
            {
                "event": "FRESH_RUN_LEDGER_MATERIALIZED",
                "scope": str(architecture_scope),
            }
        ],
    }
    _refresh(plan, ledger)
    validate_ledger(plan, ledger)
    return ledger


def _local_module_path(module_name: str) -> Path | None:
    rel = Path(*str(module_name).split("."))
    file_path = (ROOT / rel).with_suffix(".py")
    if file_path.is_file():
        return file_path.resolve()
    init_path = ROOT / rel / "__init__.py"
    if init_path.is_file():
        return init_path.resolve()
    return None


def _local_import_closure(module_name: str) -> tuple[tuple[str, str], ...]:
    seen: set[str] = set()
    rows: list[tuple[str, str]] = []

    def visit(name: str) -> None:
        if name in seen:
            return
        path = _local_module_path(name)
        if path is None:
            return
        seen.add(name)
        rows.append((name, sha256_file(path)))
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        package = name.rpartition(".")[0]
        for node in ast.walk(tree):
            candidates: list[str] = []
            if isinstance(node, ast.Import):
                candidates.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    relative = "." * int(node.level) + (node.module or "")
                    try:
                        base = importlib.util.resolve_name(relative, package or name)
                    except Exception:
                        base = ""
                else:
                    base = str(node.module or "")
                if base:
                    candidates.append(base)
                    for alias in node.names:
                        child = f"{base}.{alias.name}"
                        if _local_module_path(child) is not None:
                            candidates.append(child)
            for candidate in candidates:
                visit(candidate)

    visit(module_name)
    return tuple(sorted(rows))


def _adapter_impl_hash(adapter: str) -> str:
    module_name, separator, function_name = adapter.partition(":")
    if not separator or not module_name or not function_name:
        raise RuntimeError(f"MAINLINE_V2_ADAPTER_ID_INVALID:{adapter}")
    module = importlib.import_module(module_name)
    if not callable(getattr(module, function_name, None)):
        raise RuntimeError(f"MAINLINE_V2_ADAPTER_CALLABLE_MISSING:{adapter}")
    closure = _local_import_closure(module_name)
    if not closure:
        raise RuntimeError(f"MAINLINE_V2_IMPLEMENTATION_CLOSURE_EMPTY:{adapter}")
    return content_sha256(
        {
            "schema": "RealSaS.AdapterImplementationClosure.v2",
            "adapter": adapter,
            "local_python_import_closure": [
                {"module": name, "sha256": digest} for name, digest in closure
            ],
        }
    )


def _manifest_subset(manifest: dict, stage: dict) -> dict:
    """Return only manifest state semantically consumed by this stage.

    External-fit preregistration must remain immutable when execution artifacts
    are filled in later. Stages 26/30 read only the preregistration ref; their
    downstream execution stages fingerprint the complete fit section.
    """
    stage_id = str(stage.get("id") or "")
    subset = {}
    for key in stage["manifest_keys"]:
        value = manifest.get(key)
        if (
            stage_id == "26_GEPPETTO_FIT_PREREGISTERED"
            and key == "geppetto_fit"
        ) or (
            stage_id == "30_ARACHNE_FIT_PREREGISTERED"
            and key == "arachne_fit"
        ):
            cfg = dict(value or {})
            subset[key] = _canon(
                {"preregistration": cfg.get("preregistration")}
            )
        else:
            subset[key] = _canon(value)
    return subset


def _path_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _outputs_verify(row: dict, *, allowed_root: Path | None = None) -> bool:
    outputs = list(row.get("outputs") or ())
    if not outputs:
        return False
    for output in outputs:
        path = Path(str(output.get("path", ""))).expanduser().resolve()
        digest = str(output.get("sha256", ""))
        schema = str(output.get("schema", "") or "")
        authority_class = str(output.get("authority_class", "") or "")
        expected_bytes = int(output.get("bytes", -1))
        if allowed_root is not None and not _path_within(path, allowed_root):
            return False
        if (
            not path.is_file()
            or len(digest) != 64
            or schema in {"", "UNSPECIFIED"}
            or not authority_class
            or expected_bytes < 0
            or path.stat().st_size != expected_bytes
            or sha256_file(path) != digest
        ):
            return False
    return True


def _fingerprint(
    plan: dict,
    ledger: dict,
    manifest: dict,
    stage: dict,
    implementation_hash: str,
) -> tuple[str, str]:
    by_id = _ledger_map(ledger)
    dependency_outputs: list[dict[str, Any]] = []
    for dependency in stage.get("depends_on", ()):
        for output in by_id[str(dependency)].get("outputs", ()):
            digest = str(output.get("sha256", "") or "")
            if not digest:
                continue
            dependency_outputs.append(
                {
                    "dependency_stage_id": str(dependency),
                    "sha256": digest,
                    "bytes": int(output.get("bytes", -1)),
                    "schema": str(output.get("schema", "") or ""),
                    "authority_class": str(
                        output.get("authority_class", "") or ""
                    ),
                }
            )
    policy_hash = content_sha256(stage["policy"])
    fingerprint = content_sha256(
        {
            "schema": "RealSaS.StageInputFingerprint.v2",
            "run_id": ledger["run_id"],
            "stage_id": stage["id"],
            "manifest_subset": _manifest_subset(manifest, stage),
            "dependency_output_identities": dependency_outputs,
            "policy_hash": policy_hash,
            "implementation_hash": implementation_hash,
            "pipeline_plan_sha256": ledger["pipeline_plan_sha256"],
        }
    )
    return fingerprint, policy_hash


def dependency_failure_ids(plan: dict, ledger: dict, stage_id: str) -> tuple[str, ...]:
    stages = _stage_map(plan)
    rows = _ledger_map(ledger)
    failures: set[str] = set()
    stack = list(map(str, stages[stage_id].get("depends_on", ())))
    seen: set[str] = set()
    while stack:
        dependency = stack.pop()
        if dependency in seen:
            continue
        seen.add(dependency)
        status = str(rows[dependency].get("status", "PENDING"))
        if status in FAIL_STATUSES:
            failures.add(dependency)
        stack.extend(map(str, stages[dependency].get("depends_on", ())))
    return tuple(sorted(failures))


def ready_stage_ids(plan: dict, ledger: dict) -> tuple[str, ...]:
    stages = _stage_map(plan)
    rows = _ledger_map(ledger)
    ordinal = {stage_id: int(stage["ordinal"]) for stage_id, stage in stages.items()}
    ready: list[str] = []
    for stage_id, stage in stages.items():
        row = rows[stage_id]
        if row.get("status") != "PENDING":
            continue
        dependencies = tuple(map(str, stage.get("depends_on", ())))
        if all(
            dependency_status_admissible(
                ledger, str(rows[dependency].get("status") or "")
            )
            for dependency in dependencies
        ):
            ready.append(stage_id)
    return tuple(sorted(ready, key=lambda stage_id: (ordinal[stage_id], stage_id)))


def _refresh(plan: dict, ledger: dict) -> None:
    rows = ledger["stages"]
    ledger["completed_count"] = sum(
        dependency_status_admissible(ledger, str(row.get("status") or ""))
        for row in rows
    )
    ledger["failed_count"] = sum(row.get("status") in FAIL_STATUSES for row in rows)
    ledger["total_count"] = len(rows)
    ledger["ready_stage_ids"] = list(ready_stage_ids(plan, ledger))
    ledger["failed_stage_ids"] = [
        row["id"] for row in rows if row.get("status") in FAIL_STATUSES
    ]
    ledger["blocked_by_failed_dependency"] = {
        stage["id"]: list(dependency_failure_ids(plan, ledger, stage["id"]))
        for stage in plan["stages"]
        if dependency_failure_ids(plan, ledger, stage["id"])
        and _ledger_map(ledger)[stage["id"]].get("status") == "PENDING"
    }
    if ledger["completed_count"] == len(rows):
        ledger["status"] = (
            "PASS_DEMO_ONLY__ALL_46_STAGES"
            if any(str(row.get("status") or "") == DEMO_ONLY_STATUS for row in rows)
            else "PASS__ALL_46_STAGES"
        )
    elif ledger["failed_count"]:
        ledger["status"] = "ACTIVE_WITH_FAILED_BRANCHES"
    elif ledger["ready_stage_ids"]:
        ledger["status"] = "ACTIVE"
    else:
        ledger["status"] = "WAITING_FOR_EXTERNAL_INPUT_OR_STALE_REPAIR"


def _invalidate_dependents(
    plan: dict, ledger: dict, stage_id: str, reason: str
) -> tuple[str, ...]:
    invalid = {str(stage_id)}
    changed = True
    while changed:
        changed = False
        for stage in plan["stages"]:
            current = str(stage["id"])
            if current in invalid:
                continue
            if any(str(dep) in invalid for dep in stage.get("depends_on", ())):
                invalid.add(current)
                changed = True
    invalidated: list[str] = []
    for row in ledger["stages"]:
        if row["id"] not in invalid:
            continue
        row.update(
            status="PENDING",
            input_fingerprint="",
            implementation_hash="",
            policy_hash="",
            outputs=[],
            diagnostics_hash="",
            blockers=[],
            wall_seconds=0.0,
            performance={},
        )
        invalidated.append(row["id"])
    ledger.setdefault("history", []).append(
        {
            "event": "DEPENDENCY_SUBGRAPH_INVALIDATED",
            "source_stage": str(stage_id),
            "reason": reason,
            "invalidated_stages": invalidated,
        }
    )
    _refresh(plan, ledger)
    return tuple(invalidated)


def _seal_outputs(
    outputs: list[dict],
    *,
    allowed_root: Path,
) -> list[dict]:
    sealed: list[dict] = []
    allowed_root = allowed_root.expanduser().resolve()
    for output in outputs:
        path = Path(str(output["path"])).expanduser().resolve()
        if not _path_within(path, allowed_root):
            raise RuntimeError(
                f"STAGE_OUTPUT_OUTSIDE_STAGE_AUTHORITY_ROOT:{path}:{allowed_root}"
            )
        if not path.is_file():
            raise RuntimeError(f"STAGE_OUTPUT_MISSING:{path}")
        digest = sha256_file(path)
        expected = str(output.get("sha256", "") or "")
        if expected and expected != digest:
            raise RuntimeError(f"STAGE_OUTPUT_SHA_MISMATCH:{path}")
        authority_class = str(output.get("authority_class", "") or "")
        schema = str(output.get("schema", "") or "")
        if not authority_class:
            raise RuntimeError(f"STAGE_OUTPUT_AUTHORITY_CLASS_REQUIRED:{path}")
        if schema in {"", "UNSPECIFIED"}:
            raise RuntimeError(f"STAGE_OUTPUT_SCHEMA_REQUIRED:{path}")
        sealed.append(
            {
                "path": str(path),
                "sha256": digest,
                "bytes": path.stat().st_size,
                "authority_class": authority_class,
                "schema": schema,
            }
        )
    if not sealed:
        raise RuntimeError("STAGE_PASS_REQUIRES_OUTPUT")
    return sealed


def _verify_existing_passes(plan: dict, ledger: dict, manifest: dict) -> bool:
    changed = False
    for stage_id in topological_stage_ids(plan):
        stage = _stage_map(plan)[stage_id]
        row = _ledger_map(ledger)[stage_id]
        if not dependency_status_admissible(ledger, str(row.get("status") or "")):
            continue
        implementation_hash = _adapter_impl_hash(stage["adapter"])
        fingerprint, policy_hash = _fingerprint(
            plan, ledger, manifest, stage, implementation_hash
        )
        if (
            row.get("input_fingerprint") != fingerprint
            or row.get("implementation_hash") != implementation_hash
            or row.get("policy_hash") != policy_hash
            or not _outputs_verify(
                row,
                allowed_root=(
                    authority_root()
                    / "runs"
                    / str(ledger["run_id"])
                    / "artifacts"
                    / stage_id
                ),
            )
        ):
            _invalidate_dependents(
                plan, ledger, stage_id, "STALE_PASS_IDENTITY"
            )
            changed = True
            break
    return changed


def _target_closure(plan: dict, targets: tuple[str, ...]) -> set[str]:
    stages = _stage_map(plan)
    if not targets:
        return set(stages)
    unknown = [target for target in targets if target not in stages]
    if unknown:
        raise RuntimeError("MAINLINE_V2_TARGET_UNKNOWN:" + ",".join(unknown))
    closure: set[str] = set()

    def add(stage_id: str) -> None:
        if stage_id in closure:
            return
        closure.add(stage_id)
        for dependency in stages[stage_id].get("depends_on", ()):
            add(str(dependency))

    for target in targets:
        add(target)
    return closure


def _run_stage(
    *,
    plan: dict,
    ledger: dict,
    manifest: dict,
    manifest_path: Path,
    run_id: str,
    stage_id: str,
    ledger_path: Path,
) -> None:
    stage = _stage_map(plan)[stage_id]
    row = _ledger_map(ledger)[stage_id]
    implementation_hash = _adapter_impl_hash(stage["adapter"])
    fingerprint, policy_hash = _fingerprint(
        plan, ledger, manifest, stage, implementation_hash
    )

    module_name, _, function_name = stage["adapter"].partition(":")
    function = getattr(importlib.import_module(module_name), function_name)
    row.update(
        status="RUNNING",
        attempts=int(row.get("attempts", 0)) + 1,
        input_fingerprint=fingerprint,
        implementation_hash=implementation_hash,
        policy_hash=policy_hash,
        outputs=[],
        diagnostics_hash="",
        blockers=[],
    )
    _refresh(plan, ledger)
    atomic_json(ledger_path, ledger)

    ctx = {
        "repo_root": ROOT,
        "authority_root": authority_root(),
        "run_root": authority_root() / "runs" / run_id,
        "run_id": run_id,
        "run_manifest_path": manifest_path,
        "run_manifest": manifest,
        "stage": stage,
        "ledger": ledger,
    }
    started = perf_counter()
    try:
        result = dict(function(ctx) or {})
        elapsed = perf_counter() - started
        reported_status = str(result.get("status", "FAIL")).upper()
        status = normalize_success_status(
            plan=plan,
            ledger=ledger,
            stage_id=stage_id,
            reported_status=reported_status,
        )
        if status != reported_status and status == DEMO_ONLY_STATUS:
            ledger.setdefault("history", []).append(
                {
                    "event": "DEMO_ONLY_TAINT_PROPAGATED",
                    "stage_id": stage_id,
                    "reported_status": reported_status,
                    "persisted_status": status,
                    "demo_only_dependencies": [
                        str(dep)
                        for dep in stage.get("depends_on", ())
                        if str(_ledger_map(ledger)[str(dep)].get("status") or "")
                        == DEMO_ONLY_STATUS
                    ],
                }
            )
        if status not in PASS_STATUSES:
            row.update(
                status=status if status in FAIL_STATUSES else "FAIL",
                diagnostics_hash=content_sha256(result.get("diagnostics", {})),
                blockers=list(
                    result.get("blockers")
                    or ["STAGE_ADAPTER_REPORTED_FAILURE"]
                ),
                wall_seconds=float(elapsed),
                performance=dict(result.get("performance") or {}),
            )
        else:
            row.update(
                status=status,
                outputs=_seal_outputs(
                    list(result.get("outputs") or ()),
                    allowed_root=(
                        authority_root()
                        / "runs"
                        / run_id
                        / "artifacts"
                        / stage_id
                    ),
                ),
                diagnostics_hash=content_sha256(result.get("diagnostics", {})),
                blockers=[],
                wall_seconds=float(elapsed),
                performance=dict(result.get("performance") or {}),
            )
    except Exception as exc:
        elapsed = perf_counter() - started
        row.update(
            status="FAIL",
            outputs=[],
            diagnostics_hash=content_sha256(
                {"exception_type": type(exc).__name__, "message": str(exc)}
            ),
            blockers=[f"EXCEPTION:{type(exc).__name__}:{exc}"],
            wall_seconds=float(elapsed),
            performance={},
        )
        ledger.setdefault("history", []).append(
            {
                "event": "STAGE_EXCEPTION",
                "stage_id": stage_id,
                "exception_type": type(exc).__name__,
                "message": str(exc),
            }
        )
    _refresh(plan, ledger)
    atomic_json(ledger_path, ledger)


def _validate_run_manifest_identity(
    manifest: dict,
    *,
    run_id: str,
    subject_id: str,
) -> None:
    if str(manifest.get("run_id") or "") != str(run_id):
        raise RuntimeError("RUN_MANIFEST_RUN_ID_DRIFT")
    if str(manifest.get("subject_id") or "") != str(subject_id):
        raise RuntimeError("RUN_MANIFEST_SUBJECT_ID_DRIFT")


def execute(
    run_id: str,
    *,
    targets: tuple[str, ...] = (),
    resume: bool = True,
) -> int:
    plan = load_json(PLAN_PATH)
    validate_plan(plan)
    ledger_path = run_ledger_path(run_id)
    if not ledger_path.is_file():
        raise RuntimeError(f"RUN_LEDGER_MISSING:{ledger_path}")
    ledger = load_json(ledger_path)
    validate_ledger(plan, ledger)
    execution_class = str(ledger.get("execution_class") or "WITNESS")
    if execution_class == "WITNESS":
        validate_witness_authorization(plan)
    elif execution_class == "IMPLEMENTATION_AUDIT":
        manifest_preview = load_json(run_manifest_path(run_id))
        if manifest_preview.get("implementation_audit") is not True:
            raise RuntimeError("IMPLEMENTATION_AUDIT_MANIFEST_FLAG_REQUIRED")
        if str(manifest_preview.get("subject_id") or "") != str(ledger.get("subject_id") or ""):
            raise RuntimeError("IMPLEMENTATION_AUDIT_SUBJECT_ID_DRIFT")
    elif execution_class == "DEMO_WITNESS":
        manifest_preview = load_json(run_manifest_path(run_id))
        validate_demo_witness_authorization(
            manifest_preview,
            subject_id=str(ledger.get("subject_id") or ""),
        )
    if ledger["run_id"] != run_id:
        raise RuntimeError(
            f"ACTIVE_RUN_V2_ID_MISMATCH:{ledger['run_id']}!={run_id}"
        )

    manifest_path = run_manifest_path(run_id)
    if not manifest_path.is_file():
        raise RuntimeError(f"RUN_MANIFEST_MISSING:{manifest_path}")
    manifest = load_json(manifest_path)
    _validate_run_manifest_identity(
        manifest,
        run_id=run_id,
        subject_id=str(ledger.get("subject_id") or ""),
    )
    target_set = _target_closure(plan, targets)

    if _verify_existing_passes(plan, ledger, manifest):
        atomic_json(ledger_path, ledger)

    if not resume:
        for stage_id in sorted(
            target_set,
            key=lambda sid: int(_stage_map(plan)[sid]["ordinal"]),
        ):
            if dependency_status_admissible(
                ledger, str(_ledger_map(ledger)[stage_id].get("status") or "")
            ):
                _invalidate_dependents(plan, ledger, stage_id, "FORCED_RERUN")

    while True:
        ready = [
            stage_id
            for stage_id in ready_stage_ids(plan, ledger)
            if stage_id in target_set
        ]
        if not ready:
            break
        for stage_id in ready:
            _run_stage(
                plan=plan,
                ledger=ledger,
                manifest=manifest,
                manifest_path=manifest_path,
                run_id=run_id,
                stage_id=stage_id,
                ledger_path=ledger_path,
            )

    _refresh(plan, ledger)
    atomic_json(ledger_path, ledger)
    validate_ledger(plan, ledger)

    rows = _ledger_map(ledger)
    target_failures = [
        stage_id
        for stage_id in target_set
        if rows[stage_id].get("status") in FAIL_STATUSES
    ]
    unresolved = [
        stage_id
        for stage_id in target_set
        if rows[stage_id].get("status") == "PENDING"
        and dependency_failure_ids(plan, ledger, stage_id)
    ]
    return 2 if target_failures or unresolved else 0


def status_text(plan: dict, ledger: dict) -> str:
    validate_ledger(plan, ledger)
    lines = [
        (
            f"run={ledger['run_id']} status={ledger['status']} "
            f"progress={ledger['completed_count']}/{ledger['total_count']} "
            f"failed={ledger.get('failed_count', 0)}"
        ),
        "ready=" + (
            ",".join(ledger.get("ready_stage_ids") or ())
            if ledger.get("ready_stage_ids")
            else "NONE"
        ),
    ]
    by_id = _ledger_map(ledger)
    for stage in sorted(plan["stages"], key=lambda item: int(item["ordinal"])):
        row = by_id[stage["id"]]
        mark = (
            "x"
            if dependency_status_admissible(ledger, str(row["status"]))
            else "!"
            if row["status"] in FAIL_STATUSES
            else "~"
            if row["status"] == "RUNNING"
            else " "
        )
        failures = dependency_failure_ids(plan, ledger, stage["id"])
        suffix = (
            " blocked_by=" + ",".join(failures)
            if row["status"] == "PENDING" and failures
            else ""
        )
        lines.append(
            f"[{mark}] {stage['ordinal']:02d}/46 {stage['id']} "
            f"[{stage['group']}] — {row['status']}{suffix}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("validate-plan")
    sub.add_parser("validate-readiness")
    sub.add_parser("validate-witness-authorization")
    sub.add_parser("implementation-closure")
    status_parser = sub.add_parser("status")
    status_parser.add_argument(
        "--run-id",
        default="",
        help="Show a run-local ledger. With no run id, show canonical assembly governance ledger.",
    )
    init = sub.add_parser("init-run")
    init.add_argument("--run-id", required=True)
    init.add_argument("--subject-id", required=True)
    init.add_argument(
        "--architecture-scope",
        default="REALSAS_V2_FRESH_WITNESS",
    )
    init.add_argument(
        "--execution-class",
        choices=("WITNESS", "IMPLEMENTATION_AUDIT", "DEMO_WITNESS"),
        default="WITNESS",
    )
    run = sub.add_parser("execute")
    run.add_argument("--run-id", required=True)
    run.add_argument(
        "--target",
        action="append",
        default=[],
        help="Execute this stage and its dependency closure. Repeatable. "
        "With no target, saturate the whole DAG.",
    )
    run.add_argument("--no-resume", action="store_true")

    args = parser.parse_args(argv)
    plan = load_json(PLAN_PATH)

    if args.command == "validate-plan":
        digest = validate_plan(plan)
        print(f"MAINLINE_V2_PLAN_PASS stages=46 plan_sha256={digest}")
        return 0
    if args.command == "validate-readiness":
        digest = validate_readiness(plan)
        print(f"MAINLINE_V2_READINESS_PASS readiness_sha256={digest}")
        return 0
    if args.command == "validate-witness-authorization":
        digest = validate_witness_authorization(plan)
        print(
            "MAINLINE_V2_WITNESS_AUTHORIZATION_PASS "
            f"readiness_sha256={digest}"
        )
        return 0
    if args.command == "implementation-closure":
        manifest = implementation_closure_manifest(plan)
        digest = content_sha256(manifest)
        print(
            json.dumps(
                {
                    "implementation_closure_sha256": digest,
                    "manifest": manifest,
                },
                sort_keys=True,
            )
        )
        return 0

    if args.command == "status":
        ledger = (
            load_json(run_ledger_path(args.run_id))
            if str(args.run_id).strip()
            else load_json(LEDGER_PATH)
        )
        print(status_text(plan, ledger))
        return 0
    if args.command == "init-run":
        if args.execution_class == "WITNESS":
            validate_witness_authorization(plan)
        else:
            validate_plan(plan)
        manifest_path = run_manifest_path(args.run_id)
        if not manifest_path.is_file():
            raise RuntimeError(f"RUN_MANIFEST_MISSING:{manifest_path}")
        manifest_preview = load_json(manifest_path)
        _validate_run_manifest_identity(
            manifest_preview,
            run_id=args.run_id,
            subject_id=args.subject_id,
        )
        if args.execution_class == "DEMO_WITNESS":
            validate_demo_witness_authorization(manifest_preview, subject_id=args.subject_id)
        if args.execution_class == "IMPLEMENTATION_AUDIT":
            if manifest_preview.get("implementation_audit") is not True:
                raise RuntimeError("IMPLEMENTATION_AUDIT_MANIFEST_FLAG_REQUIRED")
            if str(manifest_preview.get("subject_id") or "") != str(args.subject_id):
                raise RuntimeError("IMPLEMENTATION_AUDIT_SUBJECT_ID_DRIFT")
        path = run_ledger_path(args.run_id)
        if path.exists():
            raise RuntimeError(f"RUN_LEDGER_ALREADY_EXISTS:{path}")
        ledger = build_fresh_run_ledger(
            plan,
            run_id=args.run_id,
            subject_id=args.subject_id,
            manifest_ref=str(manifest_path),
            architecture_scope=args.architecture_scope,
            execution_class=args.execution_class,
        )
        atomic_json(path, ledger)
        print(
            f"MAINLINE_V2_RUN_LEDGER_INIT run={args.run_id} "
            f"subject={args.subject_id} path={path}"
        )
        return 0
    return execute(
        args.run_id,
        targets=tuple(map(str, args.target)),
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    raise SystemExit(main())
