from __future__ import annotations

"""Deterministic source-owned version identities for non-stage product capabilities.

Compiler stage capability versions come from EngineRelease stage bindings. The
product capability semantic contract is shared with the Go Platform; the
responsibility ledger is a second orthogonal contract for architecture terrain.
Together they define the exact metadata and learned-model requirements sealed
into a release snapshot.
"""

import json
from pathlib import Path
from typing import Any

from compiler.realsas_compiler_services.orchestrator import mainline

ROOT = Path(__file__).resolve().parents[1]
WORKER_PATH = ROOT / "compiler/realsas_compiler_services/platform_worker/worker.py"
CAPABILITY_SEMANTICS_PATH = (
    ROOT / "platform/internal/capability/contracts/product_capabilities_v1.json"
)
RESPONSIBILITY_LEDGER_PATH = (
    ROOT / "platform/internal/capability/contracts/product_responsibilities_v1.json"
)


def _responsibility_ledger() -> dict[str, Any]:
    value = json.loads(RESPONSIBILITY_LEDGER_PATH.read_text(encoding="utf-8"))
    if value.get("schema") != "RealSaS.ProductResponsibilityLedger.v1":
        raise RuntimeError("RESPONSIBILITY_LEDGER_SCHEMA_DRIFT")
    return value


def _responsibility_ids(role_id: str) -> list[str]:
    value = _responsibility_ledger()
    rows = sorted(
        str(row["id"])
        for row in value.get("responsibilities", [])
        if str(row.get("role_id") or "") == role_id
        and bool(row.get("required_when_role_present", False))
    )
    if not rows:
        raise RuntimeError("RESPONSIBILITY_LEDGER_ROLE_EMPTY:" + role_id)
    return rows


def _required_model_families_by_role() -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for row in _responsibility_ledger().get("responsibilities", []):
        if not bool(row.get("required_when_role_present", False)):
            continue
        family = str(row.get("model_family") or "").strip()
        if not family:
            continue
        role_id = str(row.get("role_id") or "").strip()
        if not role_id:
            raise RuntimeError("RESPONSIBILITY_MODEL_ROLE_MISSING:" + family)
        out.setdefault(role_id, set()).add(family)
    return out


def _capability_semantics() -> dict[str, Any]:
    value = json.loads(CAPABILITY_SEMANTICS_PATH.read_text(encoding="utf-8"))
    if value.get("schema") != "RealSaS.ProductCapabilitySemantics.v1":
        raise RuntimeError("PRODUCT_CAPABILITY_SEMANTICS_SCHEMA_DRIFT")
    capabilities = value.get("capabilities")
    if not isinstance(capabilities, dict) or not capabilities:
        raise RuntimeError("PRODUCT_CAPABILITY_SEMANTICS_EMPTY")
    return capabilities


def capability_specs() -> dict[str, dict[str, Any]]:
    capabilities = _capability_semantics()
    out: dict[str, dict[str, Any]] = {}
    for capability_id, raw in sorted(capabilities.items()):
        if not isinstance(raw, dict):
            raise RuntimeError("PRODUCT_CAPABILITY_SEMANTICS_INVALID:" + str(capability_id))
        entry_module = str(raw.get("entry_module") or "")
        policy = raw.get("policy")
        metadata = raw.get("metadata")
        if (
            not capability_id.startswith("product.")
            or not entry_module
            or not isinstance(policy, dict)
            or not isinstance(metadata, dict)
        ):
            raise RuntimeError("PRODUCT_CAPABILITY_SEMANTICS_INVALID:" + capability_id)
        role_id = str(metadata.get("product_role_id") or "")
        parameters = dict(metadata)
        parameters["responsibility_ids"] = _responsibility_ids(role_id)
        out[capability_id] = {
            "entry_module": entry_module,
            "policy": dict(policy),
            "parameters": parameters,
        }
    return out


def product_model_requirements() -> list[dict[str, str]]:
    required_by_role = _required_model_families_by_role()
    capabilities = _capability_semantics()
    active_roles: set[str] = set()
    global_architecture: dict[str, str] = {}
    for capability_id, raw in sorted(capabilities.items()):
        metadata = raw.get("metadata")
        if not isinstance(metadata, dict):
            raise RuntimeError("PRODUCT_CAPABILITY_METADATA_INVALID:" + capability_id)
        role_id = str(metadata.get("product_role_id") or "").strip()
        if not role_id:
            raise RuntimeError("PRODUCT_CAPABILITY_ROLE_MISSING:" + capability_id)
        active_roles.add(role_id)
        declared_rows = metadata.get("required_models") or []
        if not isinstance(declared_rows, list):
            raise RuntimeError("PRODUCT_MODEL_REQUIREMENTS_TYPE:" + capability_id)
        declared: dict[str, str] = {}
        for row in declared_rows:
            if not isinstance(row, dict):
                raise RuntimeError("PRODUCT_MODEL_REQUIREMENT_ENTRY_TYPE:" + capability_id)
            family = str(row.get("family") or "").strip()
            architecture_id = str(row.get("architecture_id") or "").strip()
            if not family or not architecture_id:
                raise RuntimeError("PRODUCT_MODEL_REQUIREMENT_INVALID:" + capability_id)
            if family in declared:
                raise RuntimeError(
                    "PRODUCT_MODEL_REQUIREMENT_DUPLICATE_FAMILY:"
                    + capability_id
                    + ":"
                    + family
                )
            declared[family] = architecture_id
            prior = global_architecture.get(family)
            if prior is not None and prior != architecture_id:
                raise RuntimeError(
                    "PRODUCT_MODEL_ARCHITECTURE_DRIFT:"
                    + family
                    + ":"
                    + prior
                    + "!="
                    + architecture_id
                )
            global_architecture[family] = architecture_id
        expected = required_by_role.get(role_id, set())
        if set(declared) != expected:
            raise RuntimeError(
                "PRODUCT_MODEL_REQUIREMENT_ROLE_DRIFT:"
                + capability_id
                + ":"
                + ",".join(sorted(declared))
                + "!="
                + ",".join(sorted(expected))
            )
    expected_global = (
        set().union(*(required_by_role.get(role, set()) for role in active_roles))
        if active_roles
        else set()
    )
    if set(global_architecture) != expected_global:
        raise RuntimeError("PRODUCT_MODEL_REQUIREMENT_GLOBAL_DRIFT")
    return [
        {"family": family, "architecture_id": global_architecture[family]}
        for family in sorted(global_architecture)
    ]


def _implementation_sha256(capability_id: str, entry_module: str) -> str:
    closure = mainline._local_import_closure(entry_module)
    if not closure or entry_module not in {name for name, _digest in closure}:
        raise RuntimeError("CAPABILITY_IMPLEMENTATION_CLOSURE_INVALID:" + capability_id)
    return mainline.content_sha256(
        {
            "schema": "RealSaS.CapabilityImplementationClosure.v1",
            "capability_id": capability_id,
            "entry_module": entry_module,
            "local_python_import_closure": [
                {"module": name, "sha256": digest} for name, digest in closure
            ],
            "worker_dispatch_sha256": mainline.sha256_file(WORKER_PATH),
        }
    )


def capability_versions() -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for capability_id, spec in sorted(capability_specs().items()):
        out[capability_id] = {
            "implementation_sha256": _implementation_sha256(
                capability_id, str(spec["entry_module"])
            ),
            "policy_sha256": mainline.content_sha256(spec["policy"]),
            "parameters_sha256": mainline.content_sha256(spec["parameters"]),
        }
    return out
