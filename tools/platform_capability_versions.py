from __future__ import annotations

"""Deterministic source-owned version identities for non-stage product capabilities.

Compiler stage capability versions come from EngineRelease stage bindings. Product
module policy/metadata are loaded from the same versioned JSON contract embedded
and validated by the Go Platform, eliminating a second semantic authority.
"""

import json
from pathlib import Path
from typing import Any

from compiler.realsas_compiler_services.orchestrator import mainline

ROOT = Path(__file__).resolve().parents[1]
WORKER_PATH = ROOT / "compiler/realsas_compiler_services/platform_worker/worker.py"
CONTRACT_PATH = (
    ROOT
    / "platform/internal/capability/contracts/product_capabilities_v1.json"
)


def capability_specs() -> dict[str, dict[str, Any]]:
    value = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    if value.get("schema") != "RealSaS.ProductCapabilityContracts.v1":
        raise RuntimeError("PRODUCT_CAPABILITY_CONTRACT_SCHEMA_DRIFT")
    capabilities = value.get("capabilities")
    if not isinstance(capabilities, dict) or not capabilities:
        raise RuntimeError("PRODUCT_CAPABILITY_CONTRACT_EMPTY")
    for capability_id, spec in capabilities.items():
        if (
            not isinstance(capability_id, str)
            or not capability_id.startswith("product.")
            or not isinstance(spec, dict)
            or not str(spec.get("entry_module") or "")
            or not isinstance(spec.get("policy"), dict)
            or not isinstance(spec.get("metadata"), dict)
        ):
            raise RuntimeError(
                "PRODUCT_CAPABILITY_CONTRACT_INVALID:" + str(capability_id)
            )
    return capabilities


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
            # Dispatch is generic operational code and intentionally not imported
            # by the capability executor; bind its exact source explicitly.
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
            "parameters_sha256": mainline.content_sha256(spec["metadata"]),
        }
    return out
