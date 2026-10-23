from __future__ import annotations

"""Deterministic source-owned version identities for non-stage product capabilities.

Compiler stage capability versions come from EngineRelease stage bindings. This
module versions only first-class product modules which sit outside the legacy
stage DAG. Implementation identity is an AST-derived local Python import closure
plus the generic worker dispatch file; policy and parameters are explicit
semantic contracts mirrored by the Go capability descriptors.
"""

from pathlib import Path
from typing import Any

from compiler.realsas_compiler_services.orchestrator import mainline

ROOT = Path(__file__).resolve().parents[1]
WORKER_PATH = ROOT / "compiler/realsas_compiler_services/platform_worker/worker.py"

CAPABILITY_SPECS: dict[str, dict[str, Any]] = {
    "product.input": {
        "entry_module": "compiler.realsas_compiler_services.platform_worker.input_executor_v1",
        "policy": {
            "allowed_modes": ["DEVELOPER"],
            "product_pass_authority": False,
        },
        "parameters": {
            "executor_contract": "RealSaS.ProductInputInference.v1",
            "input_artifact_type": "RealSaS.SourceObservation",
            "input_schema_version": "v1",
            "output_artifact_type": "RealSaS.ShapeInput",
            "output_schema_version": "v1",
            "platform_view_count_invariant": False,
            "source_truth_copied_into_output": False,
            "scientific_contract_is_replaceable": True,
        },
    },
    "product.shape": {
        "entry_module": "compiler.realsas_compiler_services.platform_worker.shape_executor_v1",
        "policy": {
            "allowed_modes": ["DEVELOPER"],
            "product_pass_authority": False,
            "dependencies": ["product.input"],
        },
        "parameters": {
            "executor_contract": "RealSaS.ProductShapeInference.v1",
            "input_artifact_type": "RealSaS.ShapeInput",
            "input_schema_version": "v1",
            "output_artifact_type": "RealSaS.ShapeProposal",
            "output_schema_version": "v1",
        },
    },
}


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
    for capability_id, spec in sorted(CAPABILITY_SPECS.items()):
        out[capability_id] = {
            "implementation_sha256": _implementation_sha256(
                capability_id, str(spec["entry_module"])
            ),
            "policy_sha256": mainline.content_sha256(spec["policy"]),
            "parameters_sha256": mainline.content_sha256(spec["parameters"]),
        }
    return out
