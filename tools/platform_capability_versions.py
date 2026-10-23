from __future__ import annotations

"""Deterministic source-owned version identities for non-stage product capabilities.

Compiler stage capability versions come from EngineRelease stage bindings. This
module versions only first-class product modules which sit outside the legacy
stage DAG. Implementation identity is an AST-derived local Python import closure
plus the generic worker dispatch file; policy and parameters mirror the exact
semantic Go capability descriptors and are independently re-derived by Platform
when a release is sealed.
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
            "product_role_id": "input",
            "implementation_class": "HYBRID",
            "interface_contract": "RealSaS.ProductInputInference.v1",
            "executor_contract": "RealSaS.ProductInputInference.v1",
            "input_artifact_type": "RealSaS.SourceObservation",
            "input_schema_version": "v1",
            "output_artifact_type": "RealSaS.ShapeInput",
            "output_schema_version": "v1",
            "platform_view_count_invariant": False,
            "source_truth_copied_into_output": False,
            "scientific_contract_is_replaceable": True,
            "implementation_owner_may_change": True,
            "stable_product_role_survives_rewiring": True,
        },
    },
    "product.shape": {
        "entry_module": "compiler.realsas_compiler_services.platform_worker.shape_executor_v2",
        "policy": {
            "allowed_modes": ["DEVELOPER"],
            "product_pass_authority": False,
            "dependencies": ["product.input"],
        },
        "parameters": {
            "product_role_id": "shape",
            "implementation_class": "MODEL",
            "interface_contract": "RealSaS.ProductShapeInference.v2",
            "executor_contract": "RealSaS.ProductShapeInference.v2",
            "input_artifact_type": "RealSaS.ShapeInput",
            "input_schema_version": "v1",
            "output_artifact_type": "RealSaS.ShapeProposal",
            "output_schema_version": "v2",
            "decoded_topology_is_proposal_evidence": True,
            "canonical_mesh_authority": False,
            "implementation_owner_may_change": True,
            "stable_product_role_survives_rewiring": True,
        },
    },
    "product.mesh": {
        "entry_module": "compiler.realsas_compiler_services.platform_worker.mesh_executor_v1",
        "policy": {
            "allowed_modes": ["DEVELOPER"],
            "product_pass_authority": False,
            "dependencies": ["product.shape"],
        },
        "parameters": {
            "product_role_id": "mesh",
            "implementation_class": "COMPILER",
            "interface_contract": "RealSaS.ProductMeshCompile.v1",
            "executor_contract": "RealSaS.ProductMeshCompile.v1",
            "source_artifact_type": "RealSaS.SourceObservation",
            "source_schema_version": "v1",
            "shape_artifact_type": "RealSaS.ShapeProposal",
            "shape_schema_version": "v2",
            "output_artifact_type": "RealSaS.CanonicalMesh",
            "output_schema_version": "v1",
            "source_fidelity_policy": "Stage13-P999",
            "dynamic_qualification_required_at_birth": False,
            "implementation_owner_may_change": True,
            "stable_product_role_survives_rewiring": True,
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
