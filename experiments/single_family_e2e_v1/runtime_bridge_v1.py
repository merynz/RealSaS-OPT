from __future__ import annotations

from compiler.realsas_compiler_core.v4 import project_runtime_package_v3, require_current_proof_bundle
from runtime.reference_v4.consumer import consume_product_v4_reference


def build_and_consume_reference_runtime_v1(*,product,proof_bundle,manifest=None):
    require_current_proof_bundle(product,proof_bundle,require_pass=True)
    payload=dict(manifest or {})
    payload.setdefault('bridge','RealSaS.ReferenceRuntimeBridge.v1')
    runtime=project_runtime_package_v3(product,proof_bundle,manifest=payload,runtime_payload_ref='reference_v4/in_memory')
    report=consume_product_v4_reference(product,proof_bundle,runtime)
    return runtime,report
