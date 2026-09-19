from __future__ import annotations

from compiler.realsas_compiler_services.orchestrator.mainline import (
    _adapter_impl_hash,
    _local_import_closure,
)


def test_runtime_native_adapter_hash_covers_local_transitive_implementation():
    module="compiler.realsas_compiler_services.orchestrator.adapters.runtime_native_v1"
    rows=dict(_local_import_closure(module))
    assert module in rows
    assert "compiler.realsas_compiler_core.runtime_package_v1" in rows
    assert "compiler.realsas_compiler_services.cache.reference_render_batch" in rows
    assert "compiler.realsas_compiler_services.cache.reference_render" in rows
    digest=_adapter_impl_hash(module+":visual_motion_render_bake_stage")
    assert len(digest)==64


def test_motion_dynamic_adapter_hash_covers_core_implementation():
    module="compiler.realsas_compiler_services.orchestrator.adapters.motion_dynamic_proof_v1"
    rows=dict(_local_import_closure(module))
    assert "compiler.realsas_compiler_core.motion_dynamic_proof_v1" in rows
    digest=_adapter_impl_hash(module+":prove_dynamic_motion_stage")
    assert len(digest)==64
