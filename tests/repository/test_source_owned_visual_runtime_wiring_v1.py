from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _text(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_source_owned_visual_runtime_cannot_silently_fall_back_to_mechanical_render_mesh():
    appearance = _text(
        "compiler/realsas_compiler_services/orchestrator/adapters/appearance_v2.py"
    )
    product_state = _text(
        "compiler/realsas_compiler_services/orchestrator/adapters/product_state_v2.py"
    )
    runtime = _text(
        "compiler/realsas_compiler_services/orchestrator/adapters/runtime_v2.py"
    )
    runtime_ir = _text(
        "compiler/realsas_compiler_core/runtime_authority_v2.py"
    )
    package = _text(
        "compiler/realsas_compiler_core/runtime_package_v2.py"
    )

    # Current appearance/presentation authority explicitly revokes the mechanical
    # mesh as render geometry in source-owned visual mode.
    assert '"mechanical_mesh_render_authority": False' in appearance
    assert '"mechanical_mesh_render_authority": False' in product_state
    assert "STAGE18_SOURCE_OWNED_VISUAL_MESH_SET" in product_state

    # Therefore Stage42 must either consume a typed visual-mesh/binding authority
    # or fail closed before materializing a mechanical render package.
    consumes_visual_mesh = (
        "visual_mesh_set_from_dict" in runtime
        or "VisualMeshSetIR" in runtime
        or "visual_mesh_set_binding_hash" in runtime_ir
    )
    fail_closed = (
        "SOURCE_OWNED_VISUAL_RUNTIME_NOT_IMPLEMENTED" in runtime
        or "RUNTIME_V2_SOURCE_OWNED_VISUAL_BINDING_REQUIRED" in runtime
    )
    assert consumes_visual_mesh or fail_closed, (
        "Stage20-38 revoke mechanical render authority, but Stage42 has neither "
        "a VisualMeshSet consumer nor an explicit fail-closed blocker."
    )

    # A source-owned per-view visual mesh cannot be represented by silently
    # reusing the legacy single mechanical mesh.bin contract.
    if consumes_visual_mesh:
        supports_visual_package = (
            "visual_mesh" in package.lower()
            or "visual_mesh_set_binding_hash" in runtime_ir
        )
        assert supports_visual_package, (
            "Runtime projection claims visual-mesh consumption but the package/IR "
            "has no visual presentation geometry transport."
        )
