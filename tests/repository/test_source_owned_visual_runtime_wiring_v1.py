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
    ) + _text(
        "compiler/realsas_compiler_services/orchestrator/adapters/product_state_legacy_v2.py"
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
    assert "STAGE37_QUALIFIED_SOURCE_OWNED_VISUAL_PRESENTATION" in product_state
    assert "QualifiedVisualPresentationSetIR" in product_state

    # Stage42 must now consume the qualified Stage37 presentation through a
    # typed runtime projection. The old fail-closed token is no longer sufficient:
    # canonical recovery requires real transport through Stage43 and native Stage44.
    assert "SourceOwnedVisualRuntimeProjectionV1IR" in runtime
    assert "source_owned_visual_runtime_projection_from_dict" in runtime
    assert "build_source_owned_visual_rss_v2_entries" in runtime
    assert "SourceOwnedVisualRuntimeProjectionV1IR" in package
    assert "SOURCE_OWNED_VISUAL_PRESENTATION_V1" in package

    native = _text(
        "runtime/realsas_cpp/src/runtime_v2_caa_reference.cpp"
    )
    assert "SOURCE_OWNED_VISUAL_PRESENTATION_V1" in native
    assert "REALSAS_V2_SOURCE_OWNED_VISUAL_2D" in native

    # The product path must not regain the mechanical relation mesh as visual
    # authority while adding the carrier.
    assert '"mechanical_mesh_render_authority": False' in runtime
    assert "mechanical_mesh_render_authority=0" in package

    # Recovery has crossed the former explicit blocker: keeping the old blocker
    # token around would make it ambiguous whether runtime transport is real.
    assert "RUNTIME_V2_SOURCE_OWNED_VISUAL_PRESENTATION_BINDING_REQUIRED" not in runtime



def test_stage37_declares_direct_visual_topology_dependencies():
    import json

    plan = json.loads(
        (ROOT / "canonical/MAINLINE_EXECUTION_PLAN_V2.json").read_text(
            encoding="utf-8"
        )
    )
    stage37 = next(row for row in plan["stages"] if row["ordinal"] == 37)
    required = {
        "05_CAMERA_CONTRACT_SOLVED",
        "07_OBSERVATION_CONTRACT_QUALIFIED",
        "18_CANONICAL_MESH_ADDRESSING_BUILD",
        "35_DYNAMIC_MECHANICAL_MESH_QUALIFIED",
        "36_QUALIFIED_MESH_SKIN_TRANSFER",
    }
    assert required.issubset(set(stage37["depends_on"]))
    assert "observation" in set(stage37["manifest_keys"])
    assert "source-owned visual presentation topology" in stage37["title"]
