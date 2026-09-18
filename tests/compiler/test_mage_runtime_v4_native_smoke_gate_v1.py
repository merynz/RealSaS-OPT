from __future__ import annotations

from types import SimpleNamespace

import pytest

import experiments.playback_stack_v1.run_mage_full_assembly_runtime_v4_admission_v1 as admission
import experiments.playback_stack_v1.run_mage_full_assembly_runtime_v4_native_smoke_v1 as smoke


def test_full_mage_admission_never_persists_giant_product_graph(monkeypatch, tmp_path):
    seen = {}

    def fake_build_final_state(args, *, persist=True):
        seen["persist"] = persist
        raise RuntimeError("STOP_AFTER_PRODUCT_ENTRY")

    monkeypatch.setattr(admission.product_v4, "build_final_state", fake_build_final_state)
    args = SimpleNamespace(output_dir=str(tmp_path))
    with pytest.raises(RuntimeError, match="STOP_AFTER_PRODUCT_ENTRY"):
        admission.run(args)
    assert seen["persist"] is False


def test_native_smoke_cannot_package_or_render_before_exact_admission_pass(monkeypatch, tmp_path):
    calls = {"package": 0, "stage": 0}

    monkeypatch.setattr(
        smoke.admission,
        "run",
        lambda args: {
            "report": {
                "schema": admission.SCHEMA,
                "status": "FAIL__BLOCKED_BEFORE_RENDER",
            }
        },
    )

    def forbidden_stage(*args, **kwargs):
        calls["stage"] += 1
        raise AssertionError("texture staging must not run")

    def forbidden_package(*args, **kwargs):
        calls["package"] += 1
        raise AssertionError("package materialization must not run")

    monkeypatch.setattr(smoke, "_stage_textures", forbidden_stage)
    monkeypatch.setattr(smoke, "materialize_runtime_v4_archive_cached", forbidden_package)

    args = SimpleNamespace(output_dir=str(tmp_path))
    with pytest.raises(RuntimeError, match="MAGE_V4_NATIVE_ADMISSION_STATUS_NOT_PASS"):
        smoke.run(args)

    assert calls == {"package": 0, "stage": 0}


def test_reference_render_admission_has_no_global_boundary_certificate_dependency():
    # Regression guard: renderer admission is structural/source-authority validation.
    # Dynamic topology/conditioning diagnostics may exist separately but may not become
    # an implicit prerequisite for triangle rasterization.
    assert not hasattr(admission, "certify_directional_runtime_v4_assembly_v1")
    assert admission.SCHEMA.endswith(".v3")
    assert "REFERENCE_INPUT_ADMITTED" in admission.PASS_STATUS

def test_runtime_product_path_is_component_first_and_has_no_legacy_full_subject_loader():
    product_v4 = admission.product_v4
    assert product_v4.SCHEMA.endswith(".v5.component_first")
    assert hasattr(product_v4, "_observation_contexts")
    assert hasattr(product_v4, "materialize_mechanical_component_view")
    assert not hasattr(product_v4, "_load_p1q_state")
    assert not hasattr(product_v4, "_sanitize_legacy_p1q_carrier")
    assert not hasattr(product_v4, "legacy_p1q")
    assert not hasattr(product_v4, "legacy_state")
    assert not hasattr(product_v4, "v5base")
    assert not hasattr(product_v4, "project_mesh_to_mechanical_components")

