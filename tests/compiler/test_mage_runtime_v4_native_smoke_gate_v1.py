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

    monkeypatch.setattr(admission.product_v3, "build_final_state", fake_build_final_state)
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
