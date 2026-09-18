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



def test_legacy_p1q_sanitization_receives_mechanical_authority(monkeypatch, tmp_path):
    product_v4 = admission.product_v4
    manifest = {
        "status": "PASS__FIT2_P1Q_CURRENT_AUTHORITY_V0_V7_FROZEN_FACE_POLICY",
        "source_truth_sha256": product_v4.LEGACY_P1Q_SOURCE_TRUTH_SHA256,
        "current_gsa_lineage_hash": "SURFACE",
        "current_skeleton_lineage_hash": "SKELETON",
        "current_skin_lineage_hash": "SKIN",
        "views": [
            {
                "view": view,
                "mesh_lineage_hash": "MESH",
                "files": {
                    "mesh": f"V{view}_mesh.json",
                    "skin": f"V{view}_skin.json",
                    "appearance": f"V{view}_appearance.json",
                },
            }
            for view in range(8)
        ],
    }
    import json
    (tmp_path / "P1Q_FIT2_CURRENT_AUTHORITY_MATERIALIZATION_MANIFEST.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    mesh = SimpleNamespace(mesh_lineage_hash="MESH")
    mesh_skin = SimpleNamespace(
        surface_binding_hash="SURFACE",
        skeleton_binding_hash="SKELETON",
        skin_binding_hash="SKIN",
    )
    appearance = SimpleNamespace(mesh_binding_hash="MESH")
    mechanical = object()
    seen = []

    monkeypatch.setattr(product_v4.v5base, "_load_mesh", lambda *a, **k: mesh)
    monkeypatch.setattr(product_v4.legacy_p1q, "_load_mesh_skin", lambda *a, **k: mesh_skin)
    monkeypatch.setattr(product_v4.legacy_state, "_load_appearance", lambda *a, **k: appearance)

    def fake_sanitize(**kwargs):
        seen.append(kwargs["mechanical"])
        return mesh, mesh_skin, appearance, {"view": kwargs["view"]}

    monkeypatch.setattr(product_v4, "_sanitize_legacy_p1q_carrier", fake_sanitize)

    args = SimpleNamespace(p1q_dir=str(tmp_path), expected_p1q_manifest="")
    surface = SimpleNamespace(geometry_lineage_hash="SURFACE")
    skeleton = SimpleNamespace(skeleton_lineage_hash="SKELETON")
    skin = SimpleNamespace(skin_lineage_hash="SKIN")

    _path, _sha, effective, loaded = product_v4._load_p1q_state(
        args,
        surface,
        skeleton,
        skin,
        mechanical,
    )

    assert seen == [mechanical] * 8
    assert set(loaded) == set(range(8))
    assert effective["legacy_source_truth_witness_sanitized"] is True
    assert effective["legacy_source_truth_witness_runtime_authority"] is False
