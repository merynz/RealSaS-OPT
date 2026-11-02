from __future__ import annotations

from pathlib import Path


def test_source_reference_v2_preserves_evaluated_uv_and_color_attributes() -> None:
    text = Path("tools/truth_foundry/blender_source_ray_appearance_reference_v2.py").read_text()
    assert "evaluated.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph)" in text
    assert "SOURCE_RAY_REFERENCE_V2_FACE_WORLD_REBIND_DRIFT" in text
    assert "SOURCE_RAY_REFERENCE_V2_FACE_MATERIAL_REBIND_DRIFT" in text
    assert "SOURCE_RAY_REFERENCE_V2_ACTIVE_UV_REBIND_DRIFT" in text
    assert 'domain == "CORNER"' in text
    assert 'domain == "POINT"' in text
    assert 'domain == "FACE"' in text
    assert "mesh.color_attributes.new" in text
    assert "active_color_index" in text
    assert "render_color_index" in text
    assert "sealed_v1_reference_logic_changed" in text
    assert '"tolerance_relaxed": False' in text


def test_stage48_v2_routes_only_source_reference_adapter() -> None:
    text = Path("tools/corpus/run_truth139_stage48_witness_subject_v2.py").read_text()
    assert 'program.name == "blender_source_ray_appearance_reference_v1.py"' in text
    assert "blender_source_ray_appearance_reference_v2.py" in text
    assert "base.run_blender = _route_reference_v2" in text


def test_full139_v4_routes_stage48_v2_without_replacing_v3_logic() -> None:
    text = Path("tools/corpus/run_truth139_v32_finalize_remaining_full139_v4.py").read_text()
    assert "run_truth139_v32_finalize_remaining_full139_v3 as base" in text
    assert 'text.endswith("tools/corpus/run_truth139_stage48_witness_subject_v1.py")' in text
    assert "run_truth139_stage48_witness_subject_v2.py" in text
    assert "base._ORIGINAL_RUN = _route_stage48_v2" in text


def test_release_v5_routes_canary_and_full_run_through_v2_lane() -> None:
    text = Path("tools/notebooks/truth139_v32_finalize_release_v5.py").read_text()
    assert 'text.endswith("tools/corpus/run_truth139_stage48_witness_subject_v1.py")' in text
    assert "run_truth139_stage48_witness_subject_v2.py" in text
    assert "tools.corpus.run_truth139_v32_finalize_remaining_full139_v4" in text
    assert '"release_revision"] = 5' in text
    assert '"--response-svd-rescue"' in text
    assert '"--response-svd-rescue-receipt"' not in text
