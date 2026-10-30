from pathlib import Path


ADAPTER = Path("tools/truth_foundry/blender_presentation_ray_measure_v1.py")


def test_stage48_v2_adapter_preserves_uv_and_color_attributes() -> None:
    text = ADAPTER.read_text()
    assert "def _copy_uv_layers(" in text
    assert "source_mesh.uv_layers.active" in text
    assert "target_mesh.uv_layers.new" in text
    assert "def _copy_source_color_attributes(" in text
    assert "target_mesh.color_attributes.new" in text
    assert 'domain == "CORNER"' in text
    assert 'domain == "POINT"' in text
    assert 'domain == "FACE"' in text


def test_stage48_v2_adapter_installs_attribute_preserving_soup() -> None:
    text = ADAPTER.read_text()
    assert "def _build_triangle_soup_with_source_attributes(" in text
    assert "evaluated.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph)" in text
    assert "PRESENTATION_RAY_V2_ACTIVE_UV_REBIND_DRIFT" in text
    assert "PRESENTATION_RAY_V2_FACE_MATERIAL_REBIND_DRIFT" in text
    assert "measured._build_triangle_soup = _build_triangle_soup_with_source_attributes" in text
