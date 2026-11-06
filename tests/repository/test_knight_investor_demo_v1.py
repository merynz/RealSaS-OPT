from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXPORTER = ROOT / "tools/investor_demo/export_knight_investor_demo_v1.py"
WEB = ROOT / "tools/investor_demo/web"
DOC = ROOT / "docs/presentation/KNIGHT_INVESTOR_INTERACTIVE_DEMO_V1.md"


def test_investor_demo_shell_is_offline_and_dependency_free():
    for name in ("START_DEMO.html", "styles.css", "app.js"):
        text = (WEB / name).read_text(encoding="utf-8")
        assert "https://" not in text
        assert "http://" not in text
    html = (WEB / "START_DEMO.html").read_text(encoding="utf-8")
    assert 'src="data/demo-data.js"' in html
    assert 'src="app.js"' in html
    assert 'href="styles.css"' in html


def test_investor_exporter_binds_current_single_m_render_line_and_typed_mgw():
    text = EXPORTER.read_text(encoding="utf-8")
    assert "render_knight_m_caa_v1" in text
    assert "RealSaS.QualifiedMeshIR.v1" in text
    assert "RealSaS.QualifiedSkeletonIR.v1" in text
    assert "RealSaS.QualifiedMeshSkinIR.v1" in text
    assert "INVESTOR_DEMO_M_REST_NOT_RENDER_BASELINE" in text
    assert "INVESTOR_DEMO_M_TOPOLOGY_NOT_RENDER_BASELINE" in text
    assert '"clips": []' in text
    # The investor exporter must discover authorities by schema rather than freeze
    # another historical stage numbering scheme into the presentation package.
    assert 'stage_output_payload(' not in text


def test_investor_viewer_contains_real_structure_modes():
    html = (WEB / "START_DEMO.html").read_text(encoding="utf-8")
    js = (WEB / "app.js").read_text(encoding="utf-8")
    for token in ("Rig", "Canonical topology", "Skin weights", "X-Ray"):
        assert token in html
    assert "boneShape" in js
    assert "data.mesh.edges" in js
    assert "data.weights.influences" in js
    assert "gl.drawArrays(gl.TRIANGLES" in js
    assert "joint_world_positions" in js
    assert "artwork_by_view" in js


def test_investor_demo_design_doc_keeps_motion_separate_from_fixed_core():
    text = DOC.read_text(encoding="utf-8")
    assert "M/G/W/CAA" in text
    assert "Motion is deliberately outside this first commit" in text
    assert "21 recorded M poses across all eight directions, 168 fresh CPU frames" in text
