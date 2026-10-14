from tools.corpus.fit8_render_vertex_equivalence_v1 import (
    RenderVertexEvidenceV1,
    qualify_render_vertex_equivalence_v1,
    verify_equivalence_motion_witness_v1,
)


def _vertex(index, x, mechanics, render):
    return RenderVertexEvidenceV1(index, (float(x), 0.0, 0.0), mechanics, render)


def test_render_seam_merges_when_mechanics_match():
    report = qualify_render_vertex_equivalence_v1(
        [_vertex(0, 0, "m", "uv0"), _vertex(1, 0, "m", "uv1")]
    )
    assert report.geometric_vertex_count == 1
    assert report.render_seam_group_count == 1
    assert report.mechanical_conflict_free


def test_coincident_mechanical_conflict_does_not_merge():
    report = qualify_render_vertex_equivalence_v1(
        [_vertex(0, 0, "m0", "a"), _vertex(1, 0, "m1", "a")]
    )
    assert report.geometric_vertex_count == 2
    assert report.mechanically_conflicted_position_count == 1


def test_motion_witness_detects_divergence():
    report = qualify_render_vertex_equivalence_v1(
        [_vertex(0, 0, "m", "a"), _vertex(1, 0, "m", "b")]
    )
    witness = verify_equivalence_motion_witness_v1(
        report,
        {"pose0": [(0, 0, 0), (0.01, 0, 0)]},
        tolerance=1e-5,
    )
    assert not witness["pass"]
    assert len(witness["violations"]) == 1


def test_motion_witness_accepts_exact_coincidence():
    report = qualify_render_vertex_equivalence_v1(
        [_vertex(0, 0, "m", "a"), _vertex(1, 0, "m", "b")]
    )
    witness = verify_equivalence_motion_witness_v1(
        report,
        {"pose0": [(1, 2, 3), (1, 2, 3)]},
        tolerance=0.0,
    )
    assert witness["pass"]
    assert witness["worst_separation"] == 0.0
