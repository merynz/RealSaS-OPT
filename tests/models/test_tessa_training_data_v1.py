import numpy as np

from models.tessa.v1 import (
    TESSAConfigV1,
    build_teacher_charts_v1,
    connected_face_components_v1,
    deterministic_face_charts_v1,
)


def _strip_mesh(face_count: int):
    # Connected triangle strip in normalized object coordinates.
    vertices = []
    for i in range(face_count + 2):
        x = -0.49 + 0.98 * (i / max(1, face_count + 1))
        y = -0.1 if i % 2 == 0 else 0.1
        vertices.append((x, y, 0.0))
    faces = [(i, i + 1, i + 2) for i in range(face_count)]
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def test_connected_face_components_separates_disconnected_islands():
    faces = np.asarray([[0, 1, 2], [1, 2, 3], [4, 5, 6]], dtype=np.int64)
    comps = connected_face_components_v1(faces)
    assert comps == ((0, 1), (2,))


def test_face_charting_is_bounded_complete_and_nonoverlapping():
    _, faces = _strip_mesh(11)
    charts = deterministic_face_charts_v1(faces, max_faces_per_chart=4)
    face_ids = [fi for _, _, ids in charts for fi in ids]
    assert sorted(face_ids) == list(range(11))
    assert len(face_ids) == len(set(face_ids))
    assert max(len(ids) for _, _, ids in charts) <= 4


def test_teacher_chart_builder_preserves_exact_teacher_face_set():
    vertices, faces = _strip_mesh(17)
    cfg = TESSAConfigV1(
        d_model=64,
        n_heads=8,
        surface_layers=1,
        decoder_layers=1,
        surface_latent_count=64,
        local_attention_window=128,
        query_chunk_size=32,
        max_faces=128,
        max_vertices=128,
        max_faces_per_chart=5,
    )
    charts = build_teacher_charts_v1(vertices, faces, cfg=cfg)
    seen = [fi for chart in charts for fi in chart.source_face_indices]
    assert sorted(seen) == list(range(len(faces)))
    assert all(chart.sequence.face_count <= 5 for chart in charts)
    assert all(chart.sequence.token_ids[-1] == cfg.EOS for chart in charts)


def test_knight_scale_policy_requires_multiple_charts_not_face_truncation():
    cfg = TESSAConfigV1()
    assert cfg.scaling_policy().admits(vertex_count=3665, face_count=6952)
    minimum_chart_count = (6952 + cfg.max_faces_per_chart - 1) // cfg.max_faces_per_chart
    assert minimum_chart_count >= 2
