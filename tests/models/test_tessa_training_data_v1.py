import numpy as np

from models.tessa.v1 import (
    TESSAConfigV1,
    build_teacher_asset_sequence_v1,
    build_truncated_teacher_windows_v1,
    connected_face_components_v1,
    decode_tessa_asset_sequence_v1,
    deterministic_face_charts_v1,
    quantize_tessa_xyz_v1,
)


def _strip_mesh(face_count: int):
    vertices = []
    for i in range(face_count + 2):
        x = -0.49 + 0.98 * (i / max(1, face_count + 1))
        y = -0.1 if i % 2 == 0 else 0.1
        vertices.append((x, y, 0.0))
    faces = [(i, i + 1, i + 2) for i in range(face_count)]
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def _cfg():
    return TESSAConfigV1(
        d_model=64,
        n_heads=8,
        surface_layers=1,
        decoder_layers=1,
        surface_latent_count=64,
        local_attention_window=128,
        query_chunk_size=32,
        max_faces=8192,
        max_vertices=8192,
    )


def _geometric_face_keys(vertices, faces, cfg):
    q = quantize_tessa_xyz_v1(np.asarray(vertices), cfg.coordinate_bins)
    return sorted(
        tuple(sorted(tuple(map(int, q[int(vi)].tolist())) for vi in face))
        for face in np.asarray(faces, dtype=np.int64).tolist()
    )


def test_connected_face_components_separates_disconnected_islands():
    faces = np.asarray([[0, 1, 2], [1, 2, 3], [4, 5, 6]], dtype=np.int64)
    comps = connected_face_components_v1(faces)
    assert comps == ((0, 1), (2,))


def test_face_charting_remains_diagnostic_only_and_complete():
    _, faces = _strip_mesh(11)
    charts = deterministic_face_charts_v1(faces, max_faces_per_chart=4)
    face_ids = [fi for _, _, ids in charts for fi in ids]
    assert sorted(face_ids) == list(range(11))
    assert len(face_ids) == len(set(face_ids))
    assert max(len(ids) for _, _, ids in charts) <= 4


def test_teacher_asset_sequence_roundtrips_connectivity_without_artificial_chart_split():
    vertices, faces = _strip_mesh(17)
    cfg = _cfg()
    seq = build_teacher_asset_sequence_v1(vertices, faces, cfg=cfg)
    decoded = decode_tessa_asset_sequence_v1(seq.token_ids, cfg=cfg)
    assert seq.face_count == 17
    assert seq.vertex_count == 19
    assert seq.component_count == 1
    assert seq.source_component_face_indices == (tuple(range(17)),)
    assert decoded.component_count == 1
    assert decoded.orientation_qualified is False
    assert _geometric_face_keys(vertices, faces, cfg) == _geometric_face_keys(
        decoded.vertices_normalized, decoded.faces, cfg
    )


def test_disconnected_source_components_never_weld_equal_xyz_across_components():
    vertices = np.asarray(
        [
            [0.0, 0.0, 0.0], [-0.4, -0.2, 0.0], [-0.3, 0.1, 0.0],
            [0.0, 0.0, 0.0], [0.4, -0.2, 0.0], [0.3, 0.1, 0.0],
        ],
        dtype=np.float64,
    )
    faces = np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    cfg = _cfg()
    seq = build_teacher_asset_sequence_v1(vertices, faces, cfg=cfg)
    decoded = decode_tessa_asset_sequence_v1(seq.token_ids, cfg=cfg)
    assert seq.component_count == 2
    assert seq.token_ids.count(cfg.COMPONENT_BEGIN) == 2
    assert decoded.component_count == 2
    assert len(decoded.vertices_normalized) == 6
    coincident = np.where(np.linalg.norm(decoded.vertices_normalized, axis=1) < 1e-3)[0]
    assert len(coincident) == 2
    assert decoded.vertex_component_indices[int(coincident[0])] != decoded.vertex_component_indices[int(coincident[1])]


def test_truncated_windows_score_every_global_input_position_exactly_once():
    vertices, faces = _strip_mesh(41)
    cfg = _cfg()
    seq = build_teacher_asset_sequence_v1(vertices, faces, cfg=cfg)
    windows = build_truncated_teacher_windows_v1(seq, cfg=cfg, target_tokens=37)
    scored = []
    for w in windows:
        assert len(w.input_ids) == len(w.labels)
        assert len(w.input_ids) <= cfg.local_attention_window + 37
        prefix = w.target_input_start - w.sequence_position_offset
        assert all(x == cfg.PAD for x in w.labels[:prefix])
        scored.extend(range(w.target_input_start, w.target_input_end))
    assert scored == list(range(len(seq.token_ids) - 1))


def test_knight_scale_policy_is_asset_level_not_chart_count():
    cfg = TESSAConfigV1()
    assert cfg.scaling_policy().admits(vertex_count=3665, face_count=6952)
    assert cfg.max_faces == 32768
