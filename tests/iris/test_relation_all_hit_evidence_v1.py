from __future__ import annotations

from pathlib import Path

import numpy as np

from tools.iris_vnext.build_fit8_relation_all_hit_evidence_v1 import (
    _face_edge_components,
    _oriented_pair_evidence,
    _ray_triangle_all_parallel_grid,
    _view_stats,
)


def test_all_hit_grid_returns_complete_depth_order_for_two_layers():
    vertices = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    faces = np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    components = _face_edge_components(faces)
    hits = _ray_triangle_all_parallel_grid(
        np.asarray([[0.25, 0.25, -1.0]], dtype=np.float64),
        np.asarray([[0.0, 0.0, 1.0]], dtype=np.float64),
        vertices,
        faces,
        components,
    )
    assert hits["offsets"].tolist() == [0, 2]
    assert hits["face"].tolist() == [0, 1]
    assert np.allclose(hits["depth"], [1.0, 2.0])
    assert np.allclose(hits["point"], [[0.25, 0.25, 0.0], [0.25, 0.25, 1.0]])


def test_three_hit_stack_decouples_pairwise_front_from_global_visibility():
    all_hits = {
        "offsets": np.asarray([0, 3], dtype=np.int64),
        "face": np.asarray([10, 20, 30], dtype=np.int64),
        "depth": np.asarray([1.0, 2.0, 3.0], dtype=np.float64),
        "point": np.asarray(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 2.0]],
            dtype=np.float64,
        ),
    }
    pixels = np.asarray([[7, 9]], dtype=np.int64)
    pairs = _oriented_pair_evidence(
        all_hits=all_hits,
        pixels=pixels,
        center=np.zeros(3, dtype=np.float64),
        half=2.0,
    )
    assert pairs["front_probability"].tolist() == [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    assert pairs["visibility_probability_a"].tolist() == [1.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    assert pairs["visibility_probability_b"].tolist() == [0.0, 1.0, 0.0, 1.0, 0.0, 0.0]
    assert pairs["occlusion_positive_support"].tolist() == [1.0] * 6
    assert pairs["base_pair_id"].tolist() == [0, 0, 1, 1, 2, 2]
    assert pairs["rank_a"].tolist() == [0, 1, 0, 2, 1, 2]
    assert pairs["rank_b"].tolist() == [1, 0, 2, 0, 2, 1]
    assert np.allclose(pairs["point_a"][0], [0.0, 0.0, 0.0])
    assert np.allclose(pairs["point_b"][0], [0.0, 0.0, 0.5])

    stats = _view_stats(all_hits=all_hits, pairs=pairs, ray_count=1)
    assert stats["three_or_more_hit_rays"] == 1
    assert stats["maximum_stack_depth"] == 3
    assert stats["base_pair_count"] == 3
    assert stats["deep_hidden_pair_rows"] == 2
    assert stats["front_visibility_decoupled_rows"] == 2
    assert stats["front_visibility_decoupled_fraction"] == 2.0 / 6.0


def test_all_hit_evidence_is_nontraining_and_exact_coordinate_bound():
    source = Path('tools/iris_vnext/build_fit8_relation_all_hit_evidence_v1.py').read_text(encoding='utf-8')
    for forbidden in (
        'optimizer.step',
        'torch.optim',
        'IRISVNextFocusedRelationTrainer',
        'AUTHORIZE_IRIS_V9_FOCUSED_RELATION_OPTIMIZER_STEPS',
    ):
        assert forbidden not in source
    assert '"training_authorized": False' in source
    assert '--legacy-shard-dir' in source
    assert 'EXACT_LEGACY_IRIS_NORMALIZED_SOURCE_WORLD' in source
    assert 'POSITIVE_SUPPORT_ONLY__SAME_RAY_PAIR__NO_BINARY_NEGATIVE_AUTHORITY' in source
    assert 'NOT_EMITTED__KEEP_SEPARATE_CURRENT_VIEW_RGBA_AUTHORITY' in source
