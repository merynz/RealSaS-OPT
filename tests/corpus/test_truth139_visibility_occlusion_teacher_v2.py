from __future__ import annotations

import numpy as np

from tools.corpus.build_truth139_visibility_occlusion_teacher_v2 import VISIBILITY, classify_target


def _arrays():
    lookup = np.full((5, 5), -1, dtype=np.int32)
    # The projected centre (2,2) is background; the same surface exists one pixel to the right.
    lookup[2, 3] = 0
    return {
        "row_lookup": lookup,
        "target_xyz_canonical": np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
        "target_depth": np.asarray([4.0], dtype=np.float32),
        "target_triangle": np.asarray([17], dtype=np.int32),
        "target_facing": np.asarray([0.75], dtype=np.float32),
        "target_source_support": np.asarray([True], dtype=np.bool_),
    }


def test_neighbor_same_surface_wins_before_centre_background():
    result = classify_target(
        sample_canonical=np.asarray([0.01, 0.0, 0.0]),
        sample_depth=4.0,
        projected_x=2,
        projected_y=2,
        surface_tolerance=0.05,
        depth_epsilon=0.01,
        search_radius_px=1,
        **_arrays(),
    )
    assert result[0] == VISIBILITY["VISIBLE_SAME_SURFACE"]
    assert result[1] == 2 * 5 + 3
    assert result[2] == 17
    assert result[5] is True
    assert result[6] is True


def test_no_neighbor_support_preserves_background_state():
    result = classify_target(
        sample_canonical=np.asarray([9.0, 0.0, 0.0]),
        sample_depth=4.0,
        projected_x=2,
        projected_y=2,
        surface_tolerance=0.05,
        depth_epsilon=0.01,
        search_radius_px=1,
        **_arrays(),
    )
    assert result[0] == VISIBILITY["PROJECTED_TO_BACKGROUND_OR_UNSUPPORTED"]
    assert result[1] == 2 * 5 + 2
    assert result[2] == -1
    assert result[6] is False
