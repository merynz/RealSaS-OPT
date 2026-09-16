from __future__ import annotations

import inspect

import numpy as np
import pytest

from compiler.realsas_compiler_core.types import QualificationError
from experiments.playback_stack_v1 import build_test_subject_001_same_view_appearance_authority_v1 as r2


def test_direct_source_face_requires_nonempty_first_hit_and_zero_bad_pixels():
    first_hit = np.asarray(
        [
            [0, 0, 1, 1],
            [0, -1, 1, -1],
            [2, 2, -1, -1],
        ],
        dtype=np.int64,
    )
    owner = np.asarray(
        [
            [0, 0, 0, 2],  # face 1 has one rigid-owned first-hit pixel -> reject whole face.
            [0, -1, 0, -1],
            [0, 0, -1, -1],
        ],
        dtype=np.int16,
    )
    alpha = np.ones(first_hit.shape, dtype=np.bool_)
    alpha[2, 1] = False  # face 2 has one source-alpha failure -> reject whole face.

    direct, stats = r2.classify_direct_source_faces_v1(
        first_hit,
        owner,
        alpha,
        face_count=4,
    )

    assert direct == (0,)
    assert stats["visible_face_count"] == 3
    assert stats["direct_source_face_count"] == 1
    assert stats["rejected_visible_face_count"] == 2
    assert stats["first_hit_non_body_pixel_count"] == 1
    assert stats["first_hit_alpha_fail_pixel_count"] == 1
    assert stats["source_alpha_threshold"] == 8


def test_unseen_face_is_not_promoted_without_a_first_hit_pixel():
    first_hit = np.full((2, 2), -1, dtype=np.int64)
    owner = np.zeros((2, 2), dtype=np.int16)
    alpha = np.ones((2, 2), dtype=np.bool_)
    direct, stats = r2.classify_direct_source_faces_v1(first_hit, owner, alpha, face_count=3)
    assert direct == ()
    assert stats["visible_face_count"] == 0
    assert stats["direct_source_face_count"] == 0


def test_classifier_fails_closed_on_shape_and_face_id_drift():
    with pytest.raises(QualificationError, match="SHAPE_MISMATCH"):
        r2.classify_direct_source_faces_v1(
            np.zeros((2, 2), dtype=np.int64),
            np.zeros((2, 3), dtype=np.int16),
            np.ones((2, 2), dtype=np.bool_),
            face_count=1,
        )
    with pytest.raises(QualificationError, match="FACE_ID_OUT_OF_RANGE"):
        r2.classify_direct_source_faces_v1(
            np.asarray([[2]], dtype=np.int64),
            np.asarray([[0]], dtype=np.int16),
            np.asarray([[True]], dtype=np.bool_),
            face_count=2,
        )


def test_clip_mapping_preserves_runtime_screen_centers_and_near_depth_order():
    xyz = np.asarray(
        [
            [0.0, 0.0, 1.0],
            [4.0, 4.0, 2.0],
        ],
        dtype=np.float64,
    )
    clip = r2._projected_xyz_to_clip(xyz, resolution=4)
    assert clip.dtype == np.float32
    assert clip.flags.c_contiguous
    assert clip[0, :2] == pytest.approx([-1.0, 1.0])
    assert clip[1, :2] == pytest.approx([1.0, -1.0])
    assert clip[0, 2] < clip[1, 2]
    assert np.all(clip[:, 3] == 1.0)


def test_r2_v1_has_no_result_tuned_percentage_threshold_or_geometry_selection_authority():
    source = inspect.getsource(r2)
    assert r2.DIRECT_RULE == "FIRST_HIT_PIXELS_NONEMPTY_AND_ALL_OWNER_BODY0_AND_ALL_ALPHA_GE_8"
    assert r2.APPEARANCE_POLICY == "SAME_VIEW_DIRECT_SOURCE_ONLY__ALL_OTHER_FACES_UNSEEN"
    assert "0.95" not in source
    assert "0.98" not in source
    assert "selected_face_indices" not in source
    assert "build_dense_zero_surface_candidate" not in source
