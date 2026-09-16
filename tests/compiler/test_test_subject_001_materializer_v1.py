from __future__ import annotations

import inspect

import numpy as np
import pytest

from compiler.realsas_compiler_core.playback_runtime_v3 import AppearanceProvenance
from experiments.playback_stack_v1 import materialize_test_subject_001_v1 as subject


def _camera(view_index: int) -> dict:
    return {
        "view_index": view_index,
        "origin": [0.0, 0.0, -10.0],
        "right": [1.0, 0.0, 0.0],
        "screen_up": [0.0, 1.0, 0.0],
        "forward": [0.0, 0.0, 1.0],
        "half_extent": 1.0,
        "resolution": 4,
    }


def test_source_pixel_center_uv_matches_native_bilinear_coordinate_contract():
    xyz = np.asarray(
        [
            [0.5, 0.5, 0.0],
            [3.5, 3.5, 0.0],
            [-1.0, 5.0, 0.0],
        ],
        dtype=np.float64,
    )
    raw, stored = subject.source_pixel_center_uv_v1(xyz, resolution=4)
    assert raw[0] == pytest.approx([0.0, 0.0])
    assert raw[1] == pytest.approx([1.0, 1.0])
    assert raw[2, 0] < 0.0
    assert raw[2, 1] > 1.0
    assert stored[2] == pytest.approx([0.0, 1.0])


def test_same_view_direct_source_authority_is_explicit_and_every_other_face_is_unseen():
    vertices = np.asarray(
        [
            [-0.5, -0.5, 0.0],
            [0.5, -0.5, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 0.2],
        ],
        dtype=np.float64,
    )
    faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    cameras = {f"V{i}": _camera(i) for i in range(8)}
    direct = {f"V{i}": (0,) for i in range(8)}

    uv_by_view, authority_by_view, counts = subject._build_uv_and_face_authorities(
        vertices,
        faces,
        cameras,
        direct,
    )

    assert set(uv_by_view) == set(subject.VIEW_IDS)
    for view_index, view_id in enumerate(subject.VIEW_IDS):
        uv = uv_by_view[view_id]
        assert uv.shape == (len(vertices), 2)
        assert np.isfinite(uv).all()
        assert np.all((uv >= 0.0) & (uv <= 1.0))

        authority = authority_by_view[view_id]
        assert len(authority) == len(faces)
        assert authority[0].provenance == AppearanceProvenance.DIRECT_SOURCE
        assert authority[0].donor_view_index == view_index
        assert authority[0].atlas_id == f"SOURCE_TEXTURE:{view_id}"
        assert authority[1].provenance == AppearanceProvenance.UNSEEN
        assert authority[1].donor_view_index is None
        assert authority[1].atlas_id is None
        assert counts[view_id] == {"direct_source_faces": 1, "unseen_faces": 1}


def test_test_subject_001_fixture_cannot_reintroduce_historical_face_selection_authority():
    source = inspect.getsource(subject)
    assert "build_dense_zero_surface_candidate" not in source
    assert "qualify_dense_zero_surface_mesh" not in source
    assert "selected_face_indices" not in source
    assert subject.APPEARANCE_POLICY == "SAME_VIEW_DIRECT_SOURCE_ONLY__ALL_OTHER_FACES_UNSEEN"


def test_first_smoke_sampling_is_bounded_odd_and_endpoint_exact():
    times = subject._uniform_times(0.8, 9)
    assert len(times) == 9
    assert times[0] == pytest.approx(0.0)
    assert times[-1] == pytest.approx(0.8)
    assert np.diff(times).min() > 0.0
    with pytest.raises(RuntimeError, match="ODD"):
        subject._uniform_times(0.8, 8)
