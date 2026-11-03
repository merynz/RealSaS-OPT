"""Projection-only numeric court for zero-area edge-on source primitives.

The old absolute area guard allowed near-collinear hundred-pixel segments to
become source foreground, causing observed 76px/101px false mask gaps. Do not
change the 2px policy threshold or suppress real raster foreground.
"""
import numpy as np

from tools.corpus import iris_observation_admission_v1 as historical
from tools.corpus import iris_observation_admission_v2 as repaired


CAMERA = {
    "origin": [0., 0., 4.], "right": [1., 0., 0.],
    "screen_up": [0., 1., 0.], "half_extent": 1.,
    "resolution": 128,
}


def from_pixels(points):
    # Exact inverse of the observation camera's orthographic XY conversion.
    x = np.asarray([p[0] / 64. - 1. for p in points])
    y = np.asarray([1. - p[1] / 64. for p in points])
    return np.column_stack([x, y, np.zeros(len(points))])


def test_long_numerical_zero_area_triangle_cannot_mint_source_coverage():
    v = from_pixels([(64.5, 12.), (64.5, 100.),
                     (64.5000000002, 90.)])
    faces = np.asarray([[0, 1, 2]], dtype=np.int64)
    old = historical.project_mask(v, faces, CAMERA)
    new = repaired.project_mask(v, faces, CAMERA)
    assert np.count_nonzero(old) >= 70  # old phantom long edge
    assert not new.any()  # source surface still exists; projection unobservable


def test_true_visible_narrow_triangle_is_not_culled():
    v = from_pixels([(64.5, 12.), (64.5, 100.), (65.2, 90.)])
    mask = repaired.project_mask(v, np.asarray([[0, 1, 2]]), CAMERA)
    assert np.count_nonzero(mask) > 30


def test_opposite_winding_does_not_change_source_mask():
    v = from_pixels([(64.5, 12.), (64.5, 100.), (65.2, 90.)])
    a = repaired.project_mask(v, np.asarray([[0, 1, 2]]), CAMERA)
    b = repaired.project_mask(v, np.asarray([[0, 2, 1]]), CAMERA)
    np.testing.assert_array_equal(a, b)


def test_projected_face_cannot_be_replaced_with_inferred_geometry():
    v = from_pixels([(64.5, 12.), (64.5, 100.),
                     (64.5000000002, 90.)])
    near = repaired.project_mask(v, np.asarray([[0, 1, 2]]), CAMERA)
    v[2, 0] = 0.25  # measured real non-collinear point, no guessed edit
    real = repaired.project_mask(v, np.asarray([[0, 1, 2]]), CAMERA)
    assert not near.any()
    assert real.any()


def test_projection_diagnostics_report_abstained_face_not_certain_background():
    v = from_pixels([(64.5, 12.), (64.5, 100.), (64.5000000002, 90.)])
    diag = {}
    mask = repaired.project_mask(v, np.asarray([[0, 1, 2]]),
                                 CAMERA, diagnostics=diag)
    assert mask.sum() == 0
    assert diag["numerical_zero_projected_area_face_count"] == 1
    assert diag["zero_area_policy"].startswith("RELATIVE_PROJECTED_AREA")
