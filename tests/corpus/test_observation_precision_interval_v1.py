"""Independent repro of float32 screen-orientation interval abstention.

These tests isolate a source face that differs from exact collinearity only
within input-coordinate precision. Historical absolute/relative gates would
claim a long line of source-positive pixels that Blender does not render.
"""
import numpy as np
from tools.corpus import iris_observation_admission_v2 as earlier
from tools.corpus import iris_observation_admission_v3 as current

CAM = {"origin": [0.,0.,4.], "right": [1.,0.,0.],
       "screen_up": [0.,1.,0.], "half_extent": 1., "resolution": 128}
F = np.asarray([[0,1,2]],np.int64)


def coords(width):
    xy=np.asarray([(64.5,10.),(64.5,110.),(64.5+width,65.)])
    return np.column_stack([xy[:,0]/64.-1.,1.-xy[:,1]/64.,np.zeros(3)])


def test_float32_ambiguous_hundred_pixel_line_is_not_source_raster_positive():
    verts=coords(1e-6)
    assert earlier.project_mask(verts,F,CAM).sum()>=90
    diag={}
    result=current.project_mask(verts,F,CAM,diagnostics=diag)
    assert not result.any()
    assert diag["numerical_zero_projected_area_face_count"]==1
    assert diag["zero_area_policy"].startswith("FLOAT32_EIGHT_ULP")


def test_materially_nonzero_triangle_retains_raster_support():
    verts=coords(.7)
    old=earlier.project_mask(verts,F,CAM)
    new=current.project_mask(verts,F,CAM)
    assert old.any() and new.any()
    np.testing.assert_array_equal(old,new)


def test_zero_triangle_remains_abstention_for_opposite_winding():
    verts=coords(1e-6)
    a=current.project_mask(verts,F,CAM)
    b=current.project_mask(verts,F[:,::-1],CAM)
    assert not a.any() and not b.any()


def test_declared_precision_is_not_source_geometry_deletion():
    verts=coords(1e-6)
    original=verts.copy()
    current.project_mask(verts,F,CAM)
    np.testing.assert_array_equal(verts,original)
