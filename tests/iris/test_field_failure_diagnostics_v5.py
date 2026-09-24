from __future__ import annotations

import numpy as np

from models.iris.v5.field_failure_diagnostics_v5 import (
    finite_difference_gradient_numpy_v5,
    replay_staleness_report_v5,
    source_signed_field_per_view_numpy_v5,
    top_two_view_margin_v5,
    zero_set_conditioning_proxy_v5,
)
from models.iris.v5.source_signed_field_v5 import SourceSignedFieldPolicyV5, source_signed_field_numpy_v5


def _camera_arrays():
    origins=np.zeros((2,3),np.float32)
    right=np.asarray([[1,0,0],[1,0,0]],np.float32)
    up=np.asarray([[0,1,0],[0,1,0]],np.float32)
    half=np.ones(2,np.float32)
    return origins,right,up,half


def test_per_view_decomposition_reproduces_canonical_max():
    policy=SourceSignedFieldPolicyV5(required_views=2,required_resolution=4)
    signed=np.zeros((2,4,4),np.float32)
    signed[0]=0.10
    signed[1]=0.20
    origins,right,up,half=_camera_arrays()
    q=np.asarray([[0,0,0],[0.25,0.25,0]],np.float32)
    rep=source_signed_field_per_view_numpy_v5(
        q,signed_2d=signed,center_xyz=np.zeros(3,np.float32),
        normalization_half_extent=1.0,camera_origins=origins,
        camera_right=right,camera_screen_up=up,camera_half_extent=half,
        policy=policy,
    )
    canon=source_signed_field_numpy_v5(
        q,signed_2d=signed,center_xyz=np.zeros(3,np.float32),
        normalization_half_extent=1.0,camera_origins=origins,
        camera_right=right,camera_screen_up=up,camera_half_extent=half,
        policy=policy,
    )
    np.testing.assert_allclose(rep["aggregate_clamped"],canon,rtol=0,atol=2e-7)
    m=top_two_view_margin_v5(rep["per_view_unclamped"],rep["valid"])
    assert np.all(m["top1_view"]==1)
    assert np.all(m["margin"]>0)


def test_finite_difference_gradient_and_conditioning_proxy_linear_field():
    q=np.asarray([[0.1,-0.2,0.3],[0.5,0.0,-0.4]],np.float32)
    def fn(x):
        return (2*x[:,0]-3*x[:,1]+4*x[:,2]+0.25).astype(np.float32)
    g=finite_difference_gradient_numpy_v5(q,fn,epsilon=1e-3)
    expected=np.tile(np.asarray([2,-3,4],np.float32),(len(q),1))
    np.testing.assert_allclose(g,expected,rtol=2e-4,atol=2e-4)
    f=fn(q)
    rep=zero_set_conditioning_proxy_v5(f,g,pixel_scale_normalized_units=0.01)
    expected_offset=np.abs(f)/np.sqrt(29.0)
    np.testing.assert_allclose(rep["offset_proxy_normalized"],expected_offset,rtol=2e-4,atol=2e-4)
    np.testing.assert_allclose(rep["offset_proxy_pixels"],expected_offset/0.01,rtol=2e-4,atol=2e-4)


def test_replay_staleness_detects_shifted_hard_set():
    base=np.asarray([10,9,8,7,1,1,1,1],np.float64)
    current=np.asarray([1,1,1,1,10,9,8,7],np.float64)
    rep=replay_staleness_report_v5(base,current,bank_size=4)
    assert rep["hard_set_overlap_count"]==0
    assert rep["current_hard_set_captured_by_frozen_bank_fraction"]==0.0
    assert rep["current_hard_residual_mean"]>rep["frozen_bank_current_residual_mean"]
