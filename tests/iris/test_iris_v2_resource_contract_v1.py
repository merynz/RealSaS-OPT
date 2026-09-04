from __future__ import annotations

from models.iris.v2.resource_contract_v2 import estimate_current_iris_materialization_v2


def _estimate(stride: int, depth_count: int = 32):
    q = (1024 // stride) ** 2
    return estimate_current_iris_materialization_v2(
        batch_size=1,
        q_count=q,
        depth_count=depth_count,
        descriptor_dim=384 + (32 + 48 + 64 + 96 + 128),
        hidden_dim=192,
        spatial_neighbors=8,
        view_count=8,
        element_bytes=4,
    )


def test_current_stride8_and_stride16_forward_live_set_exceed_six_gib_before_backward() -> None:
    six_gib = 6 * 2**30
    stride8 = _estimate(8)
    stride16 = _estimate(16)
    assert stride8.q_count == 16384
    assert stride16.q_count == 4096
    assert stride8.forward_live_lower_bound_bytes > six_gib
    assert stride16.forward_live_lower_bound_bytes > six_gib
    assert abs(stride16.forward_live_lower_bound_gib - 6.8125) < 1e-9


def test_stride32_forward_lower_bound_is_below_six_gib_but_is_not_a_training_fit_claim() -> None:
    estimate = _estimate(32)
    assert estimate.q_count == 1024
    assert abs(estimate.forward_live_lower_bound_gib - 1.703125) < 1e-9
    assert estimate.forward_live_lower_bound_bytes < 6 * 2**30
    # The estimator is intentionally a lower bound: passing this comparison does
    # not authorize a training/FIT claim because backward, parameters and feature
    # maps are not included.
    assert estimate.schema == "RealSaS.IRISMaterializationEstimate.v2"


def test_stride8_descriptor_alone_is_multi_gib_at_current_dinov2s_plus_native_width() -> None:
    estimate = _estimate(8)
    assert abs(estimate.descriptor_bytes / 2**30 - 11.75) < 1e-9
