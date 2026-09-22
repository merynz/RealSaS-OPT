import math

import pytest
import torch

from models.iris.v3.dense_source_coverage_v3 import (
    DenseSourceCoveragePolicyV3,
    component_balanced_source_coverage_loss,
    deterministic_phase_fraction,
    historical_sparse_objective_v3,
    phase_jittered_unit_cube_ray_points,
    query_ray_foreground_probability_v3,
    ray_foreground_probability_from_sdf_samples,
)
from models.iris.v3.scene_first_signed_v3 import (
    SceneFirstSignedGeometryConfigV3,
    SceneFirstSignedGeometryV3,
)


POLICY = DenseSourceCoveragePolicyV3()


def test_historical_sparse_objective_is_exact_20260919_formula():
    ss = torch.tensor([[0.03, -0.08]], dtype=torch.float32)
    ts = torch.tensor([[0.01, -0.10]], dtype=torch.float32)
    sb = torch.tensor([[0.02, 0.06]], dtype=torch.float32)
    so = torch.tensor([[0.01, 0.08]], dtype=torch.float32)
    sz = torch.tensor([[0.02, -0.01]], dtype=torch.float32)
    got = historical_sparse_objective_v3(ss, ts, sb, so, sz, positive_margin=0.04)
    local = torch.nn.functional.smooth_l1_loss(ss, ts, beta=0.02)
    bg = torch.nn.functional.relu(0.04 - sb).mean()
    outside = torch.nn.functional.relu(0.04 - so).mean()
    zero = torch.nn.functional.smooth_l1_loss(sz, torch.zeros_like(sz), beta=0.01)
    expected = local + 0.25 * bg + 0.25 * outside + 0.50 * zero
    assert torch.allclose(got["total"], expected, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("fixture", ["THIN_BLADE", "THIN_STRAP"])
def test_minimum_width_feature_is_not_missed_by_phase_bank(fixture):
    origin = torch.tensor([[0.0, 0.0, -2.0]], dtype=torch.float64)
    direction = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)
    width = POLICY.minimum_feature_width_normalized
    center_z = 0.173 if fixture == "THIN_BLADE" else -0.287
    for step in range(64):
        phase = deterministic_phase_fraction(26091909, step, (0, 4))
        points = phase_jittered_unit_cube_ray_points(
            origin,
            direction,
            phase_fraction=phase,
            policy=POLICY,
        )
        z = points[0, :, 2]
        signed = torch.abs(z - center_z) - width / 2.0
        assert float(torch.min(signed)) < 0.0, (fixture, step, phase)
        gaps = torch.diff(points[0, :, 2])
        assert float(torch.max(gaps)) <= POLICY.maximum_depth_step_normalized + 1e-9


def test_small_component_keeps_equal_group_bce_authority():
    body_n = 1024
    probability = torch.cat(
        [
            torch.full((body_n,), 0.90),
            torch.tensor([0.10]),
            torch.full((body_n,), 0.10),
        ]
    ).requires_grad_(True)
    target = torch.cat(
        [torch.ones(body_n + 1), torch.zeros(body_n)]
    )
    component_id = torch.cat(
        [
            torch.zeros(body_n, dtype=torch.long),
            torch.ones(1, dtype=torch.long),
            torch.full((body_n,), -1, dtype=torch.long),
        ]
    )
    boundary = torch.zeros_like(probability)
    out = component_balanced_source_coverage_loss(
        probability,
        target,
        component_id,
        boundary,
        policy=POLICY,
    )
    out["component_balanced_bce"].backward()
    small_grad = abs(float(probability.grad[body_n]))
    body_grad = abs(float(probability.grad[0]))
    assert int(out["group_count"]) == 3
    assert small_grad > 100.0 * body_grad


def test_thin_boundary_component_remains_consequential():
    probability = torch.cat(
        [torch.full((1000,), 0.95), torch.full((4,), 0.20)]
    )
    target = torch.ones_like(probability)
    component_id = torch.cat(
        [torch.zeros(1000, dtype=torch.long), torch.ones(4, dtype=torch.long)]
    )
    boundary = torch.cat([torch.zeros(1000), torch.ones(4)])
    out = component_balanced_source_coverage_loss(
        probability,
        target,
        component_id,
        boundary,
        policy=POLICY,
    )
    expected_thin = -math.log(0.20)
    expected_body = -math.log(0.95)
    assert int(out["group_count"]) == 2
    assert abs(float(out["component_balanced_bce"]) - 0.5 * (expected_body + expected_thin)) < 1e-5


@pytest.mark.parametrize("fixture", ["FORK_GAP", "CONCAVE_NOTCH"])
def test_negative_space_background_ray_is_pushed_positive_not_filled(fixture):
    sdf = torch.tensor([[0.20, 0.10, -0.01, 0.15]], dtype=torch.float64, requires_grad=True)
    probability = ray_foreground_probability_from_sdf_samples(sdf, policy=POLICY)
    out = component_balanced_source_coverage_loss(
        probability,
        torch.zeros_like(probability),
        torch.full_like(probability, -1, dtype=torch.long),
        torch.zeros_like(probability),
        policy=POLICY,
    )
    out["total"].backward()
    assert torch.isfinite(sdf.grad).all(), fixture
    assert float(sdf.grad[0, 2]) < 0.0, fixture
    assert float(torch.count_nonzero(sdf.grad)) > 0, fixture


def test_missing_foreground_ray_is_pushed_negative_with_finite_gradient():
    sdf = torch.tensor([[0.04, 0.08, 0.12]], dtype=torch.float64, requires_grad=True)
    probability = ray_foreground_probability_from_sdf_samples(sdf, policy=POLICY)
    out = component_balanced_source_coverage_loss(
        probability,
        torch.ones_like(probability),
        torch.zeros_like(probability, dtype=torch.long),
        torch.ones_like(probability),
        policy=POLICY,
    )
    out["total"].backward()
    assert torch.isfinite(sdf.grad).all()
    assert float(sdf.grad[0, 0]) > 0.0
    assert float(torch.count_nonzero(sdf.grad)) > 0


def test_dense_loss_backpropagates_through_actual_v3_signed_field():
    cfg = SceneFirstSignedGeometryConfigV3(
        image_token_dim=8,
        camera_feature_dim=13,
        scene_dim=24,
        camera_embed_dim=12,
        plane_size=4,
        plane_channels=6,
        field_hidden_dim=16,
        attention_heads=4,
        scene_layers=1,
        feedforward_multiplier=2,
        dropout=0.0,
    )
    torch.manual_seed(91)
    model = SceneFirstSignedGeometryV3(cfg)
    image = torch.randn(1, 2, 5, 8)
    token_xy = torch.randn(1, 2, 5, 2).clamp(-1, 1)
    camera = torch.randn(1, 2, 13)
    planes = model.encode_scene(image, token_xy, camera)

    origins = torch.tensor(
        [[0.0, 0.0, -2.0], [0.2, 0.0, -2.0]], dtype=torch.float32
    )
    directions = torch.tensor(
        [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=torch.float32
    )
    points = phase_jittered_unit_cube_ray_points(
        origins,
        directions,
        phase_fraction=0.375,
        policy=POLICY,
    ).unsqueeze(0)
    probability = query_ray_foreground_probability_v3(
        model,
        planes,
        points,
        policy=POLICY,
        query_chunk=4096,
    )
    loss = component_balanced_source_coverage_loss(
        probability,
        torch.tensor([[1.0, 0.0]]),
        torch.tensor([[0, -1]], dtype=torch.long),
        torch.tensor([[1.0, 0.0]]),
        policy=POLICY,
    )["total"]
    loss.backward()
    grad = model.surface_field.sdf_head.weight.grad
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert float(torch.linalg.vector_norm(grad)) > 0.0
