import torch

from models.iris.v3.dense_source_coverage_v3 import (
    DenseSourceCoveragePolicyV3,
    component_balanced_source_coverage_loss,
    phase_jittered_depth_lattice,
    ray_foreground_probability_from_sdf_samples,
)


def test_policy_is_derived_from_vf11_minimum_feature_scale():
    p = DenseSourceCoveragePolicyV3()
    assert abs(p.reference_voxel_normalized - 2.0 / 255.0) < 1e-15
    assert abs(p.minimum_feature_width_normalized - 0.75 * 2.0 / 255.0) < 1e-15
    assert abs(
        p.maximum_depth_step_normalized - p.minimum_feature_width_normalized / 2.0
    ) < 1e-15
    assert abs(
        p.occupancy_beta_normalized - p.minimum_feature_width_normalized / 8.0
    ) < 1e-15


def test_global_phase_jitter_preserves_maximum_spacing_and_changes_phase():
    p = DenseSourceCoveragePolicyV3()
    g0 = torch.Generator().manual_seed(1)
    g1 = torch.Generator().manual_seed(2)
    a = phase_jittered_depth_lattice(-1.0, 1.0, policy=p, generator=g0)
    b = phase_jittered_depth_lattice(-1.0, 1.0, policy=p, generator=g1)
    assert len(a) == len(b)
    assert not torch.allclose(a, b)
    spacing = a[1:] - a[:-1]
    assert torch.max(spacing).item() <= p.maximum_depth_step_normalized + 1e-7
    assert a[0].item() - (-1.0) <= p.maximum_depth_step_normalized + 1e-7
    assert 1.0 - a[-1].item() <= p.maximum_depth_step_normalized + 1e-7


def test_hard_min_has_no_multiplicity_bias():
    p = DenseSourceCoveragePolicyV3()
    a = torch.tensor([[0.4, -0.01, 0.2]], requires_grad=True)
    b = torch.tensor([[0.4, -0.01, -0.01, -0.01, 0.2]], requires_grad=True)
    pa = ray_foreground_probability_from_sdf_samples(a, policy=p)
    pb = ray_foreground_probability_from_sdf_samples(b, policy=p)
    assert torch.allclose(pa, pb, atol=0, rtol=0)
    pa.sum().backward()
    pb.sum().backward()
    assert torch.isfinite(a.grad).all()
    assert torch.isfinite(b.grad).all()


def test_beta_marks_center_of_thinnest_fixture_as_strong_foreground():
    p = DenseSourceCoveragePolicyV3()
    center_sdf = -0.5 * p.minimum_feature_width_normalized
    prob = ray_foreground_probability_from_sdf_samples(
        torch.tensor([[center_sdf]]), policy=p
    )
    assert prob.item() > 0.98


def test_component_balancing_prevents_large_component_from_dominating_bce():
    p = DenseSourceCoveragePolicyV3(soft_dice_weight=0.0)
    probability = torch.cat(
        [torch.full((100,), 0.99), torch.tensor([0.01]), torch.full((100,), 0.01)]
    )
    target = torch.cat([torch.ones(101), torch.zeros(100)])
    component = torch.cat(
        [
            torch.zeros(100, dtype=torch.long),
            torch.ones(1, dtype=torch.long),
            -torch.ones(100, dtype=torch.long),
        ]
    )
    boundary = torch.zeros_like(probability)
    out = component_balanced_source_coverage_loss(
        probability, target, component, boundary, policy=p
    )
    assert out["group_count"].item() == 3
    assert out["component_balanced_bce"].item() > 1.0


def test_boundary_multiplier_is_normalized_within_each_group():
    p = DenseSourceCoveragePolicyV3(soft_dice_weight=0.0, boundary_multiplier=4.0)
    probability = torch.tensor([0.9, 0.5, 0.1, 0.1])
    target = torch.tensor([1.0, 1.0, 0.0, 0.0])
    component = torch.tensor([0, 0, -1, -1])
    no_boundary = torch.zeros(4)
    boundary = torch.tensor([0.0, 1.0, 0.0, 0.0])
    a = component_balanced_source_coverage_loss(
        probability, target, component, no_boundary, policy=p
    )
    b = component_balanced_source_coverage_loss(
        probability, target, component, boundary, policy=p
    )
    assert b["component_balanced_bce"].item() > a["component_balanced_bce"].item()
