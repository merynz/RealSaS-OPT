import torch
import torch.nn.functional as F

from models.iris.v3.dense_source_coverage_v3 import (
    DenseSourceCoveragePolicyV3,
    ray_foreground_logit_from_sdf_samples,
)


def _scaled_bce_gradient(sdf_value: float, target: float) -> float:
    policy = DenseSourceCoveragePolicyV3()
    sdf = torch.tensor([[sdf_value]], dtype=torch.float64, requires_grad=True)
    logit = ray_foreground_logit_from_sdf_samples(sdf, policy=policy)
    loss = policy.top_level_loss_weight * F.binary_cross_entropy_with_logits(
        logit,
        torch.tensor([target], dtype=logit.dtype),
    )
    loss.backward()
    return float(sdf.grad.item())


def test_top_level_dense_weight_is_exactly_frozen_beta():
    policy = DenseSourceCoveragePolicyV3()
    assert abs(policy.top_level_loss_weight - policy.occupancy_beta_normalized) < 1e-15


def test_severe_missing_foreground_scaled_gradient_is_positive_and_o1():
    grad = _scaled_bce_gradient(0.04, 1.0)
    assert 0.99 <= grad <= 1.01


def test_severe_false_foreground_scaled_gradient_is_negative_and_o1():
    grad = _scaled_bce_gradient(-0.04, 0.0)
    assert -1.01 <= grad <= -0.99
