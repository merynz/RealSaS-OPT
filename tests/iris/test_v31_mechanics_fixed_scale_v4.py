from __future__ import annotations

import numpy as np
import torch

from models.iris.v31.mechanics_fixed_scale_student_t_v4 import (
    conditioning_gain_report_v4,
    constant_prior_raw_v4,
    fixed_scale_student_t_mechanical_loss_v4,
    mechanics_regression_metrics_v4,
    robust_fixed_scale_v4,
)


def test_fixed_scale_objective_has_no_log_scale_gradient():
    raw = torch.tensor(
        [[[0.2, 9.0, 0.4, -9.0], [0.6, -3.0, 0.8, 3.0]]],
        dtype=torch.float32,
        requires_grad=True,
    )
    target = torch.tensor([[[0.3, 0.5], [0.7, 0.9]]], dtype=torch.float32)
    valid = torch.ones((1, 2), dtype=torch.bool)
    loss = fixed_scale_student_t_mechanical_loss_v4(
        raw, target, valid, fixed_scale=(0.2, 0.3), degrees_of_freedom=3.0
    )
    loss.backward()
    assert raw.grad is not None
    assert float(raw.grad[..., 0].abs().sum()) > 0.0
    assert float(raw.grad[..., 2].abs().sum()) > 0.0
    assert torch.equal(raw.grad[..., 1], torch.zeros_like(raw.grad[..., 1]))
    assert torch.equal(raw.grad[..., 3], torch.zeros_like(raw.grad[..., 3]))


def test_train_only_mad_scale_respects_floor():
    target = torch.tensor([[[0.1, 0.4], [0.1, 0.8], [0.1, 1.2]]], dtype=torch.float32)
    valid = torch.ones((1, 3), dtype=torch.bool)
    scale = robust_fixed_scale_v4(target, valid, minimum_scale=0.05)
    assert scale[0] == 0.05
    assert scale[1] > 0.05


def test_metrics_and_shortcut_report_reward_scene_conditioned_mean():
    target = torch.tensor([[[0.1, 0.2], [0.3, 0.5], [0.7, 0.9], [1.0, 1.3]]])
    valid = torch.ones((1, 4), dtype=torch.bool)
    model = torch.zeros((1, 4, 4))
    model[..., 0] = target[..., 0] + 0.01
    model[..., 2] = target[..., 1] - 0.01
    prior = constant_prior_raw_v4(target, valid, model.shape)
    shuffled = model[:, torch.tensor([3, 2, 1, 0]), :]

    model_metrics = mechanics_regression_metrics_v4(model, target, valid, tail_quantile=0.75)
    prior_metrics = mechanics_regression_metrics_v4(prior, target, valid, tail_quantile=0.75)
    shuffled_metrics = mechanics_regression_metrics_v4(shuffled, target, valid, tail_quantile=0.75)
    report = conditioning_gain_report_v4(model_metrics, prior_metrics, shuffled_metrics)

    assert model_metrics["mean_log1p_mae"] < prior_metrics["mean_log1p_mae"]
    assert model_metrics["mean_log1p_mae"] < shuffled_metrics["mean_log1p_mae"]
    assert model_metrics["mean_spearman"] > 0.99
    assert report["relative_gain_vs_train_prior_log1p_mae"] > 0.0
    assert report["relative_gain_vs_shuffled_scene_log1p_mae"] > 0.0
    assert report["pass_claimed"] is False


def test_metric_ties_are_finite():
    raw = np.zeros((1, 3, 4), dtype=np.float32)
    target = np.zeros((1, 3, 2), dtype=np.float32)
    valid = np.ones((1, 3), dtype=bool)
    metrics = mechanics_regression_metrics_v4(raw, target, valid)
    assert np.isfinite(metrics["mean_spearman"])
    assert metrics["mean_spearman"] == 0.0
