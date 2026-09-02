import torch

from experiments.geppetto_arachne_r6_20260901.arachne_tail_remediation_v1 import (
    heteroscedastic_calibration_loss_v1,
    mean_field_remediation_loss_v1,
    row_l1_cvar_v1,
)


def test_row_l1_cvar_targets_tail_not_family_identity():
    teacher = torch.tensor([[[1.,0.],[1.,0.],[1.,0.],[1.,0.]]])
    predicted = torch.tensor([[[1.,0.],[.9,.1],[.6,.4],[.2,.8]]])
    mask = torch.ones((1,4), dtype=torch.bool)
    tail = row_l1_cvar_v1(predicted, teacher, mask, tail_fraction=.25)
    assert torch.allclose(tail, torch.tensor(1.6))


def test_mean_field_remediation_is_finite_and_differentiable():
    mean = torch.zeros((1,2,3), requires_grad=True)
    target_lat = torch.ones_like(mean)
    pred = torch.softmax(torch.randn(1,5,2, requires_grad=True), dim=-1)
    teacher = torch.softmax(torch.randn(1,5,2), dim=-1)
    mask = torch.ones((1,5), dtype=torch.bool)
    out = mean_field_remediation_loss_v1(
        predicted_latent_mean=mean,
        teacher_latents=target_lat,
        predicted_weights=pred,
        teacher_weights=teacher,
        surface_mask=mask,
        reconstruction_loss=(pred-teacher).abs().mean(),
        deformation_mse=(pred-teacher).square().mean(),
    )
    out['total'].backward()
    assert mean.grad is not None


def test_uncertainty_calibration_has_gradient_only_through_sigma_when_mean_detached_by_caller():
    mean = torch.zeros((1,2,3))
    sigma = torch.zeros((1,2,3), requires_grad=True)
    target = torch.ones_like(mean)
    mask = torch.ones((1,2), dtype=torch.bool)
    loss = heteroscedastic_calibration_loss_v1(mean, sigma, target, mask)
    loss.backward()
    assert sigma.grad is not None and torch.isfinite(sigma.grad).all()
