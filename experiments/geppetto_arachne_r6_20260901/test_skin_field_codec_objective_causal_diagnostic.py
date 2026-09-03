from __future__ import annotations

import json

import torch

from experiments.geppetto_arachne_r6_20260901.skin_field_codec_v1 import skin_field_codec_loss_v1


def test_active_emphasis_preserves_exact_teacher_truth_stationarity():
    """Permanent regression for the repaired Codec A0 reconstruction objective.

    Mixed active/inactive teacher influences are the adversarial case that
    falsified the historical pairwise active weighting. The repaired objective
    may emphasize such a row, but it must not alter the relative teacher target
    inside that row. Therefore exact teacher probabilities must be stationary
    with respect to the pre-softmax logits.
    """
    teacher = torch.tensor([[[0.6000, 0.3991, 0.0009]]], dtype=torch.float64)
    logits = torch.log(teacher).detach().clone().requires_grad_(True)
    decoded = torch.softmax(logits, dim=-1)
    surface_mask = torch.ones((1, 1), dtype=torch.bool)
    joint_mask = torch.ones((1, 3), dtype=torch.bool)

    losses = skin_field_codec_loss_v1(
        decoded,
        teacher,
        surface_mask,
        joint_mask,
        active_threshold=1e-3,
        active_weight=2.0,
    )
    losses["cross_entropy"].backward()
    grad = logits.grad.detach()

    result = {
        "teacher": teacher[0, 0].tolist(),
        "exact_truth_ce_logit_grad": grad[0, 0].tolist(),
        "exact_truth_ce_logit_grad_abs_max": float(grad.abs().max()),
        "exact_truth_is_stationary": bool(float(grad.abs().max()) <= 1e-10),
    }
    print("SKIN_FIELD_CODEC_OBJECTIVE_STATIONARITY_REGRESSION=" + json.dumps(result, sort_keys=True))

    assert result["exact_truth_is_stationary"], result
    assert result["exact_truth_ce_logit_grad_abs_max"] <= 1e-10
