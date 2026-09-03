from __future__ import annotations

import json

import torch

from experiments.geppetto_arachne_r6_20260901.skin_field_codec_v1 import skin_field_codec_loss_v1


def test_active_weighted_cross_entropy_moves_exact_teacher_truth():
    """Historical positive diagnostic for the pre-repair Codec A0 objective.

    With mixed active/inactive teacher influences, the current CE multiplies
    classes by different constants. Its simplex optimum is therefore
    proportional to multiplier*teacher, not teacher itself. At exact teacher
    probabilities the CE gradient with respect to logits must be non-zero.
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

    multiplier = torch.tensor([3.0, 3.0, 1.0], dtype=torch.float64)
    weighted_target = teacher[0, 0] * multiplier
    weighted_optimum = weighted_target / weighted_target.sum()
    truth_to_weighted_optimum_l1 = float((weighted_optimum - teacher[0, 0]).abs().sum())
    result = {
        "teacher": teacher[0, 0].tolist(),
        "weighted_ce_optimum": weighted_optimum.tolist(),
        "truth_to_weighted_ce_optimum_l1": truth_to_weighted_optimum_l1,
        "exact_truth_ce_logit_grad": grad[0, 0].tolist(),
        "exact_truth_ce_logit_grad_abs_max": float(grad.abs().max()),
        "exact_truth_is_stationary": bool(float(grad.abs().max()) <= 1e-10),
    }
    print("SKIN_FIELD_CODEC_OBJECTIVE_CAUSAL_DIAGNOSTIC=" + json.dumps(result, sort_keys=True))

    # Positive historical diagnostic: pre-repair source is expected to prove the
    # objective is not truth-stationary. This assertion will be inverted into a
    # permanent negative regression when the source repair is committed.
    assert result["exact_truth_ce_logit_grad_abs_max"] > 1e-5
    assert not result["exact_truth_is_stationary"]
