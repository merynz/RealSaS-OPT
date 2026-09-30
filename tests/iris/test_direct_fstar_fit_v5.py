from __future__ import annotations

import torch
import pytest

from models.iris.v5.direct_fstar_fit_v5 import (
    DirectFStarFitPolicyV5,
    DirectFStarCheckpointCandidateV5,
    checkpoint_selection_key_v5,
    direct_fstar_loss_v5,
    direct_fstar_metrics_v5,
    select_direct_fstar_checkpoint_v5,
)


def _candidate(
    step: int,
    *,
    sign: int,
    mae: float,
    near: float,
    p95: float = 0.02,
    rmse: float = 0.01,
) -> DirectFStarCheckpointCandidateV5:
    return DirectFStarCheckpointCandidateV5(
        step=step,
        heldout_count=1000,
        sign_disagreement_count=sign,
        heldout_mae=mae,
        heldout_rmse=rmse,
        heldout_p95_abs=p95,
        near_zero_count=200,
        near_zero_mae=near,
    )


def test_direct_fstar_loss_is_only_direct_target_regression():
    pred = torch.tensor([-0.2, -0.01, 0.03, 0.2], requires_grad=True)
    target = torch.tensor([-0.25, 0.0, 0.05, 0.25])
    out = direct_fstar_loss_v5(
        pred,
        target,
        policy=DirectFStarFitPolicyV5(field_clamp=0.25, huber_beta=0.01),
    )
    expected = torch.nn.functional.smooth_l1_loss(pred.float(), target.float(), beta=0.01)
    torch.testing.assert_close(out["total"], expected)
    torch.testing.assert_close(out["direct_huber"], expected)
    out["total"].backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


def test_direct_fstar_loss_rejects_target_outside_bounded_contract():
    with pytest.raises(ValueError, match="bounded V5 field contract"):
        direct_fstar_loss_v5(
            torch.zeros(2),
            torch.tensor([0.0, 0.3]),
        )


def test_metrics_count_sign_and_near_zero_without_stage13_semantics():
    pred = torch.tensor([-0.2, 0.02, -0.01, 0.2])
    target = torch.tensor([-0.2, -0.01, 0.01, 0.2])
    m = direct_fstar_metrics_v5(pred, target, near_zero_band=0.02)
    assert m["count"] == 4
    assert m["sign_disagreement_count"] == 2
    assert m["near_zero_count"] == 2
    assert m["mae"] >= 0.0
    assert m["near_zero_mae"] >= 0.0


def test_sign_clean_checkpoint_outranks_lower_mae_sign_dirty_candidate():
    clean = _candidate(100, sign=0, mae=0.010, near=0.009)
    dirty = _candidate(200, sign=1, mae=0.001, near=0.001)
    assert checkpoint_selection_key_v5(clean) < checkpoint_selection_key_v5(dirty)
    assert select_direct_fstar_checkpoint_v5([dirty, clean]) == clean


def test_among_sign_clean_checkpoints_heldout_mae_is_primary():
    a = _candidate(100, sign=0, mae=0.008, near=0.001)
    b = _candidate(200, sign=0, mae=0.006, near=0.020)
    assert select_direct_fstar_checkpoint_v5([a, b]) == b


def test_if_none_sign_clean_least_sign_disagreement_is_retained_diagnostically():
    a = _candidate(100, sign=4, mae=0.001, near=0.001)
    b = _candidate(200, sign=2, mae=0.020, near=0.020)
    assert select_direct_fstar_checkpoint_v5([a, b]) == b


def test_duplicate_checkpoint_steps_fail_closed():
    a = _candidate(100, sign=0, mae=0.01, near=0.01)
    b = _candidate(100, sign=0, mae=0.02, near=0.02)
    with pytest.raises(ValueError, match="unique"):
        select_direct_fstar_checkpoint_v5([a, b])
