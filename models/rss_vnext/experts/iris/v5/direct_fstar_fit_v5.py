from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class DirectFStarFitPolicyV5:
    """Single-authority direct regression policy for the bounded V5 F* field."""

    field_clamp: float = 0.25
    huber_beta: float = 0.01

    def validate(self) -> None:
        if not math.isfinite(float(self.field_clamp)) or float(self.field_clamp) <= 0.0:
            raise ValueError("field_clamp must be finite and positive")
        if not math.isfinite(float(self.huber_beta)) or float(self.huber_beta) <= 0.0:
            raise ValueError("huber_beta must be finite and positive")
        if float(self.huber_beta) > 2.0 * float(self.field_clamp):
            raise ValueError("huber_beta is implausibly larger than the bounded field range")


@dataclass(frozen=True)
class DirectFStarCheckpointCandidateV5:
    """Stage13-blind checkpoint evidence.

    sign_disagreement_count is a qualification axis on the frozen held-out F* audit
    sample. Among sign-clean checkpoints, held-out field residual determines the winner.
    If no checkpoint is sign-clean, the least sign-disagreeing checkpoint is retained only
    as a diagnostic candidate; it is not promoted by this selector.
    """

    step: int
    heldout_count: int
    sign_disagreement_count: int
    heldout_mae: float
    heldout_rmse: float
    heldout_p95_abs: float
    near_zero_count: int
    near_zero_mae: float

    def validate(self) -> None:
        if int(self.step) <= 0:
            raise ValueError("checkpoint step must be positive")
        if int(self.heldout_count) <= 0:
            raise ValueError("heldout_count must be positive")
        if not (0 <= int(self.sign_disagreement_count) <= int(self.heldout_count)):
            raise ValueError("invalid sign_disagreement_count")
        if not (0 <= int(self.near_zero_count) <= int(self.heldout_count)):
            raise ValueError("invalid near_zero_count")
        for name in ("heldout_mae", "heldout_rmse", "heldout_p95_abs", "near_zero_mae"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"invalid checkpoint metric: {name}")

    @property
    def sign_clean(self) -> bool:
        self.validate()
        return int(self.sign_disagreement_count) == 0

    @property
    def sign_disagreement_fraction(self) -> float:
        self.validate()
        return float(self.sign_disagreement_count / self.heldout_count)


def direct_fstar_loss_v5(
    predicted_field: torch.Tensor,
    target_fstar: torch.Tensor,
    *,
    policy: DirectFStarFitPolicyV5 = DirectFStarFitPolicyV5(),
) -> dict[str, torch.Tensor]:
    """The complete V5 primary optimizer objective: direct Huber regression to F* only."""

    policy.validate()
    pred = predicted_field.float().reshape(-1)
    target = target_fstar.float().reshape(-1)
    if pred.shape != target.shape or pred.numel() == 0:
        raise ValueError("predicted_field and target_fstar must match and be non-empty")
    if not bool(torch.isfinite(pred).all()) or not bool(torch.isfinite(target).all()):
        raise ValueError("direct F* loss inputs must be finite")
    clamp = float(policy.field_clamp)
    tol = 1e-6
    if bool(torch.any(target < -clamp - tol)) or bool(torch.any(target > clamp + tol)):
        raise ValueError("target_fstar violates the bounded V5 field contract")

    residual = pred - target
    abs_residual = torch.abs(residual)
    total = F.smooth_l1_loss(
        pred,
        target,
        beta=float(policy.huber_beta),
        reduction="mean",
    )
    return {
        "total": total,
        "direct_huber": total,
        "mae": abs_residual.mean(),
        "rmse": torch.sqrt(torch.mean(residual.square())),
        "maximum_abs": abs_residual.max(),
    }


@torch.no_grad()
def direct_fstar_metrics_v5(
    predicted_field: torch.Tensor,
    target_fstar: torch.Tensor,
    *,
    near_zero_band: float,
) -> dict[str, Any]:
    pred = predicted_field.float().reshape(-1)
    target = target_fstar.float().reshape(-1)
    if pred.shape != target.shape or pred.numel() == 0:
        raise ValueError("predicted_field and target_fstar must match and be non-empty")
    if not bool(torch.isfinite(pred).all()) or not bool(torch.isfinite(target).all()):
        raise ValueError("direct F* metrics inputs must be finite")
    band = float(near_zero_band)
    if not math.isfinite(band) or band <= 0.0:
        raise ValueError("near_zero_band must be finite and positive")

    residual = pred - target
    abs_residual = torch.abs(residual)
    # Zero is a measure-zero target event in random held-out samples. Treat an exact
    # target zero as requiring a nonnegative prediction for deterministic accounting.
    target_positive = target >= 0.0
    pred_positive = pred >= 0.0
    sign_disagreement = target_positive != pred_positive
    near = torch.abs(target) <= band
    near_mae = (
        torch.mean(abs_residual[near])
        if bool(torch.any(near))
        else torch.zeros((), dtype=abs_residual.dtype, device=abs_residual.device)
    )
    return {
        "count": int(pred.numel()),
        "mae": float(abs_residual.mean().cpu()),
        "rmse": float(torch.sqrt(torch.mean(residual.square())).cpu()),
        "p95_abs": float(torch.quantile(abs_residual, 0.95).cpu()),
        "maximum_abs": float(abs_residual.max().cpu()),
        "sign_disagreement_count": int(sign_disagreement.sum().cpu()),
        "sign_disagreement_fraction": float(sign_disagreement.float().mean().cpu()),
        "near_zero_band": band,
        "near_zero_count": int(near.sum().cpu()),
        "near_zero_mae": float(near_mae.cpu()),
    }


def checkpoint_selection_key_v5(
    candidate: DirectFStarCheckpointCandidateV5,
) -> tuple[float, ...]:
    """Frozen lexicographic selector; lower is better and Stage13 is absent by design."""

    candidate.validate()
    if candidate.sign_clean:
        return (
            0.0,
            float(candidate.heldout_mae),
            float(candidate.near_zero_mae),
            float(candidate.heldout_p95_abs),
            float(candidate.heldout_rmse),
            float(candidate.step),
        )
    return (
        1.0,
        float(candidate.sign_disagreement_fraction),
        float(candidate.heldout_mae),
        float(candidate.near_zero_mae),
        float(candidate.heldout_p95_abs),
        float(candidate.heldout_rmse),
        float(candidate.step),
    )


def select_direct_fstar_checkpoint_v5(
    candidates: list[DirectFStarCheckpointCandidateV5] | tuple[DirectFStarCheckpointCandidateV5, ...],
) -> DirectFStarCheckpointCandidateV5:
    if not candidates:
        raise ValueError("at least one checkpoint candidate is required")
    steps = [int(row.step) for row in candidates]
    if len(set(steps)) != len(steps):
        raise ValueError("checkpoint steps must be unique")
    for row in candidates:
        row.validate()
    return min(candidates, key=checkpoint_selection_key_v5)
