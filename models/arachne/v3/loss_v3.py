from __future__ import annotations

"""Behavior-first A1 objective for the frozen unordered K4 decoder.

No ordered latent regression is used. The loss supervises decoded scalar fields
and the coupled post-normalization skin row directly.
"""

from dataclasses import asdict, dataclass
from hashlib import sha256
import json

import torch
import torch.nn.functional as F


def _hash(payload: object) -> str:
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class ArachneA1LossConfigV3:
    scalar_bce_weight: float = 1.0
    scalar_mse_weight: float = 0.1
    scalar_dice_weight: float = 1.0
    normalized_row_l1_weight: float = 1.0
    dice_epsilon: float = 1e-4
    architecture_id: str = "RealSaS.Arachne.A1.BehaviorFirstLoss.v3"

    def validate(self) -> None:
        if min(
            self.scalar_bce_weight,
            self.scalar_mse_weight,
            self.scalar_dice_weight,
            self.normalized_row_l1_weight,
        ) < 0:
            raise ValueError("loss weights must be nonnegative")
        if self.scalar_bce_weight + self.scalar_mse_weight + self.scalar_dice_weight <= 0:
            raise ValueError("scalar reconstruction loss cannot be disabled")
        if self.normalized_row_l1_weight <= 0:
            raise ValueError("coupled normalized-row supervision is mandatory")
        if self.dice_epsilon <= 0:
            raise ValueError("invalid Dice epsilon")

    @property
    def config_hash(self) -> str:
        self.validate()
        return _hash(asdict(self))


def normalize_joint_fields(raw_prob: torch.Tensor, joint_mask: torch.Tensor | None = None) -> torch.Tensor:
    if raw_prob.ndim != 3:
        raise ValueError("raw_prob must be [B,N,J]")
    p = raw_prob
    if joint_mask is not None:
        if joint_mask.shape != (p.shape[0], p.shape[2]):
            raise ValueError("joint_mask shape drift")
        p = p * joint_mask[:, None, :].to(p.dtype)
    return p / p.sum(dim=-1, keepdim=True).clamp_min(1e-8)


def arachne_a1_behavior_loss_v3(
    logits: torch.Tensor,
    truth: torch.Tensor,
    row_mask: torch.Tensor,
    *,
    joint_mask: torch.Tensor | None = None,
    config: ArachneA1LossConfigV3 = ArachneA1LossConfigV3(),
) -> dict[str, torch.Tensor]:
    config.validate()
    if logits.shape != truth.shape or logits.ndim != 3:
        raise ValueError("A1 logits/truth must match [B,N,J]")
    if row_mask.shape != logits.shape[:2]:
        raise ValueError("A1 row mask shape drift")
    if not bool(row_mask.any()):
        raise ValueError("empty A1 supervision mask")

    with torch.autocast(device_type=logits.device.type, enabled=False):
        z = logits.float()
        t = truth.float()
        rm = row_mask.bool()
        em = rm[..., None].expand_as(z)
        if joint_mask is not None:
            em = em & joint_mask[:, None, :].bool()
        zv = z[em]
        tv = t[em]
        bce = F.binary_cross_entropy_with_logits(zv, tv)
        raw = torch.sigmoid(z)
        mse = F.mse_loss(raw[em], tv)

        # Dice is computed per joint over the supervised surface set and then
        # averaged over valid joints; this preserves the A0 scalar-field intent.
        m = rm[..., None].to(raw.dtype)
        if joint_mask is not None:
            m = m * joint_mask[:, None, :].to(raw.dtype)
        pr = raw * m
        tr = t * m
        numer = 2.0 * (pr * tr).sum(dim=1) + float(config.dice_epsilon)
        denom = pr.square().sum(dim=1) + tr.square().sum(dim=1) + float(config.dice_epsilon)
        dice_per_joint = 1.0 - numer / denom
        if joint_mask is None:
            dice = dice_per_joint.mean()
        else:
            jm = joint_mask.to(dice_per_joint.dtype)
            dice = (dice_per_joint * jm).sum() / jm.sum().clamp_min(1.0)

        pred = normalize_joint_fields(raw, joint_mask)
        row_l1 = (pred - t).abs().sum(dim=-1)
        coupled = row_l1[rm].mean()

        scalar = (
            float(config.scalar_bce_weight) * bce
            + float(config.scalar_mse_weight) * mse
            + float(config.scalar_dice_weight) * dice
        )
        total = scalar + float(config.normalized_row_l1_weight) * coupled

    return {
        "total": total,
        "scalar_total": scalar,
        "bce": bce,
        "mse": mse,
        "dice": dice,
        "normalized_row_l1": coupled,
        "raw_probability_mean": raw[em].mean(),
    }


__all__ = [
    "ArachneA1LossConfigV3",
    "normalize_joint_fields",
    "arachne_a1_behavior_loss_v3",
]
