from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Any

import torch


@dataclass(frozen=True)
class JointBalancedTrainingV4Contract:
    joint_count: int = 22
    optimizer_steps: int = 192
    check_every: int = 4
    checkpoint_every: int = 16
    target_field_reconstructions: int = 4224

    def validate(self) -> None:
        if self.joint_count != 22:
            raise ValueError("V4 joint-balanced contract requires 22 joints")
        if self.optimizer_steps * self.joint_count != self.target_field_reconstructions:
            raise ValueError("V4 joint-balanced field-reconstruction budget drift")
        if min(self.optimizer_steps, self.check_every, self.checkpoint_every) <= 0:
            raise ValueError("V4 joint-balanced positive cadence required")


DEFAULT_CONTRACT = JointBalancedTrainingV4Contract()
DEFAULT_CONTRACT.validate()


def balanced_backward_step(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    make_joint_loss: Callable[[int], tuple[torch.Tensor, Mapping[str, Any]]],
    contract: JointBalancedTrainingV4Contract = DEFAULT_CONTRACT,
) -> list[Mapping[str, Any]]:
    """Accumulate the mean loss from every joint before a single optimizer update.

    `make_joint_loss(j)` must return `(scalar_loss, diagnostics)` for joint `j`.
    Each loss is scaled by `1 / joint_count` before backward, making the update
    equal to the mean per-joint reconstruction gradient while keeping only one
    joint graph alive at a time.
    """
    contract.validate()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    reports: list[Mapping[str, Any]] = []

    for joint_index in range(contract.joint_count):
        loss, report = make_joint_loss(joint_index)
        if loss.ndim != 0 or not torch.isfinite(loss):
            raise RuntimeError(f"V4_JOINT_BALANCED_INVALID_LOSS::{joint_index}")
        (loss / float(contract.joint_count)).backward()
        reports.append(report)

    optimizer.step()
    scheduler.step()
    return reports


__all__ = [
    "JointBalancedTrainingV4Contract",
    "DEFAULT_CONTRACT",
    "balanced_backward_step",
]
