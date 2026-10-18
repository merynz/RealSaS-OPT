from __future__ import annotations

"""Post-V8 specialist optimizer and convergence telemetry contracts.

Historical V8 checkpoints remain bound to their original combined
``evidence_heads`` AdamW group.  This module is for future focused post-V8
experiments only.  It keeps the shared scene carrier sealed while giving each
specialist family its own optimizer identity so LR, gradient and update telemetry
cannot be conflated across heads.
"""

from dataclasses import dataclass
import math
from typing import Mapping

import torch

from .model_v1 import IRISVNextV1
from .quality_policy_v1 import DOWNSTREAM_SPECIALIST_STAGES, quality_policy_v1


SPECIALIST_OPTIMIZER_GROUPS_V2 = (
    "scene_core",
    "mechanical_head",
    "appearance_head",
    "relation_head",
    "structure",
)


@dataclass(frozen=True)
class SpecialistOptimizationPolicyV2:
    """Explicit LR policy for one focused experiment.

    No default specialist LRs are supplied deliberately: an experiment must
    preregister them.  The only architectural default is that ``scene_core`` is
    sealed at LR zero.
    """

    mechanical_learning_rate: float
    appearance_learning_rate: float
    relation_learning_rate: float
    structure_learning_rate: float
    scene_learning_rate: float = 0.0
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.95)

    def validate(self) -> None:
        if float(self.scene_learning_rate) != 0.0:
            raise ValueError("POST_V8_SPECIALIST_SCENE_LR_MUST_BE_ZERO")
        specialist_lrs = (
            self.mechanical_learning_rate,
            self.appearance_learning_rate,
            self.relation_learning_rate,
            self.structure_learning_rate,
        )
        if any(not math.isfinite(float(v)) or float(v) <= 0.0 for v in specialist_lrs):
            raise ValueError("POST_V8_SPECIALIST_LR_INVALID")
        if not math.isfinite(float(self.weight_decay)) or float(self.weight_decay) < 0.0:
            raise ValueError("POST_V8_SPECIALIST_WEIGHT_DECAY_INVALID")
        if len(self.betas) != 2 or any(not 0.0 <= float(v) < 1.0 for v in self.betas):
            raise ValueError("POST_V8_SPECIALIST_BETAS_INVALID")


def specialist_parameter_groups_v2(model: IRISVNextV1) -> dict[str, tuple[torch.nn.Parameter, ...]]:
    """Return disjoint, complete post-V8 parameter families."""

    groups = {
        "scene_core": tuple(model.scene_core.parameters()),
        "mechanical_head": tuple(model.mechanical_demand.parameters()),
        "appearance_head": tuple(model.appearance.parameters()),
        "relation_head": tuple(model.relations.parameters()),
        "structure": tuple(model.structure.parameters()),
    }
    if tuple(groups) != SPECIALIST_OPTIMIZER_GROUPS_V2:
        raise RuntimeError("POST_V8_SPECIALIST_GROUP_ORDER_DRIFT")
    if any(not params for params in groups.values()):
        raise RuntimeError("POST_V8_SPECIALIST_EMPTY_PARAMETER_GROUP")

    seen: set[int] = set()
    for name, params in groups.items():
        for parameter in params:
            pid = id(parameter)
            if pid in seen:
                raise RuntimeError(f"POST_V8_SPECIALIST_PARAMETER_ALIAS:{name}")
            seen.add(pid)

    expected = {id(parameter) for parameter in model.parameters()}
    if seen != expected:
        missing = len(expected - seen)
        extra = len(seen - expected)
        raise RuntimeError(
            f"POST_V8_SPECIALIST_PARAMETER_COVERAGE_DRIFT:missing={missing}:extra={extra}"
        )
    return groups


def build_specialist_optimizer_v2(
    model: IRISVNextV1,
    policy: SpecialistOptimizationPolicyV2,
) -> torch.optim.AdamW:
    """Build a head-separated AdamW optimizer with a hard-sealed carrier."""

    policy.validate()
    groups = specialist_parameter_groups_v2(model)
    learning_rates = {
        "scene_core": float(policy.scene_learning_rate),
        "mechanical_head": float(policy.mechanical_learning_rate),
        "appearance_head": float(policy.appearance_learning_rate),
        "relation_head": float(policy.relation_learning_rate),
        "structure": float(policy.structure_learning_rate),
    }
    kwargs = {
        "betas": tuple(map(float, policy.betas)),
        "weight_decay": float(policy.weight_decay),
    }
    if next(model.parameters()).device.type == "cuda":
        kwargs["fused"] = True
    optimizer = torch.optim.AdamW(
        [
            {
                "params": groups[name],
                "lr": learning_rates[name],
                "name": name,
            }
            for name in SPECIALIST_OPTIMIZER_GROUPS_V2
        ],
        **kwargs,
    )

    # Reuse the same fail-closed carrier authority used by the V8 source line.
    seal_policy, _ = quality_policy_v1()
    actual_lrs = optimizer_group_lrs_v2(optimizer)
    for stage_name in DOWNSTREAM_SPECIALIST_STAGES:
        seal_policy.assert_optimizer_configuration(
            stage_name=stage_name,
            group_lrs=actual_lrs,
        )
    return optimizer


def optimizer_group_lrs_v2(optimizer: torch.optim.Optimizer) -> dict[str, float]:
    result: dict[str, float] = {}
    for group in optimizer.param_groups:
        name = str(group.get("name") or "")
        if not name or name in result:
            raise RuntimeError("POST_V8_SPECIALIST_OPTIMIZER_GROUP_NAME_INVALID")
        result[name] = float(group["lr"])
    if tuple(result) != SPECIALIST_OPTIMIZER_GROUPS_V2:
        raise RuntimeError("POST_V8_SPECIALIST_OPTIMIZER_GROUP_SET_DRIFT")
    return result


def capture_group_parameters_v2(
    optimizer: torch.optim.Optimizer,
) -> dict[str, tuple[torch.Tensor, ...]]:
    """Capture detached tensors for update-norm telemetry around one step."""

    snapshots: dict[str, tuple[torch.Tensor, ...]] = {}
    for group in optimizer.param_groups:
        name = str(group.get("name") or "")
        if not name or name in snapshots:
            raise RuntimeError("POST_V8_SPECIALIST_SNAPSHOT_GROUP_INVALID")
        snapshots[name] = tuple(parameter.detach().clone() for parameter in group["params"])
    return snapshots


def group_l2_telemetry_v2(
    optimizer: torch.optim.Optimizer,
    *,
    before: Mapping[str, tuple[torch.Tensor, ...]] | None = None,
) -> dict[str, dict[str, float]]:
    """Return parameter, gradient and optional update L2 norms per optimizer group."""

    result: dict[str, dict[str, float]] = {}
    for group in optimizer.param_groups:
        name = str(group.get("name") or "")
        params = tuple(group["params"])
        if not name or name in result:
            raise RuntimeError("POST_V8_SPECIALIST_TELEMETRY_GROUP_INVALID")

        parameter_sq = 0.0
        gradient_sq = 0.0
        for parameter in params:
            p = parameter.detach().float()
            parameter_sq += float(torch.sum(p * p).item())
            if parameter.grad is not None:
                g = parameter.grad.detach().float()
                gradient_sq += float(torch.sum(g * g).item())

        row = {
            "learning_rate": float(group["lr"]),
            "parameter_l2": math.sqrt(parameter_sq),
            "gradient_l2": math.sqrt(gradient_sq),
        }

        if before is not None:
            if name not in before or len(before[name]) != len(params):
                raise RuntimeError(f"POST_V8_SPECIALIST_TELEMETRY_SNAPSHOT_DRIFT:{name}")
            update_sq = 0.0
            for parameter, old in zip(params, before[name]):
                delta = parameter.detach().float() - old.to(device=parameter.device, dtype=torch.float32)
                update_sq += float(torch.sum(delta * delta).item())
            update_l2 = math.sqrt(update_sq)
            row["update_l2"] = update_l2
            row["update_parameter_ratio"] = update_l2 / max(row["parameter_l2"], 1.0e-30)

        result[name] = row
    return result
