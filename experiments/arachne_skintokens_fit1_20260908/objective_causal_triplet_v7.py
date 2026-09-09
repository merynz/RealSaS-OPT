from __future__ import annotations

"""Preregistered Arachne Mage A0 FIT1 objective-causality triplet.

This module freezes the scientific comparison used after the V7 replay autopsy.
It does not implement the 278M model; the production notebook imports/snapshots the
existing V7 codec contract and changes only the training objective/sampling arm.

Arms
----
C0_BIASED_SCALAR
    V7 objective/sampling control under the shared causal compute policy: per-joint
    384-query decoder sampling with 50% active support + 50% global supervised rows;
    independent BCEWithLogits + 0.1*MSE + Dice.

A_UNIFORM_SCALAR
    Same V7 model/optimizer/schedule/compute policy, but no active-heavy importance
    sampler. Each joint is trained on all supervised rows with the same independent
    scalar loss. C0 -> A isolates the effect of the uncorrected sampling prior.

B_COUPLED_ROW_L1
    Same V7 model/optimizer/schedule/compute policy, all supervised rows, but the 22
    joint fields are normalized jointly and optimized directly with mean row L1. A
    low-memory exact two-pass VJP is used: first compute dL/d(logits) for the coupled
    row objective, then rerun each joint field and accumulate J^T v before one AdamW
    step. A -> B isolates cross-joint objective coupling.

Shared compute policy
---------------------
All three arms disable Flash, memory-efficient and cuDNN SDPA backends and force the
PyTorch math SDPA backend. TF32 is disabled. This policy was added after A100 preflight
showed a 0.03525 max-logit mismatch between the no-grad and grad-enabled passes of the
coupled two-pass VJP under the default BF16 attention backend selection. The shared
math-SDPA policy is applied to C0/A/B alike so the causal contrast does not confound
objective choice with attention-kernel choice. The coupled pass must fail closed if
pass-1 and pass-2 logits differ by more than 5e-4.

All arms begin from the exact same serialized initial FP32 model state and reuse the
same frozen nested-prefix schedule. No FSQ is present. FIT1 remains one Mage character.
Acceptance metrics/gates are unchanged and no arm is allowed to redefine closure.
"""

from dataclasses import dataclass
from typing import Literal

import torch

ArmName = Literal[
    "C0_BIASED_SCALAR",
    "A_UNIFORM_SCALAR",
    "B_COUPLED_ROW_L1",
]


@dataclass(frozen=True)
class ObjectiveCausalTripletV7Contract:
    seed: int = 20260908
    optimizer_steps: int = 192
    joint_count: int = 22
    lr: float = 2e-4
    weight_decay: float = 1e-4
    scheduler: str = "CosineAnnealingLR_Tmax_192"
    control_query_count: int = 384
    control_dense_fraction: float = 0.5
    uniform_uses_all_supervised_rows: bool = True
    coupled_uses_all_supervised_rows: bool = True
    nested_prefix_min: int = 1
    nested_prefix_max: int = 4
    fsq_present: bool = False
    field_transport: str = "CONTINUOUS_NO_QUANTIZER"
    architecture_id: str = "RealSaS.Arachne.SkinFieldCodec.v7"
    expected_parameter_count: int = 278_010_880
    attention_backend_policy: str = "MATH_SDPA_ONLY__FLASH_OFF__MEM_EFFICIENT_OFF__CUDNN_SDPA_OFF__TF32_OFF"
    coupled_two_pass_logit_replay_max_abs: float = 5e-4
    arms: tuple[ArmName, ...] = (
        "C0_BIASED_SCALAR",
        "A_UNIFORM_SCALAR",
        "B_COUPLED_ROW_L1",
    )


def coupled_mean_row_l1_from_logits(
    logits_matrix: torch.Tensor,
    teacher_weights: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Authoritative coupled objective over [N,J] logits and simplex teacher rows."""
    if logits_matrix.shape != teacher_weights.shape or logits_matrix.ndim != 2:
        raise ValueError("expected matching [N,J] logits/teacher tensors")
    probs = torch.sigmoid(logits_matrix.float())
    pred = probs / probs.sum(dim=-1, keepdim=True).clamp_min(float(eps))
    return (pred - teacher_weights.float()).abs().sum(dim=-1).mean()


def coupled_logit_gradient(
    logits_matrix: torch.Tensor,
    teacher_weights: torch.Tensor,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return detached scalar loss and exact dL/dlogits for low-memory two-pass VJP."""
    z = logits_matrix.detach().float().requires_grad_(True)
    loss = coupled_mean_row_l1_from_logits(z, teacher_weights, eps=eps)
    (grad,) = torch.autograd.grad(loss, z, create_graph=False, retain_graph=False)
    return loss.detach(), grad.detach()


CONTRACT = ObjectiveCausalTripletV7Contract()

__all__ = [
    "ArmName",
    "ObjectiveCausalTripletV7Contract",
    "CONTRACT",
    "coupled_mean_row_l1_from_logits",
    "coupled_logit_gradient",
]
