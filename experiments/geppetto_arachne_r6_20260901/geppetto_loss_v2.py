from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import numpy as np
import torch
import torch.nn.functional as F
try:
    from scipy.optimize import linear_sum_assignment
except ImportError as exc:
    raise ImportError("GeppettoLossV2 requires scipy.optimize.linear_sum_assignment") from exc

from .geppetto_candidate_v2 import GeppettoRawOutputV2
from .training_targets_v1 import GeppettoTeacherTargetV1


@dataclass(frozen=True)
class GeppettoLossWeightsV2:
    position_nll: float = 5.0
    existence: float = 1.0
    stop: float = 1.0
    root: float = 1.0
    parent: float = 2.0
    support: float = 0.5
    support_presence: float = 0.25
    abstain: float = 0.5
    match_root_cost: float = 0.25


def _independent_zero_accumulators(zero: torch.Tensor, names: tuple[str, ...]) -> dict[str, torch.Tensor]:
    return {name: zero.clone() for name in names}


class GeppettoLossV2:
    def __init__(self, weights: GeppettoLossWeightsV2 = GeppettoLossWeightsV2(), support_topk: int = 8):
        self.weights = weights
        self.support_topk = int(support_topk)

    def _match(self, output: GeppettoRawOutputV2, b: int, root: torch.Tensor, target, J: int):
        tp = torch.as_tensor(target.positions_normalized, device=output.positions_normalized.device, dtype=output.positions_normalized.dtype)
        tr = torch.as_tensor(target.root_mask, device=root.device, dtype=root.dtype)
        modes = output.position_modes_normalized[b, :J]  # [J,M,3]
        locus_cost = (modes[:, :, None, :] - tp[None, None, :, :]).abs().sum(dim=-1).min(dim=1).values
        cost = locus_cost + self.weights.match_root_cost * torch.abs(torch.sigmoid(root[:J])[:, None] - tr[None, :])
        q, t = linear_sum_assignment(cost.detach().cpu().numpy())
        return np.asarray(q, np.int64), np.asarray(t, np.int64)

    @staticmethod
    def _mixture_position_nll(output: GeppettoRawOutputV2, b: int, q: torch.Tensor, target_positions: torch.Tensor) -> torch.Tensor:
        means = output.position_modes_normalized[b, q]
        log_sigma = output.position_mode_log_sigma[b, q]
        mix_logp = torch.log_softmax(output.position_mode_logits[b, q], dim=-1)
        residual = means - target_positions[:, None, :]
        component_logp = mix_logp - 0.5 * (torch.exp(-2.0 * log_sigma) * residual.square()).sum(dim=-1) - log_sigma.sum(dim=-1)
        return -torch.logsumexp(component_logp, dim=-1).mean()

    def __call__(self, output: GeppettoRawOutputV2, targets: Sequence[GeppettoTeacherTargetV1], surface_positions_normalized: torch.Tensor, valid_mask: torch.Tensor):
        B, K = output.existence_logits.shape
        if len(targets) != B or surface_positions_normalized.shape[:2] != valid_mask.shape or surface_positions_normalized.shape[-1] != 3:
            raise ValueError("Geppetto V2 loss batch mismatch")
        if output.position_modes_normalized.shape[:2] != (B, K) or output.position_modes_normalized.shape[-1] != 3:
            raise ValueError("Geppetto V2 multimodal locus output missing")
        zero = output.existence_logits.sum() * 0.0
        names = ("position_nll", "existence", "stop", "root", "parent", "support", "support_presence", "abstain")
        total = _independent_zero_accumulators(zero, names)
        matched = 0
        for b, target in enumerate(targets):
            J = int(len(target.positions_normalized)) if target.valid else 0
            N = int(valid_mask[b].sum().item())
            if J > K:
                raise ValueError(f"teacher joint count {J} exceeds decoded steps {K}")
            if J > N:
                raise ValueError(f"teacher joint count {J} exceeds surface resource guard {N}")
            ex_target = torch.zeros(K, device=output.existence_logits.device, dtype=output.existence_logits.dtype)
            ex_target[:J] = 1.0
            total["existence"] += F.binary_cross_entropy_with_logits(output.existence_logits[b], ex_target)
            abst = torch.tensor(0.0 if target.valid and J > 0 else 1.0, device=output.abstain_logits.device, dtype=output.abstain_logits.dtype)
            total["abstain"] += F.binary_cross_entropy_with_logits(output.abstain_logits[b], abst)
            if J == 0:
                continue
            stop_target = torch.zeros(J, device=output.stop_logits.device, dtype=output.stop_logits.dtype)
            stop_target[-1] = 1.0
            total["stop"] += F.binary_cross_entropy_with_logits(output.stop_logits[b, :J], stop_target)
            q_np, t_np = self._match(output, b, output.root_logits[b], target, J)
            q = torch.as_tensor(q_np, device=output.positions_normalized.device)
            t = torch.as_tensor(t_np, device=output.positions_normalized.device)
            matched += J
            tp = torch.as_tensor(target.positions_normalized, device=output.positions_normalized.device, dtype=output.positions_normalized.dtype)[t]
            total["position_nll"] += self._mixture_position_nll(output, b, q, tp)
            tr = torch.as_tensor(target.root_mask, device=output.root_logits.device, dtype=output.root_logits.dtype)[t]
            total["root"] += F.binary_cross_entropy_with_logits(output.root_logits[b, q], tr)
            teacher_parent = np.asarray(target.parent_indices, np.int64)
            t_to_q = {int(tt): int(qq) for qq, tt in zip(q_np.tolist(), t_np.tolist())}
            rel_l = []
            rel_y = []
            for child_t in t_np.tolist():
                for parent_t in t_np.tolist():
                    if child_t == parent_t:
                        continue
                    rel_l.append(output.parent_logits[b, t_to_q[child_t], t_to_q[parent_t]])
                    rel_y.append(1.0 if int(teacher_parent[child_t]) == int(parent_t) else 0.0)
            if rel_l:
                logits = torch.stack(rel_l)
                labels = torch.tensor(rel_y, device=logits.device, dtype=logits.dtype)
                total["parent"] += F.binary_cross_entropy_with_logits(logits, labels)
            surface = surface_positions_normalized[b, :N]
            support_losses = []
            presence = []
            raw_teacher = torch.as_tensor(target.positions_normalized, device=surface.device, dtype=surface.dtype)
            for qq, tt in zip(q_np.tolist(), t_np.tolist()):
                d = torch.linalg.norm(surface - raw_teacher[tt][None], dim=-1)
                top = min(self.support_topk, N)
                ids = torch.topk(d, k=top, largest=False).indices
                y = torch.zeros(N, device=surface.device, dtype=output.support_logits.dtype)
                y[ids] = 1.0
                support_losses.append(F.binary_cross_entropy_with_logits(output.support_logits[b, qq, :N], y))
                presence.append(output.support_presence_logits[b, qq])
            total["support"] += torch.stack(support_losses).mean()
            total["support_presence"] += F.binary_cross_entropy_with_logits(
                torch.stack(presence),
                torch.ones(len(presence), device=surface.device, dtype=output.support_presence_logits.dtype),
            )
        for n in total:
            total[n] = total[n] / float(max(B, 1))
        w = self.weights
        loss = (
            w.position_nll * total["position_nll"]
            + w.existence * total["existence"]
            + w.stop * total["stop"]
            + w.root * total["root"]
            + w.parent * total["parent"]
            + w.support * total["support"]
            + w.support_presence * total["support_presence"]
            + w.abstain * total["abstain"]
        )
        return {"total": loss, **total, "matched_joint_count": torch.tensor(float(matched), device=loss.device)}
