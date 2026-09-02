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


# GeppettoCandidateV2 clamps every continuous-locus log sigma to this range.
# Keeping the bound explicit here lets the Gaussian-mixture NLL be shifted by
# its exact mathematical lower bound. The shift is constant and therefore does
# not change gradients; it only prevents a secondary tiny-sigma hypothesis from
# numerically cancelling shipping-decode errors in the reported total objective.
POSITION_LOG_SIGMA_MIN = -8.0
POSITION_LOG_SIGMA_MAX = 4.0
POSITION_DIMS = 3


@dataclass(frozen=True)
class GeppettoLossWeightsV2:
    # 5.0 is the historical direct shipping-locus weight from Geppetto V1.
    # Heteroscedastic multimodal likelihood is retained as an auxiliary term,
    # rather than being allowed to define the product representative by itself.
    position_primary: float = 5.0
    position_nll: float = 1.0
    mode_rank: float = 1.0
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


def _pairwise_rank_loss(positive_logits: torch.Tensor, negative_logits: torch.Tensor) -> torch.Tensor:
    """Logistic pair ranking: every positive should outrank every negative.

    This is used only where the shipping consumer itself is rank-based (root
    selection or support top-k). It has no family-dependent margin/threshold.
    """
    if positive_logits.numel() == 0:
        return negative_logits.sum() * 0.0
    if negative_logits.numel() == 0:
        return positive_logits.sum() * 0.0
    return F.softplus(negative_logits[:, None] - positive_logits[None, :]).mean()


def _first_hit_stop_loss(stop_logits: torch.Tensor) -> torch.Tensor:
    """Align training with generate(): first sigmoid(logit)>=0.5 is STOP.

    The terminal positive and the worst earlier false-stop candidate have equal
    authority. A long sequence therefore cannot dilute the one terminal event.
    """
    if stop_logits.ndim != 1 or stop_logits.numel() < 1:
        raise ValueError("STOP boundary requires a non-empty 1D prefix")
    terminal = F.softplus(-stop_logits[-1])
    if stop_logits.numel() == 1:
        return terminal
    worst_early = F.softplus(stop_logits[:-1]).max()
    return 0.5 * (terminal + worst_early)


class GeppettoLossV2:
    def __init__(self, weights: GeppettoLossWeightsV2 = GeppettoLossWeightsV2(), support_topk: int = 8):
        self.weights = weights
        self.support_topk = int(support_topk)

    def _match(self, output: GeppettoRawOutputV2, b: int, root: torch.Tensor, target, J: int):
        """Match teacher controls to the actual shipping MAP representatives.

        Secondary locus hypotheses remain available as uncertainty/alternatives,
        but they cannot define training identity while a different MAP locus is
        autoregressively fed back and emitted to the Compiler.
        """
        tp = torch.as_tensor(target.positions_normalized, device=output.positions_normalized.device, dtype=output.positions_normalized.dtype)
        tr = torch.as_tensor(target.root_mask, device=root.device, dtype=root.dtype)
        primary = output.positions_normalized[b, :J]
        locus_cost = torch.cdist(primary, tp, p=1)
        cost = locus_cost + self.weights.match_root_cost * torch.abs(torch.sigmoid(root[:J])[:, None] - tr[None, :])
        q, t = linear_sum_assignment(cost.detach().cpu().numpy())
        return np.asarray(q, np.int64), np.asarray(t, np.int64)

    @staticmethod
    def _mixture_position_nll(output: GeppettoRawOutputV2, b: int, q: torch.Tensor, target_positions: torch.Tensor) -> torch.Tensor:
        means = output.position_modes_normalized[b, q]
        log_sigma = output.position_mode_log_sigma[b, q].clamp(POSITION_LOG_SIGMA_MIN, POSITION_LOG_SIGMA_MAX)
        mix_logp = torch.log_softmax(output.position_mode_logits[b, q], dim=-1)
        residual = means - target_positions[:, None, :]
        component_logp = mix_logp - 0.5 * (torch.exp(-2.0 * log_sigma) * residual.square()).sum(dim=-1) - log_sigma.sum(dim=-1)
        raw_nll = -torch.logsumexp(component_logp, dim=-1)
        # Since log_sigma >= -8 in each of three dimensions and mixture weights
        # sum to one, raw_nll >= 3 * -8 = -24 exactly. Shift by +24.
        shifted = raw_nll - float(POSITION_DIMS * POSITION_LOG_SIGMA_MIN)
        return shifted.mean()

    @staticmethod
    def _mode_rank_loss(output: GeppettoRawOutputV2, b: int, q: torch.Tensor, target_positions: torch.Tensor) -> torch.Tensor:
        modes = output.position_modes_normalized[b, q]
        dist2 = (modes - target_positions[:, None, :]).square().sum(dim=-1)
        teacher_supported_mode = torch.argmin(dist2.detach(), dim=-1)
        return F.cross_entropy(output.position_mode_logits[b, q], teacher_supported_mode)

    def __call__(self, output: GeppettoRawOutputV2, targets: Sequence[GeppettoTeacherTargetV1], surface_positions_normalized: torch.Tensor, valid_mask: torch.Tensor):
        B, K = output.existence_logits.shape
        if len(targets) != B or surface_positions_normalized.shape[:2] != valid_mask.shape or surface_positions_normalized.shape[-1] != 3:
            raise ValueError("Geppetto V2 loss batch mismatch")
        if output.position_modes_normalized.shape[:2] != (B, K) or output.position_modes_normalized.shape[-1] != 3:
            raise ValueError("Geppetto V2 multimodal locus output missing")
        zero = output.existence_logits.sum() * 0.0
        names = (
            "position_primary",
            "position_nll",
            "mode_rank",
            "existence",
            "stop",
            "root",
            "parent",
            "support",
            "support_presence",
            "abstain",
        )
        total = _independent_zero_accumulators(zero, names)
        matched = 0
        for b, target in enumerate(targets):
            J = int(len(target.positions_normalized)) if target.valid else 0
            N = int(valid_mask[b].sum().item())
            if J > K:
                raise ValueError(f"teacher joint count {J} exceeds decoded steps {K}")
            if J > N:
                raise ValueError(f"teacher joint count {J} exceeds surface resource guard {N}")

            # Existence is confidence evidence for decoded controls, not the
            # cardinality authority. STOP remains the sole generation boundary.
            ex_target = torch.zeros(K, device=output.existence_logits.device, dtype=output.existence_logits.dtype)
            ex_target[:J] = 1.0
            total["existence"] += F.binary_cross_entropy_with_logits(output.existence_logits[b], ex_target)
            abst = torch.tensor(0.0 if target.valid and J > 0 else 1.0, device=output.abstain_logits.device, dtype=output.abstain_logits.dtype)
            total["abstain"] += F.binary_cross_entropy_with_logits(output.abstain_logits[b], abst)
            if J == 0:
                continue

            total["stop"] += _first_hit_stop_loss(output.stop_logits[b, :J])

            q_np, t_np = self._match(output, b, output.root_logits[b], target, J)
            q = torch.as_tensor(q_np, device=output.positions_normalized.device)
            t = torch.as_tensor(t_np, device=output.positions_normalized.device)
            matched += J

            tp = torch.as_tensor(target.positions_normalized, device=output.positions_normalized.device, dtype=output.positions_normalized.dtype)[t]
            total["position_primary"] += F.smooth_l1_loss(output.positions_normalized[b, q], tp, reduction="mean")
            total["position_nll"] += self._mixture_position_nll(output, b, q, tp)
            total["mode_rank"] += self._mode_rank_loss(output, b, q, tp)

            tr = torch.as_tensor(target.root_mask, device=output.root_logits.device, dtype=torch.bool)[t]
            matched_root_logits = output.root_logits[b, q]
            root_pos = matched_root_logits[tr]
            root_neg = matched_root_logits[~tr]
            if root_pos.numel() and root_neg.numel():
                total["root"] += _pairwise_rank_loss(root_pos, root_neg)
            elif root_pos.numel():
                total["root"] += F.softplus(-root_pos).mean()
            else:
                total["root"] += F.softplus(root_neg).mean()

            # Parent evidence is consumed by a one-parent/global-tree optimizer,
            # so train one categorical parent decision per non-root child rather
            # than a sparse all-pairs binary classification problem.
            teacher_parent = np.asarray(target.parent_indices, np.int64)
            t_to_q = {int(tt): int(qq) for qq, tt in zip(q_np.tolist(), t_np.tolist())}
            parent_losses = []
            matched_teacher_ids = [int(x) for x in t_np.tolist()]
            for child_t in matched_teacher_ids:
                parent_t = int(teacher_parent[child_t])
                if parent_t < 0:
                    continue
                if parent_t not in t_to_q:
                    raise ValueError("teacher parent missing from Hungarian match")
                candidate_teacher_ids = [pt for pt in matched_teacher_ids if pt != child_t]
                logits = torch.stack([
                    output.parent_logits[b, t_to_q[child_t], t_to_q[pt]]
                    for pt in candidate_teacher_ids
                ])
                target_index = candidate_teacher_ids.index(parent_t)
                parent_losses.append(F.cross_entropy(logits[None], torch.tensor([target_index], device=logits.device)))
            if parent_losses:
                total["parent"] += torch.stack(parent_losses).mean()

            # Support is consumed only through ranking/top-k. Train that actual
            # decision instead of a highly imbalanced sparse BCE over the raster.
            surface = surface_positions_normalized[b, :N]
            support_losses = []
            presence = []
            raw_teacher = torch.as_tensor(target.positions_normalized, device=surface.device, dtype=surface.dtype)
            all_surface_ids = torch.arange(N, device=surface.device)
            for qq, tt in zip(q_np.tolist(), t_np.tolist()):
                d = torch.linalg.norm(surface - raw_teacher[tt][None], dim=-1)
                top = min(self.support_topk, N)
                ids = torch.topk(d, k=top, largest=False).indices
                logits = output.support_logits[b, qq, :N]
                positive = logits[ids]
                if top < N:
                    negative_mask = torch.ones(N, device=surface.device, dtype=torch.bool)
                    negative_mask[ids] = False
                    negative = logits[all_surface_ids[negative_mask]]
                    support_losses.append(_pairwise_rank_loss(positive, negative))
                else:
                    support_losses.append(logits.sum() * 0.0)
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
            w.position_primary * total["position_primary"]
            + w.position_nll * total["position_nll"]
            + w.mode_rank * total["mode_rank"]
            + w.existence * total["existence"]
            + w.stop * total["stop"]
            + w.root * total["root"]
            + w.parent * total["parent"]
            + w.support * total["support"]
            + w.support_presence * total["support_presence"]
            + w.abstain * total["abstain"]
        )
        return {"total": loss, **total, "matched_joint_count": torch.tensor(float(matched), device=loss.device)}
