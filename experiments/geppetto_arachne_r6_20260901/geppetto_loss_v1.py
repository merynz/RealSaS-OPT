from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

try:
    from scipy.optimize import linear_sum_assignment
except ImportError as exc:
    raise ImportError("GeppettoLossV1 requires scipy.optimize.linear_sum_assignment") from exc

from .geppetto_candidate_v1 import GeppettoRawOutputV1
from .training_targets_v1 import GeppettoTeacherTargetV1


@dataclass(frozen=True)
class GeppettoLossWeightsV1:
    position: float = 5.0
    existence: float = 1.0
    count: float = 1.0
    root: float = 1.0
    parent: float = 2.0
    abstain: float = 0.5
    match_root_cost: float = 0.25


class GeppettoLossV1:
    """Permutation-invariant Geppetto supervision with exact Hungarian matching.

    Teacher control order is supervision serialization only. Model query indices are
    internal coordinates and are matched anew for every sample; they never become
    canonical joint identity.
    """

    def __init__(self, weights: GeppettoLossWeightsV1 = GeppettoLossWeightsV1()):
        self.weights = weights

    def _match(self, positions: torch.Tensor, roots: torch.Tensor, target: GeppettoTeacherTargetV1) -> tuple[np.ndarray, np.ndarray]:
        teacher_pos = torch.as_tensor(target.positions_normalized, device=positions.device, dtype=positions.dtype)
        teacher_root = torch.as_tensor(target.root_mask, device=positions.device, dtype=positions.dtype)
        if teacher_pos.ndim != 2 or teacher_pos.shape[-1] != 3:
            raise ValueError("teacher positions must be [J,3]")
        if len(teacher_pos) == 0:
            return np.empty(0, np.int64), np.empty(0, np.int64)
        pos_cost = torch.cdist(positions, teacher_pos, p=1)
        root_prob = torch.sigmoid(roots)[:, None]
        root_cost = torch.abs(root_prob - teacher_root[None, :])
        cost = (pos_cost + self.weights.match_root_cost * root_cost).detach().cpu().numpy()
        q_idx, t_idx = linear_sum_assignment(cost)
        return q_idx.astype(np.int64), t_idx.astype(np.int64)

    def __call__(self, output: GeppettoRawOutputV1, targets: Sequence[GeppettoTeacherTargetV1]) -> dict[str, torch.Tensor]:
        bsz, k = output.existence_logits.shape
        if len(targets) != bsz:
            raise ValueError("target batch size mismatch")
        zero = output.existence_logits.sum() * 0.0
        totals = {name: zero for name in ("position", "existence", "count", "root", "parent", "abstain")}
        matched_total = 0
        for b, target in enumerate(targets):
            j = int(len(target.positions_normalized)) if target.valid else 0
            if j > k:
                raise ValueError(f"teacher joint count {j} exceeds query capacity {k}")
            existence_target = torch.zeros(k, device=output.existence_logits.device, dtype=output.existence_logits.dtype)
            if target.valid and j > 0:
                q_idx_np, t_idx_np = self._match(output.positions_normalized[b], output.root_logits[b], target)
                q_idx = torch.as_tensor(q_idx_np, device=output.existence_logits.device, dtype=torch.long)
                t_idx = torch.as_tensor(t_idx_np, device=output.existence_logits.device, dtype=torch.long)
                existence_target[q_idx] = 1.0
                matched_total += len(q_idx_np)
                teacher_pos = torch.as_tensor(target.positions_normalized, device=output.positions_normalized.device, dtype=output.positions_normalized.dtype)
                totals["position"] = totals["position"] + F.smooth_l1_loss(output.positions_normalized[b, q_idx], teacher_pos[t_idx], reduction="mean")
                teacher_root = torch.as_tensor(target.root_mask, device=output.root_logits.device, dtype=output.root_logits.dtype)
                totals["root"] = totals["root"] + F.binary_cross_entropy_with_logits(output.root_logits[b, q_idx], teacher_root[t_idx], reduction="mean")
                teacher_parent = np.asarray(target.parent_indices, dtype=np.int64)
                t_to_q = {int(t): int(q) for q, t in zip(q_idx_np.tolist(), t_idx_np.tolist())}
                rel_logits = []
                rel_labels = []
                for child_t in t_idx_np.tolist():
                    child_q = t_to_q[int(child_t)]
                    for parent_t in t_idx_np.tolist():
                        if child_t == parent_t:
                            continue
                        parent_q = t_to_q[int(parent_t)]
                        rel_logits.append(output.parent_logits[b, child_q, parent_q])
                        rel_labels.append(1.0 if int(teacher_parent[child_t]) == int(parent_t) else 0.0)
                if rel_logits:
                    logits = torch.stack(rel_logits)
                    labels = torch.tensor(rel_labels, device=logits.device, dtype=logits.dtype)
                    totals["parent"] = totals["parent"] + F.binary_cross_entropy_with_logits(logits, labels)
            totals["existence"] = totals["existence"] + F.binary_cross_entropy_with_logits(output.existence_logits[b], existence_target)
            count_target = torch.tensor([j], device=output.count_logits.device, dtype=torch.long)
            totals["count"] = totals["count"] + F.cross_entropy(output.count_logits[b:b + 1], count_target)
            abstain_target = torch.tensor(0.0 if target.valid and j > 0 else 1.0, device=output.abstain_logits.device, dtype=output.abstain_logits.dtype)
            totals["abstain"] = totals["abstain"] + F.binary_cross_entropy_with_logits(output.abstain_logits[b], abstain_target)
        for name in totals:
            totals[name] = totals[name] / float(max(bsz, 1))
        w = self.weights
        total = (
            w.position * totals["position"]
            + w.existence * totals["existence"]
            + w.count * totals["count"]
            + w.root * totals["root"]
            + w.parent * totals["parent"]
            + w.abstain * totals["abstain"]
        )
        return {"total": total, **totals, "matched_joint_count": torch.tensor(float(matched_total), device=total.device)}
