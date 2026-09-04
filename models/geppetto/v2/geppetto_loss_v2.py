from __future__ import annotations

from dataclasses import dataclass
import itertools
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


POSITION_LOG_SIGMA_MIN = -8.0
POSITION_LOG_SIGMA_MAX = 4.0
POSITION_DIMS = 3

# Shipping loci are float32 in a normalized common frame. Matching is a discrete
# identity decision, so sub-ULP/BLAS drift must not become a different training
# correspondence. 2^-16 is exactly representable and equals 128 float32 eps at
# unit scale; it is a numerical identity grid, not a product geometry tolerance.
MATCH_GEOMETRY_COST_QUANTIZATION = float(2.0 ** -16)
_MAX_EXACT_SCIPY_INTEGER = 2**52 - 1
_MAX_COINCIDENT_TOPOLOGY_VARIANTS = 256


@dataclass(frozen=True)
class GeppettoLossWeightsV2:
    # Preserve the historical Geppetto V1 direct-locus authority (total 5.0),
    # split equally between the locus shipped now and the WTA hypothesis that
    # confidence is being trained to select next. If both are the same mode the
    # aggregate direct geometric weight is still exactly 5.0.
    position_primary: float = 2.5
    position_winner: float = 2.5
    position_nll: float = 1.0
    mode_rank: float = 1.0
    existence: float = 1.0
    stop: float = 1.0
    root: float = 1.0
    parent: float = 2.0
    support: float = 0.5
    support_presence: float = 0.25
    abstain: float = 0.5


def _independent_zero_accumulators(zero: torch.Tensor, names: tuple[str, ...]) -> dict[str, torch.Tensor]:
    return {name: zero.clone() for name in names}


def _pairwise_rank_loss(positive_logits: torch.Tensor, negative_logits: torch.Tensor) -> torch.Tensor:
    """Logistic pair ranking for consumers whose real decision is rank-based."""
    if positive_logits.numel() == 0:
        return negative_logits.sum() * 0.0
    if negative_logits.numel() == 0:
        return positive_logits.sum() * 0.0
    return F.softplus(negative_logits[:, None] - positive_logits[None, :]).mean()


def _first_hit_stop_loss(stop_logits: torch.Tensor) -> torch.Tensor:
    """Align training with generate(): first sigmoid(logit)>=0.5 is STOP.

    The terminal positive and the worst earlier false-stop candidate have equal
    authority, so a long sequence cannot dilute the one terminal event.
    """
    if stop_logits.ndim != 1 or stop_logits.numel() < 1:
        raise ValueError("STOP boundary requires a non-empty 1D prefix")
    terminal = F.softplus(-stop_logits[-1])
    if stop_logits.numel() == 1:
        return terminal
    worst_early = F.softplus(stop_logits[:-1]).max()
    return 0.5 * (terminal + worst_early)


def _canonical_integer_tie_matrix(n: int) -> np.ndarray:
    """Unique deterministic secondary edge ranks with no fractional float authority."""
    if n < 1:
        raise ValueError("assignment cardinality must be positive")
    i, j = np.indices((n, n), dtype=np.uint64)
    x = ((i + 1) * np.uint64(0x9E3779B1)) ^ ((j + 1) * np.uint64(0x85EBCA77))
    x ^= ((i + 1) * (j + 1) * np.uint64(0xC2B2AE3D))
    flat_i = i.reshape(-1).astype(np.int64)
    flat_j = j.reshape(-1).astype(np.int64)
    flat_x = (x & np.uint64(0xFFFFFFFF)).reshape(-1)
    # x is the pseudo-random primary secondary key; i/j only make hash collisions
    # unique. The resulting ranks are exactly the integers 0..n^2-1.
    order = np.lexsort((flat_j, flat_i, flat_x))
    ranks = np.empty(n * n, dtype=np.int64)
    ranks[order] = np.arange(n * n, dtype=np.int64)
    return ranks.reshape(n, n)


def _canonical_teacher_geometry_order(teacher: np.ndarray) -> np.ndarray:
    """Canonicalize teacher columns by geometry, never by supplied row identity."""
    bins = np.rint(teacher / MATCH_GEOMETRY_COST_QUANTIZATION)
    if not np.isfinite(bins).all() or float(np.max(np.abs(bins))) >= float(2**52):
        raise ValueError("teacher geometry outside canonical matching range")
    # Quantized geometry is primary. Raw geometry only breaks two teacher loci
    # that land in the same numerical bin. Exact coincident rows remain an
    # equivalence class and are handled permutation-invariantly by topology loss.
    return np.lexsort((
        teacher[:, 2], teacher[:, 1], teacher[:, 0],
        bins[:, 2], bins[:, 1], bins[:, 0],
    )).astype(np.int64)


def canonical_geometry_assignment_v2(primary_positions, teacher_positions) -> tuple[np.ndarray, np.ndarray]:
    """Hard anonymous geometry match stable to machine-scale float32 drift.

    Teacher root/tree labels are intentionally absent: canonical root/tree remain
    Compiler authority and may not condition anonymous teacher<->query identity.
    The secondary objective is exact integer lexicographic authority, not a
    fractional float64 perturbation whose mantissa can disappear at large costs.
    """
    if isinstance(primary_positions, torch.Tensor):
        primary = primary_positions.detach().cpu().numpy()
    else:
        primary = np.asarray(primary_positions)
    if isinstance(teacher_positions, torch.Tensor):
        teacher = teacher_positions.detach().cpu().numpy()
    else:
        teacher = np.asarray(teacher_positions)
    primary = np.asarray(primary, dtype=np.float64)
    teacher = np.asarray(teacher, dtype=np.float64)
    if primary.ndim != 2 or teacher.ndim != 2 or primary.shape != teacher.shape or primary.shape[1] != POSITION_DIMS:
        raise ValueError("canonical geometry assignment requires equal [J,3] arrays")
    if primary.shape[0] < 1 or not np.isfinite(primary).all() or not np.isfinite(teacher).all():
        raise ValueError("canonical geometry assignment requires finite non-empty loci")

    teacher_order = _canonical_teacher_geometry_order(teacher)
    teacher_canonical = teacher[teacher_order]
    raw_cost = np.abs(primary[:, None, :] - teacher_canonical[None, :, :]).sum(axis=-1)
    quantized = np.rint(raw_cost / MATCH_GEOMETRY_COST_QUANTIZATION)
    if not np.isfinite(quantized).all() or (quantized < 0).any():
        raise ValueError("canonical geometry assignment cost invalid")
    primary_integer_cost = quantized.astype(np.int64)
    n = int(primary.shape[0])
    secondary = _canonical_integer_tie_matrix(n)
    # Any complete assignment has secondary sum < n^3. Multiplying each primary
    # edge by n^3+1 therefore makes a one-bin primary improvement dominate the
    # entire secondary assignment, exactly. The final integer is also guarded to
    # stay <=2^52 so scipy implementations that internally use float64 retain it.
    scale = int(n**3 + 1)
    max_primary = int(primary_integer_cost.max(initial=0))
    max_secondary = int(secondary.max(initial=0))
    if max_primary > (_MAX_EXACT_SCIPY_INTEGER - max_secondary) // scale:
        raise ValueError("canonical geometry assignment composite integer exceeds exact scipy range")
    deterministic_cost = primary_integer_cost * np.int64(scale) + secondary
    q, teacher_canonical_index = linear_sum_assignment(deterministic_cost)
    t = teacher_order[np.asarray(teacher_canonical_index, np.int64)]
    return np.asarray(q, np.int64), np.asarray(t, np.int64)


def _coincident_topology_mappings(q_np: np.ndarray, t_np: np.ndarray, teacher_positions: np.ndarray) -> list[dict[int, int]]:
    """Enumerate only topology-label bijections inside exact coincident loci.

    Geometry has no information that can distinguish two teacher rows at exactly
    the same locus. Instead of leaking teacher row identity into matching, root and
    parent supervision is minimized over the finite equivalence-class bijections.
    """
    base = {int(tt): int(qq) for qq, tt in zip(q_np.tolist(), t_np.tolist())}
    by_locus: dict[tuple[float, float, float], list[int]] = {}
    positions = np.asarray(teacher_positions, np.float64)
    for tt in sorted(base):
        key = tuple(0.0 if float(x) == 0.0 else float(x) for x in positions[tt])
        by_locus.setdefault(key, []).append(tt)
    groups = [tuple(ids) for ids in by_locus.values() if len(ids) > 1]
    variants = [base]
    for group in groups:
        q_values = tuple(base[t] for t in group)
        expanded: list[dict[int, int]] = []
        for mapping in variants:
            for q_perm in itertools.permutations(q_values):
                candidate = dict(mapping)
                for teacher_id, query_id in zip(group, q_perm):
                    candidate[int(teacher_id)] = int(query_id)
                expanded.append(candidate)
                if len(expanded) > _MAX_COINCIDENT_TOPOLOGY_VARIANTS:
                    raise ValueError("coincident teacher topology symmetry exceeds bounded permutation budget")
        variants = expanded
    return variants


class GeppettoLossV2:
    def __init__(self, weights: GeppettoLossWeightsV2 = GeppettoLossWeightsV2(), support_topk: int = 8):
        self.weights = weights
        self.support_topk = int(support_topk)

    def _match(self, output: GeppettoRawOutputV2, b: int, target, J: int):
        """Match teacher controls to shipping MAP representatives by geometry only."""
        tp = torch.as_tensor(
            target.positions_normalized,
            device=output.positions_normalized.device,
            dtype=output.positions_normalized.dtype,
        )
        primary = output.positions_normalized[b, :J]
        return canonical_geometry_assignment_v2(primary, tp)

    @staticmethod
    def _winner_indices(output: GeppettoRawOutputV2, b: int, q: torch.Tensor, target_positions: torch.Tensor) -> torch.Tensor:
        modes = output.position_modes_normalized[b, q]
        dist2 = (modes.detach() - target_positions[:, None, :]).square().sum(dim=-1)
        return torch.argsort(dist2, dim=-1, stable=True)[..., 0]

    @staticmethod
    def _winner_positions(output: GeppettoRawOutputV2, b: int, q: torch.Tensor, winner: torch.Tensor) -> torch.Tensor:
        modes = output.position_modes_normalized[b, q]
        gather = winner[:, None, None].expand(len(winner), 1, POSITION_DIMS)
        return torch.gather(modes, 1, gather).squeeze(1)

    @staticmethod
    def _mixture_position_nll(output: GeppettoRawOutputV2, b: int, q: torch.Tensor, target_positions: torch.Tensor) -> torch.Tensor:
        """Calibrate multimodal uncertainty without moving locus means or scores.

        Geometry is learned by explicit shipping/WTA regression and discrete mode
        choice by mode_rank. Candidate construction disconnects sigma-head inputs
        from the shared latent state, so this likelihood calibrates only sigma-head
        parameters and cannot perturb locus/cardinality/topology evidence.
        """
        means = output.position_modes_normalized[b, q].detach()
        log_sigma = output.position_mode_log_sigma[b, q].clamp(POSITION_LOG_SIGMA_MIN, POSITION_LOG_SIGMA_MAX)
        mix_logp = torch.log_softmax(output.position_mode_logits[b, q].detach(), dim=-1)
        residual = means - target_positions[:, None, :]
        component_logp = mix_logp - 0.5 * (torch.exp(-2.0 * log_sigma) * residual.square()).sum(dim=-1) - log_sigma.sum(dim=-1)
        raw_nll = -torch.logsumexp(component_logp, dim=-1)
        return (raw_nll - float(POSITION_DIMS * POSITION_LOG_SIGMA_MIN)).mean()

    @staticmethod
    def _mode_rank_loss(output: GeppettoRawOutputV2, b: int, q: torch.Tensor, winner: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(output.position_mode_logits[b, q], winner)

    @staticmethod
    def _root_parent_loss_for_mapping(output: GeppettoRawOutputV2, b: int, target: GeppettoTeacherTargetV1, mapping: dict[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
        teacher_ids = sorted(mapping)
        q_ids = torch.as_tensor([mapping[t] for t in teacher_ids], device=output.root_logits.device, dtype=torch.long)
        tr_all = np.asarray(target.root_mask, dtype=bool)
        tr = torch.as_tensor([bool(tr_all[t]) for t in teacher_ids], device=output.root_logits.device, dtype=torch.bool)
        matched_root_logits = output.root_logits[b, q_ids]
        root_pos = matched_root_logits[tr]
        root_neg = matched_root_logits[~tr]
        if root_pos.numel() and root_neg.numel():
            root_loss = _pairwise_rank_loss(root_pos, root_neg)
        elif root_pos.numel():
            root_loss = F.softplus(-root_pos).mean()
        else:
            root_loss = F.softplus(root_neg).mean()

        teacher_parent = np.asarray(target.parent_indices, np.int64)
        parent_losses = []
        for child_t in teacher_ids:
            parent_t = int(teacher_parent[child_t])
            if parent_t < 0:
                continue
            if parent_t not in mapping:
                raise ValueError("teacher parent missing from Hungarian match")
            candidates = [pt for pt in teacher_ids if pt != child_t]
            logits = torch.stack([output.parent_logits[b, mapping[child_t], mapping[pt]] for pt in candidates])
            target_index = candidates.index(parent_t)
            parent_losses.append(F.cross_entropy(logits[None], torch.tensor([target_index], device=logits.device)))
        parent_loss = torch.stack(parent_losses).mean() if parent_losses else root_loss * 0.0
        return root_loss, parent_loss

    def __call__(self, output: GeppettoRawOutputV2, targets: Sequence[GeppettoTeacherTargetV1], surface_positions_normalized: torch.Tensor, valid_mask: torch.Tensor):
        B, K = output.existence_logits.shape
        if len(targets) != B or surface_positions_normalized.shape[:2] != valid_mask.shape or surface_positions_normalized.shape[-1] != 3:
            raise ValueError("Geppetto V2 loss batch mismatch")
        if output.position_modes_normalized.shape[:2] != (B, K) or output.position_modes_normalized.shape[-1] != 3:
            raise ValueError("Geppetto V2 multimodal locus output missing")
        zero = output.existence_logits.sum() * 0.0
        names = (
            "position_primary",
            "position_winner",
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

            ex_target = torch.zeros(K, device=output.existence_logits.device, dtype=output.existence_logits.dtype)
            ex_target[:J] = 1.0
            total["existence"] += F.binary_cross_entropy_with_logits(output.existence_logits[b], ex_target)
            abst = torch.tensor(0.0 if target.valid and J > 0 else 1.0, device=output.abstain_logits.device, dtype=output.abstain_logits.dtype)
            total["abstain"] += F.binary_cross_entropy_with_logits(output.abstain_logits[b], abst)
            if J == 0:
                continue

            total["stop"] += _first_hit_stop_loss(output.stop_logits[b, :J])

            q_np, t_np = self._match(output, b, target, J)
            q = torch.as_tensor(q_np, device=output.positions_normalized.device)
            t = torch.as_tensor(t_np, device=output.positions_normalized.device)
            matched += J

            tp = torch.as_tensor(target.positions_normalized, device=output.positions_normalized.device, dtype=output.positions_normalized.dtype)[t]
            winner = self._winner_indices(output, b, q, tp)
            winner_positions = self._winner_positions(output, b, q, winner)

            total["position_primary"] += F.smooth_l1_loss(output.positions_normalized[b, q], tp, reduction="mean")
            total["position_winner"] += F.smooth_l1_loss(winner_positions, tp, reduction="mean")
            total["position_nll"] += self._mixture_position_nll(output, b, q, tp)
            total["mode_rank"] += self._mode_rank_loss(output, b, q, winner)

            variants = _coincident_topology_mappings(q_np, t_np, np.asarray(target.positions_normalized, np.float64))
            topology_candidates = []
            for mapping in variants:
                root_loss, parent_loss = self._root_parent_loss_for_mapping(output, b, target, mapping)
                weighted = self.weights.root * root_loss + self.weights.parent * parent_loss
                topology_candidates.append((weighted, root_loss, parent_loss))
            # Select by the actual weighted topology objective. Secondary detached
            # keys make equal weighted optima report the same components regardless
            # of enumeration order; no teacher row id becomes product identity.
            best = min(
                range(len(topology_candidates)),
                key=lambda i: (
                    float(topology_candidates[i][0].detach().cpu()),
                    float(topology_candidates[i][1].detach().cpu()),
                    float(topology_candidates[i][2].detach().cpu()),
                ),
            )
            total["root"] += topology_candidates[best][1]
            total["parent"] += topology_candidates[best][2]

            surface = surface_positions_normalized[b, :N]
            support_losses = []
            presence = []
            raw_teacher = torch.as_tensor(target.positions_normalized, device=surface.device, dtype=surface.dtype)
            all_surface_ids = torch.arange(N, device=surface.device)
            for qq, tt in zip(q_np.tolist(), t_np.tolist()):
                d = torch.linalg.norm(surface - raw_teacher[tt][None], dim=-1)
                top = min(self.support_topk, N)
                ids = torch.argsort(d, stable=True)[:top]
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
            + w.position_winner * total["position_winner"]
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
