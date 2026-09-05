from __future__ import annotations

import itertools
import numpy as np
import torch
import torch.nn.functional as F

from models.geppetto.v2.geppetto_loss_v2 import (
    GeppettoLossV2,
    _first_hit_stop_loss,
    _independent_zero_accumulators,
    _pairwise_rank_loss,
)

MAX_COINCIDENT_TOPOLOGY_VARIANTS_V3 = 4096


def coincident_mapping_matrix_v3(
    q_np: np.ndarray,
    t_np: np.ndarray,
    teacher_positions: np.ndarray,
    *,
    max_variants: int = MAX_COINCIDENT_TOPOLOGY_VARIANTS_V3,
) -> np.ndarray:
    """Enumerate true exact-locus teacher symmetries as a dense mapping matrix."""
    q_np = np.asarray(q_np, np.int64)
    t_np = np.asarray(t_np, np.int64)
    positions = np.asarray(teacher_positions, np.float64)
    if q_np.ndim != 1 or t_np.shape != q_np.shape or positions.shape != (len(q_np), 3):
        raise ValueError("coincident mapping requires matched [J] indices and [J,3] teacher positions")
    j = len(q_np)
    if j < 1 or set(q_np.tolist()) != set(range(j)) or set(t_np.tolist()) != set(range(j)):
        raise ValueError("coincident mapping requires complete J-way assignment")
    if not np.isfinite(positions).all() or int(max_variants) < 1:
        raise ValueError("invalid coincident mapping input")
    base = np.full(j, -1, dtype=np.int64)
    base[t_np] = q_np
    by_locus: dict[tuple[float, float, float], list[int]] = {}
    for teacher_id in range(j):
        key = tuple(0.0 if float(x) == 0.0 else float(x) for x in positions[teacher_id])
        by_locus.setdefault(key, []).append(teacher_id)
    groups = [tuple(ids) for ids in by_locus.values() if len(ids) > 1]
    variants = base[None, :]
    for group in groups:
        q_values = tuple(int(base[t]) for t in group)
        permutations = np.asarray(tuple(itertools.permutations(q_values)), dtype=np.int64)
        requested = int(len(variants) * len(permutations))
        if requested > int(max_variants):
            raise ValueError(
                f"coincident teacher topology symmetry exceeds V3 bounded permutation budget:{requested}>{int(max_variants)}"
            )
        expanded = np.repeat(variants, len(permutations), axis=0)
        expanded[:, np.asarray(group, np.int64)] = np.tile(permutations, (len(variants), 1))
        variants = expanded
    if len(np.unique(variants, axis=0)) != len(variants):
        raise AssertionError("coincident mapping enumeration produced duplicates")
    return variants


def vectorized_root_parent_loss_v3(output, b: int, target, mappings_np: np.ndarray, *, root_weight: float, parent_weight: float) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Exact V2 root/parent semantics, vectorized over anonymous coincident mappings."""
    device = output.root_logits.device
    mappings = torch.as_tensor(mappings_np, device=device, dtype=torch.long)
    if mappings.ndim != 2:
        raise ValueError("mapping matrix must be [V,J]")
    v, j = mappings.shape
    if v < 1 or j < 1:
        raise ValueError("mapping matrix cannot be empty")
    root_mask_np = np.asarray(target.root_mask, dtype=bool)
    parent_np = np.asarray(target.parent_indices, dtype=np.int64)
    if root_mask_np.shape != (j,) or parent_np.shape != (j,):
        raise ValueError("target topology cardinality drift")
    root_by_teacher = output.root_logits[b][mappings]
    root_mask = torch.as_tensor(root_mask_np, device=device, dtype=torch.bool)
    root_pos = root_by_teacher[:, root_mask]
    root_neg = root_by_teacher[:, ~root_mask]
    if root_pos.numel() and root_neg.numel():
        root_loss = F.softplus(root_neg[:, :, None] - root_pos[:, None, :]).mean(dim=(1, 2))
    elif root_pos.numel():
        root_loss = F.softplus(-root_pos).mean(dim=1)
    else:
        root_loss = F.softplus(root_neg).mean(dim=1)
    child_t_np = np.flatnonzero(parent_np >= 0).astype(np.int64)
    if len(child_t_np):
        parent_t_np = parent_np[child_t_np]
        child_t = torch.as_tensor(child_t_np, device=device, dtype=torch.long)
        parent_t = torch.as_tensor(parent_t_np, device=device, dtype=torch.long)
        q_child = mappings[:, child_t]
        q_parent = mappings[:, parent_t]
        logits_matrix = output.parent_logits[b, :j, :j]
        rows = logits_matrix[q_child]
        self_mask = torch.arange(j, device=device)[None, None, :] == q_child[:, :, None]
        rows = rows.masked_fill(self_mask, float("-inf"))
        target_logits = torch.gather(rows, 2, q_parent[:, :, None]).squeeze(-1)
        parent_loss = (torch.logsumexp(rows, dim=-1) - target_logits).mean(dim=1)
    else:
        parent_loss = root_loss * 0.0
    combined = float(root_weight) * root_loss + float(parent_weight) * parent_loss
    score = np.stack([
        combined.detach().float().cpu().numpy(),
        root_loss.detach().float().cpu().numpy(),
        parent_loss.detach().float().cpu().numpy(),
    ], axis=1)
    best = int(np.lexsort((score[:, 2], score[:, 1], score[:, 0]))[0])
    return root_loss[best], parent_loss[best], best


class GeppettoLossV3CoincidentVectorized(GeppettoLossV2):
    """V2 loss with scalable exact-locus topology symmetry handling only."""
    def __call__(self, output, targets, surface_positions_normalized: torch.Tensor, valid_mask: torch.Tensor):
        B, K = output.existence_logits.shape
        if len(targets) != B or surface_positions_normalized.shape[:2] != valid_mask.shape or surface_positions_normalized.shape[-1] != 3:
            raise ValueError("Geppetto V3 loss batch mismatch")
        if output.position_modes_normalized.shape[:2] != (B, K) or output.position_modes_normalized.shape[-1] != 3:
            raise ValueError("Geppetto V3 multimodal locus output missing")
        zero = output.existence_logits.sum() * 0.0
        names = ("position_primary","position_winner","position_nll","mode_rank","existence","stop","root","parent","support","support_presence","abstain")
        total = _independent_zero_accumulators(zero, names)
        matched = 0
        for b, target in enumerate(targets):
            J = int(len(target.positions_normalized)) if target.valid else 0
            N = int(valid_mask[b].sum().item())
            if J > K: raise ValueError(f"teacher joint count {J} exceeds decoded steps {K}")
            if J > N: raise ValueError(f"teacher joint count {J} exceeds surface resource guard {N}")
            ex_target = torch.zeros(K, device=output.existence_logits.device, dtype=output.existence_logits.dtype); ex_target[:J] = 1.0
            total["existence"] += F.binary_cross_entropy_with_logits(output.existence_logits[b], ex_target)
            abst = torch.tensor(0.0 if target.valid and J > 0 else 1.0, device=output.abstain_logits.device, dtype=output.abstain_logits.dtype)
            total["abstain"] += F.binary_cross_entropy_with_logits(output.abstain_logits[b], abst)
            if J == 0: continue
            total["stop"] += _first_hit_stop_loss(output.stop_logits[b, :J])
            q_np, t_np = self._match(output, b, target, J)
            q = torch.as_tensor(q_np, device=output.positions_normalized.device); t = torch.as_tensor(t_np, device=output.positions_normalized.device)
            matched += J
            tp = torch.as_tensor(target.positions_normalized, device=output.positions_normalized.device, dtype=output.positions_normalized.dtype)[t]
            winner = self._winner_indices(output, b, q, tp); winner_positions = self._winner_positions(output, b, q, winner)
            total["position_primary"] += F.smooth_l1_loss(output.positions_normalized[b, q], tp, reduction="mean")
            total["position_winner"] += F.smooth_l1_loss(winner_positions, tp, reduction="mean")
            total["position_nll"] += self._mixture_position_nll(output, b, q, tp)
            total["mode_rank"] += self._mode_rank_loss(output, b, q, winner)
            mappings = coincident_mapping_matrix_v3(q_np, t_np, np.asarray(target.positions_normalized, np.float64))
            root_loss, parent_loss, _best = vectorized_root_parent_loss_v3(
                output, b, target, mappings, root_weight=self.weights.root, parent_weight=self.weights.parent
            )
            total["root"] += root_loss; total["parent"] += parent_loss
            surface = surface_positions_normalized[b, :N]; support_losses=[]; presence=[]
            raw_teacher = torch.as_tensor(target.positions_normalized, device=surface.device, dtype=surface.dtype); all_surface_ids=torch.arange(N,device=surface.device)
            for qq, tt in zip(q_np.tolist(), t_np.tolist()):
                d=torch.linalg.norm(surface-raw_teacher[tt][None],dim=-1); top=min(self.support_topk,N); ids=torch.argsort(d,stable=True)[:top]
                logits=output.support_logits[b,qq,:N]; positive=logits[ids]
                if top<N:
                    negative_mask=torch.ones(N,device=surface.device,dtype=torch.bool); negative_mask[ids]=False; negative=logits[all_surface_ids[negative_mask]]; support_losses.append(_pairwise_rank_loss(positive,negative))
                else: support_losses.append(logits.sum()*0.0)
                presence.append(output.support_presence_logits[b,qq])
            total["support"] += torch.stack(support_losses).mean()
            total["support_presence"] += F.binary_cross_entropy_with_logits(torch.stack(presence),torch.ones(len(presence),device=surface.device,dtype=output.support_presence_logits.dtype))
        for n in total: total[n]=total[n]/float(max(B,1))
        w=self.weights
        loss=(w.position_primary*total["position_primary"]+w.position_winner*total["position_winner"]+w.position_nll*total["position_nll"]+w.mode_rank*total["mode_rank"]+w.existence*total["existence"]+w.stop*total["stop"]+w.root*total["root"]+w.parent*total["parent"]+w.support*total["support"]+w.support_presence*total["support_presence"]+w.abstain*total["abstain"])
        return {"total":loss,**total,"matched_joint_count":torch.tensor(float(matched),device=loss.device)}
