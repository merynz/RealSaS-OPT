from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn

from realsas_compiler_core.types import (
    RiggingSurfaceIR,
    SkeletonProposalEdge,
    SkeletonProposalIR,
    SkeletonProposalJoint,
)

from .point_context_v2 import LocalPointContextEncoder, sinusoidal_position


@dataclass(frozen=True)
class GeppettoFitV2Config:
    token_dim: int = 192
    hidden_dim: int = 384
    surface_layers: int = 3
    decoder_layers: int = 4
    heads: int = 8
    local_k: int = 16
    stop_threshold: float = 0.5
    position_extent: float = 0.85


@dataclass
class GeppettoStepOutput:
    position_norm: torch.Tensor
    parent_logits: torch.Tensor
    stop_logit: torch.Tensor
    state: torch.Tensor


@dataclass
class GeppettoTeacherTrace:
    ordered_teacher_indices: tuple[int, ...]
    ordered_parent_local: tuple[int, ...]
    steps: tuple[GeppettoStepOutput, ...]
    teacher_position_norm: torch.Tensor


def _surface_tensor(surface: RiggingSurfaceIR, device: torch.device):
    if not surface.surface_nodes:
        raise ValueError('Geppetto requires non-empty surface')
    p = torch.tensor([n.P for n in surface.surface_nodes], dtype=torch.float32, device=device)
    if not torch.isfinite(p).all():
        raise ValueError('surface contains non-finite P')
    return p


def _geometry_bfs_order(positions: torch.Tensor, parent_index: torch.Tensor) -> tuple[int, ...]:
    """Deterministic training serialization only; never product identity authority."""
    p = positions.detach().cpu()
    parent = parent_index.detach().cpu().tolist()
    roots = [i for i, x in enumerate(parent) if int(x) < 0]
    if not roots:
        raise ValueError('teacher skeleton has no root')
    children: dict[int, list[int]] = {i: [] for i in range(len(parent))}
    for child, par in enumerate(parent):
        if par >= 0:
            if par >= len(parent):
                raise ValueError('teacher parent index out of range')
            children[par].append(child)

    def key(i: int):
        q = p[i]
        return (float(q[0]), float(q[1]), float(q[2]), int(i))

    roots.sort(key=key)
    out: list[int] = []
    queue = list(roots)
    seen = set()
    while queue:
        i = queue.pop(0)
        if i in seen:
            raise ValueError('teacher parent cycle/duplicate')
        seen.add(i)
        out.append(i)
        queue.extend(sorted(children[i], key=key))
    if len(out) != len(parent):
        raise ValueError('teacher skeleton disconnected or cyclic')
    return tuple(out)


def _ordered_parent_local(order: Sequence[int], parent_index: torch.Tensor) -> tuple[int, ...]:
    source_to_local = {src: i for i, src in enumerate(order)}
    out = []
    parent = parent_index.detach().cpu().tolist()
    for local, src in enumerate(order):
        p = int(parent[src])
        if p < 0:
            out.append(-1)
        else:
            if p not in source_to_local or source_to_local[p] >= local:
                raise ValueError('teacher serialization violates parent-before-child')
            out.append(source_to_local[p])
    return tuple(out)


class GeppettoFitV2(nn.Module):
    """Variable-cardinality causal skeleton proposer.

    Joint count is not a model hyperparameter. Generation terminates through a learned
    STOP decision. The runtime safety budget is derived from available surface evidence,
    not from a character-class cap. Parent decisions enter the history state used to
    generate later joints. Compiler remains final root/tree/canonical-ID authority.
    """

    def __init__(self, cfg: GeppettoFitV2Config = GeppettoFitV2Config()):
        super().__init__()
        self.cfg = cfg
        d = cfg.token_dim
        self.surface_encoder = LocalPointContextEncoder(
            dim=d, hidden_dim=cfg.hidden_dim, layers=cfg.surface_layers,
            heads=cfg.heads, local_k=cfg.local_k,
        )
        self.history_mlp = nn.Sequential(
            nn.Linear(11, d), nn.GELU(), nn.Linear(d, d), nn.LayerNorm(d)
        )
        self.next_query = nn.Parameter(torch.zeros(d))
        nn.init.normal_(self.next_query, std=0.02)
        layer = nn.TransformerDecoderLayer(
            d_model=d, nhead=cfg.heads, dim_feedforward=cfg.hidden_dim,
            dropout=0.0, batch_first=True, norm_first=True, activation='gelu'
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=cfg.decoder_layers)
        self.out_norm = nn.LayerNorm(d)
        self.position_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 3))
        self.parent_query = nn.Linear(d, d, bias=False)
        self.parent_key = nn.Linear(d, d, bias=False)
        self.root_key = nn.Parameter(torch.zeros(d))
        nn.init.normal_(self.root_key, std=0.02)
        self.stop_head = nn.Sequential(nn.Linear(d, d // 2), nn.GELU(), nn.Linear(d // 2, 1))

    def encode_surface(self, surface_points: torch.Tensor):
        return self.surface_encoder(surface_points)

    def _history_features(self, positions: torch.Tensor, parent_local: Sequence[int]) -> torch.Tensor:
        b, k, _ = positions.shape
        if k == 0:
            return positions.new_zeros((b, 0, 11))
        parent_pos = []
        root = []
        depth = []
        for i, p in enumerate(parent_local):
            if p < 0:
                parent_pos.append(positions[:, i])
                root.append(1.0)
                depth.append(0.0)
            else:
                parent_pos.append(positions[:, p])
                root.append(0.0)
                d = 1
                q = int(p)
                while parent_local[q] >= 0:
                    d += 1
                    q = int(parent_local[q])
                    if d > k:
                        raise ValueError('parent cycle in history')
                depth.append(float(d))
        pp = torch.stack(parent_pos, dim=1)
        rel = positions - pp
        length = torch.linalg.norm(rel, dim=-1, keepdim=True)
        root_t = positions.new_tensor(root)[None, :, None].expand(b, -1, -1)
        depth_t = positions.new_tensor(depth)[None, :, None].expand(b, -1, -1)
        depth_t = depth_t / max(1.0, float(k))
        return torch.cat([positions, pp, rel, length, root_t, depth_t], dim=-1)

    def _step(self, memory: torch.Tensor, previous_positions_norm: torch.Tensor, previous_parent_local: Sequence[int]) -> GeppettoStepOutput:
        b, k, _ = previous_positions_norm.shape
        if k:
            hist = self.history_mlp(self._history_features(previous_positions_norm, previous_parent_local))
            pos = torch.stack([
                sinusoidal_position(i, self.cfg.token_dim, device=hist.device, dtype=hist.dtype)
                for i in range(k)
            ], dim=0)[None].expand(b, -1, -1)
            hist = hist + pos
        else:
            hist = previous_positions_norm.new_zeros((b, 0, self.cfg.token_dim))
        q = self.next_query[None, None, :].expand(b, 1, -1)
        q = q + sinusoidal_position(k, self.cfg.token_dim, device=q.device, dtype=q.dtype)[None, None]
        tgt = torch.cat([hist, q], dim=1)
        t = tgt.shape[1]
        causal = torch.full((t, t), float('-inf'), device=tgt.device, dtype=tgt.dtype)
        causal = torch.triu(causal, diagonal=1)
        decoded = self.out_norm(self.decoder(tgt=tgt, memory=memory, tgt_mask=causal))
        state = decoded[:, -1]
        position = self.cfg.position_extent * torch.tanh(self.position_head(state))
        qv = self.parent_query(state)
        root = torch.sum(qv * self.root_key[None], dim=-1, keepdim=True) / (self.cfg.token_dim ** 0.5)
        if k:
            keys = self.parent_key(decoded[:, :k])
            prior = torch.einsum('bd,bkd->bk', qv, keys) / (self.cfg.token_dim ** 0.5)
            parent_logits = torch.cat([root, prior], dim=-1)
        else:
            parent_logits = root
        stop = self.stop_head(state).squeeze(-1)
        return GeppettoStepOutput(position, parent_logits, stop, state)

    def teacher_forced_trace(self, surface_points: torch.Tensor, teacher_positions_world: torch.Tensor, teacher_parent_index: torch.Tensor) -> GeppettoTeacherTrace:
        if surface_points.ndim != 3 or surface_points.shape[0] != 1:
            raise ValueError('demo Geppetto teacher forcing requires surface [1,N,3]')
        memory, _, center, scale = self.encode_surface(surface_points)
        order = _geometry_bfs_order(teacher_positions_world, teacher_parent_index)
        parent_local = _ordered_parent_local(order, teacher_parent_index)
        ordered = teacher_positions_world[list(order)].to(surface_points.device)
        teacher_norm = (ordered[None] - center) / scale
        steps = []
        for k in range(len(order) + 1):
            steps.append(self._step(memory, teacher_norm[:, :k], parent_local[:k]))
        return GeppettoTeacherTrace(order, parent_local, tuple(steps), teacher_norm[0])

    @torch.no_grad()
    def propose(self, surface: RiggingSurfaceIR, *, device: torch.device | str = 'cpu') -> SkeletonProposalIR:
        device = torch.device(device)
        self.eval()
        p = _surface_tensor(surface, device)
        memory, _, center, scale = self.encode_surface(p[None])
        previous = p.new_zeros((1, 0, 3))
        parents: list[int] = []
        step_parent_prob: list[torch.Tensor] = []
        step_stop_prob: list[float] = []
        safety_budget = max(1, len(surface.surface_nodes))
        stopped = False
        for k in range(safety_budget + 1):
            out = self._step(memory, previous, parents)
            stop_prob = float(torch.sigmoid(out.stop_logit[0]).item())
            if k > 0 and stop_prob >= self.cfg.stop_threshold:
                stopped = True
                break
            if k >= safety_budget:
                break
            probs = torch.softmax(out.parent_logits[0], dim=-1)
            parent_class = int(torch.argmax(probs).item())
            parent_local = parent_class - 1
            previous = torch.cat([previous, out.position_norm[:, None, :]], dim=1)
            parents.append(parent_local)
            step_parent_prob.append(probs.detach().cpu())
            step_stop_prob.append(stop_prob)
        if not stopped:
            raise RuntimeError('GEPPETTO_LEARNED_STOP_NOT_REACHED_WITHIN_SURFACE_EVIDENCE_BUDGET')
        if previous.shape[1] == 0:
            raise RuntimeError('GEPPETTO_GENERATED_EMPTY_SKELETON')

        world = previous * scale + center
        joints = []
        edges = []
        surface_ids = tuple(n.surface_id for n in surface.surface_nodes)
        for i in range(previous.shape[1]):
            wp = world[0, i]
            d2 = torch.sum((p - wp) ** 2, dim=-1)
            support_k = min(12, len(surface_ids))
            nearest = torch.topk(d2, k=support_k, largest=False).indices.tolist()
            probs = step_parent_prob[i]
            root_score = float(probs[0].item())
            confidence = float(max(0.0, 1.0 - step_stop_prob[i]))
            joints.append(SkeletonProposalJoint(
                proposal_id=f'G2:{i:04d}',
                position=tuple(float(x) for x in wp.tolist()),
                root_score=root_score,
                confidence=confidence,
                support_surface_ids=tuple(surface_ids[x] for x in nearest),
                metadata={'generation_index': i, 'variable_cardinality': True, 'model': 'GeppettoFitV2'},
            ))
            for parent_i in range(i):
                score = float(probs[parent_i + 1].item())
                edges.append(SkeletonProposalEdge(
                    edge_id=f'G2E:{parent_i:04d}>{i:04d}',
                    parent_proposal_id=f'G2:{parent_i:04d}',
                    child_proposal_id=f'G2:{i:04d}',
                    score=score,
                    confidence=score,
                    reason='GEPPETTO_V2_CAUSAL_PARENT_EVIDENCE',
                    metadata={'parent_generated_before_child': True},
                ))
        return SkeletonProposalIR(
            joints=tuple(joints), edges=tuple(edges),
            surface_binding_hash=surface.geometry_lineage_hash,
            model_provenance='RealSaS.GeppettoFitV2.experimental',
            metadata={
                'generalization_claim': False,
                'variable_joint_count': True,
                'joint_count_hyperparameter': None,
                'learned_termination': True,
                'serialization_is_not_canonical_identity': True,
                'compiler_owns_final_root_parent_ids': True,
            },
        )
