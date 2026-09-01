from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch
from torch import nn
import torch.nn.functional as F

from realsas_compiler_core.types import (
    RiggingSurfaceIR,
    SkeletonProposalEdge,
    SkeletonProposalIR,
    SkeletonProposalJoint,
)
from .conditioning import GeppettoConditioningConfig, GeppettoConditioningBatch, build_geppetto_conditioning


@dataclass(frozen=True)
class GeppettoR6Config:
    """Full product-intent Geppetto configuration.

    There is intentionally no max_joints/max_controls product parameter. The
    output cardinality is endogenous through STOP. A runtime resource budget is
    derived from the current admitted surface only to prevent infinite loops.
    """
    model_dim: int = 256
    hidden_dim: int = 768
    knn_k: int = 16
    encoder_layers: int = 4
    global_layers: int = 3
    decoder_layers: int = 4
    relation_layers: int = 2
    dropout: float = 0.0
    stop_threshold: float = 0.5
    unsupported_threshold: float = 0.8
    support_surface_k: int = 16


class LocalGeometryAttention(nn.Module):
    """Permutation-equivariant local point attention with relative geometry."""
    def __init__(self, dim: int, k: int, dropout: float):
        super().__init__()
        self.k = int(k)
        self.q = nn.Linear(dim, dim, bias=False)
        self.kv = nn.Linear(dim, 2 * dim, bias=False)
        self.rel = nn.Sequential(nn.Linear(4, dim), nn.GELU(), nn.Linear(dim, dim))
        self.proj = nn.Linear(dim, dim)
        self.ff = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * dim, dim))
        self.n1 = nn.LayerNorm(dim)
        self.n2 = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    @staticmethod
    def _gather(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        b = torch.arange(x.shape[0], device=x.device)[:, None, None]
        return x[b, idx]

    def forward(self, h: torch.Tensor, xyz: torch.Tensor) -> torch.Tensor:
        B, N, D = h.shape
        if N == 1:
            return self.n2(h + self.drop(self.ff(self.n1(h))))
        d2 = torch.cdist(xyz.float(), xyz.float()).square()
        eye = torch.eye(N, device=d2.device, dtype=torch.bool)[None]
        d2 = d2.masked_fill(eye, float("inf"))
        k = min(max(1, self.k), N - 1)
        idx = torch.topk(d2, k=k, dim=-1, largest=False).indices
        hn = self._gather(h, idx)
        xn = self._gather(xyz, idx)
        delta = xn - xyz[:, :, None, :]
        dist = torch.linalg.norm(delta, dim=-1, keepdim=True)
        rel = self.rel(torch.cat([delta, dist], dim=-1))
        q = self.q(h)[:, :, None]
        kk, vv = self.kv(hn).chunk(2, dim=-1)
        attn = torch.softmax((q * (kk + rel)).sum(-1) / math.sqrt(D), dim=-1)
        msg = (attn[..., None] * (vv + rel)).sum(2)
        h = self.n1(h + self.drop(self.proj(msg)))
        return self.n2(h + self.drop(self.ff(h)))


class SurfaceEncoder(nn.Module):
    def __init__(self, cfg: GeppettoR6Config):
        super().__init__()
        d = cfg.model_dim
        self.stem = nn.Sequential(nn.Linear(23, d), nn.GELU(), nn.Linear(d, d), nn.LayerNorm(d))
        self.local = nn.ModuleList([LocalGeometryAttention(d, cfg.knn_k, cfg.dropout) for _ in range(cfg.encoder_layers)])
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=8, dim_feedforward=cfg.hidden_dim,
            dropout=cfg.dropout, batch_first=True, norm_first=True,
            activation="gelu",
        )
        self.global_encoder = nn.TransformerEncoder(layer, num_layers=cfg.global_layers)
        self.norm = nn.LayerNorm(d)

    def forward(self, cond: GeppettoConditioningBatch):
        h = self.stem(cond.features)
        for block in self.local:
            h = block(h, cond.points_norm)
        h = self.norm(self.global_encoder(h))
        return h, h.mean(dim=1)


class DirectedRelationRefiner(nn.Module):
    """Full directed pair evidence after temporary controls exist."""
    def __init__(self, cfg: GeppettoR6Config):
        super().__init__()
        d = cfg.model_dim
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=8, dim_feedforward=cfg.hidden_dim,
            dropout=cfg.dropout, batch_first=True, norm_first=True,
            activation="gelu",
        )
        self.ctx = nn.TransformerEncoder(layer, num_layers=cfg.relation_layers)
        self.root = nn.Linear(d, 1)
        self.edge = nn.Sequential(
            nn.Linear(2 * d + 4, cfg.hidden_dim), nn.GELU(),
            nn.Linear(cfg.hidden_dim, d), nn.GELU(), nn.Linear(d, 1),
        )

    def forward(self, states: torch.Tensor, pos: torch.Tensor):
        z = self.ctx(states)
        B, K, D = z.shape
        zp = z[:, :, None].expand(B, K, K, D)
        zc = z[:, None, :].expand(B, K, K, D)
        delta = pos[:, None, :, :] - pos[:, :, None, :]
        dist = torch.linalg.norm(delta, dim=-1, keepdim=True)
        logits = self.edge(torch.cat([zp, zc, delta, dist], dim=-1)).squeeze(-1)
        eye = torch.eye(K, device=logits.device, dtype=torch.bool)[None]
        logits = logits.masked_fill(eye, -30.0)
        return logits, self.root(z).squeeze(-1), z


class GeppettoR6(nn.Module):
    """Variable-cardinality anonymous control generator.

    Product authority remains Compiler-side. Generation order is only a temporary
    representation. Parent selection is fed back into the recurrent mechanical
    state, so topology is causal rather than a passive side head.
    """
    def __init__(self, cfg: GeppettoR6Config = GeppettoR6Config()):
        super().__init__()
        self.cfg = cfg
        d = cfg.model_dim
        self.encoder = SurfaceEncoder(cfg)
        self.start = nn.Parameter(torch.zeros(d))
        # Previous generated mechanical state: xyz, parent xyz, relative bone,
        # bone length, depth, root flag, uncertainty = 15D.
        self.history = nn.Sequential(nn.Linear(15, d), nn.GELU(), nn.Linear(d, d), nn.LayerNorm(d))
        self.decoder = nn.ModuleList([nn.GRUCell(2 * d, d) for _ in range(cfg.decoder_layers)])
        self.pos = nn.Linear(d, 3)
        self.log_sigma = nn.Linear(d, 3)
        self.root = nn.Linear(d, 1)
        self.stop = nn.Linear(d, 1)
        self.unsupported = nn.Linear(d, 1)
        self.parent_query = nn.Linear(d, d, bias=False)
        self.parent_key = nn.Linear(d, d, bias=False)
        self.root_key = nn.Parameter(torch.zeros(d))
        self.relations = DirectedRelationRefiner(cfg)
        nn.init.normal_(self.start, std=0.02)
        nn.init.normal_(self.root_key, std=0.02)

    @staticmethod
    def _depth_of(i: int, parents: Sequence[int]) -> int:
        d = 0
        q = int(i)
        seen = set()
        while q >= 0 and parents[q] >= 0:
            if q in seen:
                raise ValueError("GEPPETTO_TEMPORARY_PARENT_CYCLE")
            seen.add(q); q = int(parents[q]); d += 1
        return d

    def _history_token(
        self,
        i: int,
        positions: torch.Tensor,
        parents: Sequence[int],
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        p = int(parents[i])
        cur = positions[:, i]
        if p < 0:
            pp = cur
            root = cur.new_ones((cur.shape[0], 1))
        else:
            pp = positions[:, p]
            root = cur.new_zeros((cur.shape[0], 1))
        rel = cur - pp
        length = torch.linalg.norm(rel, dim=-1, keepdim=True)
        depth = cur.new_full((cur.shape[0], 1), float(self._depth_of(i, parents)))
        depth = depth / max(1.0, float(len(parents)))
        x = torch.cat([cur, pp, rel, length, depth, root, sigma[:, i]], dim=-1)
        return self.history(x)

    def _step(
        self,
        pooled: torch.Tensor,
        states: list[torch.Tensor],
        positions: torch.Tensor,
        parents: Sequence[int],
        sigma: torch.Tensor,
    ):
        B, D = pooled.shape
        if positions.shape[1] == 0:
            prev = self.start[None].expand(B, -1)
            history_tokens = pooled.new_zeros((B, 0, D))
        else:
            history_tokens = torch.stack([
                self._history_token(i, positions, parents, sigma)
                for i in range(positions.shape[1])
            ], dim=1)
            prev = history_tokens[:, -1]
        x = torch.cat([pooled, prev], dim=-1)
        for li, cell in enumerate(self.decoder):
            states[li] = cell(x, states[li])
            x = torch.cat([pooled, states[li]], dim=-1)
        h = states[-1]
        pos = torch.tanh(self.pos(h))
        log_sigma = self.log_sigma(h).clamp(-8.0, 2.0)
        root_logit = self.root(h).squeeze(-1)
        stop_logit = self.stop(h).squeeze(-1)
        unsupported_logit = self.unsupported(h).squeeze(-1)
        q = self.parent_query(h)
        root_parent = (q * self.root_key[None]).sum(-1, keepdim=True) / math.sqrt(D)
        if history_tokens.shape[1]:
            keys = self.parent_key(history_tokens)
            old = torch.einsum("bd,bkd->bk", q, keys) / math.sqrt(D)
            parent_logits = torch.cat([root_parent, old], dim=-1)
        else:
            parent_logits = root_parent
        return h, pos, log_sigma, root_logit, stop_logit, unsupported_logit, parent_logits

    def teacher_forced(
        self,
        cond: GeppettoConditioningBatch,
        positions_world: torch.Tensor,
        parent_local: Sequence[int],
    ) -> dict[str, torch.Tensor]:
        """Training-only serialization. No oracle count exists in final inference."""
        _, pooled = self.encoder(cond)
        target = (positions_world[None].to(pooled.device) - cond.center) / cond.scale
        B, K, _ = target.shape
        if B != 1 or len(parent_local) != K:
            raise ValueError("GEPPETTO_BAD_TEACHER_SEQUENCE")
        states = [pooled.new_zeros((B, self.cfg.model_dim)) for _ in self.decoder]
        previous = target[:, :0]
        sigma_hist = target[:, :0]
        step_out = []
        for i in range(K + 1):
            out = self._step(pooled, states, previous, parent_local[:i], sigma_hist)
            step_out.append(out)
            if i < K:
                previous = target[:, : i + 1]
                # Teacher history never imports source uncertainty; zero is neutral.
                sigma_hist = torch.zeros_like(previous)
        hist = torch.stack([
            self._history_token(i, target, parent_local, torch.zeros_like(target))
            for i in range(K)
        ], dim=1)
        pair, refined_root, _ = self.relations(hist, target)
        return {
            "steps": step_out,
            "target_positions_norm": target[0],
            "pair_logits": pair[0],
            "refined_root_logits": refined_root[0],
        }

    @torch.no_grad()
    def propose(
        self,
        surface: RiggingSurfaceIR,
        *,
        device: torch.device | str = "cpu",
    ) -> SkeletonProposalIR:
        self.eval(); device = torch.device(device)
        cond = build_geppetto_conditioning(
            surface, device=device,
            cfg=GeppettoConditioningConfig(local_k=self.cfg.knn_k),
        )
        _, pooled = self.encoder(cond)
        B = pooled.shape[0]
        states = [pooled.new_zeros((B, self.cfg.model_dim)) for _ in self.decoder]
        positions = pooled.new_zeros((B, 0, 3))
        sigma = pooled.new_zeros((B, 0, 3))
        parents: list[int] = []
        unsupported_probs: list[float] = []
        stop_probs: list[float] = []
        # Resource guard only, not a product cardinality hyperparameter.
        resource_budget = len(surface.surface_nodes)
        stopped = False
        for i in range(resource_budget + 1):
            h, p, ls, r, st, un, parent_logits = self._step(
                pooled, states, positions, parents, sigma
            )
            p_stop = float(torch.sigmoid(st[0]).item())
            p_unsupported = float(torch.sigmoid(un[0]).item())
            if i == 0 and p_unsupported >= self.cfg.unsupported_threshold:
                raise RuntimeError("GEPPETTO_UNSUPPORTED_INPUT")
            if i > 0 and p_stop >= self.cfg.stop_threshold:
                stopped = True; break
            if i >= resource_budget:
                break
            parent = int(torch.argmax(parent_logits[0]).item()) - 1
            positions = torch.cat([positions, p[:, None]], dim=1)
            sigma = torch.cat([sigma, torch.exp(ls)[:, None]], dim=1)
            parents.append(parent)
            stop_probs.append(p_stop); unsupported_probs.append(p_unsupported)
        if not stopped:
            raise RuntimeError("GEPPETTO_LEARNED_STOP_NOT_REACHED")
        if not parents:
            raise RuntimeError("GEPPETTO_EMPTY_PROPOSAL")

        hist = torch.stack([
            self._history_token(i, positions, parents, sigma)
            for i in range(len(parents))
        ], dim=1)
        pair_logits, refined_root, _ = self.relations(hist, positions)
        pair_prob = torch.sigmoid(pair_logits[0])
        root_prob = torch.sigmoid(refined_root[0])
        world = positions * cond.scale + cond.center
        joints = []
        for i in range(len(parents)):
            d2 = ((cond.points_world[0] - world[0, i]) ** 2).sum(-1)
            k = min(self.cfg.support_surface_k, len(cond.surface_ids))
            near = torch.topk(d2, k=k, largest=False).indices.tolist()
            sig_world = sigma[0, i] * cond.scale[0, 0, 0]
            conf = math.exp(-min(20.0, float(torch.linalg.norm(sig_world))))
            conf *= 1.0 - unsupported_probs[i]
            joints.append(SkeletonProposalJoint(
                proposal_id=f"GR6:{i:05d}",
                position=tuple(map(float, world[0, i].tolist())),
                root_score=float(root_prob[i]),
                confidence=float(max(0.0, min(1.0, conf))),
                support_surface_ids=tuple(cond.surface_ids[j] for j in near),
                metadata={
                    "anonymous_control": True,
                    "temporary_generation_index": i,
                    "temporary_generation_parent": parents[i],
                    "position_sigma_world": tuple(map(float, sig_world.tolist())),
                    "learned_stop_probability_before_emit": stop_probs[i],
                    "canonical_authority": False,
                },
            ))
        edges = []
        K = len(joints)
        for parent in range(K):
            for child in range(K):
                if parent == child:
                    continue
                score = float(pair_prob[parent, child])
                edges.append(SkeletonProposalEdge(
                    edge_id=f"GR6E:{parent:05d}>{child:05d}",
                    parent_proposal_id=joints[parent].proposal_id,
                    child_proposal_id=joints[child].proposal_id,
                    score=score,
                    confidence=float(abs(score - 0.5) * 2.0),
                    reason="GEPPETTO_R6_DIRECTED_PARENT_EVIDENCE",
                    metadata={"canonical_authority": False},
                ))
        return SkeletonProposalIR(
            joints=tuple(joints), edges=tuple(edges),
            surface_binding_hash=surface.geometry_lineage_hash,
            model_provenance="RealSaS.GeppettoR6.full_product_single_specimen_fit",
            metadata={
                "variable_cardinality": True,
                "oracle_count_input": False,
                "fixed_product_control_cap": False,
                "learned_stop": True,
                "topology_in_generation_state": True,
                "proposal_ids_are_ephemeral": True,
                "compiler_owns_root_tree_and_canonical_ids": True,
                "generalization_claim": False,
            },
        )
