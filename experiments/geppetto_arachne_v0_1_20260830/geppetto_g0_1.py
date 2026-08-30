from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class GeppettoG01Config:
    """Development scaffold configuration; values are NOT architecture-sealed."""

    token_dim: int = 27
    model_dim: int = 128
    knn_k: int = 16
    encoder_layers: int = 3
    decoder_layers: int = 2
    max_controls: int = 64
    dropout: float = 0.0


class LocalGeometryAttention(nn.Module):
    """Permutation-equivariant KNN attention with deterministic relative geometry."""

    def __init__(self, dim: int, k: int, dropout: float = 0.0):
        super().__init__()
        self.k = int(k)
        self.q = nn.Linear(dim, dim, bias=False)
        self.kv = nn.Linear(dim, dim * 2, bias=False)
        self.rel = nn.Sequential(nn.Linear(4, dim), nn.GELU(), nn.Linear(dim, dim))
        self.proj = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 4, dim)
        )
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _gather(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        # x [B,N,D], idx [B,N,K] -> [B,N,K,D]
        b = torch.arange(x.shape[0], device=x.device)[:, None, None]
        return x[b, idx]

    def forward(self, h: torch.Tensor, xyz: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if h.ndim != 3 or xyz.shape[:2] != h.shape[:2] or xyz.shape[-1] != 3:
            raise ValueError("bad h/xyz shape")
        B, N, D = h.shape
        if mask.shape != (B, N):
            raise ValueError("bad token mask")
        if N == 0:
            return h
        # Pair distances are only an implementation scaffold. Invalid tokens can never be neighbors.
        d2 = torch.cdist(xyz.float(), xyz.float(), p=2).square()
        inf = torch.tensor(float("inf"), device=d2.device, dtype=d2.dtype)
        d2 = d2.masked_fill(~mask[:, None, :], inf)
        # Include self where valid, which avoids undefined all-inf neighborhoods for one-token sets.
        k = min(max(1, self.k), N)
        idx = torch.topk(d2, k=k, dim=-1, largest=False, sorted=True).indices

        hn = self._gather(h, idx)
        xn = self._gather(xyz, idx)
        delta = xn - xyz[:, :, None, :]
        dist = torch.linalg.norm(delta, dim=-1, keepdim=True)
        r = self.rel(torch.cat([delta, dist], dim=-1))

        q = self.q(h)[:, :, None, :]
        kk, vv = self.kv(hn).chunk(2, dim=-1)
        kk = kk + r
        score = (q * kk).sum(dim=-1) / (D ** 0.5)
        neigh_valid = self._gather(mask[..., None].float(), idx).squeeze(-1).bool()
        score = score.masked_fill(~neigh_valid, -1e9)
        attn = torch.softmax(score, dim=-1)
        msg = (attn[..., None] * (vv + r)).sum(dim=2)
        h = self.norm1(h + self.dropout(self.proj(msg)))
        h = self.norm2(h + self.dropout(self.ff(h)))
        return h * mask[..., None].to(h.dtype)


class GeppettoTokenEncoder(nn.Module):
    def __init__(self, cfg: GeppettoG01Config):
        super().__init__()
        self.cfg = cfg
        self.stem = nn.Sequential(
            nn.Linear(cfg.token_dim, cfg.model_dim), nn.GELU(), nn.Linear(cfg.model_dim, cfg.model_dim)
        )
        self.layers = nn.ModuleList(
            [LocalGeometryAttention(cfg.model_dim, cfg.knn_k, cfg.dropout) for _ in range(cfg.encoder_layers)]
        )
        self.out_norm = nn.LayerNorm(cfg.model_dim)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if tokens.ndim != 3 or tokens.shape[-1] != self.cfg.token_dim:
            raise ValueError(f"tokens must be [B,N,{self.cfg.token_dim}]")
        if mask.shape != tokens.shape[:2]:
            raise ValueError("mask shape mismatch")
        xyz = tokens[..., :3]
        h = self.stem(tokens) * mask[..., None].to(tokens.dtype)
        for layer in self.layers:
            h = layer(h, xyz, mask)
        h = self.out_norm(h) * mask[..., None].to(h.dtype)
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1).to(h.dtype)
        pooled = h.sum(dim=1) / denom
        return h, pooled


class GeppettoG01(nn.Module):
    """G0.1 anonymous-control proposal scaffold.

    This module does NOT own canonical IDs, root selection, or final topology.
    It predicts anonymous control geometry plus root/directed-parent evidence for
    the existing Compiler to qualify globally.
    """

    def __init__(self, cfg: GeppettoG01Config = GeppettoG01Config()):
        super().__init__()
        self.cfg = cfg
        d = cfg.model_dim
        self.encoder = GeppettoTokenEncoder(cfg)
        self.start = nn.Parameter(torch.zeros(d))
        self.pos_embed = nn.Sequential(nn.Linear(3, d), nn.GELU(), nn.Linear(d, d))
        self.decoder_cells = nn.ModuleList([nn.GRUCell(d * 2, d) for _ in range(cfg.decoder_layers)])
        self.position_head = nn.Linear(d, 3)
        self.log_sigma_head = nn.Linear(d, 3)
        self.exist_head = nn.Linear(d, 1)
        self.root_head = nn.Linear(d, 1)
        self.parent_head = nn.Sequential(nn.Linear(d * 2 + 4, d), nn.GELU(), nn.Linear(d, 1))

    def _decode(
        self,
        pooled: torch.Tensor,
        *,
        steps: int,
        teacher_positions: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, D = pooled.shape
        if steps < 1 or steps > self.cfg.max_controls:
            raise ValueError(f"steps must be 1..{self.cfg.max_controls}")
        if teacher_positions is not None and teacher_positions.shape != (B, steps, 3):
            raise ValueError("teacher_positions shape mismatch")
        states = [pooled.new_zeros((B, D)) for _ in self.decoder_cells]
        prev_pos = pooled.new_zeros((B, 3))
        out_h, out_p, out_s, out_e, out_r = [], [], [], [], []
        for t in range(steps):
            prev = self.start[None].expand(B, -1) if t == 0 else self.pos_embed(prev_pos)
            x = torch.cat([pooled, prev], dim=-1)
            for li, cell in enumerate(self.decoder_cells):
                states[li] = cell(x, states[li])
                x = torch.cat([pooled, states[li]], dim=-1)
            h = states[-1]
            p = self.position_head(h)
            log_sigma = self.log_sigma_head(h).clamp(-8.0, 4.0)
            exist = self.exist_head(h).squeeze(-1)
            root = self.root_head(h).squeeze(-1)
            out_h.append(h); out_p.append(p); out_s.append(log_sigma); out_e.append(exist); out_r.append(root)
            prev_pos = teacher_positions[:, t] if teacher_positions is not None else p
        return (
            torch.stack(out_h, 1), torch.stack(out_p, 1), torch.stack(out_s, 1),
            torch.stack(out_e, 1), torch.stack(out_r, 1),
        )

    def _parent_logits(self, h: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        B, K, D = h.shape
        hp = h[:, :, None, :].expand(B, K, K, D)
        hc = h[:, None, :, :].expand(B, K, K, D)
        delta = pos[:, None, :, :] - pos[:, :, None, :]  # parent i -> child j
        dist = torch.linalg.norm(delta, dim=-1, keepdim=True)
        pair = torch.cat([hp, hc, delta, dist], dim=-1)
        logits = self.parent_head(pair).squeeze(-1)
        eye = torch.eye(K, device=logits.device, dtype=torch.bool)[None]
        return logits.masked_fill(eye, -1e9)

    def forward(
        self,
        tokens: torch.Tensor,
        token_mask: torch.Tensor,
        *,
        steps: int | None = None,
        teacher_positions: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        _, pooled = self.encoder(tokens, token_mask.bool())
        if teacher_positions is not None:
            steps = int(teacher_positions.shape[1])
        elif steps is None:
            steps = self.cfg.max_controls
        h, pos, log_sigma, exist_logits, root_logits = self._decode(
            pooled, steps=int(steps), teacher_positions=teacher_positions
        )
        parent_logits = self._parent_logits(h, pos)
        return {
            "control_states": h,
            "positions": pos,
            "position_log_sigma": log_sigma,
            "exist_logits": exist_logits,
            "root_logits": root_logits,
            "parent_logits": parent_logits,
        }

    @torch.no_grad()
    def proposal_from_output(
        self,
        output: dict[str, torch.Tensor],
        *,
        surface_binding_hash: str,
        batch_index: int = 0,
        active_count: int | None = None,
        model_provenance: str = "GEPPETTO_G0_1_UNSEALED",
    ) -> Any:
        """Convert one batch item to current Compiler `SkeletonProposalIR`.

        Import is intentionally runtime-local so the neural module can be unit
        tested outside the compiler package. `active_count` is an evaluation /
        generation decision, never a canonical product count.
        """
        from realsas_compiler_core.types import (
            SkeletonProposalJoint, SkeletonProposalEdge, SkeletonProposalIR,
        )

        pos = output["positions"][batch_index].detach().cpu()
        sig = output["position_log_sigma"][batch_index].detach().cpu().exp()
        ex = output["exist_logits"][batch_index].detach().cpu().sigmoid()
        root = output["root_logits"][batch_index].detach().cpu().sigmoid()
        edge = output["parent_logits"][batch_index].detach().cpu().sigmoid()
        K = int(pos.shape[0] if active_count is None else active_count)
        if K < 1 or K > pos.shape[0]:
            raise ValueError("bad active_count")
        joints = []
        for i in range(K):
            confidence = float((ex[i] * torch.exp(-sig[i].mean())).clamp(0.0, 1.0))
            joints.append(SkeletonProposalJoint(
                proposal_id=f"G01P:{i:04d}",
                position=tuple(map(float, pos[i].tolist())),
                root_score=float(root[i]),
                confidence=confidence,
                support_surface_ids=(),
                metadata={
                    "anonymous_control": True,
                    "position_sigma": tuple(map(float, sig[i].tolist())),
                    "exist_probability": float(ex[i]),
                    "canonical_authority": False,
                },
            ))
        edges = []
        for parent in range(K):
            for child in range(K):
                if parent == child:
                    continue
                score = float(edge[parent, child])
                edges.append(SkeletonProposalEdge(
                    edge_id=f"G01E:{parent:04d}>{child:04d}",
                    parent_proposal_id=f"G01P:{parent:04d}",
                    child_proposal_id=f"G01P:{child:04d}",
                    score=score,
                    confidence=max(score, 1.0 - score),
                    hard_required=False,
                    hard_forbidden=False,
                    reason="G0.1 directed parent evidence",
                    metadata={"canonical_authority": False},
                ))
        return SkeletonProposalIR(
            joints=tuple(joints),
            edges=tuple(edges),
            surface_binding_hash=surface_binding_hash,
            model_provenance=model_provenance,
            metadata={
                "architecture": "GEPPETTO_G0_1",
                "proposal_ids_are_ephemeral": True,
                "compiler_owns_root_tree_and_canonical_ids": True,
                "sealed": False,
            },
        )
