from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from models.geppetto.v2.geppetto_candidate_v2 import GeppettoCandidateConfigV2, SurfaceSetEncoderV2


def _timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """Sinusoidal diffusion timestep embedding."""
    if t.ndim != 1:
        raise ValueError("t must be [B]")
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(half, device=t.device, dtype=torch.float32)
        / max(half, 1)
    )
    args = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


def _cosine_alpha_bar(train_steps: int, s: float = 0.008) -> torch.Tensor:
    if train_steps < 2:
        raise ValueError("train_steps must be >=2")
    x = torch.linspace(0, train_steps, train_steps + 1, dtype=torch.float64)
    f = torch.cos(((x / train_steps) + s) / (1 + s) * math.pi * 0.5).square()
    f = f / f[0]
    # index t means cumulative alpha after t+1 forward transitions.
    return f[1:].clamp(1e-8, 1.0).float()


class _FiLMResBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.mod = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 3))
        self.ff = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift, scale, gate = self.mod(cond).chunk(3, dim=-1)
        h = self.norm(x) * (1.0 + scale) + shift
        return x + gate * self.ff(h)


class ConditionalJointDiffusionV1(nn.Module):
    """Small 3D conditional diffusion head adapted from RigAnything's mechanism.

    This is a research challenger, not product authority.  It uses a cosine forward
    schedule and an epsilon-prediction objective.  Inference uses deterministic DDIM
    stepping over a configurable subset of the training schedule.
    """

    def __init__(
        self,
        context_dim: int,
        hidden_dim: int = 384,
        depth: int = 3,
        train_steps: int = 1000,
        sample_steps: int = 50,
        position_clip: float = 1.25,
    ):
        super().__init__()
        if min(context_dim, hidden_dim, depth, train_steps, sample_steps) <= 0:
            raise ValueError("invalid diffusion configuration")
        if sample_steps > train_steps:
            raise ValueError("sample_steps cannot exceed train_steps")
        self.context_dim = int(context_dim)
        self.hidden_dim = int(hidden_dim)
        self.train_steps = int(train_steps)
        self.sample_steps = int(sample_steps)
        self.position_clip = float(position_clip)

        self.input_proj = nn.Linear(3, hidden_dim)
        self.time_proj = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.context_proj = nn.Linear(context_dim, hidden_dim)
        self.blocks = nn.ModuleList([_FiLMResBlock(hidden_dim) for _ in range(depth)])
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out = nn.Linear(hidden_dim, 3)

        alpha_bar = _cosine_alpha_bar(self.train_steps)
        self.register_buffer("alpha_bar", alpha_bar, persistent=True)

    def _predict_eps(self, x_t: torch.Tensor, t: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        if x_t.shape[-1] != 3 or context.ndim != 2 or x_t.shape[0] != context.shape[0]:
            raise ValueError("diffusion input/context shape mismatch")
        if t.shape != (x_t.shape[0],):
            raise ValueError("diffusion timestep shape mismatch")
        h = self.input_proj(x_t)
        te = self.time_proj(_timestep_embedding(t, self.hidden_dim))
        ce = self.context_proj(context)
        cond = te + ce
        for block in self.blocks:
            h = block(h, cond)
        return self.out(self.out_norm(h))

    def loss(
        self,
        target: torch.Tensor,
        context: torch.Tensor,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        if target.ndim != 2 or target.shape[-1] != 3:
            raise ValueError("diffusion target must be [B,3]")
        b = target.shape[0]
        t = torch.randint(0, self.train_steps, (b,), device=target.device, generator=generator)
        noise = torch.randn(target.shape, device=target.device, dtype=target.dtype, generator=generator)
        abar = self.alpha_bar[t].to(target.dtype)[:, None]
        x_t = abar.sqrt() * target + (1.0 - abar).sqrt() * noise
        pred = self._predict_eps(x_t, t, context)
        return F.mse_loss(pred, noise)

    @torch.no_grad()
    def sample(
        self,
        context: torch.Tensor,
        *,
        generator: Optional[torch.Generator] = None,
        sample_steps: Optional[int] = None,
    ) -> torch.Tensor:
        b = context.shape[0]
        steps = int(self.sample_steps if sample_steps is None else sample_steps)
        if steps < 1 or steps > self.train_steps:
            raise ValueError("invalid diffusion sample_steps")
        x = torch.randn((b, 3), device=context.device, dtype=context.dtype, generator=generator)
        schedule = torch.linspace(self.train_steps - 1, 0, steps, device=context.device).round().long()
        schedule = torch.unique_consecutive(schedule)
        for i, t_scalar in enumerate(schedule):
            t = torch.full((b,), int(t_scalar.item()), device=context.device, dtype=torch.long)
            abar_t = self.alpha_bar[t].to(context.dtype)[:, None]
            eps = self._predict_eps(x, t, context)
            x0 = (x - (1.0 - abar_t).sqrt() * eps) / abar_t.sqrt().clamp_min(1e-6)
            x0 = x0.clamp(-self.position_clip, self.position_clip)
            if i == len(schedule) - 1:
                x = x0
                break
            prev_t = int(schedule[i + 1].item())
            abar_prev = self.alpha_bar[prev_t].to(context.dtype)
            # Deterministic DDIM (eta=0).  The initial noise still makes the model
            # probabilistic, but repeated denoising does not inject extra noise.
            x = abar_prev.sqrt() * x0 + (1.0 - abar_prev).sqrt() * eps
        return x


@dataclass(frozen=True)
class RigAnythingMechanismConfigV1:
    base: GeppettoCandidateConfigV2 = GeppettoCandidateConfigV2()
    diffusion_hidden: int = 384
    diffusion_depth: int = 3
    diffusion_train_steps: int = 1000
    diffusion_sample_steps: int = 50
    max_sequence_steps: int = 256
    architecture_id: str = "RealSaS.GeppettoChallenger.RigAnythingMechanisms.v1"

    def validate(self) -> None:
        self.base.validate()
        if min(
            self.diffusion_hidden,
            self.diffusion_depth,
            self.diffusion_train_steps,
            self.diffusion_sample_steps,
            self.max_sequence_steps,
        ) <= 0:
            raise ValueError("invalid RigAnything challenger configuration")
        if self.diffusion_sample_steps > self.diffusion_train_steps:
            raise ValueError("diffusion sample horizon invalid")


@dataclass
class RigAnythingMechanismOutputV1:
    positions_normalized: torch.Tensor
    control_contexts: torch.Tensor
    parent_logits: torch.Tensor
    surface_attention: torch.Tensor
    diffusion_loss: Optional[torch.Tensor]
    teacher_forced: bool


class GeppettoRigAnythingMechanismChallengerV1(nn.Module):
    """Research-only Geppetto challenger carrying RigAnything's key mechanisms.

    Mechanisms intentionally isolated here instead of mutating canonical Geppetto V2:
      1) every autoregressive control step cross-attends to the full surface-token memory;
      2) the continuous 3D locus is modeled with a conditional diffusion objective;
      3) the current joint geometry and its parent geometry are fused into the token
         fed to subsequent autoregressive steps (teacher-forced during training,
         sampled/predicted during inference).

    The richer RealSaS 24D scene-first conditioning and deterministic Compiler authority
    are retained.  No raw teacher identity becomes product identity.
    """

    def __init__(self, config: RigAnythingMechanismConfigV1 = RigAnythingMechanismConfigV1()):
        super().__init__()
        config.validate()
        self.config = config
        d = config.base.model_dim
        self.encoder = SurfaceSetEncoderV2(config.base)
        self.start = nn.Parameter(torch.zeros(d))
        self.step_embed = nn.Embedding(config.max_sequence_steps, d)
        self.surface_cross_attn = nn.MultiheadAttention(d, config.base.attention_heads, batch_first=True)
        self.surface_fuse = nn.Sequential(nn.LayerNorm(d * 2), nn.Linear(d * 2, d), nn.GELU(), nn.Linear(d, d))
        self.recurrent = nn.GRUCell(d * 3, d)
        self.joint_token = nn.Sequential(nn.Linear(3, d), nn.GELU(), nn.Linear(d, d))
        self.feedback_fuse = nn.Sequential(nn.LayerNorm(d * 4), nn.Linear(d * 4, d * 2), nn.GELU(), nn.Linear(d * 2, d))
        self.parent_pair = nn.Sequential(nn.Linear(d * 2 + 4, d), nn.GELU(), nn.Linear(d, 1))
        self.diffusion = ConditionalJointDiffusionV1(
            d,
            hidden_dim=config.diffusion_hidden,
            depth=config.diffusion_depth,
            train_steps=config.diffusion_train_steps,
            sample_steps=config.diffusion_sample_steps,
            position_clip=config.base.position_scale,
        )

    def _attend_surface(
        self,
        state: torch.Tensor,
        memory: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q = state[:, None, :]
        ctx, attn = self.surface_cross_attn(
            q,
            memory,
            memory,
            key_padding_mask=~mask.bool(),
            need_weights=True,
            average_attn_weights=True,
        )
        fused = self.surface_fuse(torch.cat([state, ctx[:, 0]], dim=-1))
        return fused, attn[:, 0]

    def _parent_logits(
        self,
        current_context: torch.Tensor,
        current_position: torch.Tensor,
        previous_tokens: list[torch.Tensor],
        previous_positions: list[torch.Tensor],
    ) -> torch.Tensor:
        b, d = current_context.shape
        k = len(previous_tokens)
        if k == 0:
            return current_context.new_empty((b, 0))
        ptok = torch.stack(previous_tokens, dim=1)
        ppos = torch.stack(previous_positions, dim=1)
        cur = current_context[:, None, :].expand(b, k, d)
        cpos = current_position[:, None, :].expand(b, k, 3)
        delta = ppos - cpos
        dist = torch.linalg.norm(delta, dim=-1, keepdim=True)
        return self.parent_pair(torch.cat([cur, ptok, delta, dist], dim=-1)).squeeze(-1)

    def _feedback_token(
        self,
        context: torch.Tensor,
        joint_position: torch.Tensor,
        parent_position: torch.Tensor,
        step: int,
    ) -> torch.Tensor:
        idx = torch.full((context.shape[0],), step, device=context.device, dtype=torch.long)
        step_emb = self.step_embed(idx)
        return self.feedback_fuse(
            torch.cat(
                [context, self.joint_token(joint_position), self.joint_token(parent_position), step_emb],
                dim=-1,
            )
        )

    def forward(
        self,
        features: torch.Tensor,
        positions_normalized: torch.Tensor,
        valid_mask: torch.Tensor,
        *,
        decode_steps: int,
        teacher_positions: Optional[torch.Tensor] = None,
        teacher_parent_indices: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        sample_steps: Optional[int] = None,
    ) -> RigAnythingMechanismOutputV1:
        if decode_steps < 1 or decode_steps > self.config.max_sequence_steps:
            raise ValueError("decode_steps outside challenger sequence guard")
        if features.ndim != 3 or features.shape[-1] != self.config.base.surface_feature_dim:
            raise ValueError("challenger feature shape mismatch")
        if positions_normalized.shape != (*features.shape[:2], 3) or valid_mask.shape != features.shape[:2]:
            raise ValueError("challenger surface shape mismatch")
        if int(valid_mask.sum(1).min().item()) < decode_steps:
            raise ValueError("decode_steps exceeds admitted surface support")
        teacher_forced = teacher_positions is not None
        if teacher_forced:
            if teacher_parent_indices is None:
                raise ValueError("teacher_parent_indices required with teacher_positions")
            if teacher_positions.shape != (features.shape[0], decode_steps, 3):
                raise ValueError("teacher_positions shape mismatch")
            if teacher_parent_indices.shape != (features.shape[0], decode_steps):
                raise ValueError("teacher_parent_indices shape mismatch")

        memory, pooled = self.encoder(features, positions_normalized, valid_mask.bool())
        b, _, d = memory.shape
        state = pooled.new_zeros((b, d))
        prev_token = self.start[None].expand(b, -1)
        previous_tokens: list[torch.Tensor] = []
        previous_positions: list[torch.Tensor] = []
        contexts = []
        positions = []
        parent_rows = []
        attn_rows = []
        diff_losses = []

        for step in range(decode_steps):
            step_id = torch.full((b,), step, device=features.device, dtype=torch.long)
            step_emb = self.step_embed(step_id)
            # pooled remains a global shape summary, but no longer the sole shape path:
            # the recurrent query receives a fresh full-memory cross-attention context.
            provisional = self.recurrent(torch.cat([pooled, prev_token, step_emb], dim=-1), state)
            context, attn = self._attend_surface(provisional, memory, valid_mask)
            state = context

            if teacher_forced:
                current_position = teacher_positions[:, step]
                diff_losses.append(self.diffusion.loss(current_position, context, generator=generator))
            else:
                current_position = self.diffusion.sample(context, generator=generator, sample_steps=sample_steps)

            logits = self._parent_logits(context, current_position, previous_tokens, previous_positions)
            padded = context.new_full((b, decode_steps), -1e4)
            if logits.numel():
                padded[:, :step] = logits
            parent_rows.append(padded)

            if teacher_forced:
                parent_idx = teacher_parent_indices[:, step].long()
                if step == 0:
                    if (parent_idx >= 0).any():
                        raise ValueError("first teacher-forced step must be a root")
                    parent_position = torch.zeros_like(current_position)
                else:
                    if ((parent_idx < 0) | (parent_idx >= step)).any():
                        raise ValueError("teacher parent must precede child")
                    prior = torch.stack(previous_positions, dim=1)
                    gather = parent_idx[:, None, None].expand(b, 1, 3)
                    parent_position = torch.gather(prior, 1, gather).squeeze(1)
            else:
                if step == 0:
                    parent_position = torch.zeros_like(current_position)
                else:
                    parent_idx = torch.argsort(logits, dim=-1, descending=True, stable=True)[:, 0]
                    prior = torch.stack(previous_positions, dim=1)
                    gather = parent_idx[:, None, None].expand(b, 1, 3)
                    parent_position = torch.gather(prior, 1, gather).squeeze(1)

            token = self._feedback_token(context, current_position, parent_position, step)
            previous_tokens.append(token)
            previous_positions.append(current_position)
            prev_token = token
            contexts.append(context)
            positions.append(current_position)
            attn_rows.append(attn)

        diffusion_loss = torch.stack(diff_losses).mean() if diff_losses else None
        return RigAnythingMechanismOutputV1(
            positions_normalized=torch.stack(positions, dim=1),
            control_contexts=torch.stack(contexts, dim=1),
            parent_logits=torch.stack(parent_rows, dim=1),
            surface_attention=torch.stack(attn_rows, dim=1),
            diffusion_loss=diffusion_loss,
            teacher_forced=teacher_forced,
        )


def randomize_bfs_depth_order_v1(
    positions: torch.Tensor,
    parent_indices: torch.Tensor,
    *,
    generator: Optional[torch.Generator] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Randomize equivalent same-depth ordering while preserving parent-before-child.

    This is the RealSaS training translation of RigAnything's sibling/BFS-order
    ambiguity handling.  It operates on one tree [J,3]/[J] and returns
    (positions_reordered, parents_reindexed, permutation_new_to_old).
    """
    if positions.ndim != 2 or positions.shape[-1] != 3 or parent_indices.shape != (positions.shape[0],):
        raise ValueError("tree shape mismatch")
    j = positions.shape[0]
    parents = parent_indices.detach().cpu().tolist()
    depths = [-1] * j

    def depth_of(i: int, stack: set[int]) -> int:
        if depths[i] >= 0:
            return depths[i]
        if i in stack:
            raise ValueError("parent cycle")
        p = int(parents[i])
        if p < 0:
            depths[i] = 0
            return 0
        if p >= j:
            raise ValueError("parent index out of range")
        depths[i] = depth_of(p, stack | {i}) + 1
        return depths[i]

    for i in range(j):
        depth_of(i, set())
    groups: dict[int, list[int]] = {}
    for i, dep in enumerate(depths):
        groups.setdefault(dep, []).append(i)
    permutation = []
    for dep in sorted(groups):
        ids = torch.tensor(groups[dep], dtype=torch.long)
        if len(ids) > 1:
            order = torch.randperm(len(ids), generator=generator)
            ids = ids[order]
        permutation.extend(ids.tolist())
    perm = torch.tensor(permutation, device=positions.device, dtype=torch.long)
    inverse = torch.empty(j, device=positions.device, dtype=torch.long)
    inverse[perm] = torch.arange(j, device=positions.device)
    old_parent = parent_indices.to(device=positions.device, dtype=torch.long)
    new_parent = torch.full((j,), -1, device=positions.device, dtype=torch.long)
    for new_i, old_i in enumerate(perm.tolist()):
        p = int(old_parent[old_i].item())
        if p >= 0:
            new_parent[new_i] = inverse[p]
            if int(new_parent[new_i].item()) >= new_i:
                raise ValueError("randomized BFS order violated parent-before-child")
    return positions[perm], new_parent, perm
