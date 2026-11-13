from __future__ import annotations

"""Reference-strength, shipping-faithful Geppetto research candidate.

This module is experiment-only. It consumes the factorized tensors produced by
``rigging_surface_tensorization_v1`` rather than the legacy 24D summary.

Scientific ownership:
- shipping-available RiggingSurfaceIR evidence is learner input;
- teacher skeleton geometry/topology is training/evaluation target only;
- generation recurrence is prediction-only (no teacher-forced geometry/parent);
- final directed parent evidence is emitted all-pairs for Compiler qualification;
- generation index is never canonical identity.
"""

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from typing import Optional

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from compiler.realsas_compiler_core.types import (
    SkeletonProposalEdge,
    SkeletonProposalIR,
    SkeletonProposalJoint,
)
from experiments.geppetto_reference_strength_fullstack_v1.rigging_surface_tensorization_v1 import (
    RiggingSurfaceTensorV1,
)


def _hash(payload: object) -> str:
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def _timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    if t.ndim != 1:
        raise ValueError("diffusion timestep must be [B]")
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
        raise ValueError("diffusion_train_steps must be >=2")
    x = torch.linspace(0, train_steps, train_steps + 1, dtype=torch.float64)
    f = torch.cos(((x / train_steps) + s) / (1 + s) * math.pi * 0.5).square()
    f = f / f[0]
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
        return x + torch.tanh(gate) * self.ff(h)


class ConditionalResidualDiffusionV1(nn.Module):
    """3D residual diffusion around a learned coarse locus.

    During training the target residual is detached from the coarse prediction
    before entering this objective. This prevents the diffusion loss from
    turning teacher geometry into a recurrent-state shortcut.
    """

    def __init__(
        self,
        context_dim: int,
        *,
        hidden_dim: int,
        depth: int,
        train_steps: int,
        sample_steps: int,
        residual_clip: float,
    ):
        super().__init__()
        if min(context_dim, hidden_dim, depth, train_steps, sample_steps) <= 0:
            raise ValueError("invalid diffusion configuration")
        if sample_steps > train_steps:
            raise ValueError("diffusion sample_steps exceeds train_steps")
        if residual_clip <= 0:
            raise ValueError("diffusion residual_clip must be positive")
        self.context_dim = int(context_dim)
        self.hidden_dim = int(hidden_dim)
        self.train_steps = int(train_steps)
        self.sample_steps = int(sample_steps)
        self.residual_clip = float(residual_clip)
        self.input_proj = nn.Linear(3, hidden_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.context_proj = nn.Linear(context_dim, hidden_dim)
        self.blocks = nn.ModuleList([_FiLMResBlock(hidden_dim) for _ in range(depth)])
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out = nn.Linear(hidden_dim, 3)
        self.register_buffer("alpha_bar", _cosine_alpha_bar(train_steps), persistent=True)

    def _predict_eps(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        if x_t.ndim != 2 or x_t.shape[-1] != 3:
            raise ValueError("diffusion x_t must be [B,3]")
        if context.ndim != 2 or context.shape[0] != x_t.shape[0]:
            raise ValueError("diffusion context shape mismatch")
        if t.shape != (x_t.shape[0],):
            raise ValueError("diffusion timestep shape mismatch")
        h = self.input_proj(x_t)
        cond = self.time_proj(_timestep_embedding(t, self.hidden_dim)) + self.context_proj(context)
        for block in self.blocks:
            h = block(h, cond)
        return self.out(self.out_norm(h))

    def loss(
        self,
        target_residual: torch.Tensor,
        context: torch.Tensor,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        if target_residual.ndim != 2 or target_residual.shape[-1] != 3:
            raise ValueError("diffusion residual target must be [B,3]")
        target = target_residual.clamp(-self.residual_clip, self.residual_clip)
        b = target.shape[0]
        t = torch.randint(
            0, self.train_steps, (b,), device=target.device, generator=generator
        )
        noise = torch.randn(
            target.shape,
            device=target.device,
            dtype=target.dtype,
            generator=generator,
        )
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
        schedule = (
            torch.linspace(self.train_steps - 1, 0, steps, device=context.device)
            .round()
            .long()
        )
        schedule = torch.unique_consecutive(schedule)
        for i, t_scalar in enumerate(schedule):
            t = torch.full(
                (b,), int(t_scalar.item()), device=context.device, dtype=torch.long
            )
            abar_t = self.alpha_bar[t].to(context.dtype)[:, None]
            eps = self._predict_eps(x, t, context)
            x0 = (x - (1.0 - abar_t).sqrt() * eps) / abar_t.sqrt().clamp_min(1e-6)
            x0 = x0.clamp(-self.residual_clip, self.residual_clip)
            if i == len(schedule) - 1:
                x = x0
                break
            prev_t = int(schedule[i + 1].item())
            abar_prev = self.alpha_bar[prev_t].to(context.dtype)
            x = abar_prev.sqrt() * x0 + (1.0 - abar_prev).sqrt() * eps
        return x.clamp(-self.residual_clip, self.residual_clip)


class _ExactRelationBlock(nn.Module):
    """Message passing only across deterministic GSA local relations."""

    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.msg = nn.Sequential(
            nn.Linear(dim * 2 + 8, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.gate = nn.Sequential(nn.Linear(dim * 2 + 8, dim), nn.Sigmoid())
        self.proj = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 3),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 3, dim),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(
        self,
        h: torch.Tensor,
        xyz: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        if h.ndim != 2 or xyz.shape != (h.shape[0], 3):
            raise ValueError("relation block node shape mismatch")
        if edge_index.ndim != 2 or edge_index.shape[-1] != 2:
            raise ValueError("edge_index must be [E,2]")
        if edge_attr.shape != (edge_index.shape[0], 4):
            raise ValueError("edge_attr must be [E,4]")
        if edge_index.numel() == 0:
            return self.norm2(h + self.ff(self.norm1(h)))

        a = edge_index[:, 0].long()
        b = edge_index[:, 1].long()
        if int(edge_index.min()) < 0 or int(edge_index.max()) >= h.shape[0]:
            raise ValueError("edge_index out of range")
        agg = torch.zeros_like(h)
        deg = torch.zeros((h.shape[0], 1), dtype=h.dtype, device=h.device)
        for src, dst in ((a, b), (b, a)):
            delta = xyz[src] - xyz[dst]
            dist = torch.linalg.norm(delta, dim=-1, keepdim=True)
            rel = torch.cat([delta, dist, edge_attr], dim=-1)
            pair = torch.cat([h[dst], h[src], rel], dim=-1)
            msg = self.msg(pair) * self.gate(pair)
            agg.index_add_(0, dst, msg)
            deg.index_add_(0, dst, torch.ones((len(dst), 1), dtype=h.dtype, device=h.device))
        agg = agg / deg.clamp_min(1.0)
        h = self.norm1(h + self.proj(agg))
        return self.norm2(h + self.ff(h))


@dataclass(frozen=True)
class GeppettoReferenceStrengthConfigV1:
    model_dim: int = 256
    view_dim: int = 64
    graph_layers: int = 3
    global_layers: int = 4
    causal_layers: int = 2
    attention_heads: int = 8
    feedforward_dim: int = 768
    dropout: float = 0.0
    support_topk: int = 8
    stop_probability: float = 0.5
    support_presence_probability: float = 0.2
    position_clip: float = 1.25
    diffusion_residual_clip: float = 0.50
    diffusion_hidden: int = 384
    diffusion_depth: int = 3
    diffusion_train_steps: int = 1000
    diffusion_sample_steps: int = 32
    edge_pair_chunk: int = 64
    architecture_id: str = "RealSaS.Geppetto.ReferenceStrength.DirectSurfaceCausalDiffusion.v1"

    def validate(self) -> None:
        ints = (
            self.model_dim,
            self.view_dim,
            self.graph_layers,
            self.global_layers,
            self.causal_layers,
            self.attention_heads,
            self.feedforward_dim,
            self.support_topk,
            self.diffusion_hidden,
            self.diffusion_depth,
            self.diffusion_train_steps,
            self.diffusion_sample_steps,
            self.edge_pair_chunk,
        )
        if min(ints) <= 0:
            raise ValueError("invalid reference-strength architecture cardinality")
        if self.model_dim % self.attention_heads:
            raise ValueError("model_dim must divide attention_heads")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError("dropout out of range")
        if not (0.0 < self.stop_probability < 1.0):
            raise ValueError("stop_probability out of range")
        if not (0.0 <= self.support_presence_probability < 1.0):
            raise ValueError("support_presence_probability out of range")
        if self.position_clip <= 0 or self.diffusion_residual_clip <= 0:
            raise ValueError("position clips must be positive")
        if self.diffusion_sample_steps > self.diffusion_train_steps:
            raise ValueError("diffusion sample horizon invalid")

    @property
    def config_hash(self) -> str:
        self.validate()
        return _hash(asdict(self))


class DirectRiggingSurfaceEncoderV1(nn.Module):
    """Structured encoder for all shipping-available RiggingSurfaceIR evidence."""

    GEOMETRY_DIM = 10
    VIEW_PAYLOAD_DIM = 4

    def __init__(self, cfg: GeppettoReferenceStrengthConfigV1):
        super().__init__()
        d = cfg.model_dim
        self.cfg = cfg
        self.geometry_stem = nn.Sequential(
            nn.Linear(self.GEOMETRY_DIM + 8, d),
            nn.GELU(),
            nn.Linear(d, d),
        )
        self.view_index = nn.Parameter(torch.zeros(8, cfg.view_dim))
        nn.init.normal_(self.view_index, std=0.02)
        self.view_mlp = nn.Sequential(
            nn.Linear(self.VIEW_PAYLOAD_DIM + cfg.view_dim, cfg.view_dim),
            nn.GELU(),
            nn.Linear(cfg.view_dim, cfg.view_dim),
        )
        self.view_fuse = nn.Sequential(
            nn.Linear(d + cfg.view_dim, d),
            nn.GELU(),
            nn.Linear(d, d),
        )
        self.graph = nn.ModuleList(
            [_ExactRelationBlock(d, cfg.dropout) for _ in range(cfg.graph_layers)]
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=cfg.attention_heads,
            dim_feedforward=cfg.feedforward_dim,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.global_encoder = nn.TransformerEncoder(
            layer, num_layers=cfg.global_layers, norm=nn.LayerNorm(d)
        )

    def forward(
        self,
        *,
        positions: torch.Tensor,
        normals: torch.Tensor,
        normal_valid: torch.Tensor,
        support: torch.Tensor,
        raster_xy: torch.Tensor,
        raster_valid: torch.Tensor,
        observed: torch.Tensor,
        completed: torch.Tensor,
        degree: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n = positions.shape[0]
        if positions.shape != (n, 3) or normals.shape != (n, 3):
            raise ValueError("surface geometry shape mismatch")
        if normal_valid.shape != (n,) or observed.shape != (n,) or completed.shape != (n,):
            raise ValueError("surface scalar mask shape mismatch")
        if support.shape != (n, 8) or raster_valid.shape != (n, 8):
            raise ValueError("surface view mask shape mismatch")
        if raster_xy.shape != (n, 8, 2):
            raise ValueError("surface raster shape mismatch")
        if degree.shape != (n,):
            raise ValueError("surface degree shape mismatch")
        if n < 1:
            raise ValueError("surface must contain nodes")

        support_f = support.to(positions.dtype)
        geom = torch.cat(
            [
                positions,
                normals,
                normal_valid[:, None].to(positions.dtype),
                observed[:, None].to(positions.dtype),
                completed[:, None].to(positions.dtype),
                torch.log1p(degree[:, None].to(positions.dtype)),
                support_f,
            ],
            dim=-1,
        )
        h = self.geometry_stem(geom)

        vembed = self.view_index[None].expand(n, -1, -1)
        view_payload = torch.cat(
            [
                raster_xy,
                raster_valid[..., None].to(positions.dtype),
                support[..., None].to(positions.dtype),
                vembed,
            ],
            dim=-1,
        )
        vh = self.view_mlp(view_payload)
        active = (support | raster_valid).to(vh.dtype)[..., None]
        denom = active.sum(dim=1).clamp_min(1.0)
        view_summary = (vh * active).sum(dim=1) / denom
        h = self.view_fuse(torch.cat([h, view_summary], dim=-1))

        for block in self.graph:
            h = block(h, positions, edge_index, edge_attr)
        memory = self.global_encoder(h[None])[0]
        pooled = memory.mean(dim=0, keepdim=True)
        return memory, pooled


@dataclass
class GeppettoReferenceStrengthRawOutputV1:
    coarse_positions_normalized: torch.Tensor
    positions_normalized: torch.Tensor
    control_states: torch.Tensor
    stop_logits: torch.Tensor
    existence_logits: torch.Tensor
    root_logits: torch.Tensor
    salience_logits: torch.Tensor
    support_presence_logits: torch.Tensor
    support_logits: torch.Tensor
    internal_parent_logits: torch.Tensor
    all_pair_parent_logits: torch.Tensor
    position_log_sigma: torch.Tensor
    surface_attention: torch.Tensor
    diffusion_loss: Optional[torch.Tensor]
    teacher_target_used: bool
    teacher_feedback_used: bool = False


class GeppettoReferenceStrengthCandidateV1(nn.Module):
    """Reference-strength candidate with shipping-consistent prediction-only recurrence."""

    def __init__(
        self,
        config: GeppettoReferenceStrengthConfigV1 = GeppettoReferenceStrengthConfigV1(),
    ):
        super().__init__()
        config.validate()
        self.config = config
        d = config.model_dim
        self.encoder = DirectRiggingSurfaceEncoderV1(config)
        self.start = nn.Parameter(torch.zeros(d))
        self.state_init = nn.Linear(d, d)
        self.step_proj = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.cross_attn = nn.MultiheadAttention(
            d, config.attention_heads, dropout=config.dropout, batch_first=True
        )
        self.cross_fuse = nn.Sequential(
            nn.LayerNorm(d * 2),
            nn.Linear(d * 2, d),
            nn.GELU(),
            nn.Linear(d, d),
        )
        self.cells = nn.ModuleList(
            [nn.GRUCell(d * 4, d) for _ in range(config.causal_layers)]
        )
        self.coarse_position = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, 3),
        )
        self.position_log_sigma = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 3))
        self.stop = nn.Linear(d, 1)
        self.existence = nn.Linear(d, 1)
        self.root = nn.Linear(d, 1)
        self.salience = nn.Linear(d, 1)
        self.support_presence = nn.Linear(d, 1)
        self.support_q = nn.Linear(d, d, bias=False)
        self.support_k = nn.Linear(d, d, bias=False)
        self.internal_parent = nn.Sequential(
            nn.Linear(d * 2 + 4, d),
            nn.GELU(),
            nn.Linear(d, 1),
        )
        self.final_parent = nn.Sequential(
            nn.Linear(d * 2 + 4, d),
            nn.GELU(),
            nn.Linear(d, 1),
        )
        self.feedback = nn.Sequential(
            nn.LayerNorm(d * 3 + 6),
            nn.Linear(d * 3 + 6, d * 2),
            nn.GELU(),
            nn.Linear(d * 2, d),
        )
        self.diffusion = ConditionalResidualDiffusionV1(
            d,
            hidden_dim=config.diffusion_hidden,
            depth=config.diffusion_depth,
            train_steps=config.diffusion_train_steps,
            sample_steps=config.diffusion_sample_steps,
            residual_clip=config.diffusion_residual_clip,
        )

    @staticmethod
    def _sequence_embedding(step: int, dim: int, device, dtype) -> torch.Tensor:
        half = dim // 2
        if half == 0:
            return torch.zeros((1, dim), device=device, dtype=dtype)
        freq = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=device, dtype=torch.float32)
            / max(half - 1, 1)
        )
        arg = float(step) * freq
        emb = torch.cat([torch.sin(arg), torch.cos(arg)], dim=0)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros(1, device=device)])
        return emb[None].to(dtype=dtype)

    def _surface_tensors(
        self,
        surface: RiggingSurfaceTensorV1,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        if surface.node_count < 1:
            raise ValueError("surface tensor has no nodes")
        f32 = torch.float32
        positions = torch.as_tensor(surface.positions_normalized, device=device, dtype=f32)
        normals = torch.as_tensor(surface.normals, device=device, dtype=f32)
        normal_valid = torch.as_tensor(surface.normal_valid, device=device, dtype=torch.bool)
        support = torch.as_tensor(surface.support, device=device, dtype=torch.bool)
        raster_xy = torch.as_tensor(surface.raster_xy_normalized, device=device, dtype=f32)
        raster_valid = torch.as_tensor(surface.raster_valid, device=device, dtype=torch.bool)
        observed = torch.as_tensor(surface.observed, device=device, dtype=torch.bool)
        completed = torch.as_tensor(surface.completed, device=device, dtype=torch.bool)
        degree = torch.as_tensor(surface.degree, device=device, dtype=torch.long)
        edge_index = torch.as_tensor(surface.edge_index, device=device, dtype=torch.long)
        edge_attr = torch.stack(
            [
                torch.as_tensor(surface.edge_score, device=device, dtype=f32),
                torch.as_tensor(surface.edge_distance_normalized, device=device, dtype=f32),
                torch.as_tensor(surface.edge_crosses_unknown, device=device, dtype=f32),
                torch.as_tensor(surface.edge_unknown_bridge, device=device, dtype=f32),
            ],
            dim=-1,
        )
        return {
            "positions": positions,
            "normals": normals,
            "normal_valid": normal_valid,
            "support": support,
            "raster_xy": raster_xy,
            "raster_valid": raster_valid,
            "observed": observed,
            "completed": completed,
            "degree": degree,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
        }

    def _previous_parent_distribution(
        self,
        current_state: torch.Tensor,
        current_position: torch.Tensor,
        previous_states: list[torch.Tensor],
        previous_positions: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, d = current_state.shape
        k = len(previous_states)
        if k == 0:
            return (
                current_state.new_empty((b, 0)),
                torch.zeros_like(current_position),
                torch.zeros_like(current_state),
            )
        ps = torch.stack(previous_states, dim=1)
        pp = torch.stack(previous_positions, dim=1)
        cur = current_state[:, None, :].expand(b, k, d)
        cp = current_position[:, None, :].expand(b, k, 3)
        delta = pp - cp
        dist = torch.linalg.norm(delta, dim=-1, keepdim=True)
        logits = self.internal_parent(
            torch.cat([cur, ps, delta, dist], dim=-1)
        ).squeeze(-1)
        prob = torch.softmax(logits, dim=-1)
        parent_pos = torch.einsum("bk,bkq->bq", prob, pp)
        parent_state = torch.einsum("bk,bkd->bd", prob, ps)
        return logits, parent_pos, parent_state

    def _all_pair_parent_logits(
        self,
        states: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        if states.ndim != 3 or positions.shape != (*states.shape[:2], 3):
            raise ValueError("all-pair state/position shape mismatch")
        b, k, d = states.shape
        rows = []
        chunk = int(self.config.edge_pair_chunk)
        for start in range(0, k, chunk):
            end = min(k, start + chunk)
            c = end - start
            child = states[:, start:end, None, :].expand(b, c, k, d)
            parent = states[:, None, :, :].expand(b, c, k, d)
            child_pos = positions[:, start:end, None, :].expand(b, c, k, 3)
            parent_pos = positions[:, None, :, :].expand(b, c, k, 3)
            delta = parent_pos - child_pos
            dist = torch.linalg.norm(delta, dim=-1, keepdim=True)
            rows.append(
                self.final_parent(
                    torch.cat([child, parent, delta, dist], dim=-1)
                ).squeeze(-1)
            )
        out = torch.cat(rows, dim=1)
        eye = torch.eye(k, device=out.device, dtype=torch.bool)[None]
        return out.masked_fill(eye, -1e4)

    def forward_surface(
        self,
        surface: RiggingSurfaceTensorV1,
        *,
        decode_steps: int,
        teacher_positions_normalized: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        diffusion_sample_steps: Optional[int] = None,
    ) -> GeppettoReferenceStrengthRawOutputV1:
        if decode_steps < 1 or decode_steps > surface.node_count:
            raise ValueError("decode_steps exceeds surface-cardinality resource guard")
        device = next(self.parameters()).device
        st = self._surface_tensors(surface, device)
        memory, pooled = self.encoder(**st)
        d = self.config.model_dim
        state_layers = [self.state_init(pooled) for _ in range(self.config.causal_layers)]
        prev_token = self.start[None].to(device=device).expand(1, -1)
        previous_states: list[torch.Tensor] = []
        previous_positions: list[torch.Tensor] = []
        coarse_rows = []
        position_rows = []
        state_rows = []
        stop_rows = []
        exist_rows = []
        root_rows = []
        salience_rows = []
        support_presence_rows = []
        support_rows = []
        parent_rows = []
        sigma_rows = []
        attn_rows = []
        diffusion_losses = []

        if teacher_positions_normalized is not None:
            teacher = torch.as_tensor(
                teacher_positions_normalized, device=device, dtype=torch.float32
            )
            if teacher.shape != (decode_steps, 3):
                raise ValueError("teacher_positions_normalized must be [decode_steps,3]")
        else:
            teacher = None

        for step in range(decode_steps):
            seq = self._sequence_embedding(
                step, d, device=device, dtype=pooled.dtype
            )
            query0 = self.step_proj(seq + state_layers[-1])
            ctx, attn = self.cross_attn(
                query0[:, None, :],
                memory[None],
                memory[None],
                need_weights=True,
                average_attn_weights=True,
            )
            surface_ctx = self.cross_fuse(
                torch.cat([query0, ctx[:, 0]], dim=-1)
            )

            if previous_positions:
                provisional_pos = previous_positions[-1]
                _ilogits, parent_pos, parent_state = self._previous_parent_distribution(
                    state_layers[-1],
                    provisional_pos,
                    previous_states,
                    previous_positions,
                )
            else:
                parent_pos = torch.zeros((1, 3), device=device, dtype=pooled.dtype)
                parent_state = torch.zeros_like(pooled)

            prev_pos = (
                previous_positions[-1]
                if previous_positions
                else torch.zeros((1, 3), device=device, dtype=pooled.dtype)
            )
            fb = self.feedback(
                torch.cat(
                    [
                        surface_ctx,
                        prev_token,
                        parent_state,
                        prev_pos,
                        parent_pos,
                    ],
                    dim=-1,
                )
            )
            x = torch.cat([surface_ctx, prev_token, fb, seq], dim=-1)
            for li, cell in enumerate(self.cells):
                state_layers[li] = cell(x, state_layers[li])
                x = torch.cat(
                    [surface_ctx, state_layers[li], fb, seq], dim=-1
                )
            state = state_layers[-1]
            coarse = (
                torch.tanh(self.coarse_position(state))
                * self.config.position_clip
            )

            internal_logits, parent_pos2, parent_state2 = self._previous_parent_distribution(
                state,
                coarse,
                previous_states,
                previous_positions,
            )
            parent_rows.append(internal_logits)

            if teacher is not None:
                residual_target = teacher[step : step + 1] - coarse.detach()
                diffusion_losses.append(
                    self.diffusion.loss(
                        residual_target,
                        state,
                        generator=generator,
                    )
                )
                # Training recurrence remains prediction-only: no teacher residual
                # is injected into the generated geometry state.
                position = coarse
            else:
                residual = self.diffusion.sample(
                    state,
                    generator=generator,
                    sample_steps=diffusion_sample_steps,
                )
                position = (
                    coarse + residual
                ).clamp(-self.config.position_clip, self.config.position_clip)

            # Causal recurrence uses the coarse prediction in both training and
            # inference. Diffusion is an output refinement, not a recurrent-state
            # input, so sampling noise cannot create a second exposure gap.
            token = self.feedback(
                torch.cat(
                    [
                        state,
                        surface_ctx,
                        parent_state2,
                        coarse,
                        parent_pos2,
                    ],
                    dim=-1,
                )
            )
            previous_states.append(token)
            previous_positions.append(coarse)
            prev_token = token

            support_logits = (
                torch.einsum(
                    "bd,nd->bn",
                    self.support_q(state),
                    self.support_k(memory),
                )
                / math.sqrt(d)
            )
            coarse_rows.append(coarse)
            position_rows.append(position)
            state_rows.append(state)
            stop_rows.append(self.stop(state).squeeze(-1))
            exist_rows.append(self.existence(state).squeeze(-1))
            root_rows.append(self.root(state).squeeze(-1))
            salience_rows.append(self.salience(state).squeeze(-1))
            support_presence_rows.append(
                self.support_presence(state).squeeze(-1)
            )
            support_rows.append(support_logits)
            sigma_rows.append(
                self.position_log_sigma(state.detach()).clamp(-8.0, 4.0)
            )
            attn_rows.append(attn[:, 0])

        states = torch.stack(state_rows, dim=1)
        coarse_positions = torch.stack(coarse_rows, dim=1)
        positions = torch.stack(position_rows, dim=1)
        all_pair = self._all_pair_parent_logits(states, positions)
        k = decode_steps
        internal = states.new_full((1, k, k), -1e4)
        for child, logits in enumerate(parent_rows):
            if child:
                internal[:, child, :child] = logits

        diffusion_loss = (
            torch.stack(diffusion_losses).mean()
            if diffusion_losses
            else None
        )
        return GeppettoReferenceStrengthRawOutputV1(
            coarse_positions_normalized=coarse_positions,
            positions_normalized=positions,
            control_states=states,
            stop_logits=torch.stack(stop_rows, dim=1),
            existence_logits=torch.stack(exist_rows, dim=1),
            root_logits=torch.stack(root_rows, dim=1),
            salience_logits=torch.stack(salience_rows, dim=1),
            support_presence_logits=torch.stack(
                support_presence_rows, dim=1
            ),
            support_logits=torch.stack(support_rows, dim=1),
            internal_parent_logits=internal,
            all_pair_parent_logits=all_pair,
            position_log_sigma=torch.stack(sigma_rows, dim=1),
            surface_attention=torch.stack(attn_rows, dim=1),
            diffusion_loss=diffusion_loss,
            teacher_target_used=teacher is not None,
            teacher_feedback_used=False,
        )

    @torch.no_grad()
    def generate_surface(
        self,
        surface: RiggingSurfaceTensorV1,
        *,
        resource_step_limit: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
        diffusion_sample_steps: Optional[int] = None,
    ) -> tuple[GeppettoReferenceStrengthRawOutputV1, int]:
        max_steps = (
            surface.node_count
            if resource_step_limit is None
            else int(resource_step_limit)
        )
        if max_steps < 1 or max_steps > surface.node_count:
            raise ValueError("invalid resource_step_limit")
        # Decode the explicit resource window once, then native STOP selects count.
        out = self.forward_surface(
            surface,
            decode_steps=max_steps,
            teacher_positions_normalized=None,
            generator=generator,
            diffusion_sample_steps=diffusion_sample_steps,
        )
        stop_prob = torch.sigmoid(out.stop_logits[0])
        hits = torch.nonzero(
            stop_prob >= self.config.stop_probability, as_tuple=False
        ).flatten()
        count = int(hits[0].item() + 1) if len(hits) else max_steps
        return out, count

    @torch.no_grad()
    def propose(
        self,
        surface: RiggingSurfaceTensorV1,
        *,
        resource_step_limit: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
        diffusion_sample_steps: Optional[int] = None,
    ) -> SkeletonProposalIR:
        self.eval()
        out, count = self.generate_surface(
            surface,
            resource_step_limit=resource_step_limit,
            generator=generator,
            diffusion_sample_steps=diffusion_sample_steps,
        )
        ids = tuple(f"P:GRS:{i:04d}" for i in range(count))
        joints = []
        sp = torch.sigmoid(out.support_presence_logits[0, :count])
        ex = torch.sigmoid(out.existence_logits[0, :count])
        roots = torch.sigmoid(out.root_logits[0, :count])
        salience = torch.sigmoid(out.salience_logits[0, :count])
        for i in range(count):
            pn = out.positions_normalized[0, i].detach().cpu().numpy()
            world = (
                pn.astype(np.float64) * float(surface.normalization_scale)
                + np.asarray(surface.normalization_center, dtype=np.float64)
            )
            sigma = torch.exp(out.position_log_sigma[0, i]).mean()
            support_ids: tuple[str, ...] = ()
            if float(sp[i]) >= self.config.support_presence_probability:
                top = min(self.config.support_topk, surface.node_count)
                idx = torch.argsort(
                    out.support_logits[0, i],
                    descending=True,
                    stable=True,
                )[:top].tolist()
                support_ids = tuple(surface.surface_ids[j] for j in idx)
            confidence = float(
                (ex[i] * torch.exp(-sigma)).clamp(0.0, 1.0)
            )
            joints.append(
                SkeletonProposalJoint(
                    proposal_id=ids[i],
                    position=tuple(map(float, world)),
                    root_score=float(roots[i]),
                    confidence=confidence,
                    support_surface_ids=support_ids,
                    metadata={
                        "generation_index_internal_only": i,
                        "mechanical_salience_probability": float(salience[i]),
                        "position_sigma_normalized": tuple(
                            map(
                                float,
                                torch.exp(
                                    out.position_log_sigma[0, i]
                                ).cpu().tolist(),
                            )
                        ),
                        "canonical_authority": False,
                        "teacher_feedback_used": False,
                    },
                )
            )

        edges = []
        for child in range(count):
            for parent in range(count):
                if child == parent:
                    continue
                score = float(
                    torch.sigmoid(
                        out.all_pair_parent_logits[0, child, parent]
                    )
                )
                edges.append(
                    SkeletonProposalEdge(
                        edge_id=f"E:{ids[parent]}->{ids[child]}",
                        parent_proposal_id=ids[parent],
                        child_proposal_id=ids[child],
                        score=score,
                        confidence=max(score, 1.0 - score),
                        hard_required=False,
                        hard_forbidden=False,
                        reason="GEPPETTO_REFERENCE_STRENGTH_ALL_PAIR_SOFT_EDGE_EVIDENCE",
                        metadata={
                            "internal_causal_parent_is_not_final_authority": True,
                            "compiler_owns_tree_selection": True,
                        },
                    )
                )
        return SkeletonProposalIR(
            joints=tuple(joints),
            edges=tuple(edges),
            surface_binding_hash=surface.source_surface_hash,
            model_provenance=self.config.config_hash,
            metadata={
                "candidate_architecture": self.config.architecture_id,
                "surface_tensorization_hash": surface.tensorization_hash,
                "surface_certificate_hash": surface.certificate_hash,
                "dynamic_cardinality": True,
                "fixed_product_joint_cap": False,
                "native_stop": True,
                "generation_index_is_not_identity": True,
                "teacher_feedback_at_training": False,
                "teacher_feedback_at_inference": False,
                "soft_predicted_parent_feedback": True,
                "final_parent_evidence": "SEPARATE_ALL_PAIRS_DIRECTED_SOFT_EVIDENCE",
                "compiler_owns_root_tree_ids": True,
                "legacy_24d_conditioning_used": False,
                "exact_gsa_topology_consumed": True,
                "per_view_raster_support_consumed": True,
                "full_surface_cross_attention_each_step": True,
                "conditional_residual_diffusion": True,
                "generalization_claim": False,
                "promotion_authorized": False,
            },
        )


__all__ = [
    "GeppettoReferenceStrengthConfigV1",
    "GeppettoReferenceStrengthRawOutputV1",
    "DirectRiggingSurfaceEncoderV1",
    "ConditionalResidualDiffusionV1",
    "GeppettoReferenceStrengthCandidateV1",
]
