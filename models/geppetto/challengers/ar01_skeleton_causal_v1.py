from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from models.geppetto.v2.geppetto_candidate_v2 import (
    GeppettoCandidateConfigV2,
    GeppettoCandidateV2,
    GeppettoRawOutputV2,
)


@dataclass(frozen=True)
class GeppettoAR01ConfigV1:
    base: GeppettoCandidateConfigV2 = GeppettoCandidateConfigV2()
    architecture_id: str = "RealSaS.GeppettoChallenger.SkeletonCausalAutoregression.AR01.v1"

    def validate(self) -> None:
        self.base.validate()


class AR01ViewEvidenceAdapterV1(nn.Module):
    """Lossless fixed-eight-view evidence side path used by both AR-01 arms."""

    input_dim = 32

    def __init__(self, dim: int):
        super().__init__()
        self.pre = nn.LayerNorm(self.input_dim)
        self.fc1 = nn.Linear(self.input_dim, dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim, dim)
        # Preserve the historical trunk exactly at initialization.
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(
        self,
        raster_xy_by_view: torch.Tensor,
        raster_valid_by_view: torch.Tensor,
        support_by_view: torch.Tensor,
    ) -> torch.Tensor:
        if raster_xy_by_view.ndim != 4 or raster_xy_by_view.shape[-2:] != (8, 2):
            raise ValueError("AR01_RASTER_XY_INPUT_SHAPE")
        if (
            raster_valid_by_view.shape != raster_xy_by_view.shape[:3]
            or support_by_view.shape != raster_xy_by_view.shape[:3]
        ):
            raise ValueError("AR01_VIEW_MASK_INPUT_SHAPE")
        evidence = torch.cat(
            [
                raster_xy_by_view.reshape(*raster_xy_by_view.shape[:2], 16),
                raster_valid_by_view.to(raster_xy_by_view.dtype),
                support_by_view.to(raster_xy_by_view.dtype),
            ],
            dim=-1,
        )
        return self.fc2(self.act(self.fc1(self.pre(evidence))))


class GeppettoAR01SkeletonCausalV1(GeppettoCandidateV2):
    """Research-only causal-isolation challenger for skeleton feedback.

    Both scientific arms use this exact class and exact parameterization.

    AR0:
        mechanical_feedback_enabled=False
        The recurrent history remains latent-only after per-step full-surface attention.

    AR1:
        mechanical_feedback_enabled=True
        Teacher joint + parent geometry is fed into the next recurrent state during
        training; generated joint + predicted-parent geometry is fed back at inference.

    The feedback is a residual delta on the recurrent state, with a zero-initialized
    final projection. Therefore AR0 and AR1 are bit-identical at initialization and
    differ only after the feedback channel learns. Final root/parent/tree legality
    remains Compiler authority; this module emits causal parent evidence only.
    """

    arm_id = "AR01_SHARED_CLASS_SINGLE_FEEDBACK_GATE"
    uses_lossless_view_evidence = True
    uses_full_surface_cross_attention = True
    uses_diffusion = False
    consumes_topology = False

    def __init__(self, config: GeppettoAR01ConfigV1 = GeppettoAR01ConfigV1()):
        config.validate()
        super().__init__(config.base)
        d = config.base.model_dim
        self.ar01_config = config
        self.view_evidence_adapter = AR01ViewEvidenceAdapterV1(d)
        self.surface_cross_attn = nn.MultiheadAttention(
            d,
            config.base.attention_heads,
            batch_first=True,
            dropout=0.0,
        )
        self.surface_fuse = nn.Sequential(
            nn.LayerNorm(d * 2),
            nn.Linear(d * 2, d),
            nn.GELU(),
            nn.Linear(d, d),
        )
        self.mechanical_joint_embed = nn.Sequential(
            nn.Linear(3, d),
            nn.GELU(),
            nn.Linear(d, d),
        )
        self.mechanical_relation_embed = nn.Sequential(
            nn.Linear(4, d),
            nn.GELU(),
            nn.Linear(d, d),
        )
        self.mechanical_feedback_fuse = nn.Sequential(
            nn.LayerNorm(d * 4),
            nn.Linear(d * 4, d * 2),
            nn.GELU(),
            nn.Linear(d * 2, d),
        )
        nn.init.zeros_(self.mechanical_feedback_fuse[-1].weight)
        nn.init.zeros_(self.mechanical_feedback_fuse[-1].bias)
        self._last_attention_token_count: int | None = None
        self._last_attention_entropy: float | None = None

    def _encode_lossless(
        self,
        trunk_features: torch.Tensor,
        positions_normalized: torch.Tensor,
        valid_mask: torch.Tensor,
        raster_xy_by_view: torch.Tensor,
        raster_valid_by_view: torch.Tensor,
        support_by_view: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        memory, _ = self.encoder(trunk_features, positions_normalized, valid_mask.bool())
        delta = self.view_evidence_adapter(
            raster_xy_by_view,
            raster_valid_by_view,
            support_by_view,
        )
        memory = (memory + delta) * valid_mask[..., None].to(memory.dtype)
        denom = valid_mask.sum(1, keepdim=True).clamp_min(1).to(memory.dtype)
        return memory, memory.sum(1) / denom

    def _causal_parent_logits(
        self,
        current_context: torch.Tensor,
        current_position: torch.Tensor,
        previous_contexts: list[torch.Tensor],
        previous_positions: list[torch.Tensor],
    ) -> torch.Tensor:
        b, d = current_context.shape
        k = len(previous_contexts)
        if k == 0:
            return current_context.new_empty((b, 0))
        parent_h = torch.stack(previous_contexts, dim=1)
        parent_pos = torch.stack(previous_positions, dim=1)
        child_h = current_context[:, None, :].expand(b, k, d)
        child_pos = current_position[:, None, :].expand(b, k, 3)
        delta = parent_pos - child_pos
        dist = torch.linalg.norm(delta, dim=-1, keepdim=True)
        return self.parent_pair(
            torch.cat([child_h, parent_h, delta, dist], dim=-1)
        ).squeeze(-1)

    def _mechanical_feedback_delta(
        self,
        context: torch.Tensor,
        joint_position: torch.Tensor,
        parent_position: torch.Tensor,
    ) -> torch.Tensor:
        joint = self.mechanical_joint_embed(joint_position)
        parent = self.mechanical_joint_embed(parent_position)
        delta = parent_position - joint_position
        dist = torch.linalg.norm(delta, dim=-1, keepdim=True)
        relation = self.mechanical_relation_embed(torch.cat([delta, dist], dim=-1))
        return self.mechanical_feedback_fuse(
            torch.cat([context, joint, parent, relation], dim=-1)
        )

    @staticmethod
    def _teacher_parent_position(
        teacher_positions: torch.Tensor,
        teacher_parent_indices: torch.Tensor,
        step: int,
    ) -> torch.Tensor:
        current = teacher_positions[:, step]
        if step == 0:
            if (teacher_parent_indices[:, step] >= 0).any():
                raise ValueError("AR01_FIRST_TEACHER_STEP_MUST_BE_ROOT")
            return torch.zeros_like(current)
        parent_idx = teacher_parent_indices[:, step].long()
        is_root = parent_idx < 0
        if ((parent_idx >= step) & ~is_root).any():
            raise ValueError("AR01_TEACHER_PARENT_MUST_PRECEDE_CHILD")
        safe_idx = parent_idx.clamp_min(0)
        prior = teacher_positions[:, :step]
        gather = safe_idx[:, None, None].expand(current.shape[0], 1, 3)
        parent = torch.gather(prior, 1, gather).squeeze(1)
        return torch.where(is_root[:, None], torch.zeros_like(parent), parent)

    def forward_ar01(
        self,
        trunk_features: torch.Tensor,
        positions_normalized: torch.Tensor,
        valid_mask: torch.Tensor,
        raster_xy_by_view: torch.Tensor,
        raster_valid_by_view: torch.Tensor,
        support_by_view: torch.Tensor,
        *,
        decode_steps: int,
        mechanical_feedback_enabled: bool,
        teacher_positions: torch.Tensor | None = None,
        teacher_parent_indices: torch.Tensor | None = None,
    ) -> GeppettoRawOutputV2:
        memory, pooled = self._encode_lossless(
            trunk_features,
            positions_normalized,
            valid_mask,
            raster_xy_by_view,
            raster_valid_by_view,
            support_by_view,
        )
        b, _, d = memory.shape
        modes_n = int(self.config.position_modes)
        limits = self.resource_limits(valid_mask)
        if decode_steps < 1 or int(decode_steps) > int(limits.max().item()):
            raise ValueError("AR01_DECODE_STEPS_EXCEED_SURFACE_GUARD")

        teacher_forced = teacher_positions is not None
        if teacher_forced:
            if teacher_parent_indices is None:
                raise ValueError("AR01_TEACHER_PARENTS_REQUIRED")
            if teacher_positions.shape != (b, int(decode_steps), 3):
                raise ValueError("AR01_TEACHER_POSITION_SHAPE")
            if teacher_parent_indices.shape != (b, int(decode_steps)):
                raise ValueError("AR01_TEACHER_PARENT_SHAPE")
        elif teacher_parent_indices is not None:
            raise ValueError("AR01_TEACHER_PARENT_WITHOUT_POSITIONS")

        states = [pooled.new_zeros((b, d)) for _ in self.cells]
        history_states: list[torch.Tensor] = []
        previous_contexts: list[torch.Tensor] = []
        previous_positions: list[torch.Tensor] = []
        out_rows = []
        parent_rows = []
        entropies = []

        for step in range(int(decode_steps)):
            prev = self.start[None].expand(b, -1) if step == 0 else history_states[-1]
            query = self.rel_query(torch.cat([pooled, prev], dim=-1))
            if history_states:
                hist = torch.stack(history_states, dim=1)
                score = torch.einsum("bd,btd->bt", query, hist) / (d ** 0.5)
                rel = torch.einsum("bt,btd->bd", torch.softmax(score, dim=-1), hist)
            else:
                rel = torch.zeros_like(pooled)

            x = torch.cat([pooled, prev, rel], dim=-1)
            for layer_idx, cell in enumerate(self.cells):
                states[layer_idx] = cell(x, states[layer_idx])
                x = torch.cat([pooled, states[layer_idx], rel], dim=-1)
            h_base = states[-1]

            ctx, attn = self.surface_cross_attn(
                h_base[:, None, :],
                memory,
                memory,
                key_padding_mask=~valid_mask.bool(),
                need_weights=True,
                average_attn_weights=True,
            )
            h = self.surface_fuse(torch.cat([h_base, ctx[:, 0]], dim=-1))
            self._last_attention_token_count = int(attn.shape[-1])
            p = attn[:, 0].clamp_min(1e-12)
            entropies.append((-(p * torch.log(p)).sum(-1)).mean())

            modes = (
                torch.tanh(self.position(h).reshape(b, modes_n, 3))
                * self.config.position_scale
            )
            mode_ls = (
                self.log_sigma(h.detach())
                .reshape(b, modes_n, 3)
                .clamp(-8.0, 4.0)
            )
            mode_logits = self.position_mode_logits(h)
            pos, rep_ls, _ = self._map_representative(modes, mode_ls, mode_logits)

            causal_parent = self._causal_parent_logits(
                h,
                pos,
                previous_contexts,
                previous_positions,
            )
            padded_parent = h.new_full((b, int(decode_steps)), -1e4)
            if causal_parent.numel():
                padded_parent[:, :step] = causal_parent
            parent_rows.append(padded_parent)

            if teacher_forced:
                feedback_joint = teacher_positions[:, step]
                feedback_parent = self._teacher_parent_position(
                    teacher_positions,
                    teacher_parent_indices,
                    step,
                )
            else:
                feedback_joint = pos
                if step == 0:
                    feedback_parent = torch.zeros_like(pos)
                else:
                    parent_idx = torch.argsort(
                        causal_parent,
                        dim=-1,
                        descending=True,
                        stable=True,
                    )[:, 0]
                    prior = torch.stack(previous_positions, dim=1)
                    gather = parent_idx[:, None, None].expand(b, 1, 3)
                    feedback_parent = torch.gather(prior, 1, gather).squeeze(1)

            if mechanical_feedback_enabled:
                history_state = h + self._mechanical_feedback_delta(
                    h,
                    feedback_joint,
                    feedback_parent,
                )
            else:
                history_state = h

            # The mechanical state is a true recurrent-state update, not merely
            # an auxiliary head. AR0 is exactly the zero-treatment path.
            states[-1] = history_state
            history_states.append(history_state)
            previous_contexts.append(h)
            previous_positions.append(pos)

            out_rows.append(
                (
                    h,
                    pos,
                    rep_ls,
                    modes,
                    mode_ls,
                    mode_logits,
                    self.existence(h).squeeze(-1),
                    self.stop(h).squeeze(-1),
                    self.root(h).squeeze(-1),
                    self.support_presence(h).squeeze(-1),
                )
            )

        self._last_attention_entropy = float(
            torch.stack(entropies).mean().detach().cpu()
        )
        fields = list(zip(*out_rows))
        h, pos, ls, modes, mode_ls, mode_logits, ex, stop, root, sp = (
            torch.stack(x, 1) for x in fields
        )
        parent = torch.stack(parent_rows, dim=1)
        support = (
            torch.einsum(
                "bkd,bnd->bkn",
                self.support_q(h),
                self.support_k(memory),
            )
            / (d ** 0.5)
        )
        support = support.masked_fill(~valid_mask[:, None, :].bool(), -1e4)
        return GeppettoRawOutputV2(
            pos,
            ls,
            ex,
            stop,
            root,
            sp,
            parent,
            support,
            h,
            self.abstain(pooled).squeeze(-1),
            modes,
            mode_ls,
            mode_logits,
        )


__all__ = [
    "GeppettoAR01ConfigV1",
    "AR01ViewEvidenceAdapterV1",
    "GeppettoAR01SkeletonCausalV1",
]
