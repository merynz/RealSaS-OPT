from __future__ import annotations

"""AXIS V7: large conditional causal rig+control+SkinCode model.

Mechanics is the learning target. Graphics is NOT part of this architecture.
Uses the typed V6 program language as a lossless migration seam; a separate
V7 interactability language may be introduced only after truth/courts approve it.

This research model outputs rig graph, rest frames, controls, constraints,
attachments and MIRA V7's 64k skin codes within ONE causal hypothesis. Skin
logits are computed sparsely only at actual SKIN_CODE positions to avoid the
B x sequence_length x 64000 activation tensor during teacher forcing.

Research checkpoint ONLY; no trained weights, ModelRelease or ProductRevision.
"""

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from models.axis.rig_program_model_v6 import (
    AXISSequenceV6, AXISRigProgramGrammarV6Product,
    RigGrammarStateV6, RigProgramEventV6, RigSectionV6, RigTokenKindV6,
    CONTROL_KINDS_V6, CONSTRAINT_OPERATORS_V6, DOF_LABELS_V6,
)
from models.mira.skin_fsq_codec_v7 import MIRACodecConfigV7


AXIS_V7_ARCHITECTURE_ID = "RealSaS.AXIS.UnifiedRigControlSkin.CausalCrossAttention.v7_0"


@dataclass(frozen=True)
class AXISRigProgramConfigV7:
    scene_dim: int = 47
    hidden_dim: int = 768
    layers: int = 16
    heads: int = 12
    feedforward_mult: int = 4
    max_sequence_length: int = 4096
    max_pointer_entities: int = 256
    coord_bins: int = 256
    coord_min: float = -1.0
    coord_max: float = 1.0
    skin_vocab_size: int = 64000
    tokens_per_joint: int = 4
    dropout: float = 0.0

    def __post_init__(self):
        if min(self.scene_dim, self.hidden_dim, self.layers, self.heads,
               self.feedforward_mult, self.max_pointer_entities,
               self.max_sequence_length, self.coord_bins, self.skin_vocab_size,
               self.tokens_per_joint) < 1:
            raise ValueError("AXIS_V7_CONFIG_NONPOSITIVE")
        if self.hidden_dim % self.heads:
            raise ValueError("AXIS_V7_HEAD_DIVISIBILITY")
        if not self.coord_max > self.coord_min:
            raise ValueError("AXIS_V7_COORD_RANGE_INVALID")
        if self.max_sequence_length > 65536:
            raise ValueError("AXIS_V7_SEQUENCE_POLICY_LIMIT")

    def to_dict(self) -> dict:
        return asdict(self)


class AXISRigProgramModelV7(nn.Module):
    architecture_id = AXIS_V7_ARCHITECTURE_ID

    def __init__(self, config: AXISRigProgramConfigV7 = AXISRigProgramConfigV7()):
        super().__init__()
        self.config = config
        h = config.hidden_dim
        self.kind_embedding = nn.Embedding(len(RigTokenKindV6), h)
        self.coord_embedding = nn.Embedding(config.coord_bins + 1, h)
        self.pointer_embedding = nn.Embedding(config.max_pointer_entities + 1, h)
        self.aux_embedding = nn.Embedding(max(len(CONTROL_KINDS_V6), len(CONSTRAINT_OPERATORS_V6)) + 1, h)
        self.skin_embedding = nn.Embedding(config.skin_vocab_size + 1, h)
        self.anchor_parent_embedding = nn.Embedding(3, h)
        self.position_embedding = nn.Embedding(config.max_sequence_length, h)
        self.dof_proj = nn.Linear(len(DOF_LABELS_V6), h, bias=False)
        self.frame_proj = nn.Linear(6, h, bias=False)
        self.scalar_proj = nn.Linear(4, h, bias=False)
        self.input_norm = nn.LayerNorm(h)
        self.scene_projection = nn.Sequential(nn.Linear(config.scene_dim, h),
                                              nn.LayerNorm(h), nn.GELU())
        self.scene_type_embedding = nn.Parameter(torch.zeros(1, 1, h))
        layer = nn.TransformerDecoderLayer(
            d_model=h, nhead=config.heads,
            dim_feedforward=h * config.feedforward_mult,
            dropout=config.dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerDecoder(layer, num_layers=config.layers,
                                                  norm=nn.LayerNorm(h))
        self.kind_head = nn.Linear(h, len(RigTokenKindV6))
        self.coord_head = nn.Linear(h, 3 * config.coord_bins)
        self.pointer_a_head = nn.Linear(h, config.max_pointer_entities + 1)
        self.pointer_b_head = nn.Linear(h, config.max_pointer_entities + 1)
        self.pointer_c_head = nn.Linear(h, config.max_pointer_entities + 1)
        self.control_kind_head = nn.Linear(h, len(CONTROL_KINDS_V6))
        self.constraint_operator_head = nn.Linear(h, len(CONSTRAINT_OPERATORS_V6))
        self.dof_head = nn.Linear(h, len(DOF_LABELS_V6))
        self.frame_6d_head = nn.Linear(h, 6)
        self.scalar_head = nn.Linear(h, 4)
        self.anchor_parent_head = nn.Linear(h, 2)
        self.skin_bias = nn.Parameter(torch.zeros(config.skin_vocab_size))

    @staticmethod
    def _shift(value: Tensor, maximum: int) -> Tensor:
        return (value.to(torch.long) + 1).clamp(0, maximum)

    def _embed(self, seq: AXISSequenceV6) -> Tensor:
        seq.validate(self.config)
        b, length = seq.kind_ids.shape
        positions = torch.arange(length, device=seq.kind_ids.device)[None].expand(b, -1)
        x = self.kind_embedding(seq.kind_ids) + self.position_embedding(positions)
        x = x + self.coord_embedding(self._shift(seq.coord_indices, self.config.coord_bins)).sum(dim=2)
        for ptr in (seq.pointer_a, seq.pointer_b, seq.pointer_c):
            x = x + self.pointer_embedding(self._shift(ptr, self.config.max_pointer_entities))
        x = x + self.aux_embedding(self._shift(seq.aux_type_ids, self.aux_embedding.num_embeddings - 1))
        x = x + self.skin_embedding(self._shift(seq.skin_code_ids, self.config.skin_vocab_size))
        x = x + self.anchor_parent_embedding(self._shift(seq.anchor_parent_kind, 2))
        x = x + self.dof_proj(seq.dof_masks.to(x.dtype))
        x = x + self.frame_proj(seq.frame_6d.to(x.dtype))
        x = x + self.scalar_proj(seq.scalar_params.to(x.dtype))
        return self.input_norm(x)

    def forward(self, scene_memory: Tensor, sequence: AXISSequenceV6) -> dict[str, Tensor]:
        if scene_memory.ndim != 3 or scene_memory.shape[0] != sequence.kind_ids.shape[0]:
            raise ValueError("AXIS_V7_SCENE_BATCH_OR_RANK_INVALID")
        if scene_memory.shape[-1] != self.config.scene_dim or scene_memory.shape[1] < 1:
            raise ValueError("AXIS_V7_SCENE_DIM_INVALID")
        if not torch.isfinite(scene_memory).all():
            raise ValueError("AXIS_V7_SCENE_NONFINITE")
        x = self._embed(sequence)
        s = x.shape[1]
        # True values mask forbidden future tokens; no attention to targets.
        causal_mask = torch.triu(
            torch.ones((s, s), device=x.device, dtype=torch.bool), diagonal=1,
        )
        scene = self.scene_projection(scene_memory.to(x.dtype)) + self.scene_type_embedding
        hidden = self.transformer(
            x, scene, tgt_mask=causal_mask,
            tgt_key_padding_mask=~sequence.attention_mask.to(torch.bool),
        )
        return {
            "hidden": hidden,
            "kind_logits": self.kind_head(hidden),
            "coord_logits": self.coord_head(hidden).reshape(*hidden.shape[:2], 3, self.config.coord_bins),
            "pointer_a_logits": self.pointer_a_head(hidden),
            "pointer_b_logits": self.pointer_b_head(hidden),
            "pointer_c_logits": self.pointer_c_head(hidden),
            "control_kind_logits": self.control_kind_head(hidden),
            "constraint_operator_logits": self.constraint_operator_head(hidden),
            "dof_logits": self.dof_head(hidden),
            "frame_6d": self.frame_6d_head(hidden),
            "scalar_params": self.scalar_head(hidden),
            "anchor_parent_logits": self.anchor_parent_head(hidden),
        }

    def skin_code_logits(self, hidden: Tensor) -> Tensor:
        # Tied code embedding avoids a second 64000 x hidden_dim weight matrix.
        return F.linear(hidden, self.skin_embedding.weight[1:], self.skin_bias)

    @staticmethod
    def _pointer(logits: Tensor, count: int, allow_null: bool) -> int:
        if count < 0 or count + 1 > logits.shape[-1]:
            raise ValueError("AXIS_V7_POINTER_RESOURCE_GUARD")
        values = logits[:count + 1].clone()
        if not allow_null:
            if count < 1:
                raise ValueError("AXIS_V7_REQUIRED_POINTER_EMPTY")
            values[0] = float("-inf")
        return int(torch.argmax(values).item()) - 1

    @torch.no_grad()
    def generate_events(self, scene_memory: Tensor, *, max_steps: int | None = None
                        ) -> tuple[RigProgramEventV6, ...]:
        """Exact V6 product event wire; autoregressive correctness baseline.

        Full-prefix decoding is intentionally not represented as a low-latency
        product path. It requires KV-cache benchmarking before shipping.
        """
        self.eval()
        if scene_memory.ndim != 3 or scene_memory.shape[0] != 1:
            raise ValueError("AXIS_V7_GENERATION_BATCH_ONE_REQUIRED")
        limit = min(max_steps or self.config.max_sequence_length, self.config.max_sequence_length)
        state = AXISRigProgramGrammarV6Product.advance(
            RigGrammarStateV6(), RigTokenKindV6.BOS,
            tokens_per_joint=self.config.tokens_per_joint,
        )
        events = [RigProgramEventV6(RigTokenKindV6.BOS)]
        for _ in range(limit - 1):
            prefix = AXISSequenceV6.from_events(events, device=scene_memory.device)
            out = self(scene_memory, prefix)
            kind_logits = AXISRigProgramGrammarV6Product.mask(
                out["kind_logits"][0, -1], state,
                tokens_per_joint=self.config.tokens_per_joint,
            )
            kind = RigTokenKindV6(int(torch.argmax(kind_logits).item()))
            coord = (-1, -1, -1)
            pa = pb = aux = code = anchor_kind = -1
            frame = (0.0,) * 6
            dof = (0.0,) * len(DOF_LABELS_V6)
            params = (0.0,) * 4
            last = {key: value[0, -1] for key, value in out.items() if key != "hidden"}
            ptr = lambda which, count, nullable=False: self._pointer(
                last[which], count, nullable,
            )
            if kind in (RigTokenKindV6.JOINT, RigTokenKindV6.CONTROL, RigTokenKindV6.ANCHOR):
                coord = tuple(int(x) for x in torch.argmax(last["coord_logits"], dim=-1).tolist())
            if kind == RigTokenKindV6.JOINT:
                pa = -1 if not state.joint_count else ptr("pointer_a_logits", state.joint_count, True)
            elif kind == RigTokenKindV6.FRAME:
                pa = ptr("pointer_a_logits", state.joint_count)
                frame = tuple(float(x) for x in last["frame_6d"].tolist())
                dof = tuple(float(x) for x in (torch.sigmoid(last["dof_logits"]) >= 0.5).tolist())
            elif kind == RigTokenKindV6.CONTROL:
                aux = int(torch.argmax(last["control_kind_logits"]).item())
                pa = ptr("pointer_a_logits", state.control_count, True)
            elif kind == RigTokenKindV6.CONTROL_TARGET:
                pa = ptr("pointer_a_logits", state.joint_count)
            elif kind == RigTokenKindV6.CHAIN:
                pa = ptr("pointer_a_logits", state.joint_count)
                pb = ptr("pointer_b_logits", state.joint_count)
            elif kind == RigTokenKindV6.CONSTRAINT:
                aux = int(torch.argmax(last["constraint_operator_logits"]).item())
                pa = ptr("pointer_a_logits", state.chain_count, True)
                params = tuple(float(x) for x in last["scalar_params"].tolist())
            elif kind == RigTokenKindV6.CONSTRAINT_INPUT:
                pa = ptr("pointer_a_logits", state.control_count)
            elif kind == RigTokenKindV6.CONSTRAINT_OUTPUT:
                pa = ptr("pointer_a_logits", state.joint_count)
            elif kind == RigTokenKindV6.ANCHOR:
                anchor_kind = int(torch.argmax(last["anchor_parent_logits"]).item())
                if anchor_kind == 0:
                    pa = ptr("pointer_a_logits", state.joint_count)
                else:
                    pb = ptr("pointer_b_logits", state.control_count)
            elif kind == RigTokenKindV6.SKIN_CODE:
                code = int(torch.argmax(self.skin_code_logits(out["hidden"][0, -1])).item())
            event = RigProgramEventV6(
                kind=kind, coord_indices=coord, pointer_a=pa, pointer_b=pb,
                aux_type_id=aux, dof_mask=dof, frame_6d=frame,
                scalar_params=params, skin_code_id=code,
                anchor_parent_kind=anchor_kind,
            )
            events.append(event)
            state = AXISRigProgramGrammarV6Product.advance(
                state, kind, tokens_per_joint=self.config.tokens_per_joint,
            )
            if state.section == RigSectionV6.DONE:
                return tuple(events)
        raise RuntimeError("AXIS_V7_GENERATION_RESOURCE_LIMIT_EXHAUSTED")


def axis_v7_teacher_forced_objective(
    model: AXISRigProgramModelV7, scene_memory: Tensor,
    inputs: AXISSequenceV6, targets: AXISSequenceV6,
) -> dict[str, Tensor]:
    """All token-field losses; sparse code CE and inclusive nullable pointers."""
    out = model(scene_memory, inputs)
    active = targets.attention_mask.to(torch.bool)
    kinds = targets.kind_ids

    def ce(key: str, truth: Tensor, mask: Tensor) -> Tensor:
        if not bool(mask.any()):
            return out["hidden"].sum() * 0.0
        return F.cross_entropy(out[key][mask], truth[mask])

    def selected(*event_kinds: RigTokenKindV6) -> Tensor:
        mask = torch.zeros_like(active)
        for kind in event_kinds:
            mask = mask | (kinds == int(kind))
        return active & mask

    losses = {}
    losses["kind"] = ce("kind_logits", kinds, active)
    xyz = selected(RigTokenKindV6.JOINT, RigTokenKindV6.CONTROL, RigTokenKindV6.ANCHOR)
    losses["coordinate"] = sum(
        F.cross_entropy(out["coord_logits"][:, :, axis][xyz],
                        targets.coord_indices[:, :, axis][xyz])
        if bool(xyz.any()) else out["hidden"].sum() * 0.0
        for axis in range(3)
    ) / 3.0
    ptr_a = selected(RigTokenKindV6.JOINT, RigTokenKindV6.FRAME,
                     RigTokenKindV6.CONTROL, RigTokenKindV6.CONTROL_TARGET,
                     RigTokenKindV6.CHAIN, RigTokenKindV6.CONSTRAINT,
                     RigTokenKindV6.CONSTRAINT_INPUT, RigTokenKindV6.CONSTRAINT_OUTPUT,
                     RigTokenKindV6.ANCHOR)
    ptr_b = selected(RigTokenKindV6.CHAIN) | (
        selected(RigTokenKindV6.ANCHOR) & (targets.anchor_parent_kind == 1)
    )
    losses["pointer"] = (ce("pointer_a_logits", (targets.pointer_a + 1).clamp_min(0), ptr_a)
                         + ce("pointer_b_logits", (targets.pointer_b + 1).clamp_min(0), ptr_b))
    control = selected(RigTokenKindV6.CONTROL)
    constraint = selected(RigTokenKindV6.CONSTRAINT)
    losses["control_kind"] = ce("control_kind_logits", targets.aux_type_ids.clamp_min(0), control)
    losses["constraint_operator"] = ce("constraint_operator_logits", targets.aux_type_ids.clamp_min(0), constraint)
    frame = selected(RigTokenKindV6.FRAME)
    if bool(frame.any()):
        losses["dof"] = F.binary_cross_entropy_with_logits(out["dof_logits"][frame], targets.dof_masks[frame])
        losses["frame"] = F.smooth_l1_loss(out["frame_6d"][frame], targets.frame_6d[frame])
    else:
        losses["dof"] = losses["frame"] = out["hidden"].sum() * 0.0
    losses["constraint_scalar"] = (
        F.smooth_l1_loss(out["scalar_params"][constraint], targets.scalar_params[constraint])
        if bool(constraint.any()) else out["hidden"].sum() * 0.0
    )
    anchor = selected(RigTokenKindV6.ANCHOR)
    losses["anchor_parent"] = ce("anchor_parent_logits", targets.anchor_parent_kind.clamp_min(0), anchor)
    code_mask = selected(RigTokenKindV6.SKIN_CODE)
    if bool(code_mask.any()):
        logits = model.skin_code_logits(out["hidden"][code_mask])
        losses["skin_code"] = F.cross_entropy(logits, targets.skin_code_ids[code_mask])
    else:
        losses["skin_code"] = out["hidden"].sum() * 0.0
    losses["total"] = sum(losses.values())
    return losses


def assert_axis_mira_v7_binding(axis: AXISRigProgramConfigV7, mira: MIRACodecConfigV7) -> None:
    if axis.skin_vocab_size != mira.vocab_size:
        raise ValueError("AXIS_MIRA_V7_VOCAB_BINDING_DRIFT")
    if axis.tokens_per_joint != mira.tokens_per_joint:
        raise ValueError("AXIS_MIRA_V7_TOKEN_CARDINALITY_DRIFT")


__all__ = [
    "AXIS_V7_ARCHITECTURE_ID", "AXISRigProgramConfigV7",
    "AXISRigProgramModelV7", "axis_v7_teacher_forced_objective",
    "assert_axis_mira_v7_binding",
]
