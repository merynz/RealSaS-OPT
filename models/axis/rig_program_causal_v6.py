from __future__ import annotations

"""Early AXIS V6 typed causal scaffold for control-ready rig programs.

V5.4.1 generated only a deform tree + XYZ. This scaffold expanded the learned
language to local frames, animator controls, kinematic chains, constraints,
attachment anchors and MIRA discrete skin codes. The shipping-target V6 model is
`rig_program_model_v6.py`; this file remains only for source/regression comparison.
"""

from dataclasses import dataclass
from enum import IntEnum

import torch
from torch import Tensor, nn


AXIS_V6_ARCHITECTURE_ID = "RealSaS.AXIS.RigProgram.TypedCausal.SharedSkinCodes.v6_0"
AXIS_V6_COORD_BINS = 256


class RigEventKind(IntEnum):
    PAD = 0
    BOS = 1
    JOINT = 2
    FRAME = 3
    CONTROL = 4
    CHAIN = 5
    CONSTRAINT = 6
    ANCHOR = 7
    SKIN_SWITCH = 8
    SKIN_TOKEN = 9
    EOS = 10


class RigProgramPhase(IntEnum):
    START = 0
    JOINTS = 1
    FRAMES = 2
    CONTROLS = 3
    CHAINS = 4
    CONSTRAINTS = 5
    ANCHORS = 6
    SKIN = 7
    DONE = 8


CONTROL_KINDS = (
    "GLOBAL", "ROOT", "FK", "IK_EFFECTOR", "IK_POLE", "SPLINE", "AIM_TARGET", "SOCKET_CONTROL",
)
CONSTRAINT_OPERATORS = (
    "IK", "AIM", "LIMIT_ROTATION", "LIMIT_TRANSLATION", "FOLLOW", "COPY_TRANSFORM", "PATH", "STRETCH",
)
DOF_LABELS = ("TX", "TY", "TZ", "RX", "RY", "RZ", "SX", "SY", "SZ")


@dataclass(frozen=True)
class AXISV6Config:
    scene_dim: int
    hidden_dim: int = 384
    layers: int = 8
    heads: int = 8
    ff_mult: int = 4
    max_sequence_length: int = 2048
    max_pointer_entities: int = 256
    coord_bins: int = AXIS_V6_COORD_BINS
    skin_vocab_size: int = 625
    dropout: float = 0.0


@dataclass
class RigProgramGrammarState:
    phase: RigProgramPhase = RigProgramPhase.START
    joint_count: int = 0
    frame_count: int = 0
    control_count: int = 0
    chain_count: int = 0
    constraint_count: int = 0
    anchor_count: int = 0
    skin_token_count: int = 0


class AXISRigProgramGrammarV6:
    @staticmethod
    def allowed_kinds(state: RigProgramGrammarState) -> tuple[RigEventKind, ...]:
        p = state.phase
        if p == RigProgramPhase.START:
            return (RigEventKind.BOS,)
        if p == RigProgramPhase.JOINTS:
            return (RigEventKind.JOINT, RigEventKind.FRAME)
        if p == RigProgramPhase.FRAMES:
            return (RigEventKind.FRAME, RigEventKind.CONTROL)
        if p == RigProgramPhase.CONTROLS:
            return (RigEventKind.CONTROL, RigEventKind.CHAIN)
        if p == RigProgramPhase.CHAINS:
            return (RigEventKind.CHAIN, RigEventKind.CONSTRAINT)
        if p == RigProgramPhase.CONSTRAINTS:
            return (RigEventKind.CONSTRAINT, RigEventKind.ANCHOR, RigEventKind.SKIN_SWITCH)
        if p == RigProgramPhase.ANCHORS:
            return (RigEventKind.ANCHOR, RigEventKind.SKIN_SWITCH)
        if p == RigProgramPhase.SKIN:
            return (RigEventKind.SKIN_TOKEN, RigEventKind.EOS)
        return ()

    @staticmethod
    def advance(state: RigProgramGrammarState, kind: RigEventKind) -> RigProgramGrammarState:
        if kind not in AXISRigProgramGrammarV6.allowed_kinds(state):
            raise ValueError(f"AXIS_V6_GRAMMAR_ILLEGAL_TRANSITION:{state.phase.name}->{kind.name}")
        out = RigProgramGrammarState(**state.__dict__)
        if state.phase == RigProgramPhase.START:
            out.phase = RigProgramPhase.JOINTS
        elif kind == RigEventKind.JOINT:
            out.joint_count += 1
        elif kind == RigEventKind.FRAME:
            if state.phase == RigProgramPhase.JOINTS:
                if state.joint_count < 1:
                    raise ValueError("AXIS_V6_GRAMMAR_FRAME_BEFORE_JOINT")
                out.phase = RigProgramPhase.FRAMES
            out.frame_count += 1
        elif kind == RigEventKind.CONTROL:
            if state.phase == RigProgramPhase.FRAMES:
                out.phase = RigProgramPhase.CONTROLS
            out.control_count += 1
        elif kind == RigEventKind.CHAIN:
            if state.phase == RigProgramPhase.CONTROLS:
                out.phase = RigProgramPhase.CHAINS
            out.chain_count += 1
        elif kind == RigEventKind.CONSTRAINT:
            if state.phase == RigProgramPhase.CHAINS:
                out.phase = RigProgramPhase.CONSTRAINTS
            out.constraint_count += 1
        elif kind == RigEventKind.ANCHOR:
            if state.phase == RigProgramPhase.CONSTRAINTS:
                out.phase = RigProgramPhase.ANCHORS
            out.anchor_count += 1
        elif kind == RigEventKind.SKIN_SWITCH:
            out.phase = RigProgramPhase.SKIN
        elif kind == RigEventKind.SKIN_TOKEN:
            out.skin_token_count += 1
        elif kind == RigEventKind.EOS:
            out.phase = RigProgramPhase.DONE
        return out

    @staticmethod
    def mask_kind_logits(logits: Tensor, state: RigProgramGrammarState) -> Tensor:
        if logits.shape[-1] != len(RigEventKind):
            raise ValueError("AXIS_V6_KIND_VOCAB_DRIFT")
        mask = torch.full_like(logits, float("-inf"))
        for kind in AXISRigProgramGrammarV6.allowed_kinds(state):
            mask[..., int(kind)] = 0.0
        return logits + mask


@dataclass
class AXISV6TeacherSequence:
    kind_ids: Tensor
    coord_indices: Tensor
    pointer_a: Tensor
    pointer_b: Tensor
    aux_type_ids: Tensor
    dof_masks: Tensor
    skin_token_ids: Tensor
    attention_mask: Tensor

    def validate(self, config: AXISV6Config) -> None:
        if self.kind_ids.ndim != 2:
            raise ValueError("AXIS_V6_TEACHER_KIND_RANK")
        b, s = self.kind_ids.shape
        expected2 = (b, s)
        if self.coord_indices.shape != (b, s, 3):
            raise ValueError("AXIS_V6_TEACHER_COORD_SHAPE")
        for value in (self.pointer_a, self.pointer_b, self.aux_type_ids, self.skin_token_ids, self.attention_mask):
            if value.shape != expected2:
                raise ValueError("AXIS_V6_TEACHER_FIELD_SHAPE")
        if self.dof_masks.shape != (b, s, len(DOF_LABELS)):
            raise ValueError("AXIS_V6_TEACHER_DOF_SHAPE")
        if s > config.max_sequence_length:
            raise ValueError("AXIS_V6_TEACHER_SEQUENCE_RESOURCE_GUARD")


class AXISRigProgramCausalV6(nn.Module):
    architecture_id = AXIS_V6_ARCHITECTURE_ID

    def __init__(self, config: AXISV6Config):
        super().__init__()
        self.config = config
        h = config.hidden_dim
        self.kind_embedding = nn.Embedding(len(RigEventKind), h)
        self.coord_embedding = nn.Embedding(config.coord_bins + 1, h)
        self.pointer_embedding = nn.Embedding(config.max_pointer_entities + 1, h)
        self.aux_embedding = nn.Embedding(max(len(CONTROL_KINDS), len(CONSTRAINT_OPERATORS)) + 1, h)
        self.skin_embedding = nn.Embedding(config.skin_vocab_size + 1, h)
        self.position_embedding = nn.Embedding(config.max_sequence_length, h)
        self.dof_projection = nn.Linear(len(DOF_LABELS), h, bias=False)
        layer = nn.TransformerEncoderLayer(
            d_model=h, nhead=config.heads, dim_feedforward=h * config.ff_mult,
            dropout=config.dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.causal = nn.TransformerEncoder(layer, num_layers=config.layers, norm=nn.LayerNorm(h))
        self.scene_projection = nn.Linear(config.scene_dim, h)
        self.scene_cross_attention = nn.MultiheadAttention(h, config.heads, dropout=config.dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(h)
        self.kind_head = nn.Linear(h, len(RigEventKind))
        self.coord_head = nn.Linear(h, 3 * config.coord_bins)
        self.pointer_a_head = nn.Linear(h, config.max_pointer_entities)
        self.pointer_b_head = nn.Linear(h, config.max_pointer_entities)
        self.control_kind_head = nn.Linear(h, len(CONTROL_KINDS))
        self.constraint_operator_head = nn.Linear(h, len(CONSTRAINT_OPERATORS))
        self.dof_head = nn.Linear(h, len(DOF_LABELS))
        self.skin_token_head = nn.Linear(h, config.skin_vocab_size)

    @staticmethod
    def _shift_for_embedding(values: Tensor, max_value: int) -> Tensor:
        return (values.to(torch.long) + 1).clamp(min=0, max=max_value)

    def _teacher_embeddings(self, teacher: AXISV6TeacherSequence) -> Tensor:
        cfg = self.config
        teacher.validate(cfg)
        b, s = teacher.kind_ids.shape
        pos = torch.arange(s, device=teacher.kind_ids.device)[None].expand(b, -1)
        x = self.kind_embedding(teacher.kind_ids.to(torch.long)) + self.position_embedding(pos)
        coords = self._shift_for_embedding(teacher.coord_indices, cfg.coord_bins)
        x = x + self.coord_embedding(coords).sum(dim=2)
        x = x + self.pointer_embedding(self._shift_for_embedding(teacher.pointer_a, cfg.max_pointer_entities))
        x = x + self.pointer_embedding(self._shift_for_embedding(teacher.pointer_b, cfg.max_pointer_entities))
        aux_cap = self.aux_embedding.num_embeddings - 1
        x = x + self.aux_embedding(self._shift_for_embedding(teacher.aux_type_ids, aux_cap))
        x = x + self.skin_embedding(self._shift_for_embedding(teacher.skin_token_ids, cfg.skin_vocab_size))
        x = x + self.dof_projection(teacher.dof_masks.to(dtype=x.dtype))
        return x

    def forward(self, scene_memory: Tensor, teacher: AXISV6TeacherSequence) -> dict[str, Tensor]:
        if scene_memory.ndim != 3 or scene_memory.shape[-1] != self.config.scene_dim:
            raise ValueError("AXIS_V6_SCENE_MEMORY_SHAPE_DRIFT")
        if scene_memory.shape[0] != teacher.kind_ids.shape[0]:
            raise ValueError("AXIS_V6_SCENE_BATCH_DRIFT")
        x = self._teacher_embeddings(teacher)
        s = x.shape[1]
        causal_mask = torch.full((s, s), float("-inf"), device=x.device, dtype=x.dtype)
        causal_mask = torch.triu(causal_mask, diagonal=1)
        x = self.causal(x, mask=causal_mask, src_key_padding_mask=~teacher.attention_mask.to(torch.bool))
        scene = self.scene_projection(scene_memory)
        cross, _ = self.scene_cross_attention(x, scene, scene, need_weights=False)
        x = self.cross_norm(x + cross)
        return {
            "kind_logits": self.kind_head(x),
            "coord_logits": self.coord_head(x).view(*x.shape[:2], 3, self.config.coord_bins),
            "pointer_a_logits": self.pointer_a_head(x),
            "pointer_b_logits": self.pointer_b_head(x),
            "control_kind_logits": self.control_kind_head(x),
            "constraint_operator_logits": self.constraint_operator_head(x),
            "dof_logits": self.dof_head(x),
            "skin_token_logits": self.skin_token_head(x),
            "hidden": x,
        }


__all__ = [
    "AXISRigProgramCausalV6", "AXISRigProgramGrammarV6", "AXISV6Config", "AXISV6TeacherSequence",
    "AXIS_V6_ARCHITECTURE_ID", "AXIS_V6_COORD_BINS", "CONSTRAINT_OPERATORS", "CONTROL_KINDS",
    "DOF_LABELS", "RigEventKind", "RigProgramGrammarState", "RigProgramPhase",
]
