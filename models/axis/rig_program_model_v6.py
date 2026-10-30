from __future__ import annotations

"""Shipping-target AXIS V6 causal rig-program model.

The model emits a typed mechanical program, not merely a skeleton. Explicit
section tokens make the language unambiguous while allowing zero-cardinality
optional sections. MIRA skin tokens share the same causal state.

Grammar enforcement belongs here at generation time. Compiler never completes,
reparents, repairs, or searches the generated program.
"""

from dataclasses import dataclass
from enum import IntEnum
import math
from typing import Iterable, Sequence

import torch
from torch import Tensor, nn


AXIS_V6_PRODUCT_ARCHITECTURE_ID = "RealSaS.AXIS.RigProgram.TypedCausal.ControlSkin.v6_0"
AXIS_V6_COORD_BINS = 256
AXIS_V6_SCALAR_COUNT = 4

CONTROL_KINDS_V6 = (
    "GLOBAL",
    "ROOT",
    "FK",
    "IK_EFFECTOR",
    "IK_POLE",
    "SPLINE",
    "AIM_TARGET",
    "SOCKET_CONTROL",
)
CONSTRAINT_OPERATORS_V6 = (
    "IK",
    "AIM",
    "LIMIT_ROTATION",
    "LIMIT_TRANSLATION",
    "FOLLOW",
    "COPY_TRANSFORM",
    "PATH",
    "STRETCH",
)
DOF_LABELS_V6 = ("TX", "TY", "TZ", "RX", "RY", "RZ", "SX", "SY", "SZ")


class RigTokenKindV6(IntEnum):
    PAD = 0
    BOS = 1
    JOINT = 2
    FRAMES_START = 3
    FRAME = 4
    CONTROLS_START = 5
    CONTROL = 6
    CHAINS_START = 7
    CHAIN = 8
    CONSTRAINTS_START = 9
    CONSTRAINT = 10
    ANCHORS_START = 11
    ANCHOR = 12
    SKIN_START = 13
    SKIN_TOKEN = 14
    EOS = 15


class RigSectionV6(IntEnum):
    START = 0
    JOINTS = 1
    FRAMES = 2
    CONTROLS = 3
    CHAINS = 4
    CONSTRAINTS = 5
    ANCHORS = 6
    SKIN = 7
    DONE = 8


@dataclass(frozen=True)
class AXISRigProgramConfigV6:
    scene_dim: int
    hidden_dim: int = 384
    layers: int = 8
    heads: int = 8
    ff_mult: int = 4
    max_sequence_length: int = 4096
    max_pointer_entities: int = 256
    coord_bins: int = AXIS_V6_COORD_BINS
    coord_min: float = -1.0
    coord_max: float = 1.0
    skin_vocab_size: int = 625
    dropout: float = 0.0

    def to_dict(self) -> dict:
        return {
            "scene_dim": self.scene_dim,
            "hidden_dim": self.hidden_dim,
            "layers": self.layers,
            "heads": self.heads,
            "ff_mult": self.ff_mult,
            "max_sequence_length": self.max_sequence_length,
            "max_pointer_entities": self.max_pointer_entities,
            "coord_bins": self.coord_bins,
            "coord_min": self.coord_min,
            "coord_max": self.coord_max,
            "skin_vocab_size": self.skin_vocab_size,
            "dropout": self.dropout,
        }

    @classmethod
    def from_dict(cls, value: dict) -> "AXISRigProgramConfigV6":
        return cls(**{k: value[k] for k in cls.__dataclass_fields__ if k in value})


@dataclass(frozen=True)
class RigProgramEventV6:
    kind: RigTokenKindV6
    coord_indices: tuple[int, int, int] = (-1, -1, -1)
    pointer_a: int = -1
    pointer_b: int = -1
    pointer_c: int = -1
    aux_type_id: int = -1
    dof_mask: tuple[float, ...] = (0.0,) * len(DOF_LABELS_V6)
    frame_6d: tuple[float, ...] = (0.0,) * 6
    scalar_params: tuple[float, ...] = (0.0,) * AXIS_V6_SCALAR_COUNT
    skin_token_id: int = -1
    anchor_parent_kind: int = -1  # 0=joint, 1=control


@dataclass(frozen=True)
class RigGrammarStateV6:
    section: RigSectionV6 = RigSectionV6.START
    joint_count: int = 0
    frame_count: int = 0
    control_count: int = 0
    chain_count: int = 0
    constraint_count: int = 0
    anchor_count: int = 0
    skin_token_count: int = 0


class AXISRigProgramGrammarV6Product:
    @staticmethod
    def allowed_kinds(state: RigGrammarStateV6, *, tokens_per_joint: int) -> tuple[RigTokenKindV6, ...]:
        s = state.section
        if s == RigSectionV6.START:
            return (RigTokenKindV6.BOS,)
        if s == RigSectionV6.JOINTS:
            # At least one deform joint is required.
            return ((RigTokenKindV6.JOINT,) if state.joint_count == 0 else
                    (RigTokenKindV6.JOINT, RigTokenKindV6.FRAMES_START))
        if s == RigSectionV6.FRAMES:
            # Frame coverage is a Compiler court, not grammar synthesis. The model
            # may transition early and fail qualification.
            return (RigTokenKindV6.FRAME, RigTokenKindV6.CONTROLS_START)
        if s == RigSectionV6.CONTROLS:
            return (RigTokenKindV6.CONTROL, RigTokenKindV6.CHAINS_START)
        if s == RigSectionV6.CHAINS:
            return (RigTokenKindV6.CHAIN, RigTokenKindV6.CONSTRAINTS_START)
        if s == RigSectionV6.CONSTRAINTS:
            return (RigTokenKindV6.CONSTRAINT, RigTokenKindV6.ANCHORS_START)
        if s == RigSectionV6.ANCHORS:
            return (RigTokenKindV6.ANCHOR, RigTokenKindV6.SKIN_START)
        if s == RigSectionV6.SKIN:
            required = state.joint_count * int(tokens_per_joint)
            if state.skin_token_count < required:
                return (RigTokenKindV6.SKIN_TOKEN,)
            return (RigTokenKindV6.EOS,)
        return ()

    @staticmethod
    def advance(state: RigGrammarStateV6, kind: RigTokenKindV6, *, tokens_per_joint: int) -> RigGrammarStateV6:
        if kind not in AXISRigProgramGrammarV6Product.allowed_kinds(state, tokens_per_joint=tokens_per_joint):
            raise ValueError(f"AXIS_V6_PRODUCT_GRAMMAR_ILLEGAL:{state.section.name}->{kind.name}")
        d = dict(state.__dict__)
        if kind == RigTokenKindV6.BOS:
            d["section"] = RigSectionV6.JOINTS
        elif kind == RigTokenKindV6.JOINT:
            d["joint_count"] += 1
        elif kind == RigTokenKindV6.FRAMES_START:
            d["section"] = RigSectionV6.FRAMES
        elif kind == RigTokenKindV6.FRAME:
            d["frame_count"] += 1
        elif kind == RigTokenKindV6.CONTROLS_START:
            d["section"] = RigSectionV6.CONTROLS
        elif kind == RigTokenKindV6.CONTROL:
            d["control_count"] += 1
        elif kind == RigTokenKindV6.CHAINS_START:
            d["section"] = RigSectionV6.CHAINS
        elif kind == RigTokenKindV6.CHAIN:
            d["chain_count"] += 1
        elif kind == RigTokenKindV6.CONSTRAINTS_START:
            d["section"] = RigSectionV6.CONSTRAINTS
        elif kind == RigTokenKindV6.CONSTRAINT:
            d["constraint_count"] += 1
        elif kind == RigTokenKindV6.ANCHORS_START:
            d["section"] = RigSectionV6.ANCHORS
        elif kind == RigTokenKindV6.ANCHOR:
            d["anchor_count"] += 1
        elif kind == RigTokenKindV6.SKIN_START:
            d["section"] = RigSectionV6.SKIN
        elif kind == RigTokenKindV6.SKIN_TOKEN:
            d["skin_token_count"] += 1
        elif kind == RigTokenKindV6.EOS:
            d["section"] = RigSectionV6.DONE
        return RigGrammarStateV6(**d)

    @staticmethod
    def mask(logits: Tensor, state: RigGrammarStateV6, *, tokens_per_joint: int) -> Tensor:
        if logits.shape[-1] != len(RigTokenKindV6):
            raise ValueError("AXIS_V6_PRODUCT_KIND_VOCAB_DRIFT")
        mask = torch.full_like(logits, float("-inf"))
        for kind in AXISRigProgramGrammarV6Product.allowed_kinds(state, tokens_per_joint=tokens_per_joint):
            mask[..., int(kind)] = 0.0
        return logits + mask


@dataclass
class AXISSequenceV6:
    kind_ids: Tensor
    coord_indices: Tensor
    pointer_a: Tensor
    pointer_b: Tensor
    pointer_c: Tensor
    aux_type_ids: Tensor
    dof_masks: Tensor
    frame_6d: Tensor
    scalar_params: Tensor
    skin_token_ids: Tensor
    anchor_parent_kind: Tensor
    attention_mask: Tensor

    def validate(self, config: AXISRigProgramConfigV6) -> None:
        if self.kind_ids.ndim != 2:
            raise ValueError("AXIS_V6_SEQUENCE_KIND_RANK")
        b, s = self.kind_ids.shape
        if s > config.max_sequence_length:
            raise ValueError("AXIS_V6_SEQUENCE_RESOURCE_GUARD")
        for value in (self.pointer_a, self.pointer_b, self.pointer_c, self.aux_type_ids,
                      self.skin_token_ids, self.anchor_parent_kind, self.attention_mask):
            if value.shape != (b, s):
                raise ValueError("AXIS_V6_SEQUENCE_FIELD_SHAPE")
        if self.coord_indices.shape != (b, s, 3):
            raise ValueError("AXIS_V6_SEQUENCE_COORD_SHAPE")
        if self.dof_masks.shape != (b, s, len(DOF_LABELS_V6)):
            raise ValueError("AXIS_V6_SEQUENCE_DOF_SHAPE")
        if self.frame_6d.shape != (b, s, 6):
            raise ValueError("AXIS_V6_SEQUENCE_FRAME_SHAPE")
        if self.scalar_params.shape != (b, s, AXIS_V6_SCALAR_COUNT):
            raise ValueError("AXIS_V6_SEQUENCE_SCALAR_SHAPE")

    @classmethod
    def from_events(cls, events: Sequence[RigProgramEventV6], *, device: torch.device) -> "AXISSequenceV6":
        if not events:
            raise ValueError("AXIS_V6_EMPTY_EVENT_PREFIX")
        def row(name):
            return [getattr(e, name) for e in events]
        return cls(
            kind_ids=torch.tensor([[int(e.kind) for e in events]], device=device, dtype=torch.long),
            coord_indices=torch.tensor([[e.coord_indices for e in events]], device=device, dtype=torch.long),
            pointer_a=torch.tensor([row("pointer_a")], device=device, dtype=torch.long),
            pointer_b=torch.tensor([row("pointer_b")], device=device, dtype=torch.long),
            pointer_c=torch.tensor([row("pointer_c")], device=device, dtype=torch.long),
            aux_type_ids=torch.tensor([row("aux_type_id")], device=device, dtype=torch.long),
            dof_masks=torch.tensor([[e.dof_mask for e in events]], device=device, dtype=torch.float32),
            frame_6d=torch.tensor([[e.frame_6d for e in events]], device=device, dtype=torch.float32),
            scalar_params=torch.tensor([[e.scalar_params for e in events]], device=device, dtype=torch.float32),
            skin_token_ids=torch.tensor([row("skin_token_id")], device=device, dtype=torch.long),
            anchor_parent_kind=torch.tensor([row("anchor_parent_kind")], device=device, dtype=torch.long),
            attention_mask=torch.ones((1, len(events)), device=device, dtype=torch.bool),
        )


class AXISRigProgramModelV6(nn.Module):
    architecture_id = AXIS_V6_PRODUCT_ARCHITECTURE_ID

    def __init__(self, config: AXISRigProgramConfigV6):
        super().__init__()
        self.config = config
        h = config.hidden_dim
        if h % config.heads:
            raise ValueError("AXIS_V6_HIDDEN_HEAD_DIVISIBILITY")
        self.kind_embedding = nn.Embedding(len(RigTokenKindV6), h)
        self.coord_embedding = nn.Embedding(config.coord_bins + 1, h)
        self.pointer_embedding = nn.Embedding(config.max_pointer_entities + 1, h)
        self.aux_embedding = nn.Embedding(max(len(CONTROL_KINDS_V6), len(CONSTRAINT_OPERATORS_V6)) + 1, h)
        self.skin_embedding = nn.Embedding(config.skin_vocab_size + 1, h)
        self.anchor_parent_embedding = nn.Embedding(3, h)  # absent/joint/control
        self.position_embedding = nn.Embedding(config.max_sequence_length, h)
        self.dof_projection = nn.Linear(len(DOF_LABELS_V6), h, bias=False)
        self.frame_projection = nn.Linear(6, h, bias=False)
        self.scalar_projection = nn.Linear(AXIS_V6_SCALAR_COUNT, h, bias=False)

        layer = nn.TransformerEncoderLayer(
            d_model=h, nhead=config.heads, dim_feedforward=h * config.ff_mult,
            dropout=config.dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.causal = nn.TransformerEncoder(layer, num_layers=config.layers, norm=nn.LayerNorm(h))
        self.scene_projection = nn.Linear(config.scene_dim, h)
        self.scene_cross_attention = nn.MultiheadAttention(h, config.heads, dropout=config.dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(h)

        self.kind_head = nn.Linear(h, len(RigTokenKindV6))
        self.coord_head = nn.Linear(h, 3 * config.coord_bins)
        # Pointer vocab reserves 0=NULL, 1..N=entity index 0..N-1.
        self.pointer_a_head = nn.Linear(h, config.max_pointer_entities + 1)
        self.pointer_b_head = nn.Linear(h, config.max_pointer_entities + 1)
        self.pointer_c_head = nn.Linear(h, config.max_pointer_entities + 1)
        self.control_kind_head = nn.Linear(h, len(CONTROL_KINDS_V6))
        self.constraint_operator_head = nn.Linear(h, len(CONSTRAINT_OPERATORS_V6))
        self.dof_head = nn.Linear(h, len(DOF_LABELS_V6))
        self.frame_6d_head = nn.Linear(h, 6)
        self.scalar_head = nn.Linear(h, AXIS_V6_SCALAR_COUNT)
        self.skin_token_head = nn.Linear(h, config.skin_vocab_size)
        self.anchor_parent_head = nn.Linear(h, 2)

    @staticmethod
    def _shift(values: Tensor, cap: int) -> Tensor:
        return (values.to(torch.long) + 1).clamp(min=0, max=cap)

    def _embed(self, seq: AXISSequenceV6) -> Tensor:
        cfg = self.config
        seq.validate(cfg)
        b, s = seq.kind_ids.shape
        pos = torch.arange(s, device=seq.kind_ids.device)[None].expand(b, -1)
        x = self.kind_embedding(seq.kind_ids) + self.position_embedding(pos)
        x = x + self.coord_embedding(self._shift(seq.coord_indices, cfg.coord_bins)).sum(dim=2)
        for ptr in (seq.pointer_a, seq.pointer_b, seq.pointer_c):
            x = x + self.pointer_embedding(self._shift(ptr, cfg.max_pointer_entities))
        x = x + self.aux_embedding(self._shift(seq.aux_type_ids, self.aux_embedding.num_embeddings - 1))
        x = x + self.skin_embedding(self._shift(seq.skin_token_ids, cfg.skin_vocab_size))
        x = x + self.anchor_parent_embedding(self._shift(seq.anchor_parent_kind, 2))
        x = x + self.dof_projection(seq.dof_masks.to(dtype=x.dtype))
        x = x + self.frame_projection(seq.frame_6d.to(dtype=x.dtype))
        x = x + self.scalar_projection(seq.scalar_params.to(dtype=x.dtype))
        return x

    def forward(self, scene_memory: Tensor, seq: AXISSequenceV6) -> dict[str, Tensor]:
        if scene_memory.ndim != 3 or scene_memory.shape[-1] != self.config.scene_dim:
            raise ValueError("AXIS_V6_SCENE_MEMORY_SHAPE_DRIFT")
        if scene_memory.shape[0] != seq.kind_ids.shape[0]:
            raise ValueError("AXIS_V6_SCENE_BATCH_DRIFT")
        x = self._embed(seq)
        s = x.shape[1]
        mask = torch.full((s, s), float("-inf"), device=x.device, dtype=x.dtype)
        mask = torch.triu(mask, diagonal=1)
        x = self.causal(x, mask=mask, src_key_padding_mask=~seq.attention_mask.to(torch.bool))
        scene = self.scene_projection(scene_memory.to(dtype=x.dtype))
        cross, _ = self.scene_cross_attention(x, scene, scene, need_weights=False)
        x = self.cross_norm(x + cross)
        return {
            "kind_logits": self.kind_head(x),
            "coord_logits": self.coord_head(x).view(*x.shape[:2], 3, self.config.coord_bins),
            "pointer_a_logits": self.pointer_a_head(x),
            "pointer_b_logits": self.pointer_b_head(x),
            "pointer_c_logits": self.pointer_c_head(x),
            "control_kind_logits": self.control_kind_head(x),
            "constraint_operator_logits": self.constraint_operator_head(x),
            "dof_logits": self.dof_head(x),
            "frame_6d": self.frame_6d_head(x),
            "scalar_params": self.scalar_head(x),
            "skin_token_logits": self.skin_token_head(x),
            "anchor_parent_logits": self.anchor_parent_head(x),
            "hidden": x,
        }

    @staticmethod
    def _pointer(logits: Tensor, count: int, *, allow_null: bool) -> int:
        if count < 0:
            raise ValueError("AXIS_V6_POINTER_COUNT_INVALID")
        available = count + 1  # includes NULL at index 0
        if available > logits.shape[-1]:
            raise RuntimeError("AXIS_V6_POINTER_RESOURCE_GUARD")
        values = logits[..., :available].clone()
        if not allow_null:
            values[..., 0] = float("-inf")
        raw = int(torch.argmax(values, dim=-1).item())
        return raw - 1

    @torch.no_grad()
    def generate_events(
        self,
        scene_memory: Tensor,
        *,
        tokens_per_joint: int,
        max_steps: int | None = None,
    ) -> tuple[RigProgramEventV6, ...]:
        self.eval()
        if scene_memory.ndim != 3 or scene_memory.shape[0] != 1:
            raise ValueError("AXIS_V6_GENERATION_REQUIRES_BATCH_ONE")
        if tokens_per_joint < 1:
            raise ValueError("AXIS_V6_TOKENS_PER_JOINT_INVALID")
        limit = min(int(max_steps or self.config.max_sequence_length), self.config.max_sequence_length)
        if limit < 2:
            raise ValueError("AXIS_V6_GENERATION_LIMIT_INVALID")

        device = scene_memory.device
        events: list[RigProgramEventV6] = [RigProgramEventV6(RigTokenKindV6.BOS)]
        state = AXISRigProgramGrammarV6Product.advance(
            RigGrammarStateV6(), RigTokenKindV6.BOS, tokens_per_joint=tokens_per_joint
        )

        for _ in range(limit - 1):
            seq = AXISSequenceV6.from_events(events, device=device)
            out = self(scene_memory, seq)
            kind_logits = AXISRigProgramGrammarV6Product.mask(
                out["kind_logits"][0, -1], state, tokens_per_joint=tokens_per_joint
            )
            kind = RigTokenKindV6(int(torch.argmax(kind_logits).item()))
            last = {key: value[0, -1] for key, value in out.items() if key != "hidden"}

            coord = (-1, -1, -1)
            pa = pb = pc = aux = skin = anchor_kind = -1
            dof = (0.0,) * len(DOF_LABELS_V6)
            frame = (0.0,) * 6
            scalars = (0.0,) * AXIS_V6_SCALAR_COUNT

            if kind in (RigTokenKindV6.JOINT, RigTokenKindV6.CONTROL):
                coord = tuple(int(v) for v in torch.argmax(last["coord_logits"], dim=-1).tolist())  # type: ignore[assignment]
            if kind == RigTokenKindV6.JOINT:
                pa = -1 if state.joint_count == 0 else self._pointer(last["pointer_a_logits"], state.joint_count, allow_null=True)
            elif kind == RigTokenKindV6.FRAME:
                pa = self._pointer(last["pointer_a_logits"], state.joint_count, allow_null=False)
                frame = tuple(float(v) for v in last["frame_6d"].tolist())
                dof = tuple(float(v) for v in (torch.sigmoid(last["dof_logits"]) >= 0.5).to(torch.float32).tolist())
            elif kind == RigTokenKindV6.CONTROL:
                aux = int(torch.argmax(last["control_kind_logits"]).item())
                pa = self._pointer(last["pointer_a_logits"], state.joint_count, allow_null=True)
                pb = self._pointer(last["pointer_b_logits"], state.control_count, allow_null=True)
            elif kind == RigTokenKindV6.CHAIN:
                pa = self._pointer(last["pointer_a_logits"], state.joint_count, allow_null=False)
                pb = self._pointer(last["pointer_b_logits"], state.joint_count, allow_null=False)
            elif kind == RigTokenKindV6.CONSTRAINT:
                aux = int(torch.argmax(last["constraint_operator_logits"]).item())
                pa = self._pointer(last["pointer_a_logits"], state.control_count, allow_null=True)
                pb = self._pointer(last["pointer_b_logits"], state.chain_count, allow_null=True)
                pc = self._pointer(last["pointer_c_logits"], state.joint_count, allow_null=True)
                scalars = tuple(float(v) for v in last["scalar_params"].tolist())
            elif kind == RigTokenKindV6.ANCHOR:
                anchor_kind = int(torch.argmax(last["anchor_parent_logits"]).item())
                if anchor_kind == 0:
                    pa = self._pointer(last["pointer_a_logits"], state.joint_count, allow_null=False)
                else:
                    pb = self._pointer(last["pointer_b_logits"], state.control_count, allow_null=False)
                coord = tuple(int(v) for v in torch.argmax(last["coord_logits"], dim=-1).tolist())  # type: ignore[assignment]
            elif kind == RigTokenKindV6.SKIN_TOKEN:
                skin = int(torch.argmax(last["skin_token_logits"]).item())

            event = RigProgramEventV6(
                kind=kind, coord_indices=coord, pointer_a=pa, pointer_b=pb, pointer_c=pc,
                aux_type_id=aux, dof_mask=dof, frame_6d=frame, scalar_params=scalars,
                skin_token_id=skin, anchor_parent_kind=anchor_kind,
            )
            events.append(event)
            state = AXISRigProgramGrammarV6Product.advance(state, kind, tokens_per_joint=tokens_per_joint)
            if state.section == RigSectionV6.DONE:
                return tuple(events)
        raise RuntimeError("AXIS_V6_GENERATION_RESOURCE_LIMIT_EXHAUSTED")


def rotation_6d_to_basis_v6(value: Sequence[float]) -> tuple[tuple[float, float, float], ...]:
    """Continuous 6D rotation representation -> right-handed orthonormal basis."""
    if len(value) != 6 or not all(math.isfinite(float(v)) for v in value):
        raise ValueError("AXIS_V6_FRAME6D_INVALID")
    a = torch.tensor(value[:3], dtype=torch.float64)
    b = torch.tensor(value[3:], dtype=torch.float64)
    na = torch.linalg.norm(a)
    if float(na) <= 1e-10:
        raise ValueError("AXIS_V6_FRAME6D_FIRST_AXIS_DEGENERATE")
    x = a / na
    b_ortho = b - torch.dot(x, b) * x
    nb = torch.linalg.norm(b_ortho)
    if float(nb) <= 1e-10:
        raise ValueError("AXIS_V6_FRAME6D_SECOND_AXIS_DEGENERATE")
    y = b_ortho / nb
    z = torch.linalg.cross(x, y)
    rows = torch.stack((x, y, z), dim=0).tolist()
    return tuple(tuple(float(v) for v in row) for row in rows)


__all__ = [
    "AXISRigProgramConfigV6", "AXISRigProgramGrammarV6Product", "AXISRigProgramModelV6",
    "AXISSequenceV6", "AXIS_V6_PRODUCT_ARCHITECTURE_ID", "CONTROL_KINDS_V6",
    "CONSTRAINT_OPERATORS_V6", "DOF_LABELS_V6", "RigGrammarStateV6", "RigProgramEventV6",
    "RigSectionV6", "RigTokenKindV6", "rotation_6d_to_basis_v6",
]
