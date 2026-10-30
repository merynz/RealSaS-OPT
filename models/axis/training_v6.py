from __future__ import annotations

"""Teacher serialization and typed causal objective for AXIS V6.

Training and product runtime share exactly the same event language. The teacher is
a qualified RigProgram proposal plus the exact MIRA discrete-code proposal bound
to it. No notebook-only target schema exists.
"""

from dataclasses import dataclass
from typing import Iterable, Sequence
import math

import torch
from torch import Tensor
import torch.nn.functional as F

from compiler.realsas_compiler_core.rig_program_ir_v1 import RigProgramProposalIR
from compiler.realsas_compiler_core.skin_token_ir_v1 import SkinTokenProposalIR, validate_skin_token_proposal_v1
from models.axis.rig_program_model_v6 import (
    AXISRigProgramConfigV6,
    AXISRigProgramModelV6,
    AXISSequenceV6,
    AXIS_V6_SCALAR_COUNT,
    CONSTRAINT_OPERATORS_V6,
    CONTROL_KINDS_V6,
    DOF_LABELS_V6,
    RigProgramEventV6,
    RigTokenKindV6,
)


@dataclass(frozen=True)
class AXISV6LossWeights:
    kind: float = 1.0
    coordinate: float = 1.0
    pointer: float = 1.0
    control_kind: float = 1.0
    constraint_operator: float = 1.0
    dof: float = 0.25
    frame: float = 0.5
    constraint_scalar: float = 0.25
    skin_code: float = 1.0
    anchor_parent: float = 0.5


@dataclass
class AXISV6TrainingBatch:
    inputs: AXISSequenceV6
    targets: AXISSequenceV6


def _coord_index(config: AXISRigProgramConfigV6, value: float) -> int:
    x = float(value)
    if not math.isfinite(x):
        raise ValueError("AXIS_V6_TEACHER_COORD_NONFINITE")
    lo, hi = float(config.coord_min), float(config.coord_max)
    if not lo < hi:
        raise ValueError("AXIS_V6_TEACHER_COORD_DOMAIN_INVALID")
    tolerance = max(1e-6, abs(hi - lo) * 1e-6)
    if x < lo - tolerance or x > hi + tolerance:
        raise ValueError(f"AXIS_V6_TEACHER_COORD_OUT_OF_DOMAIN:{x}:{lo}:{hi}")
    x = min(max(x, lo), hi)
    alpha = (x - lo) / (hi - lo)
    return int(round(alpha * float(config.coord_bins - 1)))


def _quantize_world(config, point, *, center, scale):
    p = tuple((float(point[i]) - float(center[i])) / float(scale) for i in range(3))
    return tuple(_coord_index(config, v) for v in p)


def _quantize_local(config, point, *, scale):
    p = tuple(float(point[i]) / float(scale) for i in range(3))
    return tuple(_coord_index(config, v) for v in p)


def _basis_to_6d(basis) -> tuple[float, ...]:
    rows = tuple(tuple(float(v) for v in row) for row in basis)
    if len(rows) != 3 or any(len(row) != 3 for row in rows):
        raise ValueError("AXIS_V6_TEACHER_FRAME_SHAPE_INVALID")
    values = rows[0] + rows[1]
    if not all(math.isfinite(v) for v in values):
        raise ValueError("AXIS_V6_TEACHER_FRAME_NONFINITE")
    return values


def _topological_controls(proposal: RigProgramProposalIR):
    by_id = {str(row.proposal_control_id): row for row in proposal.controls}
    if len(by_id) != len(proposal.controls):
        raise ValueError("AXIS_V6_TEACHER_CONTROL_ID_DUPLICATE")
    ordered = []
    emitted: set[str] = set()
    remaining = list(proposal.controls)
    while remaining:
        next_remaining = []
        progress = False
        for row in remaining:
            parent = row.parent_proposal_control_id
            if parent is None or str(parent) in emitted:
                ordered.append(row)
                emitted.add(str(row.proposal_control_id))
                progress = True
            else:
                next_remaining.append(row)
        if not progress:
            raise ValueError("AXIS_V6_TEACHER_CONTROL_GRAPH_NOT_CAUSAL")
        remaining = next_remaining
    return tuple(ordered)


def _constraint_scalars(parameters: dict) -> tuple[float, ...]:
    allowed = {f"p{i}" for i in range(AXIS_V6_SCALAR_COUNT)}
    extra = set(map(str, parameters)) - allowed
    if extra:
        raise ValueError("AXIS_V6_TEACHER_CONSTRAINT_PARAMETER_SCHEMA_UNBOUND:" + ",".join(sorted(extra)))
    values = tuple(float(parameters.get(f"p{i}", 0.0)) for i in range(AXIS_V6_SCALAR_COUNT))
    if not all(math.isfinite(v) for v in values):
        raise ValueError("AXIS_V6_TEACHER_CONSTRAINT_PARAMETER_NONFINITE")
    return values


def serialize_axis_v6_teacher_events(
    proposal: RigProgramProposalIR,
    code_proposal: SkinTokenProposalIR,
    *,
    config: AXISRigProgramConfigV6,
    normalization_center: Sequence[float],
    normalization_scale: float,
) -> tuple[RigProgramEventV6, ...]:
    """Serialize product proposals to the exact free-running AXIS event language."""
    center = tuple(float(v) for v in normalization_center)
    scale = float(normalization_scale)
    if len(center) != 3 or not all(math.isfinite(v) for v in center):
        raise ValueError("AXIS_V6_TEACHER_NORMALIZATION_CENTER_INVALID")
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("AXIS_V6_TEACHER_NORMALIZATION_SCALE_INVALID")

    joints = tuple(sorted(proposal.joints, key=lambda row: int(row.sequence_index)))
    joint_ids = tuple(str(row.proposal_joint_id) for row in joints)
    if len(set(joint_ids)) != len(joint_ids) or not joint_ids:
        raise ValueError("AXIS_V6_TEACHER_JOINT_ID_INVALID")
    joint_index = {jid: i for i, jid in enumerate(joint_ids)}
    for i, row in enumerate(joints):
        if int(row.sequence_index) != i:
            raise ValueError("AXIS_V6_TEACHER_JOINT_SEQUENCE_NOT_DENSE")
        parent = row.parent_proposal_joint_id
        if parent is not None and (str(parent) not in joint_index or joint_index[str(parent)] >= i):
            raise ValueError("AXIS_V6_TEACHER_JOINT_PARENT_NOT_CAUSAL")

    validate_skin_token_proposal_v1(
        code_proposal,
        expected_proposal_joint_ids=joint_ids,
        expected_rig_program_proposal_hash=proposal.proposal_lineage_hash,
    )
    code_by_joint = {str(row.proposal_joint_id): tuple(map(int, row.token_ids)) for row in code_proposal.rows}

    frame_by = {str(row.proposal_joint_id): row for row in proposal.local_frames}
    if set(frame_by) != set(joint_ids):
        raise ValueError("AXIS_V6_TEACHER_FRAME_COVERAGE_DRIFT")

    events: list[RigProgramEventV6] = [RigProgramEventV6(RigTokenKindV6.BOS)]
    for i, row in enumerate(joints):
        parent = row.parent_proposal_joint_id
        events.append(RigProgramEventV6(
            RigTokenKindV6.JOINT,
            coord_indices=_quantize_world(config, row.P, center=center, scale=scale),
            pointer_a=-1 if parent is None else joint_index[str(parent)],
        ))

    events.append(RigProgramEventV6(RigTokenKindV6.FRAMES_START))
    dof_index = {name: i for i, name in enumerate(DOF_LABELS_V6)}
    for jid in joint_ids:
        frame = frame_by[jid]
        unknown_dof = set(map(str, frame.dof_labels)) - set(dof_index)
        if unknown_dof:
            raise ValueError("AXIS_V6_TEACHER_DOF_VOCAB_UNBOUND:" + ",".join(sorted(unknown_dof)))
        mask = tuple(1.0 if name in set(frame.dof_labels) else 0.0 for name in DOF_LABELS_V6)
        events.append(RigProgramEventV6(
            RigTokenKindV6.FRAME,
            pointer_a=joint_index[jid],
            dof_mask=mask,
            frame_6d=_basis_to_6d(frame.basis),
        ))

    events.append(RigProgramEventV6(RigTokenKindV6.CONTROLS_START))
    controls = _topological_controls(proposal)
    control_index: dict[str, int] = {}
    for row in controls:
        cid = str(row.proposal_control_id)
        if str(row.control_kind) not in CONTROL_KINDS_V6:
            raise ValueError("AXIS_V6_TEACHER_CONTROL_KIND_UNBOUND:" + str(row.control_kind))
        parent = row.parent_proposal_control_id
        if parent is not None and str(parent) not in control_index:
            raise ValueError("AXIS_V6_TEACHER_CONTROL_PARENT_NOT_CAUSAL")
        events.append(RigProgramEventV6(
            RigTokenKindV6.CONTROL,
            coord_indices=_quantize_world(config, row.rest_P, center=center, scale=scale),
            pointer_a=-1 if parent is None else control_index[str(parent)],
            aux_type_id=CONTROL_KINDS_V6.index(str(row.control_kind)),
        ))
        control_index[cid] = len(control_index)
        for target in row.target_proposal_joint_ids:
            if str(target) not in joint_index:
                raise ValueError("AXIS_V6_TEACHER_CONTROL_TARGET_UNKNOWN")
            events.append(RigProgramEventV6(RigTokenKindV6.CONTROL_TARGET, pointer_a=joint_index[str(target)]))

    events.append(RigProgramEventV6(RigTokenKindV6.CHAINS_START))
    chain_index: dict[str, int] = {}
    for row in proposal.chains:
        members = tuple(map(str, row.proposal_joint_ids))
        if not members or any(jid not in joint_index for jid in members):
            raise ValueError("AXIS_V6_TEACHER_CHAIN_MEMBER_INVALID")
        for parent, child in zip(members, members[1:]):
            if joints[joint_index[child]].parent_proposal_joint_id != parent:
                raise ValueError("AXIS_V6_TEACHER_CHAIN_NOT_PARENT_CONTIGUOUS")
        events.append(RigProgramEventV6(
            RigTokenKindV6.CHAIN,
            pointer_a=joint_index[members[0]],
            pointer_b=joint_index[members[-1]],
        ))
        chain_index[str(row.proposal_chain_id)] = len(chain_index)

    events.append(RigProgramEventV6(RigTokenKindV6.CONSTRAINTS_START))
    for row in proposal.constraints:
        operator = str(row.operator_id)
        if operator not in CONSTRAINT_OPERATORS_V6:
            raise ValueError("AXIS_V6_TEACHER_CONSTRAINT_OPERATOR_UNBOUND:" + operator)
        chain = row.proposal_chain_id
        if chain is not None and str(chain) not in chain_index:
            raise ValueError("AXIS_V6_TEACHER_CONSTRAINT_CHAIN_UNKNOWN")
        events.append(RigProgramEventV6(
            RigTokenKindV6.CONSTRAINT,
            pointer_a=-1 if chain is None else chain_index[str(chain)],
            aux_type_id=CONSTRAINT_OPERATORS_V6.index(operator),
            scalar_params=_constraint_scalars(dict(row.parameters or {})),
        ))
        for cid in row.input_proposal_control_ids:
            if str(cid) not in control_index:
                raise ValueError("AXIS_V6_TEACHER_CONSTRAINT_CONTROL_UNKNOWN")
            events.append(RigProgramEventV6(RigTokenKindV6.CONSTRAINT_INPUT, pointer_a=control_index[str(cid)]))
        for jid in row.output_proposal_joint_ids:
            if str(jid) not in joint_index:
                raise ValueError("AXIS_V6_TEACHER_CONSTRAINT_JOINT_UNKNOWN")
            events.append(RigProgramEventV6(RigTokenKindV6.CONSTRAINT_OUTPUT, pointer_a=joint_index[str(jid)]))

    events.append(RigProgramEventV6(RigTokenKindV6.ANCHORS_START))
    for row in proposal.attachment_anchors:
        has_joint = row.parent_proposal_joint_id is not None
        has_control = row.parent_proposal_control_id is not None
        if has_joint == has_control:
            raise ValueError("AXIS_V6_TEACHER_ANCHOR_PARENT_INVALID")
        if has_joint:
            jid = str(row.parent_proposal_joint_id)
            if jid not in joint_index:
                raise ValueError("AXIS_V6_TEACHER_ANCHOR_JOINT_UNKNOWN")
            parent_kind, pa, pb = 0, joint_index[jid], -1
        else:
            cid = str(row.parent_proposal_control_id)
            if cid not in control_index:
                raise ValueError("AXIS_V6_TEACHER_ANCHOR_CONTROL_UNKNOWN")
            parent_kind, pa, pb = 1, -1, control_index[cid]
        events.append(RigProgramEventV6(
            RigTokenKindV6.ANCHOR,
            coord_indices=_quantize_local(config, row.local_P, scale=scale),
            pointer_a=pa,
            pointer_b=pb,
            anchor_parent_kind=parent_kind,
        ))

    events.append(RigProgramEventV6(RigTokenKindV6.SKIN_START))
    for jid in joint_ids:
        codes = code_by_joint[jid]
        if len(codes) != code_proposal.tokens_per_joint:
            raise ValueError("AXIS_V6_TEACHER_SKIN_CODE_CARDINALITY_DRIFT")
        for code in codes:
            if code < 0 or code >= config.skin_vocab_size:
                raise ValueError("AXIS_V6_TEACHER_SKIN_CODE_OUT_OF_RANGE")
            events.append(RigProgramEventV6(RigTokenKindV6.SKIN_CODE, skin_code_id=int(code)))
    events.append(RigProgramEventV6(RigTokenKindV6.EOS))
    if len(events) > config.max_sequence_length:
        raise ValueError("AXIS_V6_TEACHER_SEQUENCE_RESOURCE_GUARD")
    return tuple(events)


def _event_batch(events_batch: Sequence[Sequence[RigProgramEventV6]], *, device: torch.device) -> AXISSequenceV6:
    if not events_batch or any(not rows for rows in events_batch):
        raise ValueError("AXIS_V6_TRAINING_BATCH_EMPTY")
    width = max(len(rows) for rows in events_batch)
    pad = RigProgramEventV6(RigTokenKindV6.PAD)
    padded = [list(rows) + [pad] * (width - len(rows)) for rows in events_batch]
    def matrix(name, dtype=torch.long):
        return torch.tensor([[getattr(e, name) for e in rows] for rows in padded], device=device, dtype=dtype)
    return AXISSequenceV6(
        kind_ids=torch.tensor([[int(e.kind) for e in rows] for rows in padded], device=device, dtype=torch.long),
        coord_indices=torch.tensor([[e.coord_indices for e in rows] for rows in padded], device=device, dtype=torch.long),
        pointer_a=matrix("pointer_a"),
        pointer_b=matrix("pointer_b"),
        pointer_c=matrix("pointer_c"),
        aux_type_ids=matrix("aux_type_id"),
        dof_masks=torch.tensor([[e.dof_mask for e in rows] for rows in padded], device=device, dtype=torch.float32),
        frame_6d=torch.tensor([[e.frame_6d for e in rows] for rows in padded], device=device, dtype=torch.float32),
        scalar_params=torch.tensor([[e.scalar_params for e in rows] for rows in padded], device=device, dtype=torch.float32),
        skin_code_ids=matrix("skin_code_id"),
        anchor_parent_kind=matrix("anchor_parent_kind"),
        attention_mask=torch.tensor(
            [[i < len(events_batch[b]) for i in range(width)] for b in range(len(events_batch))],
            device=device, dtype=torch.bool,
        ),
    )


def collate_axis_v6_causal_batch(
    teacher_events: Sequence[Sequence[RigProgramEventV6]], *, device: torch.device
) -> AXISV6TrainingBatch:
    if any(len(rows) < 2 for rows in teacher_events):
        raise ValueError("AXIS_V6_TRAINING_SEQUENCE_TOO_SHORT")
    inputs = _event_batch([rows[:-1] for rows in teacher_events], device=device)
    targets = _event_batch([rows[1:] for rows in teacher_events], device=device)
    return AXISV6TrainingBatch(inputs=inputs, targets=targets)


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    selected = values[mask]
    return selected.mean() if selected.numel() else values.sum() * 0.0


def _ce(logits: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    if not bool(mask.any()):
        return logits.sum() * 0.0
    return F.cross_entropy(logits[mask], target[mask])


def axis_v6_causal_loss(
    model: AXISRigProgramModelV6,
    scene_memory: Tensor,
    batch: AXISV6TrainingBatch,
    *,
    weights: AXISV6LossWeights = AXISV6LossWeights(),
) -> dict[str, Tensor]:
    out = model(scene_memory, batch.inputs)
    target = batch.targets
    active = target.attention_mask.to(torch.bool)
    kinds = target.kind_ids

    losses: dict[str, Tensor] = {}
    losses["kind"] = _ce(out["kind_logits"], kinds, active)

    coord_mask = active & (
        (kinds == int(RigTokenKindV6.JOINT)) |
        (kinds == int(RigTokenKindV6.CONTROL)) |
        (kinds == int(RigTokenKindV6.ANCHOR))
    )
    coord_loss = out["coord_logits"].sum() * 0.0
    for axis in range(3):
        coord_loss = coord_loss + _ce(out["coord_logits"][:, :, axis], target.coord_indices[:, :, axis], coord_mask)
    losses["coordinate"] = coord_loss / 3.0

    ptr_a_kinds = {
        RigTokenKindV6.JOINT, RigTokenKindV6.FRAME, RigTokenKindV6.CONTROL,
        RigTokenKindV6.CONTROL_TARGET, RigTokenKindV6.CHAIN, RigTokenKindV6.CONSTRAINT,
        RigTokenKindV6.CONSTRAINT_INPUT, RigTokenKindV6.CONSTRAINT_OUTPUT, RigTokenKindV6.ANCHOR,
    }
    pointer_a_mask = active & torch.zeros_like(active)
    for kind in ptr_a_kinds:
        pointer_a_mask |= active & (kinds == int(kind))
    pointer_a_target = target.pointer_a + 1
    pointer_a_valid = pointer_a_mask & (pointer_a_target >= 0)
    losses["pointer_a"] = _ce(out["pointer_a_logits"], pointer_a_target.clamp_min(0), pointer_a_valid)

    pointer_b_mask = active & (
        (kinds == int(RigTokenKindV6.CHAIN)) |
        ((kinds == int(RigTokenKindV6.ANCHOR)) & (target.anchor_parent_kind == 1))
    )
    pointer_b_target = target.pointer_b + 1
    losses["pointer_b"] = _ce(out["pointer_b_logits"], pointer_b_target.clamp_min(0), pointer_b_mask)
    losses["pointer"] = losses["pointer_a"] + losses["pointer_b"]

    control_mask = active & (kinds == int(RigTokenKindV6.CONTROL))
    losses["control_kind"] = _ce(out["control_kind_logits"], target.aux_type_ids.clamp_min(0), control_mask)
    constraint_mask = active & (kinds == int(RigTokenKindV6.CONSTRAINT))
    losses["constraint_operator"] = _ce(out["constraint_operator_logits"], target.aux_type_ids.clamp_min(0), constraint_mask)

    frame_mask = active & (kinds == int(RigTokenKindV6.FRAME))
    dof_raw = F.binary_cross_entropy_with_logits(out["dof_logits"], target.dof_masks, reduction="none").mean(dim=-1)
    losses["dof"] = _masked_mean(dof_raw, frame_mask)
    frame_raw = F.smooth_l1_loss(out["frame_6d"], target.frame_6d, reduction="none").mean(dim=-1)
    losses["frame"] = _masked_mean(frame_raw, frame_mask)

    scalar_raw = F.smooth_l1_loss(out["scalar_params"], target.scalar_params, reduction="none").mean(dim=-1)
    losses["constraint_scalar"] = _masked_mean(scalar_raw, constraint_mask)

    code_mask = active & (kinds == int(RigTokenKindV6.SKIN_CODE))
    losses["skin_code"] = _ce(out["skin_code_logits"], target.skin_code_ids.clamp_min(0), code_mask)

    anchor_mask = active & (kinds == int(RigTokenKindV6.ANCHOR))
    losses["anchor_parent"] = _ce(out["anchor_parent_logits"], target.anchor_parent_kind.clamp_min(0), anchor_mask)

    total = (
        weights.kind * losses["kind"] +
        weights.coordinate * losses["coordinate"] +
        weights.pointer * losses["pointer"] +
        weights.control_kind * losses["control_kind"] +
        weights.constraint_operator * losses["constraint_operator"] +
        weights.dof * losses["dof"] +
        weights.frame * losses["frame"] +
        weights.constraint_scalar * losses["constraint_scalar"] +
        weights.skin_code * losses["skin_code"] +
        weights.anchor_parent * losses["anchor_parent"]
    )
    losses["total"] = total
    return losses


__all__ = [
    "AXISV6LossWeights", "AXISV6TrainingBatch", "axis_v6_causal_loss",
    "collate_axis_v6_causal_batch", "serialize_axis_v6_teacher_events",
]
