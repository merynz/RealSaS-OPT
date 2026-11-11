from __future__ import annotations

"""Lossless typed adapter from AXIS V6 generated events to Compiler proposals.

This adapter interprets the model language; it does not improve it. Invalid
pointers, missing frame coverage, non-contiguous chains, degenerate frames or
other malformed model output fail here or in the Compiler court. No learned
semantics are synthesized downstream.
"""

from dataclasses import dataclass
from typing import Sequence
import math

from compiler.realsas_compiler_core.rig_program_ir_v1 import (
    RigProgramAttachmentAnchorProposalIR,
    RigProgramChainProposalIR,
    RigProgramConstraintProposalIR,
    RigProgramControlProposalIR,
    RigProgramJointProposalIR,
    RigProgramLocalFrameProposalIR,
    RigProgramProposalIR,
    seal_rig_program_proposal_v1,
)
from compiler.realsas_compiler_core.skin_code_ir_v1 import (
    JointSkinCodeRowIR,
    SkinCodeProposalIR,
    seal_skin_code_proposal_v1,
)
from models.axis.rig_program_model_v6 import (
    AXISRigProgramConfigV6,
    CONSTRAINT_OPERATORS_V6,
    CONTROL_KINDS_V6,
    DOF_LABELS_V6,
    AXISRigProgramGrammarV6Product,
    RigGrammarStateV6,
    RigSectionV6,
    RigProgramEventV6,
    RigTokenKindV6,
    rotation_6d_to_basis_v6,
)


@dataclass(frozen=True)
class AXISV6CodecBinding:
    architecture_id: str
    checkpoint_hash: str
    codebook_size: int
    tokens_per_joint: int


def _center(config: AXISRigProgramConfigV6, index: int) -> float:
    if index < 0 or index >= config.coord_bins:
        raise ValueError("AXIS_V6_COORD_INDEX_OUT_OF_RANGE")
    if config.coord_bins < 2:
        raise ValueError("AXIS_V6_COORD_BIN_DOMAIN_INVALID")
    alpha = float(index) / float(config.coord_bins - 1)
    return float(config.coord_min + alpha * (config.coord_max - config.coord_min))


def _world_point(config, indices, *, center, scale):
    p = tuple(_center(config, int(v)) for v in indices)
    return tuple(float(center[i]) + float(scale) * p[i] for i in range(3))


def _local_vector(config, indices, *, scale):
    p = tuple(_center(config, int(v)) for v in indices)
    return tuple(float(scale) * p[i] for i in range(3))


def _at(values: Sequence[str], index: int, code: str) -> str:
    if index < 0 or index >= len(values):
        raise ValueError(code)
    return str(values[index])


def _chain_path(start_id: str, end_id: str, parent_by: dict[str, str | None]) -> tuple[str, ...]:
    if start_id == end_id:
        return (start_id,)
    path = [end_id]
    cur = end_id
    seen = {cur}
    while parent_by.get(cur) is not None:
        cur = str(parent_by[cur])
        if cur in seen:
            return (start_id, end_id)
        seen.add(cur)
        path.append(cur)
        if cur == start_id:
            return tuple(reversed(path))
    return (start_id, end_id)


def validate_axis_v6_product_event_stream(
    events: Sequence[RigProgramEventV6],
    *,
    config: AXISRigProgramConfigV6,
    codec_binding: AXISV6CodecBinding,
) -> None:
    """Independent consumer court for AXIS's causal rig+skin grammar.

    The autoregressive producer masks illegal kinds, but the product decoder
    also accepts persisted/model-provided sequences. BOS/EOS alone cannot
    establish section ordering or the exact joint-coupled skin-code cardinality.
    Do not repair or truncate the proposed program at this boundary.
    """
    if codec_binding.tokens_per_joint < 1 or codec_binding.codebook_size < 2:
        raise ValueError("AXIS_V6_CODEC_BINDING_INVALID")
    if (
        type(config.skin_vocab_size) is not int
        or config.skin_vocab_size != codec_binding.codebook_size
    ):
        raise ValueError("AXIS_V6_CODEC_VOCABULARY_DRIFT")
    if not events or len(events) > config.max_sequence_length:
        raise ValueError("AXIS_V6_EVENT_STREAM_RESOURCE_LIMIT")
    if events[0].kind != RigTokenKindV6.BOS or events[-1].kind != RigTokenKindV6.EOS:
        raise ValueError("AXIS_V6_EVENT_BOUNDARY_INVALID")

    state = RigGrammarStateV6()
    for index, event in enumerate(events):
        if type(event) is not RigProgramEventV6 or type(event.kind) is not RigTokenKindV6:
            raise ValueError("AXIS_V6_EVENT_TYPED_DOMAIN_INVALID")
        if event.kind not in AXISRigProgramGrammarV6Product.allowed_kinds(
            state, tokens_per_joint=codec_binding.tokens_per_joint,
        ):
            raise ValueError("AXIS_V6_EVENT_GRAMMAR_INVALID")

        for value in (event.pointer_a, event.pointer_b, event.pointer_c):
            if type(value) is not int or value < -1 or value >= config.max_pointer_entities:
                raise ValueError("AXIS_V6_EVENT_POINTER_DOMAIN_INVALID")
        # Aux is a vocabulary class index, not a pointer into causal
        # entities. Never bound it by max_pointer_entities.
        if type(event.aux_type_id) is not int:
            raise ValueError("AXIS_V6_EVENT_AUX_DOMAIN_INVALID")
        if event.kind == RigTokenKindV6.CONTROL:
            if not 0 <= event.aux_type_id < len(CONTROL_KINDS_V6):
                raise ValueError("AXIS_V6_EVENT_AUX_DOMAIN_INVALID")
        elif event.kind == RigTokenKindV6.CONSTRAINT:
            if not 0 <= event.aux_type_id < len(CONSTRAINT_OPERATORS_V6):
                raise ValueError("AXIS_V6_EVENT_AUX_DOMAIN_INVALID")
        elif event.aux_type_id != -1:
            raise ValueError("AXIS_V6_EVENT_UNUSED_AUX_INVALID")
        if type(event.skin_code_id) is not int or type(event.anchor_parent_kind) is not int:
            raise ValueError("AXIS_V6_EVENT_INTEGER_DOMAIN_INVALID")
        for values, width in (
            (event.coord_indices, 3), (event.dof_mask, len(DOF_LABELS_V6)),
            (event.frame_6d, 6), (event.scalar_params, 4),
        ):
            if not isinstance(values, (tuple, list)) or len(values) != width:
                raise ValueError("AXIS_V6_EVENT_VECTOR_DOMAIN_INVALID")
        if any(type(x) is not int or x < -1 for x in event.coord_indices):
            raise ValueError("AXIS_V6_EVENT_COORD_DOMAIN_INVALID")
        coordinate_kinds = (
            RigTokenKindV6.JOINT, RigTokenKindV6.CONTROL, RigTokenKindV6.ANCHOR,
        )
        if event.kind in coordinate_kinds:
            if any(x < 0 or x >= config.coord_bins for x in event.coord_indices):
                raise ValueError("AXIS_V6_EVENT_COORD_DOMAIN_INVALID")
        elif tuple(event.coord_indices) != (-1, -1, -1):
            raise ValueError("AXIS_V6_EVENT_UNUSED_COORD_INVALID")
        if any(
            isinstance(x, bool) or not isinstance(x, (float, int)) or not math.isfinite(x)
            for x in (*event.dof_mask, *event.frame_6d, *event.scalar_params)
        ):
            raise ValueError("AXIS_V6_EVENT_CONTINUOUS_DOMAIN_INVALID")
        if any(x not in (0.0, 1.0) for x in event.dof_mask):
            raise ValueError("AXIS_V6_EVENT_DOF_DOMAIN_INVALID")
        if event.kind == RigTokenKindV6.SKIN_CODE:
            if not 0 <= event.skin_code_id < codec_binding.codebook_size:
                raise ValueError("AXIS_V6_SKIN_CODE_OUT_OF_CODEC_DOMAIN")
        elif event.skin_code_id != -1:
            raise ValueError("AXIS_V6_EVENT_UNUSED_SKIN_CODE_INVALID")
        if event.kind == RigTokenKindV6.ANCHOR:
            if event.anchor_parent_kind not in (0, 1):
                raise ValueError("AXIS_V6_ANCHOR_PARENT_KIND_INVALID")
        elif event.anchor_parent_kind != -1:
            raise ValueError("AXIS_V6_EVENT_UNUSED_ANCHOR_KIND_INVALID")

        state = AXISRigProgramGrammarV6Product.advance(
            state, event.kind, tokens_per_joint=codec_binding.tokens_per_joint,
        )
        if max(state.joint_count, state.control_count, state.chain_count) > config.max_pointer_entities:
            raise ValueError("AXIS_V6_EVENT_ENTITY_RESOURCE_LIMIT")
        if state.section == RigSectionV6.DONE and index != len(events) - 1:
            raise ValueError("AXIS_V6_EVENT_TRAILING_AFTER_EOS")

    if state.section != RigSectionV6.DONE:
        raise ValueError("AXIS_V6_EVENT_UNFINISHED_GRAMMAR")


def decode_axis_v6_events(
    events: Sequence[RigProgramEventV6],
    *,
    config: AXISRigProgramConfigV6,
    normalization_center: Sequence[float],
    normalization_scale: float,
    mesh_lineage_hash: str,
    feature_evidence_binding_hash: str,
    carrier_evidence_hash: str,
    carrier_topology_hash: str,
    carrier_geometry_hash: str,
    model_provenance: str,
    codec_binding: AXISV6CodecBinding,
) -> tuple[RigProgramProposalIR, SkinCodeProposalIR]:
    validate_axis_v6_product_event_stream(
        events, config=config, codec_binding=codec_binding,
    )
    center = tuple(float(v) for v in normalization_center)
    if len(center) != 3 or not all(math.isfinite(v) for v in center):
        raise ValueError("AXIS_V6_NORMALIZATION_CENTER_INVALID")
    scale = float(normalization_scale)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("AXIS_V6_NORMALIZATION_SCALE_INVALID")
    if codec_binding.tokens_per_joint < 1 or codec_binding.codebook_size < 2:
        raise ValueError("AXIS_V6_CODEC_BINDING_INVALID")

    joints: list[RigProgramJointProposalIR] = []
    frames: list[RigProgramLocalFrameProposalIR] = []
    chains: list[RigProgramChainProposalIR] = []
    anchors: list[RigProgramAttachmentAnchorProposalIR] = []
    control_rows: list[dict] = []
    constraint_rows: list[dict] = []
    skin_codes: list[int] = []

    joint_ids: list[str] = []
    control_ids: list[str] = []
    chain_ids: list[str] = []
    parent_by: dict[str, str | None] = {}

    for event in events[1:-1]:
        kind = event.kind
        if kind == RigTokenKindV6.JOINT:
            jid = f"P:J:{len(joint_ids):04d}"
            parent = None if event.pointer_a < 0 else _at(joint_ids, event.pointer_a, "AXIS_V6_JOINT_PARENT_POINTER_INVALID")
            joints.append(RigProgramJointProposalIR(
                proposal_joint_id=jid,
                P=_world_point(config, event.coord_indices, center=center, scale=scale),
                parent_proposal_joint_id=parent,
                support_evidence_ids=(),
                sequence_index=len(joint_ids),
                metadata={
                    "generation_index_internal_only": len(joint_ids),
                    "tree_serialization": "CAUSAL_PARENT_BEFORE_CHILD",
                    "discrete_xyz_bins": config.coord_bins,
                },
            ))
            joint_ids.append(jid)
            parent_by[jid] = parent
        elif kind == RigTokenKindV6.FRAME:
            jid = _at(joint_ids, event.pointer_a, "AXIS_V6_FRAME_JOINT_POINTER_INVALID")
            labels = tuple(label for label, enabled in zip(DOF_LABELS_V6, event.dof_mask) if float(enabled) >= 0.5)
            frames.append(RigProgramLocalFrameProposalIR(
                proposal_joint_id=jid,
                basis=rotation_6d_to_basis_v6(event.frame_6d),
                dof_labels=labels,
                metadata={"orientation_representation": "CONTINUOUS_6D_TO_ORTHONORMAL_BASIS"},
            ))
        elif kind == RigTokenKindV6.CONTROL:
            cid = f"P:C:{len(control_ids):04d}"
            if event.aux_type_id < 0 or event.aux_type_id >= len(CONTROL_KINDS_V6):
                raise ValueError("AXIS_V6_CONTROL_KIND_INVALID")
            parent_control = None if event.pointer_a < 0 else _at(control_ids, event.pointer_a, "AXIS_V6_CONTROL_PARENT_POINTER_INVALID")
            control_rows.append({
                "id": cid,
                "kind": CONTROL_KINDS_V6[event.aux_type_id],
                "parent": parent_control,
                "rest_P": _world_point(config, event.coord_indices, center=center, scale=scale),
                "targets": [],
            })
            control_ids.append(cid)
        elif kind == RigTokenKindV6.CONTROL_TARGET:
            if not control_rows:
                raise ValueError("AXIS_V6_CONTROL_TARGET_WITHOUT_CONTROL")
            target = _at(joint_ids, event.pointer_a, "AXIS_V6_CONTROL_TARGET_POINTER_INVALID")
            control_rows[-1]["targets"].append(target)
        elif kind == RigTokenKindV6.CHAIN:
            kid = f"P:K:{len(chain_ids):04d}"
            start = _at(joint_ids, event.pointer_a, "AXIS_V6_CHAIN_START_POINTER_INVALID")
            end = _at(joint_ids, event.pointer_b, "AXIS_V6_CHAIN_END_POINTER_INVALID")
            chains.append(RigProgramChainProposalIR(
                proposal_chain_id=kid,
                proposal_joint_ids=_chain_path(start, end, parent_by),
                chain_kind="GENERIC_KINEMATIC_CHAIN",
                metadata={"path_interpretation": "DECLARED_START_END_ON_DEFORM_PARENT_GRAPH"},
            ))
            chain_ids.append(kid)
        elif kind == RigTokenKindV6.CONSTRAINT:
            qid = f"P:Q:{len(constraint_rows):04d}"
            if event.aux_type_id < 0 or event.aux_type_id >= len(CONSTRAINT_OPERATORS_V6):
                raise ValueError("AXIS_V6_CONSTRAINT_OPERATOR_INVALID")
            chain_id = None if event.pointer_a < 0 else _at(chain_ids, event.pointer_a, "AXIS_V6_CONSTRAINT_CHAIN_POINTER_INVALID")
            constraint_rows.append({
                "id": qid,
                "operator": CONSTRAINT_OPERATORS_V6[event.aux_type_id],
                "chain": chain_id,
                "inputs": [],
                "outputs": [],
                "parameters": {f"p{i}": float(v) for i, v in enumerate(event.scalar_params)},
            })
        elif kind == RigTokenKindV6.CONSTRAINT_INPUT:
            if not constraint_rows:
                raise ValueError("AXIS_V6_CONSTRAINT_INPUT_WITHOUT_CONSTRAINT")
            cid = _at(control_ids, event.pointer_a, "AXIS_V6_CONSTRAINT_CONTROL_POINTER_INVALID")
            constraint_rows[-1]["inputs"].append(cid)
        elif kind == RigTokenKindV6.CONSTRAINT_OUTPUT:
            if not constraint_rows:
                raise ValueError("AXIS_V6_CONSTRAINT_OUTPUT_WITHOUT_CONSTRAINT")
            jid = _at(joint_ids, event.pointer_a, "AXIS_V6_CONSTRAINT_OUTPUT_POINTER_INVALID")
            constraint_rows[-1]["outputs"].append(jid)
        elif kind == RigTokenKindV6.ANCHOR:
            aid = f"P:A:{len(anchors):04d}"
            if event.anchor_parent_kind == 0:
                pj = _at(joint_ids, event.pointer_a, "AXIS_V6_ANCHOR_JOINT_POINTER_INVALID")
                pc = None
            elif event.anchor_parent_kind == 1:
                pj = None
                pc = _at(control_ids, event.pointer_b, "AXIS_V6_ANCHOR_CONTROL_POINTER_INVALID")
            else:
                raise ValueError("AXIS_V6_ANCHOR_PARENT_KIND_INVALID")
            anchors.append(RigProgramAttachmentAnchorProposalIR(
                proposal_anchor_id=aid,
                parent_proposal_joint_id=pj,
                parent_proposal_control_id=pc,
                local_P=_local_vector(config, event.coord_indices, scale=scale),
                metadata={"mechanical_attachment_anchor": True},
            ))
        elif kind == RigTokenKindV6.SKIN_CODE:
            if event.skin_code_id < 0 or event.skin_code_id >= codec_binding.codebook_size:
                raise ValueError("AXIS_V6_SKIN_CODE_OUT_OF_CODEC_DOMAIN")
            skin_codes.append(int(event.skin_code_id))
        elif kind in {
            RigTokenKindV6.FRAMES_START, RigTokenKindV6.CONTROLS_START,
            RigTokenKindV6.CHAINS_START, RigTokenKindV6.CONSTRAINTS_START,
            RigTokenKindV6.ANCHORS_START, RigTokenKindV6.SKIN_START,
        }:
            continue
        else:
            raise ValueError(f"AXIS_V6_UNEXPECTED_EVENT:{kind.name}")

    controls = tuple(
        RigProgramControlProposalIR(
            proposal_control_id=row["id"],
            control_kind=row["kind"],
            target_proposal_joint_ids=tuple(row["targets"]),
            parent_proposal_control_id=row["parent"],
            rest_P=row["rest_P"],
            metadata={"animator_facing": True, "target_arity": len(row["targets"])},
        )
        for row in control_rows
    )
    constraints = tuple(
        RigProgramConstraintProposalIR(
            proposal_constraint_id=row["id"],
            operator_id=row["operator"],
            input_proposal_control_ids=tuple(row["inputs"]),
            output_proposal_joint_ids=tuple(row["outputs"]),
            proposal_chain_id=row["chain"],
            parameters=dict(row["parameters"]),
            metadata={
                "solver_execution_performed": False,
                "input_arity": len(row["inputs"]),
                "output_arity": len(row["outputs"]),
            },
        )
        for row in constraint_rows
    )

    proposal = seal_rig_program_proposal_v1(RigProgramProposalIR(
        joints=tuple(joints),
        local_frames=tuple(frames),
        controls=controls,
        chains=tuple(chains),
        constraints=constraints,
        attachment_anchors=tuple(anchors),
        mesh_lineage_hash=str(mesh_lineage_hash),
        feature_evidence_binding_hash=str(feature_evidence_binding_hash),
        model_provenance=str(model_provenance),
        metadata={
            "compiler_role": "VALIDATE_AND_MATERIALIZE_ONLY",
            "compiler_parent_reselection_allowed": False,
            "compiler_control_synthesis_allowed": False,
            "compiler_constraint_synthesis_allowed": False,
            "variable_arity_control_targets": True,
            "variable_arity_constraint_relations": True,
            "skin_codes_generated_in_same_causal_run": True,
        },
    ))

    required = len(joint_ids) * codec_binding.tokens_per_joint
    if len(skin_codes) != required:
        raise ValueError(f"AXIS_V6_SKIN_CODE_COUNT_DRIFT:{len(skin_codes)}:{required}")
    rows = tuple(
        JointSkinCodeRowIR(
            proposal_joint_id=jid,
            code_ids=tuple(skin_codes[i * codec_binding.tokens_per_joint:(i + 1) * codec_binding.tokens_per_joint]),
            metadata={"joint_sequence_index": i},
        )
        for i, jid in enumerate(joint_ids)
    )
    code_proposal = seal_skin_code_proposal_v1(SkinCodeProposalIR(
        rows=rows,
        rig_program_proposal_hash=proposal.proposal_lineage_hash,
        carrier_evidence_hash=str(carrier_evidence_hash),
        carrier_topology_hash=str(carrier_topology_hash),
        carrier_geometry_hash=str(carrier_geometry_hash),
        codec_architecture_id=str(codec_binding.architecture_id),
        codec_checkpoint_hash=str(codec_binding.checkpoint_hash),
        codebook_size=int(codec_binding.codebook_size),
        codes_per_joint=int(codec_binding.tokens_per_joint),
        model_provenance=str(model_provenance),
        metadata={
            "generated_joint_conditionally_in_same_axis_sequence": True,
            "compiler_code_search_allowed": False,
            "compiler_code_repair_allowed": False,
        },
    ))
    return proposal, code_proposal


__all__ = ["AXISV6CodecBinding", "decode_axis_v6_events"]
