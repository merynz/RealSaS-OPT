from __future__ import annotations

"""Exact executable Mechanics operator grammar for AXIS V6 event programs.

No missing controls, outputs or kinematic chains are synthesized downstream.
The only arity authority is the same RigConstraintOperatorSchemaV1 consumed
by Compiler/Mechanics. AXIS masks impossible next actions and constraint
operator selections; source teachers and serialized event consumers must
satisfy the identical contract. This does not infer semantic wing/cloth truth.
"""
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor

from compiler.realsas_compiler_core.rig_constraint_schema_v1 import (
    RIG_CONSTRAINT_OPERATOR_IDS_V1, rig_constraint_operator_schema_v1,
)
from models.axis.rig_program_model_v6 import (
    RigProgramEventV6, RigGrammarStateV6, RigSectionV6, RigTokenKindV6,
    CONSTRAINT_OPERATORS_V6,
)


@dataclass(frozen=True)
class AXISExecutableConstraintStateV1:
    active_operator: str | None
    active_control_inputs: int
    active_joint_outputs: int
    viable_new_operators: tuple[str, ...]
    source_truth_claimed: bool = False


def _viable(
    *,
    control_count: int,
    chain_count: int,
    executable_operators: Sequence[str],
) -> tuple[str,...]:
    if control_count<0 or chain_count<0:
        raise ValueError("AXIS_V6_EXECUTABLE_CONSTRAINT_STATE_INVALID")
    names=tuple(str(x) for x in executable_operators)
    if len(set(names))!=len(names) or set(names)-set(RIG_CONSTRAINT_OPERATOR_IDS_V1):
        raise ValueError("AXIS_V6_EXECUTABLE_CONSTRAINT_OPERATOR_UNSUPPORTED")
    viable=[]
    for name in names:
        schema=rig_constraint_operator_schema_v1(name)
        if schema.min_control_inputs>control_count:
            continue
        if schema.chain_requirement=="REQUIRED" and chain_count==0:
            continue
        viable.append(name)
    return tuple(viable)


def inspect_axis_v6_constraint_events_v1(
    events: Sequence[RigProgramEventV6],
    *,
    executable_operators: Sequence[str]=RIG_CONSTRAINT_OPERATOR_IDS_V1,
) -> AXISExecutableConstraintStateV1:
    """Validate every completed event and any active trailing constraint."""
    controls=chains=joints=0
    active_name=None
    n_input=n_output=0
    def finish() -> None:
        if active_name is None:
            return
        schema=rig_constraint_operator_schema_v1(active_name)
        if not schema.min_control_inputs <= n_input <= schema.max_control_inputs:
            raise ValueError("AXIS_V6_CONSTRAINT_CONTROL_ARITY_INVALID")
        if not schema.min_joint_outputs <= n_output <= schema.max_joint_outputs:
            raise ValueError("AXIS_V6_CONSTRAINT_JOINT_OUTPUT_ARITY_INVALID")

    allowed=tuple(str(x) for x in executable_operators)
    _viable(control_count=0,chain_count=0,executable_operators=allowed)
    for event in events:
        if not isinstance(event,RigProgramEventV6):
            raise ValueError("AXIS_V6_CONSTRAINT_TYPED_EVENT_REQUIRED")
        k=event.kind
        if k==RigTokenKindV6.JOINT:
            joints+=1
        elif k==RigTokenKindV6.CONTROL:
            controls+=1
        elif k==RigTokenKindV6.CHAIN:
            chains+=1
        elif k==RigTokenKindV6.CONSTRAINT:
            finish()
            op_index=event.aux_type_id
            if not 0<=op_index<len(CONSTRAINT_OPERATORS_V6):
                raise ValueError("AXIS_V6_CONSTRAINT_OPERATOR_INDEX_INVALID")
            name=CONSTRAINT_OPERATORS_V6[op_index]
            if name not in allowed or name not in _viable(
                control_count=controls,chain_count=chains,
                executable_operators=allowed,
            ):
                raise ValueError("AXIS_V6_CONSTRAINT_UNEXECUTABLE_OPERATOR")
            schema=rig_constraint_operator_schema_v1(name)
            if (schema.chain_requirement=="REQUIRED" and
                not 0<=event.pointer_a<chains):
                raise ValueError("AXIS_V6_CONSTRAINT_REQUIRED_CHAIN_MISSING")
            if (schema.chain_requirement=="FORBIDDEN" and event.pointer_a!=-1):
                raise ValueError("AXIS_V6_CONSTRAINT_FORBIDDEN_CHAIN_DECLARED")
            if event.pointer_a < -1 or event.pointer_a >= chains:
                raise ValueError("AXIS_V6_CONSTRAINT_CHAIN_POINTER_INVALID")
            active_name=name
            n_input=n_output=0
        elif k==RigTokenKindV6.CONSTRAINT_INPUT:
            if active_name is None or event.pointer_a<0 or event.pointer_a>=controls:
                raise ValueError("AXIS_V6_CONSTRAINT_CONTROL_POINTER_INVALID")
            n_input+=1
            if n_input>rig_constraint_operator_schema_v1(active_name).max_control_inputs:
                raise ValueError("AXIS_V6_CONSTRAINT_CONTROL_ARITY_INVALID")
        elif k==RigTokenKindV6.CONSTRAINT_OUTPUT:
            if active_name is None or event.pointer_a<0 or event.pointer_a>=joints:
                raise ValueError("AXIS_V6_CONSTRAINT_JOINT_POINTER_INVALID")
            n_output+=1
            if n_output>rig_constraint_operator_schema_v1(active_name).max_joint_outputs:
                raise ValueError("AXIS_V6_CONSTRAINT_JOINT_OUTPUT_ARITY_INVALID")
        elif k==RigTokenKindV6.ANCHORS_START:
            finish()
            active_name=None
            n_input=n_output=0
    # In inference this may be an incomplete causal prefix. Only completed
    # constraints must meet minimum arity; the active one stays repair-free.
    return AXISExecutableConstraintStateV1(
        active_operator=active_name,
        active_control_inputs=n_input,
        active_joint_outputs=n_output,
        viable_new_operators=_viable(
            control_count=controls,chain_count=chains,
            executable_operators=allowed,
        ),
    )


def mask_axis_v6_executable_constraint_kinds_v1(
    kind_logits:Tensor,
    *,
    events:Sequence[RigProgramEventV6],
    state:RigGrammarStateV6,
    executable_operators:Sequence[str]=RIG_CONSTRAINT_OPERATOR_IDS_V1,
) -> Tensor:
    """Mask impossible next kinds after AXIS's basic causal grammar mask."""
    if state.section != RigSectionV6.CONSTRAINTS:
        return kind_logits
    if kind_logits.ndim!=1 or kind_logits.shape[0]!=len(RigTokenKindV6):
        raise ValueError("AXIS_V6_CONSTRAINT_KIND_LOGIT_DOMAIN_INVALID")
    court=inspect_axis_v6_constraint_events_v1(
        events,executable_operators=executable_operators,
    )
    forbidden=set()
    if not court.viable_new_operators:
        forbidden.add(RigTokenKindV6.CONSTRAINT)
    if court.active_operator is None:
        forbidden.update((
            RigTokenKindV6.CONSTRAINT_INPUT,
            RigTokenKindV6.CONSTRAINT_OUTPUT,
        ))
    else:
        schema=rig_constraint_operator_schema_v1(court.active_operator)
        if court.active_control_inputs < schema.min_control_inputs:
            forbidden.update((RigTokenKindV6.CONSTRAINT,RigTokenKindV6.ANCHORS_START,
                              RigTokenKindV6.CONSTRAINT_OUTPUT))
        elif court.active_joint_outputs < schema.min_joint_outputs:
            forbidden.update((RigTokenKindV6.CONSTRAINT,RigTokenKindV6.ANCHORS_START,
                              RigTokenKindV6.CONSTRAINT_INPUT))
        if court.active_control_inputs >= schema.max_control_inputs:
            forbidden.add(RigTokenKindV6.CONSTRAINT_INPUT)
        if court.active_joint_outputs >= schema.max_joint_outputs:
            forbidden.add(RigTokenKindV6.CONSTRAINT_OUTPUT)
    masked=kind_logits.clone()
    for kind in forbidden:
        masked[int(kind)]=float("-inf")
    if not bool(torch.isfinite(masked).any()):
        raise ValueError("AXIS_V6_CONSTRAINT_NO_EXECUTABLE_NEXT_KIND")
    return masked


def mask_axis_v6_executable_constraint_operators_v1(
    logits:Tensor, *,
    control_count:int, chain_count:int,
    executable_operators:Sequence[str]=RIG_CONSTRAINT_OPERATOR_IDS_V1,
) -> Tensor:
    if logits.ndim!=1 or logits.shape[0]!=len(CONSTRAINT_OPERATORS_V6):
        raise ValueError("AXIS_V6_CONSTRAINT_OPERATOR_LOGIT_DOMAIN_INVALID")
    viable=_viable(
        control_count=control_count,chain_count=chain_count,
        executable_operators=executable_operators,
    )
    masked=torch.full_like(logits,float("-inf"))
    for name in viable:
        masked[CONSTRAINT_OPERATORS_V6.index(name)]=logits[
            CONSTRAINT_OPERATORS_V6.index(name)
        ]
    if not bool(torch.isfinite(masked).any()):
        raise ValueError("AXIS_V6_CONSTRAINT_NO_EXECUTABLE_OPERATOR")
    return masked


__all__=[
    "AXISExecutableConstraintStateV1",
    "inspect_axis_v6_constraint_events_v1",
    "mask_axis_v6_executable_constraint_kinds_v1",
    "mask_axis_v6_executable_constraint_operators_v1",
]
