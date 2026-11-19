from __future__ import annotations

"""AXIS/Compiler share one executable rig-constraint grammar, no synthesis."""
from dataclasses import replace
import pytest
import torch

from models.axis.rig_program_model_v6 import (
    RigGrammarStateV6, RigSectionV6, RigTokenKindV6 as K,
    RigProgramEventV6 as E, CONSTRAINT_OPERATORS_V6,
)
from models.rss_vnext.axis_executable_constraint_grammar_v1 import (
    inspect_axis_v6_constraint_events_v1 as inspect,
    mask_axis_v6_executable_constraint_kinds_v1 as mask_kinds,
    mask_axis_v6_executable_constraint_operators_v1 as mask_ops,
)
from tests.models.test_axis_mira_v6_product_runtime import (
    _minimal_product_events, _decode_product_events,
)


def _prefix(operator="IK",with_chain=True):
    return (
        E(K.BOS), E(K.JOINT,pointer_a=-1),
        E(K.CONTROL,aux_type_id=0),
        *( (E(K.CHAIN,pointer_a=0,pointer_b=0),) if with_chain else () ),
        E(K.CONSTRAINT,
          aux_type_id=CONSTRAINT_OPERATORS_V6.index(operator),
          pointer_a=(0 if with_chain and operator in ("IK","STRETCH") else -1)),
    )


def _state(*,chain_count=1):
    return RigGrammarStateV6(
        section=RigSectionV6.CONSTRAINTS,
        joint_count=1,control_count=1,chain_count=chain_count,
        constraint_count=1,
    )


def test_compiler_schema_forces_control_input_and_aim_output_before_exit():
    logits=torch.zeros(len(K))
    ik=_prefix()
    out=mask_kinds(logits,events=ik,state=_state())
    assert torch.isfinite(out[int(K.CONSTRAINT_INPUT)])
    assert torch.isneginf(out[int(K.CONSTRAINT)])
    assert torch.isneginf(out[int(K.ANCHORS_START)])
    assert torch.isneginf(out[int(K.CONSTRAINT_OUTPUT)])
    after=mask_kinds(logits,events=ik+(E(K.CONSTRAINT_INPUT,pointer_a=0),),state=_state())
    assert torch.isfinite(after[int(K.ANCHORS_START)])

    aim=_prefix("AIM",with_chain=False)+(E(K.CONSTRAINT_INPUT,pointer_a=0),)
    out=mask_kinds(logits,events=aim,state=_state(chain_count=0))
    assert torch.isfinite(out[int(K.CONSTRAINT_OUTPUT)])
    assert torch.isneginf(out[int(K.ANCHORS_START)])
    assert torch.isneginf(out[int(K.CONSTRAINT)])
    completed=mask_kinds(
        logits,events=aim+(E(K.CONSTRAINT_OUTPUT,pointer_a=0),),
        state=_state(chain_count=0),
    )
    assert torch.isfinite(completed[int(K.ANCHORS_START)])


def test_unrealizable_operator_or_absent_controller_cannot_be_generated():
    logits=torch.zeros(len(CONSTRAINT_OPERATORS_V6))
    out=mask_ops(logits,control_count=1,chain_count=0)
    for name in ("IK","STRETCH","LIMIT_ROTATION","PATH"):
        assert torch.isneginf(out[CONSTRAINT_OPERATORS_V6.index(name)])
    for name in ("AIM","FOLLOW","COPY_TRANSFORM"):
        assert torch.isfinite(out[CONSTRAINT_OPERATORS_V6.index(name)])
    kinds=mask_kinds(
        torch.zeros(len(K)),
        events=(E(K.BOS),E(K.JOINT)),
        state=RigGrammarStateV6(
            section=RigSectionV6.CONSTRAINTS,joint_count=1,
        ),
    )
    assert torch.isneginf(kinds[int(K.CONSTRAINT)])
    assert torch.isfinite(kinds[int(K.ANCHORS_START)])


def test_incomplete_or_forged_constraint_rejected_without_fixing():
    with pytest.raises(ValueError,match="CONTROL_ARITY_INVALID"):
        inspect(_prefix()+(E(K.ANCHORS_START),))
    with pytest.raises(ValueError,match="REQUIRED_CHAIN_MISSING"):
        inspect((
            E(K.BOS),E(K.JOINT),E(K.CONTROL),
            E(K.CONSTRAINT,aux_type_id=CONSTRAINT_OPERATORS_V6.index("IK")),
        ))
    with pytest.raises(ValueError,match="UNEXECUTABLE_OPERATOR"):
        inspect(_prefix("LIMIT_ROTATION"))
    with pytest.raises(ValueError,match="JOINT_OUTPUT_ARITY_INVALID"):
        inspect(_prefix("AIM",False)+(
            E(K.CONSTRAINT_INPUT,pointer_a=0),E(K.ANCHORS_START),
        ))
    with pytest.raises(ValueError,match="CONTROL_ARITY_INVALID"):
        inspect(_prefix("AIM",False)+(
            E(K.CONSTRAINT_INPUT,pointer_a=0),
            E(K.CONSTRAINT_INPUT,pointer_a=0),
        ))


def test_product_decoder_exactly_replays_ik_and_refuses_unfinished_ik():
    base=list(_minimal_product_events())
    control=E(K.CONTROL,aux_type_id=0,coord_indices=(128,128,128))
    controls_index=next(i for i,x in enumerate(base) if x.kind==K.CHAINS_START)
    base[controls_index:controls_index]=[
        control,E(K.CONTROL_TARGET,pointer_a=0),
    ]
    start=next(i for i,x in enumerate(base) if x.kind==K.CONSTRAINTS_START)
    base[start:start]=[E(K.CHAIN,pointer_a=0,pointer_b=0)]
    anchor=next(i for i,x in enumerate(base) if x.kind==K.ANCHORS_START)
    base[anchor:anchor]=[
        E(K.CONSTRAINT,aux_type_id=CONSTRAINT_OPERATORS_V6.index("IK"),pointer_a=0),
        E(K.CONSTRAINT_INPUT,pointer_a=0),
    ]
    rig, _=_decode_product_events(tuple(base))
    assert len(rig.constraints)==1
    assert len(rig.constraints[0].input_proposal_control_ids)==1
    missing=tuple(x for x in base if x.kind!=K.CONSTRAINT_INPUT)
    with pytest.raises(ValueError,match="CONTROL_ARITY_INVALID"):
        _decode_product_events(missing)


def test_source_teacher_trains_declared_executable_ik_and_rejects_unfinished_ik():
    from models.axis.rig_program_model_v6 import (
        AXISSequenceV6, qualified_axis_rig_skin_sequence_loss_v6,
    )
    from tests.models.test_rss_vnext_axis_structural_chain_training_v1 import _setup
    model,_=_setup(with_control=True)
    events=(
        E(K.BOS),
        E(K.JOINT,coord_indices=(1,2,3),pointer_a=-1),
        E(K.FRAMES_START),
        E(K.FRAME,pointer_a=0,frame_6d=(1.,0.,0.,0.,1.,0.)),
        E(K.CONTROLS_START),
        E(K.CONTROL,coord_indices=(1,2,3),pointer_a=-1,aux_type_id=0),
        E(K.CONTROL_TARGET,pointer_a=0),
        E(K.CHAINS_START),
        E(K.CHAIN,pointer_a=0,pointer_b=0),
        E(K.CONSTRAINTS_START),
        E(K.CONSTRAINT,pointer_a=0,aux_type_id=CONSTRAINT_OPERATORS_V6.index("IK")),
        E(K.CONSTRAINT_INPUT,pointer_a=0),
        E(K.ANCHORS_START),
        E(K.SKIN_START),
        *(E(K.SKIN_CODE,skin_code_id=i) for i in range(4)),
        E(K.EOS),
    )
    def loss(xs):
        seq=AXISSequenceV6.from_events(xs,device=torch.device("cpu"))
        return qualified_axis_rig_skin_sequence_loss_v6(
            model,torch.zeros(1,4,32),seq,
            tokens_per_joint=4,
            executable_constraint_operators=("IK",),
            control_intervention_qualified=True,
            structural_chains_qualified=True,
        )[0]
    value=loss(events)
    assert torch.isfinite(value)
    value.backward()
    assert model.kind_head.weight.grad is not None
    bad=tuple(x for x in events if x.kind!=K.CONSTRAINT_INPUT)
    with pytest.raises(ValueError,match="CONSTRAINT_CONTROL_ARITY_INVALID"):
        loss(bad)
