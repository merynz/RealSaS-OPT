from __future__ import annotations

"""Exact deform-joint frame ownership in emitted and source-trained AXIS V6."""
from dataclasses import replace
import pytest
import torch

from models.axis.rig_program_model_v6 import (
    AXISRigProgramGrammarV6Product as Grammar,
    RigGrammarStateV6, RigTokenKindV6 as K,
    RigProgramEventV6 as E,
)
from tests.models.test_axis_mira_v6_product_runtime import (
    _minimal_product_events, _decode_product_events,
)
from tests.models.test_rss_vnext_axis_structural_chain_training_v1 import (
    _setup, _loss,
)


def test_product_grammar_cannot_leave_frames_before_full_joint_coverage():
    state=RigGrammarStateV6()
    for token in (K.BOS,K.JOINT,K.JOINT,K.FRAMES_START):
        state=Grammar.advance(state,token,tokens_per_joint=2)
    assert Grammar.allowed_kinds(state,tokens_per_joint=2)==(K.FRAME,)
    state=Grammar.advance(state,K.FRAME,tokens_per_joint=2)
    assert Grammar.allowed_kinds(state,tokens_per_joint=2)==(K.FRAME,)
    with pytest.raises(ValueError,match="PRODUCT_GRAMMAR_ILLEGAL"):
        Grammar.advance(state,K.CONTROLS_START,tokens_per_joint=2)
    state=Grammar.advance(state,K.FRAME,tokens_per_joint=2)
    assert Grammar.allowed_kinds(state,tokens_per_joint=2)==(K.CONTROLS_START,)
    with pytest.raises(ValueError,match="PRODUCT_GRAMMAR_ILLEGAL"):
        Grammar.advance(state,K.FRAME,tokens_per_joint=2)


def test_product_decoder_rejects_missing_frames_and_accepts_two_unique_frames():
    single=_minimal_product_events()
    with pytest.raises(ValueError,match="EVENT_GRAMMAR_INVALID"):
        _decode_product_events(single[:3]+single[4:])
    seq=list(single)
    seq.insert(2,E(K.JOINT,coord_indices=(130,128,128),pointer_a=0))
    seq.insert(5,E(K.FRAME,pointer_a=1,frame_6d=(1.,0.,0.,0.,1.,0.)))
    seq[-1:-1]=[E(K.SKIN_CODE,skin_code_id=4),E(K.SKIN_CODE,skin_code_id=5)]
    rig,skin=_decode_product_events(tuple(seq))
    assert len(rig.joints)==len(rig.local_frames)==2
    assert len(skin.rows)==2
    seq[5]=replace(seq[5],pointer_a=0)
    with pytest.raises(ValueError,match="JOINT_FRAME_DUPLICATE_OR_UNKNOWN"):
        _decode_product_events(tuple(seq))


def test_source_teacher_cannot_repeat_a_frame_pointer_with_qualified_chains():
    net,seq=_setup()
    positions=torch.where(seq.kind_ids[0]==int(K.FRAME))[0]
    assert len(positions)==2
    seq.pointer_a[0,int(positions[1])]=0
    with pytest.raises(ValueError,match="TRAIN_JOINT_FRAME_DUPLICATE_OR_UNKNOWN"):
        _loss(net,seq,structural_chains_qualified=True)
