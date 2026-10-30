from __future__ import annotations

import pytest
import torch

from models.axis.rig_program_causal_v6 import (
    AXISRigProgramCausalV6,
    AXISRigProgramGrammarV6,
    AXISV6Config,
    AXISV6TeacherSequence,
    AXIS_V6_ARCHITECTURE_ID,
    DOF_LABELS,
    RigEventKind,
    RigProgramGrammarState,
    RigProgramPhase,
)


def _teacher(cfg: AXISV6Config) -> AXISV6TeacherSequence:
    kinds = torch.tensor([[1, 2, 3, 4, 5, 6, 8, 9, 10]], dtype=torch.long)
    b, s = kinds.shape
    return AXISV6TeacherSequence(
        kind_ids=kinds,
        coord_indices=torch.full((b, s, 3), -1, dtype=torch.long),
        pointer_a=torch.full((b, s), -1, dtype=torch.long),
        pointer_b=torch.full((b, s), -1, dtype=torch.long),
        aux_type_ids=torch.full((b, s), -1, dtype=torch.long),
        dof_masks=torch.zeros((b, s, len(DOF_LABELS)), dtype=torch.float32),
        skin_token_ids=torch.full((b, s), -1, dtype=torch.long),
        attention_mask=torch.ones((b, s), dtype=torch.bool),
    )


def test_axis_v6_grammar_requires_typed_section_order():
    state = RigProgramGrammarState()
    for kind in (
        RigEventKind.BOS,
        RigEventKind.JOINT,
        RigEventKind.FRAME,
        RigEventKind.CONTROL,
        RigEventKind.CHAIN,
        RigEventKind.CONSTRAINT,
        RigEventKind.SKIN_SWITCH,
        RigEventKind.SKIN_TOKEN,
        RigEventKind.EOS,
    ):
        state = AXISRigProgramGrammarV6.advance(state, kind)
    assert state.phase == RigProgramPhase.DONE
    assert state.joint_count == 1
    assert state.frame_count == 1
    assert state.control_count == 1
    assert state.chain_count == 1
    assert state.constraint_count == 1
    assert state.skin_token_count == 1


def test_axis_v6_grammar_forbids_skipping_from_joint_directly_to_constraint():
    state = AXISRigProgramGrammarV6.advance(RigProgramGrammarState(), RigEventKind.BOS)
    state = AXISRigProgramGrammarV6.advance(state, RigEventKind.JOINT)
    with pytest.raises(ValueError, match="ILLEGAL_TRANSITION"):
        AXISRigProgramGrammarV6.advance(state, RigEventKind.CONSTRAINT)


def test_axis_v6_model_predicts_all_typed_heads_from_shared_causal_state():
    torch.manual_seed(11)
    cfg = AXISV6Config(
        scene_dim=20,
        hidden_dim=32,
        layers=2,
        heads=4,
        ff_mult=2,
        max_sequence_length=32,
        max_pointer_entities=16,
        skin_vocab_size=81,
    )
    model = AXISRigProgramCausalV6(cfg)
    assert model.architecture_id == AXIS_V6_ARCHITECTURE_ID
    teacher = _teacher(cfg)
    scene = torch.randn(1, 13, cfg.scene_dim)
    out = model(scene, teacher)
    s = teacher.kind_ids.shape[1]
    assert out["kind_logits"].shape == (1, s, len(RigEventKind))
    assert out["coord_logits"].shape == (1, s, 3, cfg.coord_bins)
    assert out["pointer_a_logits"].shape == (1, s, cfg.max_pointer_entities)
    assert out["pointer_b_logits"].shape == (1, s, cfg.max_pointer_entities)
    assert out["dof_logits"].shape == (1, s, len(DOF_LABELS))
    assert out["skin_token_logits"].shape == (1, s, cfg.skin_vocab_size)


def test_axis_v6_generation_mask_is_model_side_not_compiler_repair():
    state = AXISRigProgramGrammarV6.advance(RigProgramGrammarState(), RigEventKind.BOS)
    logits = torch.zeros(len(RigEventKind))
    masked = AXISRigProgramGrammarV6.mask_kind_logits(logits, state)
    finite = torch.where(torch.isfinite(masked))[0].tolist()
    assert finite == [int(RigEventKind.JOINT), int(RigEventKind.FRAME)]
