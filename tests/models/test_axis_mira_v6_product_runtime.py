from __future__ import annotations

import hashlib

import torch

from compiler.realsas_compiler_core.rig_program_ir_v1 import qualify_rig_program_v1
from compiler.realsas_compiler_core.skin_token_ir_v1 import validate_skin_token_proposal_v1
from models.axis.rig_program_decode_v6 import AXISV6CodecBinding, decode_axis_v6_events
from models.axis.rig_program_model_v6 import (
    AXISRigProgramConfigV6,
    AXISRigProgramGrammarV6Product,
    AXISRigProgramModelV6,
    DOF_LABELS_V6,
    RigGrammarStateV6,
    RigProgramEventV6,
    RigSectionV6,
    RigTokenKindV6,
)
from models.axis.runtime_checkpoint_v2 import (
    build_axis_runtime_checkpoint_v2,
    load_axis_runtime_model_v2,
)
from models.axis.scene_conditioning_v6 import AXIS_V6_SURFACE_SCENE_DIM
from models.mira.conditioning_v6 import MIRA_V6_JOINT_FEATURE_DIM, MIRA_V6_VERTEX_FEATURE_DIM
from models.mira.runtime_checkpoint_v2 import build_mira_runtime_checkpoint_v2, load_mira_runtime_model_v2
from models.mira.skin_fsq_codec_v6 import MIRASkinFSQCodecConfig, MIRASkinFSQCodecV6, MIRA_V6_CODEC_ID


def test_product_grammar_allows_empty_optional_sections_but_exact_skin_cardinality():
    state = AXISRigProgramGrammarV6Product.advance(RigGrammarStateV6(), RigTokenKindV6.BOS, tokens_per_joint=2)
    state = AXISRigProgramGrammarV6Product.advance(state, RigTokenKindV6.JOINT, tokens_per_joint=2)
    for marker in (
        RigTokenKindV6.FRAMES_START,
        RigTokenKindV6.CONTROLS_START,
        RigTokenKindV6.CHAINS_START,
        RigTokenKindV6.CONSTRAINTS_START,
        RigTokenKindV6.ANCHORS_START,
        RigTokenKindV6.SKIN_START,
    ):
        state = AXISRigProgramGrammarV6Product.advance(state, marker, tokens_per_joint=2)
    assert AXISRigProgramGrammarV6Product.allowed_kinds(state, tokens_per_joint=2) == (RigTokenKindV6.SKIN_CODE,)
    state = AXISRigProgramGrammarV6Product.advance(state, RigTokenKindV6.SKIN_CODE, tokens_per_joint=2)
    state = AXISRigProgramGrammarV6Product.advance(state, RigTokenKindV6.SKIN_CODE, tokens_per_joint=2)
    assert AXISRigProgramGrammarV6Product.allowed_kinds(state, tokens_per_joint=2) == (RigTokenKindV6.EOS,)
    state = AXISRigProgramGrammarV6Product.advance(state, RigTokenKindV6.EOS, tokens_per_joint=2)
    assert state.section == RigSectionV6.DONE


def test_typed_event_decode_preserves_variable_arity_controls_constraints_and_skin_codes():
    events = (
        RigProgramEventV6(RigTokenKindV6.BOS),
        RigProgramEventV6(RigTokenKindV6.JOINT, coord_indices=(128, 128, 128), pointer_a=-1),
        RigProgramEventV6(RigTokenKindV6.FRAMES_START),
        RigProgramEventV6(
            RigTokenKindV6.FRAME,
            pointer_a=0,
            frame_6d=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            dof_mask=tuple(1.0 if name in {"RX", "RY", "RZ"} else 0.0 for name in DOF_LABELS_V6),
        ),
        RigProgramEventV6(RigTokenKindV6.CONTROLS_START),
        RigProgramEventV6(RigTokenKindV6.CONTROL, coord_indices=(128, 128, 128), pointer_a=-1, aux_type_id=1),
        RigProgramEventV6(RigTokenKindV6.CONTROL_TARGET, pointer_a=0),
        RigProgramEventV6(RigTokenKindV6.CONTROL, coord_indices=(128, 128, 128), pointer_a=0, aux_type_id=3),
        RigProgramEventV6(RigTokenKindV6.CONTROL_TARGET, pointer_a=0),
        RigProgramEventV6(RigTokenKindV6.CONTROL, coord_indices=(128, 140, 128), pointer_a=0, aux_type_id=4),
        RigProgramEventV6(RigTokenKindV6.CHAINS_START),
        RigProgramEventV6(RigTokenKindV6.CHAIN, pointer_a=0, pointer_b=0),
        RigProgramEventV6(RigTokenKindV6.CONSTRAINTS_START),
        RigProgramEventV6(RigTokenKindV6.CONSTRAINT, pointer_a=0, aux_type_id=0),
        RigProgramEventV6(RigTokenKindV6.CONSTRAINT_INPUT, pointer_a=1),
        RigProgramEventV6(RigTokenKindV6.CONSTRAINT_INPUT, pointer_a=2),
        RigProgramEventV6(RigTokenKindV6.CONSTRAINT_OUTPUT, pointer_a=0),
        RigProgramEventV6(RigTokenKindV6.ANCHORS_START),
        RigProgramEventV6(RigTokenKindV6.SKIN_START),
        RigProgramEventV6(RigTokenKindV6.SKIN_CODE, skin_code_id=7),
        RigProgramEventV6(RigTokenKindV6.SKIN_CODE, skin_code_id=8),
        RigProgramEventV6(RigTokenKindV6.EOS),
    )
    cfg = AXISRigProgramConfigV6(scene_dim=AXIS_V6_SURFACE_SCENE_DIM, hidden_dim=32, layers=1, heads=4, skin_vocab_size=81)
    codec = AXISV6CodecBinding(MIRA_V6_CODEC_ID, "a" * 64, 81, 2)
    rig, skin = decode_axis_v6_events(
        events,
        config=cfg,
        normalization_center=(0.0, 0.0, 0.0), normalization_scale=2.0,
        mesh_lineage_hash="1" * 64, feature_evidence_binding_hash="2" * 64,
        carrier_evidence_hash="3" * 64, carrier_topology_hash="4" * 64, carrier_geometry_hash="5" * 64,
        model_provenance="axis-v6-test", codec_binding=codec,
    )
    qualified = qualify_rig_program_v1(rig)
    assert qualified.qualification_report["control_synthesis_performed"] is False
    assert len(rig.controls) == 3
    assert rig.controls[0].control_kind == "ROOT"
    assert rig.controls[1].control_kind == "IK_EFFECTOR"
    assert rig.controls[2].control_kind == "IK_POLE"
    assert rig.constraints[0].operator_id == "IK"
    assert rig.constraints[0].input_proposal_control_ids == ("P:C:0001", "P:C:0002")
    assert rig.constraints[0].output_proposal_joint_ids == ("P:J:0000",)
    validate_skin_token_proposal_v1(
        skin,
        expected_proposal_joint_ids=(rig.joints[0].proposal_joint_id,),
        expected_rig_program_proposal_hash=rig.proposal_lineage_hash,
    )
    assert skin.rows[0].token_ids == (7, 8)


def test_runtime_packages_bind_axis_to_exact_mira_codec(tmp_path):
    torch.manual_seed(5)
    mira_cfg = MIRASkinFSQCodecConfig(
        MIRA_V6_VERTEX_FEATURE_DIM,
        MIRA_V6_JOINT_FEATURE_DIM,
        hidden_dim=24,
        tokens_per_joint=2,
        levels=(3, 3, 3),
    )
    mira = MIRASkinFSQCodecV6(mira_cfg)
    mira_path = tmp_path / "mira_v6.pt"
    torch.save(build_mira_runtime_checkpoint_v2(mira), mira_path)
    mira_sha = hashlib.sha256(mira_path.read_bytes()).hexdigest()
    loaded_mira, _ = load_mira_runtime_model_v2(mira_path, device=torch.device("cpu"))
    assert loaded_mira.config.vocab_size == mira_cfg.vocab_size

    axis_cfg = AXISRigProgramConfigV6(
        scene_dim=AXIS_V6_SURFACE_SCENE_DIM,
        hidden_dim=32,
        layers=1,
        heads=4,
        ff_mult=2,
        max_sequence_length=64,
        max_pointer_entities=16,
        skin_vocab_size=mira_cfg.vocab_size,
    )
    axis = AXISRigProgramModelV6(axis_cfg)
    package = build_axis_runtime_checkpoint_v2(
        axis,
        skin_codec_architecture_id=MIRA_V6_CODEC_ID,
        skin_codec_checkpoint_sha256=mira_sha,
        skin_codebook_size=mira_cfg.vocab_size,
        tokens_per_joint=mira_cfg.tokens_per_joint,
    )
    axis_path = tmp_path / "axis_v6.pt"
    torch.save(package, axis_path)
    loaded_axis, receipt = load_axis_runtime_model_v2(axis_path, device=torch.device("cpu"))
    assert loaded_axis.config.skin_vocab_size == mira_cfg.vocab_size
    assert receipt["skin_codec_binding"]["checkpoint_sha256"] == mira_sha
    assert receipt["runtime_rules"]["compiler_token_search_allowed"] is False
