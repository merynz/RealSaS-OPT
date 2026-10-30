from __future__ import annotations

import torch

from compiler.realsas_compiler_core.rig_program_ir_v1 import (
    RigProgramChainProposalIR,
    RigProgramConstraintProposalIR,
    RigProgramControlProposalIR,
    RigProgramJointProposalIR,
    RigProgramLocalFrameProposalIR,
    RigProgramProposalIR,
    seal_rig_program_proposal_v1,
)
from compiler.realsas_compiler_core.skin_token_ir_v1 import (
    JointSkinTokenRowIR,
    SkinTokenProposalIR,
    seal_skin_token_proposal_v1,
)
from models.axis.rig_program_decode_v6 import AXISV6CodecBinding, decode_axis_v6_events
from models.axis.rig_program_model_v6 import (
    AXISRigProgramConfigV6,
    AXISRigProgramModelV6,
    RigTokenKindV6,
)
from models.axis.training_v6 import (
    axis_v6_causal_loss,
    collate_axis_v6_causal_batch,
    serialize_axis_v6_teacher_events,
)


def _truth():
    proposal = seal_rig_program_proposal_v1(RigProgramProposalIR(
        joints=(
            RigProgramJointProposalIR("J0", (0.0, 0.0, 0.0), None, sequence_index=0),
            RigProgramJointProposalIR("J1", (0.0, 1.0, 0.0), "J0", sequence_index=1),
        ),
        local_frames=(
            RigProgramLocalFrameProposalIR("J0", ((1, 0, 0), (0, 1, 0), (0, 0, 1)), ("RX", "RY", "RZ")),
            RigProgramLocalFrameProposalIR("J1", ((1, 0, 0), (0, 1, 0), (0, 0, 1)), ("RX", "RY")),
        ),
        controls=(
            RigProgramControlProposalIR("C0", "ROOT", ("J0",), None, (0.0, 0.0, 0.0)),
            RigProgramControlProposalIR("C1", "IK_EFFECTOR", ("J1",), "C0", (0.0, 1.0, 0.0)),
            RigProgramControlProposalIR("C2", "IK_POLE", (), "C0", (0.5, 0.5, 0.0)),
        ),
        chains=(RigProgramChainProposalIR("K0", ("J0", "J1")),),
        constraints=(
            RigProgramConstraintProposalIR(
                "Q0", "IK", ("C1", "C2"), ("J1",), "K0",
                parameters={"p0": 1.0, "p1": 0.0},
            ),
        ),
        attachment_anchors=(),
        mesh_lineage_hash="1" * 64,
        feature_evidence_binding_hash="2" * 64,
        model_provenance="truth",
    ))
    codes = seal_skin_token_proposal_v1(SkinTokenProposalIR(
        rows=(
            JointSkinTokenRowIR("J0", (1, 2)),
            JointSkinTokenRowIR("J1", (3, 4)),
        ),
        rig_program_proposal_hash=proposal.proposal_lineage_hash,
        carrier_evidence_hash="3" * 64,
        carrier_topology_hash="4" * 64,
        carrier_geometry_hash="5" * 64,
        codec_architecture_id="RealSaS.MIRA.CarrierNativeFSQSkinCodec.v6_0",
        codec_checkpoint_hash="a" * 64,
        codebook_size=27,
        tokens_per_joint=2,
        model_provenance="truth",
    ))
    return proposal, codes


def test_teacher_serialization_is_runtime_language_and_roundtrips_variable_arity():
    cfg = AXISRigProgramConfigV6(scene_dim=12, hidden_dim=32, layers=1, heads=4, max_pointer_entities=16, skin_vocab_size=27)
    proposal, codes = _truth()
    events = serialize_axis_v6_teacher_events(
        proposal, codes, config=cfg, normalization_center=(0.0, 0.0, 0.0), normalization_scale=2.0,
    )
    assert events[0].kind == RigTokenKindV6.BOS
    assert events[-1].kind == RigTokenKindV6.EOS
    assert sum(e.kind == RigTokenKindV6.CONSTRAINT_INPUT for e in events) == 2
    assert sum(e.kind == RigTokenKindV6.CONTROL_TARGET for e in events) == 2
    assert sum(e.kind == RigTokenKindV6.SKIN_CODE for e in events) == 4

    decoded, decoded_codes = decode_axis_v6_events(
        events,
        config=cfg,
        normalization_center=(0.0, 0.0, 0.0), normalization_scale=2.0,
        mesh_lineage_hash=proposal.mesh_lineage_hash,
        feature_evidence_binding_hash=proposal.feature_evidence_binding_hash,
        carrier_evidence_hash=codes.carrier_evidence_hash,
        carrier_topology_hash=codes.carrier_topology_hash,
        carrier_geometry_hash=codes.carrier_geometry_hash,
        model_provenance="roundtrip",
        codec_binding=AXISV6CodecBinding(codes.codec_architecture_id, codes.codec_checkpoint_hash, 27, 2),
    )
    assert len(decoded.joints) == 2
    assert tuple(c.control_kind for c in decoded.controls) == ("ROOT", "IK_EFFECTOR", "IK_POLE")
    assert decoded.constraints[0].input_proposal_control_ids == ("P:C:0001", "P:C:0002")
    assert decoded.constraints[0].output_proposal_joint_ids == ("P:J:0001",)
    assert tuple(row.token_ids for row in decoded_codes.rows) == ((1, 2), (3, 4))


def test_typed_causal_objective_is_finite_and_backpropagates():
    torch.manual_seed(17)
    cfg = AXISRigProgramConfigV6(
        scene_dim=12, hidden_dim=32, layers=1, heads=4, ff_mult=2,
        max_sequence_length=128, max_pointer_entities=16, skin_vocab_size=27,
    )
    proposal, codes = _truth()
    events = serialize_axis_v6_teacher_events(
        proposal, codes, config=cfg, normalization_center=(0.0, 0.0, 0.0), normalization_scale=2.0,
    )
    batch = collate_axis_v6_causal_batch((events, events), device=torch.device("cpu"))
    model = AXISRigProgramModelV6(cfg)
    scene = torch.randn(2, 9, cfg.scene_dim)
    losses = axis_v6_causal_loss(model, scene, batch)
    assert torch.isfinite(losses["total"])
    assert float(losses["skin_code"]) > 0.0
    assert float(losses["kind"]) > 0.0
    losses["total"].backward()
    grad = model.kind_head.weight.grad
    assert grad is not None and torch.isfinite(grad).all()
    assert float(grad.abs().sum()) > 0.0
