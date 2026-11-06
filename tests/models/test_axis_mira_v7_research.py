from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from models.mira.skin_fsq_codec_v7 import (
    MIRACodecConfigV7, MIRASkinCodecV7,
)
from models.axis.rig_program_model_v7 import (
    AXISRigProgramConfigV7, AXISRigProgramModelV7,
    assert_axis_mira_v7_binding, axis_v7_teacher_forced_objective,
)
from models.axis.rig_program_model_v6 import (
    AXISSequenceV6, RigProgramEventV6, RigTokenKindV6,
)
from compiler.realsas_compiler_core.mechanics_interaction_v1 import (
    select_mechanics_controls_from_surface_v1,
)
from compiler.realsas_compiler_core.types import QualificationError


def test_mira_v7_reconstruction_gradient_and_hard_token_parity():
    torch.manual_seed(701)
    cfg = MIRACodecConfigV7(
        vertex_feature_dim=7, joint_feature_dim=25,
        hidden_dim=32, tokens_per_joint=2, levels=(3, 3),
        max_vertices=128, max_joints=8,
    )
    model = MIRASkinCodecV7(cfg)
    v = torch.randn(2, 12, 7)
    j = torch.randn(2, 3, 25)
    w = torch.softmax(torch.randn(2, 12, 3), dim=-1)
    result = model(w, v, j)
    assert result["token_ids"].shape == (2, 3, 2)
    assert torch.isfinite(result["loss"])
    assert result["simplex_max_abs"] < 1e-6
    # Critical regression: reconstruction alone MUST train the MIRA encoder.
    result["reconstruction_loss"].backward()
    grad = model.latent_encoder[-1].weight.grad
    assert grad is not None and torch.isfinite(grad).all()
    assert float(grad.abs().sum()) > 0.0

    with torch.no_grad():
        decoded = model.decode_tokens(result["token_ids"], v, j)
        assert torch.allclose(decoded, result["reconstructed_weights"], atol=1e-6, rtol=0.0)


def test_mira_v7_sparse_positive_weights_not_all_zero():
    model = MIRASkinCodecV7(MIRACodecConfigV7(
        hidden_dim=24, tokens_per_joint=2, levels=(3, 3),
        max_vertices=24, max_joints=4,
    ))
    vertices = torch.randn(1, 5, 7)
    joints = torch.randn(1, 2, 25)
    weights = torch.tensor([[[1., 0.], [0., 1.], [1., 0.], [0.5, 0.5], [0., 1.]]])
    result = model(weights, vertices, joints, mesh_edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]]))
    assert torch.isfinite(result["edge_gradient_loss"])
    with pytest.raises(ValueError, match="UNCOVERED"):
        model(torch.zeros_like(weights), vertices, joints)


def test_axis_v7_large_vocab_contract_and_sparse_skin_loss():
    cfg = AXISRigProgramConfigV7(
        scene_dim=47, hidden_dim=32, layers=1, heads=4,
        feedforward_mult=2, max_sequence_length=20,
        max_pointer_entities=6, skin_vocab_size=9, tokens_per_joint=2,
    )
    assert_axis_mira_v7_binding(cfg, MIRACodecConfigV7(
        hidden_dim=32, tokens_per_joint=2, levels=(3, 3)
    ))
    with pytest.raises(ValueError, match="VOCAB"):
        assert_axis_mira_v7_binding(cfg, MIRACodecConfigV7(
            hidden_dim=32, tokens_per_joint=2, levels=(5, 5)
        ))
    model = AXISRigProgramModelV7(cfg)
    inputs = AXISSequenceV6.from_events((
        RigProgramEventV6(RigTokenKindV6.BOS),
        RigProgramEventV6(RigTokenKindV6.JOINT, coord_indices=(1, 2, 3), pointer_a=-1),
    ), device=torch.device("cpu"))
    targets = AXISSequenceV6.from_events((
        RigProgramEventV6(RigTokenKindV6.JOINT, coord_indices=(1, 2, 3), pointer_a=-1),
        RigProgramEventV6(RigTokenKindV6.SKIN_CODE, skin_code_id=4),
    ), device=torch.device("cpu"))
    result = axis_v7_teacher_forced_objective(
        model, torch.randn(1, 5, 47), inputs, targets,
    )
    assert torch.isfinite(result["total"])
    assert result["skin_code"] > 0
    result["total"].backward()
    assert model.skin_embedding.weight.grad is not None
    assert model.kind_head.weight.grad is not None


def _mechanics_fixture():
    joints = (
        SimpleNamespace(canonical_joint_id="J0"),
        SimpleNamespace(canonical_joint_id="J1"),
    )
    controls = (
        SimpleNamespace(canonical_control_id="FK1", control_kind="FK",
                        target_canonical_joint_ids=("J1",)),
        SimpleNamespace(canonical_control_id="IK1", control_kind="IK_EFFECTOR",
                        target_canonical_joint_ids=()),
        SimpleNamespace(canonical_control_id="GLOBAL", control_kind="GLOBAL",
                        target_canonical_joint_ids=()),
    )
    chains = (SimpleNamespace(canonical_chain_id="CH", canonical_joint_ids=("J0", "J1")),)
    constraints = (
        SimpleNamespace(operator_id="IK", canonical_chain_id="CH",
                        input_canonical_control_ids=("IK1",)),
    )
    rig = SimpleNamespace(
        mesh_lineage_hash="mesh", deform_structure_hash="deform",
        rig_lineage_hash="rig", joints=joints, controls=controls,
        chains=chains, constraints=constraints,
    )
    rows = (
        SimpleNamespace(canonical_mesh_vertex_id="V0", influences=(("J0", 0.1), ("J1", 0.9))),
        SimpleNamespace(canonical_mesh_vertex_id="V1", influences=(("J1", 1.0),)),
        SimpleNamespace(canonical_mesh_vertex_id="V2", influences=(("J0", 0.2), ("J1", 0.8))),
    )
    skin = SimpleNamespace(
        mesh_lineage_hash="mesh", deform_structure_hash="deform",
        skin_lineage_hash="skin", rows=rows,
    )
    return rig, skin


def test_mechanics_pick_is_anonymous_readonly_and_declared_only():
    rig, skin = _mechanics_fixture()
    out = select_mechanics_controls_from_surface_v1(
        rig, skin, triangle_vertex_ids=("V0", "V1", "V2"),
        barycentric=(0.2, 0.5, 0.3),
    )
    assert out.status == "CANDIDATES"
    assert tuple(c.canonical_control_id for c in out.candidates) == ("FK1", "IK1")
    assert out.candidates[0].influence_score > 0.9
    assert rig.controls[1].target_canonical_joint_ids == ()  # never synthesized
    with pytest.raises(QualificationError, match="SIMPLEX"):
        select_mechanics_controls_from_surface_v1(
            rig, skin, triangle_vertex_ids=("V0", "V1", "V2"),
            barycentric=(0.5, 0.5, 0.5),
        )
    with pytest.raises(QualificationError, match="LINEAGE"):
        bad_skin = SimpleNamespace(**{**skin.__dict__, "deform_structure_hash": "wrong"})
        select_mechanics_controls_from_surface_v1(
            rig, bad_skin, triangle_vertex_ids=("V0", "V1", "V2"),
            barycentric=(0.2, 0.5, 0.3),
        )


def test_mira_v7_attention_cv_ae_active_sampling_gradient_and_token_parity():
    from models.mira.skin_attention_codec_v7 import (
        MIRAAttentionConfigV7, MIRAAttentionSkinCodecV7,
    )
    torch.manual_seed(711)
    cfg = MIRAAttentionConfigV7(
        hidden_dim=32, attention_heads=4, attention_layers=2,
        active_samples_per_joint=5, tokens_per_joint=2,
        levels=(3, 3), max_vertices=128, max_joints=8,
    )
    model = MIRAAttentionSkinCodecV7(cfg)
    vertices = torch.randn(2, 19, 7)
    joints = torch.randn(2, 4, 25)
    truth = torch.softmax(torch.randn(2, 19, 4) * 2, dim=-1)
    indices, weights = model._active_samples(truth)
    assert indices.shape == (2, 4, 5)
    assert torch.all(weights[:, :, :-1] >= weights[:, :, 1:])
    out = model(truth, vertices, joints)
    assert out["token_ids"].shape == (2, 4, 2)
    out["reconstruction_loss"].backward()
    assert model.learned_queries.grad is not None
    assert float(model.learned_queries.grad.abs().sum()) > 0.0
    assert model.attention_latent.weight.grad is not None
    assert float(model.attention_latent.weight.grad.abs().sum()) > 0.0
    with torch.no_grad():
        decoded = model.decode_tokens(out["token_ids"], vertices, joints)
        assert torch.allclose(out["reconstructed_weights"], decoded, atol=1e-6, rtol=0.0)
