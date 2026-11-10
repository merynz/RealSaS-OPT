from __future__ import annotations

"""Falsifiable SkinTokens→RSS shared-carrier and exact indexed-skin tests."""
from dataclasses import replace
import numpy as np
import pytest
import torch

from models.axis.rig_program_model_v6 import (
    AXISSequenceV6, RigProgramEventV6, RigTokenKindV6,
)
from models.rss_vnext import (
    RSSvNextConfig, RSSvNextPatchEncoderForTests,
    RSSvNextUnifiedCharacterModel,
)
from models.rss_vnext.skin_query_field import RSSvNextJointSkinQueryField
from models.tessa.v1.model_v1 import TESSAConfigV1
from models.tessa.v1.tokenization_v1 import encode_adjacent_teacher_mesh_v1


@pytest.fixture(autouse=True)
def reproducible():
    torch.manual_seed(41)
    torch.set_num_threads(1)


def _query(chunk=2):
    return RSSvNextJointSkinQueryField(
        12, 8, 4, tokens_per_joint=4, hidden_dim=16,
        heads=4, max_query_chunk=chunk,
    )


def _source_query_inputs():
    vh = torch.randn(1, 7, 12, requires_grad=True)
    xyz = torch.randn(1, 7, 3, requires_grad=True)
    nrm = torch.randn(1, 7, 3, requires_grad=True)
    joints = torch.randn(1, 3, 8, requires_grad=True)
    code = torch.randn(1, 3, 4, 4, requires_grad=True)
    mask = torch.tensor([[True, False, True]])
    return vh, xyz, nrm, joints, code, mask


def test_full_code_token_cross_attention_is_vertex_addressable_and_differentiable():
    field = _query()
    vh, xyz, nrm, joints, code, mask = _source_query_inputs()
    result = field(
        vh, xyz, joints, code, mask, vertex_normals=nrm,
    )
    assert result.weights.shape == (1, 7, 3)
    assert result.unnormalized_logits.shape == (1, 7, 3)
    assert result.valid_joint_mask.equal(mask)
    assert torch.equal(result.weights[:, :, 1], torch.zeros(1, 7))
    assert torch.allclose(
        result.weights.sum(-1), torch.ones(1, 7), atol=1e-6,
    )
    # Do not optimize a constant sum. Different joint targets must produce a
    # real gradient to *both* query H and FSQ code attention.
    objective = (
        ((result.weights[0, :, 0] - torch.linspace(.1, .9, 7)) ** 2).mean()
        + .01 * result.unnormalized_logits[:, :, 0].square().mean()
    )
    objective.backward()
    for tensor in (vh, xyz, nrm, joints, code):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
        assert tensor.grad.abs().sum() > 0
    assert field.cross_attention.in_proj_weight.grad is not None
    assert field.cross_attention.in_proj_weight.grad.abs().sum() > 0


def test_streamed_chunk_size_does_not_change_query_results_or_gradients():
    a = _query(chunk=1)
    b = _query(chunk=7)
    b.load_state_dict(a.state_dict(), strict=True)
    vh, xyz, nrm, joints, code, mask = _source_query_inputs()
    x = a(vh, xyz, joints, code, mask, vertex_normals=nrm).weights
    y = b(vh, xyz, joints, code, mask, vertex_normals=nrm).weights
    assert torch.allclose(x, y, atol=1e-6, rtol=1e-6)
    grad_x = torch.autograd.grad(
        x[:, :, 0].square().mean(), vh, retain_graph=True,
    )[0]
    grad_y = torch.autograd.grad(
        y[:, :, 0].square().mean(), vh,
    )[0]
    assert torch.allclose(grad_x, grad_y, atol=1e-6, rtol=1e-5)


def test_no_joint_evidence_and_address_corruption_abstain():
    field = _query()
    vh, xyz, nrm, joints, code, mask = _source_query_inputs()
    with pytest.raises(ValueError, match="ALL_JOINTS_UNKNOWN"):
        field(vh, xyz, joints, code, torch.zeros_like(mask))
    with pytest.raises(ValueError, match="CODE_ADDRESS"):
        field(vh, xyz, joints, code[:, :, :2], mask)
    with pytest.raises(ValueError, match="NORMAL_ADDRESS"):
        field(vh, xyz, joints, code, mask, vertex_normals=nrm[:, :4])
    with pytest.raises(ValueError, match="NONFINITE"):
        bad = vh.detach().clone()
        bad[:, 0] = float("nan")
        field(bad, xyz, joints, code, mask)


def _rig_sequence():
    kinds = [
        RigTokenKindV6.BOS,
        RigTokenKindV6.JOINT, RigTokenKindV6.JOINT,
        RigTokenKindV6.FRAMES_START,
        RigTokenKindV6.FRAME, RigTokenKindV6.FRAME,
        RigTokenKindV6.CONTROLS_START, RigTokenKindV6.CHAINS_START,
        RigTokenKindV6.CONSTRAINTS_START, RigTokenKindV6.ANCHORS_START,
        RigTokenKindV6.SKIN_START,
        *([RigTokenKindV6.SKIN_CODE] * 8),
        RigTokenKindV6.EOS,
    ]
    return AXISSequenceV6.from_events(
        [RigProgramEventV6(x) for x in kinds],
        device=torch.device("cpu"),
    )


def _net():
    cfg = replace(
        RSSvNextConfig.tiny_for_test(),
        skin_decoder_mode="POINT_QUERY_CROSS_ATTENTION",
        skin_query_heads=4, skin_query_chunk=2,
    )
    return RSSvNextUnifiedCharacterModel(
        cfg, vision_encoder=RSSvNextPatchEncoderForTests(
            cfg.iris.image_token_dim,
        ),
    )


def _source():
    rgba = torch.rand(1, 2, 4, 8, 8)
    cam = torch.zeros(1, 2, 13)
    cam[:, :, 3] = cam[:, :, 7] = cam[:, :, 11] = cam[:, :, 12] = 1
    return rgba, cam


def _mesh_sequence(model):
    s = encode_adjacent_teacher_mesh_v1(
        np.array([[-.25, -.25, 0], [.25, -.25, 0], [.25, .25, 0]],dtype=np.float64),
        np.array([[0,1,2]],dtype=np.int64),
        component_id="synthetic-one-component",
        chart_id="synthetic-one-chart",
        cfg=TESSAConfigV1(coordinate_bins=model.cfg.tessa.coordinate_bins),
    )
    return torch.tensor(s.token_ids, dtype=torch.long)


def test_shared_source_to_exact_mesh_to_control_to_query_skin_gradient():
    model = _net().train()
    rgba, cam = _source()
    seq = _rig_sequence()
    out = model(
        rgba, cam, mesh_teacher_tokens=_mesh_sequence(model),
        rig_sequence=seq, topology_prefix=torch.full((1,4), model.cfg.tessa.BOS),
    )
    assert out.mesh_candidate is not None
    mesh = out.mesh_candidate
    assert mesh.faces.shape == (1, 3)
    assert out.skin_weight_proposal.shape == (1, mesh.rest_vertices_iris.shape[1], 2)
    assert out.skin_joint_valid_mask.tolist() == [[True, True]]
    assert out.dynamic_mesh_candidate is not None
    # The same exact topology address domain must traverse all experts.
    assert mesh.vertex_component_ids.shape == (mesh.rest_vertices_iris.shape[1],)
    assert out.dynamic_mesh_candidate.vertices.shape[:2] == mesh.rest_vertices_iris.shape[:2]
    objective = (
        (out.skin_weight_proposal[:, :, 0] - .83).square().mean()
        + .01 * out.structural_logits.square().mean()
    )
    objective.backward()
    for param in (
        model.vision.patch.weight,
        model.iris.scene.scene_queries,
        model.tessa.memory.scene.queries,
        model.axis.skin_code_head.weight,
        model.mira.spatial_decoder.cross_attention.in_proj_weight,
        model.mira.spatial_decoder.vertex_projection[1].weight,
        model.mira.spatial_decoder.code_projection[1].weight,
    ):
        assert param.grad is not None and torch.isfinite(param.grad).all()
        assert param.grad.abs().sum() > 0


def test_first_skin_token_label_cannot_leak_to_its_prediction():
    model = _net().eval()
    rgba, cam = _source()
    source = _rig_sequence()
    from copy import deepcopy
    forged = deepcopy(source)
    first = int(torch.where(
        source.kind_ids[0] == int(RigTokenKindV6.SKIN_CODE),
    )[0][0])
    forged.skin_code_ids[0, first:] = 13
    with torch.no_grad():
        a = model(rgba, cam, rig_sequence=source)
        b = model(rgba, cam, rig_sequence=forged)
    assert torch.allclose(
        a.control_logits["skin_code_logits"][0, first - 1],
        b.control_logits["skin_code_logits"][0, first - 1],
        atol=1e-6,
    )
    assert a.state.source_view_authority == (
        "SOURCE_OBSERVED_FRONT", "SOURCE_OBSERVED_BACK",
    )
