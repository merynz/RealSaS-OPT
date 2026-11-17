from __future__ import annotations

"""Full H(x) and separate garment-vs-body surface identity remain learnable.

Synthetic mesh is a gradient/identity regression, not a performance result.
"""
from dataclasses import replace

import pytest
import torch

from models.rss_vnext import (
    RSSvNextConfig, RSSvNextPatchEncoderForTests,
    RSSvNextUnifiedCharacterModel,
)
from models.rss_vnext.tessa_seeded_surface_v1 import (
    build_tessa_seeded_surface_v1,
)
from tests.models.test_rss_vnext_unified_character_model import sequence_two_joints


def _net():
    torch.manual_seed(710)
    torch.set_num_threads(1)
    cfg = RSSvNextConfig.tiny_for_test()
    return RSSvNextUnifiedCharacterModel(
        cfg, vision_encoder=RSSvNextPatchEncoderForTests(cfg.iris.image_token_dim),
    ).train()


def _state(net):
    rgba = torch.rand(1, 2, 4, 8, 8)
    cam = torch.zeros(1, 2, 13)
    cam[..., 3] = cam[..., 7] = cam[..., 11] = cam[..., 12] = 1
    state, _, _ = net.encode_observations(rgba, cam)
    return state


def _seed(x=None, component=None):
    if x is None:
        x = torch.tensor([[
            [-0.35, 0.3, 0.0], [0.4, 0.3, 0.0],
            [-0.35, -0.45, 0.0], [0.4, -0.45, 0.0],
        ]], requires_grad=True)
    if component is None:
        component = torch.zeros(x.shape[1], dtype=torch.long)
    return build_tessa_seeded_surface_v1(
        iris_surface_xyz=x,
        iris_surface_faces=torch.tensor([[0, 2, 1], [1, 2, 3]]),
        component_ids_by_vertex=component,
        scene_sha256="a" * 64, original_surface_sha256="b" * 64,
    )


def test_seeded_axis_mira_same_mesh_full_h_and_hard_fsq_decoding():
    net = _net()
    state = _state(net)
    seed = _seed()
    result = net.build_seeded_mechanics_candidate_v1(
        state, seed, rig_sequence=sequence_two_joints(),
    )
    m = result["mesh_candidate"]
    assert m.seeded_surface is seed
    assert m.faces is seed.faces
    assert m.local_features.shape == m.mechanics_local_features.shape
    assert result["axis_control_conditioning"] == (
        "FULL_RAW_H_PLUS_ALL_INDEXED_FACE_CONTEXT"
    )
    assert torch.allclose(m.local_features, m.mechanics_local_features)
    w = result["skin_weight_proposal"]
    hard = result["discrete_skin_weight_proposal"]
    assert w.shape == hard.shape == (1, seed.xyz.shape[1], 2)
    torch.testing.assert_close(w, hard, atol=1e-5, rtol=1e-5)
    assert result["dynamic_mesh_candidate"].source_mesh is m
    # Neither H -> face context nor H -> MIRA is detached.
    loss = (w[:, :, 0] - 0.35).square().mean()
    loss.backward()
    last = net.surface_mechanics_context.conditioner[-1]
    assert last.weight.grad is not None
    assert last.weight.grad.abs().sum() > 0
    assert net.iris.scene.scene_queries.grad is not None
    assert net.iris.scene.scene_queries.grad.abs().sum() > 0
    assert seed.xyz.grad is not None and seed.xyz.grad.abs().sum() > 0


def test_coincident_covering_and_body_remain_distinguishable_with_same_H():
    net = _net()
    state = _state(net)
    # Two disconnected shells coincide in XYZ; they must not weld.
    xyz = torch.tensor([[
        [-0.2, 0.2, 0.0], [0.4, 0.2, 0.0], [-0.2, -0.4, 0.0],
        [-0.2, 0.2, 0.0], [0.4, 0.2, 0.0], [-0.2, -0.4, 0.0],
    ]], requires_grad=True)
    seed = build_tessa_seeded_surface_v1(
        iris_surface_xyz=xyz,
        iris_surface_faces=torch.tensor([[0, 2, 1], [3, 5, 4]]),
        component_ids_by_vertex=torch.tensor([0, 0, 0, 1, 1, 1]),
        scene_sha256="a"*64, original_surface_sha256="b"*64,
    )
    mesh = net.mesh_bridge.forward_seeded(
        state.scene_triplanes, seed,
        refine=net.surface_refine, semantics=net.semantic,
        affordances=net.affordance, graphics=net.graphics,
        quality=net.deformation_quality, udf=net.udf,
        sign_logit=net.sign_logit, role_head=net.presentation_role_head,
        role_schema_id=net.cfg.semantic_role_schema_id,
    )
    # Geometry/H identical for first and second component. The original
    # semantic evidence can DIFFER when source-qualified 3D supervision exists.
    assert torch.allclose(mesh.local_features[:, 0], mesh.local_features[:, 3])
    roles = mesh.presentation_role_logits.detach().clone()
    roles[0, :3] = torch.tensor([-8., -8., 8., -8.])
    roles[0, 3:] = torch.tensor([-8., -8., -8., 8.])
    roles.requires_grad_()
    mesh = replace(mesh, presentation_role_logits=roles)
    ctx = net.surface_mechanics_context
    with torch.no_grad():
        ctx.conditioner[-1].weight.normal_(0, 0.15)
    augmented = ctx(mesh)
    assert not torch.allclose(augmented[:, 0], augmented[:, 3], atol=1e-6)
    assert torch.allclose(mesh.local_features[:, 0], mesh.local_features[:, 3])
    (augmented[:, 3] - augmented[:, 0]).square().mean().backward()
    assert roles.grad is not None
    assert roles.grad.abs().sum() > 0
    assert xyz.grad is not None
    assert torch.isfinite(xyz.grad).all()


def test_conditioned_MIRA_features_have_explicit_shape_and_address_check():
    net = _net()
    state = _state(net)
    mesh = net.build_seeded_mechanics_candidate_v1(state, _seed())["mesh_candidate"]
    with pytest.raises(ValueError, match="MECHANICS_FEATURE_ADDRESS_DRIFT"):
        replace(
            mesh, mechanics_local_features=torch.randn(
                1, mesh.rest_vertices_iris.shape[1] - 1,
                net.cfg.iris.local_feature_dim,
            ),
        ).validate()
