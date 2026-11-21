from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from models.iris.vnext.contracts_v1 import IRISVNextConfigV1
from models.iris.vnext.scene_core_v1 import SharedSceneTriplaneVNextV1
from models.rss_vnext import (
    RSSvNextConfig, RSSvNextPatchEncoderForTests,
    RSSvNextUnifiedCharacterModel,
)
from models.rss_vnext.flexicubes_surface_evidence import (
    RSSvNextFlexiCubesSurfaceEvidence,
)
from models.rss_vnext.surface_topology_context_v1 import (
    iris_surface_incidence_v1,
)
from models.rss_vnext.training_engine import (
    RSSvNextStructuralTrainer, RSSvNextStructuralTrainingBatch,
    RSSvNextTrainingWeights,
)


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.manual_seed(31)
    torch.set_num_threads(1)


def test_production_iris_matches_stdgen_transformer_scale_with_large_encoder():
    cfg = RSSvNextConfig()
    assert cfg.iris.image_token_dim == 1024  # Frozen DINOv2-L/14 + four registers.
    assert cfg.iris.scene_dim == 1024
    assert cfg.iris.scene_layers == 16
    assert cfg.iris.attention_heads == 16
    assert cfg.iris.plane_size == 32
    assert cfg.iris.plane_channels == 80
    assert cfg.iris.scene_gradient_checkpointing
    assert cfg.tessa_surface_requirement == "REQUIRE_SIGNED_IRIS_SURFACE"
    assert cfg.semantic_role_schema().count == 4
    cfg.validate()


def test_scene_activation_checkpointing_matches_plain_transformer_and_gradients():
    cfg = replace(
        RSSvNextConfig.tiny_for_test().iris,
        scene_layers=2, scene_gradient_checkpointing=True,
    )
    a = SharedSceneTriplaneVNextV1(cfg).train()
    b = SharedSceneTriplaneVNextV1(
        replace(cfg, scene_gradient_checkpointing=False),
    ).train()
    b.load_state_dict(a.state_dict())
    memory = torch.randn(1, 9, cfg.scene_dim, requires_grad=True)
    memory_copy = memory.detach().clone().requires_grad_()
    x = a(memory)
    y = b(memory_copy)
    torch.testing.assert_close(x, y, atol=1.e-6, rtol=1.e-6)
    x.square().mean().backward()
    y.square().mean().backward()
    torch.testing.assert_close(memory.grad, memory_copy.grad, atol=1.e-6, rtol=1.e-5)
    torch.testing.assert_close(
        a.scene_queries.grad, b.scene_queries.grad, atol=1.e-6, rtol=1.e-5,
    )


def test_iris_topology_descriptors_change_with_faces_and_backprop_to_xyz():
    xyz = torch.tensor([[
        [-0.3, -0.4, 0.0], [0.5, -0.2, 0.1],
        [0.2, 0.6, -0.2], [0.0, 0.0, 0.7],
    ]], requires_grad=True)
    fa = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.long)
    fb = torch.tensor([[0, 1, 3], [1, 2, 3]], dtype=torch.long)
    va, ta = iris_surface_incidence_v1(xyz, fa)
    vb, tb = iris_surface_incidence_v1(xyz, fb)
    assert va.shape == vb.shape == (1, 4, 8)
    assert ta.shape == tb.shape == (1, 2, 10)
    assert not torch.allclose(va, vb)
    assert not torch.allclose(ta, tb)
    vr, tr = iris_surface_incidence_v1(xyz, fa[:, [1, 2, 0]])
    torch.testing.assert_close(va, vr, atol=1.e-6, rtol=1.e-6)
    torch.testing.assert_close(ta, tr, atol=1.e-6, rtol=1.e-6)
    (va.square().mean() + ta.square().mean()).backward()
    assert xyz.grad is not None
    assert torch.isfinite(xyz.grad).all()
    assert xyz.grad.abs().sum() > 0


def test_tessa_memory_responds_to_iris_face_incidence_not_only_vertex_positions():
    cfg = RSSvNextConfig.tiny_for_test()
    net = RSSvNextUnifiedCharacterModel(
        cfg, vision_encoder=RSSvNextPatchEncoderForTests(cfg.iris.image_token_dim),
    ).train()
    rgba = torch.rand(1, 2, 4, 8, 8)
    cam = torch.zeros(1, 2, 13)
    cam[..., 3] = cam[..., 7] = cam[..., 11] = cam[..., 12] = 1
    state, aux, _ = net.encode_observations(rgba, cam)
    # All geometry, local H and semantic evidence stay fixed. Only face
    # incidence changes. TESSA must see that difference.
    xyz = torch.tensor([[
        [-0.3, -0.4, 0.0], [0.5, -0.2, 0.1],
        [0.2, 0.6, -0.2], [0.0, 0.0, 0.7],
    ]], requires_grad=True)
    h = net.iris.query(state.scene_triplanes, xyz)["H"]
    witness = RSSvNextFlexiCubesSurfaceEvidence(
        xyz=xyz, faces=torch.tensor([[0,1,2],[0,2,3]], dtype=torch.long),
        local_features=h,
        presentation_role_logits=net.presentation_role_head(h),
        regularizer=h.new_zeros(1),
        sample_grid_xyz=state.query_xyz,
        signed_field_grid=state.geometry["field"],
        cube_parameters=h.new_zeros(1, 21),
        source_policy="SIGNED_HYPOTHESIS_ONLY",
        layer_name="all",
    )
    witnessed_a = net._structural_memory(state, aux, flexicubes=witness)
    witnessed_b = net._structural_memory(
        state, aux,
        flexicubes=replace(
            witness, faces=torch.tensor([[0,1,3],[1,2,3]], dtype=torch.long),
        ),
    )
    assert not torch.allclose(witnessed_a, witnessed_b)
    logits = net.tessa.decode_tokens(
        memory=witnessed_a,
        input_ids=torch.full((1, 4), cfg.tessa.BOS, dtype=torch.long),
    ).logits
    logits.square().mean().backward()
    assert net.fc_surface_incidence[1].weight.grad is not None
    assert net.fc_surface_incidence[1].weight.grad.abs().sum() > 0
    assert net.fc_face_geometry[1].weight.grad is not None
    assert net.fc_face_geometry[1].weight.grad.abs().sum() > 0
    assert xyz.grad is not None and xyz.grad.abs().sum() > 0
    assert net.iris.scene.scene_queries.grad is not None


def test_product_topology_training_refuses_missing_iris_surface():
    cfg = replace(
        RSSvNextConfig.tiny_for_test(),
        tessa_surface_requirement="REQUIRE_SIGNED_IRIS_SURFACE",
    )
    net = RSSvNextUnifiedCharacterModel(
        cfg, vision_encoder=RSSvNextPatchEncoderForTests(cfg.iris.image_token_dim),
    ).train()
    weights = RSSvNextTrainingWeights(
        topology=1.0, motion_preference=0.0, source_semantics=0.0,
        flexicubes_surface=0.0,
    )
    trainer = RSSvNextStructuralTrainer(
        net, torch.optim.AdamW(net.parameters(), lr=1.e-4),
        dataset_manifest_sha256="a" * 64, weights=weights,
    )
    rgba = torch.rand(1, 2, 4, 8, 8)
    cam = torch.zeros(1, 2, 13)
    cam[..., 3] = cam[..., 7] = cam[..., 11] = cam[..., 12] = 1
    batch = RSSvNextStructuralTrainingBatch(
        rgba=rgba, camera_features=cam,
        topology_teacher_tokens=torch.tensor([[
            cfg.tessa.BOS, cfg.tessa.COMPONENT_BEGIN, cfg.tessa.EOS,
        ]]),
        topology_valid_mask=torch.tensor([True]),
        source_qualification_key="synthetic-only",
        admission_status="QUALIFIED_FOR_THIS_SUPERVISION_FAMILY",
    )
    with pytest.raises(ValueError, match="TESSA_IRIS_SURFACE_REQUIRED"):
        trainer.objective(batch)


def test_alignment_loss_cannot_be_activated_without_source_receipt():
    cfg = RSSvNextConfig.tiny_for_test()
    net = RSSvNextUnifiedCharacterModel(
        cfg, vision_encoder=RSSvNextPatchEncoderForTests(cfg.iris.image_token_dim),
    ).train()
    weights = RSSvNextTrainingWeights(
        topology=0.0, motion_preference=0.0, source_semantics=0.0,
        flexicubes_surface=0.0, tessa_iris_surface_alignment=1.0,
    )
    trainer = RSSvNextStructuralTrainer(
        net, torch.optim.AdamW(net.parameters(), lr=1.e-4),
        dataset_manifest_sha256="a" * 64, weights=weights,
    )
    rgba = torch.rand(1, 2, 4, 8, 8)
    cam = torch.zeros(1, 2, 13)
    cam[..., 3] = cam[..., 7] = cam[..., 11] = cam[..., 12] = 1
    with pytest.raises(ValueError, match="TESSA_IRIS_ALIGNMENT_SOURCE_REQUIRED"):
        trainer.objective(RSSvNextStructuralTrainingBatch(
            rgba=rgba, camera_features=cam,
            source_qualification_key="synthetic-only",
            admission_status="QUALIFIED_FOR_THIS_SUPERVISION_FAMILY",
        ))
