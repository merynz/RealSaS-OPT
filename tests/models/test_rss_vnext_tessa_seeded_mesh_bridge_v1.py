from __future__ import annotations

"""Actual shared-H seeded TESSA -> AXIS/MIRA-addressed mesh candidate court.

Tiny random weights verify exact identity and gradient plumbing only.
They do not establish any learned reconstruction or cloth motion quality.
"""

from dataclasses import replace
import torch
import pytest

from models.rss_vnext import (
    RSSvNextConfig, RSSvNextPatchEncoderForTests,
    RSSvNextUnifiedCharacterModel,
)
from models.rss_vnext.tessa_seeded_surface_v1 import (
    build_tessa_seeded_surface_v1,
)
from models.rss_vnext.animation_ready_mesh import RSSvNextStructuralMeshCandidate


def test_real_unified_graph_builds_seeded_mesh_without_independent_topology_decoder():
    torch.manual_seed(17011)
    torch.set_num_threads(1)
    cfg = RSSvNextConfig.tiny_for_test()
    model = RSSvNextUnifiedCharacterModel(
        cfg, vision_encoder=RSSvNextPatchEncoderForTests(cfg.iris.image_token_dim),
    ).train()
    rgba = torch.rand(1, 2, 4, 8, 8)
    cameras = torch.zeros(1, 2, 13)
    cameras[..., 3] = cameras[..., 7] = cameras[..., 11] = cameras[..., 12] = 1
    state, _, _ = model.encode_observations(rgba, cameras)
    source_xyz = torch.tensor(
        [[[-0.3, 0.35, 0.0], [0.4, 0.35, 0.0],
          [-0.3, -0.4, 0.0], [0.4, -0.4, 0.0]]],
        requires_grad=True,
    )
    source_faces = torch.tensor([[0, 2, 1], [1, 2, 3]], dtype=torch.long)
    seeded = build_tessa_seeded_surface_v1(
        iris_surface_xyz=source_xyz,
        iris_surface_faces=source_faces,
        component_ids_by_vertex=torch.zeros(4, dtype=torch.long),
        scene_sha256="a" * 64,
        original_surface_sha256="b" * 64,
        one_to_four_refine=True,
    )
    mesh = model.mesh_bridge.forward_seeded(
        state.scene_triplanes, seeded,
        refine=model.surface_refine,
        semantics=model.semantic,
        affordances=model.affordance,
        graphics=model.graphics,
        quality=model.deformation_quality,
        udf=model.udf,
        sign_logit=model.sign_logit,
        role_head=model.presentation_role_head,
        role_schema_id=cfg.semantic_role_schema_id,
    )
    assert type(mesh) is RSSvNextStructuralMeshCandidate
    assert mesh.seeded_surface is seeded
    assert mesh.topology_token_ids == ()
    assert mesh.topology_court is None
    assert mesh.faces is seeded.faces
    assert mesh.base_vertices_iris is seeded.xyz
    assert mesh.rest_vertices_iris.shape == seeded.xyz.shape
    assert mesh.local_features.shape[:2] == (1, seeded.xyz.shape[1])
    assert mesh.presentation_role_logits.shape[-1] == cfg.semantic_role_schema().count
    assert not mesh.canonical_mesh_authority
    loss = (mesh.rest_vertices_iris.square().mean()
            + mesh.graphics_logits.square().mean()
            + mesh.normals.square().mean())
    loss.backward()
    assert source_xyz.grad is not None
    assert torch.isfinite(source_xyz.grad).all()
    assert source_xyz.grad.abs().sum() > 0
    assert model.surface_refine[-1].weight.grad is not None
    assert model.surface_refine[-1].weight.grad.abs().sum() > 0
    assert model.iris.scene.scene_queries.grad is not None

    # The candidate cannot mutate faces/identity independently after TESSA.
    altered = replace(mesh, faces=mesh.faces[:, [0, 2, 1]])
    with pytest.raises(ValueError, match="SEEDED_TOPOLOGY_IDENTITY_DRIFT"):
        altered.validate()
