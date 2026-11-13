from __future__ import annotations

"""Actual pinned NVIDIA FC surface with source animation teacher, CPU court."""

import os
from pathlib import Path

import pytest
import torch
from torch import nn

from models.iris.vnext.flexicubes_v1 import PinnedFlexiCubesV1
from models.rss_vnext.flexicubes_surface_evidence import (
    RSSvNextFlexiCubesEvidenceBridge,
)
from models.rss_vnext.flexicubes_animation_supervision import (
    RSSvNextFCAnimationTeacher,
    RSSvNextFCAnimationLossConfig,
    differentiable_source_skin_transfer,
    source_animation_flexicubes_loss,
)
from models.rss_vnext.animation_ready_mesh import (
    posed_lbs_from_joint_matrices,
    _area_weighted_surface_witnesses,
)
from models.rss_vnext.semantic_geometry_prior import (
    RSSvNextSemanticLayerSpec,
    realsas_style_generic_schema,
)
from models.rss_vnext import (
    RSSvNextConfig, RSSvNextPatchEncoderForTests,
    RSSvNextUnifiedCharacterModel,
)
from models.rss_vnext.training_engine import (
    RSSvNextStructuralTrainer,
    RSSvNextStructuralTrainingBatch,
    RSSvNextTrainingWeights,
)
from models.tessa.v1.model_v1 import TESSAConfigV1
from models.tessa.v1.tokenization_v1 import encode_adjacent_teacher_mesh_v1
import numpy as np


@pytest.fixture(autouse=True)
def deterministic():
    torch.manual_seed(37)
    torch.set_num_threads(1)


def _pinned():
    source_root = os.getenv("REALSAS_FLEXICUBES_ROOT")
    if not source_root:
        pytest.skip("Only runs against checked-out pinned NVIDIA source")
    return PinnedFlexiCubesV1(Path(source_root), device="cpu")


class AnalyticCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(0.03))
        self.feature_gain = nn.Parameter(torch.tensor(1.0))
        self.cube_extraction = nn.Sequential(
            nn.Flatten(-2), nn.Linear(8 * 8, 21),
        )

    def query(self, scene_planes, xyz):
        offset = scene_planes.mean() * .04
        return {
            "field": xyz.norm(dim=-1) - 0.65 + self.bias + offset,
            "H": torch.cat((xyz, xyz.square(), xyz[..., :2]), dim=-1)
                 * self.feature_gain + offset,
            "extraction_delta": torch.zeros_like(xyz),
        }


def _source_from_extracted(surface):
    """Synthetic only: the teacher is a fixed copy of an animated reference.

    It is formed AFTER the student forward but detached and cannot affect
    student source-image encoder. A real dataset needs native qualified
    independent source mesh/rig/poses, not this self-referential fixture.
    """
    source_rest = surface.xyz.detach()[0].clone()
    n = source_rest.shape[0]
    assert n >= 10
    # Smooth source influence transition along x; no human joint prior.
    influence = torch.sigmoid(8.0 * source_rest[:, :1])
    source_weights = torch.cat((influence, 1.0 - influence), dim=-1)
    source_matrices = torch.eye(4)[None, None].expand(2, 2, 4, 4).clone()
    source_matrices[1, 0, 0, 3] = .30
    source_matrices[1, 1, 2, 3] = -.30
    source_poses = posed_lbs_from_joint_matrices(
        source_rest[None], source_weights[None], source_matrices[None],
    )[0]
    posed_samples = torch.stack([
        _area_weighted_surface_witnesses(source_poses[p][None], surface.faces).detach()
        for p in range(2)
    ])
    return RSSvNextFCAnimationTeacher(
        source_asset_sha256="synthetic-fake-asset-unit-only",
        source_rig_binding_sha256="synthetic-fake-rig-unit-only",
        source_pose_suite_sha256="synthetic-fake-pose-suite-unit-only",
        source_admission_witness_id="synthetic-test-not-admission",
        source_component_domain="SINGLE_QUALIFIED_CLOSED_COMPONENT",
        rest_vertices=source_rest,
        rest_skin_weights=source_weights,
        posed_skin_matrices=source_matrices,
        rest_surface_samples=_area_weighted_surface_witnesses(
            source_rest[None], surface.faces,
        ).detach(),
        pose_surface_samples=posed_samples,
        pose_valid_mask=torch.tensor([True, True]),
    )


def _real_fc_surface():
    core = AnalyticCore()
    role = nn.Linear(8, 4)
    planes = torch.randn(1, 3, 2, 2, 2, requires_grad=True)
    bridge = RSSvNextFlexiCubesEvidenceBridge(core, resolution=5)
    surface = bridge(
        planes, extractor=_pinned(), role_head=role,
        schema=realsas_style_generic_schema(),
        layer=RSSvNextSemanticLayerSpec(
            "all", realsas_style_generic_schema().roles,
        ),
        field_policy="SIGNED_HYPOTHESIS_ONLY",
    )
    return surface, core, planes


def test_knn_skin_transfer_is_differentiable_between_neighbor_switches():
    source = torch.tensor([
        [-1., 0., 0.], [0., 0., 0.], [1., 0., 0.],
    ])
    weights = torch.tensor([
        [1., 0.], [.5, .5], [0., 1.],
    ])
    points = torch.tensor([[
        [-.38, .1, 0.], [.19, .13, 0.], [.66, -.2, 0.],
    ]], requires_grad=True)
    estimated = differentiable_source_skin_transfer(
        points, source, weights, k=3,
    )
    assert estimated.shape == (1, 3, 2)
    assert torch.allclose(
        estimated.sum(-1), torch.ones(1, 3), atol=1e-6,
    )
    estimated[0, 0, 0].backward()
    assert points.grad is not None
    assert torch.isfinite(points.grad).all()
    assert points.grad.abs().sum() > 1e-7


def test_actual_pinned_fc_animation_poses_backprop_to_geometry_and_cube():
    surface, core, source_planes = _real_fc_surface()
    teacher = _source_from_extracted(surface)
    objective = source_animation_flexicubes_loss(
        surface, teacher,
        config=RSSvNextFCAnimationLossConfig(knn=10, rest_weight=0.1),
    )
    assert objective.pose_count == 2
    assert objective.posed_vertices.shape == (
        1, 2, surface.xyz.shape[1], 3,
    )
    assert objective.transferred_skin.shape == (1, surface.xyz.shape[1], 2)
    assert float(objective.posed_surface.detach()) > 0
    objective.total.backward()
    for gradient in (
        core.bias.grad,
        core.cube_extraction[-1].weight.grad,
        source_planes.grad,
    ):
        assert gradient is not None and torch.isfinite(gradient).all()
        assert gradient.abs().sum() > 1e-9


def test_source_component_and_pose_qualification_fail_closed():
    surface, _, _ = _real_fc_surface()
    teacher = _source_from_extracted(surface)
    from dataclasses import replace
    with pytest.raises(ValueError, match="COMPONENT_TRANSFER_UNSAFE"):
        source_animation_flexicubes_loss(
            surface, replace(teacher, source_component_domain="MODULAR_OPEN"),
        )
    with pytest.raises(ValueError, match="NO_QUALIFIED_POSE"):
        source_animation_flexicubes_loss(
            surface, replace(teacher, pose_valid_mask=torch.tensor([False, False])),
        )
    with pytest.raises(ValueError, match="SOURCE_WEIGHT_SIMPLEX"):
        source_animation_flexicubes_loss(
            surface, replace(teacher, rest_skin_weights=teacher.rest_skin_weights * .5),
        )


def test_one_optimizer_step_combines_fc_animation_and_tessa_structure():
    """NVIDIA teacher pose never enters model observation; TESSA gets FC H(x)."""
    cfg = RSSvNextConfig.tiny_for_test()
    net = RSSvNextUnifiedCharacterModel(
        cfg, vision_encoder=RSSvNextPatchEncoderForTests(cfg.iris.image_token_dim),
    ).train()
    real_query = net.iris.query

    def analytic_field_with_student_dependencies(scene, xyz):
        result = real_query(scene, xyz)
        result["field"] = xyz.norm(dim=-1) - .65 + .03 * result["field"]
        return result

    net.iris.query = analytic_field_with_student_dependencies
    rgba = torch.rand(1, 2, 4, 8, 8)
    camera = torch.zeros(1, 2, 13)
    camera[..., 3] = camera[..., 7] = camera[..., 11] = camera[..., 12] = 1

    layer = RSSvNextSemanticLayerSpec("all", cfg.semantic_role_schema().roles)
    # Construct training fixture from a prior student pass. All teacher values
    # are detached; the model then gets only two RGBA views as observations.
    state, _, _ = net.encode_observations(rgba, camera)
    surface = net.fc_surface_bridge(
        state.scene_triplanes,
        extractor=_pinned(),
        role_head=net.presentation_role_head,
        schema=cfg.semantic_role_schema(),
        layer=layer, field_policy="SIGNED_HYPOTHESIS_ONLY",
    )
    teacher = _source_from_extracted(surface)
    triangle = encode_adjacent_teacher_mesh_v1(
        np.array([[-.25, -.25, 0], [.25, -.25, 0], [.25, .25, 0]],
                 dtype=np.float64),
        np.array([[0, 1, 2]], dtype=np.int64),
        component_id="synthetic", chart_id="synthetic",
        cfg=TESSAConfigV1(coordinate_bins=cfg.tessa.coordinate_bins),
    )
    tokens = torch.tensor(triangle.token_ids, dtype=torch.long)[None]
    # Synthetic teacher identity stands in for an independent signed-source
    # admission producer; the unit test does not qualify real character truth.
    signed_receipt = {
        "admission_status": "QUALIFIED_SOURCE_SIGNED_SOLID",
        "source_component_domain": "SINGLE_QUALIFIED_CLOSED_COMPONENT",
        "dataset_manifest_sha256": "f"*64,
        "source_qualification_key": "synthetic-only",
        "role_schema_id": cfg.semantic_role_schema_id,
        "layer_name": layer.name,
        "surface_target_sha256": None,
        "animation_source_asset_sha256": teacher.source_asset_sha256,
        "animation_source_rig_sha256": teacher.source_rig_binding_sha256,
        "animation_source_poses_sha256": teacher.source_pose_suite_sha256,
    }
    trainer = RSSvNextStructuralTrainer(
        net, torch.optim.AdamW(net.parameters(), lr=1e-4),
        dataset_manifest_sha256="f" * 64,
        trusted_signed_source_registry={"synthetic-only": signed_receipt},
        weights=RSSvNextTrainingWeights(
            topology=1., motion_preference=0., source_semantics=0.,
            flexicubes_surface=0., flexicubes_animation=1.,
        ),
    )
    batch = RSSvNextStructuralTrainingBatch(
        rgba=rgba, camera_features=camera,
        topology_teacher_tokens=tokens,
        topology_valid_mask=torch.tensor([True]),
        flexicubes_extractor=_pinned(),
        flexicubes_layer=layer,
        flexicubes_animation_teacher=teacher,
        flexicubes_signed_source_admission_key="synthetic-only",
        source_qualification_key="synthetic-only",
        admission_status="QUALIFIED_FOR_THIS_SUPERVISION_FAMILY",
    )
    result = trainer.train_step(batch)
    assert result.flexicubes_animation_loss is not None
    assert result.topology_loss is not None
    assert result.eligible_animation_pose_count == 2
    assert trainer.global_step == 1
    assert net.iris.field.field_head.weight.grad is not None
    assert net.iris.cube_extraction.net[-1].weight.grad is not None
    assert net.iris.cube_extraction.net[-1].weight.grad.abs().sum() > 0
    assert net.tessa.lm_head.weight.grad is not None
    assert net.tessa.lm_head.weight.grad.abs().sum() > 0
    assert net.vision.patch.weight.grad is not None

    batch.flexicubes_signed_source_admission_key = ""
    with pytest.raises(ValueError, match="SIGNED_SOURCE_PROOF_REQUIRED"):
        trainer.objective(batch)
    batch.flexicubes_signed_source_admission_key = "other-source"
    with pytest.raises(ValueError, match="FC_SOURCE_SIGNED_DOMAIN_RECEIPT_DRIFT"):
        trainer.objective(batch)
