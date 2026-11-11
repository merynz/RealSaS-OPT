from __future__ import annotations

"""8-way RSS.vNext training plumbing; synthetic tests do NOT claim art fidelity.

Only FRONT/BACK reach the student scene encoder. The six other angles enter
the objective as masked, provenance-gated teacher targets and are never
promoted to user-observed inputs or source appearance authority.
"""
from dataclasses import replace
from hashlib import sha256

import pytest
import torch

from models.rss_vnext import (
    RSSvNextConfig,
    RSSvNextPatchEncoderForTests,
    RSSvNextUnifiedCharacterModel,
)
from models.rss_vnext.training_engine import (
    RSSvNextStructuralTrainingBatch,
    RSSvNextStructuralTrainer,
    RSSvNextTrainingWeights,
)


@pytest.fixture(autouse=True)
def deterministic():
    torch.manual_seed(71)
    torch.set_num_threads(1)


def _camera():
    source = torch.zeros(1, 2, 13)
    source[:, 0, 2] = 2
    source[:, 1, 2] = -2
    source[:, 0, 3] = 1
    source[:, 1, 3] = -1
    source[:, :, 7] = 1
    source[:, 0, 11] = -1
    source[:, 1, 11] = 1
    source[:, :, 12] = 1
    return source


def _fixture():
    cfg = replace(RSSvNextConfig.tiny_for_test(), constructed_total_views=8)
    model = RSSvNextUnifiedCharacterModel(
        cfg, vision_encoder=RSSvNextPatchEncoderForTests(cfg.iris.image_token_dim),
    ).train()
    rgba = torch.rand(1, 2, 4, 8, 8)
    cameras = _camera()
    # Fixed independent synthetic target; never a detached model prediction.
    teacher = torch.zeros(1, 6, 4, 3, 3)
    teacher[:, :, 0] = .25
    teacher[:, :, 1] = .15
    teacher[:, :, 2] = .85
    teacher[:, :, 3] = 1.
    normal = torch.zeros(1, 6, 3, 3, 3)
    normal[:, :, 2] = 1.
    mask = torch.ones(1, 6, 3, 3, dtype=torch.bool)
    batch = RSSvNextStructuralTrainingBatch(
        rgba=rgba,
        camera_features=cameras,
        novel_view_rgba=teacher,
        novel_view_normal=normal,
        novel_view_rgb_mask=mask,
        novel_view_normal_mask=mask.clone(),
        novel_view_provenance="QUALIFIED_SOURCE_RENDER:synthetic-test-only",
        source_qualification_key="synthetic-test-only",
        admission_status="QUALIFIED_FOR_THIS_SUPERVISION_FAMILY",
    )
    weights = RSSvNextTrainingWeights(
        topology=0, motion_preference=0, source_semantics=0,
        flexicubes_surface=0, flexicubes_animation=0, novel_view=1.0,
    )
    trainer = RSSvNextStructuralTrainer(
        model, torch.optim.AdamW(model.parameters(), lr=1e-3),
        dataset_manifest_sha256=sha256(b"synthetic-8-way-test").hexdigest(),
        weights=weights,
    )
    return model, batch, trainer


def test_eight_way_source_render_loss_trains_shared_carrier_without_six_extra_observations():
    model, batch, trainer = _fixture()
    rgba_before = batch.rgba.clone()
    out = trainer.train_step(batch)
    assert out.eligible_novel_view_pixels == 6 * 3 * 3
    assert out.novel_view_loss is not None and torch.isfinite(out.novel_view_loss)
    assert out.topology_loss is None
    assert out.gradient_norm is not None and torch.isfinite(out.gradient_norm)
    assert model.constructed_view_renderer.rgb.weight.grad is not None
    assert model.constructed_view_renderer.rgb.weight.grad.abs().sum() > 0
    assert model.iris.scene.scene_queries.grad is not None
    assert model.iris.scene.scene_queries.grad.abs().sum() > 0
    assert model.vision.patch.weight.grad is not None
    assert model.vision.patch.weight.grad.abs().sum() > 0
    assert batch.rgba.shape[1] == 2
    assert torch.equal(rgba_before, batch.rgba)


def test_eight_way_rejects_unqualified_partial_or_unconfigured_teacher():
    model, batch, trainer = _fixture()
    batch.novel_view_provenance = "PREDICTED_LATENT_NOT_SOURCE"
    with pytest.raises(ValueError, match="SOURCE_RENDER_PROOF_REQUIRED"):
        trainer.objective(batch)
    batch.novel_view_provenance = "QUALIFIED_SOURCE_RENDER:synthetic-test-only"
    batch.novel_view_normal_mask = None
    with pytest.raises(ValueError, match="TEACHER_INCOMPLETE"):
        trainer.objective(batch)
    batch.novel_view_normal_mask = torch.ones_like(batch.novel_view_rgb_mask)
    batch.admission_status = "UNQUALIFIED"
    with pytest.raises(ValueError, match="ADMISSION_NOT_QUALIFIED"):
        trainer.objective(batch)
    batch.admission_status = "QUALIFIED_FOR_THIS_SUPERVISION_FAMILY"
    trainer.weights = replace(trainer.weights, novel_view=0, topology=1)
    with pytest.raises(ValueError, match="OBJECTIVE_NOT_ENABLED"):
        trainer.objective(batch)


def test_eight_way_mask_and_target_fail_closed():
    _, batch, trainer = _fixture()
    batch.novel_view_normal_mask.fill_(False)
    with pytest.raises(ValueError, match="NO_QUALIFIED_TARGET"):
        trainer.objective(batch)
    batch.novel_view_normal_mask.fill_(True)
    batch.novel_view_rgba[:, :, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="TEACHER_NONFINITE"):
        trainer.objective(batch)
