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
from models.rss_vnext.eight_way_teacher_court_v1 import (
    RSSvNextEightWayTeacherReceiptV1,
    source_teacher_receipt_hash_v1,
    expected_eight_way_tensor_hashes_v1,
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
        novel_view_provenance="",
        source_qualification_key="synthetic-test-only",
        admission_status="QUALIFIED_FOR_THIS_SUPERVISION_FAMILY",
    )
    weights = RSSvNextTrainingWeights(
        topology=0, motion_preference=0, source_semantics=0,
        flexicubes_surface=0, flexicubes_animation=0, novel_view=1.0,
    )
    manifest = sha256(b"synthetic-8-way-test").hexdigest()
    # A synthetic *test-only* registry explicitly simulates the independent
    # corpus administrator. Production must use the verified source manifest.
    hashes = expected_eight_way_tensor_hashes_v1(
        observed_rgba=batch.rgba, source_cameras=batch.camera_features,
        novel_rgba=batch.novel_view_rgba, novel_normal=batch.novel_view_normal,
        rgb_mask=batch.novel_view_rgb_mask,
        normal_mask=batch.novel_view_normal_mask, index=0,
    )
    receipt = RSSvNextEightWayTeacherReceiptV1(
        source_asset_sha256="a" * 64,
        native_scene_sha256="b" * 64,
        native_render_artifact_sha256="c" * 64,
        source_qualification_key=batch.source_qualification_key,
        dataset_manifest_sha256=manifest,
        **hashes,
    )
    receipt = replace(receipt, record_hash=source_teacher_receipt_hash_v1(receipt))
    batch.novel_view_receipts = (receipt,)
    trainer = RSSvNextStructuralTrainer(
        model, torch.optim.AdamW(model.parameters(), lr=1e-3),
        dataset_manifest_sha256=manifest,
        weights=weights,
        trusted_render_registry={receipt.record_hash: receipt.as_record()},
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
    # Crucial: eight-way appearance now trains the actual IRIS Graphics head,
    # instead of only the detached post-scene view renderer.
    assert model.graphics[-1].weight.grad is not None
    assert model.graphics[-1].weight.grad.abs().sum() > 0
    assert model.iris.scene.scene_queries.grad is not None
    assert model.iris.scene.scene_queries.grad.abs().sum() > 0
    assert model.vision.patch.weight.grad is not None
    assert model.vision.patch.weight.grad.abs().sum() > 0
    assert batch.rgba.shape[1] == 2
    assert torch.equal(rgba_before, batch.rgba)


def test_eight_way_rejects_unqualified_partial_or_unconfigured_teacher():
    model, batch, trainer = _fixture()
    batch.novel_view_provenance = "PREDICTED_LATENT_NOT_SOURCE"
    with pytest.raises(ValueError, match="UNTRUSTED_PROVENANCE"):
        trainer.objective(batch)
    batch.novel_view_provenance = ""
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
    with pytest.raises(ValueError, match="PAYLOAD_DRIFT"):
        trainer.objective(batch)
    batch.novel_view_normal_mask.fill_(True)
    batch.novel_view_rgba[:, :, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="TEACHER_TENSOR_NONFINITE"):
        trainer.objective(batch)


def test_five_channel_graphics_contract_and_invalid_field_rejected():
    from dataclasses import replace
    from torch import nn
    with pytest.raises(ValueError, match="GRAPHICS_FIELD_REQUIRES_FIVE"):
        replace(RSSvNextConfig.tiny_for_test(), graphics_channels=4).validate()
    model, batch, _ = _fixture()
    state, _, _ = model.encode_observations(batch.rgba, batch.camera_features)
    model.graphics = nn.Linear(model.cfg.iris.local_feature_dim, 4)
    with pytest.raises(ValueError, match="GRAPHICS_FIELD_INVALID"):
        model.construct_novel_views(
            state, height=3, width=3,
            method="EXPERIMENTAL_POST_SCENE_H_RENDER",
        )


def test_eight_way_requires_independent_registry_and_source_binding():
    _, batch, trainer = _fixture()
    correct = trainer.trusted_render_registry
    trainer.trusted_render_registry = {}
    with pytest.raises(ValueError, match="TRUSTED_REGISTRY_REQUIRED"):
        trainer.objective(batch)
    trainer.trusted_render_registry = correct
    batch.novel_view_rgba[:, :, 0, 0, 0] += 0.1
    with pytest.raises(ValueError, match="PAYLOAD_DRIFT:novel_rgba"):
        trainer.objective(batch)


def test_eight_way_rejects_changed_source_and_receipt():
    _, batch, trainer = _fixture()
    original = batch.rgba.clone()
    batch.rgba[:, 0, :, 0, 0] += .01
    with pytest.raises(ValueError, match="PAYLOAD_DRIFT:front_back_rgba"):
        trainer.objective(batch)
    batch.rgba.copy_(original)
    old = batch.novel_view_receipts[0]
    batch.novel_view_receipts = (replace(old, native_scene_sha256="d" * 64),)
    with pytest.raises(ValueError, match="RECEIPT_HASH_DRIFT"):
        trainer.objective(batch)


def test_eight_way_registry_is_checkpoint_bound(tmp_path):
    model, batch, trainer = _fixture()
    trainer.train_step(batch)
    path = tmp_path / "source-bound.pt"
    trainer.save_checkpoint(path, sampler_cursor=2)
    mirror, _, matched = _fixture()
    assert matched.load_checkpoint(path) == 2
    untrusted = RSSvNextStructuralTrainer(
        mirror, torch.optim.AdamW(mirror.parameters()),
        dataset_manifest_sha256=matched.dataset_manifest_sha256,
        weights=matched.weights,
    )
    with pytest.raises(ValueError, match="RENDER_REGISTRY_DRIFT"):
        untrusted.load_checkpoint(path)
