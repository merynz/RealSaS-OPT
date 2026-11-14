from __future__ import annotations

from hashlib import sha256

import numpy as np
import pytest
import torch

from models.rss_vnext import (
    RSSvNextConfig, RSSvNextPatchEncoderForTests, RSSvNextUnifiedCharacterModel,
)
from models.rss_vnext.realsas_virtual_camera_policy_v1 import (
    target_canonical_and_six_cameras_v1,
)
from models.rss_vnext.realsas_two_source_construction_v1 import (
    paired_generation_binding_sha256_v1,
)
from models.rss_vnext.training_engine import (
    RSSvNextStructuralTrainingBatch, RSSvNextStructuralTrainer,
    RSSvNextTrainingWeights, source_surface_tensor_sha256,
)
from models.tessa.v1.model_v1 import TESSAConfigV1
from models.tessa.v1.tokenization_v1 import encode_adjacent_teacher_mesh_v1


@pytest.fixture(autouse=True)
def _set_seed():
    torch.manual_seed(21)
    torch.set_num_threads(1)


DATASET = sha256(b"synthetic-product-compatibility-test").hexdigest()


def _fixture():
    cfg = RSSvNextConfig.tiny_for_test()
    net = RSSvNextUnifiedCharacterModel(
        cfg, vision_encoder=RSSvNextPatchEncoderForTests(cfg.iris.image_token_dim),
    )
    source = torch.rand(1, 2, 4, 16, 16)
    calibrated_teacher_camera = torch.zeros((1, 2, 13))
    for index in (3, 7, 11, 12):
        calibrated_teacher_camera[:, :, index] = 1.
    _, virtual = target_canonical_and_six_cameras_v1(batch_size=1)
    generated = torch.rand((1, 6, 4, 16, 16))
    normals = torch.rand((1, 6, 3, 16, 16))*2.-1.
    xyz = np.array([[-.25, -.25, 0.], [.25, -.25, 0.], [.25, .25, 0.]], dtype=np.float64)
    topo = encode_adjacent_teacher_mesh_v1(
        xyz, np.array([[0, 1, 2]], dtype=np.int64),
        component_id="source", chart_id="source",
        cfg=TESSAConfigV1(coordinate_bins=cfg.tessa.coordinate_bins),
    ).token_ids
    batch = RSSvNextStructuralTrainingBatch(
        rgba=source, camera_features=calibrated_teacher_camera,
        topology_teacher_tokens=torch.as_tensor(topo, dtype=torch.long)[None],
        topology_valid_mask=torch.tensor([True]),
        source_qualification_key="synthetic-unit-no-production-source-claim",
        admission_status="QUALIFIED_FOR_THIS_SUPERVISION_FAMILY",
        product_generated_rgba=generated,
        product_generated_normals=normals,
        product_generated_cameras=virtual,
        product_source_binding_sha256="a"*64,
        product_canonical_checkpoint_sha256="b"*64,
        product_multiview_checkpoint_sha256="c"*64,
        product_generation_checkpoint_sha256=paired_generation_binding_sha256_v1(
            source_binding_sha256="a"*64,
            canonical_checkpoint_sha256="b"*64,
            multiview_checkpoint_sha256="c"*64,
        ),
        product_conditioning_admission_key="synthetic-pinned-generator-receipt",
    )
    record = {
        "admission_status": "PINNED_GENERATOR_OUTPUT_NOT_SOURCE_TEACHER",
        "dataset_manifest_sha256": DATASET,
        "source_qualification_key": batch.source_qualification_key,
        "source_observed_rgba_sha256": source_surface_tensor_sha256(source),
        "original_camera_is_model_input": False,
        "generated_rgba_sha256": source_surface_tensor_sha256(generated),
        "generated_normals_sha256": source_surface_tensor_sha256(normals),
        "generated_cameras_sha256": source_surface_tensor_sha256(virtual),
        "source_binding_sha256": batch.product_source_binding_sha256,
        "canonical_stage_checkpoint_sha256": batch.product_canonical_checkpoint_sha256,
        "multiview_stage_checkpoint_sha256": batch.product_multiview_checkpoint_sha256,
        "combined_generation_checkpoint_sha256": batch.product_generation_checkpoint_sha256,
        "generated_views_are_observed_source": False,
    }
    return net, batch, record


def _trainer(net, registry, policy="PRODUCT_GENERATED_SIX_VIEW"):
    return RSSvNextStructuralTrainer(
        net, torch.optim.AdamW(net.parameters(), lr=1.e-3),
        dataset_manifest_sha256=DATASET,
        scene_conditioning_policy=policy,
        trusted_generated_conditioning_registry=registry,
        weights=RSSvNextTrainingWeights(
            topology=1., motion_preference=0., source_semantics=0.,
            flexicubes_surface=0.,
        ),
    )


def test_product_trainer_constructs_same_six_view_H_without_source_camera(monkeypatch):
    net, batch, record = _fixture()
    trainer = _trainer(net, {batch.product_conditioning_admission_key: record})
    observed = {}
    original = net.encode_generated_multiview

    def observe(observed_rgba, original_camera_features, **kwargs):
        observed["camera"] = original_camera_features
        observed["views"] = kwargs["generated_rgba"].shape[1]
        state = original(observed_rgba, original_camera_features, **kwargs)
        observed["authority"] = state.scene_conditioning_authority
        observed["state_camera"] = state.camera_features
        return state

    monkeypatch.setattr(net, "encode_generated_multiview", observe)
    outcome = trainer.train_step(batch)
    assert outcome.topology_loss is not None and torch.isfinite(outcome.total_loss)
    assert trainer.global_step == 1
    assert observed == {
        "camera": None, "views": 6, "authority": "GENERATED_SIX_VIEW_SHARED_SCENE",
        "state_camera": None,
    }


def test_product_trainer_rejects_unbound_or_tampered_generated_inputs():
    net, batch, record = _fixture()
    trainer = _trainer(net, {batch.product_conditioning_admission_key: record})
    batch.product_generated_rgba = batch.product_generated_rgba.clone()
    batch.product_generated_rgba[0, 0, 0, 0, 0] += .01
    with pytest.raises(ValueError, match="GENERATED_SCENE_RECEIPT_DRIFT"):
        trainer.objective(batch)
    batch.product_generated_rgba = None
    with pytest.raises(ValueError, match="PRODUCT_GENERATED_SCENE_REQUIRED"):
        trainer.objective(batch)


def test_product_trainer_rejects_untrusted_virtual_camera_or_wrong_policy():
    net, batch, record = _fixture()
    trainer = _trainer(net, {batch.product_conditioning_admission_key: record})
    batch.product_generated_cameras = batch.product_generated_cameras.clone()
    batch.product_generated_cameras[:, 1, :3] += .1
    with pytest.raises(ValueError, match="GENERATOR_CAMERA_CONTRACT_DRIFT"):
        trainer.objective(batch)
    batch.product_generated_cameras = None
    research = _trainer(net, {}, policy="RESEARCH_CALIBRATED_TWO_VIEW")
    with pytest.raises(ValueError, match="GENERATED_SCENE_UNEXPECTED"):
        research.objective(batch)


def test_product_generator_registry_and_policy_are_checkpoint_identity(tmp_path):
    net, batch, record = _fixture()
    first = _trainer(net, {batch.product_conditioning_admission_key: record})
    checkpoint = tmp_path/"product-six-view.pt"
    first.save_checkpoint(checkpoint, sampler_cursor=17)
    clone, _, _ = _fixture()
    restored = _trainer(clone, {batch.product_conditioning_admission_key: record})
    assert restored.load_checkpoint(checkpoint) == 17
    altered = _trainer(clone, {batch.product_conditioning_admission_key: {
        **record, "generated_views_are_observed_source": True,
    }})
    with pytest.raises(ValueError, match="GENERATOR_REGISTRY_DRIFT"):
        altered.load_checkpoint(checkpoint)
    research = _trainer(clone, {}, policy="RESEARCH_CALIBRATED_TWO_VIEW")
    with pytest.raises(ValueError, match="OBJECTIVE_POLICY_DRIFT"):
        research.load_checkpoint(checkpoint)



def test_product_trainer_recomputes_generator_binding_not_just_trusts_receipt():
    net, batch, record = _fixture()
    # A malicious batch and even a compromised registry agreeing on a fake
    # binding must be rejected by the same immutable composition as inference.
    batch.product_generation_checkpoint_sha256 = "d"*64
    trainer = _trainer(net, {batch.product_conditioning_admission_key: {
        **record, "combined_generation_checkpoint_sha256": "d"*64,
    }})
    with pytest.raises(ValueError, match="GENERATOR_BINDING_RECOMPUTE_DRIFT"):
        trainer.objective(batch)
