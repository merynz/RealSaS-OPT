from __future__ import annotations

import pytest
import torch

from models.rss_vnext import (
    RSSvNextConfig, RSSvNextPatchEncoderForTests,
    RSSvNextUnifiedCharacterModel,
)
from models.rss_vnext.early_semantic_field import masked_source_semantic_evidence_loss
from models.rss_vnext.training_engine import (
    RSSvNextStructuralTrainer, RSSvNextStructuralTrainingBatch,
    RSSvNextTrainingWeights,
)


@pytest.fixture(autouse=True)
def deterministic():
    torch.manual_seed(42)
    torch.set_num_threads(1)


def make():
    cfg = RSSvNextConfig.tiny_for_test()
    return RSSvNextUnifiedCharacterModel(
        cfg, vision_encoder=RSSvNextPatchEncoderForTests(cfg.iris.image_token_dim),
    )


def source():
    rgba = torch.rand(1, 2, 4, 8, 8)
    camera = torch.zeros((1, 2, 13))
    camera[..., 3] = 1
    camera[..., 7] = 1
    camera[..., 11] = 1
    camera[..., 12] = 1
    return rgba, camera


def test_semantic_evidence_forms_before_shared_scene_and_keeps_original_tokens():
    net = make()
    rgba, camera = source()
    out = net(rgba, camera, topology_prefix=torch.full((1, 4), net.cfg.tessa.BOS))
    state = out.state
    original, _ = net.vision(rgba)
    assert torch.equal(state.observed_image_tokens, original)
    assert state.conditioned_image_tokens.shape == original.shape
    assert not torch.allclose(state.conditioned_image_tokens, original)
    assert state.source_semantic_patch_logits.shape == (1, 2, 4, 5)
    assert state.semantic_evidence_logits.shape == (1, 8, 5)
    assert state.region_relation_tokens.shape == (1, 4, 32)
    assert state.region_relation_logits.shape == (1, 4, 4, 3)
    assert state.region_relation_valid_mask[0].diagonal().sum() == 0

    # A topology objective, with NO segmentation labels, can still train the
    # scene-semantic prior AND the structural relation graph by end-to-end grad.
    out.structural_logits.square().mean().backward()
    assert net.early_semantics.image_evidence[-1].weight.grad is not None
    assert net.early_semantics.image_evidence[-1].weight.grad.abs().sum() > 0
    assert net.semantic_structure.pair[-1].weight.grad is not None
    assert net.semantic_structure.pair[-1].weight.grad.abs().sum() > 0
    assert net.region_graph_to_memory[-1].weight.grad is not None
    assert net.region_graph_to_memory[-1].weight.grad.abs().sum() > 0
    assert net.iris.scene.scene_queries.grad is not None


def test_source_semantic_evidence_is_independent_axes_with_explicit_unknown_masks():
    logits = torch.zeros((1, 2, 3, 5), requires_grad=True)
    target = torch.full_like(logits, float("nan"))
    mask = torch.zeros_like(logits, dtype=torch.bool)
    mask[0, 0, 1, 0] = True
    mask[0, 1, 2, 4] = True
    target[mask] = torch.tensor([1.0, 0.0])
    loss = masked_source_semantic_evidence_loss(logits, target, mask)
    assert float(loss) == pytest.approx(0.693147, rel=1e-4)
    loss.backward()
    assert logits.grad is not None
    assert logits.grad[~mask].abs().sum() == 0
    assert logits.grad[mask].abs().sum() > 0


def test_semantic_only_admitted_batch_trains_early_prior_and_shared_encoder():
    net = make()
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3)
    trainer = RSSvNextStructuralTrainer(
        net, optimizer, dataset_manifest_sha256="3" * 64,
        weights=RSSvNextTrainingWeights(
            topology=0, motion_preference=0, source_semantics=1,
        ),
    )
    img, cam = source()
    # Source-space patch supervision remains 2 views; completely unlabeled
    # material/role axes do not get forced into negative examples.
    target = torch.rand((1, 2, 4, net.cfg.semantic_evidence_channels)).round()
    mask = torch.zeros_like(target, dtype=torch.bool)
    mask[:, 0, :, :2] = True
    mask[:, 1, ::2, 2:3] = True
    batch = RSSvNextStructuralTrainingBatch(
        rgba=img, camera_features=cam,
        semantic_patch_target=target, semantic_patch_mask=mask,
        semantic_qualification_key="qualified-source-render-to-patch-binding",
        source_qualification_key="synthetic-unit-only",
        admission_status="QUALIFIED_FOR_THIS_SUPERVISION_FAMILY",
    )
    result = trainer.train_step(batch)
    assert result.source_semantic_loss is not None
    assert result.topology_loss is None
    assert result.eligible_semantic_patch_labels == int(mask.sum())
    assert trainer.global_step == 1
    assert net.early_semantics.image_evidence[-1].weight.grad is not None
    assert net.vision.patch.weight.grad is not None
    # No source supervision in the unknown axes should be manufactured.
    batch.semantic_qualification_key = ""
    with pytest.raises(ValueError, match="SEMANTIC_QUALIFICATION_REQUIRED"):
        trainer.objective(batch)


def test_semantic_targets_can_never_be_injected_as_a_third_observation():
    net = make()
    img, cam = source()
    with pytest.raises(ValueError, match="TWO_SOURCE_RGBA"):
        net(torch.cat((img, img[:, 1:2]), dim=1), cam)
