from __future__ import annotations

import pytest
import torch

from models.rss_vnext import (
    RSSvNextConfig, RSSvNextPatchEncoderForTests,
    RSSvNextUnifiedCharacterModel,
)
from models.rss_vnext.early_semantic_field import masked_source_semantic_evidence_loss
from models.rss_vnext.source_ray_evidence import sample_front_back_source_rays
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
    assert state.source_semantic_patch_logits.shape == (1, 2, 4, 4)
    assert state.semantic_evidence_logits.shape == (1, 8, 4)
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



def test_original_source_rays_are_queryable_but_cannot_claim_point_visibility():
    img = torch.zeros((1, 2, 4, 8, 8))
    img[:, 0, 0] = .2
    img[:, 1, 1] = .8
    img[:, :, 3] = 1
    cam = torch.zeros((1, 2, 13))
    cam[:, :, 3] = 1  # right X
    cam[:, :, 7] = 1  # up Y
    cam[:, :, 11] = 1  # forward Z
    cam[:, :, 12] = 1  # half-extent
    xyz = torch.tensor([[[-.875, -.875, 0], [3, 0, 0]]])
    source = sample_front_back_source_rays(img, cam, xyz)
    assert source.rgba.shape == (1, 2, 2, 4)
    assert source.projection_valid[0, 0].tolist() == [True, True]
    assert source.projection_valid[0, 1].tolist() == [False, False]
    assert source.rgba[0, 0, 0, 0] == pytest.approx(.2)
    assert source.rgba[0, 0, 1, 1] == pytest.approx(.8)
    assert source.ray_depth_authority == "UNKNOWN_NO_SOURCE_DEPTH"
    assert source.appearance_authority == "ORIGINAL_RGBA_CAMERA_RAY_NOT_POINT_TEXTURE"


def test_mesh_candidate_carries_source_ray_evidence_without_appearance_claim():
    # Reuse a strict component-safe three-vertex sequence from source codec.
    import numpy as np
    from models.tessa.v1.model_v1 import TESSAConfigV1
    from models.tessa.v1.tokenization_v1 import encode_adjacent_teacher_mesh_v1
    net = make()
    rgba, cam = source()
    state, _, _ = net.encode_observations(rgba, cam)
    seq = encode_adjacent_teacher_mesh_v1(
        np.asarray([[-.25, -.25, 0], [.25, -.25, 0], [.25, .25, 0]], dtype=np.float64),
        np.asarray([[0, 1, 2]], dtype=np.int64),
        component_id="toy", chart_id="toy",
        cfg=TESSAConfigV1(coordinate_bins=net.cfg.tessa.coordinate_bins),
    )
    mesh = net.build_mesh_candidate(
        state, torch.tensor(seq.token_ids),
        source_authority="TEACHER_FORCED_TRAINING_ONLY",
    )
    assert mesh.source_ray_evidence is not None
    assert mesh.source_ray_evidence.rgba.shape[1] == mesh.rest_vertices_iris.shape[1]
    assert mesh.source_ray_evidence.source_view_count == 2
    assert not mesh.canonical_mesh_authority
