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
    RSSvNextTrainingWeights, source_surface_tensor_sha256,
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


def _qualified_3d_role_case():
    net = make()
    rgba, cam = source()
    state, _, _ = net.encode_observations(rgba, cam)
    target = (torch.arange(state.query_xyz.shape[1]).reshape(1, -1) % net.cfg.semantic_role_schema().count).long()
    mask = torch.zeros_like(target, dtype=torch.bool)
    mask[0, 0] = True
    mask[0, 2] = True
    mask[0, 5] = True
    key = "test-only-source-surface-receipt"
    registry = {key: {
        "admission_status": "QUALIFIED_SOURCE_SURFACE_ROLE",
        "dataset_manifest_sha256": "3" * 64,
        "source_qualification_key": "toy-source",
        "role_schema_id": net.cfg.semantic_role_schema_id,
        "query_xyz_sha256": source_surface_tensor_sha256(state.query_xyz),
        "role_indices_sha256": source_surface_tensor_sha256(target),
        "role_mask_sha256": source_surface_tensor_sha256(mask),
    }}
    batch = RSSvNextStructuralTrainingBatch(
        rgba=rgba, camera_features=cam,
        source_surface_role_indices=target,
        source_surface_role_mask=mask,
        source_surface_role_admission_key=key,
        source_qualification_key="toy-source",
        admission_status="QUALIFIED_FOR_THIS_SUPERVISION_FAMILY",
    )
    return net, registry, batch


def test_source_admitted_3d_presentation_role_head_receives_real_gradient():
    net, registry, batch = _qualified_3d_role_case()
    trainer = RSSvNextStructuralTrainer(
        net, torch.optim.AdamW(net.parameters(), lr=1e-3),
        dataset_manifest_sha256="3" * 64,
        weights=RSSvNextTrainingWeights(
            topology=0, motion_preference=0, source_semantics=0,
            surface_roles=1, flexicubes_surface=0,
        ),
        trusted_surface_role_registry=registry,
    )
    result = trainer.train_step(batch)
    assert result.surface_role_loss is not None
    assert torch.isfinite(result.surface_role_loss)
    assert result.eligible_surface_role_points == 3
    assert result.source_semantic_loss is None
    assert net.presentation_role_head[-1].weight.grad is not None
    assert net.presentation_role_head[-1].weight.grad.abs().sum() > 0
    assert net.iris.scene.scene_queries.grad is not None
    assert net.iris.scene.scene_queries.grad.abs().sum() > 0


def test_3d_surface_semantic_receipt_cannot_be_self_certified_or_reused():
    net, registry, batch = _qualified_3d_role_case()
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3)
    weights = RSSvNextTrainingWeights(
        topology=0, motion_preference=0, source_semantics=0,
        surface_roles=1, flexicubes_surface=0,
    )
    without_registry = RSSvNextStructuralTrainer(
        net, optimizer, dataset_manifest_sha256="3" * 64, weights=weights,
    )
    with pytest.raises(ValueError, match="SURFACE_ROLE_UNTRUSTED_ADMISSION"):
        without_registry.objective(batch)
    trainer = RSSvNextStructuralTrainer(
        net, optimizer, dataset_manifest_sha256="3" * 64,
        weights=weights, trusted_surface_role_registry=registry,
    )
    batch.source_surface_role_indices = batch.source_surface_role_indices.clone()
    batch.source_surface_role_indices[0, 0] = 2
    with pytest.raises(ValueError, match="RECEIPT_SOURCE_OR_ADDRESS_DRIFT"):
        trainer.objective(batch)

    batch.source_surface_role_indices[0, 0] = 0
    batch.source_qualification_key = "different-source"
    with pytest.raises(ValueError, match="RECEIPT_SOURCE_OR_ADDRESS_DRIFT"):
        trainer.objective(batch)


def test_3d_surface_semantic_masked_unknown_not_imputed_and_resume_registry_bound(tmp_path):
    from models.rss_vnext.semantic_geometry_prior import masked_qualified_surface_role_loss
    net, registry, batch = _qualified_3d_role_case()
    logits = torch.zeros((1, 8, 4), requires_grad=True)
    labels = batch.source_surface_role_indices.clone()
    labels[~batch.source_surface_role_mask] = -1
    loss = masked_qualified_surface_role_loss(
        logits, labels, batch.source_surface_role_mask,
        schema=net.cfg.semantic_role_schema(),
    )
    loss.backward()
    assert logits.grad[~batch.source_surface_role_mask].abs().sum() == 0
    assert logits.grad[batch.source_surface_role_mask].abs().sum() > 0
    with pytest.raises(ValueError, match="NO_ADMITTED_POINTS"):
        masked_qualified_surface_role_loss(
            logits, labels, torch.zeros_like(batch.source_surface_role_mask),
            schema=net.cfg.semantic_role_schema(),
        )
    weights = RSSvNextTrainingWeights(
        topology=0, motion_preference=0, source_semantics=0,
        surface_roles=1, flexicubes_surface=0,
    )
    trainer = RSSvNextStructuralTrainer(
        net, torch.optim.AdamW(net.parameters(), lr=1e-3),
        dataset_manifest_sha256="3" * 64,
        weights=weights, trusted_surface_role_registry=registry,
    )
    checkpoint = tmp_path / "surface-role-receipted.pt"
    trainer.save_checkpoint(checkpoint, sampler_cursor=0)
    incompatible = RSSvNextStructuralTrainer(
        net, torch.optim.AdamW(net.parameters(), lr=1e-3),
        dataset_manifest_sha256="3" * 64, weights=weights,
    )
    with pytest.raises(ValueError, match="SURFACE_ROLE_REGISTRY_DRIFT"):
        incompatible.load_checkpoint(checkpoint)


def test_structural_relation_pool_ignores_diagonal_self_edges_and_keeps_gradients():
    net = make()
    rgba, camera = source()
    state, _, _ = net.encode_observations(rgba, camera)
    raw = state.region_relation_logits
    valid = state.region_relation_valid_mask.unsqueeze(-1)
    assert valid.shape == (1, net.cfg.region_slots, net.cfg.region_slots, 1)
    assert not bool(valid[:, torch.arange(net.cfg.region_slots),
                          torch.arange(net.cfg.region_slots)].any())
    masked = raw.masked_fill(~valid, 0.0)
    outgoing = masked.sum(2) / valid.sum(2).clamp_min(1)
    incoming = masked.sum(1) / valid.sum(1).clamp_min(1)
    expected = state.region_slot_features + net.region_graph_to_memory(
        torch.cat((outgoing, incoming), dim=-1)
    )
    assert torch.allclose(state.region_relation_tokens, expected, atol=1e-7)

    # Even extremely large invalid diagonal scores must not alter the
    # input to topology/control (a self-edge cannot be motion evidence).
    augmented = raw + torch.eye(raw.shape[1])[None, :, :, None] * 1e4
    other = augmented.masked_fill(~valid, 0.0)
    changed = state.region_slot_features + net.region_graph_to_memory(
        torch.cat((
            other.sum(2) / valid.sum(2).clamp_min(1),
            other.sum(1) / valid.sum(1).clamp_min(1),
        ), dim=-1)
    )
    assert torch.allclose(changed, expected, atol=1e-7)
    state.region_relation_tokens.square().mean().backward()
    assert net.semantic_structure.pair[-1].weight.grad is not None
    assert net.semantic_structure.pair[-1].weight.grad.abs().sum() > 0


def _qualified_region_pairs(net, rgba, camera):
    from models.rss_vnext.training_engine import source_surface_tensor_sha256
    state, _, _ = net.encode_observations(rgba, camera)
    pairs = torch.tensor([[[0, 1], [2, 3], [4, 5]]], dtype=torch.long)
    targets = torch.tensor([[1., 0., 1.]], dtype=torch.float32)
    mask = torch.tensor([[True, True, False]], dtype=torch.bool)
    receipt = {
        "admission_status": "QUALIFIED_SOURCE_REGION_PAIR",
        "dataset_manifest_sha256": "3" * 64,
        "source_qualification_key": "synthetic-unit-only",
        "supervision_semantics": "SOURCE_REGION_CO_MEMBERSHIP_NOT_DRAWABLE",
        "query_xyz_sha256": source_surface_tensor_sha256(state.query_xyz),
        "pair_indices_sha256": source_surface_tensor_sha256(pairs),
        "pair_targets_sha256": source_surface_tensor_sha256(targets),
        "pair_mask_sha256": source_surface_tensor_sha256(mask),
    }
    return pairs, targets, mask, receipt


def test_source_admitted_anonymous_region_pair_loss_reaches_real_membership_gradient():
    net = make()
    rgba, camera = source()
    pairs, targets, mask, receipt = _qualified_region_pairs(net, rgba, camera)
    trainer = RSSvNextStructuralTrainer(
        net, torch.optim.AdamW(net.parameters(), lr=1e-3),
        dataset_manifest_sha256="3" * 64,
        weights=RSSvNextTrainingWeights(
            topology=0, motion_preference=0, source_semantics=0,
            flexicubes_surface=0, region_pairs=1,
        ),
        trusted_region_pair_registry={"admitted-pair": receipt},
    )
    batch = RSSvNextStructuralTrainingBatch(
        rgba=rgba, camera_features=camera,
        source_region_pair_indices=pairs,
        source_region_pair_is_same=targets,
        source_region_pair_mask=mask,
        source_region_pair_admission_key="admitted-pair",
        source_qualification_key="synthetic-unit-only",
        admission_status="QUALIFIED_FOR_THIS_SUPERVISION_FAMILY",
    )
    result = trainer.train_step(batch)
    assert result.topology_loss is None
    assert result.region_pair_loss is not None
    assert result.eligible_region_pairs == 2
    assert torch.isfinite(result.total_loss)
    assert net.semantic.weight.grad is not None
    assert net.semantic.weight.grad.abs().sum() > 0
    assert net.semantic_structure.point[1].weight.grad is not None
    assert net.semantic_structure.point[1].weight.grad.abs().sum() > 0
    assert trainer.region_pair_registry_sha256


def test_unadmitted_or_modified_source_region_pairs_cannot_train():
    net = make()
    rgba, camera = source()
    pairs, targets, mask, receipt = _qualified_region_pairs(net, rgba, camera)
    trainer = RSSvNextStructuralTrainer(
        net, torch.optim.AdamW(net.parameters(), lr=1e-3),
        dataset_manifest_sha256="3" * 64,
        weights=RSSvNextTrainingWeights(
            topology=0, motion_preference=0, source_semantics=0,
            flexicubes_surface=0, region_pairs=1,
        ),
        trusted_region_pair_registry={"approved": receipt},
    )
    batch = RSSvNextStructuralTrainingBatch(
        rgba=rgba, camera_features=camera,
        source_region_pair_indices=pairs,
        source_region_pair_is_same=targets,
        source_region_pair_mask=mask,
        source_region_pair_admission_key="approved",
        source_qualification_key="synthetic-unit-only",
        admission_status="QUALIFIED_FOR_THIS_SUPERVISION_FAMILY",
    )
    trainer.objective(batch)
    batch.source_region_pair_is_same = 1 - targets
    with pytest.raises(ValueError, match="REGION_PAIR_RECEIPT_SOURCE_OR_ADDRESS_DRIFT"):
        trainer.objective(batch)
    batch.source_region_pair_is_same = targets
    batch.source_region_pair_admission_key = "untrusted"
    with pytest.raises(ValueError, match="REGION_PAIR_UNTRUSTED_ADMISSION"):
        trainer.objective(batch)


def test_region_pair_unknown_is_not_a_positive_label_or_vacuous_negative():
    from models.rss_vnext.semantic_structure import qualified_source_region_pair_loss

    pairs = torch.tensor([[[0,1]]], dtype=torch.long)
    mask = torch.ones((1,1), dtype=torch.bool)
    unknown = torch.tensor([[[-9.,-9.,9.],[-9.,-9.,9.]]], requires_grad=True)
    same = torch.tensor([[1.]])
    different = torch.tensor([[0.]])
    loss_same = qualified_source_region_pair_loss(unknown, pairs, same, mask)
    loss_different = qualified_source_region_pair_loss(unknown, pairs, different, mask)
    assert loss_same > 3.0
    assert loss_different > 3.0
    (loss_same + loss_different).backward()
    assert unknown.grad is not None and torch.isfinite(unknown.grad).all()

    with pytest.raises(ValueError, match="REGION_PAIR_SELF_PAIR_FORBIDDEN"):
        qualified_source_region_pair_loss(
            unknown.detach(), torch.tensor([[[0,0]]]), same, mask,
        )
    with pytest.raises(ValueError, match="REGION_PAIR_NO_QUALIFIED_LABELS"):
        qualified_source_region_pair_loss(
            unknown.detach(), pairs, same, torch.zeros_like(mask),
        )


def test_region_pair_checkpoint_refuses_swapped_teacher_registry(tmp_path):
    net = make()
    rho, camera = source()
    pairs, targets, mask, receipt = _qualified_region_pairs(net, rho, camera)
    weights = RSSvNextTrainingWeights(
        topology=0, motion_preference=0, source_semantics=0,
        flexicubes_surface=0, region_pairs=1,
    )
    trainer = RSSvNextStructuralTrainer(
        net, torch.optim.AdamW(net.parameters()),
        dataset_manifest_sha256="3" * 64,
        weights=weights,
        trusted_region_pair_registry={"qualified": receipt},
    )
    path = tmp_path / "region_source_receipt.pt"
    trainer.save_checkpoint(path, sampler_cursor=4)
    restored_model = make()
    restored = RSSvNextStructuralTrainer(
        restored_model, torch.optim.AdamW(restored_model.parameters()),
        dataset_manifest_sha256="3" * 64,
        weights=weights,
        trusted_region_pair_registry={
            "qualified": {**receipt, "supervision_semantics": "UNTRUSTED_DRAWABLE_LABEL"}
        },
    )
    with pytest.raises(ValueError, match="CHECKPOINT_REGION_PAIR_REGISTRY_DRIFT"):
        restored.load_checkpoint(path)


def test_source_region_pair_builder_preserves_unknown_and_never_invents_negative():
    from models.rss_vnext.semantic_structure import source_region_pairs_from_qualified_labels
    ids = torch.tensor([[5,5,8,-1]], dtype=torch.long)
    mask = torch.tensor([[True,True,True,False]])
    pairs = torch.tensor([[[0,1],[0,2],[0,3],[1,3]]],dtype=torch.long)
    same, qualified = source_region_pairs_from_qualified_labels(ids,mask,pairs)
    assert qualified.tolist() == [[True,True,False,False]]
    assert same.tolist() == [[1.,0.,0.,0.]]
    # No two unnamed unknown points are declared to be a common region.
    with pytest.raises(ValueError,match="SOURCE_REGION_ADMITTED_LABEL_INVALID"):
        source_region_pairs_from_qualified_labels(
            ids,torch.ones_like(mask),pairs,
        )
    with pytest.raises(ValueError,match="SOURCE_REGION_PAIR_ADDRESSES_INVALID"):
        source_region_pairs_from_qualified_labels(
            ids,mask,torch.tensor([[[0,0]]],dtype=torch.long),
        )
