from __future__ import annotations

"""RSS.vNext typed Semantic/Graphics/Mechanics relation supervision.

The synthetic *source* court here exercises trust and gradients only.
Actual PartVerse part tags never self-certify drawable identity or contact.
Only the externally admitted teacher channel may train the corresponding
model-owned directed or symmetric relation output.
"""

from dataclasses import replace as dc_replace

import pytest
import torch

from models.rss_vnext import (
    RSSvNextConfig, RSSvNextPatchEncoderForTests,
    RSSvNextUnifiedCharacterModel,
)
from models.rss_vnext.semantic_structure import (
    TYPED_REGION_RELATION_CHANNELS_V1,
    qualified_source_typed_relation_loss,
)
from models.rss_vnext.training_engine import (
    RSSvNextStructuralTrainingBatch, RSSvNextStructuralTrainer,
    RSSvNextTrainingWeights, source_surface_tensor_sha256,
)


@pytest.fixture(autouse=True)
def _deterministic():
    torch.manual_seed(171)
    torch.set_num_threads(1)


def _simple_source():
    rgba = torch.rand(1, 2, 4, 8, 8)
    cam = torch.zeros(1, 2, 13)
    cam[:, :, 3] = cam[:, :, 7] = cam[:, :, 11] = cam[:, :, 12] = 1.
    return rgba, cam


def _model():
    cfg = RSSvNextConfig.tiny_for_test()
    return RSSvNextUnifiedCharacterModel(
        cfg, vision_encoder=RSSvNextPatchEncoderForTests(
            cfg.iris.image_token_dim,
        ),
    )


def _admitted_cut_only_teacher(model, rgba, cam):
    with torch.no_grad():
        state, _, _ = model.encode_observations(rgba, cam)
    indices = torch.tensor([[[0, 1], [2, 3], [4, 5]]], dtype=torch.long)
    target = torch.full((1, 3, 3), float("nan"))
    mask = torch.zeros_like(target, dtype=torch.bool)
    target[0, :, 1] = torch.tensor([1., 0., 1.])
    mask[0, :, 1] = True
    authorities = {
        "same_drawable_affinity": "UNAVAILABLE",
        "cut_affinity": "QUALIFIED_SOURCE_GEOMETRIC_SEPARABILITY",
        "directed_attachment_affinity": "UNAVAILABLE",
    }
    receipt = {
        "admission_status": "QUALIFIED_SOURCE_TYPED_REGION_RELATIONS",
        "dataset_manifest_sha256": "3" * 64,
        "source_qualification_key": "SYNTHETIC_SOURCE_COURT_NOT_DATASET_RIGHTS",
        "query_xyz_sha256": source_surface_tensor_sha256(state.query_xyz),
        "pair_indices_sha256": source_surface_tensor_sha256(indices),
        "typed_targets_sha256": source_surface_tensor_sha256(target),
        "typed_mask_sha256": source_surface_tensor_sha256(mask),
        "original_rgba_sha256": source_surface_tensor_sha256(rgba),
        "original_camera_sha256": source_surface_tensor_sha256(cam),
        "supervision_semantics": (
            "SOURCE_QUERY_PAIR_TYPED_DRAWABLE_CUT_DIRECTED_ATTACHMENT_MASKED_V1"
        ),
        "channel_source_authority": authorities,
    }
    batch = RSSvNextStructuralTrainingBatch(
        rgba=rgba, camera_features=cam,
        source_qualification_key=receipt["source_qualification_key"],
        admission_status="QUALIFIED_FOR_THIS_SUPERVISION_FAMILY",
        source_typed_relation_pair_indices=indices,
        source_typed_relation_target=target,
        source_typed_relation_mask=mask,
        source_typed_relation_admission_key="synthetic-cut-provenance",
    )
    return receipt, batch


def test_permutation_invariant_directed_attachment_pair_loss():
    # Two anonymous slots, two source query points. Attachment is directional
    # even when same-drawable/cut are symmetric: no hardcoded human part names.
    member = torch.tensor([[[8., -8., -10.], [-8., 8., -10.]]])
    relation = torch.zeros((1, 2, 2, 3), requires_grad=True)
    with torch.no_grad():
        relation[0, 0, 1, 2] = 4.
        relation[0, 1, 0, 2] = -4.
    valid = ~torch.eye(2, dtype=torch.bool)[None]
    pairs = torch.tensor([[[0, 1], [1, 0]]], dtype=torch.long)
    targets = torch.full((1, 2, 3), float("nan"))
    mask = torch.zeros((1, 2, 3), dtype=torch.bool)
    targets[0, 0, 2] = 1.
    targets[0, 1, 2] = 0.
    mask[..., 2] = True
    loss, counts = qualified_source_typed_relation_loss(
        member, relation, valid, pairs, targets, mask,
    )
    assert counts == (0, 0, 2)
    assert float(loss) < .05
    order = torch.tensor([1, 0])
    permuted_member = member[:, :, torch.tensor([1, 0, 2])]
    permuted_relation = relation[:, order][:, :, order]
    permuted_mask = valid[:, order][:, :, order]
    permuted, _ = qualified_source_typed_relation_loss(
        permuted_member, permuted_relation, permuted_mask,
        pairs, targets, mask,
    )
    assert torch.allclose(loss, permuted, atol=1.e-6, rtol=0)
    assert TYPED_REGION_RELATION_CHANNELS_V1 == (
        "same_drawable_affinity", "cut_affinity", "directed_attachment_affinity",
    )
    loss.backward()
    assert relation.grad is not None
    assert torch.count_nonzero(relation.grad[..., :2]) == 0
    assert relation.grad[..., 2].abs().sum() > 0


def test_cut_supervision_only_trains_cut_not_drawable_or_attachment():
    model = _model()
    rgba, cam = _simple_source()
    receipt, batch = _admitted_cut_only_teacher(model, rgba, cam)
    trainer = RSSvNextStructuralTrainer(
        model, torch.optim.AdamW(model.parameters(), lr=1.e-3),
        dataset_manifest_sha256="3" * 64,
        weights=RSSvNextTrainingWeights(
            topology=0, motion_preference=0, source_semantics=0,
            typed_relations=1, flexicubes_surface=0,
        ),
        trusted_typed_relation_registry={"synthetic-cut-provenance": receipt},
    )
    result = trainer.train_step(batch)
    assert result.typed_relation_loss is not None
    assert torch.isfinite(result.total_loss)
    assert result.eligible_typed_relation_channels == (0, 3, 0)
    assert result.region_pair_loss is None
    assert result.axis_rig_skin_loss is None
    # Projection head has three separate output rows; a cut teacher cannot
    # manufacture gradients on independently unknown attachment/drawable rows.
    final = model.semantic_structure.pair[-1].weight.grad
    assert final is not None
    assert final[1].abs().sum() > 0
    assert torch.count_nonzero(final[0]) == 0
    assert torch.count_nonzero(final[2]) == 0
    assert model.vision.patch.weight.grad is not None


def test_teacher_registry_rejects_rehashed_labels_missing_masks_and_spoofed_attachment():
    model = _model()
    rgba, cam = _simple_source()
    receipt, batch = _admitted_cut_only_teacher(model, rgba, cam)
    weights = RSSvNextTrainingWeights(
        topology=0, motion_preference=0, source_semantics=0,
        typed_relations=1, flexicubes_surface=0,
    )
    trainer = RSSvNextStructuralTrainer(
        model, torch.optim.AdamW(model.parameters()),
        dataset_manifest_sha256="3" * 64, weights=weights,
        trusted_typed_relation_registry={"synthetic-cut-provenance": receipt},
    )
    bad = batch.source_typed_relation_target.clone()
    bad[0, 0, 1] = 0.
    with pytest.raises(ValueError, match="RECEIPT_SOURCE_OR_ADDRESS_DRIFT"):
        trainer.objective(dc_replace(batch, source_typed_relation_target=bad))
    with pytest.raises(ValueError, match="TEACHER_INCOMPLETE"):
        trainer.objective(dc_replace(batch, source_typed_relation_mask=None))
    bad_mask = batch.source_typed_relation_mask.clone()
    bad_mask[0, 0, 2] = True
    with pytest.raises(ValueError, match="RECEIPT_SOURCE_OR_ADDRESS_DRIFT"):
        trainer.objective(dc_replace(batch, source_typed_relation_mask=bad_mask))
    pretend = dict(receipt)
    pretend["channel_source_authority"] = {
        **receipt["channel_source_authority"],
        "cut_affinity": "QUALIFIED_SOURCE_CONTACT_AND_MOTION_INTERVENTION",
    }
    untrusted = RSSvNextStructuralTrainer(
        model, torch.optim.AdamW(model.parameters()),
        dataset_manifest_sha256="3" * 64, weights=weights,
        trusted_typed_relation_registry={"synthetic-cut-provenance": pretend},
    )
    with pytest.raises(ValueError, match="CHANNEL_AUTHORITY_INVALID"):
        untrusted.objective(batch)
    # A segmentation-only source is never a valid attachment teacher.
    pretend_attach = dict(receipt)
    pretend_attach["channel_source_authority"] = {
        **receipt["channel_source_authority"],
        "directed_attachment_affinity": "SOURCE_PART_LABEL_ONLY",
    }
    untrusted_attach = RSSvNextStructuralTrainer(
        model, torch.optim.AdamW(model.parameters()),
        dataset_manifest_sha256="3" * 64, weights=weights,
        trusted_typed_relation_registry={"synthetic-cut-provenance": pretend_attach},
    )
    with pytest.raises(ValueError, match="CHANNEL_AUTHORITY_INVALID"):
        untrusted_attach.objective(batch)


def test_typed_relation_trainer_resume_binds_teacher_registry_and_objective(tmp_path):
    model = _model()
    rgba, cam = _simple_source()
    receipt, batch = _admitted_cut_only_teacher(model, rgba, cam)
    weights = RSSvNextTrainingWeights(
        topology=0, motion_preference=0, source_semantics=0,
        typed_relations=1, flexicubes_surface=0,
    )
    registry = {"synthetic-cut-provenance": receipt}
    trainer = RSSvNextStructuralTrainer(
        model, torch.optim.AdamW(model.parameters()),
        dataset_manifest_sha256="3" * 64, weights=weights,
        trusted_typed_relation_registry=registry,
    )
    path = tmp_path / "typed-relation-trainer.pt"
    trainer.save_checkpoint(path, sampler_cursor=5)
    second = RSSvNextStructuralTrainer(
        model, torch.optim.AdamW(model.parameters()),
        dataset_manifest_sha256="3" * 64, weights=weights,
        trusted_typed_relation_registry=registry,
    )
    assert second.load_checkpoint(path) == 5
    changed = dict(receipt)
    changed["channel_source_authority"] = {
        **receipt["channel_source_authority"],
        "cut_affinity": "UNAVAILABLE",
    }
    changed_trainer = RSSvNextStructuralTrainer(
        model, torch.optim.AdamW(model.parameters()),
        dataset_manifest_sha256="3" * 64, weights=weights,
        trusted_typed_relation_registry={"synthetic-cut-provenance": changed},
    )
    with pytest.raises(ValueError, match="CHECKPOINT_TYPED_RELATION_REGISTRY_DRIFT"):
        changed_trainer.load_checkpoint(path)
