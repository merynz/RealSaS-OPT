from __future__ import annotations

from collections import OrderedDict
import pytest
import torch

from models.rss_vnext.training_optimizer_ownership_v1 import (
    qualify_rss_vnext_optimizer_ownership_v1,
)
from models.rss_vnext.training_engine import (
    RSSvNextStructuralTrainer, RSSvNextTrainingWeights,
)
from tests.models.test_rss_vnext_structural_trainer import (
    make_model, make_sample, manifest_hash,
)


@pytest.fixture(autouse=True)
def deterministic():
    torch.manual_seed(105)
    torch.set_num_threads(1)


def test_active_product_optimizer_binds_all_trainable_named_model_tensors():
    net=make_model()
    optimizer=torch.optim.AdamW(net.parameters(),lr=1e-3)
    owner=qualify_rss_vnext_optimizer_ownership_v1(net,optimizer)
    assert owner.trainable_model_parameter_tensors>10
    assert owner.optimizer_registered_parameter_tensors>=owner.trainable_model_parameter_tensors
    assert len(owner.binding_sha256)==64
    assert owner.schema.endswith("TrainingOptimizerOwnership.v1")


def test_subset_optimizer_without_explicit_freeze_fails_at_trainer_creation():
    net=make_model()
    params=list(net.parameters())
    assert all(p.requires_grad for p in params[:2])
    incomplete=torch.optim.AdamW(params[1:],lr=1e-3)
    with pytest.raises(ValueError,match="UNOWNED_TRAINABLE_PARAMETER"):
        RSSvNextStructuralTrainer(
            net,incomplete,dataset_manifest_sha256=manifest_hash(),
        )
    foreign=make_model()
    with pytest.raises(ValueError,match="FOREIGN_PARAMETER"):
        qualify_rss_vnext_optimizer_ownership_v1(
            net,torch.optim.AdamW(foreign.parameters()),
        )


def test_actual_tessa_only_stage_is_safe_with_IRIS_frozen_and_optimizer_subset():
    net=make_model()
    frozen=[p for name,p in net.named_parameters() if not name.startswith("tessa.")]
    tessa=[p for name,p in net.named_parameters() if name.startswith("tessa.")]
    assert frozen and tessa
    for p in frozen:
        p.requires_grad_(False)
    # A former stage may have left a stale gradient on a now-frozen tensor.
    frozen[0].grad=torch.ones_like(frozen[0])
    trainer=RSSvNextStructuralTrainer(
        net,torch.optim.AdamW(tessa,lr=1e-3),
        dataset_manifest_sha256=manifest_hash(),
        weights=RSSvNextTrainingWeights(
            topology=1., motion_preference=0., source_semantics=0.,
            flexicubes_surface=0.,
        ),
    )
    receipt=trainer.optimizer_ownership
    assert receipt.trainable_model_parameter_tensors==len(tessa)
    assert receipt.optimizer_registered_parameter_tensors==len(tessa)
    frozen_snapshot=[p.detach().clone() for p in frozen]
    outcome=trainer.train_step(make_sample(net))
    assert torch.isfinite(outcome.total_loss)
    assert trainer.global_step==1
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in tessa)
    assert all(p.grad is None for p in frozen)
    assert all(torch.equal(a,p) for a,p in zip(frozen_snapshot,frozen))


def test_stage_freeze_or_optimizer_group_changes_are_detected_before_backward():
    net=make_model()
    optimizer=torch.optim.AdamW(net.parameters(),lr=1e-3)
    trainer=RSSvNextStructuralTrainer(
        net,optimizer,dataset_manifest_sha256=manifest_hash(),
    )
    outside=net.iris.scene.scene_queries
    outside.requires_grad_(False)
    with pytest.raises(RuntimeError,match="OPTIMIZER_PHASE_IDENTITY_DRIFT"):
        trainer.train_step(make_sample(net))
    outside.requires_grad_(True)
    assert trainer._assert_optimizer_ownership() is None
    optimizer.param_groups[0]["params"].append(next(net.parameters()))
    with pytest.raises(ValueError,match="PARAMETER_DUPLICATE"):
        trainer.train_step(make_sample(net))


def test_optimizer_stage_ownership_is_checkpoint_identity(tmp_path):
    net=make_model()
    optimizer=torch.optim.AdamW(net.parameters(),lr=1e-3)
    trainer=RSSvNextStructuralTrainer(
        net,optimizer,dataset_manifest_sha256=manifest_hash(),
    )
    path=tmp_path/"owned-stage.pt"
    trainer.save_checkpoint(path,sampler_cursor=8)
    new=make_model()
    ordered=list(new.parameters())
    # Same model architecture, but a different parameter-to-optimizer
    # ordering silently maps optimizer moments to the wrong expert.
    permuted=torch.optim.AdamW(list(reversed(ordered)),lr=1e-3)
    challenger=RSSvNextStructuralTrainer(
        new,permuted,dataset_manifest_sha256=manifest_hash(),
    )
    with pytest.raises(ValueError,match="CHECKPOINT_OPTIMIZER_OWNERSHIP_DRIFT"):
        challenger.load_checkpoint(path)
    restored=make_model()
    correct=RSSvNextStructuralTrainer(
        restored,torch.optim.AdamW(restored.parameters(),lr=1e-3),
        dataset_manifest_sha256=manifest_hash(),
    )
    assert correct.load_checkpoint(path)==8
