from __future__ import annotations

from dataclasses import replace
import pytest
import torch

from models.rss_vnext.training_engine import (
    RSSvNextStructuralTrainer, RSSvNextTrainingWeights,
)
from models.rss_vnext.training_phase_contract_v1 import (
    RSSvNextStructuralTrainingPhaseContractV1,
)
from tests.models.test_rss_vnext_structural_trainer import (
    make_model, make_sample, manifest_hash,
)
from tests.models.test_rss_vnext_training_phase_contract_v1 import (
    source_receipt,
)


@pytest.fixture(autouse=True)
def seeded():
    torch.manual_seed(1228)
    torch.set_num_threads(1)


def _weights():
    return RSSvNextTrainingWeights(
        topology=1., motion_preference=0., source_semantics=0.,
        flexicubes_surface=0.,
    )


def _case(contract):
    model=make_model()
    contract.freeze_for_phase(model)
    batch=make_sample(model)
    batch.source_topology_admission_key="source-topology"
    optimizer=torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],lr=1e-3,
    )
    trainer=RSSvNextStructuralTrainer(
        model,optimizer,weights=_weights(),
        dataset_manifest_sha256=manifest_hash(),
        phase_contract=contract,
        trusted_topology_teacher_registry={
            "source-topology":source_receipt(batch),
        },
    )
    return model,batch,trainer


def _contract(*, owners=("tessa.",), grads=("tessa.",)):
    return RSSvNextStructuralTrainingPhaseContractV1(
        name="TOPOLOGY_GRADIENT_REACHABILITY",
        trainable_parameter_prefixes=owners,
        allowed_positive_objectives=("topology",),
        required_any_qualified_loss_groups=(("topology",),),
        required_gradient_parameter_prefixes=grads,
    )


def test_supervised_tessa_receives_actual_gradient_before_optimizer_step():
    model,batch,trainer=_case(_contract())
    result=trainer.train_step(batch)
    assert result.eligible_topology_examples==1
    assert trainer.global_step==1
    assert any(
        p.grad is not None and bool(torch.isfinite(p.grad).all())
        for n,p in model.named_parameters() if n.startswith("tessa.")
    )


def test_dead_dynamic_quality_head_cannot_silently_pass_as_trained():
    policy=_contract(
        owners=("tessa.","dynamic_refine."),
        grads=("tessa.","dynamic_refine.quality_head."),
    )
    model,batch,trainer=_case(policy)
    with pytest.raises(ValueError,match="REQUIRED_GRADIENT_MISSING:dynamic_refine.quality_head."):
        trainer.train_step(batch)
    assert trainer.global_step==0
    assert not trainer.optimizer.state
    assert all(
        p.grad is None
        for n,p in model.named_parameters()
        if n.startswith("dynamic_refine.quality_head.")
    )


def test_gradient_requirement_must_be_within_owned_trainable_scope():
    with pytest.raises(ValueError,match="REQUIRED_GRADIENT_PREFIX_INVALID"):
        _contract(owners=("tessa.",),grads=("dynamic_refine.",)).validate()
    with pytest.raises(ValueError,match="REQUIRED_GRADIENT_PREFIX_INVALID"):
        _contract(grads=("tessa",)).validate()


def test_phase_checkpoint_identity_cannot_disable_gradient_requirements(tmp_path):
    model,batch,trainer=_case(_contract())
    path=tmp_path/"phase.pt"
    trainer.save_checkpoint(path,sampler_cursor=0)
    model2,batch2,without=_case(_contract(grads=()))
    with pytest.raises(ValueError,match="OBJECTIVE_POLICY_DRIFT"):
        without.load_checkpoint(path)
    assert without.global_step==0


def test_legitimate_zero_gradient_is_not_a_false_failure():
    model,batch,trainer=_case(_contract())
    trainer.train_step(batch)
    for name,p in model.named_parameters():
        if name.startswith("tessa.") and p.grad is not None:
            p.grad.zero_()
    trainer.phase_contract.assert_required_gradients(model)
