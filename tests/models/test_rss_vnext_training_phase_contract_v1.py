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


@pytest.fixture(autouse=True)
def deterministic():
    torch.manual_seed(712)
    torch.set_num_threads(1)


def policy(name="TESSA_TOPOLOGY"):
    return RSSvNextStructuralTrainingPhaseContractV1(
        name=name,
        trainable_parameter_prefixes=("tessa.",),
        allowed_positive_objectives=("topology", "motion_preference"),
        required_any_qualified_loss_groups=(("topology", "motion_preference"),),
    )


def weights():
    return RSSvNextTrainingWeights(
        topology=1., motion_preference=0., source_semantics=0.,
        flexicubes_surface=0.,
    )


def make_phase_trainer(*, contract=None):
    model=make_model()
    phase=contract or policy()
    owners=phase.freeze_for_phase(model)
    assert owners and all(n.startswith("tessa.") for n in owners)
    optimizer=torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),lr=1.e-3,
    )
    return model,RSSvNextStructuralTrainer(
        model,optimizer,dataset_manifest_sha256=manifest_hash(),
        weights=weights(),phase_contract=phase,
    )


def test_training_phase_freezes_other_experts_and_trains_tessa():
    model,trainer=make_phase_trainer()
    frozen=[(n,p.detach().clone()) for n,p in model.named_parameters()
            if not p.requires_grad]
    result=trainer.train_step(make_sample(model))
    assert result.eligible_topology_examples==1
    assert trainer.global_step==1
    assert any(p.grad is not None for n,p in model.named_parameters()
               if n.startswith("tessa.") and p.requires_grad)
    assert all(torch.equal(snapshot,dict(model.named_parameters())[n])
               for n,snapshot in frozen)
    assert all(p.grad is None for n,p in model.named_parameters()
               if not p.requires_grad)


def test_phase_rejects_unfrozen_owner_and_wrong_objective_before_backward():
    model=make_model()
    with pytest.raises(ValueError,match="UNAUTHORIZED_TRAINABLE_OWNER"):
        RSSvNextStructuralTrainer(
            model,torch.optim.AdamW(model.parameters()),
            dataset_manifest_sha256=manifest_hash(),
            weights=weights(),phase_contract=policy(),
        )
    policy().freeze_for_phase(model)
    with pytest.raises(ValueError,match="UNAUTHORIZED_OBJECTIVE"):
        RSSvNextStructuralTrainer(
            model,torch.optim.AdamW(p for p in model.parameters() if p.requires_grad),
            dataset_manifest_sha256=manifest_hash(),
            weights=replace(weights(),flexicubes_surface=.2),
            phase_contract=policy(),
        )
    with pytest.raises(ValueError,match="REQUIRED_OBJECTIVE_DISABLED"):
        policy().validate(weights=replace(weights(),topology=0.,source_semantics=.5))


def test_phase_source_coverage_requires_real_qualified_count():
    model,trainer=make_phase_trainer()
    result=trainer.objective(make_sample(model))
    assert result.topology_loss is not None
    contract=policy()
    contract.assert_qualified_result(result,weights=weights())
    result.eligible_topology_examples=0
    with pytest.raises(ValueError,match="REQUIRED_QUALIFIED_SUPERVISION_MISSING"):
        contract.assert_qualified_result(result,weights=weights())
    result.eligible_topology_examples=1
    result.topology_loss=None
    with pytest.raises(ValueError,match="REQUIRED_QUALIFIED_SUPERVISION_MISSING"):
        contract.assert_qualified_result(result,weights=weights())


def test_phase_owner_drift_rejected_before_training():
    model,trainer=make_phase_trainer()
    for name,p in model.named_parameters():
        if name.startswith("iris."):
            p.requires_grad_(True)
            break
    with pytest.raises(ValueError,match="UNAUTHORIZED_TRAINABLE_OWNER"):
        trainer.train_step(make_sample(model))
    assert trainer.global_step==0


def test_phase_contract_bound_to_checkpoint_and_resumption(tmp_path):
    model,trainer=make_phase_trainer()
    trainer.train_step(make_sample(model))
    path=tmp_path/"stage.pt"
    trainer.save_checkpoint(path,sampler_cursor=3)
    model2,trainer2=make_phase_trainer()
    assert trainer2.load_checkpoint(path)==3
    assert trainer2.global_step==1

    model3,alternate=make_phase_trainer(contract=policy("TESSA_SECOND_STAGE"))
    with pytest.raises(ValueError,match="OBJECTIVE_POLICY_DRIFT"):
        alternate.load_checkpoint(path)


def test_phase_contract_rejects_invalid_owners_and_missing_groups():
    with pytest.raises(ValueError,match="PHASE_CONTRACT_INVALID"):
        RSSvNextStructuralTrainingPhaseContractV1(
            name="bad",trainable_parameter_prefixes=("tessa",),
            allowed_positive_objectives=("topology",),
            required_any_qualified_loss_groups=(("topology",),),
        ).validate()
    with pytest.raises(ValueError,match="PHASE_REQUIRED_GROUP_INVALID"):
        RSSvNextStructuralTrainingPhaseContractV1(
            name="bad",trainable_parameter_prefixes=("tessa.",),
            allowed_positive_objectives=("topology",),
            required_any_qualified_loss_groups=(("axis_rig_skin_codes",),),
        ).validate()
    assert policy().digest()!=policy("ANOTHER_TESSA_PHASE").digest()
