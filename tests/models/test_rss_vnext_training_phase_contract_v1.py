from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import pytest
import torch

from models.rss_vnext.training_engine import (
    RSSvNextStructuralTrainer, RSSvNextTrainingWeights,
    source_surface_tensor_sha256,
)
from models.rss_vnext.training_phase_contract_v1 import (
    RSSvNextStructuralTrainingPhaseContractV1,
)
from models.rss_vnext.training_topology_source_admission_v1 import SCHEMA, STATUS
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


def source_receipt(batch):
    return {
        "schema":SCHEMA,
        "admission_status":STATUS,
        "dataset_manifest_sha256":manifest_hash(),
        "source_qualification_key":batch.source_qualification_key,
        "observed_rgba_sha256":source_surface_tensor_sha256(batch.rgba),
        "input_camera_features_sha256":source_surface_tensor_sha256(batch.camera_features),
        "topology_teacher_tokens_sha256":source_surface_tensor_sha256(batch.topology_teacher_tokens),
        "topology_valid_mask_sha256":source_surface_tensor_sha256(batch.topology_valid_mask),
        "teacher_semantics":"SOURCE_NATIVE_INDEXED_MESH_NOT_STUDENT_PREDICTION",
        "source_mesh_reindex_verified":True,
        "source_asset_sha256":sha256(b"source-asset-test-only").hexdigest(),
        "source_native_faces_sha256":sha256(b"faces-test-only").hexdigest(),
    }


def make_phase_trainer(*, contract=None, batch=None, receipt=None):
    model=make_model()
    phase=contract or policy()
    owners=phase.freeze_for_phase(model)
    assert owners and all(n.startswith("tessa.") for n in owners)
    if batch is None:
        batch=make_sample(model)
    batch.source_topology_admission_key="test-source-topology"
    optimizer=torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),lr=1.e-3,
    )
    trainer=RSSvNextStructuralTrainer(
        model,optimizer,dataset_manifest_sha256=manifest_hash(),
        weights=weights(),phase_contract=phase,
        trusted_topology_teacher_registry={
            batch.source_topology_admission_key:receipt or source_receipt(batch),
        },
    )
    return model,trainer,batch


def test_training_phase_freezes_other_experts_and_trains_tessa():
    model,trainer,batch=make_phase_trainer()
    frozen=[(n,p.detach().clone()) for n,p in model.named_parameters()
            if not p.requires_grad]
    result=trainer.train_step(batch)
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
        RSSvNextStructuralTrainingPhaseContractV1(
            name="REQUIRE_GEOMETRY",trainable_parameter_prefixes=("tessa.",),
            allowed_positive_objectives=("topology","source_semantics"),
            required_any_qualified_loss_groups=(("topology",),),
        ).validate(weights=replace(weights(),topology=0.,source_semantics=.5))


def test_phase_source_coverage_requires_real_qualified_count():
    model,trainer,batch=make_phase_trainer()
    result=trainer.objective(batch)
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


def test_phase_rejects_unadmitted_or_tampered_source_topology_before_backward():
    model,trainer,batch=make_phase_trainer()
    with pytest.raises(ValueError,match="TOPOLOGY_RECEIPT_DRIFT"):
        trainer.train_step(replace(batch,rgba=batch.rgba+0.01))
    damaged=batch.topology_teacher_tokens.clone()
    damaged[0,1]+=1
    with pytest.raises(ValueError,match="TOPOLOGY_RECEIPT_DRIFT"):
        trainer.train_step(replace(batch,topology_teacher_tokens=damaged))
    with pytest.raises(ValueError,match="TOPOLOGY_BATCH_INCOMPLETE"):
        trainer.train_step(replace(batch,source_topology_admission_key=""))
    assert trainer.global_step==0


def test_phase_rejects_unqualified_source_mesh_provenance():
    model=make_model()
    batch=make_sample(model)
    bad=source_receipt(batch)
    bad["source_mesh_reindex_verified"]=False
    _,trainer,batch=make_phase_trainer(batch=batch,receipt=bad)
    with pytest.raises(ValueError,match="TOPOLOGY_RECEIPT_DRIFT"):
        trainer.train_step(batch)


def test_phase_owner_drift_rejected_before_training():
    model,trainer,batch=make_phase_trainer()
    for name,p in model.named_parameters():
        if name.startswith("iris."):
            p.requires_grad_(True)
            break
    with pytest.raises(ValueError,match="UNAUTHORIZED_TRAINABLE_OWNER"):
        trainer.train_step(batch)
    assert trainer.global_step==0


def test_phase_contract_and_source_registry_bound_to_checkpoint(tmp_path):
    model,trainer,batch=make_phase_trainer()
    trainer.train_step(batch)
    path=tmp_path/"stage.pt"
    trainer.save_checkpoint(path,sampler_cursor=3)
    model2,trainer2,_=make_phase_trainer(batch=batch)
    assert trainer2.load_checkpoint(path)==3
    assert trainer2.global_step==1
    model3,alternate,_=make_phase_trainer(contract=policy("TESSA_SECOND_STAGE"),batch=batch)
    with pytest.raises(ValueError,match="OBJECTIVE_POLICY_DRIFT"):
        alternate.load_checkpoint(path)

    changed=source_receipt(batch)
    changed["source_native_faces_sha256"]=sha256(b"another-mesh").hexdigest()
    model4,changed_trainer,_=make_phase_trainer(batch=batch,receipt=changed)
    with pytest.raises(ValueError,match="CHECKPOINT_TOPOLOGY_TEACHER_REGISTRY_DRIFT"):
        changed_trainer.load_checkpoint(path)


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
