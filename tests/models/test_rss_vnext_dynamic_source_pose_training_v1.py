from __future__ import annotations

"""Synthetic regression only: does not establish real-data pose quality."""

from dataclasses import replace

import pytest
import torch

from models.rss_vnext.training_engine import (
    RSSvNextStructuralTrainer, RSSvNextTrainingWeights,
)
from models.rss_vnext.training_phase_contract_v1 import (
    RSSvNextStructuralTrainingPhaseContractV1,
)
from tests.models.test_rss_vnext_unified_posed_surface_objective_v1 import (
    _qualified_synthetic_case,
)


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(1186)
    torch.set_num_threads(1)


def configured_case(weight=1.):
    net,batch,factory,receipt=_qualified_synthetic_case()
    old=factory({"synthetic-pose":receipt})
    weights=replace(old.weights, axis_dynamic_posed_surface=weight)
    trainer=RSSvNextStructuralTrainer(
        net,torch.optim.AdamW(net.parameters(),lr=1.e-3),
        dataset_manifest_sha256="3"*64,
        weights=weights,
        axis_training_policy="PRODUCT_MESH_MATCHED_REQUIRED",
        trusted_axis_teacher_registry=old.trusted_axis_teacher_registry,
        trusted_axis_pose_registry={"synthetic-pose":receipt},
    )
    return net,batch,trainer,receipt


def test_dynamic_refinement_gets_actual_source_qualified_pose_gradients():
    net,batch,trainer,receipt=configured_case()
    result=trainer.objective(batch)
    assert result.axis_posed_surface_loss is not None
    assert result.axis_dynamic_posed_surface_loss is not None
    assert result.eligible_axis_pose_count==2
    assert result.eligible_axis_dynamic_pose_count==2
    assert torch.isfinite(result.axis_dynamic_posed_surface_loss)
    result.axis_dynamic_posed_surface_loss.backward()
    # Before this objective, these trained submodules had no source-label
    # gradient path from RSSvNextStructuralTrainer at all.
    for parameter in (
        net.dynamic_refine.vertex_head.weight,
        net.dynamic_refine.vertex_head.bias,
        net.dynamic_refine.vertex_context.weight,
    ):
        assert parameter.grad is not None
        assert bool(torch.isfinite(parameter.grad).all())
    assert net.dynamic_refine.vertex_head.bias.grad.abs().sum()>0
    assert net.axis.skin_code_head.weight.grad is not None
    assert net.surface_refine[-1].weight.grad is not None
    assert all(p.grad is None for p in net.mira.codec.parameters())


def test_dynamic_pose_uses_same_receipt_and_rejects_unverified_teacher():
    net,batch,trainer,receipt=configured_case()
    with pytest.raises(ValueError,match="POSE_RECEIPT_UNQUALIFIED"):
        trainer.objective(replace(
            batch,axis_pose_surface_samples=batch.axis_pose_surface_samples+.13,
        ))
    with pytest.raises(ValueError,match="POSE_RECEIPT_UNQUALIFIED"):
        changed=RSSvNextStructuralTrainer(
            net,torch.optim.AdamW(net.parameters()),
            dataset_manifest_sha256="3"*64,
            weights=trainer.weights,
            axis_training_policy="PRODUCT_MESH_MATCHED_REQUIRED",
            trusted_axis_teacher_registry=trainer.trusted_axis_teacher_registry,
            trusted_axis_pose_registry={"synthetic-pose":{
                **receipt,"pose_witness_source":"GENERATED_DIFFUSION",
            }},
        )
        changed.objective(batch)
    assert trainer.global_step==0


def test_dynamic_pose_opt_in_and_checkpoint_identity(tmp_path):
    net,batch,trainer,receipt=configured_case(.5)
    path=tmp_path/"with-dynamic.pt"
    trainer.save_checkpoint(path,sampler_cursor=17)
    assert trainer.load_checkpoint(path)==17
    old=RSSvNextStructuralTrainer(
        net,torch.optim.AdamW(net.parameters()),
        dataset_manifest_sha256="3"*64,
        weights=replace(trainer.weights,axis_dynamic_posed_surface=0.),
        axis_training_policy="PRODUCT_MESH_MATCHED_REQUIRED",
        trusted_axis_teacher_registry=trainer.trusted_axis_teacher_registry,
        trusted_axis_pose_registry=trainer.trusted_axis_pose_registry,
    )
    assert old.objective(batch).axis_dynamic_posed_surface_loss is None
    with pytest.raises(ValueError,match="OBJECTIVE_POLICY_DRIFT"):
        old.load_checkpoint(path)


def test_dynamic_pose_cannot_be_enabled_without_matched_axis_and_static_source_pose():
    net,batch,factory,receipt=_qualified_synthetic_case()
    basic=factory({"synthetic-pose":receipt})
    with pytest.raises(ValueError,match="DYNAMIC_POSE_REQUIRES_QUALIFIED_AXIS"):
        RSSvNextStructuralTrainer(
            net,torch.optim.AdamW(net.parameters()),
            dataset_manifest_sha256="3"*64,
            weights=replace(basic.weights,axis_posed_surface=0.,axis_dynamic_posed_surface=1.),
            axis_training_policy="PRODUCT_MESH_MATCHED_REQUIRED",
        )


def test_dynamic_pose_phase_contract_has_distinct_source_eligibility():
    p=RSSvNextStructuralTrainingPhaseContractV1(
        name="DYNAMIC_SOURCE_MOTION",
        trainable_parameter_prefixes=("dynamic_refine.",),
        allowed_positive_objectives=(
            "axis_posed_surface","axis_dynamic_posed_surface",
        ),
        required_any_qualified_loss_groups=(("axis_dynamic_posed_surface",),),
    )
    assert p.digest()
