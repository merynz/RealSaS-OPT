from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from models.rss_vnext.training_gradient_interference_v1 import (
    RSSvNextGradientCourtPolicyV1,
    audit_rss_vnext_gradient_interference_v1,
)
from models.rss_vnext.training_engine import (
    RSSvNextStructuralTrainer, RSSvNextTrainingWeights,
    source_surface_tensor_sha256,
)
from tests.models.test_rss_vnext_structural_trainer import (
    make_model, make_sample, manifest_hash,
)


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(71)
    torch.set_num_threads(1)


def probe(losses, *, named, max_params=5, max_elements=65536):
    return audit_rss_vnext_gradient_interference_v1(
        named_parameters=named,
        loss_terms=losses,
        policy=RSSvNextGradientCourtPolicyV1(
            every_n_steps=1, max_parameters_per_owner=max_params,
            max_elements_per_owner=max_elements,
        ),
    )


def test_exact_antiparallel_gradient_conflict_and_cancellation():
    p=torch.nn.Parameter(torch.tensor([1., 2., 3.]))
    before=p.detach().clone()
    p.grad=torch.full_like(p, 7.)
    out=probe(
        {"shape":(p.sum(),1.),"graphics":(-p.sum(),1.)},
        named=[("iris.scene.scene_queries",p)],
    )
    row=out["owners"]["SHARED_IRIS_SCENE"]
    assert len(row["pairwise"])==1
    assert row["pairwise"][0]["cosine"] == pytest.approx(-1.)
    assert row["pairwise_negative_count"]==1
    assert row["cancellation_ratio"]==pytest.approx(0.,abs=1e-12)
    assert row["worst_sampled_parameters"][0]["name"]=="iris.scene.scene_queries"
    assert out["modifies_gradient"] is False
    assert torch.equal(p.grad,torch.full_like(p,7.))
    assert torch.equal(p,before)


def test_orthogonal_and_aligned_objectives_do_not_forge_conflict():
    p=torch.nn.Parameter(torch.tensor([1., 1.]))
    orthogonal=probe({
        "x":(p[0]*2,1.),
        "y":(p[1]*3,1.),
    },named=[("iris.scene.scene_queries",p)])
    row=orthogonal["owners"]["SHARED_IRIS_SCENE"]
    assert row["pairwise_negative_count"]==0
    assert row["pairwise"][0]["cosine"]==pytest.approx(0.,abs=1e-12)
    assert row["cancellation_ratio"]==pytest.approx((13**.5)/5)
    aligned=probe({
        "a":(p.sum(),1.),
        "b":(p.sum()*2,3.),
    },named=[("iris.scene.scene_queries",p)])
    row=aligned["owners"]["SHARED_IRIS_SCENE"]
    assert row["pairwise"][0]["cosine"]==pytest.approx(1.)
    assert row["cancellation_ratio"]==pytest.approx(1.)


def test_undefined_gradients_remain_undefined_and_sampling_coverage_is_explicit():
    p=torch.nn.Parameter(torch.tensor([1.,2.]))
    unused=torch.nn.Parameter(torch.tensor([0.,0.]))
    other=torch.nn.Parameter(torch.tensor([2.,2.]))
    out=probe({
        "only_one_owner":(p.sum(),1.),
        "disjoint_owner":(other.sum(),1.),
    },named=[
        ("iris.scene.layer0",p),
        ("iris.scene.layer1",unused),
        ("graphics.rgb_head",other),
    ],max_params=1)
    scene=out["owners"]["SHARED_IRIS_SCENE"]
    graphics=out["owners"]["GRAPHICS_HEAD"]
    assert scene["coverage_is_complete"] is False
    assert scene["eligible_parameter_count"]==2
    assert scene["pairwise"][0]["cosine"] is None
    assert scene["pairwise_defined_count"]==0
    assert graphics["pairwise"][0]["cosine"] is None
    assert scene["per_family_gradient_norms"]["disjoint_owner"]==0.


def test_gradient_audit_rejects_malformed_terms_and_unmeasurable_owner():
    p=torch.nn.Parameter(torch.ones(2))
    with pytest.raises(ValueError,match="LOSS_DOMAIN"):
        probe({"wrong":(p.sum(),float("nan"))},
              named=[("iris.scene.p",p)])
    with pytest.raises(ValueError,match="NO_AUDITABLE_PARAMETERS"):
        probe({"ok":(p.sum(),1.)},named=[("some.unknown.param",p)])
    with pytest.raises(ValueError,match="POLICY_INVALID"):
        RSSvNextGradientCourtPolicyV1(every_n_steps=-1).validate()


def _two_objective_batch(net):
    batch=make_sample(net)
    target=torch.rand((1,2,4,net.cfg.semantic_evidence_channels)).round()
    mask=torch.zeros_like(target,dtype=torch.bool)
    mask[:,0,:,:2]=True
    mask[:,1,::2,2:]=True
    batch.semantic_patch_target=target
    batch.semantic_patch_mask=mask
    batch.semantic_qualification_key="source-mask"
    row={
        "admission_status":"QUALIFIED_SOURCE_SEMANTIC_PATCH_AXES",
        "dataset_manifest_sha256":manifest_hash(),
        "source_qualification_key":batch.source_qualification_key,
        "observed_rgba_sha256":source_surface_tensor_sha256(batch.rgba),
        "original_camera_sha256":source_surface_tensor_sha256(batch.camera_features),
        "semantic_patch_target_sha256":source_surface_tensor_sha256(target),
        "semantic_patch_mask_sha256":source_surface_tensor_sha256(mask),
        "supervision_semantics":"ORIGINAL_TWO_VIEW_MASKED_PATCH_AXES_NOT_3D_DRAWABLE",
    }
    return batch,{"source-mask":row}


def test_real_unified_IRIS_court_is_observational_and_optimizer_step_identical(tmp_path):
    net=make_model()
    reference=make_model()
    reference.load_state_dict(net.state_dict(),strict=True)
    batch,registry=_two_objective_batch(net)
    weights=RSSvNextTrainingWeights(
        topology=1., source_semantics=.6, motion_preference=0.,
        flexicubes_surface=0.,
    )
    audited=RSSvNextStructuralTrainer(
        net,torch.optim.AdamW(net.parameters(),lr=1.e-3),
        dataset_manifest_sha256=manifest_hash(),weights=weights,
        trusted_semantic_patch_registry=registry,
        gradient_court_policy=RSSvNextGradientCourtPolicyV1(
            every_n_steps=2,max_parameters_per_owner=5,
            max_elements_per_owner=2048,
        ),
    )
    plain=RSSvNextStructuralTrainer(
        reference,torch.optim.AdamW(reference.parameters(),lr=1.e-3),
        dataset_manifest_sha256=manifest_hash(),weights=weights,
        trusted_semantic_patch_registry=registry,
    )
    result=audited.train_step(batch)
    baseline=plain.train_step(batch)
    audit=result.gradient_interference_report
    assert audit is not None
    assert audit["schema"].endswith("TrainingGradientInterferenceCourt.v1")
    assert audit["loss_families"]==["source_semantics","topology"]
    assert "SHARED_IRIS_SCENE" in audit["owners"]
    assert audit["owners"]["SHARED_IRIS_SCENE"]["eligible_parameter_count"]>=1
    assert audit["owners"]["SHARED_IRIS_SCENE"]["sampled_scalar_elements"]>0
    assert torch.equal(result.total_loss,baseline.total_loss)
    assert result.gradient_norm==pytest.approx(float(baseline.gradient_norm),abs=1e-6)
    for a,b in zip(net.parameters(),reference.parameters()):
        assert torch.equal(a,b)
    assert result.gradient_interference_report["global_step_before_update"]==0
    next_step=audited.train_step(batch)
    assert next_step.gradient_interference_report is None
    ckpt=tmp_path/"audited.pt"
    audited.save_checkpoint(ckpt,sampler_cursor=3)
    swapped=RSSvNextStructuralTrainer(
        reference,torch.optim.AdamW(reference.parameters()),
        dataset_manifest_sha256=manifest_hash(),weights=weights,
        trusted_semantic_patch_registry=registry,
    )
    with pytest.raises(ValueError,match="CHECKPOINT_OBJECTIVE_POLICY_DRIFT"):
        swapped.load_checkpoint(ckpt)


def test_sampler_policy_changes_cannot_fake_complete_layer_coverage():
    p=torch.nn.Parameter(torch.ones(1024))
    q=torch.nn.Parameter(torch.ones(1024))
    with pytest.raises(ValueError,match="NO_AUDITABLE_PARAMETERS"):
        probe(
            {"left":(p.sum(),1.),"right":(q.sum(),1.)},
            named=[("iris.scene.large",p),("iris.scene.second",q)],
            max_params=5,max_elements=128,
        )
