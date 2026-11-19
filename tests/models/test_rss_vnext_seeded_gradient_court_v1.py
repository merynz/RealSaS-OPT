from __future__ import annotations

"""Read-only, source-qualified seeded gradient audit; no optimizer mutation."""
import pytest
import torch

from models.rss_vnext.training_gradient_interference_v1 import (
    RSSvNextGradientCourtPolicyV1,
)
from tests.models.test_rss_vnext_product_seeded_training_route_v1 import _scenario


@pytest.fixture(autouse=True)
def fixed_cpu():
    torch.manual_seed(20261119)
    torch.set_num_threads(1)


def _policy(*, cadence=1):
    return RSSvNextGradientCourtPolicyV1(
        every_n_steps=cadence,
        max_parameters_per_owner=3,
        max_elements_per_owner=8192,
        report_individual_parameters=False,
    )


def test_product_seeded_gradient_court_uses_true_losses_without_changing_optimizer(monkeypatch):
    first,batch_one,make_plain,*_=_scenario(monkeypatch)
    second,batch_two,make_audited,*_=_scenario(monkeypatch)
    # Identical initial neural checkpoint, independent optimizers and native
    # source receipts. No source truth becomes a model input through the audit.
    for (na,pa),(nb,pb) in zip(first.named_parameters(),second.named_parameters()):
        assert na==nb
        torch.testing.assert_close(pa,pb,atol=0.,rtol=0.)
    baseline=make_plain()
    audited=make_audited(gradient_court_policy=_policy())
    torch.manual_seed(201)
    baseline_result=baseline.train_step(batch_one)
    torch.manual_seed(201)
    audited_result=audited.train_step(batch_two)
    assert baseline_result.gradient_interference_report is None
    report=audited_result.gradient_interference_report
    assert report is not None
    assert report["training_profile"]=="PRODUCT_SEEDED_IRIS_TESSA_V1"
    assert report["source_teacher_authority"]=="SOURCE_ADMITTED_LOSSES_ONLY"
    assert report["modifies_gradient"] is False
    assert report["optimizer_step_performed"] is False
    assert report["admitted_differentiable_loss_count"]>=2
    assert "seeded_tessa.rest_surface" in report["loss_families"]
    assert "seeded_tessa.normal" in report["loss_families"]
    assert report["objective_policy_sha256"]==audited.objective_policy_sha256
    assert not any(t in report["loss_families"] for t in (
        "topology","motion_preference","source_udf",
    ))
    for (na,pa),(nb,pb) in zip(first.named_parameters(),second.named_parameters()):
        assert na==nb
        torch.testing.assert_close(pa,pb,atol=0.,rtol=0.)
    assert baseline.global_step==audited.global_step==1


def test_seeded_gradient_court_policy_cannot_silently_change_on_resume(
    monkeypatch,tmp_path,
):
    _,batch,make,*_=_scenario(monkeypatch)
    audited=make(gradient_court_policy=_policy(cadence=2))
    audited.train_step(batch)
    checkpoint=tmp_path/"seeded-gradient-court.pt"
    audited.save_checkpoint(checkpoint,sampler_cursor=7)
    assert make(gradient_court_policy=_policy(cadence=2)).load_checkpoint(
        checkpoint,
    )==7
    with pytest.raises(ValueError,match="OBJECTIVE_POLICY_DRIFT"):
        make().load_checkpoint(checkpoint)
    with pytest.raises(ValueError,match="OBJECTIVE_POLICY_DRIFT"):
        make(gradient_court_policy=_policy(cadence=1)).load_checkpoint(checkpoint)
