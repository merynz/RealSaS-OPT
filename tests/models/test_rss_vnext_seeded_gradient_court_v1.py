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


def test_product_multitask_partition_and_private_gradient_in_seeded_court():
    """Run inside the existing model-mainline-source CI gate."""
    from types import SimpleNamespace
    from models.rss_vnext.product_multitask_gradient_control_v1 import (
        ProductMultitaskGradientPolicyV1,
        admitted_product_task_losses_v1,
        prepare_product_gradient_route_v1,
    )
    from torch import nn

    class Shared(nn.Module):
        def __init__(self):
            super().__init__()
            self.iris = nn.Module()
            self.iris.scene = nn.Module()
            self.iris.scene.weight = nn.Parameter(torch.tensor([.3, -.2]))
            self.graphics = nn.Linear(2, 1, bias=False)

    net = Shared()
    x = net.iris.scene.weight
    geo = 3*x[0]+x[1]
    sem = -2*x[0]+3*x[1]
    vis = (x[0]-x[1])**2 + net.graphics.weight.square().sum()
    total = geo + sem + .5*vis
    result = SimpleNamespace(
        total_loss=total, seeded_tessa_loss=geo,
        component_udf_loss=None, product_semantic_loss=sem,
        graphics_source_rgb_loss=vis, realsas_joint_slrm_loss=None,
        product_axis_mira_loss=None, anchor_face_correspondence_loss=None,
    )
    tasks = admitted_product_task_losses_v1(
        result, component_udf_weight=0., graphics_rgb_weight=.5,
        graphics_joint_ray_weight=0., anchor_face_weight=0.,
    )
    assert set(tasks)=={"GEOMETRY","SEMANTIC","GRAPHICS"}
    route = prepare_product_gradient_route_v1(
        named_parameters=tuple(net.named_parameters()), task_losses=tasks,
        policy=ProductMultitaskGradientPolicyV1(
            mode="BOUNDED_SHARED_PCGRAD",
        ), global_step=0,
    )
    assert route is not None
    result.total_loss.backward()
    private_before=net.graphics.weight.grad.detach().clone()
    report=route.apply_after_backward()
    assert report["task_private_gradients_preserved"]
    assert report["first_order_total_descent_preserved"]
    torch.testing.assert_close(net.graphics.weight.grad,private_before,atol=0,rtol=0)


def test_source_graphics_seeded_trainer_routes_same_H_without_stealing_truth(
    monkeypatch,
):
    """One real product trainer step, not just detached synthetic vectors."""
    from models.rss_vnext.product_multitask_gradient_control_v1 import (
        ProductMultitaskGradientPolicyV1,
    )
    from tests.models.test_rss_vnext_product_graphics_source_training_v1 import (
        _appearance,
    )
    net,batch,make,*_=_appearance(monkeypatch)
    routed=make(gradient_route_policy=ProductMultitaskGradientPolicyV1(
        mode="BOUNDED_SHARED_PCGRAD",
        max_shared_elements=2_000_000,
    ))
    ordinary=make()
    assert routed.objective_policy_sha256!=ordinary.objective_policy_sha256
    result=routed.train_step(batch)
    report=result.product_gradient_route_report
    assert report is not None
    assert set(report["active_shared_task_families"])=={"GEOMETRY","GRAPHICS"}
    assert report["independent_source_truth_inferred"] is False
    assert report["product_release_qualified"] is False
    assert result.total_loss.isfinite()
