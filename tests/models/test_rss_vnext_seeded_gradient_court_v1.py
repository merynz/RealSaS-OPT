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
    assert set(report["active_shared_task_families"])=={"GEOMETRY","GRAPHICS","JOINT_RENDER"}
    assert report["independent_source_truth_inferred"] is False
    assert report["product_release_qualified"] is False
    assert result.total_loss.isfinite()


@pytest.mark.parametrize("case", [
    "test_partition_counts_source_qualified_groups_once",
    "test_partition_rejects_missing_teacher_and_false_total",
    "test_conflict_projection_changes_only_shared_gradients_and_preserves_total_descent",
    "test_no_conflict_exactly_preserves_vanilla_weighted_sum",
    "test_sparse_single_qualified_family_does_not_invent_negative_teacher",
    "test_wrong_gradient_partition_aborts_before_optimizer_mutation",
    "test_memory_cap_and_nonproduct_scope_fail_closed",
    "test_routing_cadence_and_vanilla_default_do_not_probe_graph",
])
def test_multitask_router_extended_properties_in_mandatory_source_gate(case):
    # Source court's explicit CI command runs this file, so the independent
    # pure-router regressions must also execute, not merely exist in-tree.
    from tests.models import (
        test_rss_vnext_product_multitask_gradient_control_v1 as cases,
    )
    getattr(cases, case)()


@pytest.mark.parametrize("test_name", (
    "test_canonical_surface_appearance_stays_stable_under_pose_depth_translation",
    "test_posed_mstar_raster_backpropagates_to_geometry_and_graphics",
    "test_independently_admitted_pose_raster_loss_is_differentiable",
    "test_pose_teacher_tampering_or_invented_motion_is_rejected",
))
def test_qualified_native_mstar_pose_raster_in_required_model_ci(
    monkeypatch, test_name,
):
    # The mandatory model source gate runs this file; no uncollected new test.
    from tests.models import test_rss_vnext_product_mstar_pose_graphics_v1 as court
    getattr(court, test_name)(monkeypatch)


@pytest.mark.parametrize("test_name", (
    "test_source_same_mstar_rest_and_pose_has_zero_clearance_loss",
    "test_collapse_gradients_reach_predicted_garment_and_body_mstar_vertices",
    "test_nonindependent_or_stale_source_native_pair_receipt_is_refused",
    "test_unknown_pose_mask_and_source_body_penetration_do_not_teach_as_valid",
))
def test_native_body_garment_clearance_in_required_seeded_source_gate(test_name):
    from tests.models import test_rss_vnext_product_native_layer_clearance_v1 as court
    getattr(court, test_name)()


def test_native_clearance_task_partition_respects_true_product_total():
    from types import SimpleNamespace
    from models.rss_vnext.product_multitask_gradient_control_v1 import (
        admitted_product_task_losses_v1,
    )
    x=torch.tensor([.3,-.2],requires_grad=True)
    geo=x.square().sum()
    clearance=(x[0]-.5).square()
    result=SimpleNamespace(
        total_loss=geo+.4*clearance,seeded_tessa_loss=geo,
        component_udf_loss=None,product_semantic_loss=None,
        graphics_source_rgb_loss=None,realsas_joint_slrm_loss=None,
        product_mstar_pose_raster_loss=None,
        product_native_layer_clearance_loss=clearance,
        product_axis_mira_loss=None,anchor_face_correspondence_loss=None,
    )
    groups=admitted_product_task_losses_v1(
        result,component_udf_weight=0.,graphics_rgb_weight=0.,
        graphics_joint_ray_weight=0.,anchor_face_weight=0.,
        native_layer_clearance_weight=.4,
    )
    assert set(groups)=={"GEOMETRY","POSED_LAYER_CLEARANCE"}
    torch.testing.assert_close(sum(groups.values()),result.total_loss)
