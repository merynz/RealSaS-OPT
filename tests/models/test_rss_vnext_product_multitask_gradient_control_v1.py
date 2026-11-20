from __future__ import annotations

"""No pretrained weights, source laundering, or real-asset claims in these tests."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from models.rss_vnext.product_multitask_gradient_control_v1 import (
    ProductMultitaskGradientPolicyV1,
    admitted_product_task_losses_v1,
    prepare_product_gradient_route_v1,
)


class TinyShared(nn.Module):
    def __init__(self):
        super().__init__()
        self.iris = nn.Module()
        self.iris.scene = nn.Module()
        self.iris.scene.weight = nn.Parameter(torch.tensor([0.4, -0.6]))
        self.tessa_seeded_head = nn.Linear(2, 1, bias=False)
        self.graphics = nn.Linear(2, 1, bias=False)
        self.axis = nn.Module()
        self.axis.private = nn.Parameter(torch.tensor([0.2, 0.3]))


def _result(*, geometry, semantic=None, graphics=None, mechanics=None,
            component=None, anchor=None, joint_rays=None):
    appearance_total = .75 * graphics if graphics is not None else None
    rendered_total = .25 * joint_rays if joint_rays is not None else None
    mechanical_total = mechanics
    if anchor is not None:
        mechanical_total = (.5 * anchor if mechanical_total is None
                            else mechanical_total + .5 * anchor)
    total = geometry + (2.0 * component if component is not None else 0)
    for v in (semantic, appearance_total, rendered_total, mechanical_total):
        if v is not None:
            total = total + v
    return SimpleNamespace(
        total_loss=total,
        seeded_tessa_loss=geometry,
        component_udf_loss=component,
        product_semantic_loss=semantic,
        graphics_source_rgb_loss=graphics,
        realsas_joint_slrm_loss=joint_rays,
        product_axis_mira_loss=mechanics,
        anchor_face_correspondence_loss=anchor,
    )


def _families(result):
    return admitted_product_task_losses_v1(
        result, component_udf_weight=2.,
        graphics_rgb_weight=.75, graphics_joint_ray_weight=.25,
        anchor_face_weight=.5,
    )


def _policy(**kwargs):
    return ProductMultitaskGradientPolicyV1(
        mode="BOUNDED_SHARED_PCGRAD", **kwargs
    )


def test_partition_counts_source_qualified_groups_once():
    net = TinyShared()
    x = net.iris.scene.weight
    r = _result(
        geometry=(x[0] ** 2).sum(),
        semantic=x[1] ** 2,
        graphics=(x[0] - x[1]) ** 2,
        mechanics=net.axis.private.square().sum(),
        component=x.square().sum(),
        anchor=(x[0] + 1) ** 2,
        joint_rays=(x[0] + x[1]) ** 2,
    )
    families = _families(r)
    assert set(families) == {
        "GEOMETRY", "SEMANTIC", "GRAPHICS", "JOINT_RENDER", "MECHANICS"
    }
    torch.testing.assert_close(sum(families.values()), r.total_loss)


def test_partition_rejects_missing_teacher_and_false_total():
    x = nn.Parameter(torch.tensor([0.2, 0.3]))
    r = _result(geometry=x.square().sum(), semantic=x.sum())
    r.seeded_tessa_loss = None
    with pytest.raises(RuntimeError, match="SOURCE_REQUIRED"):
        _families(r)
    r.seeded_tessa_loss = x.square().sum()
    r.total_loss = r.total_loss + 5
    with pytest.raises(RuntimeError, match="PARTITION_DRIFT"):
        _families(r)


def test_conflict_projection_changes_only_shared_gradients_and_preserves_total_descent():
    net = TinyShared()
    x, private = net.iris.scene.weight, net.axis.private
    geometry = 4*x[0] + x[1] + private.square().sum()
    semantic = -3*x[0] + 2*x[1]
    graphics = -x[0] - 4*x[1] + net.graphics.weight.square().sum()
    r = _result(geometry=geometry, semantic=semantic, graphics=graphics)
    policy = _policy()
    route = prepare_product_gradient_route_v1(
        named_parameters=tuple(net.named_parameters()),
        task_losses=_families(r), policy=policy, global_step=0,
    )
    assert route is not None
    assert route.report["task_private_gradients_preserved"]
    assert route.report["conflicting_task_pairs"]
    r.total_loss.backward()
    before_private = private.grad.detach().clone()
    before_graphics = net.graphics.weight.grad.detach().clone()
    report = route.apply_after_backward()
    assert report["first_order_total_descent_preserved"]
    torch.testing.assert_close(private.grad, before_private, rtol=0, atol=0)
    torch.testing.assert_close(net.graphics.weight.grad, before_graphics,
                               rtol=0, atol=0)
    assert torch.isfinite(x.grad).all()


def test_no_conflict_exactly_preserves_vanilla_weighted_sum():
    net = TinyShared()
    x = net.iris.scene.weight
    r = _result(geometry=2*x.sum(), semantic=3*x.sum())
    route = prepare_product_gradient_route_v1(
        named_parameters=tuple(net.named_parameters()),
        task_losses=_families(r), policy=_policy(), global_step=0,
    )
    assert route is not None
    assert route.report["projection_applied"] is False
    r.total_loss.backward()
    before = x.grad.clone()
    route.apply_after_backward()
    torch.testing.assert_close(x.grad, before, rtol=0, atol=0)


def test_sparse_single_qualified_family_does_not_invent_negative_teacher():
    net = TinyShared()
    x = net.iris.scene.weight
    r = _result(geometry=x.square().sum())
    route = prepare_product_gradient_route_v1(
        named_parameters=tuple(net.named_parameters()),
        task_losses=_families(r), policy=_policy(), global_step=0,
    )
    assert route is None


def test_wrong_gradient_partition_aborts_before_optimizer_mutation():
    net = TinyShared()
    x = net.iris.scene.weight
    a, b = x[0]**2, x[1]**2
    route = prepare_product_gradient_route_v1(
        named_parameters=tuple(net.named_parameters()),
        task_losses={"GEOMETRY": a, "SEMANTIC": b},
        policy=_policy(), global_step=0,
    )
    assert route is not None
    (a + b + x[0]).backward()
    with pytest.raises(RuntimeError, match="BACKWARD_GRAPH_DRIFT"):
        route.apply_after_backward()


def test_memory_cap_and_nonproduct_scope_fail_closed():
    net = TinyShared()
    x = net.iris.scene.weight
    tasks = {"GEOMETRY": x[0]**2, "GRAPHICS": x[1]**2}
    with pytest.raises(RuntimeError, match="MEMORY_BUDGET"):
        prepare_product_gradient_route_v1(
            named_parameters=tuple(net.named_parameters()),
            task_losses=tasks, policy=_policy(max_shared_elements=1),
            global_step=0,
        )
    with pytest.raises(ValueError, match="POLICY_INVALID"):
        _policy(shared_owner_prefixes=("axis.",)).validate()
    with pytest.raises(ValueError, match="POLICY_INVALID"):
        _policy(max_candidate_norm_ratio=10.).validate()


def test_routing_cadence_and_vanilla_default_do_not_probe_graph():
    net = TinyShared()
    x = net.iris.scene.weight
    tasks = {"GEOMETRY": x[0]**2, "SEMANTIC": x[1]**2}
    assert prepare_product_gradient_route_v1(
        named_parameters=tuple(net.named_parameters()),
        task_losses=tasks, policy=ProductMultitaskGradientPolicyV1(),
        global_step=0,
    ) is None
    assert prepare_product_gradient_route_v1(
        named_parameters=tuple(net.named_parameters()),
        task_losses=tasks, policy=_policy(every_n_steps=3),
        global_step=1,
    ) is None
