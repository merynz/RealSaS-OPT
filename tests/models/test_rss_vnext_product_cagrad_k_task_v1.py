from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from models.rss_vnext.product_cagrad_k_task_v1 import (
    k_task_admitted_shared_cagrad_v1,
)
from models.rss_vnext.product_multitask_gradient_control_v1 import (
    ProductMultitaskGradientPolicyV1,
    prepare_product_gradient_route_v1,
)


def _grads():
    return {
        "GEOMETRY": torch.tensor([3.,-1.,0.]),
        "SEMANTIC": torch.tensor([-1.,3.,0.]),
        "GRAPHICS": torch.tensor([.2,.1,1.]),
    }


def test_three_task_official_style_cagrad_simplex_and_sum_scale():
    groups = _grads()
    out = k_task_admitted_shared_cagrad_v1(groups, c=.5)
    baseline = sum(list(groups.values())[1:], list(groups.values())[0].clone())
    assert out.family_names == tuple(groups)
    assert len(out.mixture_weights) == 3
    assert sum(out.mixture_weights) == pytest.approx(1., abs=1e-6)
    assert all(0 <= w <= 1 for w in out.mixture_weights)
    assert out.candidate.shape == baseline.shape
    assert out.first_order_total_dot > 0
    assert not out.candidate.requires_grad
    assert not out.product_release_qualified
    assert out.source_authority == "ADMITTED_LOSS_GRADIENTS_ONLY_NOT_CORPUS_PROOF"
    zero = k_task_admitted_shared_cagrad_v1(groups, c=0.)
    assert torch.allclose(zero.candidate, baseline, rtol=1e-5, atol=1e-5)


def test_cagrad_rejects_degenerate_and_invalid_teacher_like_inputs():
    with pytest.raises(ValueError, match="POLICY_INVALID"):
        k_task_admitted_shared_cagrad_v1({"one":torch.ones(4)})
    with pytest.raises(ValueError, match="GRADIENT_DOMAIN"):
        k_task_admitted_shared_cagrad_v1({
            "x":torch.ones(4),"y":torch.tensor([float("nan")]*4),
        })
    with pytest.raises(RuntimeError, match="ZERO_SHARED_TOTAL_GRADIENT"):
        k_task_admitted_shared_cagrad_v1({
            "GEOMETRY":torch.ones(3),
            "GRAPHICS":-torch.ones(3),
        })


def test_cagrad_real_loss_graph_keeps_private_gradient_unmodified():
    shared = torch.nn.Parameter(torch.tensor([1.,2.,3.]))
    private = torch.nn.Parameter(torch.tensor([4.,5.,6.]))
    groups = {
        "GEOMETRY": 3.*shared[0]+shared[1]+private[0],
        "SEMANTIC": -shared[0]+2.*shared[1]+private[1],
        "GRAPHICS": shared[2]-shared[1]+private[2],
    }
    policy = ProductMultitaskGradientPolicyV1(
        mode="BOUNDED_SHARED_CAGRAD_K",
        shared_owner_prefixes=("iris.scene.",),
        projection_strength=1., max_candidate_norm_ratio=1.5,
    )
    prepared = prepare_product_gradient_route_v1(
        named_parameters=(("iris.scene.shared",shared),("graphics.private",private)),
        task_losses=groups,policy=policy,global_step=0,
    )
    assert prepared is not None
    sum(groups.values()).backward()
    original_private = private.grad.detach().clone()
    report = prepared.apply_after_backward()
    assert report["mode"] == "BOUNDED_SHARED_CAGRAD_K"
    assert report["cagrad_diagnostics"] is not None
    assert report["task_private_gradients_preserved"] is True
    assert torch.equal(private.grad,original_private)
    assert report["baseline_shared_gradient_norm"]>0
    assert report["first_order_total_descent_preserved"] is True
    assert shared.grad.shape == (3,)


def test_cagrad_cannot_route_unqualified_or_disallowed_policy():
    with pytest.raises(ValueError, match="GRADIENT_POLICY_INVALID"):
        replace(ProductMultitaskGradientPolicyV1(),
                cagrad_conflict_aversion=1.3).validate()
    with pytest.raises(ValueError, match="GRADIENT_POLICY_INVALID"):
        replace(ProductMultitaskGradientPolicyV1(),
                mode="UNREVIEWED_CAGRAD").validate()
    p=ProductMultitaskGradientPolicyV1()
    p.validate()
    shared=torch.nn.Parameter(torch.tensor([.1]))
    result=prepare_product_gradient_route_v1(
        named_parameters=(("iris.scene.shared",shared),),
        task_losses={
            "GEOMETRY": shared.square().sum(),
            "GRAPHICS": (shared-1).square().sum(),
        },
        policy=p,global_step=0,
    )
    assert result is None


def test_backward_drift_rejected_before_gradient_overwrite():
    shared=torch.nn.Parameter(torch.tensor([1., 2.]))
    grp={
        "GEOMETRY": 2*shared[0],
        "GRAPHICS": 3*shared[1],
    }
    policy=ProductMultitaskGradientPolicyV1(
        mode="BOUNDED_SHARED_CAGRAD_K",
    )
    prepared=prepare_product_gradient_route_v1(
        named_parameters=(("iris.scene.x",shared),),
        task_losses=grp,policy=policy,global_step=0,
    )
    assert prepared is not None
    (sum(grp.values())+shared[0]).backward()
    before=shared.grad.detach().clone()
    with pytest.raises(RuntimeError, match="BACKWARD_GRAPH_DRIFT"):
        prepared.apply_after_backward()
    assert torch.equal(shared.grad,before)
