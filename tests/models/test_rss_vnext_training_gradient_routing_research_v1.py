from __future__ import annotations

from math import cos, pi, sin

import pytest
import torch

from models.rss_vnext.training_gradient_routing_research_v1 import (
    ROUTES, route_two_rss_vnext_task_gradients_v1,
)


@pytest.fixture(autouse=True)
def precision_and_seed():
    torch.manual_seed(151)
    torch.set_num_threads(1)


def route(a,b,mode,c=.5):
    return route_two_rss_vnext_task_gradients_v1(
        torch.tensor(a,dtype=torch.float64),
        torch.tensor(b,dtype=torch.float64),
        mode=mode,cagrad_radius_fraction=float(c),
    )


def test_exact_unopposed_sum_and_pcgrad_leave_gradient_unchanged():
    a=[2.,0.];b=[0.,3.]
    baseline=route(a,b,"WEIGHTED_SUM")
    pc=route(a,b,"PCGRAD_2")
    assert torch.equal(baseline.candidate,torch.tensor([2.,3.]))
    assert torch.equal(pc.candidate,baseline.candidate)
    assert baseline.first_order_adverse_task_count==0
    assert baseline.cagrad_ball_radius is None


def test_pcgrad_projects_conflicting_grads_and_never_changes_inputs():
    a=torch.tensor([2.,0.,1.],requires_grad=True)
    b=torch.tensor([-1.,1.,0.],requires_grad=True)
    olda=a.detach().clone();oldb=b.detach().clone()
    result=route_two_rss_vnext_task_gradients_v1(a,b,mode="PCGRAD_2")
    assert torch.equal(a,olda) and torch.equal(b,oldb)
    assert a.grad is None and b.grad is None
    x=olda.double();y=oldb.double();dot=torch.dot(x,y)
    expected=(x-dot/torch.dot(y,y)*y)+(y-dot/torch.dot(x,x)*x)
    assert torch.allclose(result.candidate.double(),expected,atol=1e-6)
    assert result.mode=="PCGRAD_2"
    assert not result.source_or_product_qualified


def test_mgda_two_task_closed_form_minimizes_convex_hull_norm():
    a=torch.tensor([3.,0.]);b=torch.tensor([0.,1.])
    m=route_two_rss_vnext_task_gradients_v1(a,b,mode="MGDA_2")
    # min ||alpha*[3,0] + (1-alpha)*[0,1]|| => alpha=1/10.
    assert m.candidate.tolist()==pytest.approx([.3,.9],abs=1e-6)
    assert m.candidate_norm<=min(float(a.norm()),float(b.norm()))
    for alpha in range(101):
        trial=(alpha/100)*a+(1-alpha/100)*b
        assert float(trial.norm())+1e-5>=m.candidate_norm


@pytest.mark.parametrize("a,b",[
    ([1.,0.],[-.4,1.]),
    ([1.,0.],[-2.,0.]),
    ([1.,2.],[1.,2.]),
    ([1.,0.],[0.,1.]),
    ([.5,3.],[-.75,-1.]),
    ([2.,2.],[-1.,-2.]),
])
@pytest.mark.parametrize("fraction",[0.,.1,.5,.9])
def test_cagrad_ball_exact_two_task_primal_against_dense_angle_search(a,b,fraction):
    result=route(a,b,"CAGRAD_BALL_2",fraction)
    x=torch.tensor(a,dtype=torch.float64)
    y=torch.tensor(b,dtype=torch.float64)
    base=x+y
    r=float(fraction*base.norm())
    direction=result.candidate.double()
    assert float((direction-base).norm())<=r+1e-9
    optimum=min(float(x@direction),float(y@direction))
    # Numeric global search on the disk boundary (and center), independent
    # of the solver's plane/ridge enumeration. Grid bound resolution 2pi/4096.
    sampled=[min(float(x@base),float(y@base))]
    for i in range(4096):
        theta=2*pi*i/4096
        d=base+r*torch.tensor([cos(theta),sin(theta)],dtype=torch.float64)
        sampled.append(min(float(x@d),float(y@d)))
    assert optimum>=max(sampled)-1e-6
    assert result.cagrad_ball_distance is not None
    assert result.cagrad_ball_distance<=result.cagrad_ball_radius+1e-9
    if fraction==0.:
        assert torch.equal(result.candidate,base)


def test_collinear_opposite_zero_sum_and_freeze_are_deterministic():
    for mode in ROUTES:
        r=route([1.,0.],[-1.,0.],mode)
        assert not bool(r.candidate.any())
        assert r.candidate_norm==0.
        assert r.cosine_with_baseline is None
    frozen=route([1.,1.],[0.,1.],"FREEZE_SHARED")
    assert not bool(frozen.candidate.any())
    assert frozen.first_order_adverse_task_count==0


@pytest.mark.parametrize("broken",[
    {"mode":"FUTURE_ALGORITHM"},
    {"mode":"CAGRAD_BALL_2","cagrad_radius_fraction":1.},
    {"mode":"CAGRAD_BALL_2","cagrad_radius_fraction":float("nan")},
])
def test_invalid_research_policy_fails_closed(broken):
    args={"mode":"WEIGHTED_SUM","cagrad_radius_fraction":.5}
    args.update(broken)
    with pytest.raises(ValueError,match="INPUT_DOMAIN"):
        route_two_rss_vnext_task_gradients_v1(
            torch.ones(4),torch.ones(4),**args,
        )


def test_invalid_or_mismatched_task_gradients_fail_closed():
    for a,b in [
        (torch.zeros(0),torch.zeros(0)),
        (torch.ones(3),torch.ones(4)),
        (torch.ones(3),torch.ones((1,3))),
        (torch.tensor([1.,float("nan")]),torch.ones(2)),
        (torch.ones(3,dtype=torch.float32),torch.ones(3,dtype=torch.float64)),
    ]:
        with pytest.raises(ValueError,match="INPUT_DOMAIN"):
            route_two_rss_vnext_task_gradients_v1(a,b,mode="WEIGHTED_SUM")
