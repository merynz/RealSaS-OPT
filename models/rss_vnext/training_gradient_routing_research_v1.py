from __future__ import annotations

"""Reference two-task shared-gradient routers for RSS.vNext *research only*.

Algorithms are compared on the exact FULL shared owner parameter vector,
never on the diagnostic court's sampled tensors. Expert-private gradients are
outside this function and MUST retain the normal source-qualified backward.

Names denote their precise TWO-task variants, not a general K-task theorem:
WEIGHTED_SUM: the current trainer update;
PCGRAD_2: symmetric, independently projected conflicting task gradients;
MGDA_2: minimum-norm convex mixture of the two gradients;
CAGRAD_BALL_2: exact maximin first-order direction in an L2 ball centered
  on the current sum, solved by enumerating two linear regions and their
  equality ridge. This solves the CAGrad *primal objective for K=2*, not a
  general K-task dual solver nor a promise about actual AdamW or held-out risk;
FREEZE_SHARED: zero shared-gradient negative control.

This code is intentionally NOT wired into the active trainer optimizer.
Only a matched-compute controlled ablation may request a different route.
"""

from dataclasses import dataclass
from math import isfinite, sqrt

import torch
from torch import Tensor


SCHEMA = "RealSaS.RSSvNext.TwoTaskSharedGradientRoutingResearch.v1"
ROUTES = (
    "WEIGHTED_SUM",
    "PCGRAD_2",
    "MGDA_2",
    "CAGRAD_BALL_2",
    "FREEZE_SHARED",
)
_EPS = 1.e-14


@dataclass(frozen=True)
class RSSvNextTwoTaskGradientRouteV1:
    schema: str
    mode: str
    candidate: Tensor
    baseline_norm: float
    candidate_norm: float
    cosine_with_baseline: float | None
    first_order_task_dots: tuple[float, float]
    baseline_task_dots: tuple[float, float]
    first_order_adverse_task_count: int
    cagrad_ball_radius: float | None
    cagrad_ball_distance: float | None
    source_or_product_qualified: bool = False


def _l2(v: Tensor) -> float:
    value = float(torch.linalg.vector_norm(v).item())
    if not isfinite(value):
        raise RuntimeError("RSS_VNEXT_ROUTING_NONFINITE_NORM")
    return value


def _unit(v: Tensor) -> Tensor:
    n = _l2(v)
    return v / n if n > _EPS else torch.zeros_like(v)


def _cagrad_two_task_exact(g1: Tensor, g2: Tensor, base: Tensor, fraction: float) -> Tensor:
    """Exact 2-task solution of max_d min_i <g_i,d>, ||d-base||<=c||base||.

    The minimum of two linear forms is piecewise-linear concave. A maximum
    on a Euclidean ball either maximizes a single active plane, or lies on
    their equality hyperplane. Enumerating these cases is complete in
    dimension >=2; no approximate iterative dual solver or hidden seed.
    """
    r = fraction * _l2(base)
    if r <= _EPS:
        return base.clone()
    options = [base + r * _unit(g1), base + r * _unit(g2), base.clone()]
    q = g1 - g2
    q2 = float(torch.dot(q, q).item())
    if q2 > _EPS:
        correction = -(torch.dot(q, base) / q2) * q
        orth_dist = _l2(correction)
        if orth_dist <= r + 1.e-10 * max(1., r):
            orth = g1 - torch.dot(g1, q) / q2 * q
            remaining = sqrt(max(0., r*r - orth_dist*orth_dist))
            ridge = base + correction + remaining * _unit(orth)
            options.append(ridge)
    def score(d: Tensor) -> float:
        return min(float(torch.dot(g1,d).item()),
                   float(torch.dot(g2,d).item()))
    best = max(options, key=score)
    if _l2(best - base) > r + 1.e-9 * max(1.,r):
        raise RuntimeError("RSS_VNEXT_CAGRAD_BALL_CONSTRAINT_DRIFT")
    return best


def route_two_rss_vnext_task_gradients_v1(
    task_a: Tensor, task_b: Tensor, *,
    mode: str,
    cagrad_radius_fraction: float = .5,
) -> RSSvNextTwoTaskGradientRouteV1:
    """Route ALREADY weighted flattened task gradients of one shared owner.

    No gradients are taken or mutated here; no task-private layers touched.
    The baseline uses the exact existing weighted SUM, not an average.
    """
    if (
        mode not in ROUTES
        or type(cagrad_radius_fraction) is not float
        or not isfinite(cagrad_radius_fraction)
        or not 0. <= cagrad_radius_fraction < 1.
        or not isinstance(task_a, Tensor)
        or not isinstance(task_b, Tensor)
        or task_a.ndim != 1 or task_b.ndim != 1
        or task_a.shape != task_b.shape or task_a.numel() < 1
        or not torch.is_floating_point(task_a)
        or task_a.device != task_b.device or task_a.dtype != task_b.dtype
        or not bool(torch.isfinite(task_a).all())
        or not bool(torch.isfinite(task_b).all())
    ):
        raise ValueError("RSS_VNEXT_TWO_TASK_ROUTE_INPUT_DOMAIN")
    source_device = task_a.device
    source_dtype = task_a.dtype
    a = task_a.detach().to(device="cpu",dtype=torch.float64)
    b = task_b.detach().to(device="cpu",dtype=torch.float64)
    base = a+b
    if mode == "WEIGHTED_SUM":
        d=base.clone()
    elif mode == "FREEZE_SHARED":
        d=torch.zeros_like(base)
    elif mode == "PCGRAD_2":
        dot=torch.dot(a,b)
        a_projected = (a - dot / torch.dot(b,b).clamp_min(_EPS) * b
                       if dot < 0. else a)
        b_projected = (b - dot / torch.dot(a,a).clamp_min(_EPS) * a
                       if dot < 0. else b)
        d=a_projected+b_projected
    elif mode == "MGDA_2":
        delta = a-b
        sq=float(torch.dot(delta,delta).item())
        w = (
            max(0., min(1., -float(torch.dot(b,delta).item()) / sq))
            if sq > _EPS else .5
        )
        d=w*a+(1.-w)*b
    else:
        d=_cagrad_two_task_exact(a,b,base,cagrad_radius_fraction)
    if not bool(torch.isfinite(d).all()):
        raise RuntimeError("RSS_VNEXT_TWO_TASK_ROUTE_NONFINITE")
    dot_a=float(torch.dot(a,d).item())
    dot_b=float(torch.dot(b,d).item())
    bn=_l2(base)
    dn=_l2(d)
    cosine=(
        max(-1.,min(1.,float(torch.dot(base,d).item())/(bn*dn)))
        if bn>_EPS and dn>_EPS else None
    )
    return RSSvNextTwoTaskGradientRouteV1(
        schema=SCHEMA,mode=mode,
        candidate=d.to(device=source_device,dtype=source_dtype),
        baseline_norm=bn,candidate_norm=dn,
        cosine_with_baseline=cosine,
        first_order_task_dots=(dot_a,dot_b),
        baseline_task_dots=(float(torch.dot(a,base).item()),
                            float(torch.dot(b,base).item())),
        first_order_adverse_task_count=int(dot_a < -1e-10)+int(dot_b < -1e-10),
        cagrad_ball_radius=(cagrad_radius_fraction*bn
                            if mode=="CAGRAD_BALL_2" else None),
        cagrad_ball_distance=(_l2(d-base)
                              if mode=="CAGRAD_BALL_2" else None),
    )


__all__ = [
    "SCHEMA", "ROUTES", "RSSvNextTwoTaskGradientRouteV1",
    "route_two_rss_vnext_task_gradients_v1",
]
