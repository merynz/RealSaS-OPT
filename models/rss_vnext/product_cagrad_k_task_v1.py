from __future__ import annotations

"""Source-qualified K-task CAGrad research route for shared owner gradients.

Implements the *CAGrad NeurIPS 2021* small-simplex dual with SciPy SLSQP,
using the published official paper/code objective. This is a K-task option,
not the historical two-task ball variant or a claim of quality improvement.
Inputs are the ALREADY WEIGHTED, actual admitted loss-family gradients on
exact full shared-owner parameters. A single step cannot establish whether
CAGrad is a better optimizer for RSS.vNext: matched GPU/unseen ablation still
controls selection. Never route expert-private gradients or infer teachers.
"""

from dataclasses import dataclass
from typing import Mapping
import math

import numpy as np
from torch import Tensor
import torch

SCHEMA = "RealSaS.RSSvNext.AdmittedSharedKTaskCAGrad.v1"


@dataclass(frozen=True)
class CAGradSharedRouteV1:
    candidate: Tensor
    family_names: tuple[str, ...]
    mixture_weights: tuple[float, ...]
    c: float
    baseline_norm: float
    candidate_norm: float
    simplex_objective: float
    solver_iterations: int
    first_order_total_dot: float
    estimated_extra_solver_bytes: int
    source_authority: str = "ADMITTED_LOSS_GRADIENTS_ONLY_NOT_CORPUS_PROOF"
    product_release_qualified: bool = False


def k_task_admitted_shared_cagrad_v1(
    task_gradients: Mapping[str, Tensor], *,
    c: float = 0.5, solver_maxiter: int = 150,
) -> CAGradSharedRouteV1:
    """Official-style K-task dual solved on a small CPU Gram matrix.

    Normalization: the paper's average gradient g0 is used in the solver,
    while the final direction is multiplied by K to preserve the existing
    weighted-SUM magnitude convention. We use official 'rescale=1', namely
    /(1+c^2). The product gate may additionally blend/bound the candidate.
    """
    if (
        type(task_gradients) is not dict
        or not 2 <= len(task_gradients) <= 9
        or type(c) not in (int, float) or not math.isfinite(c)
        or not 0 <= c < 1
        or type(solver_maxiter) is not int
        or not 20 <= solver_maxiter <= 500
    ):
        raise ValueError("RSS_VNEXT_CAGRAD_K_TASK_POLICY_INVALID")
    names = tuple(task_gradients)
    gradients = tuple(task_gradients.values())
    if (
        len(set(names)) != len(names)
        or not all(type(n) is str and n for n in names)
        or not all(isinstance(g, Tensor) and g.ndim == 1 and
                   torch.is_floating_point(g) and g.numel()>0
                   for g in gradients)
        or any(g.shape != gradients[0].shape
               or g.dtype != gradients[0].dtype
               or g.device != gradients[0].device
               or not bool(torch.isfinite(g).all()) for g in gradients)
    ):
        raise ValueError("RSS_VNEXT_CAGRAD_K_TASK_GRADIENT_DOMAIN")
    n = len(gradients)
    # K^2 dot-products without allocating a second KxP dense matrix.
    grams = np.asarray([
        [float(torch.dot(a,b).item()) for b in gradients]
        for a in gradients
    ], dtype=np.float64)
    grams = .5 * (grams + grams.T)
    scale = max(float(np.max(np.abs(grams))), 1e-20)
    matrix = grams / scale
    uniform = np.full(n, 1./n, dtype=np.float64)
    g0norm = math.sqrt(max(0., float(uniform @ matrix @ uniform)))
    if g0norm == 0.:
        raise RuntimeError("RSS_VNEXT_CAGRAD_ZERO_SHARED_TOTAL_GRADIENT")
    phi = float(c) * g0norm

    def objective(weights: np.ndarray) -> float:
        mean_alignment = float(weights @ matrix @ uniform)
        mixed_norm = math.sqrt(max(0., float(weights @ matrix @ weights)) + 1e-12)
        return mean_alignment + phi*mixed_norm

    try:
        from scipy.optimize import minimize
    except ImportError as exc:
        raise RuntimeError("RSS_VNEXT_CAGRAD_SCIPY_SOLVER_REQUIRED") from exc
    solved = minimize(
        objective, x0=uniform,
        method="SLSQP",
        bounds=[(0.,1.) for _ in range(n)],
        constraints=[{"type":"eq", "fun":lambda x:float(np.sum(x)-1.)}],
        options={"maxiter":solver_maxiter, "ftol":1e-11, "disp":False},
    )
    w = np.asarray(solved.x, dtype=np.float64)
    if (
        not solved.success or not np.all(np.isfinite(w))
        or np.any(w < -1e-7) or np.any(w > 1+1e-7)
        or abs(float(np.sum(w))-1.)>1e-7
        or not math.isfinite(float(solved.fun))
    ):
        raise RuntimeError("RSS_VNEXT_CAGRAD_SIMPLEX_SOLVER_FAILED")
    w = np.maximum(w,0.); w/=w.sum()
    base = sum(gradients[1:],gradients[0].clone())
    mixed = sum(
        (float(w[i])*gradients[i] for i in range(1,n)),
        float(w[0])*gradients[0],
    )
    mixed_norm = torch.linalg.vector_norm(mixed)
    base_norm = torch.linalg.vector_norm(base)
    if (
        not bool(torch.isfinite(mixed_norm))
        or not bool(torch.isfinite(base_norm))
        or not bool(mixed_norm > 0)
        or not bool(base_norm > 0)
    ):
        raise RuntimeError("RSS_VNEXT_CAGRAD_DEGENERATE_SHARED_GRADIENT")
    c_real = float(c) * base_norm / n
    candidate = (
        base + (n*c_real/mixed_norm)*mixed
    ) / (1 + float(c)**2)
    alignment = torch.dot(base, candidate)
    if not bool(torch.isfinite(candidate).all()):
        raise RuntimeError("RSS_VNEXT_CAGRAD_NONFINITE_DIRECTION")
    return CAGradSharedRouteV1(
        candidate=candidate,
        family_names=names,
        mixture_weights=tuple(float(a) for a in w),
        c=float(c),
        baseline_norm=float(base_norm.item()),
        candidate_norm=float(torch.linalg.vector_norm(candidate).item()),
        simplex_objective=float(solved.fun),
        solver_iterations=int(solved.nit),
        first_order_total_dot=float(alignment.item()),
        estimated_extra_solver_bytes=8*n*n*3,
    )


__all__ = [
    "SCHEMA", "CAGradSharedRouteV1", "k_task_admitted_shared_cagrad_v1",
]
