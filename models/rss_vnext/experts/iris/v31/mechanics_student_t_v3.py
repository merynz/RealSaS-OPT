"""Heavy-tail mechanics likelihood for IRIS V3.1.

The Stage48 mechanics truth is retained exactly. Targets are transformed only by
log1p on non-negative magnitudes, as in the preceding capacity pilots. This
objective changes the residual likelihood from Gaussian to fixed-df Student-t so
rare valid tail observations cannot dominate quadratically while remaining
fully visible to the learner.
"""
from __future__ import annotations

import math

import torch

from .presentation_objective_v1 import MECHANICAL_CHANNELS_V31_PRESENTATION


def student_t_mechanical_loss_v3(
    raw: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    *,
    degrees_of_freedom: float = 3.0,
    minimum_log_variance: float = -6.0,
    maximum_log_variance: float = 3.0,
) -> torch.Tensor:
    """Exact fixed-df Student-t NLL for two conditional mechanics magnitudes.

    ``raw`` keeps the existing four-channel contract:
      mean_0, log_variance_0, mean_1, log_variance_1.

    The scale is therefore ``sqrt(exp(log_variance))``.  With fixed df, the
    Student-t likelihood has logarithmic rather than quadratic tail growth.
    No target clipping, deletion, winsorization, subject-specific weighting or
    hidden validity rule is applied here.
    """

    if (
        raw.shape[-1] != len(MECHANICAL_CHANNELS_V31_PRESENTATION)
        or target.shape != raw.shape[:-1] + (2,)
        or valid.shape != raw.shape[:-1]
    ):
        raise ValueError("IRIS_V31_STUDENT_T_MECHANICAL_LOSS_SHAPE")
    if not math.isfinite(float(degrees_of_freedom)) or float(degrees_of_freedom) <= 2.0:
        raise ValueError("IRIS_V31_STUDENT_T_DF_MUST_HAVE_FINITE_VARIANCE")
    if not (
        math.isfinite(float(minimum_log_variance))
        and math.isfinite(float(maximum_log_variance))
        and float(minimum_log_variance) < float(maximum_log_variance)
    ):
        raise ValueError("IRIS_V31_STUDENT_T_LOGVAR_BOUNDS")

    mask = valid.bool()
    if not bool(mask.any()):
        raise ValueError("IRIS_V31_STUDENT_T_MECHANICAL_LOSS_EMPTY")

    mean = torch.stack((raw[..., 0], raw[..., 2]), dim=-1)
    log_variance = torch.stack((raw[..., 1], raw[..., 3]), dim=-1).clamp(
        float(minimum_log_variance), float(maximum_log_variance)
    )
    residual = target.to(mean) - mean
    nu = mean.new_tensor(float(degrees_of_freedom))

    # NLL for Student-t(location=mean, scale=sqrt(exp(log_variance)), df=nu).
    # The normalization term is retained so logged values are true NLLs, not a
    # shifted optimization surrogate.
    constant = (
        0.5 * torch.log(nu * mean.new_tensor(math.pi))
        + torch.lgamma(0.5 * nu)
        - torch.lgamma(0.5 * (nu + 1.0))
    )
    scaled_square = residual.square() * torch.exp(-log_variance) / nu
    nll = (
        constant
        + 0.5 * log_variance
        + 0.5 * (nu + 1.0) * torch.log1p(scaled_square)
    )
    return nll[mask].mean()
