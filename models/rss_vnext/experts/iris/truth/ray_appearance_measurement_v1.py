"""Measured per-ray appearance factorization from renderer interventions.

This module does not infer material alpha/RGB from node graphs.  A renderer
adapter must independently measure, at one declared sampling footprint:

- every hit rendered standalone,
- the full ordered stack,
- the stack with each hit removed.

The standalone measurements propose one scalar effective alpha and straight
linear RGB per hit.  Those scalars are accepted only when ``ray_compositing_v1``
replays both the measured full result and every measured removal counterfactual
within the declared numeric tolerance.

This explicitly catches a dangerous raster assumption: an anti-aliased pixel can
contain correlated subpixel coverage for multiple surfaces, so in general
``E[a*b] != E[a]E[b]``.  Such a footprint cannot be silently collapsed to scalar
per-hit alpha just because standalone renders exist.

The result is a measurement/court result only.  It does not mint Truth
qualification, student supervision, training authority or product activation.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from models.iris.truth.ray_compositing_v1 import measure_ray


PASS = "PASS_MEASURED_RAY_APPEARANCE_FACTORIZATION"
FAIL_REPLAY = "FAIL_MEASURED_RAY_INTERVENTION_REPLAY"
ABSTAIN_INCOMPLETE = "ABSTAIN_INCOMPLETE_RAY_STACK"
ABSTAIN_TIE = "ABSTAIN_AMBIGUOUS_DEPTH_TIE"
ABSTAIN_FOOTPRINT = "ABSTAIN_RENDERER_FOOTPRINT_NONCONVERGED"
ABSTAIN_ZERO_ALPHA = "ABSTAIN_ZERO_ALPHA_STRAIGHT_COLOUR_UNIDENTIFIED"


@dataclass(frozen=True)
class RayAppearanceMeasurementV1:
    status: str
    valid: bool
    effective_alpha: np.ndarray
    straight_linear_rgb: np.ndarray
    incoming_transmittance: np.ndarray
    contribution: np.ndarray
    predicted_full_alpha: float
    predicted_full_premultiplied_linear_rgb: np.ndarray
    maximum_full_alpha_error: float
    maximum_full_rgb_error: float
    maximum_removal_alpha_error: float
    maximum_removal_rgb_error: float
    schema_version: str = "RealSaS.RayAppearanceMeasurement.v1"


def _finite_vector(value, shape, code: str) -> np.ndarray:
    out = np.asarray(value, dtype=np.float64)
    if out.shape != shape or not np.isfinite(out).all():
        raise ValueError(code)
    return out


def _abstain(status: str, count: int) -> RayAppearanceMeasurementV1:
    return RayAppearanceMeasurementV1(
        status=status,
        valid=False,
        effective_alpha=np.full(count, np.nan),
        straight_linear_rgb=np.full((count, 3), np.nan),
        incoming_transmittance=np.full(count, np.nan),
        contribution=np.full(count, np.nan),
        predicted_full_alpha=float("nan"),
        predicted_full_premultiplied_linear_rgb=np.full(3, np.nan),
        maximum_full_alpha_error=float("nan"),
        maximum_full_rgb_error=float("nan"),
        maximum_removal_alpha_error=float("nan"),
        maximum_removal_rgb_error=float("nan"),
    )


def qualify_measured_ray_appearance_v1(
    depth,
    standalone_alpha,
    standalone_premultiplied_linear_rgb,
    measured_full_alpha,
    measured_full_premultiplied_linear_rgb,
    measured_removal_alpha,
    measured_removal_premultiplied_linear_rgb,
    *,
    complete: bool,
    tie_epsilon: float,
    measurement_tolerance: float,
    footprint_converged: bool,
    alpha_identification_floor: float = 1.0e-8,
) -> RayAppearanceMeasurementV1:
    """Qualify scalar per-hit appearance only by measured renderer consequences.

    ``measured_removal_*[i]`` is the render of the complete stack with hit ``i``
    removed.  RGB carriers are premultiplied linear RGB; standalone straight RGB
    is therefore identified as ``premultiplied / alpha`` only for strictly
    positive measured alpha above ``alpha_identification_floor``.

    A zero-alpha geometric hit is not fabricated into a coloured transparent hit;
    this V1 court abstains because ``ray_compositing_v1`` currently requires a
    finite straight colour for every admitted per-hit composite row.
    """
    z = np.asarray(depth, dtype=np.float64)
    if z.ndim != 1 or not np.isfinite(z).all() or np.any(np.diff(z) < 0):
        raise ValueError("RAY_APPEARANCE_FINITE_ASCENDING_DEPTH_REQUIRED")
    n = len(z)
    alpha = _finite_vector(standalone_alpha, (n,), "RAY_APPEARANCE_STANDALONE_ALPHA_LAYOUT")
    premul = _finite_vector(
        standalone_premultiplied_linear_rgb,
        (n, 3),
        "RAY_APPEARANCE_STANDALONE_RGB_LAYOUT",
    )
    full_rgb = _finite_vector(
        measured_full_premultiplied_linear_rgb,
        (3,),
        "RAY_APPEARANCE_FULL_RGB_LAYOUT",
    )
    removal_alpha = _finite_vector(
        measured_removal_alpha,
        (n,),
        "RAY_APPEARANCE_REMOVAL_ALPHA_LAYOUT",
    )
    removal_rgb = _finite_vector(
        measured_removal_premultiplied_linear_rgb,
        (n, 3),
        "RAY_APPEARANCE_REMOVAL_RGB_LAYOUT",
    )
    if type(complete) is not bool or type(footprint_converged) is not bool:
        raise ValueError("RAY_APPEARANCE_BOOLEAN_CONTRACT_REQUIRED")
    if (
        not math.isfinite(tie_epsilon)
        or tie_epsilon < 0
        or not math.isfinite(measurement_tolerance)
        or measurement_tolerance < 0
        or not math.isfinite(alpha_identification_floor)
        or alpha_identification_floor <= 0
    ):
        raise ValueError("RAY_APPEARANCE_TOLERANCE_CONTRACT")
    full_alpha = float(measured_full_alpha)
    if not math.isfinite(full_alpha):
        raise ValueError("RAY_APPEARANCE_FULL_ALPHA_FINITE")
    for values, code in (
        (alpha, "RAY_APPEARANCE_STANDALONE_ALPHA_RANGE"),
        (removal_alpha, "RAY_APPEARANCE_REMOVAL_ALPHA_RANGE"),
        (np.asarray([full_alpha]), "RAY_APPEARANCE_FULL_ALPHA_RANGE"),
    ):
        if np.any((values < 0) | (values > 1)):
            raise ValueError(code)

    if not footprint_converged:
        return _abstain(ABSTAIN_FOOTPRINT, n)
    if not complete:
        return _abstain(ABSTAIN_INCOMPLETE, n)
    if np.any(np.diff(z) <= tie_epsilon):
        return _abstain(ABSTAIN_TIE, n)
    if np.any(alpha <= alpha_identification_floor):
        return _abstain(ABSTAIN_ZERO_ALPHA, n)

    straight = premul / alpha[:, None]
    if not np.isfinite(straight).all():
        raise ValueError("RAY_APPEARANCE_STRAIGHT_RGB_NONFINITE")

    predicted = measure_ray(
        z,
        alpha,
        straight,
        complete=True,
        tie_epsilon=tie_epsilon,
        blend_mode="SOURCE_OVER",
    )
    if not (predicted.alpha_valid and predicted.per_hit_valid and predicted.composite_valid):
        raise ValueError("RAY_APPEARANCE_INTERNAL_COMPOSITOR_REJECTED_ADMITTED_INPUT")

    full_alpha_error = abs(float(predicted.alpha) - full_alpha)
    full_rgb_error = float(np.max(np.abs(predicted.premultiplied_rgb - full_rgb)))

    removal_alpha_error = 0.0
    removal_rgb_error = 0.0
    for index in range(n):
        keep = np.arange(n) != index
        counterfactual = measure_ray(
            z[keep],
            alpha[keep],
            straight[keep],
            complete=True,
            tie_epsilon=tie_epsilon,
            blend_mode="SOURCE_OVER",
        )
        if not (counterfactual.alpha_valid and counterfactual.composite_valid):
            raise ValueError("RAY_APPEARANCE_INTERNAL_REMOVAL_COMPOSITOR_REJECTED")
        removal_alpha_error = max(
            removal_alpha_error,
            abs(float(counterfactual.alpha) - float(removal_alpha[index])),
        )
        removal_rgb_error = max(
            removal_rgb_error,
            float(np.max(np.abs(counterfactual.premultiplied_rgb - removal_rgb[index]))),
        )

    valid = max(
        full_alpha_error,
        full_rgb_error,
        removal_alpha_error,
        removal_rgb_error,
    ) <= measurement_tolerance
    return RayAppearanceMeasurementV1(
        status=PASS if valid else FAIL_REPLAY,
        valid=bool(valid),
        effective_alpha=alpha.copy(),
        straight_linear_rgb=straight.copy(),
        incoming_transmittance=predicted.transmittance.copy(),
        contribution=predicted.contribution.copy(),
        predicted_full_alpha=float(predicted.alpha),
        predicted_full_premultiplied_linear_rgb=predicted.premultiplied_rgb.copy(),
        maximum_full_alpha_error=float(full_alpha_error),
        maximum_full_rgb_error=float(full_rgb_error),
        maximum_removal_alpha_error=float(removal_alpha_error),
        maximum_removal_rgb_error=float(removal_rgb_error),
    )
