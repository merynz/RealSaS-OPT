"""Complete-ray source-over measurements; no tie-breaking by teacher face ID."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class RayMeasurement:
    transmittance: np.ndarray
    contribution: np.ndarray
    premultiplied_rgb: np.ndarray
    alpha: float
    per_hit_valid: bool
    composite_valid: bool
    alpha_valid: bool
    reason: str


def measure_ray(depth, alpha, linear_rgb, *, complete, tie_epsilon, blend_mode="SOURCE_OVER"):
    """Depth ascending; straight linear RGB; transparent background.

    Coverage must already be resolved at the recorded sampling footprint. NaN
    alpha means unknown, never opaque. Any tie conservatively invalidates RGB
    and per-hit weights; total alpha remains known for a complete known stack.
    Unsupported blend laws fail closed. No validity is inferred from RGB delta.
    """
    z, a, c = (np.asarray(x, dtype=np.float64) for x in (depth, alpha, linear_rgb))
    if z.ndim != 1 or a.shape != z.shape or c.shape != (len(z), 3):
        raise ValueError("matched [hit], [hit], [hit,3] arrays required")
    if not isinstance(complete, bool) or not np.isfinite(tie_epsilon) or tie_epsilon < 0:
        raise ValueError("explicit completeness and nonnegative source-unit tolerance required")
    if not np.isfinite(z).all() or np.any(np.diff(z) < 0):
        raise ValueError("finite ascending source depths required")
    known_alpha = np.isfinite(a).all()
    if np.isinf(a).any() or np.any((a < 0) | (a > 1)):
        raise ValueError("alpha outside [0,1]")
    if np.isinf(c).any():
        raise ValueError("infinite colour")
    supported = blend_mode == "SOURCE_OVER"
    alpha_valid = complete and known_alpha and supported
    total_alpha = float(1 - np.prod(1 - a)) if alpha_valid else float("nan")
    reason = ("INCOMPLETE_STACK" if not complete else
              "UNSUPPORTED_BLEND" if not supported else
              "UNKNOWN_ALPHA" if not known_alpha else
              "AMBIGUOUS_DEPTH_TIE" if np.any(np.diff(z) <= tie_epsilon) else
              "UNKNOWN_COLOUR" if not np.isfinite(c).all() else "MEASURED")
    # Unknown colour does not invalidate geometric alpha/transmittance weights.
    hit_valid = reason in ("MEASURED", "UNKNOWN_COLOUR")
    if hit_valid:
        transmittance = np.concatenate(([1.0], np.cumprod(1 - a)))[:-1]
        weights = transmittance * a
    else:
        transmittance = np.full_like(a, np.nan)
        weights = np.full_like(a, np.nan)
    rgb_valid = reason == "MEASURED"
    rgb = np.sum(weights[:, None] * c, axis=0) if rgb_valid else np.full(3, np.nan)
    return RayMeasurement(transmittance, weights, rgb, total_alpha,
                          hit_valid, rgb_valid, alpha_valid, reason)


def removal_rgb_delta(depth, alpha, linear_rgb, remove_index, **kwargs):
    """Recompose the surviving complete stack. Zero delta is not no occlusion."""
    n = len(depth)
    if isinstance(remove_index, bool) or not isinstance(remove_index, (int, np.integer)) or not 0 <= remove_index < n:
        raise ValueError("valid hit index required")
    base = measure_ray(depth, alpha, linear_rgb, **kwargs)
    keep = np.arange(n) != remove_index
    changed = measure_ray(np.asarray(depth)[keep], np.asarray(alpha)[keep],
                          np.asarray(linear_rgb)[keep], **kwargs)
    valid = base.composite_valid and changed.composite_valid
    return (base.premultiplied_rgb - changed.premultiplied_rgb if valid
            else np.full(3, np.nan)), valid
