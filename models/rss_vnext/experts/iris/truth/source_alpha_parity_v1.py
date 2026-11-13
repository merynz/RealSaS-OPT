from __future__ import annotations

"""Measured parity between native source alpha and unlit-preserve-alpha alpha.

The court is deliberately alpha-only. It does not claim RGB parity, source
lighting parity, sampler qualification, or compositing qualification. Background
agreement alone is never sufficient: at least one union-foreground sample must
be measured.
"""

from dataclasses import dataclass
import numpy as np

SCHEMA = "RealSaS.SourceAlphaTransportParity.v1"


@dataclass(frozen=True)
class AlphaParityMeasurementV1:
    valid: bool
    union_foreground_sample_count: int
    compared_sample_count: int
    maximum_absolute_error: float
    mean_absolute_error: float
    mismatch_count: int
    tolerance: float
    status: str
    schema_version: str = SCHEMA


def measure_alpha_transport_parity_v1(native_alpha, transformed_alpha, *, tolerance: float = 1.0e-6,
                                      foreground_epsilon: float = 1.0e-8) -> AlphaParityMeasurementV1:
    a = np.asarray(native_alpha, dtype=np.float64)
    b = np.asarray(transformed_alpha, dtype=np.float64)
    if a.shape != b.shape or a.ndim < 1 or a.size == 0:
        raise ValueError("SOURCE_ALPHA_PARITY_MATCHED_NONEMPTY_ARRAYS_REQUIRED")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("SOURCE_ALPHA_PARITY_FINITE_REQUIRED")
    if np.any((a < 0) | (a > 1)) or np.any((b < 0) | (b > 1)):
        raise ValueError("SOURCE_ALPHA_PARITY_RANGE")
    if not np.isfinite(tolerance) or tolerance < 0 or not np.isfinite(foreground_epsilon) or foreground_epsilon < 0:
        raise ValueError("SOURCE_ALPHA_PARITY_POLICY_INVALID")

    union = (a > foreground_epsilon) | (b > foreground_epsilon)
    count = int(np.count_nonzero(union))
    if count == 0:
        return AlphaParityMeasurementV1(
            valid=False,
            union_foreground_sample_count=0,
            compared_sample_count=0,
            maximum_absolute_error=float("inf"),
            mean_absolute_error=float("inf"),
            mismatch_count=0,
            tolerance=float(tolerance),
            status="ABSTAIN_EMPTY_FOREGROUND",
        )

    error = np.abs(a[union] - b[union])
    maximum = float(np.max(error))
    mean = float(np.mean(error))
    mismatches = int(np.count_nonzero(error > tolerance))
    status = "PASS_ALPHA_TRANSPORT_PARITY" if mismatches == 0 else "FAIL_ALPHA_TRANSPORT_DRIFT"
    return AlphaParityMeasurementV1(
        valid=mismatches == 0,
        union_foreground_sample_count=count,
        compared_sample_count=count,
        maximum_absolute_error=maximum,
        mean_absolute_error=mean,
        mismatch_count=mismatches,
        tolerance=float(tolerance),
        status=status,
    )
