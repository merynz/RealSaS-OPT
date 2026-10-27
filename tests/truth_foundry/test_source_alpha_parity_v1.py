from __future__ import annotations

import numpy as np
import pytest

from models.iris.truth.source_alpha_parity_v1 import measure_alpha_transport_parity_v1


def test_exact_foreground_passes():
    native = np.array([[0.0, 0.25], [1.0, 0.0]], dtype=np.float64)
    got = measure_alpha_transport_parity_v1(native, native.copy())
    assert got.valid is True
    assert got.status == "PASS_ALPHA_TRANSPORT_PARITY"
    assert got.union_foreground_sample_count == 2
    assert got.maximum_absolute_error == 0.0
    assert got.mismatch_count == 0


def test_drift_above_preregistered_tolerance_fails():
    native = np.array([0.0, 0.5, 1.0], dtype=np.float64)
    changed = np.array([0.0, 0.500002, 1.0], dtype=np.float64)
    got = measure_alpha_transport_parity_v1(native, changed, tolerance=1.0e-6)
    assert got.valid is False
    assert got.status == "FAIL_ALPHA_TRANSPORT_DRIFT"
    assert got.mismatch_count == 1


def test_empty_background_abstains_instead_of_passing():
    got = measure_alpha_transport_parity_v1(np.zeros(8), np.zeros(8))
    assert got.valid is False
    assert got.status == "ABSTAIN_EMPTY_FOREGROUND"
    assert got.compared_sample_count == 0


def test_invalid_alpha_is_rejected():
    with pytest.raises(ValueError, match="SOURCE_ALPHA_PARITY_RANGE"):
        measure_alpha_transport_parity_v1(np.array([1.1]), np.array([1.0]))


def test_shape_drift_is_rejected():
    with pytest.raises(ValueError, match="MATCHED_NONEMPTY"):
        measure_alpha_transport_parity_v1(np.zeros((2, 2)), np.zeros((4,)))
