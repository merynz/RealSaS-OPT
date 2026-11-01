from __future__ import annotations

import numpy as np
import pytest

from tools.corpus.build_gltf_rig_skin_authority_v1 import _quat_matrix as quat_matrix_v1
from tools.corpus.build_gltf_rig_skin_authority_v2 import _quat_matrix as quat_matrix_v2


def test_v2_normalizes_non_unit_gltf_rotation_before_matrix_construction():
    # Same rotation as [0, 0, sqrt(1/2), sqrt(1/2)] but deliberately scaled.
    q = np.asarray([0.0, 0.0, 0.7, 0.7], dtype=np.float64)
    normalized = q / np.linalg.norm(q)
    expected = quat_matrix_v1(normalized)
    actual = quat_matrix_v2(q)

    assert np.allclose(actual, expected, atol=1e-14, rtol=0)
    assert np.allclose(actual[:3, :3].T @ actual[:3, :3], np.eye(3), atol=1e-14, rtol=0)
    assert np.isclose(np.linalg.det(actual[:3, :3]), 1.0, atol=1e-14, rtol=0)
    # Regression proof: V1 leaked quaternion magnitude into the matrix.
    assert not np.allclose(quat_matrix_v1(q), expected, atol=1e-12, rtol=0)


def test_v2_rejects_zero_norm_quaternion_fail_closed():
    with pytest.raises(ValueError, match="GLTF_RIG_SKIN_V2_ROTATION_QUATERNION_ZERO_NORM"):
        quat_matrix_v2([0.0, 0.0, 0.0, 0.0])


def test_v2_rejects_nonfinite_quaternion_fail_closed():
    with pytest.raises(ValueError, match="GLTF_RIG_SKIN_V2_ROTATION_QUATERNION"):
        quat_matrix_v2([0.0, 0.0, np.nan, 1.0])
