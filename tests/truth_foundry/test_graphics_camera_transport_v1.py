import numpy as np
import pytest

from models.iris.truth.graphics_oracle_v1 import (
    CAMERA_BASIS_TRANSPORT_ABS_TOLERANCE,
    camera_basis_transport_residual,
    camera_from_mapping,
)


def _camera(forward, right, up):
    return {
        "origin": [0.0, 0.0, 5.0],
        "forward": list(forward),
        "right": list(right),
        "screen_up": list(up),
        "half_extent": 2.0,
        "resolution": 1024,
        "view_index": 0,
    }


def test_float32_serialized_orthonormal_basis_is_transported_without_normalization():
    a = float(np.float32(np.sqrt(0.5)))
    value = _camera((a, 0.0, -a), (a, 0.0, a), (0.0, 1.0, 0.0))
    residual = camera_basis_transport_residual(value)
    assert residual > 1.0e-8
    assert residual < CAMERA_BASIS_TRANSPORT_ABS_TOLERANCE
    camera = camera_from_mapping(value)
    # Transport admission must not silently repair or renormalize the sealed camera.
    assert camera.forward == tuple(value["forward"])
    assert camera.right == tuple(value["right"])
    assert camera.screen_up == tuple(value["screen_up"])


def test_real_basis_shear_still_fails_closed():
    value = _camera((0.0, 0.0, -1.0), (1.0, 1.0e-4, 0.0), (0.0, 1.0, 0.0))
    assert camera_basis_transport_residual(value) > CAMERA_BASIS_TRANSPORT_ABS_TOLERANCE
    with pytest.raises(ValueError, match="GRAPHICS_ORACLE_CAMERA_ORTHONORMAL_REQUIRED"):
        camera_from_mapping(value)
