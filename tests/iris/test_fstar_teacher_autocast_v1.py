from __future__ import annotations

import numpy as np
import torch

from models.iris.training.apparatus_v1 import targets
from models.iris.v5.source_signed_field_v5 import source_signed_field_numpy_v5, SourceSignedFieldPolicyV5


def _subject(resolution: int = 1024):
    field = np.zeros((8, resolution, resolution), dtype=np.float32)
    yy, xx = np.meshgrid(
        np.arange(resolution, dtype=np.float32),
        np.arange(resolution, dtype=np.float32),
        indexing="ij",
    )
    # Distinct smooth field so exact edge sampling can be compared to numpy.
    for view in range(8):
        field[view] = (0.001 * xx + 0.0005 * yy + 0.01 * view - 0.2).astype(np.float32)
    origins = np.zeros((8, 3), dtype=np.float32)
    right = np.tile(np.asarray([[1.0, 0.0, 0.0]], np.float32), (8, 1))
    up = np.tile(np.asarray([[0.0, 1.0, 0.0]], np.float32), (8, 1))
    half = np.ones((8,), dtype=np.float32)
    return {
        "signed_2d": torch.from_numpy(field),
        "center_xyz": torch.zeros(3, dtype=torch.float32),
        "normalization_half_extent": 1.0,
        "camera_origins": torch.from_numpy(origins),
        "camera_right": torch.from_numpy(right),
        "camera_screen_up": torch.from_numpy(up),
        "camera_half_extent": torch.from_numpy(half),
        "resolution": resolution,
    }


def test_exact_fstar_teacher_disables_inherited_bf16_autocast_at_r1024_edge():
    subject = _subject(1024)
    points = np.asarray(
        [
            [1.0, -1.0, 0.0],
            [0.999, -0.999, 0.0],
            [-0.999, 0.999, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    policy = SourceSignedFieldPolicyV5(required_resolution=1024)
    expected = source_signed_field_numpy_v5(
        points,
        signed_2d=subject["signed_2d"].numpy(),
        center_xyz=subject["center_xyz"].numpy(),
        normalization_half_extent=subject["normalization_half_extent"],
        camera_origins=subject["camera_origins"].numpy(),
        camera_right=subject["camera_right"].numpy(),
        camera_screen_up=subject["camera_screen_up"].numpy(),
        camera_half_extent=subject["camera_half_extent"].numpy(),
        policy=policy,
    )

    # Reproduce the caller condition from IRIS V3.1: the model owns an outer
    # BF16 autocast scope, but exact teacher target generation must remain FP32.
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=True):
        actual = targets(subject, points, device="cpu", torch_authority=True)

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2e-7)
