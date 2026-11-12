from __future__ import annotations

"""Exact FRONT/BACK source-ray camera parity with Canonical Runtime projection.

A camera's declared 'up' vector points to image TOP in the runtime projector;
torch.grid_sample normalized y points DOWN. This must be the same original
pixel, including on non-square and rotated calibrated camera frames. Synthetic
RGBA gradients are observation fixtures, not source-depth or graphics proof.
"""

import math

import numpy as np
import pytest
import torch

from compiler.realsas_compiler_core.camera_geometry_v2 import (
    CameraProjectionV3, project_points_xyz_v3,
)
from models.rss_vnext.source_ray_evidence import sample_front_back_source_rays


@pytest.mark.parametrize("height,width", [(14, 20), (28, 28)])
@pytest.mark.parametrize("angle", [0.0, 0.67])
def test_original_front_back_rgba_sampled_pixel_matches_runtime_camera_geometry(
    height, width, angle,
):
    c = math.cos(angle)
    s = math.sin(angle)
    right = np.array([c, s, 0.], dtype=np.float32)
    up = np.array([-s, c, 0.], dtype=np.float32)
    forward = np.array([0., 0., 1.], dtype=np.float32)
    origin = np.array([.13, -.29, 2.7], dtype=np.float32)
    features = torch.zeros((1, 2, 13), dtype=torch.float32)
    for index, sign in enumerate((1., -1.)):
        features[0, index, :3] = torch.from_numpy(origin)
        features[0, index, 3:6] = torch.from_numpy(sign * right)
        features[0, index, 6:9] = torch.from_numpy(up)
        features[0, index, 9:12] = torch.from_numpy(sign * forward)
        features[0, index, 12] = 1.

    images = torch.zeros((1, 2, 4, height, width), dtype=torch.float32)
    yy, xx = torch.meshgrid(
        torch.arange(height), torch.arange(width), indexing="ij",
    )
    for view in (0, 1):
        images[0, view, 0] = (1 + xx + width * yy + view * height * width) / (
            2 * height * width + 1
        )
        images[0, view, 1] = (yy + 1) / (height + 1)
        images[0, view, 2] = (xx + 1) / (width + 1)
        images[0, view, 3] = 1.

    pixels = [
        (2, 1), (width // 2, height // 2),
        (width - 3, height - 2),
    ]
    points = []
    for x, y in pixels:
        u = 2. * (x + .5) / width - 1.
        v = 1. - 2. * (y + .5) / height
        points.append(origin + u * right + v * up)
    xyz = torch.from_numpy(np.array(points, dtype=np.float32))[None, ...]

    rays = sample_front_back_source_rays(images, features, xyz)
    assert rays.projection_valid.all()
    assert rays.ray_depth_authority == "UNKNOWN_NO_SOURCE_DEPTH"
    assert rays.appearance_authority == "ORIGINAL_RGBA_CAMERA_RAY_NOT_POINT_TEXTURE"

    for view in (0, 1):
        cam = CameraProjectionV3(
            view_id=("FRONT", "BACK")[view], view_index=view,
            origin=tuple(float(x) for x in origin),
            right=tuple(float(x) for x in ((1. if view == 0 else -1.) * right)),
            screen_up=tuple(float(x) for x in up),
            forward=tuple(float(x) for x in ((1. if view == 0 else -1.) * forward)),
            half_extent=1., resolution=width,
        )
        projected = project_points_xyz_v3(xyz[0].numpy(), cam)
        projected[:, 1] *= height / width
        for index, (x, y) in enumerate(pixels):
            expected_x = x if view == 0 else width - 1 - x
            assert float(projected[index, 0]) == pytest.approx(expected_x + .5, abs=5e-5)
            assert float(projected[index, 1]) == pytest.approx(y + .5, abs=5e-5)
            expected = images[0, view, :, y, expected_x]
            assert torch.allclose(rays.rgba[0, index, view], expected, atol=2e-5, rtol=0)


def test_world_up_samples_top_not_bottom_and_preserves_source_alpha():
    pixels = torch.zeros((1, 2, 4, 14, 20))
    pixels[:, :, 0, 0, :] = .2
    pixels[:, :, 0, -1, :] = .8
    pixels[:, :, 3, 0, :] = .25
    pixels[:, :, 3, -1, :] = .75
    cam = torch.zeros((1, 2, 13))
    cam[:, 0, [3, 7, 11, 12]] = 1.
    cam[:, 1, [3, 7, 12]] = torch.tensor([-1., 1., 1.])
    cam[:, 1, 11] = -1.
    top = torch.tensor([[[0., 1.-1./14., 0.], [0., -(1.-1./14.), 0.]]])
    sampled = sample_front_back_source_rays(pixels, cam, top)
    for view in (0, 1):
        assert float(sampled.rgba[0, 0, view, 0]) == pytest.approx(.2, abs=1e-5)
        assert float(sampled.rgba[0, 1, view, 0]) == pytest.approx(.8, abs=1e-5)
        assert float(sampled.rgba[0, 0, view, 3]) == pytest.approx(.25, abs=1e-5)
        assert float(sampled.rgba[0, 1, view, 3]) == pytest.approx(.75, abs=1e-5)
