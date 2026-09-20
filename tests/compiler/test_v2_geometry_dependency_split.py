from __future__ import annotations

import numpy as np

from compiler.realsas_compiler_core.camera_geometry_v2 import (
    CameraProjectionV3 as NeutralCamera,
    project_points_xyz_v3 as neutral_project,
    qualify_camera_v3 as neutral_qualify,
)
from compiler.realsas_compiler_core.playback_full_surface_v3 import (
    CameraProjectionV3 as HistoricalCamera,
    project_points_xyz_v3 as historical_project,
    qualify_camera_v3 as historical_qualify,
)
from compiler.realsas_compiler_core.rest_preservation_v1 import (
    silhouette_distance_metrics as historical_silhouette,
)
from compiler.realsas_compiler_core.silhouette_metrics_v2 import (
    silhouette_distance_metrics as neutral_silhouette,
)


def _camera_payload():
    return {
        "view_index": 3,
        "origin": [0.25, -0.5, -3.0],
        "right": [1.0, 0.0, 0.0],
        "screen_up": [0.0, 1.0, 0.0],
        "forward": [0.0, 0.0, 1.0],
        "half_extent": 1.75,
        "resolution": 1024,
    }


def test_neutral_camera_type_is_historical_reexport_identity():
    assert NeutralCamera is HistoricalCamera


def test_neutral_camera_projection_is_numerically_identical_to_historical_bridge():
    payload = _camera_payload()
    a = neutral_qualify(payload, view_id="V3", view_index=3)
    b = historical_qualify(payload, view_id="V3", view_index=3)
    assert a == b
    points = np.asarray(
        [
            [-0.5, -0.25, 0.1],
            [0.0, 0.0, 0.5],
            [0.75, 0.25, 1.25],
        ],
        dtype=np.float64,
    )
    np.testing.assert_array_equal(neutral_project(points, a), historical_project(points, b))


def test_neutral_silhouette_metric_is_numerically_identical_to_stage32_metric():
    source = np.zeros((32, 32), dtype=bool)
    source[7:25, 8:24] = True
    predicted = np.zeros((32, 32), dtype=bool)
    predicted[8:26, 9:25] = True
    assert neutral_silhouette(source, predicted) == historical_silhouette(
        source, predicted
    )
