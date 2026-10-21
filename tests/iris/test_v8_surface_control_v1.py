from __future__ import annotations

import numpy as np
import torch

from tools.iris_vnext.audit_v8_raw_field_sparse_compatibility_v1 import (
    _cell_corner_extrema,
    _edge_roughness,
)
from tools.iris_vnext.audit_fit8_v8_native_flexicubes_control_v1 import (
    SCHEMA as CONTROL_SCHEMA,
)
from tools.iris_vnext.materialize_fit8_v8_native_flexicubes_v1 import (
    MATERIALIZER_ID,
    SCHEMA as MATERIALIZATION_SCHEMA,
    _raw_cube_parameters,
    parser,
)


def _grid(base: int) -> np.ndarray:
    axis = np.linspace(-1.0, 1.0, base + 1, dtype=np.float32)
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    return np.stack((x, y, z), axis=-1)


def test_distance_like_sphere_has_compact_sign_support_and_bounded_edge_variation():
    base = 8
    xyz = _grid(base)
    field = np.linalg.norm(xyz, axis=-1).astype(np.float32) - np.float32(0.55)
    sign, band_only = _cell_corner_extrema(field, slab=2)
    rough = _edge_roughness(field, 2.0 / base)
    assert len(sign) > 0
    assert len(sign) < base ** 3
    assert len(band_only) >= 0
    assert rough["sign_flip_edge_count"] > 0
    assert rough["max_abs_field_difference_over_spatial_step"] <= 1.00001


def test_oscillatory_field_is_flagged_by_edge_roughness_indicators():
    base = 8
    index = np.indices((base + 1,) * 3).sum(axis=0)
    field = np.where(index % 2 == 0, -1.0, 1.0).astype(np.float32)
    rough = _edge_roughness(field, 2.0 / base)
    assert rough["sign_flip_edge_fraction"] == 1.0
    assert rough["max_abs_field_difference_over_spatial_step"] == 8.0
    assert rough["edge_fraction_abs_difference_over_spatial_step"]["gt_4x"] == 1.0


class _FakeCubeExtraction:
    def __call__(self, corner_h: torch.Tensor) -> torch.Tensor:
        assert corner_h.ndim == 4 and corner_h.shape[-2] == 8
        shape = corner_h.shape[:2] + (21,)
        return torch.ones(shape, dtype=corner_h.dtype, device=corner_h.device)


class _FakeSceneCore:
    def __init__(self):
        self.cube_extraction = _FakeCubeExtraction()


class _FakeModel:
    def __init__(self):
        self.scene_core = _FakeSceneCore()


def test_native_control_cube_parameter_path_uses_learned_cube_head_in_chunks():
    h = torch.arange(10 * 4, dtype=torch.float32).reshape(1, 10, 4)
    cubes = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6, 7], [2, 3, 4, 5, 6, 7, 8, 9]],
        dtype=torch.long,
    )
    raw = _raw_cube_parameters(_FakeModel(), h, cubes, chunk=1)
    assert raw.shape == (1, 2, 21)
    assert torch.equal(raw, torch.ones_like(raw))


def test_native_control_cli_and_status_are_explicitly_measurement_only():
    destinations = {action.dest for action in parser()._actions}
    assert {"v8_checkpoint", "flexicubes_root", "prepared_manifest", "output_dir"} <= destinations
    assert MATERIALIZATION_SCHEMA == "RealSaS.Fit8V8NativeFlexiCubesMaterialization.v1"
    assert MATERIALIZER_ID == "RealSaS.Fit8V8NativeFlexiCubesMaterializer.v1"
    assert CONTROL_SCHEMA == "RealSaS.Fit8V8NativeFlexiCubesControlCourt.v1"
