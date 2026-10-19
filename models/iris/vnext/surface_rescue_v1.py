from __future__ import annotations

"""Resolution-aware surface rescue decoder over exact sealed V8 local evidence H.

SURF-A1 never rewrites V8 scene_core. At zero step and native R24 it reproduces
V8 field, extraction deformation and FlexiCubes raw parameters exactly. At other
resolutions the inherited deformation is expressed in cell units so the absolute
motion scales with the current lattice rather than reusing V8's absolute 0.03
normalized-coordinate bound.
"""

from dataclasses import dataclass
import math

import torch
from torch import nn


@dataclass(frozen=True)
class FrozenHSurfaceRescueConfigV1:
    local_feature_dim: int = 384
    point_width: int = 512
    point_residual_blocks: int = 4
    cube_residual_width: int = 768
    native_resolution: int = 24
    maximum_absolute_cell_offset: float = 0.45
    maximum_absolute_field_residual: float = 0.5

    def validate(self) -> None:
        if min(self.local_feature_dim, self.point_width, self.point_residual_blocks, self.cube_residual_width) <= 0:
            raise ValueError("SURF_A1_CONFIG_WIDTH_INVALID")
        if self.native_resolution < 2:
            raise ValueError("SURF_A1_NATIVE_RESOLUTION_INVALID")
        if not (0.0 < self.maximum_absolute_cell_offset < 0.5):
            raise ValueError("SURF_A1_CELL_OFFSET_BOUND_INVALID")
        if self.maximum_absolute_field_residual <= 0.0:
            raise ValueError("SURF_A1_FIELD_RESIDUAL_BOUND_INVALID")


def iris_cell_step_v1(resolution: int) -> float:
    if int(resolution) < 2:
        raise ValueError("SURF_A1_RESOLUTION_INVALID")
    return 2.0 / float(resolution)


class _ResidualBlockV1(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, width)
        self.fc2 = nn.Linear(width, width)
        self.scale = 1.0 / math.sqrt(2.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x + self.fc2(torch.nn.functional.silu(self.fc1(self.norm(x))))) * self.scale


def _zero_linear(linear: nn.Linear) -> None:
    nn.init.zeros_(linear.weight)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)


class FrozenHSurfaceRescueV1(nn.Module):
    """Trainable extraction-only adapter; all V8 inputs must be detached/frozen."""

    architecture_id = "RealSaS.IRIS.FrozenHSurfaceRescue.v1"
    teacher_truth_in_inference_forward = False
    scene_core_trainable = False

    def __init__(self, config: FrozenHSurfaceRescueConfigV1 = FrozenHSurfaceRescueConfigV1()):
        super().__init__()
        config.validate()
        self.config = config
        input_dim = config.local_feature_dim + 3 + 1 + 3 + 1
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_projection = nn.Linear(input_dim, config.point_width)
        self.blocks = nn.ModuleList([_ResidualBlockV1(config.point_width) for _ in range(config.point_residual_blocks)])
        self.final_norm = nn.LayerNorm(config.point_width)
        self.field_residual = nn.Linear(config.point_width, 1)
        self.delta_residual_logits = nn.Linear(config.point_width, 3)
        self.cube_residual = nn.Sequential(
            nn.LayerNorm(8 * config.point_width),
            nn.Linear(8 * config.point_width, config.cube_residual_width),
            nn.GELU(),
            nn.Linear(config.cube_residual_width, 21),
        )
        _zero_linear(self.field_residual)
        _zero_linear(self.delta_residual_logits)
        _zero_linear(self.cube_residual[-1])

    @staticmethod
    def _require_frozen(name: str, value: torch.Tensor) -> None:
        if value.requires_grad:
            raise ValueError("SURF_A1_FROZEN_V8_INPUT_REQUIRES_GRAD:" + name)
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError("SURF_A1_FROZEN_V8_INPUT_NONFINITE:" + name)

    def point_decode(
        self,
        *,
        h: torch.Tensor,
        xyz_iris: torch.Tensor,
        raw_v8_field: torch.Tensor,
        v8_extraction_delta: torch.Tensor,
        resolution: int,
    ) -> dict[str, torch.Tensor]:
        self._require_frozen("H", h)
        self._require_frozen("RAW_FIELD", raw_v8_field)
        self._require_frozen("EXTRACTION_DELTA", v8_extraction_delta)
        if h.ndim != 3 or h.shape[-1] != self.config.local_feature_dim:
            raise ValueError("SURF_A1_H_SHAPE_INVALID")
        if xyz_iris.shape != h.shape[:2] + (3,) or v8_extraction_delta.shape != xyz_iris.shape:
            raise ValueError("SURF_A1_POINT_SHAPE_INVALID")
        if raw_v8_field.shape != h.shape[:2]:
            raise ValueError("SURF_A1_FIELD_SHAPE_INVALID")
        if float(xyz_iris.detach().abs().max().item()) > 1.00001:
            raise ValueError("SURF_A1_XYZ_DOMAIN_INVALID")

        native_step = iris_cell_step_v1(self.config.native_resolution)
        current_step = iris_cell_step_v1(int(resolution))
        base_cells = v8_extraction_delta.float() / native_step
        maximum = float(self.config.maximum_absolute_cell_offset)
        if float(base_cells.detach().abs().max().item()) >= maximum:
            raise ValueError("SURF_A1_INHERITED_DELTA_EXCEEDS_CELL_BOUND")
        ratio = math.log2(float(resolution) / float(self.config.native_resolution))
        resolution_feature = torch.full_like(raw_v8_field[..., None], ratio)
        features = torch.cat((h, xyz_iris, raw_v8_field[..., None], base_cells, resolution_feature), dim=-1)
        z = torch.nn.functional.silu(self.input_projection(self.input_norm(features)))
        for block in self.blocks:
            z = block(z)
        z = self.final_norm(z)

        field_residual = torch.tanh(self.field_residual(z).squeeze(-1).float()) * float(self.config.maximum_absolute_field_residual)
        # Exact identity at zero residual, with a per-coordinate remaining-margin
        # bound. This is stronger than an atanh/tanh round-trip: the inherited V8
        # offset is copied bit-for-bit before any learned correction is added.
        delta_logits = self.delta_residual_logits(z).float()
        remaining_margin = (maximum - base_cells.abs()).clamp_min(0.0)
        delta_adjustment = torch.tanh(delta_logits) * remaining_margin
        delta_cells = base_cells + delta_adjustment
        deformed_xyz = xyz_iris.float() + delta_cells * current_step
        return {
            "point_features": z,
            "field": raw_v8_field.float() + field_residual,
            "field_residual": field_residual,
            "delta_cells": delta_cells,
            "deformed_xyz_iris": deformed_xyz,
            "inherited_delta_cells": base_cells,
        }

    def cube_parameters(
        self,
        *,
        point_features: torch.Tensor,
        cubes: torch.Tensor,
        inherited_v8_raw_parameters: torch.Tensor,
        cube_chunk: int = 2048,
    ) -> dict[str, torch.Tensor]:
        self._require_frozen("V8_CUBE_PARAMETERS", inherited_v8_raw_parameters)
        if cubes.ndim != 2 or cubes.shape[-1] != 8:
            raise ValueError("SURF_A1_CUBES_SHAPE_INVALID")
        if point_features.ndim != 3 or point_features.shape[-1] != self.config.point_width:
            raise ValueError("SURF_A1_POINT_FEATURE_SHAPE_INVALID")
        if inherited_v8_raw_parameters.shape != (point_features.shape[0], cubes.shape[0], 21):
            raise ValueError("SURF_A1_INHERITED_CUBE_PARAMETER_SHAPE_INVALID")
        if int(cube_chunk) <= 0:
            raise ValueError("SURF_A1_CUBE_CHUNK_INVALID")
        residual_rows = []
        for start in range(0, int(cubes.shape[0]), int(cube_chunk)):
            cube = cubes[start : start + int(cube_chunk)]
            corner = point_features[:, cube].flatten(-2)
            residual_rows.append(self.cube_residual(corner).float())
        residual = torch.cat(residual_rows, dim=1)
        if residual.shape != inherited_v8_raw_parameters.shape:
            raise RuntimeError("SURF_A1_CUBE_RESIDUAL_SHAPE_DRIFT")
        return {
            "raw_parameters": inherited_v8_raw_parameters.float() + residual,
            "cube_parameter_residual": residual,
        }

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
