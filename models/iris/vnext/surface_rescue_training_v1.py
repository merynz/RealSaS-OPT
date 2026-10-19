from __future__ import annotations

"""Shared SURF-A1 frozen-H surface-rescue training primitives."""

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any

import torch

from .flexicubes_v1 import PinnedFlexiCubesV1, deterministic_surface_witness_v1, symmetric_surface_chamfer_v1
from .surface_rescue_raster_v1 import SurfA1RasterConfigV1, raster_losses_v1, rasterize_surface_v1
from .surface_rescue_v1 import FrozenHSurfaceRescueV1


@dataclass(frozen=True)
class SurfA1ScheduleV1:
    r24_steps: int = 400
    r32_steps: int = 600
    r48_steps: int = 1400
    warmup_steps: int = 120
    base_lr: float = 1.0e-4
    minimum_lr: float = 1.0e-5

    @property
    def total_steps(self) -> int:
        return self.r24_steps + self.r32_steps + self.r48_steps

    def resolution(self, step: int) -> int:
        if not (0 <= int(step) < self.total_steps):
            raise ValueError("SURF_A1_STEP_OUT_OF_RANGE")
        if step < self.r24_steps:
            return 24
        if step < self.r24_steps + self.r32_steps:
            return 32
        return 48

    def learning_rate(self, step: int) -> float:
        if not (0 <= int(step) < self.total_steps):
            raise ValueError("SURF_A1_STEP_OUT_OF_RANGE")
        if step < self.warmup_steps:
            return self.base_lr * float(step + 1) / float(self.warmup_steps)
        progress = float(step - self.warmup_steps) / float(max(1, self.total_steps - self.warmup_steps - 1))
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))
        return self.minimum_lr + (self.base_lr - self.minimum_lr) * cosine


def module_state_sha256_v1(module: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        cpu = tensor.detach().cpu().contiguous()
        h.update(name.encode("utf-8"))
        h.update(str(cpu.dtype).encode("utf-8"))
        h.update(json.dumps(list(cpu.shape), separators=(",", ":")).encode("utf-8"))
        h.update(cpu.numpy().tobytes(order="C"))
    return h.hexdigest()


def exact_v8_grid_evidence_v1(
    v8_model: torch.nn.Module,
    scene_planes: torch.Tensor,
    extractor: PinnedFlexiCubesV1,
    *,
    resolution: int,
    cube_chunk: int = 2048,
) -> dict[str, torch.Tensor]:
    if int(cube_chunk) <= 0:
        raise ValueError("SURF_A1_V8_CUBE_CHUNK_INVALID")
    base_vertices, cubes = extractor.construct_iris_grid(int(resolution))
    base_vertices = base_vertices.to(device=scene_planes.device, dtype=torch.float32)
    cubes = cubes.to(device=scene_planes.device)
    grid = base_vertices.unsqueeze(0)
    with torch.inference_mode(), torch.autocast(device_type=scene_planes.device.type, enabled=False):
        local = v8_model.scene_core.query(scene_planes.float(), grid.float())
        h = local["H"].float().detach()
        field = local["field"].float().detach()
        delta = local["extraction_delta"].float().detach()
        raw_rows = []
        for start in range(0, int(cubes.shape[0]), int(cube_chunk)):
            cube = cubes[start : start + int(cube_chunk)]
            raw_rows.append(v8_model.scene_core.cube_extraction(h[:, cube]).float())
        raw = torch.cat(raw_rows, dim=1).detach()
    if raw.shape != (1, int(cubes.shape[0]), 21):
        raise RuntimeError("SURF_A1_V8_RAW_CUBE_PARAMETER_SHAPE_DRIFT")
    return {
        "grid": grid.detach(),
        "cubes": cubes,
        "H": h,
        "field": field,
        "extraction_delta": delta,
        "raw_cube_parameters": raw,
    }


def extract_rescued_surface_v1(
    v8_model: torch.nn.Module,
    decoder: FrozenHSurfaceRescueV1,
    scene_planes: torch.Tensor,
    extractor: PinnedFlexiCubesV1,
    *,
    resolution: int,
    cube_chunk: int = 2048,
    training: bool = True,
) -> dict[str, Any]:
    evidence = exact_v8_grid_evidence_v1(
        v8_model, scene_planes, extractor, resolution=int(resolution), cube_chunk=int(cube_chunk)
    )
    device_type = scene_planes.device.type
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=(device_type == "cuda")):
        point = decoder.point_decode(
            h=evidence["H"],
            xyz_iris=evidence["grid"],
            raw_v8_field=evidence["field"],
            v8_extraction_delta=evidence["extraction_delta"],
            resolution=int(resolution),
        )
        cube = decoder.cube_parameters(
            point_features=point["point_features"],
            cubes=evidence["cubes"],
            inherited_v8_raw_parameters=evidence["raw_cube_parameters"],
            cube_chunk=int(cube_chunk),
        )
    vertices, faces, l_dev = extractor.extract(
        vertices_iris=point["deformed_xyz_iris"][0],
        field=point["field"][0],
        cubes=evidence["cubes"],
        raw_parameters=cube["raw_parameters"][0],
        resolution=int(resolution),
        training=bool(training),
    )
    if not bool(torch.isfinite(vertices).all().item()):
        raise RuntimeError("SURF_A1_EXTRACTED_VERTEX_NONFINITE")
    return {
        "vertices_iris": vertices,
        "faces": faces,
        "l_dev": l_dev.mean() if l_dev.numel() else vertices.sum() * 0.0,
        "field_residual": point["field_residual"],
        "delta_cell_residual": point["delta_cells"] - point["inherited_delta_cells"],
        "cube_parameter_residual": cube["cube_parameter_residual"],
        "raw_cube_parameters": cube["raw_parameters"],
    }


def surf_a1_objective_v1(
    surface: dict[str, Any],
    *,
    teacher_surface_points_iris: torch.Tensor,
    raster_views: list[dict[str, Any]],
    raster_config: SurfA1RasterConfigV1 = SurfA1RasterConfigV1(),
) -> dict[str, torch.Tensor]:
    vertices = surface["vertices_iris"]
    faces = surface["faces"]
    witness, _ = deterministic_surface_witness_v1(
        vertices, faces, count=int(teacher_surface_points_iris.shape[0])
    )
    chamfer = symmetric_surface_chamfer_v1(
        witness.unsqueeze(0),
        teacher_surface_points_iris.unsqueeze(0).to(device=vertices.device, dtype=torch.float32),
    )
    raster_rows = []
    for row in raster_views:
        rendered = rasterize_surface_v1(vertices, faces, camera=row["camera"], config=raster_config)
        raster_rows.append(
            raster_losses_v1(
                rendered,
                source_mask=row["mask"],
                teacher_depth_normalized=row["depth_normalized"],
                teacher_normal_world=row["normal_world"],
                depth_normal_valid=row["depth_normal_valid"],
            )
        )
    silhouette = torch.stack([r["silhouette"] for r in raster_rows]).mean()
    depth = torch.stack([r["depth"] for r in raster_rows]).mean()
    normal = torch.stack([r["normal"] for r in raster_rows]).mean()
    overlap = torch.stack([r["depth_normal_overlap_fraction"] for r in raster_rows]).mean()
    l_dev = surface["l_dev"]
    cell_offset = surface["delta_cell_residual"].float().square().mean()
    field_residual = surface["field_residual"].float().square().mean()
    cube_residual = surface["cube_parameter_residual"].float().square().mean()
    total = (
        chamfer
        + silhouette
        + 0.5 * depth
        + 0.25 * normal
        + 0.01 * l_dev
        + 0.01 * cell_offset
        + 0.001 * field_residual
        + 0.0001 * cube_residual
    )
    return {
        "total": total,
        "surface_chamfer": chamfer,
        "silhouette": silhouette,
        "depth": depth,
        "normal": normal,
        "flexicubes_deviation": l_dev,
        "cell_offset_l2": cell_offset,
        "field_residual_l2": field_residual,
        "cube_parameter_residual_l2": cube_residual,
        "depth_normal_overlap_fraction": overlap,
    }
