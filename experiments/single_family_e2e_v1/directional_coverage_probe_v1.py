from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import math

import numpy as np
from PIL import Image, ImageDraw


@dataclass(frozen=True)
class DirectionalCoveragePolicyV1:
    min_source_alpha_coverage: float = 0.95
    max_mesh_outside_source_fraction: float = 0.01
    alpha_threshold: int = 0

    def validate(self) -> None:
        if not (0.0 < float(self.min_source_alpha_coverage) <= 1.0):
            raise ValueError("invalid minimum source-alpha coverage")
        if not (0.0 <= float(self.max_mesh_outside_source_fraction) < 1.0):
            raise ValueError("invalid mesh-outside-source fraction")
        if not (0 <= int(self.alpha_threshold) <= 254):
            raise ValueError("invalid alpha threshold")


@dataclass(frozen=True)
class DirectionalCoverageMeasurementV1:
    view_index: int
    source_alpha_pixels: int
    mesh_pixels: int
    covered_source_pixels: int
    uncovered_source_pixels: int
    mesh_outside_source_pixels: int
    source_alpha_coverage: float
    mesh_outside_source_fraction: float
    passed: bool
    policy: dict
    schema: str = "RealSaS.DirectionalCoverageMeasurement.v1"

    def to_dict(self):
        return asdict(self)


def _surface_raster(surface, view_index: int) -> dict[str, tuple[float, float]]:
    out = {}
    for node in surface.surface_nodes:
        rows = [tuple(map(float, xy)) for view, xy in node.raster_bindings if int(view) == int(view_index)]
        if len(rows) > 1:
            raise ValueError(f"duplicate surface raster binding:{node.surface_id}:V{view_index}")
        if rows:
            out[str(node.surface_id)] = rows[0]
    return out


def _mesh_vertex_raster(mesh, raster: dict[str, tuple[float, float]]) -> dict[str, tuple[float, float]]:
    out = {}
    for vertex in mesh.vertices:
        x = y = total = 0.0
        for sid, coeff in vertex.support_binding.coefficients:
            sid = str(sid)
            if sid not in raster:
                raise ValueError(f"mesh support lacks target-view raster authority:{sid}")
            c = float(coeff)
            if not math.isfinite(c) or c < -1e-12:
                raise ValueError("invalid mesh support coefficient")
            x += c * raster[sid][0]
            y += c * raster[sid][1]
            total += c
        if abs(total - 1.0) > 1e-6:
            raise ValueError("mesh support coefficients must sum to one")
        out[str(vertex.canonical_mesh_vertex_id)] = (x, y)
    return out


def measure_directional_source_alpha_coverage_v1(
    *,
    surface,
    mesh,
    source_rgba_path,
    policy: DirectionalCoveragePolicyV1 = DirectionalCoveragePolicyV1(),
) -> DirectionalCoverageMeasurementV1:
    """Diagnostic raster coverage of one qualified direction-local mesh.

    This is a measurement only: source alpha is never converted into mesh/product
    authority here. It measures whether the already-qualified mesh covers the admitted
    source silhouette and whether that mesh spills materially outside it.
    """
    policy.validate()
    view = int(mesh.view_index)
    image = Image.open(Path(source_rgba_path)).convert("RGBA")
    width, height = image.size
    alpha = np.asarray(image.getchannel("A"), dtype=np.uint8)
    source = alpha > int(policy.alpha_threshold)
    source_count = int(source.sum())
    if source_count <= 0:
        raise ValueError("source observation has empty alpha support")

    raster = _surface_raster(surface, view)
    vertex_xy = _mesh_vertex_raster(mesh, raster)
    mask_image = Image.new("1", (width, height), 0)
    draw = ImageDraw.Draw(mask_image)
    for face in mesh.faces:
        if len(face) != 3:
            raise ValueError("directional coverage probe requires triangulated mesh")
        try:
            points = [vertex_xy[str(vid)] for vid in face]
        except KeyError as exc:
            raise ValueError("mesh face references unknown vertex") from exc
        draw.polygon(points, fill=1)
    mesh_mask = np.asarray(mask_image, dtype=bool)
    mesh_count = int(mesh_mask.sum())
    if mesh_count <= 0:
        raise ValueError("qualified mesh rasterizes to empty coverage")

    covered = int(np.logical_and(mesh_mask, source).sum())
    outside = int(np.logical_and(mesh_mask, np.logical_not(source)).sum())
    uncovered = int(np.logical_and(source, np.logical_not(mesh_mask)).sum())
    coverage = float(covered / source_count)
    outside_fraction = float(outside / mesh_count)
    passed = (
        coverage >= float(policy.min_source_alpha_coverage)
        and outside_fraction <= float(policy.max_mesh_outside_source_fraction)
    )
    return DirectionalCoverageMeasurementV1(
        view_index=view,
        source_alpha_pixels=source_count,
        mesh_pixels=mesh_count,
        covered_source_pixels=covered,
        uncovered_source_pixels=uncovered,
        mesh_outside_source_pixels=outside,
        source_alpha_coverage=coverage,
        mesh_outside_source_fraction=outside_fraction,
        passed=passed,
        policy=asdict(policy),
    )
