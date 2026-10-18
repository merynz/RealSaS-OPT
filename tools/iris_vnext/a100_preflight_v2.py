from __future__ import annotations

"""A100 engineering preflight v2.

V2 preserves the v1 B=8 geometry optimizer smoke and pinned FlexiCubes smoke,
but fixes the analytic differentiability court so vertex-position, scalar-field,
and 21-D FlexiCubes parameter gradients are tested as independent inputs.

The v1 analytic sphere used ``sqrt(x^2+y^2+z^2)`` directly from the vertex
leaf. The regular grid contains the exact origin, where that analytic SDF has a
singular derivative; a NaN from the SDF construction could therefore be
misdiagnosed as a FlexiCubes vertex-gradient failure. V2 detaches the analytic
field values from the vertex leaf and promotes the field to its own leaf before
calling the exact same pinned FlexiCubes operator.
"""

from pathlib import Path
import time

import torch

from models.iris.vnext.contracts_v1 import IRISVNextConfigV1
from models.iris.vnext.flexicubes_v1 import PinnedFlexiCubesV1
from tools.iris_vnext import a100_preflight_v1 as v1


def independent_analytic_sphere_field_v2(vertices: torch.Tensor) -> torch.Tensor:
    """Return finite sphere-SDF values as an independent differentiable leaf."""
    values = torch.linalg.vector_norm(vertices.detach(), dim=-1) - 0.55
    if not bool(torch.isfinite(values).all()):
        raise RuntimeError("VNEXT_A100_ANALYTIC_FIELD_VALUE_INVALID")
    return values.detach().clone().requires_grad_(True)


def _finite_nonzero_grad(name: str, tensor: torch.Tensor) -> float:
    grad = tensor.grad
    if grad is None:
        raise RuntimeError(f"VNEXT_A100_FLEXICUBES_{name}_GRADIENT_MISSING")
    if not bool(torch.isfinite(grad).all()):
        raise RuntimeError(f"VNEXT_A100_FLEXICUBES_{name}_GRADIENT_INVALID")
    l1 = float(grad.abs().mean().detach().cpu())
    if not (l1 > 0.0):
        raise RuntimeError(f"VNEXT_A100_FLEXICUBES_{name}_GRADIENT_ZERO")
    return l1


def _analytic_flexicubes_probe_v2(root: Path, cfg: IRISVNextConfigV1):
    extractor = PinnedFlexiCubesV1(
        root, device="cuda", weight_scale=cfg.flexicubes_weight_scale
    )
    vertices, cubes = extractor.construct_iris_grid(cfg.extraction_training_resolution)
    vertices = vertices.cuda().float().detach().clone().requires_grad_(True)
    cubes = cubes.cuda()
    field = independent_analytic_sphere_field_v2(vertices)
    raw = torch.zeros((cubes.shape[0], 21), device="cuda", requires_grad=True)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    surface, faces, l_dev = extractor.extract(
        vertices_iris=vertices,
        field=field,
        cubes=cubes,
        raw_parameters=raw,
        resolution=cfg.extraction_training_resolution,
        training=True,
    )
    if not bool(torch.isfinite(surface).all()):
        raise RuntimeError("VNEXT_A100_FLEXICUBES_SURFACE_NONFINITE")
    if l_dev.numel() and not bool(torch.isfinite(l_dev).all()):
        raise RuntimeError("VNEXT_A100_FLEXICUBES_LDEV_NONFINITE")

    objective = surface.square().mean() + (
        l_dev.mean() if l_dev.numel() else surface.sum() * 0.0
    )
    objective.backward()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    if surface.numel() == 0 or faces.numel() == 0:
        raise RuntimeError("VNEXT_A100_FLEXICUBES_EMPTY_ANALYTIC_SURFACE")

    raw_l1 = _finite_nonzero_grad("PARAMETER", raw)
    field_l1 = _finite_nonzero_grad("SCALAR_FIELD", field)
    vertex_l1 = _finite_nonzero_grad("VERTEX", vertices)

    return {
        "probe_version": "INDEPENDENT_INPUT_GRADIENTS_V2",
        "resolution": int(cfg.extraction_training_resolution),
        "grid_vertices": int(vertices.shape[0]),
        "cubes": int(cubes.shape[0]),
        "surface_vertices": int(surface.shape[0]),
        "surface_faces": int(faces.shape[0]),
        "raw_parameter_gradient_l1": raw_l1,
        "scalar_field_gradient_l1": field_l1,
        "vertex_gradient_l1": vertex_l1,
        "elapsed_seconds": elapsed,
        "memory": v1._cuda_snapshot(),
        "scientific_result": "NONE__ANALYTIC_ENGINEERING_COURT_ONLY",
    }


def main():
    # Preserve the exact v1 engineering preflight orchestration and B=8 geometry
    # smoke; replace only the flawed analytic FlexiCubes gradient court.
    v1._analytic_flexicubes_probe = _analytic_flexicubes_probe_v2
    v1.main()


if __name__ == "__main__":
    main()
