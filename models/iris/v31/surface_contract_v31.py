from __future__ import annotations

"""Frozen V3.1 surface-address primitives that do not select an extractor.

The architecture freeze selected explicit IRIS/TESSA coordinate conversion and
permutation-invariant XYZ farthest-point sampling. It did not select a production
surface-extraction resolution/operator. This module implements only the selected
pieces so an extractor cannot be smuggled in as an incidental runtime detail.
"""

import math

import torch


IRIS_TO_TESSA_SCALE_V31 = 0.5
TESSA_TO_IRIS_SCALE_V31 = 2.0
WITNESS_SAMPLER_V31 = "DETERMINISTIC_CANONICAL_XYZ_FPS"
EXTRACTION_POLICY_FORM_V31 = "OBSERVABILITY_GATED_BOUNDED_RESOLUTION_ESCALATION"


def tessa_to_iris_coordinates_v31(xyz_tessa: torch.Tensor) -> torch.Tensor:
    _validate_xyz(xyz_tessa, "IRIS_V31_TESSA_COORDINATES_INVALID")
    return xyz_tessa * TESSA_TO_IRIS_SCALE_V31


def iris_to_tessa_coordinates_v31(xyz_iris: torch.Tensor) -> torch.Tensor:
    _validate_xyz(xyz_iris, "IRIS_V31_IRIS_COORDINATES_INVALID")
    return xyz_iris * IRIS_TO_TESSA_SCALE_V31


def _validate_xyz(value: torch.Tensor, code: str) -> None:
    if (
        not isinstance(value, torch.Tensor)
        or value.ndim < 2
        or value.shape[-1] != 3
        or not torch.is_floating_point(value)
        or not bool(torch.isfinite(value).all())
    ):
        raise ValueError(code)


def canonical_xyz_fps_indices_v31(points: torch.Tensor, *, count: int) -> torch.Tensor:
    """Select deterministic XYZ-FPS witnesses independent of source row order.

    The selected coordinate set is permutation invariant. Duplicate coordinates
    are equivalent witnesses; when duplicates exist, their original row identity
    is intentionally not treated as semantic information.
    """

    if (
        not isinstance(points, torch.Tensor)
        or points.ndim != 2
        or points.shape[-1] != 3
        or points.shape[0] <= 0
        or not torch.is_floating_point(points)
        or not bool(torch.isfinite(points).all())
    ):
        raise ValueError("IRIS_V31_XYZ_FPS_POINTS_INVALID")
    if isinstance(count, bool) or int(count) <= 0:
        raise ValueError("IRIS_V31_XYZ_FPS_COUNT_INVALID")

    cpu = points.detach().to(device="cpu", dtype=torch.float64)
    keys = [tuple(float(component) for component in row.tolist()) for row in cpu]
    order = sorted(range(len(keys)), key=lambda index: keys[index])
    if len(order) <= int(count):
        return torch.tensor(order, dtype=torch.long, device=points.device)

    canonical = [keys[index] for index in order]
    selected = [0]
    min_distance = [math.inf] * len(canonical)
    for _ in range(1, int(count)):
        anchor = canonical[selected[-1]]
        for index, point in enumerate(canonical):
            distance = sum(
                (anchor[axis] - point[axis]) ** 2 for axis in range(3)
            )
            if distance < min_distance[index]:
                min_distance[index] = distance
        for index in selected:
            min_distance[index] = -1.0
        best = max(min_distance)
        selected.append(next(index for index, value in enumerate(min_distance) if value == best))
    return torch.tensor([order[index] for index in selected], dtype=torch.long, device=points.device)


def canonical_xyz_fps_v31(points: torch.Tensor, *, count: int) -> torch.Tensor:
    return points.index_select(0, canonical_xyz_fps_indices_v31(points, count=count))
