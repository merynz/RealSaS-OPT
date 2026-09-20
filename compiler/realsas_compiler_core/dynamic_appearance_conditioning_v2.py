from __future__ import annotations

"""Subject-free dynamic appearance conditioning for V2 art under deformation.

Shipping gates are intrinsic to the textured surface and therefore invariant to
rigid 3D rotation/translation. Screen projection conditioning is retained only
as diagnostic evidence because legitimate foreshortening must not be mistaken
for texture deformation.
"""

from typing import Any, Mapping

import numpy as np

from .types import QualificationError


def _edge_matrix(points, *, dimension: int) -> np.ndarray:
    value = np.asarray(points, dtype=np.float64)
    if value.shape != (3, dimension) or not np.isfinite(value).all():
        raise QualificationError("DYNAMIC_APPEARANCE_TRIANGLE_INVALID")
    return np.column_stack((value[1] - value[0], value[2] - value[0]))


def _condition(matrix: np.ndarray) -> float:
    singular = np.linalg.svd(np.asarray(matrix, dtype=np.float64), compute_uv=False)
    smax = float(np.max(singular))
    smin = float(np.min(singular))
    if not np.isfinite(smax) or not np.isfinite(smin) or smin <= 1.0e-12:
        raise QualificationError("DYNAMIC_APPEARANCE_AFFINE_DEGENERATE")
    return smax / smin


def _surface_metric(edge_matrix_3d: np.ndarray) -> np.ndarray:
    edges = np.asarray(edge_matrix_3d, dtype=np.float64)
    metric = edges.T @ edges
    if metric.shape != (2, 2) or not np.isfinite(metric).all():
        raise QualificationError("DYNAMIC_APPEARANCE_SURFACE_METRIC_INVALID")
    if float(np.linalg.det(metric)) <= 1.0e-14:
        raise QualificationError("DYNAMIC_APPEARANCE_SURFACE_DEGENERATE")
    return metric


def _relative_surface_stretch(
    reference_edges_3d: np.ndarray,
    target_edges_3d: np.ndarray,
) -> tuple[float, float]:
    reference_metric = _surface_metric(reference_edges_3d)
    target_metric = _surface_metric(target_edges_3d)
    try:
        lower = np.linalg.cholesky(reference_metric)
        inv_lower = np.linalg.inv(lower)
    except np.linalg.LinAlgError as exc:
        raise QualificationError(
            "DYNAMIC_APPEARANCE_REFERENCE_METRIC_DEGENERATE"
        ) from exc
    generalized = inv_lower @ target_metric @ inv_lower.T
    generalized = 0.5 * (generalized + generalized.T)
    eigen = np.linalg.eigvalsh(generalized)
    if (
        eigen.shape != (2,)
        or not np.isfinite(eigen).all()
        or float(np.min(eigen)) <= 1.0e-12
    ):
        raise QualificationError("DYNAMIC_APPEARANCE_GENERALIZED_METRIC_INVALID")
    stretches = np.sqrt(eigen)
    smin = float(np.min(stretches))
    smax = float(np.max(stretches))
    return smax / smin, max(smax, 1.0 / smin)


def dynamic_face_conditioning_metrics(
    *,
    uv_triangle,
    posed_xyz_triangle,
    rest_xyz_triangle,
    posed_screen_triangle,
    previous_xyz_triangle=None,
    min_projected_double_area_px2: float,
) -> dict[str, Any]:
    minimum_area = float(min_projected_double_area_px2)
    if not np.isfinite(minimum_area) or minimum_area <= 0.0:
        raise QualificationError("DYNAMIC_APPEARANCE_MIN_AREA_INVALID")

    uv_edges = _edge_matrix(uv_triangle, dimension=2)
    posed_edges = _edge_matrix(posed_xyz_triangle, dimension=3)
    rest_edges = _edge_matrix(rest_xyz_triangle, dimension=3)
    screen_edges = _edge_matrix(posed_screen_triangle, dimension=2)

    uv_det = abs(float(np.linalg.det(uv_edges)))
    projected_area2 = abs(float(np.linalg.det(screen_edges)))
    if uv_det <= 1.0e-12:
        raise QualificationError("DYNAMIC_APPEARANCE_UV_DEGENERATE")

    result: dict[str, Any] = {
        "posed_projected_double_area_px2": projected_area2,
        "measurable": projected_area2 >= minimum_area,
        "uv_to_surface_condition_number": None,
        "relative_surface_condition_number": None,
        "relative_surface_principal_stretch": None,
        "adjacent_frame_surface_principal_stretch": None,
        "uv_to_screen_condition_number_diagnostic": None,
    }
    if not result["measurable"]:
        return result

    uv_to_surface = posed_edges @ np.linalg.inv(uv_edges)
    result["uv_to_surface_condition_number"] = _condition(uv_to_surface)
    result["uv_to_screen_condition_number_diagnostic"] = _condition(
        screen_edges @ np.linalg.inv(uv_edges)
    )

    relative_condition, relative_stretch = _relative_surface_stretch(
        rest_edges, posed_edges
    )
    result["relative_surface_condition_number"] = relative_condition
    result["relative_surface_principal_stretch"] = relative_stretch

    if previous_xyz_triangle is not None:
        previous_edges = _edge_matrix(previous_xyz_triangle, dimension=3)
        _step_condition, step_stretch = _relative_surface_stretch(
            previous_edges, posed_edges
        )
        result["adjacent_frame_surface_principal_stretch"] = step_stretch

    return result


def validate_dynamic_appearance_policy(
    policy: Mapping[str, Any],
) -> dict[str, float | int]:
    required = (
        "dynamic_min_visible_pixels_per_face",
        "dynamic_min_projected_double_area_px2",
        "dynamic_max_uv_to_surface_condition_number",
        "dynamic_max_relative_surface_condition_number",
        "dynamic_max_relative_surface_principal_stretch",
        "dynamic_max_adjacent_frame_surface_principal_stretch",
    )
    if any(key not in policy for key in required):
        raise QualificationError("DYNAMIC_APPEARANCE_POLICY_INCOMPLETE")
    out: dict[str, float | int] = {
        "dynamic_min_visible_pixels_per_face": int(
            policy["dynamic_min_visible_pixels_per_face"]
        ),
        "dynamic_min_projected_double_area_px2": float(
            policy["dynamic_min_projected_double_area_px2"]
        ),
        "dynamic_max_uv_to_surface_condition_number": float(
            policy["dynamic_max_uv_to_surface_condition_number"]
        ),
        "dynamic_max_relative_surface_condition_number": float(
            policy["dynamic_max_relative_surface_condition_number"]
        ),
        "dynamic_max_relative_surface_principal_stretch": float(
            policy["dynamic_max_relative_surface_principal_stretch"]
        ),
        "dynamic_max_adjacent_frame_surface_principal_stretch": float(
            policy["dynamic_max_adjacent_frame_surface_principal_stretch"]
        ),
    }
    if int(out["dynamic_min_visible_pixels_per_face"]) < 1:
        raise QualificationError("DYNAMIC_APPEARANCE_VISIBLE_PIXEL_FLOOR_INVALID")
    if float(out["dynamic_min_projected_double_area_px2"]) <= 0.0:
        raise QualificationError("DYNAMIC_APPEARANCE_PROJECTED_AREA_FLOOR_INVALID")
    for key in required[2:]:
        value = float(out[key])
        if not np.isfinite(value) or value < 1.0:
            raise QualificationError("DYNAMIC_APPEARANCE_CONDITION_LIMIT_INVALID")
    return out
