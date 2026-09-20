from __future__ import annotations

"""Subject-free screen-space appearance conditioning for V2 dynamic art."""

from typing import Any, Mapping

import numpy as np

from .types import QualificationError


def _triangle_matrix(points_xy) -> np.ndarray:
    points = np.asarray(points_xy, dtype=np.float64)
    if points.shape != (3, 2) or not np.isfinite(points).all():
        raise QualificationError("DYNAMIC_APPEARANCE_TRIANGLE_INVALID")
    return np.column_stack((points[1] - points[0], points[2] - points[0]))


def _sv_metrics(matrix: np.ndarray) -> tuple[float, float, float]:
    value = np.asarray(matrix, dtype=np.float64)
    if value.shape != (2, 2) or not np.isfinite(value).all():
        raise QualificationError("DYNAMIC_APPEARANCE_AFFINE_INVALID")
    singular = np.linalg.svd(value, compute_uv=False)
    smax = float(np.max(singular))
    smin = float(np.min(singular))
    if not np.isfinite(smax) or not np.isfinite(smin) or smin <= 1.0e-12:
        raise QualificationError("DYNAMIC_APPEARANCE_AFFINE_DEGENERATE")
    return smax, smin, smax / smin


def dynamic_face_conditioning_metrics(
    *,
    uv_triangle,
    posed_screen_triangle,
    rest_screen_triangle=None,
    previous_screen_triangle=None,
    min_projected_double_area_px2: float,
) -> dict[str, Any]:
    minimum_area = float(min_projected_double_area_px2)
    if not np.isfinite(minimum_area) or minimum_area <= 0.0:
        raise QualificationError("DYNAMIC_APPEARANCE_MIN_AREA_INVALID")

    uv_matrix = _triangle_matrix(uv_triangle)
    posed_matrix = _triangle_matrix(posed_screen_triangle)
    uv_det = abs(float(np.linalg.det(uv_matrix)))
    posed_area2 = abs(float(np.linalg.det(posed_matrix)))
    if uv_det <= 1.0e-12:
        raise QualificationError("DYNAMIC_APPEARANCE_UV_DEGENERATE")

    result: dict[str, Any] = {
        "posed_projected_double_area_px2": posed_area2,
        "measurable": posed_area2 >= minimum_area,
        "uv_to_screen_condition_number": None,
        "relative_screen_condition_number": None,
        "relative_principal_stretch": None,
        "adjacent_frame_principal_stretch": None,
    }
    if not result["measurable"]:
        return result

    uv_to_screen = posed_matrix @ np.linalg.inv(uv_matrix)
    _smax, _smin, condition = _sv_metrics(uv_to_screen)
    result["uv_to_screen_condition_number"] = condition

    if rest_screen_triangle is not None:
        rest_matrix = _triangle_matrix(rest_screen_triangle)
        rest_area2 = abs(float(np.linalg.det(rest_matrix)))
        result["rest_projected_double_area_px2"] = rest_area2
        if rest_area2 >= minimum_area:
            relative = posed_matrix @ np.linalg.inv(rest_matrix)
            smax, smin, condition = _sv_metrics(relative)
            result["relative_screen_condition_number"] = condition
            result["relative_principal_stretch"] = max(smax, 1.0 / smin)

    if previous_screen_triangle is not None:
        previous_matrix = _triangle_matrix(previous_screen_triangle)
        previous_area2 = abs(float(np.linalg.det(previous_matrix)))
        result["previous_projected_double_area_px2"] = previous_area2
        if previous_area2 >= minimum_area:
            step = posed_matrix @ np.linalg.inv(previous_matrix)
            smax, smin, _condition = _sv_metrics(step)
            result["adjacent_frame_principal_stretch"] = max(smax, 1.0 / smin)

    return result


def validate_dynamic_appearance_policy(policy: Mapping[str, Any]) -> dict[str, float | int]:
    required = (
        "dynamic_min_visible_pixels_per_face",
        "dynamic_min_projected_double_area_px2",
        "dynamic_max_uv_to_screen_condition_number",
        "dynamic_max_relative_screen_condition_number",
        "dynamic_max_relative_principal_stretch",
        "dynamic_max_adjacent_frame_principal_stretch",
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
        "dynamic_max_uv_to_screen_condition_number": float(
            policy["dynamic_max_uv_to_screen_condition_number"]
        ),
        "dynamic_max_relative_screen_condition_number": float(
            policy["dynamic_max_relative_screen_condition_number"]
        ),
        "dynamic_max_relative_principal_stretch": float(
            policy["dynamic_max_relative_principal_stretch"]
        ),
        "dynamic_max_adjacent_frame_principal_stretch": float(
            policy["dynamic_max_adjacent_frame_principal_stretch"]
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
