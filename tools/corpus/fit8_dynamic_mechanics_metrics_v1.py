from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


SCHEMA_VERSION = "RealSaS.FIT8DynamicMechanicalMetrics.v1"


@dataclass(frozen=True)
class DynamicMechanicalMetricsV1:
    unique_edge_count: int
    face_count: int
    edge_stretch_p99: float
    edge_stretch_max_diagnostic: float
    face_condition_p99: float
    face_condition_max_diagnostic: float
    face_area_ratio_p01: float
    face_area_ratio_p99: float
    face_area_ratio_min: float
    face_area_ratio_max: float
    raw_maxima_diagnostic_only: bool = True

    def validate(self) -> None:
        if self.unique_edge_count <= 0 or self.face_count <= 0:
            raise ValueError("FIT8_DYNAMIC_MECHANICS_DOMAIN_EMPTY")
        values = (
            self.edge_stretch_p99,
            self.edge_stretch_max_diagnostic,
            self.face_condition_p99,
            self.face_condition_max_diagnostic,
            self.face_area_ratio_p01,
            self.face_area_ratio_p99,
            self.face_area_ratio_min,
            self.face_area_ratio_max,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("FIT8_DYNAMIC_MECHANICS_NONFINITE")
        if self.edge_stretch_p99 <= 0.0 or self.edge_stretch_max_diagnostic < self.edge_stretch_p99:
            raise ValueError("FIT8_DYNAMIC_MECHANICS_STRETCH_INVALID")
        if self.face_condition_p99 < 1.0 or self.face_condition_max_diagnostic < self.face_condition_p99:
            raise ValueError("FIT8_DYNAMIC_MECHANICS_CONDITION_INVALID")
        if not (
            0.0
            <= self.face_area_ratio_min
            <= self.face_area_ratio_p01
            <= self.face_area_ratio_p99
            <= self.face_area_ratio_max
        ):
            raise ValueError("FIT8_DYNAMIC_MECHANICS_AREA_ORDER_INVALID")
        if not self.raw_maxima_diagnostic_only:
            raise ValueError("FIT8_DYNAMIC_MECHANICS_RAW_MAX_MUST_REMAIN_DIAGNOSTIC")


def _validate_mesh(vertices: np.ndarray, faces: np.ndarray, name: str) -> tuple[np.ndarray, np.ndarray]:
    xyz = np.asarray(vertices, dtype=np.float64)
    tri = np.asarray(faces, dtype=np.int64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not len(xyz) or not np.isfinite(xyz).all():
        raise ValueError(f"FIT8_DYNAMIC_MECHANICS_{name}_VERTICES_INVALID")
    if tri.ndim != 2 or tri.shape[1] != 3 or not len(tri) or np.any(tri < 0) or np.any(tri >= len(xyz)):
        raise ValueError(f"FIT8_DYNAMIC_MECHANICS_{name}_FACES_INVALID")
    if np.any((tri[:, 0] == tri[:, 1]) | (tri[:, 1] == tri[:, 2]) | (tri[:, 2] == tri[:, 0])):
        raise ValueError(f"FIT8_DYNAMIC_MECHANICS_{name}_FACE_DEGENERATE_INDEX")
    return xyz, tri


def _unique_edges(faces: np.ndarray) -> np.ndarray:
    edges = np.concatenate(
        (
            faces[:, [0, 1]],
            faces[:, [1, 2]],
            faces[:, [2, 0]],
        ),
        axis=0,
    )
    edges = np.sort(edges, axis=1)
    return np.unique(edges, axis=0)


def measure_dynamic_mechanical_consequence_v1(
    *,
    rest_vertices: np.ndarray,
    posed_vertices: np.ndarray,
    faces: np.ndarray,
) -> DynamicMechanicalMetricsV1:
    rest, tri = _validate_mesh(rest_vertices, faces, "REST")
    posed, posed_tri = _validate_mesh(posed_vertices, faces, "POSED")
    if not np.array_equal(tri, posed_tri) or posed.shape != rest.shape:
        raise ValueError("FIT8_DYNAMIC_MECHANICS_TOPOLOGY_DRIFT")

    edges = _unique_edges(tri)
    rest_edge_length = np.linalg.norm(rest[edges[:, 1]] - rest[edges[:, 0]], axis=1)
    posed_edge_length = np.linalg.norm(posed[edges[:, 1]] - posed[edges[:, 0]], axis=1)
    if np.any(rest_edge_length <= 1.0e-14):
        raise ValueError("FIT8_DYNAMIC_MECHANICS_ZERO_REST_EDGE")
    stretch = posed_edge_length / rest_edge_length

    rest_triangle = rest[tri]
    posed_triangle = posed[tri]
    a0 = rest_triangle[:, 1] - rest_triangle[:, 0]
    b0 = rest_triangle[:, 2] - rest_triangle[:, 0]
    a1 = posed_triangle[:, 1] - posed_triangle[:, 0]
    b1 = posed_triangle[:, 2] - posed_triangle[:, 0]
    rest_double_area = np.linalg.norm(np.cross(a0, b0), axis=1)
    posed_double_area = np.linalg.norm(np.cross(a1, b1), axis=1)
    if np.any(rest_double_area <= 1.0e-14):
        raise ValueError("FIT8_DYNAMIC_MECHANICS_ZERO_REST_FACE_AREA")
    area_ratio = posed_double_area / rest_double_area

    length_a = np.linalg.norm(a0, axis=1)
    x = np.einsum("ij,ij->i", a0, b0) / length_a
    y_squared = np.einsum("ij,ij->i", b0, b0) - x * x
    y = np.sqrt(np.maximum(y_squared, 0.0))
    if np.any(length_a <= 1.0e-14) or np.any(y <= 1.0e-14):
        raise ValueError("FIT8_DYNAMIC_MECHANICS_REST_PARAMETERIZATION_DEGENERATE")

    condition = np.empty(len(tri), dtype=np.float64)
    for face_id in range(len(tri)):
        rest_basis = np.asarray(
            [[length_a[face_id], x[face_id]], [0.0, y[face_id]]],
            dtype=np.float64,
        )
        posed_gram = np.asarray(
            [
                [np.dot(a1[face_id], a1[face_id]), np.dot(a1[face_id], b1[face_id])],
                [np.dot(a1[face_id], b1[face_id]), np.dot(b1[face_id], b1[face_id])],
            ],
            dtype=np.float64,
        )
        inverse_rest = np.linalg.inv(rest_basis)
        metric = inverse_rest.T @ posed_gram @ inverse_rest
        eigenvalues = np.maximum(np.linalg.eigvalsh(metric), 0.0)
        singular = np.sqrt(eigenvalues)
        if singular[0] <= 1.0e-12:
            condition[face_id] = singular[-1] / 1.0e-12
        else:
            condition[face_id] = singular[-1] / singular[0]
    condition = np.maximum(condition, 1.0)

    result = DynamicMechanicalMetricsV1(
        unique_edge_count=int(len(edges)),
        face_count=int(len(tri)),
        edge_stretch_p99=float(np.quantile(stretch, 0.99)),
        edge_stretch_max_diagnostic=float(np.max(stretch)),
        face_condition_p99=float(np.quantile(condition, 0.99)),
        face_condition_max_diagnostic=float(np.max(condition)),
        face_area_ratio_p01=float(np.quantile(area_ratio, 0.01)),
        face_area_ratio_p99=float(np.quantile(area_ratio, 0.99)),
        face_area_ratio_min=float(np.min(area_ratio)),
        face_area_ratio_max=float(np.max(area_ratio)),
    )
    result.validate()
    return result


__all__ = [
    "DynamicMechanicalMetricsV1",
    "SCHEMA_VERSION",
    "measure_dynamic_mechanical_consequence_v1",
]
