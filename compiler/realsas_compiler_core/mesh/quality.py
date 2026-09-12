from __future__ import annotations

from dataclasses import asdict, dataclass
import math

from .types import QualificationError


@dataclass(frozen=True)
class MeshQualityPolicyV1:
    """Fail-closed product-mesh admission policy.

    Coverage values are measured against the exact observation alpha domain. Raster
    topology values are measured on the exact directional rest mesh consumed by
    appearance/deformation. The policy is deliberately independent of any source mesh.
    """

    min_source_alpha_recall: float = 0.94
    min_precision_inside_alpha: float = 0.995
    min_alpha_iou: float = 0.935
    max_largest_uncovered_component_fraction: float = 0.015
    min_large_alpha_component_recall: float = 0.90
    large_alpha_component_min_fraction: float = 0.0025
    max_degenerate_faces: int = 0
    max_duplicate_faces: int = 0
    max_nonmanifold_edges: int = 0
    min_raster_triangle_angle_deg: float = 0.25
    max_raster_triangle_aspect_ratio: float = 250.0
    schema_version: str = "RealSaS.MeshQualityPolicy.v1"

    def validate(self) -> None:
        for name in (
            "min_source_alpha_recall",
            "min_precision_inside_alpha",
            "min_alpha_iou",
            "max_largest_uncovered_component_fraction",
            "min_large_alpha_component_recall",
            "large_alpha_component_min_fraction",
        ):
            value = float(getattr(self, name))
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"{name} must be within [0,1]")
        if self.max_degenerate_faces < 0 or self.max_duplicate_faces < 0 or self.max_nonmanifold_edges < 0:
            raise ValueError("mesh defect limits must be nonnegative")
        if not (0.0 < self.min_raster_triangle_angle_deg < 60.0):
            raise ValueError("min_raster_triangle_angle_deg must be within (0,60)")
        if self.max_raster_triangle_aspect_ratio < 1.0:
            raise ValueError("max_raster_triangle_aspect_ratio must be >=1")

    def to_dict(self) -> dict:
        self.validate()
        return asdict(self)


FIT2_PRODUCT_MESH_QUALITY_POLICY_V1 = MeshQualityPolicyV1()


def _angle_deg(a: float, b: float, c: float) -> float:
    denom = max(2.0 * a * b, 1.0e-18)
    cosine = max(-1.0, min(1.0, (a * a + b * b - c * c) / denom))
    return math.degrees(math.acos(cosine))


def _triangle_metrics(pa, pb, pc) -> tuple[float, float, float]:
    ax, ay = map(float, pa); bx, by = map(float, pb); cx, cy = map(float, pc)
    ab = math.hypot(bx - ax, by - ay)
    bc = math.hypot(cx - bx, cy - by)
    ca = math.hypot(ax - cx, ay - cy)
    area2 = abs((bx - ax) * (cy - ay) - (by - ay) * (cx - ax))
    area = 0.5 * area2
    if area <= 1.0e-12 or min(ab, bc, ca) <= 1.0e-12:
        return area, 0.0, float("inf")
    angles = (
        _angle_deg(ab, ca, bc),
        _angle_deg(ab, bc, ca),
        _angle_deg(bc, ca, ab),
    )
    longest = max(ab, bc, ca)
    shortest_altitude = (2.0 * area) / max(longest, 1.0e-18)
    aspect = longest / max(shortest_altitude, 1.0e-18)
    return area, min(angles), aspect


def mesh_raster_quality_report(mesh, *, surface=None, view_index: int | None = None) -> dict:
    """Measure exact directional mesh topology in its authoritative raster frame.

    Candidate-produced ``metadata.raster_xy`` is accepted as a cached witness. If it
    is absent, the coordinate is re-derived from the exact SurfaceSupportBinding and
    the admitted RiggingSurfaceIR raster bindings for the target view.
    """
    surface_raster = {}
    if surface is not None:
        if view_index is None:
            view_index = int(mesh.view_index)
        for node in surface.surface_nodes:
            rows = [tuple(map(float, xy)) for v, xy in node.raster_bindings if int(v) == int(view_index)]
            if len(rows) > 1:
                raise QualificationError(f"MESH_QUALITY_DUPLICATE_SURFACE_RASTER:{node.surface_id}:{view_index}")
            if rows:
                surface_raster[str(node.surface_id)] = rows[0]

    by_id = {}
    for vertex in mesh.vertices:
        cached_xy = (getattr(vertex, "metadata", {}) or {}).get("raster_xy")
        derived_xy = None
        if surface_raster:
            x = y = total = 0.0
            for sid, coefficient in vertex.support_binding.coefficients:
                if str(sid) not in surface_raster:
                    raise QualificationError(f"MESH_QUALITY_SUPPORT_NOT_RASTER_BOUND:{view_index}:{sid}")
                c = float(coefficient)
                x += c * surface_raster[str(sid)][0]
                y += c * surface_raster[str(sid)][1]
                total += c
            if abs(total - 1.0) > 1.0e-8:
                raise QualificationError("MESH_QUALITY_SUPPORT_SIMPLEX_RESIDUAL")
            derived_xy = (x, y)
            if cached_xy is not None:
                cx, cy = map(float, cached_xy)
                if max(abs(cx - x), abs(cy - y)) > 1.0e-7:
                    raise QualificationError(
                        f"MESH_QUALITY_CACHED_RASTER_BINDING_DRIFT:{vertex.canonical_mesh_vertex_id}"
                    )
        xy = derived_xy if derived_xy is not None else cached_xy
        if xy is None or len(tuple(xy)) != 2:
            raise QualificationError(
                f"MESH_QUALITY_MISSING_RASTER_BINDING:{vertex.canonical_mesh_vertex_id}"
            )
        x, y = map(float, xy)
        if not math.isfinite(x) or not math.isfinite(y):
            raise QualificationError("MESH_QUALITY_NONFINITE_RASTER_BINDING")
        by_id[vertex.canonical_mesh_vertex_id] = (x, y)

    face_keys: set[tuple[str, str, str]] = set()
    edge_incidence: dict[tuple[str, str], int] = {}
    areas = []
    angles = []
    aspects = []
    duplicate_faces = 0
    degenerate_faces = 0
    for face in mesh.faces:
        if len(face) != 3 or any(vertex_id not in by_id for vertex_id in face):
            raise QualificationError("MESH_QUALITY_REQUIRES_VALID_TRIANGULATED_MESH")
        key = tuple(sorted(map(str, face)))
        if key in face_keys:
            duplicate_faces += 1
        face_keys.add(key)
        a, b, c = map(str, face)
        area, min_angle, aspect = _triangle_metrics(by_id[a], by_id[b], by_id[c])
        areas.append(area)
        angles.append(min_angle)
        aspects.append(aspect)
        if area <= 1.0e-12:
            degenerate_faces += 1
        for u, v in ((a, b), (b, c), (c, a)):
            edge = tuple(sorted((u, v)))
            edge_incidence[edge] = edge_incidence.get(edge, 0) + 1

    nonmanifold = int(sum(count > 2 for count in edge_incidence.values()))
    boundary = int(sum(count == 1 for count in edge_incidence.values()))
    return {
        "vertex_count": len(by_id),
        "face_count": len(mesh.faces),
        "edge_count": len(edge_incidence),
        "degenerate_faces": int(degenerate_faces),
        "duplicate_faces": int(duplicate_faces),
        "nonmanifold_edges": nonmanifold,
        "boundary_edges": boundary,
        "min_raster_triangle_area": float(min(areas, default=0.0)),
        "mean_raster_triangle_area": float(sum(areas) / len(areas)) if areas else 0.0,
        "min_raster_triangle_angle_deg": float(min(angles, default=0.0)),
        "max_raster_triangle_aspect_ratio": float(max(aspects, default=0.0)),
    }


def _large_component_recalls(coverage: dict, policy: MeshQualityPolicyV1) -> tuple[float, int]:
    rows = tuple(coverage.get("foreground_component_recalls") or ())
    eligible = [
        row for row in rows
        if float(row.get("foreground_fraction", 0.0)) >= float(policy.large_alpha_component_min_fraction)
    ]
    return (
        min((float(row.get("recall", 0.0)) for row in eligible), default=1.0),
        len(eligible),
    )


def coverage_gate_failures(coverage: dict, *, policy: MeshQualityPolicyV1 = FIT2_PRODUCT_MESH_QUALITY_POLICY_V1) -> tuple[str, ...]:
    policy.validate()
    failures: list[str] = []
    if float(coverage.get("source_alpha_recall", -1.0)) < policy.min_source_alpha_recall:
        failures.append("source_alpha_recall")
    if float(coverage.get("precision_inside_alpha", -1.0)) < policy.min_precision_inside_alpha:
        failures.append("precision_inside_alpha")
    if float(coverage.get("alpha_iou", -1.0)) < policy.min_alpha_iou:
        failures.append("alpha_iou")
    if float(coverage.get("largest_uncovered_component_fraction", 1.0)) > policy.max_largest_uncovered_component_fraction:
        failures.append("largest_uncovered_component_fraction")
    min_component_recall, component_count = _large_component_recalls(coverage, policy)
    if component_count and min_component_recall < policy.min_large_alpha_component_recall:
        failures.append("large_alpha_component_recall")
    return tuple(failures)


def raster_quality_gate_failures(report: dict, *, policy: MeshQualityPolicyV1 = FIT2_PRODUCT_MESH_QUALITY_POLICY_V1) -> tuple[str, ...]:
    policy.validate()
    failures: list[str] = []
    if int(report.get("degenerate_faces", 0)) > policy.max_degenerate_faces:
        failures.append("degenerate_faces")
    if int(report.get("duplicate_faces", 0)) > policy.max_duplicate_faces:
        failures.append("duplicate_faces")
    if int(report.get("nonmanifold_edges", 0)) > policy.max_nonmanifold_edges:
        failures.append("nonmanifold_edges")
    if float(report.get("min_raster_triangle_angle_deg", 0.0)) < policy.min_raster_triangle_angle_deg:
        failures.append("min_raster_triangle_angle_deg")
    if float(report.get("max_raster_triangle_aspect_ratio", float("inf"))) > policy.max_raster_triangle_aspect_ratio:
        failures.append("max_raster_triangle_aspect_ratio")
    return tuple(failures)


def evaluate_mesh_quality(
    *,
    coverage: dict,
    raster_report: dict,
    policy: MeshQualityPolicyV1 = FIT2_PRODUCT_MESH_QUALITY_POLICY_V1,
) -> dict:
    coverage_failures = coverage_gate_failures(coverage, policy=policy)
    topology_failures = raster_quality_gate_failures(raster_report, policy=policy)
    failures = tuple(coverage_failures) + tuple(topology_failures)
    min_component_recall, large_component_count = _large_component_recalls(coverage, policy)
    return {
        "passed": not failures,
        "failure_invariants": failures,
        "policy": policy.to_dict(),
        "min_large_alpha_component_recall": float(min_component_recall),
        "large_alpha_component_count": int(large_component_count),
        **coverage,
        **raster_report,
    }
