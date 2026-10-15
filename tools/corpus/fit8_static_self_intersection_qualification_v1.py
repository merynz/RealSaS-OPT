from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np

from compiler.realsas_compiler_core.dynamic_geometry_integrity_v2 import unexpected_intersection_pairs

SCHEMA_VERSION = "RealSaS.StaticSelfIntersectionQualification.v1"


@dataclass(frozen=True)
class StaticSelfIntersectionPolicyV1:
    shared_contact_exclusion_fraction: float = 1.0e-6
    tolerance_fraction_of_shrink_clearance: float = 0.1
    sweep_multipliers: tuple[float, ...] = (0.0, 0.01, 0.1, 1.0)
    provenance: str = ""
    schema_version: str = SCHEMA_VERSION

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("STATIC_SELF_INTERSECTION_POLICY_SCHEMA_INVALID")
        if not isfinite(self.shared_contact_exclusion_fraction) or not (0.0 < self.shared_contact_exclusion_fraction < 0.01):
            raise ValueError("STATIC_SELF_INTERSECTION_EXCLUSION_FRACTION_INVALID")
        if not isfinite(self.tolerance_fraction_of_shrink_clearance) or not (0.0 < self.tolerance_fraction_of_shrink_clearance < 0.5):
            raise ValueError("STATIC_SELF_INTERSECTION_CLEARANCE_FRACTION_INVALID")
        if not self.sweep_multipliers or self.sweep_multipliers[0] != 0.0:
            raise ValueError("STATIC_SELF_INTERSECTION_SWEEP_MUST_INCLUDE_ZERO_FIRST")
        if tuple(sorted(set(self.sweep_multipliers))) != self.sweep_multipliers:
            raise ValueError("STATIC_SELF_INTERSECTION_SWEEP_NOT_CANONICAL")
        if self.sweep_multipliers[-1] != 1.0 or any(not isfinite(v) or v < 0.0 or v > 1.0 for v in self.sweep_multipliers):
            raise ValueError("STATIC_SELF_INTERSECTION_SWEEP_INVALID")
        if not self.provenance:
            raise ValueError("STATIC_SELF_INTERSECTION_POLICY_PROVENANCE_MISSING")


@dataclass(frozen=True)
class StaticSelfIntersectionQualificationV1:
    minimum_triangle_altitude: float
    maximum_test_tolerance: float
    tested_tolerances: tuple[float, ...]
    pair_sets: tuple[tuple[tuple[int, int], ...], ...]
    stable_across_tolerance_sweep: bool
    self_intersection_free: bool
    qualified: bool
    provenance: str
    schema_version: str = SCHEMA_VERSION

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("STATIC_SELF_INTERSECTION_SCHEMA_INVALID")
        if not isfinite(self.minimum_triangle_altitude) or self.minimum_triangle_altitude <= 0.0:
            raise ValueError("STATIC_SELF_INTERSECTION_MIN_ALTITUDE_INVALID")
        if not isfinite(self.maximum_test_tolerance) or self.maximum_test_tolerance <= 0.0:
            raise ValueError("STATIC_SELF_INTERSECTION_MAX_TOLERANCE_INVALID")
        if len(self.tested_tolerances) != len(self.pair_sets) or not self.tested_tolerances:
            raise ValueError("STATIC_SELF_INTERSECTION_SWEEP_ARITY_INVALID")
        if self.tested_tolerances[0] != 0.0 or self.tested_tolerances[-1] != self.maximum_test_tolerance:
            raise ValueError("STATIC_SELF_INTERSECTION_SWEEP_ENDPOINT_INVALID")
        if any(b < a for a, b in zip(self.tested_tolerances[:-1], self.tested_tolerances[1:])):
            raise ValueError("STATIC_SELF_INTERSECTION_TOLERANCE_ORDER_INVALID")
        for rows in self.pair_sets:
            if tuple(sorted(set(rows))) != rows:
                raise ValueError("STATIC_SELF_INTERSECTION_PAIR_SET_NOT_CANONICAL")
        expected_stable = all(rows == self.pair_sets[0] for rows in self.pair_sets[1:])
        if self.stable_across_tolerance_sweep != expected_stable:
            raise ValueError("STATIC_SELF_INTERSECTION_STABILITY_MISMATCH")
        if self.qualified != self.stable_across_tolerance_sweep:
            raise ValueError("STATIC_SELF_INTERSECTION_QUALIFICATION_MISMATCH")
        expected_free = self.qualified and len(self.pair_sets[0]) == 0
        if self.self_intersection_free != expected_free:
            raise ValueError("STATIC_SELF_INTERSECTION_FREE_MISMATCH")
        if not self.provenance:
            raise ValueError("STATIC_SELF_INTERSECTION_PROVENANCE_MISSING")


def _minimum_triangle_altitude(vertices: np.ndarray, faces: np.ndarray) -> float:
    triangles = vertices[faces]
    edge01 = np.linalg.norm(triangles[:, 1] - triangles[:, 0], axis=1)
    edge12 = np.linalg.norm(triangles[:, 2] - triangles[:, 1], axis=1)
    edge20 = np.linalg.norm(triangles[:, 0] - triangles[:, 2], axis=1)
    area2 = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )
    longest = np.maximum(np.maximum(edge01, edge12), edge20)
    if np.any(longest <= 0.0) or np.any(area2 <= 0.0):
        raise ValueError("STATIC_SELF_INTERSECTION_DEGENERATE_TRIANGLE")
    altitude = area2 / longest
    value = float(np.min(altitude))
    if not isfinite(value) or value <= 0.0:
        raise ValueError("STATIC_SELF_INTERSECTION_MIN_ALTITUDE_INVALID")
    return value


def qualify_static_self_intersection_v1(
    *,
    vertices: np.ndarray,
    faces: np.ndarray,
    policy: StaticSelfIntersectionPolicyV1,
) -> StaticSelfIntersectionQualificationV1:
    """Qualify static self-intersection without a free absolute tolerance choice.

    The maximum SAT tolerance is derived from the smallest triangle altitude and
    the same infinitesimal shared-contact shrink used by the Compiler intersection
    court. A source is qualified only when the exact reported pair set is stable
    from zero tolerance through that bounded scale-aware sweep.
    """
    policy.validate()
    xyz = np.asarray(vertices, dtype=np.float64)
    tri = np.asarray(faces, dtype=np.int64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not len(xyz) or not np.isfinite(xyz).all():
        raise ValueError("STATIC_SELF_INTERSECTION_VERTICES_INVALID")
    if tri.ndim != 2 or tri.shape[1] != 3 or not len(tri) or np.any(tri < 0) or np.any(tri >= len(xyz)):
        raise ValueError("STATIC_SELF_INTERSECTION_FACES_INVALID")

    minimum_altitude = _minimum_triangle_altitude(xyz, tri)
    shrink_clearance = minimum_altitude * float(policy.shared_contact_exclusion_fraction)
    maximum_tolerance = shrink_clearance * float(policy.tolerance_fraction_of_shrink_clearance)
    if not isfinite(maximum_tolerance) or maximum_tolerance <= 0.0:
        raise ValueError("STATIC_SELF_INTERSECTION_DERIVED_TOLERANCE_INVALID")

    tolerances = tuple(float(multiplier) * maximum_tolerance for multiplier in policy.sweep_multipliers)
    pair_sets = []
    for tolerance in tolerances:
        pairs = unexpected_intersection_pairs(
            vertices=xyz,
            faces=tri,
            tolerance=tolerance,
            shared_contact_exclusion_fraction=policy.shared_contact_exclusion_fraction,
        )
        pair_sets.append(tuple(sorted(set((int(a), int(b)) for a, b in pairs))))

    stable = all(rows == pair_sets[0] for rows in pair_sets[1:])
    result = StaticSelfIntersectionQualificationV1(
        minimum_triangle_altitude=minimum_altitude,
        maximum_test_tolerance=maximum_tolerance,
        tested_tolerances=tolerances,
        pair_sets=tuple(pair_sets),
        stable_across_tolerance_sweep=stable,
        self_intersection_free=bool(stable and len(pair_sets[0]) == 0),
        qualified=bool(stable),
        provenance=policy.provenance,
    )
    result.validate()
    return result


__all__ = [
    "SCHEMA_VERSION",
    "StaticSelfIntersectionPolicyV1",
    "StaticSelfIntersectionQualificationV1",
    "qualify_static_self_intersection_v1",
]
