from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from tools.corpus.fit8_static_draw_order_capacity_court_v2 import _deterministic_cycle_witness
from tools.corpus.fit8_static_ray_census_v1 import (
    CanonicalOrbitFrameV1,
    StaticRayCensusPolicyV1,
    VIEW_YAW_DEG,
    _raster_hits_one_view,
    face_edge_components_v1,
    solve_canonical_orbit_frame_v1,
)


SCHEMA_VERSION = "RealSaS.FIT8DynamicFixedFramePresentation.v1"
FRAME_SCOPE = "SUBJECT_WIDE_REST_PLUS_ALL_PREREGISTERED_POSES"


@dataclass(frozen=True)
class DynamicPresentationSnapshotV1:
    frame_sha256: str
    policy_sha256: str
    component_count: int
    mixed_pair_view_count: int
    mixed_pair_count_sum: int
    directed_cycle_view_count: int
    ordered_relation_count_sum: int
    maximum_distinct_component_stack: int
    ordered_pairs_by_view: tuple[tuple[tuple[int, int, int], ...], ...]

    def validate(self) -> None:
        if len(self.frame_sha256) != 64 or len(self.policy_sha256) != 64:
            raise ValueError("FIT8_DYNAMIC_PRESENTATION_SHA_INVALID")
        if self.component_count < 1 or len(self.ordered_pairs_by_view) != 8:
            raise ValueError("FIT8_DYNAMIC_PRESENTATION_DOMAIN_INVALID")
        if not 0 <= self.mixed_pair_view_count <= self.directed_cycle_view_count <= 8:
            raise ValueError("FIT8_DYNAMIC_PRESENTATION_VIEW_COUNTS_INVALID")
        if self.mixed_pair_count_sum < 0 or self.ordered_relation_count_sum < 0:
            raise ValueError("FIT8_DYNAMIC_PRESENTATION_PAIR_COUNTS_INVALID")
        if self.maximum_distinct_component_stack < 1:
            raise ValueError("FIT8_DYNAMIC_PRESENTATION_STACK_INVALID")
        total = 0
        for rows in self.ordered_pairs_by_view:
            seen = set()
            for a, b, count in rows:
                if a < 0 or b < 0 or a == b or count <= 0 or (a, b) in seen:
                    raise ValueError("FIT8_DYNAMIC_PRESENTATION_ORDERED_PAIR_INVALID")
                seen.add((a, b))
            total += len(rows)
        if total != self.ordered_relation_count_sum:
            raise ValueError("FIT8_DYNAMIC_PRESENTATION_ORDERED_PAIR_TOTAL_MISMATCH")


@dataclass(frozen=True)
class DynamicPresentationDeltaV1:
    new_directed_edges_vs_rest: int
    reversed_directed_edges_vs_rest: int
    strict_pair_flips_vs_rest: int

    def validate(self) -> None:
        if min(
            self.new_directed_edges_vs_rest,
            self.reversed_directed_edges_vs_rest,
            self.strict_pair_flips_vs_rest,
        ) < 0:
            raise ValueError("FIT8_DYNAMIC_PRESENTATION_DELTA_NEGATIVE")


def solve_subject_dynamic_frame_v1(
    *,
    rest_vertices: np.ndarray,
    posed_vertices: Sequence[np.ndarray],
    frame_scale: float,
) -> CanonicalOrbitFrameV1:
    rest = np.asarray(rest_vertices, dtype=np.float64)
    if rest.ndim != 2 or rest.shape[1] != 3 or not len(rest):
        raise ValueError("FIT8_DYNAMIC_FRAME_REST_INVALID")
    rows = [rest]
    for pose in posed_vertices:
        array = np.asarray(pose, dtype=np.float64)
        if array.shape != rest.shape or not np.isfinite(array).all():
            raise ValueError("FIT8_DYNAMIC_FRAME_POSE_INVALID")
        rows.append(array)
    if len(rows) < 2:
        raise ValueError("FIT8_DYNAMIC_FRAME_POSES_EMPTY")
    frame = solve_canonical_orbit_frame_v1(np.vstack(rows), frame_scale=float(frame_scale))
    frame.validate()
    return frame


def _edges(rows: tuple[tuple[int, int, int], ...]) -> set[tuple[int, int]]:
    return {(int(a), int(b)) for a, b, _ in rows}


def evaluate_fixed_frame_presentation_v1(
    *,
    vertices: np.ndarray,
    faces: np.ndarray,
    frame: CanonicalOrbitFrameV1,
    policy: StaticRayCensusPolicyV1,
    face_component_id: np.ndarray | None = None,
) -> DynamicPresentationSnapshotV1:
    frame.validate()
    policy.validate()
    xyz = np.asarray(vertices, dtype=np.float64)
    tri = np.asarray(faces, dtype=np.int64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not len(xyz) or not np.isfinite(xyz).all():
        raise ValueError("FIT8_DYNAMIC_PRESENTATION_VERTICES_INVALID")
    if tri.ndim != 2 or tri.shape[1] != 3 or not len(tri) or np.any(tri < 0) or np.any(tri >= len(xyz)):
        raise ValueError("FIT8_DYNAMIC_PRESENTATION_FACES_INVALID")
    component = (
        face_edge_components_v1(tri)
        if face_component_id is None
        else np.asarray(face_component_id, dtype=np.int32)
    )
    if component.shape != (len(tri),) or np.any(component < 0):
        raise ValueError("FIT8_DYNAMIC_PRESENTATION_COMPONENT_INVALID")
    material = np.full(len(tri), -1, dtype=np.int32)
    owner = np.full(len(tri), -1, dtype=np.int32)

    ordered_by_view: list[tuple[tuple[int, int, int], ...]] = []
    mixed_views = 0
    mixed_total = 0
    cycle_views = 0
    max_stack = 0
    for yaw in VIEW_YAW_DEG:
        result = _raster_hits_one_view(
            vertices=xyz,
            faces=tri,
            face_component_id=component,
            face_material_id=material,
            face_owner_id=owner,
            frame=frame,
            yaw_deg=float(yaw),
            policy=policy,
        )
        ordered = tuple(
            (int(a), int(b), int(count))
            for a, b, count in np.asarray(
                result["ordered_component_pair_counts"], dtype=np.int64
            ).tolist()
        )
        mixed = np.asarray(result["mixed_component_pair_counts"], dtype=np.int64)
        stack = np.asarray(result["stack_sizes"], dtype=np.int64)
        if not len(stack):
            raise ValueError("FIT8_DYNAMIC_PRESENTATION_STACK_EMPTY")
        edges = _edges(ordered)
        mixed_views += int(len(mixed) > 0)
        mixed_total += int(len(mixed))
        cycle_views += int(_deterministic_cycle_witness(edges) is not None)
        max_stack = max(max_stack, int(stack.max()))
        ordered_by_view.append(ordered)

    result = DynamicPresentationSnapshotV1(
        frame_sha256=frame.sha256(),
        policy_sha256=policy.sha256(),
        component_count=int(len(set(int(v) for v in component.tolist()))),
        mixed_pair_view_count=int(mixed_views),
        mixed_pair_count_sum=int(mixed_total),
        directed_cycle_view_count=int(cycle_views),
        ordered_relation_count_sum=int(sum(len(rows) for rows in ordered_by_view)),
        maximum_distinct_component_stack=int(max_stack),
        ordered_pairs_by_view=tuple(ordered_by_view),
    )
    result.validate()
    return result


def compare_dynamic_presentation_to_rest_v1(
    rest: DynamicPresentationSnapshotV1,
    posed: DynamicPresentationSnapshotV1,
) -> DynamicPresentationDeltaV1:
    rest.validate()
    posed.validate()
    if (
        rest.frame_sha256 != posed.frame_sha256
        or rest.policy_sha256 != posed.policy_sha256
        or rest.component_count != posed.component_count
    ):
        raise ValueError("FIT8_DYNAMIC_PRESENTATION_CONTEXT_DRIFT")
    new_edges = 0
    reversed_edges = 0
    strict_flips = 0
    for base_rows, pose_rows in zip(rest.ordered_pairs_by_view, posed.ordered_pairs_by_view):
        base = _edges(base_rows)
        current = _edges(pose_rows)
        new_edges += len(current - base)
        reversed_edges += sum((b, a) in base and (a, b) not in base for a, b in current)
        for a, b in {tuple(sorted(edge)) for edge in base | current}:
            before = ((a, b) in base, (b, a) in base)
            after = ((a, b) in current, (b, a) in current)
            if before in {(True, False), (False, True)} and after in {(True, False), (False, True)} and before != after:
                strict_flips += 1
    result = DynamicPresentationDeltaV1(
        new_directed_edges_vs_rest=int(new_edges),
        reversed_directed_edges_vs_rest=int(reversed_edges),
        strict_pair_flips_vs_rest=int(strict_flips),
    )
    result.validate()
    return result


__all__ = [
    "DynamicPresentationDeltaV1",
    "DynamicPresentationSnapshotV1",
    "FRAME_SCOPE",
    "SCHEMA_VERSION",
    "compare_dynamic_presentation_to_rest_v1",
    "evaluate_fixed_frame_presentation_v1",
    "solve_subject_dynamic_frame_v1",
]
