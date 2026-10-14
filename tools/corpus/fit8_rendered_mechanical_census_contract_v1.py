from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping
import math


SCHEMA_VERSION = "RealSaS.FIT8RenderedMechanicalCensus.v1"
CANONICAL_VIEW_COUNT = 8
SURFACE_CLASSES = {"CLOSED", "OPEN", "NONMANIFOLD", "AMBIGUOUS"}


def _finite(value: float) -> bool:
    return math.isfinite(float(value))


@dataclass(frozen=True)
class ViewVisibilityStatsV1:
    view_index: int
    foreground_pixels: int
    visible_component_count: int
    visible_material_region_count: int
    max_distinct_component_stack: int
    p95_distinct_component_stack: float
    pairwise_ordered_component_relation_count: int

    def validate(self) -> None:
        if not 0 <= self.view_index < CANONICAL_VIEW_COUNT:
            raise ValueError("FIT8_RENDERED_VIEW_INDEX_INVALID")
        if self.foreground_pixels <= 0:
            raise ValueError("FIT8_RENDERED_FOREGROUND_EMPTY")
        if self.visible_component_count <= 0:
            raise ValueError("FIT8_RENDERED_VISIBLE_COMPONENT_EMPTY")
        if self.visible_material_region_count < 0:
            raise ValueError("FIT8_RENDERED_VISIBLE_MATERIAL_NEGATIVE")
        if self.max_distinct_component_stack <= 0:
            raise ValueError("FIT8_RENDERED_STACK_EMPTY")
        if not _finite(self.p95_distinct_component_stack):
            raise ValueError("FIT8_RENDERED_STACK_P95_NONFINITE")
        if not 0.0 <= self.p95_distinct_component_stack <= self.max_distinct_component_stack:
            raise ValueError("FIT8_RENDERED_STACK_P95_INVALID")
        if self.pairwise_ordered_component_relation_count < 0:
            raise ValueError("FIT8_RENDERED_PAIR_COUNT_NEGATIVE")


@dataclass(frozen=True)
class SurfaceClassVisibilitySweepV1:
    resolution: int
    visible_fraction_by_surface_class: Mapping[str, float]

    def validate(self) -> None:
        if self.resolution < 32:
            raise ValueError("FIT8_RENDERED_SWEEP_RESOLUTION_TOO_SMALL")
        total = 0.0
        for key, value in self.visible_fraction_by_surface_class.items():
            if key not in SURFACE_CLASSES:
                raise ValueError("FIT8_RENDERED_SURFACE_CLASS_INVALID")
            if not _finite(value) or not 0.0 <= float(value) <= 1.0:
                raise ValueError("FIT8_RENDERED_SURFACE_FRACTION_INVALID")
            total += float(value)
        if abs(total - 1.0) > 1e-5:
            raise ValueError("FIT8_RENDERED_SURFACE_FRACTIONS_NOT_NORMALIZED")


@dataclass(frozen=True)
class DynamicMechanicalTailStatsV1:
    sample_count: int
    edge_stretch_p99_max_over_samples: float
    face_condition_p99_max_over_samples: float
    face_area_ratio_min_over_samples: float
    face_area_ratio_max_over_samples: float
    max_stat_is_diagnostic_only: bool = True

    def validate(self) -> None:
        if self.sample_count <= 0:
            raise ValueError("FIT8_DYNAMIC_SAMPLE_COUNT_EMPTY")
        values = (
            self.edge_stretch_p99_max_over_samples,
            self.face_condition_p99_max_over_samples,
            self.face_area_ratio_min_over_samples,
            self.face_area_ratio_max_over_samples,
        )
        if not all(_finite(x) for x in values):
            raise ValueError("FIT8_DYNAMIC_MECHANICAL_NONFINITE")
        if self.edge_stretch_p99_max_over_samples <= 0.0:
            raise ValueError("FIT8_DYNAMIC_STRETCH_NONPOSITIVE")
        if self.face_condition_p99_max_over_samples < 1.0:
            raise ValueError("FIT8_DYNAMIC_CONDITION_LT_ONE")
        if self.face_area_ratio_min_over_samples < 0.0:
            raise ValueError("FIT8_DYNAMIC_AREA_NEGATIVE")
        if self.face_area_ratio_max_over_samples < self.face_area_ratio_min_over_samples:
            raise ValueError("FIT8_DYNAMIC_AREA_RANGE_INVALID")
        if not self.max_stat_is_diagnostic_only:
            raise ValueError("FIT8_DYNAMIC_RAW_MAX_MUST_REMAIN_DIAGNOSTIC")


@dataclass(frozen=True)
class SubjectRenderedMechanicalCensusV1:
    subject_key: str
    static_views: tuple[ViewVisibilityStatsV1, ...]
    surface_class_resolution_sweep: tuple[SurfaceClassVisibilitySweepV1, ...] = ()
    dynamic_mechanical_tail: DynamicMechanicalTailStatsV1 | None = None
    dynamic_mechanics_authority: str = "UNAVAILABLE"
    notes: tuple[str, ...] = ()

    def validate(self) -> None:
        if not self.subject_key:
            raise ValueError("FIT8_RENDERED_SUBJECT_MISSING")
        if len(self.static_views) != CANONICAL_VIEW_COUNT:
            raise ValueError("FIT8_RENDERED_CANONICAL_VIEW_COUNT_INVALID")
        for row in self.static_views:
            row.validate()
        if sorted(v.view_index for v in self.static_views) != list(range(CANONICAL_VIEW_COUNT)):
            raise ValueError("FIT8_RENDERED_CANONICAL_VIEW_SET_INVALID")
        seen = set()
        for row in self.surface_class_resolution_sweep:
            row.validate()
            if row.resolution in seen:
                raise ValueError("FIT8_RENDERED_DUPLICATE_SWEEP_RESOLUTION")
            seen.add(row.resolution)
        if self.dynamic_mechanical_tail is not None:
            self.dynamic_mechanical_tail.validate()
            if self.dynamic_mechanics_authority not in {"DIAGNOSTIC", "QUALIFIED"}:
                raise ValueError("FIT8_DYNAMIC_AUTHORITY_INVALID")
        elif self.dynamic_mechanics_authority != "UNAVAILABLE":
            raise ValueError("FIT8_DYNAMIC_AUTHORITY_WITHOUT_STATS")


def surface_visibility_sweep_span(
    sweep: tuple[SurfaceClassVisibilitySweepV1, ...], surface_class: str
) -> float:
    if surface_class not in SURFACE_CLASSES:
        raise ValueError("FIT8_RENDERED_SURFACE_CLASS_INVALID")
    if not sweep:
        raise ValueError("FIT8_RENDERED_SWEEP_EMPTY")
    values = [
        float(row.visible_fraction_by_surface_class.get(surface_class, 0.0))
        for row in sweep
    ]
    return max(values) - min(values)
