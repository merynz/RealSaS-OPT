from __future__ import annotations

import pytest

from tools.corpus.fit8_rendered_mechanical_census_contract_v1 import (
    DynamicMechanicalTailStatsV1,
    SubjectRenderedMechanicalCensusV1,
    SurfaceClassVisibilitySweepV1,
    ViewVisibilityStatsV1,
    surface_visibility_sweep_span,
)


def _views() -> tuple[ViewVisibilityStatsV1, ...]:
    return tuple(
        ViewVisibilityStatsV1(
            view_index=i,
            foreground_pixels=100,
            visible_component_count=3,
            visible_material_region_count=1,
            max_distinct_component_stack=2,
            p95_distinct_component_stack=1.5,
            pairwise_ordered_component_relation_count=4,
        )
        for i in range(8)
    )


def test_valid_subject_and_resolution_sweep():
    value = SubjectRenderedMechanicalCensusV1(
        subject_key="Bat",
        static_views=_views(),
        surface_class_resolution_sweep=(
            SurfaceClassVisibilitySweepV1(
                128, {"CLOSED": 0.7277, "OPEN": 0.1551, "NONMANIFOLD": 0.1172}
            ),
            SurfaceClassVisibilitySweepV1(
                256, {"CLOSED": 0.7270, "OPEN": 0.1557, "NONMANIFOLD": 0.1173}
            ),
        ),
    )
    value.validate()
    assert surface_visibility_sweep_span(value.surface_class_resolution_sweep, "OPEN") < 0.001


def test_requires_exact_eight_canonical_views():
    with pytest.raises(ValueError, match="CANONICAL_VIEW_COUNT"):
        SubjectRenderedMechanicalCensusV1("x", _views()[:-1]).validate()


def test_surface_fractions_must_normalize():
    with pytest.raises(ValueError, match="NOT_NORMALIZED"):
        SurfaceClassVisibilitySweepV1(128, {"OPEN": 0.2}).validate()


def test_dynamic_raw_max_is_not_authority():
    DynamicMechanicalTailStatsV1(10, 1.2, 2.0, 0.5, 2.0, True).validate()
    with pytest.raises(ValueError, match="RAW_MAX"):
        DynamicMechanicalTailStatsV1(10, 1.2, 2.0, 0.5, 2.0, False).validate()


def test_dynamic_stats_require_explicit_authority():
    with pytest.raises(ValueError, match="AUTHORITY_INVALID"):
        SubjectRenderedMechanicalCensusV1(
            "x",
            _views(),
            dynamic_mechanical_tail=DynamicMechanicalTailStatsV1(1, 1.0, 1.0, 1.0, 1.0),
        ).validate()
