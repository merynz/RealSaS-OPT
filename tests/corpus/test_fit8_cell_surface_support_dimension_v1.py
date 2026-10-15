from tools.corpus.fit8_cell_surface_support_dimension_v1 import (
    SUPPORT_AREA,
    SUPPORT_LINE,
    SUPPORT_NONE,
    SUPPORT_POINT,
    SurfaceSupportDimensionPolicyV1,
    qualify_patch_cell_support_v1,
    qualify_triangle_cell_support_v1,
)
from tools.corpus.fit8_local_patch_teacher_v1 import ExtractionCellV1, SourceTriangleV1

CELL = ExtractionCellV1((0.0, 0.0, 0.0), 1.0, 2.0)
POLICY = SurfaceSupportDimensionPolicyV1(1e-8, "SYNTHETIC_SUPPORT_DIMENSION_TEST")


def _tri(triangle_id, a, b, c):
    return SourceTriangleV1(triangle_id, (a, b, c), 0)


def test_positive_area_surface_support_is_distinct_from_contact():
    triangle = _tri(0, (-0.5, -0.5, 0.0), (0.5, -0.5, 0.0), (0.0, 0.5, 0.0))
    result = qualify_triangle_cell_support_v1(CELL, triangle, policy=POLICY)
    assert result.support_kind == SUPPORT_AREA


def test_edge_only_contact_is_line_support_not_surface_patch():
    triangle = _tri(0, (1.0, -0.5, 1.0), (1.0, 0.5, 1.0), (2.0, 0.0, 2.0))
    result = qualify_triangle_cell_support_v1(CELL, triangle, policy=POLICY)
    assert result.support_kind == SUPPORT_LINE


def test_corner_only_contact_is_point_touch():
    triangle = _tri(0, (1.0, 1.0, 1.0), (2.0, 1.5, 1.0), (1.5, 2.0, 1.0))
    result = qualify_triangle_cell_support_v1(CELL, triangle, policy=POLICY)
    assert result.support_kind == SUPPORT_POINT


def test_outside_triangle_has_no_support():
    triangle = _tri(0, (2.0, 2.0, 2.0), (3.0, 2.0, 2.0), (2.0, 3.0, 2.0))
    result = qualify_triangle_cell_support_v1(CELL, triangle, policy=POLICY)
    assert result.support_kind == SUPPORT_NONE


def test_patch_support_uses_highest_geometric_dimension_without_erasing_lower_dimensional_evidence():
    point = _tri(0, (1.0, 1.0, 1.0), (2.0, 1.5, 1.0), (1.5, 2.0, 1.0))
    area = _tri(1, (-0.5, -0.5, 0.0), (0.5, -0.5, 0.0), (0.0, 0.5, 0.0))
    result = qualify_patch_cell_support_v1(CELL, (point, area), policy=POLICY)
    assert result.support_kind == SUPPORT_AREA
    assert [row.support_kind for row in result.triangle_support] == [SUPPORT_POINT, SUPPORT_AREA]
