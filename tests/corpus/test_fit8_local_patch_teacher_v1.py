import numpy as np

from tools.corpus.fit8_local_patch_teacher_v1 import (
    ExtractionCellV1,
    SourceTriangleV1,
    extract_cell_teacher_v1,
)

CELL = ExtractionCellV1((0.0, 0.0, 0.0), 1.0, 2.0)


def _plane_x(tid, x, component, owner=None):
    # Two triangles are used in real surfaces; one large triangle is enough to
    # cross several x-directed cube edges in this synthetic contract test.
    return SourceTriangleV1(
        triangle_id=tid,
        vertices_xyz=((x, -2.0, -2.0), (x, 2.0, -2.0), (x, 0.0, 2.0)),
        geometric_component_id=component,
        owner_region_id=owner,
    )


def test_two_parallel_sheets_produce_two_crossings_on_same_edge():
    r = extract_cell_teacher_v1(CELL, [_plane_x(0, -0.25, 0), _plane_x(1, 0.25, 1)])
    assert r.active
    assert r.max_distinct_crossings_per_edge == 2
    per_edge = {}
    for c in r.edge_crossings:
        per_edge.setdefault(c.edge_index, []).append(c)
    assert any(len(v) == 2 for v in per_edge.values())
    pair = next(v for v in per_edge.values() if len(v) == 2)
    assert [round(c.t, 3) for c in pair] == [0.375, 0.625]
    assert pair[0].geometric_component_ids != pair[1].geometric_component_ids


def test_same_global_component_can_have_two_local_patch_lower_bound_groups():
    tris = [_plane_x(0, -0.25, 7), _plane_x(1, 0.25, 7)]
    r = extract_cell_teacher_v1(CELL, tris, triangle_adjacency={})
    assert len(r.local_patch_lower_bound) == 2
    assert all(p.geometric_component_ids == (7,) for p in r.local_patch_lower_bound)


def test_source_adjacency_merges_patch_lower_bound_group():
    a = SourceTriangleV1(0, ((0,-1,-1),(0,1,-1),(0,1,1)), 3)
    b = SourceTriangleV1(1, ((0,-1,-1),(0,1,1),(0,-1,1)), 3)
    r = extract_cell_teacher_v1(CELL, [a,b], triangle_adjacency={0:[1],1:[0]})
    assert len(r.local_patch_lower_bound) == 1
    assert r.local_patch_lower_bound[0].source_triangle_ids == (0,1)


def test_coincident_crossing_deduplicates_but_preserves_identity_sets():
    a = _plane_x(0, 0.0, 1, owner=10)
    b = _plane_x(1, 0.0, 2, owner=11)
    r = extract_cell_teacher_v1(CELL, [a,b])
    assert r.max_distinct_crossings_per_edge == 1
    merged = [c for c in r.edge_crossings if c.geometric_component_ids == (1,2)]
    assert merged
    assert merged[0].owner_region_ids == (10,11)


def test_empty_cell_is_explicitly_inactive():
    far = SourceTriangleV1(0, ((5,5,5),(6,5,5),(5,6,5)), 0)
    r = extract_cell_teacher_v1(CELL, [far])
    assert not r.active
    assert r.intersecting_triangle_ids == ()
    assert r.edge_crossings == ()
    assert r.max_distinct_crossings_per_edge == 0


def test_no_fixed_local_patch_cap_is_imposed():
    tris=[]
    for i,x in enumerate(np.linspace(-0.8,0.8,6)):
        tris.append(_plane_x(i, float(x), i))
    r=extract_cell_teacher_v1(CELL,tris)
    assert len(r.local_patch_lower_bound) == 6
    assert r.max_distinct_crossings_per_edge == 6
