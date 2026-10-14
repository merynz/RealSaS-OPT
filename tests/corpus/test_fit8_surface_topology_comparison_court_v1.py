from tools.corpus.fit8_surface_topology_comparison_court_v1 import (
    ComponentCorrespondenceV1,
    compare_surface_topology_invariants_v1,
)
from tools.corpus.fit8_surface_topology_invariants_v1 import (
    SurfaceComponentTopologyV1,
    SurfaceTopologyReportV1,
)


def _component(
    component_id,
    *,
    v=4,
    e=6,
    f=4,
    chi=2,
    boundary_edges=0,
    boundary_graph_components=0,
    boundary_loops=0,
    nonmanifold=0,
    orientability_valid=True,
    orientable=True,
    genus_valid=True,
    genus=0,
):
    return SurfaceComponentTopologyV1(
        component_id=component_id,
        triangle_ids=(component_id,),
        vertex_count=v,
        edge_count=e,
        face_count=f,
        euler_characteristic=chi,
        boundary_edge_count=boundary_edges,
        boundary_graph_component_count=boundary_graph_components,
        boundary_loop_count=boundary_loops,
        nonmanifold_edge_count=nonmanifold,
        orientability_valid=orientability_valid,
        orientable=orientable,
        genus_valid=genus_valid,
        genus=genus,
    )


def _report(*components):
    return SurfaceTopologyReportV1(
        schema_version="RealSaS.SurfaceTopologyInvariants.v1",
        component_count=len(components),
        components=tuple(components),
        total_boundary_edges=sum(c.boundary_edge_count for c in components),
        total_nonmanifold_edges=sum(c.nonmanifold_edge_count for c in components),
    )


def test_retessellation_may_change_vef_counts_without_topology_failure():
    source = _report(_component(0, v=4, e=6, f=4))
    candidate = _report(_component(0, v=10, e=24, f=16))
    result = compare_surface_topology_invariants_v1(
        source, candidate, [ComponentCorrespondenceV1(0, 0)]
    )
    assert result.exact_topology_invariants


def test_euler_drift_is_rejected():
    source = _report(_component(0, chi=2))
    candidate = _report(_component(0, chi=0, genus=1))
    result = compare_surface_topology_invariants_v1(
        source, candidate, [ComponentCorrespondenceV1(0, 0)]
    )
    assert not result.exact_topology_invariants
    assert any(m.field == "euler_characteristic" for m in result.mismatches)


def test_validity_state_change_is_not_silently_accepted():
    source = _report(
        _component(
            0,
            boundary_loops=None,
            nonmanifold=1,
            orientability_valid=False,
            orientable=None,
            genus_valid=False,
            genus=None,
        )
    )
    candidate = _report(_component(0))
    result = compare_surface_topology_invariants_v1(
        source, candidate, [ComponentCorrespondenceV1(0, 0)]
    )
    fields = {m.field for m in result.mismatches}
    assert "boundary_loop_valid" in fields
    assert "orientability_valid" in fields
    assert "genus_valid" in fields


def test_boundary_loop_count_drift_is_rejected_when_both_are_qualified():
    source = _report(
        _component(0, chi=1, boundary_edges=3, boundary_graph_components=1, boundary_loops=1, genus=0)
    )
    candidate = _report(
        _component(0, chi=0, boundary_edges=6, boundary_graph_components=2, boundary_loops=2, genus=0)
    )
    result = compare_surface_topology_invariants_v1(
        source, candidate, [ComponentCorrespondenceV1(0, 0)]
    )
    assert not result.exact_topology_invariants
    assert any(m.field == "boundary_loop_count" for m in result.mismatches)


def test_non_bijective_component_mapping_is_rejected():
    source = _report(_component(0), _component(1))
    candidate = _report(_component(0), _component(1))
    try:
        compare_surface_topology_invariants_v1(
            source,
            candidate,
            [ComponentCorrespondenceV1(0, 0), ComponentCorrespondenceV1(1, 0)],
        )
    except ValueError as exc:
        assert "CANDIDATE_MAPPING_NOT_INJECTIVE" in str(exc)
    else:
        raise AssertionError("expected non-bijective mapping rejection")
