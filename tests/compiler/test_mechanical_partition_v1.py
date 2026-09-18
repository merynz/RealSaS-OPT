from __future__ import annotations

import pytest

from compiler.realsas_compiler_core.mechanical_partition_v1 import (
    build_structural_partition,
    propose_relation_boundary_constraints,
)
from compiler.realsas_compiler_core.product_authority_v1 import ComponentBoundaryConstraintIR
from compiler.realsas_compiler_core.types import QualificationError, RiggingSurfaceIR, SurfaceNode, SurfaceRelation


def _surface(*, alternate_path=False):
    nodes = tuple(
        SurfaceNode(f"s{i}", (float(i), 0.0, 0.0), (0,), ("src",), (f"o{i}",))
        for i in range(4)
    )
    relations = [
        SurfaceRelation("r01", "s0", "s1", "LOCAL_NEIGHBOR", 1.0, {}),
        SurfaceRelation("r12", "s1", "s2", "UNKNOWN_BRIDGE", 1.0, {"crosses_unknown": True}),
        SurfaceRelation("r23", "s2", "s3", "LOCAL_NEIGHBOR", 1.0, {}),
    ]
    if alternate_path:
        relations.append(SurfaceRelation("r02", "s0", "s2", "LOCAL_NEIGHBOR", 1.0, {}))
    return RiggingSurfaceIR(nodes, tuple(relations), "surface-hash")


def _override(a, b, decision):
    return ComponentBoundaryConstraintIR(
        f"override:{a}:{b}:{decision}",
        a,
        b,
        decision,
        ("qualified-boundary-evidence",),
        1.0,
        metadata={"evidence_class": "QUALIFIED_EXTERNAL_BOUNDARY_EVIDENCE"},
    )


def test_relation_baseline_preserves_safe_and_keeps_unsafe_unknown():
    rows = propose_relation_boundary_constraints(_surface())
    by_pair = {(r.a_surface_id, r.b_surface_id): r for r in rows}
    assert by_pair[("s0", "s1")].decision == "PRESERVE_CONTINUITY"
    assert by_pair[("s1", "s2")].decision == "UNKNOWN"
    assert by_pair[("s2", "s3")].decision == "PRESERVE_CONTINUITY"


def test_unknown_is_provisionally_preserved_but_remains_explicit():
    surface = _surface()
    partition = build_structural_partition(surface)
    assert len(partition.components) == 1
    assert partition.metadata["unknown_boundary_count"] == 1
    assert any(x.decision == "UNKNOWN" for x in partition.boundary_constraints)
    assert partition.surface_lineage_hash == surface.geometry_lineage_hash


def test_explicit_separate_cut_creates_two_components_without_mutating_surface():
    surface = _surface()
    before = surface.to_dict()
    partition = build_structural_partition(
        surface,
        boundary_overrides=(_override("s1", "s2", "SEPARATE"),),
    )
    assert sorted(len(c.surface_ids) for c in partition.components) == [2, 2]
    assert any(x.decision == "SEPARATE" for x in partition.boundary_constraints)
    assert surface.to_dict() == before


def test_separate_must_be_a_real_cut_not_bypassed_by_an_alternate_path():
    with pytest.raises(QualificationError, match="NOT_A_CUT"):
        build_structural_partition(
            _surface(alternate_path=True),
            boundary_overrides=(_override("s1", "s2", "SEPARATE"),),
        )


def test_override_cannot_invent_surface_adjacency():
    with pytest.raises(QualificationError, match="EXISTING_LOCAL_RELATION"):
        build_structural_partition(
            _surface(),
            boundary_overrides=(_override("s0", "s3", "SEPARATE"),),
        )
