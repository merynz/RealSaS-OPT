from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from compiler.realsas_compiler_core.mechanical_partition_v1 import (
    build_structural_partition,
)
from compiler.realsas_compiler_core.tessa_surface_field_v1 import (
    build_tessa_material_support_field_v1,
    material_support_binding_v1,
    tessa_material_support_field_lineage_hash,
    validate_tessa_material_support_field_v1,
)
from compiler.realsas_compiler_core.types import (
    QualificationError,
    RiggingSurfaceIR,
    SurfaceNode,
    SurfaceRelation,
)


def _node(surface_id: str, p):
    return SurfaceNode(
        surface_id=surface_id,
        P=tuple(map(float, p)),
        support_views=(0, 1),
        provenance_refs=(f"prov:{surface_id}",),
        source_observation_ids=(f"obs:{surface_id}",),
        derived_normal=(0.0, 0.0, 1.0),
        validity_flags=("OBSERVED_SIGNED_ZERO_SURFACE",),
    )


def _surface_two_components() -> RiggingSurfaceIR:
    nodes = (
        _node("a0", (0.0, 0.0, 0.0)),
        _node("a1", (1.0, 0.0, 0.0)),
        _node("a2", (0.0, 1.0, 0.0)),
        _node("b0", (10.0, 0.0, 0.0)),
        _node("b1", (11.0, 0.0, 0.0)),
        _node("b2", (10.0, 1.0, 0.0)),
    )
    rels = []
    for prefix in ("a", "b"):
        for index, (u, v) in enumerate(((0, 1), (1, 2), (2, 0))):
            rels.append(
                SurfaceRelation(
                    relation_id=f"r:{prefix}:{index}",
                    a_surface_id=f"{prefix}{u}",
                    b_surface_id=f"{prefix}{v}",
                    relation_kind="SIGNED_ZERO_SURFACE_TOPOLOGY_NEIGHBOR",
                    score=1.0,
                )
            )
    return RiggingSurfaceIR(
        surface_nodes=nodes,
        local_relations=tuple(rels),
        geometry_lineage_hash="surface-two-components-v1",
    )


def test_material_support_field_is_component_safe_and_vertex_index_free():
    surface = _surface_two_components()
    partition = build_structural_partition(surface)
    assert len(partition.components) == 2

    vertices = np.asarray(
        [
            [0.25, 0.25, 0.05],
            [0.75, 0.15, -0.02],
            [10.25, 0.25, 0.03],
            [10.70, 0.10, -0.04],
        ],
        dtype=np.float64,
    )
    decoded_components = np.asarray([0, 0, 1, 1], dtype=np.int64)
    vertex_ids = tuple(f"tv{i}" for i in range(len(vertices)))

    field = build_tessa_material_support_field_v1(
        vertices_world=vertices,
        decoded_component_indices=decoded_components,
        surface=surface,
        partition=partition,
        topology_sequence_hash="topology-sequence-v1",
        proposal_vertex_ids=vertex_ids,
        max_support_nodes=3,
        max_graph_hops=2,
    )
    validate_tessa_material_support_field_v1(
        field,
        surface=surface,
        partition=partition,
        expected_vertex_ids=vertex_ids,
    )

    owner = {
        sid: component.component_id
        for component in partition.components
        for sid in component.surface_ids
    }
    component_by_decoded = {}
    for row in field.rows:
        component_by_decoded.setdefault(
            row.decoded_component_index, row.mechanical_component_id
        )
        assert component_by_decoded[row.decoded_component_index] == row.mechanical_component_id
        assert abs(sum(weight for _, weight in row.coefficients) - 1.0) < 1e-9
        assert all(owner[sid] == row.mechanical_component_id for sid, _ in row.coefficients)
        assert row.metadata["teacher_vertex_index_used"] is False
        assert row.metadata["geometry_position_derived_from_support"] is False

        binding = material_support_binding_v1(row)
        assert binding.metadata["authority_class"] == "MATERIAL_SUPPORT_ONLY"
        assert binding.metadata["teacher_vertex_index_used"] is False

    assert component_by_decoded[0] != component_by_decoded[1]
    assert field.metadata["teacher_vertex_index_used"] is False
    assert field.metadata["product_geometry_authority_claimed"] is False
    assert field.metadata["uncovered_mechanical_component_ids"] == []


def test_material_support_validator_rejects_cross_component_coefficients():
    surface = _surface_two_components()
    partition = build_structural_partition(surface)
    vertices = np.asarray([[0.2, 0.2, 0.0]], dtype=np.float64)

    field = build_tessa_material_support_field_v1(
        vertices_world=vertices,
        decoded_component_indices=np.asarray([0], dtype=np.int64),
        surface=surface,
        partition=partition,
        topology_sequence_hash="topology-sequence-v1",
        proposal_vertex_ids=("tv0",),
    )

    row = field.rows[0]
    other_component = next(
        component
        for component in partition.components
        if component.component_id != row.mechanical_component_id
    )
    bad_sid = other_component.surface_ids[0]
    bad_row = replace(row, coefficients=((bad_sid, 1.0),))
    provisional = replace(field, rows=(bad_row,), field_lineage_hash="")
    bad = replace(
        provisional,
        field_lineage_hash=tessa_material_support_field_lineage_hash(provisional),
    )

    with pytest.raises(
        QualificationError, match="TESSA_SUPPORT_CROSS_COMPONENT_COEFFICIENT"
    ):
        validate_tessa_material_support_field_v1(
            bad,
            surface=surface,
            partition=partition,
            expected_vertex_ids=("tv0",),
        )


def test_decoded_component_assignment_is_component_level_not_per_vertex():
    surface = _surface_two_components()
    partition = build_structural_partition(surface)

    # One generated connected component has one noisy point closer to component B,
    # but its median support belongs to A. The whole decoded component must remain
    # on one mechanical owner; per-vertex nearest ownership would tear material
    # support across a connected generated surface.
    vertices = np.asarray(
        [
            [0.10, 0.10, 0.0],
            [0.20, 0.10, 0.0],
            [0.10, 0.20, 0.0],
            [9.95, 0.05, 0.0],
        ],
        dtype=np.float64,
    )
    field = build_tessa_material_support_field_v1(
        vertices_world=vertices,
        decoded_component_indices=np.zeros(4, dtype=np.int64),
        surface=surface,
        partition=partition,
        topology_sequence_hash="topology-sequence-v1",
        proposal_vertex_ids=("v0", "v1", "v2", "v3"),
    )

    owners = {row.mechanical_component_id for row in field.rows}
    assert len(owners) == 1
    assert len(field.metadata["decoded_component_assignment"]) == 1
