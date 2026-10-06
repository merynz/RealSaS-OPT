from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from compiler.realsas_compiler_core.mechanical_partition_v1 import build_structural_partition
from compiler.realsas_compiler_core.product_authority_v1 import (
    ComponentCarrierDecisionIR,
    build_component_carrier_policy,
)
from compiler.realsas_compiler_core.tessa_candidate_bridge_v1 import (
    build_tessa_candidate_bridge_v1,
)
from compiler.realsas_compiler_core.tessa_geometry_qualification_v1 import (
    qualify_tessa_learned_geometry_v1,
    tessa_learned_geometry_qualification_hash_v1,
    validate_tessa_learned_geometry_qualification_v1,
)
from compiler.realsas_compiler_core.tessa_surface_field_v1 import (
    build_tessa_material_support_field_v1,
)
from compiler.realsas_compiler_core.types import (
    QualificationError,
    RiggingSurfaceIR,
    SurfaceNode,
    SurfaceRelation,
)
from models.tessa.v1.contracts_v1 import TESSAMeshProposalV1, TESSAVertexProposalV1


def _context():
    nodes = (
        SurfaceNode("a", (0.0, 0.0, 0.0), (0, 1), ("pa",), ("oa",), derived_normal=(0.0, 0.0, 1.0)),
        SurfaceNode("b", (1.0, 0.0, 0.0), (0, 1), ("pb",), ("ob",), derived_normal=(0.0, 0.0, 1.0)),
        SurfaceNode("c", (0.0, 1.0, 0.0), (0, 1), ("pc",), ("oc",), derived_normal=(0.0, 0.0, 1.0)),
    )
    relations = (
        SurfaceRelation("ab", "a", "b", "SIGNED_ZERO_SURFACE_TOPOLOGY_NEIGHBOR", 1.0),
        SurfaceRelation("bc", "b", "c", "SIGNED_ZERO_SURFACE_TOPOLOGY_NEIGHBOR", 1.0),
        SurfaceRelation("ca", "c", "a", "SIGNED_ZERO_SURFACE_TOPOLOGY_NEIGHBOR", 1.0),
    )
    surface = RiggingSurfaceIR(nodes, relations, "surface-v1")
    partition = build_structural_partition(surface)
    carrier = build_component_carrier_policy(
        partition=partition,
        decisions=tuple(
            ComponentCarrierDecisionIR(component.component_id, "MESH", ("TEST",))
            for component in partition.components
        ),
    )
    proposal = TESSAMeshProposalV1(
        vertices=(
            TESSAVertexProposalV1("v0", (0.15, 0.15, 0.20), "a", "decoded:0"),
            TESSAVertexProposalV1("v1", (0.75, 0.10, 0.18), "b", "decoded:0"),
            TESSAVertexProposalV1("v2", (0.10, 0.75, 0.22), "c", "decoded:0"),
        ),
        faces=(("v0", "v1", "v2"),),
        source_geometry_lineage_hash=surface.geometry_lineage_hash,
        model_provenance="checkpoint:test",
        topology_sequence_hash="topology:test",
    )
    field = build_tessa_material_support_field_v1(
        vertices_world=np.asarray([v.P for v in proposal.vertices], dtype=np.float64),
        decoded_component_indices=np.zeros(3, dtype=np.int64),
        surface=surface,
        partition=partition,
        topology_sequence_hash=proposal.topology_sequence_hash,
        proposal_vertex_ids=("v0", "v1", "v2"),
        max_support_nodes=3,
        max_graph_hops=2,
    )
    candidate, bridge = build_tessa_candidate_bridge_v1(
        proposal=proposal,
        support_field=field,
        surface=surface,
        partition=partition,
        carrier_policy=carrier,
        producer_policy_hash="policy-hash",
    )
    return surface, partition, carrier, candidate, bridge


def _g5_rows(status="PASS"):
    # G1B consumes exact G5 rows only as independent source-fidelity evidence.
    # Detailed G5 row schema remains owned by product_coverage_v1; the hash binds
    # the rows byte-for-byte and every row must already report PASS.
    return (
        {
            "view_index": 0,
            "component_id": "component:test",
            "carrier_class": "MESH",
            "recall": 1.0,
            "precision": 1.0,
            "largest_coherent_hole_fraction": 0.0,
            "interior_uncovered_fraction": 0.0,
            "status": status,
        },
    )


def test_g1b_seals_exact_candidate_only_after_independent_g5_pass():
    surface, partition, carrier, candidate, bridge = _context()
    rows = _g5_rows("PASS")
    result = qualify_tessa_learned_geometry_v1(
        candidate=candidate,
        bridge_evidence=bridge,
        surface=surface,
        partition=partition,
        carrier_policy=carrier,
        g5_rows=rows,
    )
    assert result.candidate_lineage_hash == candidate.candidate_lineage_hash
    assert result.bridge_evidence_hash == bridge.evidence_hash
    assert result.material_support_field_hash == bridge.material_support_field_hash
    assert result.metadata["status"] == "PASS_LEARNED_GEOMETRY_SOURCE_FIDELITY"
    assert result.metadata["legacy_g1_replaced"] is False
    assert result.metadata["material_support_is_geometry_authority"] is False
    assert result.metadata["motion_capability_claimed"] is False
    assert result.metadata["generalization_claimed"] is False


def test_g1b_rejects_any_failed_g5_cell():
    surface, partition, carrier, candidate, bridge = _context()
    with pytest.raises(QualificationError, match="TESSA_G1B_G5_REQUIRES_ALL_CELLS_PASS"):
        qualify_tessa_learned_geometry_v1(
            candidate=candidate,
            bridge_evidence=bridge,
            surface=surface,
            partition=partition,
            carrier_policy=carrier,
            g5_rows=_g5_rows("FAIL"),
        )


def test_g1b_rejects_candidate_lineage_drift():
    surface, partition, carrier, candidate, bridge = _context()
    rows = _g5_rows("PASS")
    result = qualify_tessa_learned_geometry_v1(
        candidate=candidate,
        bridge_evidence=bridge,
        surface=surface,
        partition=partition,
        carrier_policy=carrier,
        g5_rows=rows,
    )
    bad = replace(result, candidate_lineage_hash="wrong")
    bad = replace(bad, qualification_hash=tessa_learned_geometry_qualification_hash_v1(bad))
    with pytest.raises(QualificationError, match="TESSA_G1B_CANDIDATE_LINEAGE_MISMATCH"):
        validate_tessa_learned_geometry_qualification_v1(
            bad,
            candidate=candidate,
            bridge_evidence=bridge,
            surface=surface,
            partition=partition,
            carrier_policy=carrier,
            g5_rows=rows,
        )


def test_g1b_rejects_support_geometry_conflation():
    surface, partition, carrier, candidate, bridge = _context()
    vertex = candidate.vertices[0]
    bad_vertex = replace(
        vertex,
        metadata={**vertex.metadata, "geometry_position_derived_from_material_support": True},
    )
    bad_candidate = replace(candidate, vertices=(bad_vertex,) + candidate.vertices[1:])
    # The candidate lineage must also change; G1B must reject on semantic authority
    # before any attempt to mint a replacement lineage.
    with pytest.raises(QualificationError, match="TESSA_G1B_VERTEX_GEOMETRY_SUPPORT_CONFLATION"):
        qualify_tessa_learned_geometry_v1(
            candidate=bad_candidate,
            bridge_evidence=bridge,
            surface=surface,
            partition=partition,
            carrier_policy=carrier,
            g5_rows=_g5_rows("PASS"),
        )
