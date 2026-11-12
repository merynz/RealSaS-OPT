from __future__ import annotations

"""Product bridge CPU court: same RSS H(x), no invented source correspondence.

The synthetic source substrate in these tests is built from known test
triangles and is NOT a real two-image source surface/admission receipt.
Passing this court proves safe data transport, not learned/source fidelity.
"""

from copy import deepcopy
from dataclasses import replace

import pytest

from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.product_authority_v1 import (
    ComponentCarrierDecisionIR, ComponentRegionIR,
    build_component_carrier_policy, build_mechanical_partition,
    validate_canonical_mesh_candidate,
)
from compiler.realsas_compiler_core.rss_vnext_canonical_mesh_candidate_bridge_v1 import (
    build_rss_vnext_canonical_mesh_candidate_bridge_v1,
    build_source_correspondence_receipt_payload_v1,
)
from compiler.realsas_compiler_core.rss_vnext_inference_evidence_v1 import (
    seal_inference_evidence_v1,
)
from compiler.realsas_compiler_core.types import (
    QualificationError, RiggingSurfaceIR, SurfaceNode,
    SurfaceSupportBinding,
)
from tests.models.test_rss_vnext_cross_expert_capability_court_v1 import _ready


def _parents():
    cfg, evidence, surface, _ = _ready()
    mesh = evidence.mesh
    # Test-source geometry deliberately matches TESSA BASE token geometry;
    # this is synthetic input setup, not a source-observation oracle.
    nodes = tuple(SurfaceNode(
        f"src-{i}", tuple(map(float, mesh["base_xyz"][i])),
        (0, 1), ("SYNTHETIC_FIXTURE_ONLY",), (f"source-pixel-{i}",),
    ) for i in range(len(mesh["base_xyz"])))
    src = RiggingSurfaceIR(
        nodes, (), "synthetic-test-source-geometry",
        metadata={"synthetic_not_source_truth": True},
    )
    groups = sorted(set(mesh["vertex_components"]))
    components = tuple(
        ComponentRegionIR(f"component-{c}", tuple(
            f"src-{i}" for i, owner in enumerate(mesh["vertex_components"])
            if owner == c
        ), (f"synthetic-component-{c}",))
        for c in groups
    )
    partition = build_mechanical_partition(
        surface=src, components=components, boundary_constraints=(),
        metadata={"synthetic_not_source_truth": True},
    )
    carrier = build_component_carrier_policy(
        partition=partition,
        decisions=tuple(ComponentCarrierDecisionIR(
            c.component_id, "MESH", ("synthetic-fixture-no-source-authority",),
        ) for c in components),
    )
    supports = {i: SurfaceSupportBinding(
        mode="IDENTITY_SURFACE_NODE", coefficients=((f"src-{i}", 1.0),),
        metadata={"synthetic_not_admitted": True},
    ) for i in range(len(nodes))}
    mapping = {int(c): f"component-{c}" for c in groups}
    kwargs = {
        "evidence": evidence, "computed_surface": surface,
        "model_config": cfg, "source_surface": src,
        "source_partition": partition, "source_carrier_policy": carrier,
        "source_supports_by_vertex": supports,
        "source_component_ids_by_candidate_index": mapping,
        "source_correspondence_receipt_sha256": None,
        "trusted_source_correspondence_registry": None,
    }
    # A synthetic, independently named source-fixture producer registers
    # exact source/mesh/support parent fields. This never certifies real art.
    payload = build_source_correspondence_receipt_payload_v1(
        evidence_hash=evidence.evidence_hash,
        candidate_mesh_sha256=content_sha256(mesh),
        source_surface_lineage_hash=src.geometry_lineage_hash,
        partition_lineage_hash=partition.partition_lineage_hash,
        carrier_policy_lineage_hash=carrier.carrier_policy_lineage_hash,
        source_supports_by_vertex=supports,
        source_component_ids_by_candidate_index=mapping,
        source_observation_receipt_sha256="f" * 64,
        producer_id="SYNTHETIC_INDEPENDENT_TEST_SOURCE_FIXTURE",
    )
    key = content_sha256(payload)
    kwargs["source_correspondence_receipt_sha256"] = key
    kwargs["trusted_source_correspondence_registry"] = {key: payload}
    return kwargs


def _compile(kwargs):
    return build_rss_vnext_canonical_mesh_candidate_bridge_v1(**kwargs)


def test_rss_experts_can_bind_real_candidate_triangles_into_existing_canonical_candidate():
    kw = _parents()
    before = deepcopy(kw["evidence"].to_dict())
    result = _compile(kw)
    assert result.state == "UNQUALIFIED_CANONICAL_MESH_CANDIDATE_BOUND"
    assert not result.canonical_mesh_authority
    assert not result.source_fidelity_authority
    assert not result.animation_ready_authority
    assert result.missing_requirements[0] == "STATIC_SOURCE_FIDELITY"
    candidate = result.candidate
    assert candidate is not None
    validate_canonical_mesh_candidate(
        candidate, surface=kw["source_surface"],
        partition=kw["source_partition"],
        carrier_policy=kw["source_carrier_policy"],
    )
    mesh = kw["evidence"].mesh
    assert len(candidate.vertices) == len(mesh["rest_xyz"])
    assert len(candidate.faces) == len(mesh["faces"])
    for i, vertex in enumerate(candidate.vertices):
        assert vertex.candidate_vertex_id == f"RSSV:{i:08d}"
        assert vertex.P == tuple(mesh["rest_xyz"][i])
        assert vertex.metadata["original_inference_evidence_hash"] == kw["evidence"].evidence_hash
        assert vertex.metadata["base_geometry_preserved_in_source_CAS"] is True
    assert candidate.faces == tuple(
        tuple(f"RSSV:{int(j):08d}" for j in triangle)
        for triangle in mesh["faces"]
    )
    assert result.candidate_mesh_sha256 == content_sha256(mesh)
    assert kw["evidence"].to_dict() == before
    assert candidate.metadata["static_source_fidelity_qualified"] is False
    assert candidate.metadata["canonical_mesh_authority"] is False


def test_no_source_witness_is_abstain_not_fabricated_identity_support():
    kw = _parents()
    for field in (
        "source_surface", "source_partition", "source_carrier_policy",
        "source_supports_by_vertex", "source_component_ids_by_candidate_index",
        "source_correspondence_receipt_sha256",
        "trusted_source_correspondence_registry",
    ):
        kw[field] = None
    result = _compile(kw)
    assert result.state == "ABSTAIN_SOURCE_CORRESPONDENCE_REQUIRED"
    assert result.candidate is None
    assert len(result.missing_requirements) == 7
    assert not result.canonical_mesh_authority


def test_rehashed_semantic_or_graphics_drift_cannot_mint_candidate_bridge():
    kw = _parents()
    distorted = deepcopy(dict(kw["evidence"].mesh))
    distorted["graphics_logits"][0][0] += .5
    kw["evidence"] = seal_inference_evidence_v1(replace(
        kw["evidence"], mesh=distorted, evidence_hash="",
    ))
    with pytest.raises(QualificationError, match="SOURCE_OR_AUTHORITY_DRIFT"):
        _compile(kw)


def test_source_partition_cannot_be_invented_from_disjoint_faces():
    kw = _parents()
    # The encoder emits disjoint indexed triangles in ONE TESSA component.
    # Spatially separated faces alone do not prove separate movable parts.
    mesh = kw["evidence"].mesh
    assert len(set(mesh["vertex_components"])) == 1
    from compiler.realsas_compiler_core.product_authority_v1 import (
        ComponentRegionIR, build_mechanical_partition, build_component_carrier_policy,
        ComponentCarrierDecisionIR,
    )
    src = kw["source_surface"]
    partition = build_mechanical_partition(
        surface=src,
        components=(
            ComponentRegionIR("split-A", ("src-0", "src-1", "src-2")),
            ComponentRegionIR("split-B", ("src-3", "src-4", "src-5")),
        ),
        boundary_constraints=(),
    )
    kw["source_partition"] = partition
    kw["source_carrier_policy"] = build_component_carrier_policy(
        partition=partition,
        decisions=(
            ComponentCarrierDecisionIR("split-A", "MESH", ("synthetic",)),
            ComponentCarrierDecisionIR("split-B", "MESH", ("synthetic",)),
        ),
    )
    with pytest.raises(QualificationError, match="PARTITION_COMPONENT_COVERAGE_DRIFT"):
        _compile(kw)


@pytest.mark.parametrize("mode", [
    "missing_vertex", "duplicate_source", "fractional_identity", "bad_receipt",
    "bad_component", "bad_source_lineage",
])
def test_partial_or_forged_source_correspondence_fails_closed(mode):
    kw = _parents()
    if mode == "missing_vertex":
        kw["source_supports_by_vertex"].pop(0)
    elif mode == "duplicate_source":
        kw["source_supports_by_vertex"][0] = replace(
            kw["source_supports_by_vertex"][0],
            coefficients=(("src-0", .5), ("src-0", .5)),
        )
    elif mode == "fractional_identity":
        kw["source_supports_by_vertex"][0] = replace(
            kw["source_supports_by_vertex"][0],
            coefficients=(("src-0", .5), ("src-1", .5)),
        )
    elif mode == "bad_receipt":
        kw["source_correspondence_receipt_sha256"] = "not-a-sha"
    elif mode == "bad_component":
        kw["source_component_ids_by_candidate_index"] = {0:"unmatched-component"}
    else:
        kw["source_partition"] = replace(
            kw["source_partition"], surface_lineage_hash="not-the-source",
        )
    with pytest.raises(QualificationError):
        _compile(kw)


def test_refinement_is_geometry_proposal_not_original_rgba_or_source_visibility():
    kw = _parents()
    bridge = _compile(kw)
    assert bridge.candidate is not None
    assert bridge.candidate.producer_id == "RealSaS.RSSvNext.CanonicalMeshCandidateBridge.v1"
    for vertex in bridge.candidate.vertices:
        assert vertex.refinement is None  # preserved as raw original CAS evidence
        assert vertex.metadata["canonical_authority"] is False
    assert not bridge.drawable_authority
    assert not bridge.mechanical_joint_authority


@pytest.mark.parametrize("tamper,expected", [
    ("unregistered_digest", "UNREGISTERED_OR_SCHEMA_DRIFT"),
    ("replaced_parent_hash", "CONTENT_DRIFT"),
    ("wrong_source_receipt", "CONTENT_DRIFT"),
    ("wrong_producer", "CONTENT_DRIFT"),
])
def test_source_witness_digest_is_not_sufficient_for_source_admission(tamper, expected):
    kw = _parents()
    key = kw["source_correspondence_receipt_sha256"]
    if tamper == "unregistered_digest":
        kw["source_correspondence_receipt_sha256"] = "f" * 64
    else:
        value = dict(kw["trusted_source_correspondence_registry"][key])
        if tamper == "replaced_parent_hash":
            value["source_surface_lineage_hash"] = "a" * 64
        elif tamper == "wrong_source_receipt":
            value["source_observation_receipt_sha256"] = "e" * 64
        else:
            value["producer_id"] = "RealSaS.RSSvNext.CanonicalMeshCandidateBridge"
        kw["trusted_source_correspondence_registry"] = {key: value}
    with pytest.raises(QualificationError, match=expected):
        _compile(kw)


def test_validly_rehashed_but_wrong_component_receipt_cannot_cross_parent_boundary():
    kw = _parents()
    key = kw["source_correspondence_receipt_sha256"]
    value = dict(kw["trusted_source_correspondence_registry"][key])
    value["component_correspondence_hash"] = "1" * 64
    new_key = content_sha256(value)
    kw["trusted_source_correspondence_registry"] = {new_key: value}
    kw["source_correspondence_receipt_sha256"] = new_key
    with pytest.raises(QualificationError, match="REGISTERED_PARENT_DRIFT"):
        _compile(kw)


def test_same_receipt_cannot_be_replayed_against_another_source_correspondence():
    kw = _parents()
    altered = dict(kw["source_supports_by_vertex"])
    altered[0] = replace(
        altered[0], metadata={"same_node_different_source_witness": True},
    )
    kw["source_supports_by_vertex"] = altered
    with pytest.raises(QualificationError, match="REGISTERED_PARENT_DRIFT"):
        _compile(kw)


def test_missing_source_admission_registry_abstains_without_qualifying_candidate():
    kw = _parents()
    kw["trusted_source_correspondence_registry"] = None
    result = _compile(kw)
    assert result.state == "ABSTAIN_SOURCE_CORRESPONDENCE_REQUIRED"
    assert "INDEPENDENT_SOURCE_CORRESPONDENCE_REGISTRY" in result.missing_requirements
    assert result.candidate is None
    assert result.canonical_mesh_authority is False
