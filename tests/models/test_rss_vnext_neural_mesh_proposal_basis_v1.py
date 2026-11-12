from __future__ import annotations

"""Actual RSS model + synthetic TESSA tokens through product canonical ABI.

A source image does NOT reveal all 3D coordinates; TESSA generated vertex
basis is a neural proposal only. These tests prove we can transport that
basis to existing CanonicalMeshCandidateIR without inventing observed 3D
source labels, semantic part boundaries, or a qualified puppet.
"""

from copy import deepcopy
from dataclasses import asdict, replace
import math

import numpy as np
import pytest

from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.product_authority_v1 import (
    validate_canonical_mesh_candidate,
)
from compiler.realsas_compiler_core.rss_vnext_canonical_mesh_candidate_bridge_v1 import (
    build_rss_vnext_canonical_mesh_candidate_bridge_v1,
)
from compiler.realsas_compiler_core.rss_vnext_neural_mesh_proposal_basis_v1 import (
    build_rss_vnext_neural_mesh_candidate_basis_v1,
)
from compiler.realsas_compiler_core.rss_vnext_inference_evidence_v1 import (
    seal_inference_evidence_v1,
)
from compiler.realsas_compiler_core.types import QualificationError
from tests.models.test_rss_vnext_cross_expert_capability_court_v1 import _ready


def _build():
    cfg, evidence, surface, _relations = _ready()
    result = build_rss_vnext_neural_mesh_candidate_basis_v1(
        evidence=evidence, computed_surface=surface, model_config=cfg,
    )
    return cfg, evidence, surface, result


def test_tessa_geometry_enters_existing_canonical_candidate_domain_without_source_laundering():
    cfg, evidence, surface, result = _build()
    mesh = evidence.mesh
    validate_canonical_mesh_candidate(
        result.candidate, surface=result.neural_surface,
        partition=result.neural_partition, carrier_policy=result.neural_carrier_policy,
    )
    assert result.state == "TESSA_INDEXED_CANONICAL_CANDIDATE_UNQUALIFIED"
    assert result.original_mesh_sha256 == content_sha256(mesh)
    assert result.neural_surface.builder_id.endswith("NEURAL_PROPOSAL_SURFACE_NOT_SOURCE_TRUTH")
    assert result.neural_surface.metadata["source_surface_qualified"] is False
    assert result.neural_surface.metadata["original_observed_front_sha256"] == evidence.observed_front_sha256
    assert result.neural_surface.metadata["original_observed_back_sha256"] == evidence.observed_back_sha256
    assert all(not x.support_views and not x.source_observation_ids for x in result.neural_surface.surface_nodes)
    assert len(result.candidate.vertices) == len(mesh["rest_xyz"])
    assert len(result.candidate.faces) == len(mesh["faces"])
    assert len(result.candidate.edges) == 6
    assert result.candidate.metadata["flexicubes_refinement_performed"] is False
    assert not result.source_fidelity_authority
    assert not result.canonical_mesh_authority
    assert not result.semantic_part_identity_authority
    assert not result.animation_ready_authority


def test_every_tessa_refinement_is_explicit_and_surface_support_is_a_proposal_only():
    _cfg, evidence, _surface, result = _build()
    mesh = evidence.mesh
    for i, vertex in enumerate(result.candidate.vertices):
        assert vertex.candidate_vertex_id == f"RSSV:{i:08d}"
        assert vertex.P == tuple(mesh["rest_xyz"][i])
        assert vertex.support_binding.mode == "IDENTITY_SURFACE_NODE"
        assert vertex.support_binding.metadata["basis_kind"] == "NEURAL_PROPOSAL_NOT_SOURCE_GEOMETRY"
        ref = vertex.refinement
        assert ref is not None
        base = np.asarray(mesh["base_xyz"][i], dtype=float)
        rest = np.asarray(mesh["rest_xyz"][i], dtype=float)
        assert ref.base_position == tuple(base)
        assert ref.refined_position == tuple(rest)
        assert ref.local_scale > 0
        assert math.isclose(
            np.linalg.norm(rest-base),
            math.hypot(ref.normal_component, ref.tangential_component),
            rel_tol=1.e-8, abs_tol=1.e-8,
        )
        assert ref.metadata["refinement_quality_not_qualified"] is True
        assert vertex.metadata["semantics_graphics_skin_preserved_in_Hx_CAS"] is True


def test_topological_components_are_not_inferred_from_semantic_or_spatial_distance():
    _cfg, evidence, _surface, result = _build()
    # The synthetic fixture has two spatially disconnected faces, but TESSA
    # encodes them in one *candidate* component. Do not fabricate two parts.
    assert len(set(evidence.mesh["vertex_components"])) == 1
    assert len(result.neural_partition.components) == 1
    assert result.neural_partition.metadata["mechanical_independence_qualified"] is False
    assert result.candidate.metadata["semantic_head_is_not_component_partition_authority"] is True


def test_two_view_product_neural_basis_gate_never_mints_source_or_mesh():
    cfg,evidence,surface,_ = _build()
    standard = build_rss_vnext_canonical_mesh_candidate_bridge_v1(
        evidence=evidence, computed_surface=surface, model_config=cfg,
    )
    assert standard.state == "ABSTAIN_SOURCE_CORRESPONDENCE_REQUIRED"
    assert standard.candidate is None
    enabled = build_rss_vnext_canonical_mesh_candidate_bridge_v1(
        evidence=evidence, computed_surface=surface, model_config=cfg,
        allow_unqualified_neural_basis=True,
    )
    assert enabled.state == "UNQUALIFIED_NEURAL_BASIS_CANONICAL_CANDIDATE"
    assert enabled.candidate is not None
    assert enabled.source_support_binding_sha256 is None
    assert enabled.neural_basis_lineage_sha256 is not None
    assert enabled.neural_basis_surface is not None
    assert enabled.neural_basis_partition is not None
    assert enabled.neural_basis_carrier_policy is not None
    assert "SOURCE_RASTER_FIDELITY_QUALIFICATION" in enabled.missing_requirements
    assert not any([
        enabled.source_fidelity_authority, enabled.canonical_mesh_authority,
        enabled.drawable_authority, enabled.mechanical_joint_authority,
        enabled.animation_ready_authority,
    ])


def test_rehashed_model_mesh_cannot_invent_tessa_face_order():
    cfg,evidence,surface,_ = _build()
    mesh = deepcopy(evidence.mesh)
    mesh["faces"][0][0],mesh["faces"][0][1] = mesh["faces"][0][1],mesh["faces"][0][0]
    forged = seal_inference_evidence_v1(replace(
        evidence, mesh=mesh, evidence_hash="",
    ))
    with pytest.raises(QualificationError, match="TESSA_TOKEN_MESH_DECODE_MISMATCH"):
        build_rss_vnext_neural_mesh_candidate_basis_v1(
            evidence=forged, computed_surface=surface, model_config=cfg,
        )


def test_candidate_basis_cannot_accept_an_unrelated_surface_parent():
    cfg,evidence,surface,_ = _build()
    forged_surface = replace(
        surface, source_inference_evidence_hash="e" * 64,
    )
    with pytest.raises(QualificationError, match="SOURCE_OR_AUTHORITY_DRIFT"):
        build_rss_vnext_neural_mesh_candidate_basis_v1(
            evidence=evidence, computed_surface=forged_surface,
            model_config=cfg,
        )
