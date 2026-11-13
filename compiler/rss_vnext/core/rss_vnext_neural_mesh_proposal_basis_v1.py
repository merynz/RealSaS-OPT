from __future__ import annotations

"""Two-image RSS/TESSA proposal basis for the EXISTING CanonicalMesh pipeline.

RealSaS's feed-forward multiview geometry/color/semantics and multilayer
extraction remain upstream. The canonical candidate is generated from the
actual topology-decoder output, never from invented source 3D supervision.
The substrate is explicitly a NEURAL CANDIDATE, not an independently observed
source surface. Qualify against observed source cameras/raster using existing
static_mesh_qualification_v2 before mint_canonical_mesh_v1. Different feature
heads remain untouched in original CAS. No semantic label means a joint/cut.
"""

from dataclasses import asdict, dataclass, replace
import math
from typing import Any

import numpy as np

from .hashing import content_sha256
from .product_authority_v1 import (
    CanonicalMeshCandidateIR, CanonicalMeshVertexCandidateIR,
    ComponentCarrierDecisionIR, ComponentCarrierPolicyIR, ComponentRegionIR,
    GeometricRefinementIR, MechanicalPartitionIR,
    build_component_carrier_policy, build_mechanical_partition,
    canonical_mesh_candidate_lineage_hash, validate_canonical_mesh_candidate,
)
from .rss_vnext_cross_expert_capability_court_v1 import (
    compile_rss_vnext_cross_expert_capability_court_v1,
)
from .rss_vnext_inference_evidence_v1 import RSSvNextInferenceEvidenceIRV1
from .rss_vnext_surface_decisions_v1 import RSSvNextComputedCandidateSurfaceV1
from .types import QualificationError, RiggingSurfaceIR, SurfaceNode, SurfaceSupportBinding

SCHEMA = "RealSaS.RSSvNext.TwoViewNeuralMeshCandidateBasis.v1"
_MAX_VERTICES = 250_000
_MAX_FACES = 500_000
_P = "RSS_VNEXT_NEURAL_BASIS_"


@dataclass(frozen=True)
class RSSvNextNeuralMeshCandidateBasisV1:
    inference_evidence_hash: str
    computed_surface_sha256: str
    original_mesh_sha256: str
    neural_surface: RiggingSurfaceIR
    neural_partition: MechanicalPartitionIR
    neural_carrier_policy: ComponentCarrierPolicyIR
    candidate: CanonicalMeshCandidateIR
    basis_hash: str
    schema: str = SCHEMA
    state: str = "TESSA_INDEXED_CANONICAL_CANDIDATE_UNQUALIFIED"
    source_fidelity_authority: bool = False
    canonical_mesh_authority: bool = False
    semantic_part_identity_authority: bool = False
    attachment_authority: bool = False
    drawable_authority: bool = False
    animation_ready_authority: bool = False


def _fail(reason: str) -> None:
    raise QualificationError(_P + reason)


def build_rss_vnext_neural_mesh_candidate_basis_v1(
    *, evidence: RSSvNextInferenceEvidenceIRV1,
    computed_surface: RSSvNextComputedCandidateSurfaceV1,
    model_config: Any,
) -> RSSvNextNeuralMeshCandidateBasisV1:
    """Admit TESSA geometry as a *candidate*, not as source 3D truth.

    This adapter resolves the historical 8-view source-surface dependency
    for the 2-view *candidate* stage. Existing source-fidelity/geometry
    qualification still must compare the resulting candidate against the
    original observed rasters, with actual calibrated camera contract.
    """
    from .rss_vnext_surface_decisions_v1 import _replay_topology
    if (
        type(evidence) is not RSSvNextInferenceEvidenceIRV1
        or type(computed_surface) is not RSSvNextComputedCandidateSurfaceV1
        or evidence.mesh_status != "UNQUALIFIED_MESH_CANDIDATE"
        or evidence.mesh is None
    ):
        _fail("LEARNED_INDEXED_MESH_REQUIRED")
    # Actual TESSA grammar and index replay, not merely two internally
    # consistent/rehashed JSON objects.
    _replay_topology(evidence, model_config)
    court = compile_rss_vnext_cross_expert_capability_court_v1(
        evidence=evidence, surface=computed_surface, model_config=model_config,
    )
    if (
        court.state != "TYPED_CROSS_EXPERT_PROPOSALS_PRESERVED"
        or court.candidate_mesh_sha256 != computed_surface.candidate_mesh_sha256
        or court.face_count != len(evidence.mesh["faces"])
        or court.canonical_mesh_authority
    ):
        _fail("CROSS_EXPERT_PARENT_OR_AUTHORITY_DRIFT")
    mesh = evidence.mesh
    base = np.asarray(mesh["base_xyz"], dtype=np.float64)
    rest = np.asarray(mesh["rest_xyz"], dtype=np.float64)
    indices = np.asarray(mesh["faces"], dtype=np.int64)
    vc = np.asarray(mesh["vertex_components"], dtype=np.int64)
    fc = np.asarray(mesh["face_components"], dtype=np.int64)
    n = len(base)
    f = len(indices)
    if (
        not 3 <= n <= _MAX_VERTICES or not 1 <= f <= _MAX_FACES
        or base.shape != (n, 3) or rest.shape != base.shape
        or indices.shape != (f, 3) or vc.shape != (n,) or fc.shape != (f,)
        or not np.isfinite(base).all() or not np.isfinite(rest).all()
        or np.any(indices < 0) or np.any(indices >= n)
    ):
        _fail("MESH_DOMAIN_OR_RESOURCE_LIMIT")
    if any(
        (len(set(int(v) for v in face)) != 3
         or len(set(int(vc[v]) for v in face)) != 1
         or int(vc[face[0]]) != int(fc[i]))
        for i, face in enumerate(indices)
    ):
        _fail("FACE_COMPONENT_OR_VERTEX_DOMAIN_INVALID")
    # Use actual *base indexed triangle* geometry to measure TESSA's
    # refinement. FlexiCubes may later alter this candidate only by creating
    # a NEW lineage; this function does not pretend it ran FlexiCubes.
    normals = np.zeros_like(base)
    scales = np.zeros(n, dtype=np.float64)
    for face in indices:
        a,b,c = (int(x) for x in face)
        cross = np.cross(base[b]-base[a], base[c]-base[a])
        norm = float(np.linalg.norm(cross))
        if not math.isfinite(norm) or norm <= 1.e-12:
            _fail("DEGENERATE_BASE_TRIANGLE")
        for idx in (a,b,c):
            normals[idx] += cross
        for x,y in ((a,b),(b,c),(c,a)):
            edge = float(np.linalg.norm(base[x]-base[y]))
            scales[x] = max(scales[x], edge)
            scales[y] = max(scales[y], edge)
    norm_length = np.linalg.norm(normals,axis=1)
    if np.any(norm_length <= 1.e-12) or np.any(scales <= 1.e-12):
        _fail("ISOLATED_OR_AMBIGUOUS_NORMAL_VERTEX")
    normals /= norm_length[:,None]
    mesh_hash = content_sha256(mesh)
    original_faces = tuple(tuple(int(j) for j in face) for face in indices)
    names = tuple(f"RSSV:{i:08d}" for i in range(n))
    basis_ids = tuple(f"RSSS:{mesh_hash[:16]}:{i:08d}" for i in range(n))
    source_nodes = tuple(
        SurfaceNode(
            surface_id=basis_ids[i],
            P=tuple(float(x) for x in base[i]),
            support_views=(),  # source ray projection is NOT first-hit truth
            provenance_refs=(evidence.evidence_hash,),
            source_observation_ids=(),  # never forge a raster observation id
            metadata={
                "authority": "TESSA_BASE_CANDIDATE_NOT_OBSERVED_3D",
                "candidate_vertex_index": i,
                "projection_valid_input_views_unqualified": tuple(
                    int(view) for view, flag in enumerate(
                        mesh["source_rays"]["valid"][i]
                    ) if flag
                ),
                "base_coordinate_in_original_Hx_CAS": True,
            },
        )
        for i in range(n)
    )
    basis_lineage = content_sha256({
        "schema": "RealSaS.RSSvNext.NeuralProposalSurfaceLineage.v1",
        "inference_evidence_hash": evidence.evidence_hash,
        "mesh_hash": mesh_hash,
        "base_xyz": mesh["base_xyz"],
        "faces": mesh["faces"],
        "vertex_components": mesh["vertex_components"],
    })
    source = RiggingSurfaceIR(
        surface_nodes=source_nodes,
        geometry_lineage_hash=basis_lineage,
        builder_id="RealSaS.RSSvNext.NEURAL_PROPOSAL_SURFACE_NOT_SOURCE_TRUTH",
        metadata={
            "source_surface_qualified": False,
            "source_observation_claimed": False,
            "not_an_independent_3d_observation": True,
            "external_source_fidelity_still_required": True,
            "original_observed_front_sha256": evidence.observed_front_sha256,
            "original_observed_back_sha256": evidence.observed_back_sha256,
            "original_inference_evidence_hash": evidence.evidence_hash,
        },
    )
    comp_values = tuple(sorted(set(int(c) for c in vc)))
    comp_ids = {
        c: f"RSSC:{mesh_hash[:16]}:{c}"
        for c in comp_values
    }
    partition = build_mechanical_partition(
        surface=source,
        components=tuple(ComponentRegionIR(
            component_id=comp_ids[c],
            surface_ids=tuple(basis_ids[i] for i in range(n) if int(vc[i])==c),
            evidence_refs=(evidence.evidence_hash,),
            metadata={
                "semantic_role": "UNQUALIFIED_NOT_COMPUTED",
                "mechanical_separability": "UNKNOWN",
                "source": "TESSA_INDEXED_COMPONENT_PROPOSAL",
            },
        ) for c in comp_values),
        boundary_constraints=(),
        metadata={
            "authority": "TESSA_PROPOSAL_COMPONENT_PARTITION",
            "mechanical_independence_qualified": False,
            "learned_region_affinities_are_not_partition_truth": True,
        },
    )
    policy = build_component_carrier_policy(
        partition=partition,
        decisions=tuple(ComponentCarrierDecisionIR(
            component_id=comp_ids[c], carrier_class="MESH",
            evidence_refs=(evidence.evidence_hash,),
            metadata={"carrier_geometry_proposal":True,"mechanical_carrier_qualified":False},
        ) for c in comp_values),
        metadata={"authority": "NEURAL_MESH_PROPOSAL_ONLY"},
    )
    refinement_lineage = content_sha256({
        "schema": "RealSaS.RSSvNext.TESSAIndexedRefinement.v1",
        "mesh_hash": mesh_hash,
        "base_xyz": mesh["base_xyz"],
        "rest_xyz": mesh["rest_xyz"],
    })
    vertices = []
    for i in range(n):
        delta = rest[i] - base[i]
        normal_component = float(np.dot(delta, normals[i]))
        tangent = delta - normal_component * normals[i]
        tangential_component = float(np.linalg.norm(tangent))
        ref = GeometricRefinementIR(
            dense_lineage_hash=refinement_lineage,
            base_position=tuple(float(x) for x in base[i]),
            refined_position=tuple(float(x) for x in rest[i]),
            local_scale=float(scales[i]),
            normal_component=normal_component,
            tangential_component=tangential_component,
            method="RSS_TESSA_REFINED_PROPOSAL_NO_SOURCE_AUTHORITY",
            metadata={"refinement_quality_not_qualified": True},
        )
        vertices.append(CanonicalMeshVertexCandidateIR(
            candidate_vertex_id=names[i],
            support_binding=SurfaceSupportBinding(
                mode="IDENTITY_SURFACE_NODE",
                coefficients=((basis_ids[i],1.0),),
                metadata={"basis_kind": "NEURAL_PROPOSAL_NOT_SOURCE_GEOMETRY"},
            ),
            component_id=comp_ids[int(vc[i])],
            P=tuple(float(x) for x in rest[i]),
            refinement=ref,
            metadata={
                "rss_vertex_index":i,
                "original_inference_evidence_hash":evidence.evidence_hash,
                "model_generated_geometry_not_source_truth": True,
                "semantics_graphics_skin_preserved_in_Hx_CAS": True,
            },
        ))
    faces = tuple(tuple(names[j] for j in tri) for tri in original_faces)
    edges = tuple(sorted({
        tuple(sorted((names[a], names[b])))
        for tri in original_faces
        for a,b in ((tri[0],tri[1]),(tri[1],tri[2]),(tri[2],tri[0]))
    }))
    candidate = CanonicalMeshCandidateIR(
        vertices=tuple(vertices), faces=faces, edges=edges,
        surface_binding_hash=source.geometry_lineage_hash,
        partition_binding_hash=partition.partition_lineage_hash,
        carrier_policy_binding_hash=policy.carrier_policy_lineage_hash,
        producer_id="RealSaS.RSSvNext.TwoViewNeuralMeshProposalBasis.v1",
        producer_policy_hash=content_sha256({
            "schema":SCHEMA,
            "basis_owner":"TESSA_DECODED_BASE_AND_REFINED_MESH",
            "source_fidelity_owner":"EXISTING_STATIC_MESH_QUALIFICATION_V2",
        }),
        candidate_lineage_hash="",
        metadata={
            "neural_proposal_basis":True,
            "source_geometry_was_not_independently_observed":True,
            "source_fidelity_qualified":False,
            "source_observation_raster_qualification_required":True,
            "model_identity":evidence.checkpoint_sha256,
            "inference_evidence_hash":evidence.evidence_hash,
            "original_mesh_sha256":mesh_hash,
            "computed_surface_sha256":content_sha256(asdict(computed_surface)),
            "semantic_head_is_not_component_partition_authority":True,
            "flexicubes_refinement_performed":False,
            "geometry_change_after_admission_requires_new_lineage":True,
        },
    )
    candidate = replace(candidate,
        candidate_lineage_hash=canonical_mesh_candidate_lineage_hash(candidate))
    validate_canonical_mesh_candidate(
        candidate, surface=source, partition=partition, carrier_policy=policy,
    )
    payload = {
        "schema":SCHEMA,
        "evidence":evidence.evidence_hash,
        "computed_surface":content_sha256(asdict(computed_surface)),
        "neural_surface_lineage":source.geometry_lineage_hash,
        "candidate":candidate.candidate_lineage_hash,
        "partition":partition.partition_lineage_hash,
        "carrier_policy":policy.carrier_policy_lineage_hash,
    }
    return RSSvNextNeuralMeshCandidateBasisV1(
        inference_evidence_hash=evidence.evidence_hash,
        computed_surface_sha256=content_sha256(asdict(computed_surface)),
        original_mesh_sha256=mesh_hash,
        neural_surface=source,
        neural_partition=partition,
        neural_carrier_policy=policy,
        candidate=candidate,
        basis_hash=content_sha256(payload),
    )


__all__ = [
    "SCHEMA", "RSSvNextNeuralMeshCandidateBasisV1",
    "build_rss_vnext_neural_mesh_candidate_basis_v1",
]
