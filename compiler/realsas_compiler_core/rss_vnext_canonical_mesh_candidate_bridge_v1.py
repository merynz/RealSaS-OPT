from __future__ import annotations

"""RSS.vNext -> existing CanonicalMeshCandidateIR; never new Mesh authority.

StdGEN's multiview color/geometry/semantic extraction is upstream. TESSA
owns indexed mesh proposals and FlexiCubes may refine geometry, but none of
those learned values is a source-support correspondence. This bridge only
runs if an external source-support witness explicitly binds EACH indexed
vertex to an independently constructed RiggingSurfaceIR. No correspondence
is generated from semantic probabilities or RGBA, and missing evidence
must retain ABSTAIN. The existing static_mesh_qualification_v2 and
canonical_mesh_ir_v1 remain the SOLE product Mesh admission authorities.
"""

from dataclasses import asdict, dataclass, replace
import math
from typing import Any, Mapping, Sequence

import numpy as np

from .hashing import content_sha256
from .product_authority_v1 import (
    CanonicalMeshCandidateIR,
    CanonicalMeshVertexCandidateIR,
    ComponentCarrierPolicyIR,
    MechanicalPartitionIR,
    canonical_mesh_candidate_lineage_hash,
    validate_canonical_mesh_candidate,
    validate_component_carrier_policy,
    validate_mechanical_partition,
)
from .rss_vnext_cross_expert_capability_court_v1 import (
    compile_rss_vnext_cross_expert_capability_court_v1,
)
from .rss_vnext_inference_evidence_v1 import (
    RSSvNextInferenceEvidenceIRV1,
)
from .rss_vnext_surface_decisions_v1 import (
    RSSvNextComputedCandidateSurfaceV1,
)
from .types import QualificationError, RiggingSurfaceIR, SurfaceSupportBinding

SCHEMA = "RealSaS.RSSvNext.CanonicalMeshCandidateBridge.v1"
_MAX_VERTICES = 250_000
_MAX_FACES = 500_000
_SOURCE_BASE_SUPPORT_MAX_DISTANCE = 2.0e-5


@dataclass(frozen=True)
class RSSvNextCanonicalMeshCandidateBridgeV1:
    inference_evidence_hash: str
    computed_surface_sha256: str
    candidate_mesh_sha256: str | None
    state: str
    missing_requirements: tuple[str, ...]
    candidate: CanonicalMeshCandidateIR | None
    source_support_binding_sha256: str | None
    # Optional typed TESSA-only basis. It is NEVER independently observed
    # source 3D geometry. The actual SourceFidelity court must still run.
    neural_basis_surface: RiggingSurfaceIR | None = None
    neural_basis_partition: MechanicalPartitionIR | None = None
    neural_basis_carrier_policy: ComponentCarrierPolicyIR | None = None
    neural_basis_lineage_sha256: str | None = None
    schema: str = SCHEMA
    canonical_mesh_authority: bool = False
    source_fidelity_authority: bool = False
    drawable_authority: bool = False
    mechanical_joint_authority: bool = False
    animation_ready_authority: bool = False


def _fail(code: str) -> None:
    raise QualificationError("RSS_VNEXT_CANONICAL_BRIDGE_" + code)


def _sha(value: object) -> bool:
    return type(value) is str and len(value) == 64 and all(
        c in "0123456789abcdef" for c in value
    )


def build_rss_vnext_canonical_mesh_candidate_bridge_v1(
    *,
    evidence: RSSvNextInferenceEvidenceIRV1,
    computed_surface: RSSvNextComputedCandidateSurfaceV1,
    model_config: Any,
    source_surface: RiggingSurfaceIR | None = None,
    source_partition: MechanicalPartitionIR | None = None,
    source_carrier_policy: ComponentCarrierPolicyIR | None = None,
    source_supports_by_vertex: Mapping[int, SurfaceSupportBinding] | None = None,
    source_component_ids_by_candidate_index: Mapping[int, str] | None = None,
    source_correspondence_receipt_sha256: str | None = None,
    allow_unqualified_neural_basis: bool = False,
) -> RSSvNextCanonicalMeshCandidateBridgeV1:
    """Bind existing source-support addresses to TESSA *without* minting Mesh.

    The source substrate, its partition and its correspondence are independent
    upstream artifacts. This function rejects invented, mismatched, partial
    or rehashed vertex/face/part addresses; it never synthesizes substitutes.
    Successful candidate transport can only enter the EXISTING source-fidelity
    and mechanical-carrier courts, not bypass them.
    """
    if (
        type(evidence) is not RSSvNextInferenceEvidenceIRV1
        or type(computed_surface) is not RSSvNextComputedCandidateSurfaceV1
    ):
        _fail("TYPED_NEURAL_PARENT_REQUIRED")
    # Recheck every face's Semantic, Graphics, rays and Mechanics fields against
    # the exact inference parent: hashes alone do not verify semantic equality.
    court = compile_rss_vnext_cross_expert_capability_court_v1(
        evidence=evidence, surface=computed_surface, model_config=model_config,
    )
    digest = content_sha256(asdict(computed_surface))
    parent = dict(
        inference_evidence_hash=evidence.evidence_hash,
        computed_surface_sha256=digest,
        candidate_mesh_sha256=computed_surface.candidate_mesh_sha256,
    )
    if evidence.mesh_status == "ABSTAIN":
        return RSSvNextCanonicalMeshCandidateBridgeV1(
            **parent, state="ABSTAIN_TESSA_TOPOLOGY",
            missing_requirements=("VALID_INDEXED_TOPOLOGY",),
            candidate=None, source_support_binding_sha256=None,
        )
    if court.state != "TYPED_CROSS_EXPERT_PROPOSALS_PRESERVED":
        _fail("CROSS_EXPERT_PARENT_NOT_VALID")
    supplied = (
        source_surface, source_partition, source_carrier_policy,
        source_supports_by_vertex, source_component_ids_by_candidate_index,
        source_correspondence_receipt_sha256,
    )
    if allow_unqualified_neural_basis and all(x is None for x in supplied):
        from .rss_vnext_neural_mesh_proposal_basis_v1 import (
            build_rss_vnext_neural_mesh_candidate_basis_v1,
        )
        neural = build_rss_vnext_neural_mesh_candidate_basis_v1(
            evidence=evidence, computed_surface=computed_surface,
            model_config=model_config,
        )
        if (
            neural.candidate.metadata.get("source_geometry_was_not_independently_observed")
            is not True
            or neural.canonical_mesh_authority or neural.source_fidelity_authority
        ):
            _fail("NEURAL_BASIS_CANNOT_MINT_SOURCE_AUTHORITY")
        return RSSvNextCanonicalMeshCandidateBridgeV1(
            **parent,
            state="UNQUALIFIED_NEURAL_BASIS_CANONICAL_CANDIDATE",
            missing_requirements=(
                "SOURCE_RASTER_FIDELITY_QUALIFICATION",
                "STATIC_MESH_QUALITY_AND_CONDITIONING",
                "QUALIFIED_MECHANICAL_CARRIER",
                "CANONICAL_MESH_MINT",
                "POSE_DEFORMATION_QUALITY",
                "GRAPHICS_2D_SOURCE_APPEARANCE",
            ),
            candidate=neural.candidate,
            source_support_binding_sha256=None,
            neural_basis_surface=neural.neural_surface,
            neural_basis_partition=neural.neural_partition,
            neural_basis_carrier_policy=neural.neural_carrier_policy,
            neural_basis_lineage_sha256=neural.basis_hash,
        )
    if any(x is None for x in supplied):
        missing = tuple(key for key, val in zip((
            "SOURCE_SURFACE", "SOURCE_PARTITION", "SOURCE_CARRIER_POLICY",
            "PER_VERTEX_SOURCE_SUPPORT", "COMPONENT_SOURCE_CORRESPONDENCE",
            "INDEPENDENT_SOURCE_CORRESPONDENCE_RECEIPT",
        ), supplied) if val is None)
        return RSSvNextCanonicalMeshCandidateBridgeV1(
            **parent, state="ABSTAIN_SOURCE_CORRESPONDENCE_REQUIRED",
            missing_requirements=missing, candidate=None,
            source_support_binding_sha256=None,
        )
    assert source_surface is not None and source_partition is not None
    assert source_carrier_policy is not None
    assert source_supports_by_vertex is not None
    assert source_component_ids_by_candidate_index is not None
    if (
        type(source_surface) is not RiggingSurfaceIR
        or type(source_partition) is not MechanicalPartitionIR
        or type(source_carrier_policy) is not ComponentCarrierPolicyIR
        or not _sha(source_correspondence_receipt_sha256)
    ):
        _fail("SOURCE_WITNESS_TYPE_OR_RECEIPT_INVALID")
    validate_mechanical_partition(source_partition, source_surface)
    validate_component_carrier_policy(source_carrier_policy, source_partition)
    mesh = evidence.mesh
    assert mesh is not None
    rest = np.asarray(mesh["rest_xyz"], dtype=np.float64)
    base = np.asarray(mesh["base_xyz"], dtype=np.float64)
    faces = np.asarray(mesh["faces"], dtype=np.int64)
    n = len(rest)
    if (
        not 3 <= n <= _MAX_VERTICES
        or not 1 <= len(faces) <= _MAX_FACES
        or rest.shape != (n, 3) or base.shape != rest.shape
        or len(mesh["vertex_components"]) != n
        or len(mesh["face_components"]) != len(faces)
    ):
        _fail("MESH_RESOURCE_OR_ADDRESS_DRIFT")
    if (
        type(source_supports_by_vertex) is not dict
        or set(source_supports_by_vertex) != set(range(n))
        or type(source_component_ids_by_candidate_index) is not dict
    ):
        _fail("SOURCE_SUPPORT_INCOMPLETE_OR_DUPLICATE")
    components = set(int(x) for x in mesh["vertex_components"])
    if (
        set(source_component_ids_by_candidate_index) != components
        or any(type(k) is not int or type(v) is not str or not v
               for k, v in source_component_ids_by_candidate_index.items())
        or len(set(source_component_ids_by_candidate_index.values()))
           != len(source_component_ids_by_candidate_index)
    ):
        _fail("COMPONENT_SOURCE_CORRESPONDENCE_INCOMPLETE")
    sources = {node.surface_id: node for node in source_surface.surface_nodes}
    owner = {
        sid: component.component_id
        for component in source_partition.components for sid in component.surface_ids
    }
    partitions = {c.component_id for c in source_partition.components}
    if set(source_component_ids_by_candidate_index.values()) != partitions:
        _fail("PARTITION_COMPONENT_COVERAGE_DRIFT")
    vertices = []
    for i in range(n):
        support = source_supports_by_vertex[i]
        if type(support) is not SurfaceSupportBinding:
            _fail("SUPPORT_TYPE_INVALID")
        if (
            not support.coefficients
            or any(
                type(sid) is not str or sid not in sources
                or type(coeff) not in (int, float)
                or not math.isfinite(coeff) or coeff < 0
                for sid, coeff in support.coefficients
            )
            or len({sid for sid, _ in support.coefficients}) != len(support.coefficients)
            or not math.isclose(
                sum(float(v) for _, v in support.coefficients),
                1.0, abs_tol=1.e-9, rel_tol=0.0,
            )
        ):
            _fail("SOURCE_SUPPORT_SIMPLEX_OR_LINEAGE_INVALID")
        component = source_component_ids_by_candidate_index[int(mesh["vertex_components"][i])]
        if any(owner[sid] != component for sid, _ in support.coefficients):
            _fail("SOURCE_SUPPORT_COMPONENT_CROSSING")
        if support.mode == "IDENTITY_SURFACE_NODE":
            if len(support.coefficients) != 1 or support.coefficients[0][1] != 1.0:
                _fail("SOURCE_SUPPORT_IDENTITY_CANNOT_BE_FRACTIONAL")
        # A genuine source-support witness must actually address the proposed
        # BASE geometry; never silently claim it based on part/color similarity.
        source_pos = sum(
            (float(w) * np.asarray(sources[sid].P, dtype=np.float64)
             for sid, w in support.coefficients),
            start=np.zeros(3, dtype=np.float64),
        )
        if (
            not np.isfinite(source_pos).all()
            or float(np.linalg.norm(source_pos - base[i]))
                > _SOURCE_BASE_SUPPORT_MAX_DISTANCE
        ):
            _fail("SOURCE_BASE_CORRESPONDENCE_RESIDUAL")
        vertices.append(CanonicalMeshVertexCandidateIR(
            candidate_vertex_id=f"RSSV:{i:08d}",
            support_binding=support,
            component_id=component,
            P=tuple(float(v) for v in rest[i]),
            refinement=None,  # base/refined still present in original CAS
            metadata={
                "source": "RSS_VNEXT_TESSA_TOPOLOGY_LEARNED",
                "rss_vertex_index": i,
                "original_inference_evidence_hash": evidence.evidence_hash,
                "base_geometry_preserved_in_source_CAS": True,
                "refined_geometry_preserved_in_source_CAS": True,
                "computed_source_support_displacement": float(
                    np.linalg.norm(source_pos - base[i])
                ),
                "canonical_authority": False,
            },
        ))
    face_ids = []
    edges = set()
    for fi, indices in enumerate(faces):
        vertex_components = {int(mesh["vertex_components"][int(k)]) for k in indices}
        if (
            len(vertex_components) != 1
            or next(iter(vertex_components)) != int(mesh["face_components"][fi])
        ):
            _fail("FACE_CROSSES_SOURCE_PARTITION")
        tri = tuple(f"RSSV:{int(x):08d}" for x in indices)
        face_ids.append(tri)
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            edges.add(tuple(sorted((a, b))))
    correspondence_hash = content_sha256({
        "schema": "RealSaS.RSSvNext.ExternalSourceCorrespondenceBinding.v1",
        "original_inference_evidence_hash": evidence.evidence_hash,
        "source_correspondence_receipt_sha256": source_correspondence_receipt_sha256,
        "source_surface_lineage_hash": source_surface.geometry_lineage_hash,
        "partition_lineage_hash": source_partition.partition_lineage_hash,
        "carrier_policy_lineage_hash": source_carrier_policy.carrier_policy_lineage_hash,
        "component_map": sorted(source_component_ids_by_candidate_index.items()),
        "support_rows": [asdict(source_supports_by_vertex[i]) for i in range(n)],
    })
    candidate = CanonicalMeshCandidateIR(
        vertices=tuple(vertices),
        faces=tuple(face_ids),
        edges=tuple(sorted(edges)),
        surface_binding_hash=source_surface.geometry_lineage_hash,
        partition_binding_hash=source_partition.partition_lineage_hash,
        carrier_policy_binding_hash=source_carrier_policy.carrier_policy_lineage_hash,
        producer_id="RealSaS.RSSvNext.CanonicalMeshCandidateBridge.v1",
        producer_policy_hash=content_sha256({
            "schema": SCHEMA,
            "max_source_base_support_displacement": _SOURCE_BASE_SUPPORT_MAX_DISTANCE,
            "topology_authority": "TESSA_UNQUALIFIED_INDEXED_CANDIDATE",
            "no_source_qualification_minted": True,
        }),
        candidate_lineage_hash="",
        metadata={
            "original_inference_evidence_hash": evidence.evidence_hash,
            "unified_checkpoint_sha256": evidence.checkpoint_sha256,
            "candidate_mesh_sha256": computed_surface.candidate_mesh_sha256,
            "topology_token_sequence_sha256": computed_surface.topology_token_sequence_sha256,
            "computed_surface_sha256": digest,
            "external_source_correspondence_sha256": correspondence_hash,
            "external_source_receipt_sha256": source_correspondence_receipt_sha256,
            "original_expert_evidence_in_CAS": True,
            "semantic_roles_are_not_component_ownership": True,
            "graphics_logits_are_not_source_rgb": True,
            "static_source_fidelity_qualified": False,
            "canonical_mesh_authority": False,
        },
    )
    candidate = replace(candidate,
        candidate_lineage_hash=canonical_mesh_candidate_lineage_hash(candidate))
    validate_canonical_mesh_candidate(
        candidate, surface=source_surface, partition=source_partition,
        carrier_policy=source_carrier_policy,
    )
    return RSSvNextCanonicalMeshCandidateBridgeV1(
        **parent, state="UNQUALIFIED_CANONICAL_MESH_CANDIDATE_BOUND",
        missing_requirements=(
            "STATIC_SOURCE_FIDELITY", "STATIC_MESH_QUALITY",
            "SOURCE_OBSERVATION_RAY_COURT", "MECHANICAL_CARRIER_QUALIFICATION",
            "CANONICAL_MESH_MINT",
        ),
        candidate=candidate,
        source_support_binding_sha256=correspondence_hash,
    )


__all__ = [
    "SCHEMA", "RSSvNextCanonicalMeshCandidateBridgeV1",
    "build_rss_vnext_canonical_mesh_candidate_bridge_v1",
]
