from __future__ import annotations

"""View-independent conservative canonical mesh-candidate producer.

This producer is deliberately simple: it triangulates only 3-cliques already present
in the immutable RiggingSurfaceIR local-relation graph. Every candidate vertex is an
IDENTITY_SURFACE_NODE binding. It is a safe baseline and a seam test for stage 26, not
a claim that relation-clique topology is the final product backend.

CDT/local-chart backends must emit the same CanonicalMeshCandidateIR and therefore
remain subordinate to current qualification.
"""

from itertools import combinations
import math

from .hashing import content_sha256
from .product_authority_v1 import (
    CanonicalMeshCandidateIR,
    CanonicalMeshVertexCandidateIR,
    ComponentCarrierPolicyIR,
    MechanicalPartitionIR,
    canonical_mesh_candidate_lineage_hash,
    validate_component_carrier_policy,
    validate_mechanical_partition,
)
from .types import QualificationError, RiggingSurfaceIR, SurfaceSupportBinding

_FORBIDDEN_RELATION_TOKENS = ("OCCLUDED", "UNOBSERVED", "UNSUPPORTED")


def _pair(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a < b else (b, a)


def _triangle_double_area(pa, pb, pc) -> float:
    ux, uy, uz = (float(pb[i]) - float(pa[i]) for i in range(3))
    vx, vy, vz = (float(pc[i]) - float(pa[i]) for i in range(3))
    cx = uy * vz - uz * vy
    cy = uz * vx - ux * vz
    cz = ux * vy - uy * vx
    return math.sqrt(cx * cx + cy * cy + cz * cz)


def _relation_usable(relation) -> bool:
    kind = str(relation.relation_kind).upper()
    if any(token in kind for token in _FORBIDDEN_RELATION_TOKENS):
        return False
    md = dict(getattr(relation, "metadata", {}) or {})
    if bool(md.get("unsupported", False)):
        return False
    try:
        score = float(relation.score)
    except Exception as exc:
        raise QualificationError("CANONICAL_MESH_RELATION_SCORE_INVALID") from exc
    return math.isfinite(score) and score > 0.0


def build_canonical_relation_candidate(
    surface: RiggingSurfaceIR,
    partition: MechanicalPartitionIR,
    carrier_policy: ComponentCarrierPolicyIR,
    *,
    producer_policy_hash: str,
    min_relative_double_area: float = 1e-8,
) -> CanonicalMeshCandidateIR:
    """Build a canonical candidate without any view/camera authority.

    UNKNOWN boundary adjacency may remain connected provisionally, matching the
    partition contract, but is explicitly recorded for later consequential-UNKNOWN
    qualification. SEPARATE boundaries cannot produce a cross-component face.
    """
    validate_mechanical_partition(partition, surface)
    validate_component_carrier_policy(carrier_policy, partition)
    if not producer_policy_hash:
        raise QualificationError("CANONICAL_MESH_PRODUCER_POLICY_MISSING")
    if not math.isfinite(float(min_relative_double_area)) or min_relative_double_area <= 0.0:
        raise QualificationError("CANONICAL_MESH_MIN_RELATIVE_AREA_INVALID")

    nodes = {node.surface_id: node for node in surface.surface_nodes}
    owner = {
        sid: component.component_id
        for component in partition.components
        for sid in component.surface_ids
    }
    boundary_by_pair = {
        _pair(row.a_surface_id, row.b_surface_id): row
        for row in partition.boundary_constraints
    }

    component_members = {
        component.component_id: set(component.surface_ids)
        for component in partition.components
    }
    safe_edges: set[tuple[str, str]] = set()
    unknown_edges: set[tuple[str, str]] = set()
    rejected_relation_count = 0

    for relation in surface.local_relations:
        a, b = str(relation.a_surface_id), str(relation.b_surface_id)
        if a == b or a not in nodes or b not in nodes:
            raise QualificationError("CANONICAL_MESH_RELATION_ENDPOINT_INVALID")
        if owner[a] != owner[b]:
            # Mechanical partition is the cut authority. No cross-component relation
            # can leak into product topology, regardless of relation score.
            continue
        if not _relation_usable(relation):
            rejected_relation_count += 1
            continue
        pair = _pair(a, b)
        boundary = boundary_by_pair.get(pair)
        if boundary is not None and boundary.decision == "SEPARATE":
            raise QualificationError("CANONICAL_MESH_SEPARATE_EDGE_INSIDE_COMPONENT")
        safe_edges.add(pair)
        if boundary is not None and boundary.decision == "UNKNOWN":
            unknown_edges.add(pair)

    faces_surface: list[tuple[str, str, str]] = []
    for component_id in sorted(component_members):
        members = sorted(component_members[component_id])
        neighbors = {sid: set() for sid in members}
        for a, b in safe_edges:
            if a in neighbors and b in neighbors:
                neighbors[a].add(b)
                neighbors[b].add(a)
        for a in members:
            for b, c in combinations(sorted(neighbors[a]), 2):
                if _pair(b, c) not in safe_edges:
                    continue
                tri = tuple(sorted((a, b, c)))
                if tri[0] != a:
                    # Each sorted clique is emitted once.
                    continue
                points = [nodes[sid].P for sid in tri]
                lengths = [
                    math.dist(tuple(map(float, points[i])), tuple(map(float, points[j])))
                    for i, j in ((0, 1), (1, 2), (2, 0))
                ]
                local_scale = max(lengths)
                if not math.isfinite(local_scale) or local_scale <= 0.0:
                    continue
                relative_area = _triangle_double_area(*points) / (local_scale * local_scale)
                if relative_area >= float(min_relative_double_area):
                    faces_surface.append(tri)

    faces_surface = sorted(set(faces_surface))
    if not faces_surface:
        raise QualificationError("CANONICAL_MESH_RELATION_BASELINE_NO_FACE")

    used_surface_ids = sorted({sid for face in faces_surface for sid in face})
    candidate_id = {
        sid: "CMV:" + content_sha256({
            "surface_lineage_hash": surface.geometry_lineage_hash,
            "partition_lineage_hash": partition.partition_lineage_hash,
            "surface_id": sid,
        })[:24]
        for sid in used_surface_ids
    }
    vertices = tuple(
        CanonicalMeshVertexCandidateIR(
            candidate_vertex_id=candidate_id[sid],
            support_binding=SurfaceSupportBinding(
                "IDENTITY_SURFACE_NODE",
                ((sid, 1.0),),
                metadata={"canonical_relation_baseline": True},
            ),
            component_id=owner[sid],
            P=tuple(map(float, nodes[sid].P)),
            metadata={"source_surface_id": sid, "source_mesh_used": False},
        )
        for sid in used_surface_ids
    )
    faces = tuple(tuple(candidate_id[sid] for sid in face) for face in faces_surface)
    edges_surface = sorted({
        _pair(face[i], face[j])
        for face in faces_surface
        for i, j in ((0, 1), (1, 2), (2, 0))
    })
    edges = tuple((candidate_id[a], candidate_id[b]) for a, b in edges_surface)

    provisional = CanonicalMeshCandidateIR(
        vertices=vertices,
        faces=faces,
        edges=edges,
        surface_binding_hash=surface.geometry_lineage_hash,
        partition_binding_hash=partition.partition_lineage_hash,
        carrier_policy_binding_hash=carrier_policy.carrier_policy_lineage_hash,
        producer_id="RealSaS.CanonicalRelationMeshBaseline.v1",
        producer_policy_hash=str(producer_policy_hash),
        candidate_lineage_hash="",
        metadata={
            "view_independent": True,
            "camera_authority_used": False,
            "source_mesh_used": False,
            "generated_vertex_count": 0,
            "identity_support_vertex_count": len(vertices),
            "provisional_unknown_edge_count": sum(edge in unknown_edges for edge in edges_surface),
            "rejected_relation_count": rejected_relation_count,
            "backend_role": "SAFE_BASELINE__NOT_FINAL_CDT_QUALITY_BACKEND",
        },
    )
    return CanonicalMeshCandidateIR(
        **{
            **provisional.__dict__,
            "candidate_lineage_hash": canonical_mesh_candidate_lineage_hash(provisional),
        }
    )
