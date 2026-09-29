from __future__ import annotations

"""View-independent conservative canonical mesh-candidate producer.

This producer supports two topology authorities. Legacy relation mode triangulates
3-cliques in RiggingSurfaceIR; product mode supplies exact compacted dense-face
provenance so pairwise relations cannot invent triangles. Every candidate vertex is
an IDENTITY_SURFACE_NODE binding. CDT/local-chart backends remain subordinate to
current qualification.

CDT/local-chart backends must emit the same CanonicalMeshCandidateIR and therefore
remain subordinate to current qualification.
"""

from dataclasses import replace
from itertools import combinations
import math

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

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
    explicit_face_provenance: tuple[tuple[str, str, str], ...] | None = None,
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
    if mechanical_skin_transfer not in {
        "OWNER_COPY",
        "COMPONENT_HARMONIC_DIRICHLET_V1",
    }:
        raise QualificationError("CANONICAL_MESH_MECHANICAL_SKIN_TRANSFER_UNSUPPORTED")

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
    rejected_explicit_face_count = 0
    if explicit_face_provenance is None:
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
    else:
        # Dense-face authority: a triangle is admitted only when the frozen
        # Stage12->Stage14 compaction replay provides an actual dense face witness.
        # Pairwise relation support remains mandatory, but can no longer mint a face.
        for raw_face in explicit_face_provenance:
            if len(raw_face) != 3:
                raise QualificationError("CANONICAL_MESH_EXPLICIT_FACE_ARITY_INVALID")
            tri = tuple(sorted(map(str, raw_face)))
            if len(set(tri)) != 3 or any(sid not in nodes for sid in tri):
                raise QualificationError("CANONICAL_MESH_EXPLICIT_FACE_ENDPOINT_INVALID")
            if len({owner.get(sid) for sid in tri}) != 1 or None in {owner.get(sid) for sid in tri}:
                raise QualificationError("CANONICAL_MESH_EXPLICIT_FACE_CROSS_COMPONENT")
            pairs = (_pair(tri[0], tri[1]), _pair(tri[1], tri[2]), _pair(tri[2], tri[0]))
            if any(pair not in safe_edges for pair in pairs):
                raise QualificationError("CANONICAL_MESH_EXPLICIT_FACE_RELATION_SUPPORT_MISSING")
            for pair in pairs:
                boundary = boundary_by_pair.get(pair)
                if boundary is not None and boundary.decision == "SEPARATE":
                    raise QualificationError("CANONICAL_MESH_EXPLICIT_FACE_CROSSES_SEPARATE")
            points = [nodes[sid].P for sid in tri]
            lengths = [
                math.dist(tuple(map(float, points[i])), tuple(map(float, points[j])))
                for i, j in ((0, 1), (1, 2), (2, 0))
            ]
            local_scale = max(lengths)
            if not math.isfinite(local_scale) or local_scale <= 0.0:
                rejected_explicit_face_count += 1
                continue
            relative_area = _triangle_double_area(*points) / (local_scale * local_scale)
            if relative_area < float(min_relative_double_area):
                rejected_explicit_face_count += 1
                continue
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
        producer_id=(
            "RealSaS.CanonicalDenseFaceProvenanceBaseline.v1"
            if explicit_face_provenance is not None
            else "RealSaS.CanonicalRelationMeshBaseline.v1"
        ),
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
            "face_provenance_mode": (
                "EXACT_COMPACTED_DENSE_FACE_REPLAY"
                if explicit_face_provenance is not None
                else "RELATION_GRAPH_THREE_CLIQUE"
            ),
            "explicit_face_provenance_input_count": (
                len(explicit_face_provenance)
                if explicit_face_provenance is not None
                else 0
            ),
            "rejected_explicit_face_count": int(rejected_explicit_face_count),
            "three_clique_face_minting_allowed": explicit_face_provenance is None,
            "backend_role": "SAFE_BASELINE__NOT_FINAL_CDT_QUALITY_BACKEND",
        },
    )
    return CanonicalMeshCandidateIR(
        **{
            **provisional.__dict__,
            "candidate_lineage_hash": canonical_mesh_candidate_lineage_hash(provisional),
        }
    )


def _apply_component_harmonic_skin_support(
    vertices_by_id: dict[str, CanonicalMeshVertexCandidateIR],
    out_faces: list[tuple[str, str, str]],
    *,
    support_epsilon: float = 1e-14,
) -> dict:
    """Replace generated seam owner-copy skin support with component-local harmonic support.

    Geometry support remains unchanged.  Mechanical support is represented only as
    a convex combination of IDENTITY_SURFACE_NODE anchors from the same mechanical
    component.  The operator depends on candidate geometry/connectivity and the
    partition only; QualifiedSkinIR weights are not inputs.
    """
    if not math.isfinite(float(support_epsilon)) or support_epsilon < 0.0:
        raise QualificationError("CANONICAL_MESH_HARMONIC_SUPPORT_EPS_INVALID")

    component_by_vid = {
        str(vid): str(vertex.component_id)
        for vid, vertex in vertices_by_id.items()
    }
    edge_weights: dict[tuple[str, str], float] = {}
    for face in out_faces:
        if len({component_by_vid[str(x)] for x in face}) != 1:
            raise QualificationError("CANONICAL_MESH_HARMONIC_CROSS_COMPONENT_FACE")
        for i, j in ((0, 1), (1, 2), (2, 0)):
            a, b = str(face[i]), str(face[j])
            pair = _pair(a, b)
            pa = vertices_by_id[pair[0]].P
            pb = vertices_by_id[pair[1]].P
            length = math.sqrt(sum(
                (float(pa[k]) - float(pb[k])) ** 2
                for k in range(3)
            ))
            if not math.isfinite(length) or length <= 1e-12:
                raise QualificationError("CANONICAL_MESH_HARMONIC_EDGE_DEGENERATE")
            edge_weights[pair] = max(edge_weights.get(pair, 0.0), 1.0 / length)

    neighbors: dict[str, list[tuple[str, float]]] = {
        str(vid): [] for vid in vertices_by_id
    }
    for (a, b), weight in edge_weights.items():
        if component_by_vid[a] != component_by_vid[b]:
            raise QualificationError("CANONICAL_MESH_HARMONIC_GRAPH_COMPONENT_LEAK")
        neighbors[a].append((b, float(weight)))
        neighbors[b].append((a, float(weight)))

    component_vertices: dict[str, list[str]] = {}
    for vid, component_id in component_by_vid.items():
        component_vertices.setdefault(component_id, []).append(vid)

    generated_count = 0
    stored_coefficient_count = 0
    max_support_count = 0
    component_solve_count = 0

    for component_id in sorted(component_vertices):
        vids = tuple(sorted(component_vertices[component_id]))
        unknown = tuple(
            vid for vid in vids
            if str(vertices_by_id[vid].support_binding.mode)
            == "SEAM_GEOMETRY_INTERPOLATION"
        )
        if not unknown:
            continue
        anchors = tuple(vid for vid in vids if vid not in set(unknown))
        if not anchors:
            raise QualificationError("CANONICAL_MESH_HARMONIC_COMPONENT_NO_ANCHOR")

        anchor_surface_id: dict[str, str] = {}
        for vid in anchors:
            binding = vertices_by_id[vid].support_binding
            coeffs = tuple(binding.coefficients)
            if (
                str(binding.mode) != "IDENTITY_SURFACE_NODE"
                or len(coeffs) != 1
                or abs(float(coeffs[0][1]) - 1.0) > 1e-12
            ):
                raise QualificationError("CANONICAL_MESH_HARMONIC_ANCHOR_INVALID")
            anchor_surface_id[vid] = str(coeffs[0][0])

        uindex = {vid: i for i, vid in enumerate(unknown)}
        aindex = {vid: i for i, vid in enumerate(anchors)}
        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []
        boundary = np.zeros((len(unknown), len(anchors)), dtype=np.float64)

        for row_index, vid in enumerate(unknown):
            total = 0.0
            for neighbor, weight in neighbors[vid]:
                if component_by_vid[neighbor] != component_id:
                    raise QualificationError("CANONICAL_MESH_HARMONIC_GRAPH_COMPONENT_LEAK")
                total += float(weight)
                if neighbor in uindex:
                    rows.append(row_index)
                    cols.append(uindex[neighbor])
                    vals.append(-float(weight))
                elif neighbor in aindex:
                    boundary[row_index, aindex[neighbor]] += float(weight)
                else:
                    raise QualificationError("CANONICAL_MESH_HARMONIC_NEIGHBOR_INVALID")
            if total <= 0.0:
                raise QualificationError("CANONICAL_MESH_HARMONIC_ZERO_DEGREE")
            rows.append(row_index)
            cols.append(row_index)
            vals.append(total)

        laplacian = csr_matrix(
            (vals, (rows, cols)),
            shape=(len(unknown), len(unknown)),
        )
        coordinates = np.asarray(
            spsolve(laplacian, boundary),
            dtype=np.float64,
        )
        if coordinates.ndim == 1:
            coordinates = coordinates[:, None]
        if (
            coordinates.shape != (len(unknown), len(anchors))
            or not np.isfinite(coordinates).all()
        ):
            raise QualificationError("CANONICAL_MESH_HARMONIC_SOLVE_INVALID")
        if float(np.min(coordinates)) < -1e-10:
            raise QualificationError("CANONICAL_MESH_HARMONIC_NEGATIVE_COORDINATE")

        coordinates = np.maximum(coordinates, 0.0)
        sums = coordinates.sum(axis=1, keepdims=True)
        if np.any(sums <= 1e-12):
            raise QualificationError("CANONICAL_MESH_HARMONIC_SIMPLEX_COLLAPSE")
        coordinates = coordinates / sums

        for local_index, vid in enumerate(unknown):
            row = coordinates[local_index]
            retained = np.nonzero(row > float(support_epsilon))[0]
            if not len(retained):
                retained = np.asarray([int(np.argmax(row))], dtype=np.int64)
            retained_total = float(np.sum(row[retained]))
            if retained_total <= 0.0:
                raise QualificationError("CANONICAL_MESH_HARMONIC_SUPPORT_EMPTY")
            support = tuple(sorted(
                (
                    anchor_surface_id[anchors[int(index)]],
                    float(row[int(index)] / retained_total),
                )
                for index in retained.tolist()
            ))
            if abs(sum(float(weight) for _, weight in support) - 1.0) > 1e-9:
                raise QualificationError("CANONICAL_MESH_HARMONIC_SUPPORT_SIMPLEX_INVALID")

            vertex = vertices_by_id[vid]
            binding = vertex.support_binding
            vertices_by_id[vid] = replace(
                vertex,
                support_binding=replace(
                    binding,
                    metadata={
                        **dict(binding.metadata or {}),
                        "skin_support_coefficients": support,
                        "mechanical_skin_transfer": "COMPONENT_HARMONIC_DIRICHLET_V1",
                        "harmonic_component_id": component_id,
                        "harmonic_support_count": len(support),
                        "harmonic_support_epsilon": float(support_epsilon),
                    },
                ),
            )
            generated_count += 1
            stored_coefficient_count += len(support)
            max_support_count = max(max_support_count, len(support))

        component_solve_count += 1

    return {
        "mechanical_skin_transfer": "COMPONENT_HARMONIC_DIRICHLET_V1",
        "generated_vertex_count": int(generated_count),
        "component_solve_count": int(component_solve_count),
        "stored_support_coefficient_count": int(stored_coefficient_count),
        "max_support_count_per_generated_vertex": int(max_support_count),
        "support_epsilon": float(support_epsilon),
        "cross_component_support_allowed": False,
        "qualified_skin_weights_used": False,
    }


def build_holeless_partitioned_dense_candidate(
    surface: RiggingSurfaceIR,
    partition: MechanicalPartitionIR,
    carrier_policy: ComponentCarrierPolicyIR,
    *,
    producer_policy_hash: str,
    explicit_face_provenance: tuple[tuple[str, str, str], ...],
    min_relative_double_area: float = 1e-8,
    mechanical_skin_transfer: str = "OWNER_COPY",
) -> CanonicalMeshCandidateIR:
    """Build dense-face topology after a mechanical repartition without deleting faces.

    A dense source triangle whose vertices now belong to different mechanical
    components is partitioned geometrically into component-pure sub-triangles.
    Seam geometry is shared in position but duplicated per component.  The special
    SEAM_GEOMETRY_INTERPOLATION support mode describes the geometric position; a
    separate skin_support_coefficients metadata field restricts mechanical skin
    transfer to the owning component.

    This is the product-form counterpart of the preregistered Knight holeless
    oracle: no source face is dropped and rest-area is conserved up to floating
    point tolerance.
    """
    validate_mechanical_partition(partition, surface)
    validate_component_carrier_policy(carrier_policy, partition)
    if not producer_policy_hash:
        raise QualificationError("CANONICAL_MESH_PRODUCER_POLICY_MISSING")
    if not explicit_face_provenance:
        raise QualificationError("CANONICAL_MESH_EXPLICIT_FACE_PROVENANCE_REQUIRED")
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
    safe_edges: set[tuple[str, str]] = set()
    for relation in surface.local_relations:
        a, b = str(relation.a_surface_id), str(relation.b_surface_id)
        if a == b or a not in nodes or b not in nodes:
            raise QualificationError("CANONICAL_MESH_RELATION_ENDPOINT_INVALID")
        if _relation_usable(relation):
            safe_edges.add(_pair(a, b))

    vertices_by_id: dict[str, CanonicalMeshVertexCandidateIR] = {}
    identity_id: dict[str, str] = {}

    def identity_vertex(sid: str) -> str:
        if sid in identity_id:
            return identity_id[sid]
        if sid not in nodes or sid not in owner:
            raise QualificationError("CANONICAL_MESH_EXPLICIT_FACE_ENDPOINT_INVALID")
        vid = "CMV:" + content_sha256({
            "surface_lineage_hash": surface.geometry_lineage_hash,
            "partition_lineage_hash": partition.partition_lineage_hash,
            "surface_id": sid,
        })[:24]
        identity_id[sid] = vid
        vertices_by_id[vid] = CanonicalMeshVertexCandidateIR(
            candidate_vertex_id=vid,
            support_binding=SurfaceSupportBinding(
                "IDENTITY_SURFACE_NODE",
                ((sid, 1.0),),
                metadata={"holeless_partitioned_dense": True},
            ),
            component_id=owner[sid],
            P=tuple(map(float, nodes[sid].P)),
            metadata={"source_surface_id": sid, "source_mesh_used": False},
        )
        return vid

    seam_cache: dict[tuple[str, str, str], str] = {}
    centroid_cache: dict[tuple[int, str], str] = {}

    def seam_vertex(a: str, b: str, component_id: str) -> str:
        pair = _pair(a, b)
        key = (pair[0], pair[1], component_id)
        if key in seam_cache:
            return seam_cache[key]
        owned = [sid for sid in pair if owner[sid] == component_id]
        if len(owned) != 1:
            raise QualificationError("CANONICAL_MESH_SEAM_OWNER_AMBIGUOUS")
        skin_sid = owned[0]
        coeffs = ((pair[0], 0.5), (pair[1], 0.5))
        p = tuple(
            0.5 * (float(nodes[pair[0]].P[k]) + float(nodes[pair[1]].P[k]))
            for k in range(3)
        )
        vid = "HSMV:" + content_sha256({
            "surface": surface.geometry_lineage_hash,
            "partition": partition.partition_lineage_hash,
            "edge": pair,
            "component": component_id,
        })[:24]
        vertices_by_id[vid] = CanonicalMeshVertexCandidateIR(
            candidate_vertex_id=vid,
            support_binding=SurfaceSupportBinding(
                "SEAM_GEOMETRY_INTERPOLATION",
                coeffs,
                metadata={
                    "mechanical_component_id": component_id,
                    "skin_support_coefficients": ((skin_sid, 1.0),),
                    "seam_geometry": "EDGE_MIDPOINT",
                },
            ),
            component_id=component_id,
            P=p,
            metadata={
                "generated_by": "HOLELESS_PARTITIONED_DENSE_V1",
                "seam_kind": "EDGE_MIDPOINT_COMPONENT_COPY",
                "source_mesh_used": False,
            },
        )
        seam_cache[key] = vid
        return vid

    def centroid_vertex(face_index: int, tri: tuple[str, str, str], component_id: str) -> str:
        key = (int(face_index), component_id)
        if key in centroid_cache:
            return centroid_cache[key]
        owned = [sid for sid in tri if owner[sid] == component_id]
        if len(owned) != 1:
            raise QualificationError("CANONICAL_MESH_CENTROID_OWNER_AMBIGUOUS")
        skin_sid = owned[0]
        coeffs = tuple((sid, 1.0 / 3.0) for sid in tri)
        p = tuple(sum(float(nodes[sid].P[k]) for sid in tri) / 3.0 for k in range(3))
        vid = "HSCV:" + content_sha256({
            "surface": surface.geometry_lineage_hash,
            "partition": partition.partition_lineage_hash,
            "face_index": int(face_index),
            "face": tri,
            "component": component_id,
        })[:24]
        vertices_by_id[vid] = CanonicalMeshVertexCandidateIR(
            candidate_vertex_id=vid,
            support_binding=SurfaceSupportBinding(
                "SEAM_GEOMETRY_INTERPOLATION",
                coeffs,
                metadata={
                    "mechanical_component_id": component_id,
                    "skin_support_coefficients": ((skin_sid, 1.0),),
                    "seam_geometry": "FACE_CENTROID",
                },
            ),
            component_id=component_id,
            P=p,
            metadata={
                "generated_by": "HOLELESS_PARTITIONED_DENSE_V1",
                "seam_kind": "FACE_CENTROID_COMPONENT_COPY",
                "source_mesh_used": False,
            },
        )
        centroid_cache[key] = vid
        return vid

    out_faces: list[tuple[str, str, str]] = []
    mixed_face_count = 0
    source_area = 0.0
    output_area = 0.0

    def area_by_ids(face_ids: tuple[str, str, str]) -> float:
        p = [vertices_by_id[x].P for x in face_ids]
        return 0.5 * _triangle_double_area(*p)

    for fi, raw_face in enumerate(explicit_face_provenance):
        if len(raw_face) != 3:
            raise QualificationError("CANONICAL_MESH_EXPLICIT_FACE_ARITY_INVALID")
        tri = tuple(map(str, raw_face))
        if len(set(tri)) != 3 or any(sid not in nodes or sid not in owner for sid in tri):
            raise QualificationError("CANONICAL_MESH_EXPLICIT_FACE_ENDPOINT_INVALID")
        if any(_pair(tri[i], tri[j]) not in safe_edges for i, j in ((0,1),(1,2),(2,0))):
            raise QualificationError("CANONICAL_MESH_EXPLICIT_FACE_RELATION_SUPPORT_MISSING")

        base_vids = tuple(identity_vertex(sid) for sid in tri)
        source_area += area_by_ids(base_vids)
        labels = tuple(owner[sid] for sid in tri)
        unique = tuple(sorted(set(labels)))
        made: list[tuple[str, str, str]] = []

        if len(unique) == 1:
            made = [base_vids]
        elif len(unique) == 2:
            mixed_face_count += 1
            for component_id in unique:
                own = [i for i, x in enumerate(labels) if x == component_id]
                other = [i for i, x in enumerate(labels) if x != component_id]
                if len(own) == 1:
                    i = own[0]
                    j, k = other
                    vi = base_vids[i]
                    a = seam_vertex(tri[i], tri[j], component_id)
                    b = seam_vertex(tri[i], tri[k], component_id)
                    made.append((vi, a, b))
                elif len(own) == 2:
                    i, j = own
                    k = other[0]
                    vi, vj = base_vids[i], base_vids[j]
                    a = seam_vertex(tri[i], tri[k], component_id)
                    b = seam_vertex(tri[j], tri[k], component_id)
                    made.extend(((vi, vj, b), (vi, b, a)))
                else:
                    raise QualificationError("CANONICAL_MESH_HOLELESS_TWO_REGION_SPLIT_INVALID")
        elif len(unique) == 3:
            mixed_face_count += 1
            for i in range(3):
                component_id = labels[i]
                vi = base_vids[i]
                a = seam_vertex(tri[i], tri[(i + 1) % 3], component_id)
                b = seam_vertex(tri[(i - 1) % 3], tri[i], component_id)
                cen = centroid_vertex(fi, tri, component_id)
                made.extend(((vi, a, cen), (vi, cen, b)))
        else:
            raise QualificationError("CANONICAL_MESH_HOLELESS_COMPONENT_COUNT_INVALID")

        for face in made:
            if len({vertices_by_id[x].component_id for x in face}) != 1:
                raise QualificationError("CANONICAL_MESH_HOLELESS_OUTPUT_NOT_COMPONENT_PURE")
            a = area_by_ids(face)
            if a <= 0.0 or not math.isfinite(a):
                raise QualificationError("CANONICAL_MESH_HOLELESS_DEGENERATE_OUTPUT_FACE")
            output_area += a
            out_faces.append(face)

    if not out_faces:
        raise QualificationError("CANONICAL_MESH_RELATION_BASELINE_NO_FACE")
    area_error = abs(output_area - source_area) / max(source_area, 1e-15)
    if area_error > 1e-10:
        raise QualificationError("CANONICAL_MESH_HOLELESS_REST_AREA_NOT_PRESERVED")

    skin_transfer_report = {
        "mechanical_skin_transfer": "OWNER_COPY",
        "generated_vertex_count": int(len(seam_cache) + len(centroid_cache)),
        "component_solve_count": 0,
        "stored_support_coefficient_count": int(len(seam_cache) + len(centroid_cache)),
        "max_support_count_per_generated_vertex": 1 if (seam_cache or centroid_cache) else 0,
        "support_epsilon": None,
        "cross_component_support_allowed": False,
        "qualified_skin_weights_used": False,
    }
    if mechanical_skin_transfer == "COMPONENT_HARMONIC_DIRICHLET_V1":
        skin_transfer_report = _apply_component_harmonic_skin_support(
            vertices_by_id,
            out_faces,
        )

    edges = tuple(sorted({
        tuple(sorted((str(face[i]), str(face[j]))))
        for face in out_faces
        for i, j in ((0,1),(1,2),(2,0))
    }))
    vertices = tuple(sorted(vertices_by_id.values(), key=lambda x: x.candidate_vertex_id))
    provisional = CanonicalMeshCandidateIR(
        vertices=vertices,
        faces=tuple(out_faces),
        edges=edges,
        surface_binding_hash=surface.geometry_lineage_hash,
        partition_binding_hash=partition.partition_lineage_hash,
        carrier_policy_binding_hash=carrier_policy.carrier_policy_lineage_hash,
        producer_id="RealSaS.HolelessPartitionedDenseFaceV1",
        producer_policy_hash=str(producer_policy_hash),
        candidate_lineage_hash="",
        metadata={
            "view_independent": True,
            "camera_authority_used": False,
            "source_mesh_used": False,
            "face_provenance_mode": "EXACT_COMPACTED_DENSE_FACE_REPLAY_HOLELESS_PARTITIONED",
            "three_clique_face_minting_allowed": False,
            "source_face_count": len(explicit_face_provenance),
            "mixed_source_face_count": int(mixed_face_count),
            "face_deletion_count": 0,
            "output_face_count": len(out_faces),
            "generated_seam_vertex_count": len(seam_cache),
            "generated_centroid_vertex_count": len(centroid_cache),
            "rest_area_relative_error": float(area_error),
            "dual_geometry_skin_support_required": bool(seam_cache or centroid_cache),
            "mechanical_skin_transfer": str(
                skin_transfer_report["mechanical_skin_transfer"]
            ),
            "mechanical_skin_transfer_report": dict(skin_transfer_report),
        },
    )
    return CanonicalMeshCandidateIR(
        **{
            **provisional.__dict__,
            "candidate_lineage_hash": canonical_mesh_candidate_lineage_hash(provisional),
        }
    )
