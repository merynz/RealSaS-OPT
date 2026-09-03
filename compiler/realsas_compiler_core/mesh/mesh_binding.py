from __future__ import annotations

import math
from typing import Iterable

from .hashing import content_sha256
from .types import (
    MeshDiscretizationCandidateIR,
    MeshVertexCandidate,
    QualifiedEditableMeshIR,
    QualifiedMeshVertex,
    QualifiedMeshSkinIR,
    QualifiedMeshSkinRow,
    RiggingSurfaceIR,
    QualifiedSkeletonIR,
    QualifiedSkinIR,
    SurfaceSupportBinding,
    Vec3,
    QualificationError,
)

_SIMPLEX_TOL = 1e-9
_POSITION_TOL = 1e-9
_ALLOWED_BINDING_MODES = {"IDENTITY_SURFACE_NODE", "LOCAL_CONVEX_INTERPOLATION"}


def _finite(value: float) -> bool:
    return math.isfinite(float(value))


def _without_lineage(value, field_name: str) -> dict:
    payload = value.to_dict()
    payload.pop(field_name, None)
    return payload


def mesh_candidate_lineage_hash(candidate: MeshDiscretizationCandidateIR) -> str:
    return content_sha256(_without_lineage(candidate, "candidate_lineage_hash"))


def mesh_lineage_hash(mesh: QualifiedEditableMeshIR) -> str:
    return content_sha256(_without_lineage(mesh, "mesh_lineage_hash"))


def mesh_skin_lineage_hash(mesh_skin: QualifiedMeshSkinIR) -> str:
    return content_sha256(_without_lineage(mesh_skin, "mesh_skin_lineage_hash"))


def validate_surface_support_binding(binding: SurfaceSupportBinding, *, known_surface_ids: Iterable[str] | None = None) -> None:
    if binding.mode not in _ALLOWED_BINDING_MODES:
        raise QualificationError(f"unsupported SurfaceSupportBinding mode:{binding.mode}")
    if not binding.coefficients:
        raise QualificationError("MESH_VERTEX_SUPPORT_INVALID: empty support coefficients")
    known = set(known_surface_ids) if known_surface_ids is not None else None
    seen: set[str] = set()
    total = 0.0
    for surface_id, coefficient in binding.coefficients:
        if not surface_id or surface_id in seen:
            raise QualificationError("MESH_VERTEX_SUPPORT_INVALID: duplicate/empty surface_id")
        seen.add(surface_id)
        if known is not None and surface_id not in known:
            raise QualificationError(f"MESH_VERTEX_SUPPORT_INVALID: unknown surface_id:{surface_id}")
        if not _finite(coefficient) or coefficient < 0.0:
            raise QualificationError("MESH_VERTEX_SUPPORT_INVALID: coefficient must be finite/nonnegative")
        total += float(coefficient)
    if abs(total - 1.0) > _SIMPLEX_TOL:
        raise QualificationError(f"MESH_VERTEX_SUPPORT_INVALID: coefficient simplex residual={abs(total-1.0)}")
    if binding.mode == "IDENTITY_SURFACE_NODE":
        if len(binding.coefficients) != 1 or abs(float(binding.coefficients[0][1]) - 1.0) > _SIMPLEX_TOL:
            raise QualificationError("MESH_VERTEX_SUPPORT_INVALID: identity binding must be exactly one coefficient of 1")


def derive_bound_position(surface: RiggingSurfaceIR, binding: SurfaceSupportBinding) -> Vec3:
    nodes = {node.surface_id: node for node in surface.surface_nodes}
    validate_surface_support_binding(binding, known_surface_ids=nodes)
    x = y = z = 0.0
    for surface_id, coefficient in binding.coefficients:
        p = nodes[surface_id].P
        c = float(coefficient)
        x += c * float(p[0]); y += c * float(p[1]); z += c * float(p[2])
    return (x, y, z)


def _require_position_matches(label: str, actual: Vec3, expected: Vec3) -> None:
    if any(not _finite(v) for v in actual):
        raise QualificationError(f"{label}: non-finite vertex position")
    error = max(abs(float(a) - float(b)) for a, b in zip(actual, expected))
    if error > _POSITION_TOL:
        raise QualificationError(f"{label}: rest position is not derived from admitted surface support; max_error={error}")


def _validate_topology_refs(vertex_ids: set[str], faces: tuple[tuple[str, ...], ...], edges: tuple[tuple[str, str], ...]) -> None:
    for face in faces:
        if len(face) < 3 or len(set(face)) != len(face):
            raise QualificationError("MESH_TOPOLOGY_INVALID: face must contain at least three unique vertices")
        if any(vertex_id not in vertex_ids for vertex_id in face):
            raise QualificationError("MESH_TOPOLOGY_INVALID: face references unknown vertex")
    for edge in edges:
        if len(edge) != 2 or edge[0] == edge[1]:
            raise QualificationError("MESH_TOPOLOGY_INVALID: edge must contain two distinct vertices")
        if edge[0] not in vertex_ids or edge[1] not in vertex_ids:
            raise QualificationError("MESH_TOPOLOGY_INVALID: edge references unknown vertex")


def validate_mesh_candidate(candidate: MeshDiscretizationCandidateIR, surface: RiggingSurfaceIR) -> None:
    if candidate.surface_binding_hash != surface.geometry_lineage_hash:
        raise QualificationError("MESH_SURFACE_LINEAGE_MISMATCH")
    if candidate.candidate_lineage_hash != mesh_candidate_lineage_hash(candidate):
        raise QualificationError("mesh candidate lineage hash mismatch")
    known_surface_ids = {node.surface_id for node in surface.surface_nodes}
    vertex_ids: set[str] = set()
    for vertex in candidate.vertices:
        if not vertex.candidate_vertex_id or vertex.candidate_vertex_id in vertex_ids:
            raise QualificationError("MESH_TOPOLOGY_INVALID: duplicate/empty candidate vertex id")
        vertex_ids.add(vertex.candidate_vertex_id)
        validate_surface_support_binding(vertex.support_binding, known_surface_ids=known_surface_ids)
        _require_position_matches("MESH_VERTEX_SUPPORT_INVALID", vertex.P, derive_bound_position(surface, vertex.support_binding))
    _validate_topology_refs(vertex_ids, candidate.faces, candidate.edges)


def validate_qualified_mesh(mesh: QualifiedEditableMeshIR, surface: RiggingSurfaceIR) -> None:
    if mesh.surface_binding_hash != surface.geometry_lineage_hash:
        raise QualificationError("MESH_SURFACE_LINEAGE_MISMATCH")
    if mesh.mesh_lineage_hash != mesh_lineage_hash(mesh):
        raise QualificationError("qualified mesh lineage hash mismatch")
    known_surface_ids = {node.surface_id for node in surface.surface_nodes}
    vertex_ids: set[str] = set()
    for vertex in mesh.vertices:
        if not vertex.canonical_mesh_vertex_id or vertex.canonical_mesh_vertex_id in vertex_ids:
            raise QualificationError("MESH_TOPOLOGY_INVALID: duplicate/empty canonical mesh vertex id")
        vertex_ids.add(vertex.canonical_mesh_vertex_id)
        validate_surface_support_binding(vertex.support_binding, known_surface_ids=known_surface_ids)
        _require_position_matches("MESH_VERTEX_SUPPORT_INVALID", vertex.P, derive_bound_position(surface, vertex.support_binding))
    _validate_topology_refs(vertex_ids, mesh.faces, mesh.edges)


def validate_qualified_mesh_skin(mesh_skin: QualifiedMeshSkinIR, *, surface: RiggingSurfaceIR, skeleton: QualifiedSkeletonIR, skin: QualifiedSkinIR, mesh: QualifiedEditableMeshIR) -> None:
    if mesh_skin.surface_binding_hash != surface.geometry_lineage_hash: raise QualificationError("MESH_SURFACE_LINEAGE_MISMATCH")
    if mesh_skin.skeleton_binding_hash != skeleton.skeleton_lineage_hash: raise QualificationError("MESH_WEIGHT_SKIN_LINEAGE_MISMATCH: skeleton")
    if mesh_skin.skin_binding_hash != skin.skin_lineage_hash: raise QualificationError("MESH_WEIGHT_SKIN_LINEAGE_MISMATCH")
    if mesh_skin.mesh_binding_hash != mesh.mesh_lineage_hash: raise QualificationError("MESH_WEIGHT_MESH_LINEAGE_MISMATCH")
    if mesh_skin.mesh_skin_lineage_hash != mesh_skin_lineage_hash(mesh_skin): raise QualificationError("qualified mesh-skin lineage hash mismatch")
    mesh_vertex_ids = {vertex.canonical_mesh_vertex_id for vertex in mesh.vertices}
    joint_ids = {joint.canonical_joint_id for joint in skeleton.joints}
    seen_rows: set[str] = set()
    for row in mesh_skin.rows:
        if row.canonical_mesh_vertex_id not in mesh_vertex_ids: raise QualificationError("MESH_WEIGHT_UNSUPPORTED_VERTEX")
        if row.canonical_mesh_vertex_id in seen_rows: raise QualificationError("duplicate mesh-skin row")
        seen_rows.add(row.canonical_mesh_vertex_id)
        if not row.influences: raise QualificationError("MESH_WEIGHT_SIMPLEX_RESIDUAL: empty row")
        total = 0.0
        for joint_id, weight in row.influences:
            if joint_id not in joint_ids: raise QualificationError("MESH_WEIGHT_ILLEGAL_JOINT_REFERENCE")
            if not _finite(weight) or weight < 0.0: raise QualificationError("MESH_WEIGHT_SIMPLEX_RESIDUAL: non-finite/negative weight")
            total += float(weight)
        residual = abs(total - 1.0)
        if residual > _SIMPLEX_TOL: raise QualificationError(f"MESH_WEIGHT_SIMPLEX_RESIDUAL:{residual}")
        if not _finite(row.simplex_residual_before) or not _finite(row.correction_l1): raise QualificationError("MESH_WEIGHT_SIMPLEX_RESIDUAL: non-finite diagnostics")
    if seen_rows != mesh_vertex_ids: raise QualificationError("MESH_WEIGHT_UNSUPPORTED_VERTEX: missing mesh vertex row")


def qualify_identity_subset_mesh(surface: RiggingSurfaceIR, candidate: MeshDiscretizationCandidateIR) -> QualifiedEditableMeshIR:
    """MWB-1 baseline qualifier: admit only identity-bound existing surface nodes.

    Topology is supplied by the candidate; this function does not triangulate or create geometry.
    """
    validate_mesh_candidate(candidate, surface)
    for vertex in candidate.vertices:
        if vertex.support_binding.mode != "IDENTITY_SURFACE_NODE":
            raise QualificationError("MWB1 requires IDENTITY_SURFACE_NODE bindings only")
    ordered = tuple(sorted(candidate.vertices, key=lambda v: v.candidate_vertex_id))
    id_map = {v.candidate_vertex_id: f"MV:{i:04d}" for i, v in enumerate(ordered)}
    vertices = tuple(QualifiedMeshVertex(id_map[v.candidate_vertex_id], v.P, v.support_binding, v.candidate_vertex_id, dict(v.metadata)) for v in ordered)
    faces = tuple(tuple(id_map[x] for x in face) for face in candidate.faces)
    edges = tuple((id_map[a], id_map[b]) for a, b in candidate.edges)
    report = {
        "status": "PASS_IDENTITY_SUBSET_QUALIFICATION",
        "candidate_lineage_hash": candidate.candidate_lineage_hash,
        "vertex_count": len(vertices),
        "face_count": len(faces),
        "edge_count": len(edges),
        "no_new_geometry_authority": True,
        "camera_geometry_consumed": bool(candidate.metadata.get("camera_geometry_consumed", False)),
        "numerical_solver_promoted": False,
    }
    mesh = QualifiedEditableMeshIR(vertices, faces, edges, surface.geometry_lineage_hash, candidate.view_index, candidate.camera_binding_hash, report, "", candidate.boundary_constraints, candidate.coverage_classification, metadata={"source_candidate_lineage_hash": candidate.candidate_lineage_hash})
    mesh = QualifiedEditableMeshIR(**{**mesh.__dict__, "mesh_lineage_hash": mesh_lineage_hash(mesh)})
    validate_qualified_mesh(mesh, surface)
    return mesh


def bind_identity_mesh_skin(surface: RiggingSurfaceIR, skeleton: QualifiedSkeletonIR, skin: QualifiedSkinIR, mesh: QualifiedEditableMeshIR) -> QualifiedMeshSkinIR:
    """MWB-1 exact-copy binder for identity-bound mesh vertices only."""
    validate_qualified_mesh(mesh, surface)
    if skin.surface_binding_hash != surface.geometry_lineage_hash: raise QualificationError("MESH_WEIGHT_SKIN_LINEAGE_MISMATCH: surface")
    if skin.skeleton_binding_hash != skeleton.skeleton_lineage_hash: raise QualificationError("MESH_WEIGHT_SKIN_LINEAGE_MISMATCH: skeleton")
    source_rows = {row.surface_id: row for row in skin.rows}
    rows = []
    for vertex in sorted(mesh.vertices, key=lambda v: v.canonical_mesh_vertex_id):
        binding = vertex.support_binding
        if binding.mode != "IDENTITY_SURFACE_NODE" or len(binding.coefficients) != 1 or binding.coefficients[0][1] != 1.0:
            raise QualificationError("MWB1 identity binder received non-identity mesh vertex")
        surface_id = binding.coefficients[0][0]
        if surface_id not in source_rows: raise QualificationError("MESH_WEIGHT_UNSUPPORTED_VERTEX")
        src = source_rows[surface_id]
        rows.append(QualifiedMeshSkinRow(vertex.canonical_mesh_vertex_id, src.influences, ((surface_id, 1.0),), src.simplex_residual_before, 0.0))
    report = {"status": "PASS_IDENTITY_EXACT_COPY", "row_count": len(rows), "transfer_method": "IDENTITY_SURFACE_NODE_EXACT_COPY_V1", "semantic_skin_synthesis": False}
    bound = QualifiedMeshSkinIR(tuple(rows), surface.geometry_lineage_hash, skeleton.skeleton_lineage_hash, skin.skin_lineage_hash, mesh.mesh_lineage_hash, "IDENTITY_SURFACE_NODE_EXACT_COPY_V1", report, "", metadata={"exact_copy": True})
    bound = QualifiedMeshSkinIR(**{**bound.__dict__, "mesh_skin_lineage_hash": mesh_skin_lineage_hash(bound)})
    validate_qualified_mesh_skin(bound, surface=surface, skeleton=skeleton, skin=skin, mesh=mesh)
    return bound
