from __future__ import annotations

"""Compiler-owned bridge between view-local raster samples and canonical mechanics.

The key invariant is deliberately stronger than per-vertex support validity:

    every admitted mesh face must live inside one relation-supported canonical
    surface triangle, and mesh skin must be derived from those same support
    coordinates.

This prevents a view-space triangulator from connecting two raster-near vertices
that belong to unrelated mechanical neighborhoods.  No source mesh, teacher
mesh topology, learned repair, or historical skin is consulted here.
"""

from collections import defaultdict
from typing import Iterable

from .mesh_binding import validate_qualified_mesh
from .mwb2 import build_mwb2_candidate
from .types import QualificationError, RiggingSurfaceIR


ANCHOR_UNIFIED_TOPOLOGY_METHOD = "GSA_VIEW_COMMON_ANCHOR_TOPOLOGY_V1"
_MAX_EXAMPLES = 32
_SUPPORT_EPS = 1.0e-12


def _support_ids(binding) -> tuple[str, ...]:
    ids = tuple(
        sorted(
            str(surface_id)
            for surface_id, coefficient in binding.coefficients
            if float(coefficient) > _SUPPORT_EPS
        )
    )
    if not ids:
        raise QualificationError("ANCHOR_UNIFIED_EMPTY_VERTEX_SUPPORT")
    return ids


def relation_supported_parent_triangles(
    surface: RiggingSurfaceIR,
    *,
    view_index: int,
    camera_binding_hash: str,
) -> tuple[tuple[str, str, str], ...]:
    """Return the view-visible canonical parent triangles licensed by safe GSA relations.

    ``build_mwb2_candidate`` is intentionally used as the parent authority because it
    admits only positive-area 3-cliques from safe ``RiggingSurfaceIR.local_relations``
    and removes positive-area overlap deterministically.  The returned identities are
    canonical ``surface_id`` values, never view-local candidate IDs.
    """
    candidate = build_mwb2_candidate(
        surface,
        view_index=int(view_index),
        camera_binding_hash=str(camera_binding_hash),
    )
    sid_by_candidate: dict[str, str] = {}
    for vertex in candidate.vertices:
        coeff = tuple(vertex.support_binding.coefficients)
        if (
            vertex.support_binding.mode != "IDENTITY_SURFACE_NODE"
            or len(coeff) != 1
            or abs(float(coeff[0][1]) - 1.0) > _SUPPORT_EPS
        ):
            raise QualificationError("ANCHOR_UNIFIED_PARENT_VERTEX_NOT_IDENTITY_BOUND")
        sid_by_candidate[str(vertex.candidate_vertex_id)] = str(coeff[0][0])

    parents = []
    seen = set()
    for face in candidate.faces:
        if len(face) != 3:
            raise QualificationError("ANCHOR_UNIFIED_PARENT_NOT_TRIANGULAR")
        parent = tuple(sorted(sid_by_candidate[str(vertex_id)] for vertex_id in face))
        if len(set(parent)) != 3:
            raise QualificationError("ANCHOR_UNIFIED_PARENT_DEGENERATE")
        if parent not in seen:
            seen.add(parent)
            parents.append(parent)
    if not parents:
        raise QualificationError("ANCHOR_UNIFIED_NO_RELATION_SUPPORTED_PARENT")
    return tuple(sorted(parents))


def _parent_inverted_index(parents: tuple[tuple[str, str, str], ...]):
    by_surface: dict[str, set[int]] = defaultdict(set)
    for index, parent in enumerate(parents):
        for surface_id in parent:
            by_surface[str(surface_id)].add(int(index))
    return by_surface


def _matching_parent_indices(
    support_ids: Iterable[str],
    *,
    by_surface: dict[str, set[int]],
) -> tuple[int, ...]:
    ids = tuple(sorted(set(map(str, support_ids))))
    if not ids:
        return ()
    rows = [set(by_surface.get(surface_id, ())) for surface_id in ids]
    if not rows or any(not row for row in rows):
        return ()
    common = rows[0]
    for row in rows[1:]:
        common = common & row
        if not common:
            break
    return tuple(sorted(common))


def anchor_unified_topology_report(surface: RiggingSurfaceIR, mesh) -> dict:
    """Measure the common-anchor topology invariant for one qualified mesh.

    A vertex is licensed only when all of its support coefficients fit inside at
    least one safe relation-supported canonical parent triangle.  A face is licensed
    only when the *union* of support IDs from all of its vertices fits inside one
    such parent.  Therefore topology and subsequent convex skin transfer share one
    canonical support simplex instead of meeting only after independent derivation.
    """
    validate_qualified_mesh(mesh, surface)
    parents = relation_supported_parent_triangles(
        surface,
        view_index=int(mesh.view_index),
        camera_binding_hash=str(mesh.camera_binding_hash),
    )
    by_surface = _parent_inverted_index(parents)
    vertices = {str(vertex.canonical_mesh_vertex_id): vertex for vertex in mesh.vertices}

    vertex_parent_candidates: dict[str, tuple[int, ...]] = {}
    vertex_violations = []
    for vertex_id, vertex in sorted(vertices.items()):
        support = _support_ids(vertex.support_binding)
        matched = _matching_parent_indices(support, by_surface=by_surface)
        vertex_parent_candidates[vertex_id] = matched
        if not matched and len(vertex_violations) < _MAX_EXAMPLES:
            vertex_violations.append({
                "canonical_mesh_vertex_id": vertex_id,
                "support_surface_ids": support,
            })

    face_violations = []
    admitted_faces = 0
    for face_index, face in enumerate(mesh.faces):
        support_union: set[str] = set()
        for vertex_id in face:
            support_union.update(_support_ids(vertices[str(vertex_id)].support_binding))
        matched = _matching_parent_indices(support_union, by_surface=by_surface)
        if matched:
            admitted_faces += 1
        elif len(face_violations) < _MAX_EXAMPLES:
            face_violations.append({
                "face_index": int(face_index),
                "vertex_ids": tuple(map(str, face)),
                "support_union": tuple(sorted(support_union)),
            })

    vertex_fail_count = sum(not rows for rows in vertex_parent_candidates.values())
    face_fail_count = len(mesh.faces) - admitted_faces
    passed = vertex_fail_count == 0 and face_fail_count == 0
    return {
        "status": (
            "PASS__GSA_VIEW_COMMON_ANCHOR_TOPOLOGY"
            if passed
            else "FAIL__GSA_VIEW_COMMON_ANCHOR_TOPOLOGY"
        ),
        "passed": bool(passed),
        "method": ANCHOR_UNIFIED_TOPOLOGY_METHOD,
        "surface_lineage_hash": str(surface.geometry_lineage_hash),
        "mesh_lineage_hash": str(mesh.mesh_lineage_hash),
        "view_index": int(mesh.view_index),
        "relation_supported_parent_triangle_count": len(parents),
        "vertex_count": len(mesh.vertices),
        "face_count": len(mesh.faces),
        "vertex_without_common_parent_count": int(vertex_fail_count),
        "face_without_common_parent_count": int(face_fail_count),
        "vertex_violation_examples": tuple(vertex_violations),
        "face_violation_examples": tuple(face_violations),
        "source_mesh_used": False,
        "teacher_topology_used": False,
        "topology_and_skin_share_support_simplex": True,
        "view_space_proximity_alone_can_create_edge": False,
    }


def require_anchor_unified_topology(surface: RiggingSurfaceIR, mesh) -> dict:
    report = anchor_unified_topology_report(surface, mesh)
    if not bool(report["passed"]):
        raise QualificationError(
            "ANCHOR_UNIFIED_TOPOLOGY_REQUIRED:"
            f"vertices={report['vertex_without_common_parent_count']}:"
            f"faces={report['face_without_common_parent_count']}"
        )
    return report


__all__ = [
    "ANCHOR_UNIFIED_TOPOLOGY_METHOD",
    "anchor_unified_topology_report",
    "relation_supported_parent_triangles",
    "require_anchor_unified_topology",
]
