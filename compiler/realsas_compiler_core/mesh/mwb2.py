from __future__ import annotations

from itertools import combinations

from .hashing import content_sha256
from .types import (
    MeshDiscretizationCandidateIR,
    MeshVertexCandidate,
    RiggingSurfaceIR,
    SurfaceSupportBinding,
    QualificationError,
)

_FORBIDDEN_RELATION_TOKENS = ("UNKNOWN", "AMBIGUOUS", "OCCLUDED", "UNOBSERVED")


def _visible_binding(node, view_index: int):
    if int(view_index) not in set(map(int, node.support_views)):
        return None
    vals = [tuple(map(float, xy)) for v, xy in node.raster_bindings if int(v) == int(view_index)]
    if len(vals) != 1:
        return None
    return vals[0]


def _relation_safe(relation) -> bool:
    kind = str(relation.relation_kind).upper()
    if any(tok in kind for tok in _FORBIDDEN_RELATION_TOKENS):
        return False
    md = dict(getattr(relation, "metadata", {}) or {})
    if bool(md.get("crosses_unknown", False)) or bool(md.get("unknown_bridge", False)):
        return False
    return float(relation.score) > 0.0


def _signed_area2(pa, pb, pc) -> float:
    return float((pb[0] - pa[0]) * (pc[1] - pa[1]) - (pb[1] - pa[1]) * (pc[0] - pa[0]))


def _oriented_triangle(tri: tuple[str, str, str], visible: dict[str, tuple[float, float]]) -> tuple[str, str, str]:
    a, b, c = tri
    if _signed_area2(visible[a], visible[b], visible[c]) > 0.0:
        return (a, b, c)
    return (a, c, b)


def _clip_polygon_against_edge(poly, a, b, *, eps: float = 1e-12):
    if not poly:
        return []

    def side(p):
        return float((b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]))

    out = []
    prev = poly[-1]
    prev_side = side(prev)
    prev_inside = prev_side >= -eps
    for cur in poly:
        cur_side = side(cur)
        cur_inside = cur_side >= -eps
        if cur_inside != prev_inside:
            denom = prev_side - cur_side
            if abs(denom) > eps:
                t = prev_side / denom
                out.append((prev[0] + t * (cur[0] - prev[0]), prev[1] + t * (cur[1] - prev[1])))
        if cur_inside:
            out.append(cur)
        prev = cur
        prev_side = cur_side
        prev_inside = cur_inside
    return out


def _polygon_area(poly) -> float:
    if len(poly) < 3:
        return 0.0
    total = 0.0
    for i, p in enumerate(poly):
        q = poly[(i + 1) % len(poly)]
        total += float(p[0] * q[1] - q[0] * p[1])
    return 0.5 * abs(total)


def _triangle_overlap_area(a_tri: tuple[str, str, str], b_tri: tuple[str, str, str], visible: dict[str, tuple[float, float]]) -> float:
    poly = [visible[sid] for sid in a_tri]
    clip = [visible[sid] for sid in b_tri]
    for i in range(3):
        poly = _clip_polygon_against_edge(poly, clip[i], clip[(i + 1) % 3])
        if len(poly) < 3:
            return 0.0
    return _polygon_area(poly)


def _select_nonoverlapping_faces(
    candidate_faces: set[tuple[str, str, str]],
    visible: dict[str, tuple[float, float]],
    *,
    min_face_area_grid: float,
) -> tuple[tuple[str, str, str], ...]:
    """Choose a deterministic planar subset of relation-supported triangle cliques.

    Eight-connected native raster neighborhoods deliberately contain both diagonals of
    every 2x2 cell. Treating every 3-clique as a face double-covers those cells and
    creates four-face interior edges. We retain only relation-supported triangles whose
    positive-area interiors do not overlap already admitted faces. Shared vertices and
    shared boundary edges remain legal. The output winding is consistently CCW in the
    target view and no source mesh or teacher topology is consulted.
    """
    ordered = []
    for tri in sorted(candidate_faces):
        oriented = _oriented_triangle(tri, visible)
        pa, pb, pc = (visible[x] for x in oriented)
        area = 0.5 * abs(_signed_area2(pa, pb, pc))
        if area >= float(min_face_area_grid):
            ordered.append((tri, oriented, area))

    admitted: list[tuple[str, str, str]] = []
    overlap_eps = max(1e-12, float(min_face_area_grid) * 1e-3)
    for _, oriented, _ in ordered:
        if any(_triangle_overlap_area(oriented, old, visible) > overlap_eps for old in admitted):
            continue
        admitted.append(oriented)
    return tuple(admitted)


def build_mwb2_candidate(surface: RiggingSurfaceIR, *, view_index: int, camera_binding_hash: str, min_face_area_grid: float = 1e-8) -> MeshDiscretizationCandidateIR:
    """Conservative direction-local mesh from admitted S only; never source mesh."""
    if not (0 <= int(view_index) < 8):
        raise ValueError("view_index must be in [0,7]")
    if not camera_binding_hash:
        raise ValueError("camera_binding_hash required")
    if min_face_area_grid <= 0:
        raise ValueError("min_face_area_grid must be positive")

    nodes = {n.surface_id: n for n in surface.surface_nodes}
    visible = {sid: _visible_binding(n, int(view_index)) for sid, n in nodes.items()}
    visible = {sid: xy for sid, xy in visible.items() if xy is not None}
    if len(visible) < 3:
        raise QualificationError("MWB2_INSUFFICIENT_OBSERVED_SURFACE_SUPPORT")

    safe_edges: set[tuple[str, str]] = set()
    rejected_unknown = 0
    for rel in surface.local_relations:
        a, b = str(rel.a_surface_id), str(rel.b_surface_id)
        if a == b or a not in visible or b not in visible:
            continue
        if not _relation_safe(rel):
            rejected_unknown += 1
            continue
        safe_edges.add(tuple(sorted((a, b))))
    if len(safe_edges) < 3:
        raise QualificationError("MWB2_NO_SAFE_LOCAL_RELATION_COMPLEX")

    neighbors: dict[str, set[str]] = {sid: set() for sid in visible}
    for a, b in safe_edges:
        neighbors[a].add(b)
        neighbors[b].add(a)

    clique_faces: set[tuple[str, str, str]] = set()
    for a in sorted(visible):
        for b, c in combinations(sorted(neighbors[a]), 2):
            if tuple(sorted((b, c))) not in safe_edges:
                continue
            tri = tuple(sorted((a, b, c)))
            pa, pb, pc = (visible[x] for x in tri)
            if 0.5 * abs(_signed_area2(pa, pb, pc)) >= float(min_face_area_grid):
                clique_faces.add(tri)

    faces = _select_nonoverlapping_faces(clique_faces, visible, min_face_area_grid=float(min_face_area_grid))
    if not faces:
        raise QualificationError("MWB2_NO_NONDEGENERATE_SAFE_FACE")

    used_ids = sorted({sid for tri in faces for sid in tri})
    used_edges = sorted({tuple(sorted(e)) for tri in faces for e in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[0], tri[2]))})
    candidate_id = {sid: f"MWB2:{view_index}:{i:05d}" for i, sid in enumerate(used_ids)}
    vertices = tuple(
        MeshVertexCandidate(
            candidate_id[sid],
            tuple(map(float, nodes[sid].P)),
            SurfaceSupportBinding("IDENTITY_SURFACE_NODE", ((sid, 1.0),), metadata={"observed_view": int(view_index)}),
            metadata={"source_surface_id": sid, "raster_xy": visible[sid], "source_mesh_used": False},
        )
        for sid in used_ids
    )
    face_ids = tuple(tuple(candidate_id[s] for s in tri) for tri in faces)
    edge_ids = tuple((candidate_id[a], candidate_id[b]) for a, b in used_edges)
    degree = {sid: 0 for sid in used_ids}
    for a, b in used_edges:
        degree[a] += 1
        degree[b] += 1
    boundary = tuple(
        {"candidate_vertex_id": candidate_id[sid], "kind": "LOW_RELATION_DEGREE"}
        for sid in used_ids
        if degree[sid] < 3
    )

    provisional = MeshDiscretizationCandidateIR(
        vertices,
        face_ids,
        edge_ids,
        surface.geometry_lineage_hash,
        int(view_index),
        str(camera_binding_hash),
        "",
        boundary,
        "OBSERVED_SAFE_LOCAL_RELATION_COMPLEX",
        solver_provenance={"solver": "DETERMINISTIC_NONOVERLAP_RELATION_TRIANGULATION", "cdt_promoted": False},
        residual_report={
            "face_count": len(face_ids),
            "edge_count": len(edge_ids),
            "vertex_count": len(vertices),
            "relation_clique_face_count": len(clique_faces),
            "overlap_rejected_face_count": len(clique_faces) - len(faces),
        },
        metadata={
            "producer": "RealSaS.MWB2.ObservedRelationComplex.v2",
            "source_mesh_used": False,
            "unknown_bridge_forbidden": True,
            "rejected_unknown_relation_count": int(rejected_unknown),
            "observed_view": int(view_index),
            "surface_lineage_hash": surface.geometry_lineage_hash,
            "target_view_winding": "CCW",
            "positive_area_overlap_forbidden": True,
        },
    )
    payload = provisional.to_dict()
    payload.pop("candidate_lineage_hash", None)
    return MeshDiscretizationCandidateIR(**{**provisional.__dict__, "candidate_lineage_hash": content_sha256(payload)})


def qualify_mwb2_mesh(surface: RiggingSurfaceIR, candidate: MeshDiscretizationCandidateIR):
    from .mesh_binding import qualify_identity_subset_mesh

    if candidate.metadata.get("source_mesh_used") is not False:
        raise QualificationError("MWB2_SOURCE_MESH_AUTHORITY_FORBIDDEN")
    if candidate.coverage_classification != "OBSERVED_SAFE_LOCAL_RELATION_COMPLEX":
        raise QualificationError("MWB2_COVERAGE_CLASSIFICATION_DRIFT")
    return qualify_identity_subset_mesh(surface, candidate)
