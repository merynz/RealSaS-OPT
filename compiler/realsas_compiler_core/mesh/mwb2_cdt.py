from __future__ import annotations

from dataclasses import replace
import math

from ._historical_v05 import HISTORICAL_CDT_SOURCE_SHA256, triangulate_production_cdt
from .hashing import content_sha256
from .mesh_binding import mesh_lineage_hash, qualify_identity_subset_mesh, validate_qualified_mesh
from .mwb2 import _oriented_triangle, _relation_safe, _signed_area2, _visible_binding
from .observation_domain import ObservationRasterDomain
from .types import (
    MeshDiscretizationCandidateIR,
    MeshVertexCandidate,
    QualificationError,
    RiggingSurfaceIR,
    SurfaceSupportBinding,
)

_CDT_MATCH_EPS = 1.0e-7


def _safe_surface_graph(surface: RiggingSurfaceIR, view_index: int):
    """Return full safe-S topology plus the observation-supported carriers for one view.

    Component identity belongs to S, not to a view-local induced subgraph. An S carrier
    that is temporarily invisible may therefore relay *component identity*, but it is
    never admitted as a 2D CDT vertex for that view. Unsafe/unknown relations remain
    forbidden and can never join components.
    """
    nodes = {n.surface_id: n for n in surface.surface_nodes}
    visible = {sid: _visible_binding(n, int(view_index)) for sid, n in nodes.items()}
    visible = {sid: xy for sid, xy in visible.items() if xy is not None}
    safe_edges: set[tuple[str, str]] = set()
    rejected_unknown = 0
    for rel in surface.local_relations:
        a, b = str(rel.a_surface_id), str(rel.b_surface_id)
        if a == b or a not in nodes or b not in nodes:
            continue
        if not _relation_safe(rel):
            rejected_unknown += 1
            continue
        safe_edges.add(tuple(sorted((a, b))))
    return nodes, visible, safe_edges, rejected_unknown


def _connected_components(node_ids: set[str], safe_edges: set[tuple[str, str]]) -> tuple[tuple[str, ...], ...]:
    neighbors: dict[str, set[str]] = {sid: set() for sid in node_ids}
    for a, b in safe_edges:
        if a in neighbors and b in neighbors:
            neighbors[a].add(b)
            neighbors[b].add(a)
    unseen = set(node_ids)
    out: list[tuple[str, ...]] = []
    while unseen:
        seed = min(unseen)
        stack = [seed]
        unseen.remove(seed)
        component: list[str] = []
        while stack:
            cur = stack.pop()
            component.append(cur)
            for nxt in sorted(neighbors[cur], reverse=True):
                if nxt in unseen:
                    unseen.remove(nxt)
                    stack.append(nxt)
        out.append(tuple(sorted(component)))
    return tuple(sorted(out, key=lambda comp: comp[0]))


def _visible_subsets_of_full_components(
    all_components: tuple[tuple[str, ...], ...],
    visible: dict[str, tuple[float, float]],
) -> tuple[tuple[str, ...], ...]:
    visible_ids = set(visible)
    out = []
    for component in all_components:
        subset = tuple(sid for sid in component if sid in visible_ids)
        if subset:
            out.append(subset)
    return tuple(out)


def _dedupe_projected_ids(
    ids: tuple[str, ...],
    visible: dict[str, tuple[float, float]],
    *,
    eps: float = _CDT_MATCH_EPS,
) -> tuple[str, ...]:
    kept: list[str] = []
    for sid in sorted(ids):
        p = visible[sid]
        if any(math.hypot(p[0] - visible[old][0], p[1] - visible[old][1]) <= eps for old in kept):
            continue
        kept.append(sid)
    return tuple(kept)


def _convex_hull_ids(ids: tuple[str, ...], visible: dict[str, tuple[float, float]]) -> tuple[str, ...]:
    points = sorted(
        ((float(visible[sid][0]), float(visible[sid][1]), sid) for sid in ids),
        key=lambda v: (v[0], v[1], v[2]),
    )
    if len(points) < 3:
        return ()

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 1.0e-12:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 1.0e-12:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    if len(hull) < 3:
        return ()
    area2 = sum(
        hull[i][0] * hull[(i + 1) % len(hull)][1] - hull[(i + 1) % len(hull)][0] * hull[i][1]
        for i in range(len(hull))
    )
    if abs(area2) <= 1.0e-12:
        return ()
    return tuple(p[2] for p in hull)


def _match_kernel_vertex(
    point,
    ids: tuple[str, ...],
    visible: dict[str, tuple[float, float]],
    *,
    eps: float = _CDT_MATCH_EPS,
) -> str | None:
    px, py = float(point[0]), float(point[1])
    best_sid = None
    best_dist = float("inf")
    for sid in ids:
        qx, qy = visible[sid]
        dist = math.hypot(px - float(qx), py - float(qy))
        if dist < best_dist or (
            abs(dist - best_dist) <= 1.0e-15 and (best_sid is None or sid < best_sid)
        ):
            best_dist = dist
            best_sid = sid
    return best_sid if best_dist <= eps else None

def _point_on_original_hull_segment(
    point,
    hull_ids: tuple[str, ...],
    visible: dict[str, tuple[float, float]],
    *,
    eps: float = _CDT_MATCH_EPS,
) -> tuple[str, str, float] | None:
    px, py = float(point[0]), float(point[1])
    matches: list[tuple[float, str, str, float]] = []
    for i, a_sid in enumerate(hull_ids):
        b_sid = hull_ids[(i + 1) % len(hull_ids)]
        ax, ay = visible[a_sid]
        bx, by = visible[b_sid]
        dx, dy = float(bx) - float(ax), float(by) - float(ay)
        denom = dx * dx + dy * dy
        if denom <= eps * eps:
            continue
        t = ((px - float(ax)) * dx + (py - float(ay)) * dy) / denom
        if t <= eps or t >= 1.0 - eps:
            continue
        qx, qy = float(ax) + t * dx, float(ay) + t * dy
        dist = math.hypot(px - qx, py - qy)
        if dist <= eps:
            matches.append((dist, a_sid, b_sid, float(t)))
    if not matches:
        return None
    matches.sort(key=lambda row: (row[0], row[1], row[2], row[3]))
    best = matches[0]
    if len(matches) > 1 and abs(matches[1][0] - best[0]) <= 1.0e-12:
        raise QualificationError("MWB2_CDT_AMBIGUOUS_BOUNDARY_RECOVERY_VERTEX")
    return best[1], best[2], best[3]


def _polygon_signed_area2(indices: list[int], points) -> float:
    return float(
        sum(
            points[indices[i]][0] * points[indices[(i + 1) % len(indices)]][1]
            - points[indices[(i + 1) % len(indices)]][0] * points[indices[i]][1]
            for i in range(len(indices))
        )
    )


def _point_in_triangle_xy(point, a, b, c, *, eps: float = 1.0e-10) -> bool:
    s0 = _signed_area2(a, b, point)
    s1 = _signed_area2(b, c, point)
    s2 = _signed_area2(c, a, point)
    has_neg = s0 < -eps or s1 < -eps or s2 < -eps
    has_pos = s0 > eps or s1 > eps or s2 > eps
    return not (has_neg and has_pos)


def _ear_clip_cavity(path: list[int], points) -> list[tuple[int, int, int]]:
    poly = list(path)
    if len(poly) < 3 or len(set(poly)) != len(poly):
        raise QualificationError("MWB2_CDT_BOUNDARY_CONTRACTION_INVALID_CAVITY")
    if _polygon_signed_area2(poly, points) < 0.0:
        poly.reverse()
    out: list[tuple[int, int, int]] = []
    while len(poly) > 3:
        ears = []
        for i, cur in enumerate(poly):
            a, b, c = poly[i - 1], cur, poly[(i + 1) % len(poly)]
            if _signed_area2(points[a], points[b], points[c]) <= 1.0e-10:
                continue
            tri = (a, b, c)
            if any(
                j not in tri
                and _point_in_triangle_xy(points[j], points[a], points[b], points[c])
                for j in poly
            ):
                continue
            ears.append((tuple(sorted(tri)), i, tri))
        if not ears:
            raise QualificationError("MWB2_CDT_BOUNDARY_CONTRACTION_EAR_CLIP_FAIL")
        _, index, tri = min(ears)
        out.append(tri)
        poly.pop(index)
    out.append(tuple(poly))
    return out


def _contract_boundary_recovery_vertices(result, generated_indices: tuple[int, ...]):
    """Remove solver-only boundary split vertices and fill their planar cavities."""
    triangles = [tuple(map(int, tri)) for tri in result.triangles]
    constraints = {tuple(sorted(map(int, edge))) for edge in result.constraint_edges}
    points = result.vertices
    for g in sorted(map(int, generated_indices)):
        incident = [tri for tri in triangles if g in tri]
        if not incident:
            raise QualificationError("MWB2_CDT_BOUNDARY_RECOVERY_VERTEX_UNUSED")
        constraint_incident = [edge for edge in constraints if g in edge]
        if len(constraint_incident) != 2:
            raise QualificationError("MWB2_CDT_GENERATED_VERTEX_NOT_BOUNDARY_RECOVERY")
        endpoints = [edge[0] if edge[1] == g else edge[1] for edge in constraint_incident]
        adjacency: dict[int, set[int]] = {}
        for tri in incident:
            opposite = [idx for idx in tri if idx != g]
            if len(opposite) != 2:
                raise QualificationError("MWB2_CDT_BOUNDARY_CONTRACTION_INVALID_STAR")
            a, b = opposite
            adjacency.setdefault(a, set()).add(b)
            adjacency.setdefault(b, set()).add(a)
        start, end = endpoints
        path = [start]
        previous = None
        current = start
        while current != end:
            candidates = [idx for idx in sorted(adjacency.get(current, ())) if idx != previous]
            candidates = [idx for idx in candidates if idx not in path or idx == end]
            if len(candidates) != 1:
                raise QualificationError("MWB2_CDT_BOUNDARY_CONTRACTION_NONMANIFOLD_STAR")
            nxt = candidates[0]
            previous, current = current, nxt
            path.append(current)
            if len(path) > len(adjacency) + 1:
                raise QualificationError("MWB2_CDT_BOUNDARY_CONTRACTION_PATH_LOOP")
        replacement = _ear_clip_cavity(path, points)
        triangles = [tri for tri in triangles if g not in tri] + replacement
        for edge in constraint_incident:
            constraints.remove(tuple(sorted(edge)))
        constraints.add(tuple(sorted((start, end))))
    if any(any(g in tri for g in generated_indices) for tri in triangles):
        raise QualificationError("MWB2_CDT_BOUNDARY_RECOVERY_CONTRACTION_INCOMPLETE")
    return tuple(triangles)


def build_mwb2_observation_cdt_candidate(
    surface: RiggingSurfaceIR,
    *,
    view_index: int,
    camera_binding_hash: str,
    observation_domain: ObservationRasterDomain,
    min_face_area_grid: float = 1.0e-8,
    min_component_nodes: int = 3,
) -> MeshDiscretizationCandidateIR:
    """Triangulate observed S carriers with the frozen v0.5 CDT numerical kernel.

    Current authority remains explicit: the *full* safe S graph defines legal 3D
    component identity, then the target view selects observation-supported carriers
    from each component. Exact input alpha defines the admissible 2D domain.
    Historical CDT is numerical machinery only. Quality-Steiner insertion is disabled;
    any output vertex that cannot be rebound exactly to an admitted S carrier is
    rejected fail-closed.
    """
    if not (0 <= int(view_index) < 8):
        raise ValueError("view_index must be in [0,7]")
    if not camera_binding_hash:
        raise ValueError("camera_binding_hash required")
    if min_face_area_grid <= 0.0:
        raise ValueError("min_face_area_grid must be positive")
    if int(min_component_nodes) < 3:
        raise ValueError("min_component_nodes must be >= 3")
    observation_domain.validate()
    if int(observation_domain.view_index) != int(view_index):
        raise QualificationError("MWB2_OBSERVATION_DOMAIN_VIEW_MISMATCH")

    nodes, visible, safe_edges, rejected_unknown = _safe_surface_graph(surface, int(view_index))
    if len(visible) < 3:
        raise QualificationError("MWB2_INSUFFICIENT_OBSERVED_SURFACE_SUPPORT")
    full_components = _connected_components(set(nodes), safe_edges)
    components = _visible_subsets_of_full_components(full_components, visible)
    visible_safe_edges = {
        (a, b) for a, b in safe_edges if a in visible and b in visible
    }

    admitted_faces: list[tuple[str, str, str]] = []
    admitted_face_xy: list[
        tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
    ] = []
    cdt_component_count = 0
    tiny_component_count = 0
    alpha_rejected_face_count = 0
    duplicate_projected_node_count = 0
    kernel_triangle_count = 0
    kernel_constraint_split_count = 0
    contracted_boundary_recovery_vertex_count = 0
    post_contraction_triangle_count = 0

    for component in components:
        deduped = _dedupe_projected_ids(component, visible)
        duplicate_projected_node_count += len(component) - len(deduped)
        if len(deduped) < int(min_component_nodes):
            tiny_component_count += 1
            continue
        hull_ids = _convex_hull_ids(deduped, visible)
        if len(hull_ids) < 3:
            tiny_component_count += 1
            continue
        hull_set = set(hull_ids)
        support_ids = tuple(sid for sid in deduped if sid not in hull_set)
        result = triangulate_production_cdt(
            [visible[sid] for sid in hull_ids],
            support_points=[visible[sid] for sid in support_ids],
            target_min_angle_deg=0.0,
            max_boundary_vertices=max(256, len(hull_ids) + 16),
            max_support_vertices=max(128, len(support_ids) + 16),
            max_constraint_recovery_iterations=96,
            max_quality_iterations=0,
            min_feature_spacing=1.0e-8,
        )
        if not bool(result.success):
            raise QualificationError(f"MWB2_CDT_KERNEL_FAILURE:{result.reason}")
        kernel_constraint_split_count += int(result.constraint_split_count)
        kernel_triangle_count += len(result.triangles)
        cdt_component_count += 1

        kernel_ids: list[str | None] = []
        generated_indices: list[int] = []
        for vertex_index, point in enumerate(result.vertices):
            sid = _match_kernel_vertex(point, deduped, visible)
            if sid is None:
                if _point_on_original_hull_segment(point, hull_ids, visible) is None:
                    raise QualificationError("MWB2_CDT_UNSUPPORTED_GENERATED_VERTEX")
                generated_indices.append(int(vertex_index))
            kernel_ids.append(sid)
        if generated_indices:
            if int(getattr(result, "quality_insert_count", 0)) != 0 or int(
                getattr(result, "inserted_steiner_count", 0)
            ) != 0:
                raise QualificationError("MWB2_CDT_UNBOUND_QUALITY_STEINER_FORBIDDEN")
            if len(generated_indices) != int(result.constraint_split_count):
                raise QualificationError("MWB2_CDT_CONSTRAINT_SPLIT_BINDING_COUNT_MISMATCH")
            kernel_triangles = _contract_boundary_recovery_vertices(result, tuple(generated_indices))
            contracted_boundary_recovery_vertex_count += len(generated_indices)
        else:
            kernel_triangles = tuple(result.triangles)
        matched_ids = [sid for sid in kernel_ids if sid is not None]
        if len(set(matched_ids)) != len(matched_ids):
            raise QualificationError("MWB2_CDT_VERTEX_COLLAPSE_AFTER_BINDING")
        post_contraction_triangle_count += len(kernel_triangles)

        for ia, ib, ic in kernel_triangles:
            tri_raw = (kernel_ids[int(ia)], kernel_ids[int(ib)], kernel_ids[int(ic)])
            if any(sid is None for sid in tri_raw):
                raise QualificationError("MWB2_CDT_BOUNDARY_RECOVERY_CONTRACTION_INCOMPLETE")
            tri = (str(tri_raw[0]), str(tri_raw[1]), str(tri_raw[2]))
            if len(set(tri)) != 3:
                continue
            tri = _oriented_triangle(tri, visible)
            pa, pb, pc = (visible[sid] for sid in tri)
            if 0.5 * abs(_signed_area2(pa, pb, pc)) < float(min_face_area_grid):
                continue
            tri_xy = (pa, pb, pc)
            if not observation_domain.triangle_inside(tri_xy):
                alpha_rejected_face_count += 1
                continue
            admitted_faces.append(tri)
            admitted_face_xy.append(tri_xy)

    if not admitted_faces:
        raise QualificationError("MWB2_CDT_NO_OBSERVATION_DOMAIN_FACE")

    face_keys: set[tuple[str, str, str]] = set()
    unique_faces: list[tuple[str, str, str]] = []
    unique_face_xy = []
    for tri, tri_xy in zip(admitted_faces, admitted_face_xy):
        key = tuple(sorted(tri))
        if key in face_keys:
            continue
        face_keys.add(key)
        unique_faces.append(tri)
        unique_face_xy.append(tri_xy)

    used_ids = sorted({sid for tri in unique_faces for sid in tri})
    candidate_id = {sid: f"MWB2CDT:{view_index}:{i:05d}" for i, sid in enumerate(used_ids)}
    vertices = tuple(
        MeshVertexCandidate(
            candidate_id[sid],
            tuple(map(float, nodes[sid].P)),
            SurfaceSupportBinding(
                "IDENTITY_SURFACE_NODE",
                ((sid, 1.0),),
                metadata={"observed_view": int(view_index), "cdt_kernel_vertex": True},
            ),
            metadata={
                "source_surface_id": sid,
                "raster_xy": tuple(map(float, visible[sid])),
                "source_mesh_used": False,
                "generated_geometry": False,
            },
        )
        for sid in used_ids
    )
    face_ids = tuple(tuple(candidate_id[sid] for sid in tri) for tri in unique_faces)
    used_edges = sorted(
        {
            tuple(sorted((tri[i], tri[(i + 1) % 3])))
            for tri in unique_faces
            for i in range(3)
        }
    )
    edge_ids = tuple((candidate_id[a], candidate_id[b]) for a, b in used_edges)
    coverage = observation_domain.coverage(unique_face_xy)
    boundary = (
        {
            "kind": "EXACT_OBSERVATION_ALPHA_DOMAIN",
            "view_index": int(view_index),
            "mask_sha256": observation_domain.mask_sha256,
            "source_alpha_sha256": observation_domain.source_alpha_sha256,
            "width": int(observation_domain.width),
            "height": int(observation_domain.height),
        },
    )
    residual = {
        **coverage,
        "face_count": len(face_ids),
        "edge_count": len(edge_ids),
        "vertex_count": len(vertices),
        "visible_surface_node_count": len(visible),
        "safe_relation_edge_count": len(visible_safe_edges),
        "full_safe_relation_edge_count": len(safe_edges),
        "safe_component_count": len(components),
        "full_safe_component_count": len(full_components),
        "cdt_component_count": int(cdt_component_count),
        "tiny_component_count": int(tiny_component_count),
        "kernel_triangle_count": int(kernel_triangle_count),
        "alpha_rejected_face_count": int(alpha_rejected_face_count),
        "duplicate_projected_node_count": int(duplicate_projected_node_count),
        "kernel_constraint_split_count": int(kernel_constraint_split_count),
        "contracted_boundary_recovery_vertex_count": int(contracted_boundary_recovery_vertex_count),
        "post_contraction_triangle_count": int(post_contraction_triangle_count),
    }
    provisional = MeshDiscretizationCandidateIR(
        vertices,
        face_ids,
        edge_ids,
        surface.geometry_lineage_hash,
        int(view_index),
        str(camera_binding_hash),
        "",
        boundary,
        "OBSERVATION_DOMAIN_CDT",
        solver_provenance={
            "solver": "HISTORICAL_V05_CDT_CURRENT_TYPED_ADAPTER_V3",
            "historical_cdt_source_sha256": HISTORICAL_CDT_SOURCE_SHA256,
            "historical_kernel_role": "NUMERICAL_ONLY",
            "quality_refinement_enabled": False,
            "generated_vertex_policy": "CONTRACT_BOUNDARY_RECOVERY_VERTICES__REJECT_OTHER_GENERATED_VERTICES",
            "cdt_promoted": False,
        },
        residual_report=residual,
        metadata={
            "producer": "RealSaS.MWB2.ObservationDomainCDT.v3",
            "source_mesh_used": False,
            "teacher_topology_used": False,
            "unknown_bridge_forbidden": True,
            "rejected_unknown_relation_count": int(rejected_unknown),
            "observed_view": int(view_index),
            "surface_lineage_hash": surface.geometry_lineage_hash,
            "observation_mask_sha256": observation_domain.mask_sha256,
            "source_alpha_sha256": observation_domain.source_alpha_sha256,
            "target_view_winding": "CCW",
            "component_partition_authority": "FULL_SAFE_SURFACE_RELATION_COMPONENTS_THEN_VISIBLE_SUBSET",
            "alpha_domain_authority": "EXACT_OBSERVATION_MASK",
        },
    )
    payload = provisional.to_dict()
    payload.pop("candidate_lineage_hash", None)
    return MeshDiscretizationCandidateIR(
        **{**provisional.__dict__, "candidate_lineage_hash": content_sha256(payload)}
    )


def qualify_mwb2_observation_cdt_mesh(
    surface: RiggingSurfaceIR,
    candidate: MeshDiscretizationCandidateIR,
    *,
    min_source_alpha_recall: float = 0.90,
    min_precision_inside_alpha: float = 0.995,
):
    """Fail-closed promotion gate for a current-authority CDT candidate."""
    if (
        candidate.metadata.get("source_mesh_used") is not False
        or candidate.metadata.get("teacher_topology_used") is not False
    ):
        raise QualificationError("MWB2_CDT_EXTERNAL_MESH_AUTHORITY_FORBIDDEN")
    if candidate.coverage_classification != "OBSERVATION_DOMAIN_CDT":
        raise QualificationError("MWB2_CDT_COVERAGE_CLASSIFICATION_DRIFT")
    if candidate.solver_provenance.get("historical_cdt_source_sha256") != HISTORICAL_CDT_SOURCE_SHA256:
        raise QualificationError("MWB2_CDT_HISTORICAL_KERNEL_HASH_DRIFT")
    if candidate.solver_provenance.get("quality_refinement_enabled") is not False:
        raise QualificationError("MWB2_CDT_UNBOUND_QUALITY_STEINER_FORBIDDEN")
    recall = float(candidate.residual_report.get("source_alpha_recall", -1.0))
    precision = float(candidate.residual_report.get("precision_inside_alpha", -1.0))
    if precision < float(min_precision_inside_alpha):
        raise QualificationError(f"MWB2_CDT_ALPHA_PRECISION_GATE_FAIL:{precision}")
    if recall < float(min_source_alpha_recall):
        raise QualificationError(f"MWB2_CDT_ALPHA_RECALL_GATE_FAIL:{recall}")

    mesh = qualify_identity_subset_mesh(surface, candidate)
    report = dict(mesh.qualification_report)
    report.update(
        {
            "status": "PASS_OBSERVATION_DOMAIN_CDT_QUALIFICATION",
            "source_alpha_recall": recall,
            "precision_inside_alpha": precision,
            "historical_cdt_source_sha256": HISTORICAL_CDT_SOURCE_SHA256,
            "historical_kernel_role": "NUMERICAL_ONLY",
            "cdt_behavioral_gate_pass": True,
            "numerical_solver_promoted": True,
        }
    )
    updated = replace(
        mesh,
        qualification_report=report,
        metadata={
            **mesh.metadata,
            "cdt_adapter": "RealSaS.MWB2.ObservationDomainCDT.v3",
        },
        mesh_lineage_hash="",
    )
    updated = replace(updated, mesh_lineage_hash=mesh_lineage_hash(updated))
    validate_qualified_mesh(updated, surface)
    return updated
