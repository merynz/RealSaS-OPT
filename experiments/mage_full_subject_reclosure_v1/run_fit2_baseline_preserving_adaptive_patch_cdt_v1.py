from __future__ import annotations

"""Preregistered FIT2 baseline-preserving adaptive patch CDT diagnostic.

The exact sealed CDT replay remains coverage authority. The adaptive support-derived
Delaunay solver is restricted to local quality repair and residual recovery. Every
inserted vertex is derived from admitted S by IDENTITY_SURFACE_NODE or
LOCAL_CONVEX_INTERPOLATION. This experiment never relabels the frozen GSA lineage and
never claims canonical PRODUCT PASS while byte-exact GSA replay is open.
"""

import argparse
from dataclasses import dataclass, replace
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from compiler.realsas_compiler_core.mesh._historical_v05 import triangulate_production_cdt
from compiler.realsas_compiler_core.mesh.mesh_binding import (
    mesh_candidate_lineage_hash,
    qualify_supported_mesh,
)
from compiler.realsas_compiler_core.mesh.mwb2_cdt import (
    _contract_boundary_recovery_vertices,
    build_mwb2_observation_cdt_candidate,
)
from compiler.realsas_compiler_core.mesh.observation_domain import ObservationRasterDomain
from compiler.realsas_compiler_core.mesh.quality import (
    FIT2_PRODUCT_MESH_QUALITY_POLICY_V1,
    _triangle_metrics,
    evaluate_mesh_quality,
    mesh_raster_quality_report,
)
from compiler.realsas_compiler_core.types import (
    MeshDiscretizationCandidateIR,
    MeshVertexCandidate,
    QualificationError,
    SurfaceSupportBinding,
)

import experiments.mage_full_subject_reclosure_v1.run_fit2_adaptive_boundary_quality_cdt_v1 as adaptive_v1
import experiments.mage_full_subject_reclosure_v1.run_fit2_legal_steiner_ceiling_v1 as ceiling_v1
import experiments.mage_full_subject_reclosure_v1.run_fit2_legal_steiner_ceiling_v2 as ceiling_v2


SCHEMA = "RealSaS.MageFIT2.BaselinePreservingAdaptivePatchCDTDiagnostic.v1"
PREREG_ID = "FIT2_BASELINE_PRESERVING_ADAPTIVE_PATCH_CDT_PREREG_20260913"
TREATMENTS = (
    {"name": "P1_B2_G10", "repair_hops": 1, "boundary_stride": 2, "interior_spacing": 10},
    {"name": "P2_B1_G8", "repair_hops": 2, "boundary_stride": 1, "interior_spacing": 8},
    {"name": "P2_B1_G6", "repair_hops": 2, "boundary_stride": 1, "interior_spacing": 6},
)
_EPS = 1.0e-7


@dataclass
class WorkingState:
    vertices: dict[str, MeshVertexCandidate]
    faces: list[tuple[str, str, str]]
    xy_by_id: dict[str, tuple[float, float]]


def _edge_key(a: str, b: str) -> tuple[str, str]:
    return tuple(sorted((str(a), str(b))))


def _face_key(face) -> tuple[str, str, str]:
    return tuple(sorted(map(str, face)))


def _face_edges(face):
    a, b, c = map(str, face)
    return (_edge_key(a, b), _edge_key(b, c), _edge_key(c, a))


def _signed_area2(a, b, c) -> float:
    return (
        (float(b[0]) - float(a[0])) * (float(c[1]) - float(a[1]))
        - (float(b[1]) - float(a[1])) * (float(c[0]) - float(a[0]))
    )


def _oriented_face(face, xy_by_id):
    a, b, c = map(str, face)
    return (a, b, c) if _signed_area2(xy_by_id[a], xy_by_id[b], xy_by_id[c]) > 0.0 else (a, c, b)


def _quality_ok_xy(tri_xy) -> bool:
    p = FIT2_PRODUCT_MESH_QUALITY_POLICY_V1
    area, angle, aspect = _triangle_metrics(*tri_xy)
    return bool(
        float(area) > 1.0e-12
        and float(angle) >= float(p.min_raster_triangle_angle_deg)
        and float(aspect) <= float(p.max_raster_triangle_aspect_ratio)
    )


def _rasterize_faces(faces, xy_by_id, shape: tuple[int, int]) -> np.ndarray:
    from skimage.draw import polygon

    out = np.zeros(shape, dtype=bool)
    for face in faces:
        pts = [xy_by_id[str(v)] for v in face]
        rr, cc = polygon([p[1] for p in pts], [p[0] for p in pts], shape=shape)
        out[rr, cc] = True
    return out


def _mask_triangle(tri_xy, shape: tuple[int, int]) -> np.ndarray:
    from skimage.draw import polygon

    out = np.zeros(shape, dtype=bool)
    rr, cc = polygon([p[1] for p in tri_xy], [p[0] for p in tri_xy], shape=shape)
    out[rr, cc] = True
    return out


def _quality_bad_faces(state: WorkingState) -> set[tuple[str, str, str]]:
    return {
        _face_key(face)
        for face in state.faces
        if not _quality_ok_xy(tuple(state.xy_by_id[str(v)] for v in face))
    }


def _topology_counts(faces) -> dict:
    keys = [_face_key(face) for face in faces]
    duplicate = len(keys) - len(set(keys))
    edge_counts: dict[tuple[str, str], int] = {}
    degenerate = 0
    for face in faces:
        if len(face) != 3 or len(set(face)) != 3:
            degenerate += 1
            continue
        for edge in _face_edges(face):
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
    return {
        "duplicate_faces": int(duplicate),
        "degenerate_faces": int(degenerate),
        "nonmanifold_edges": int(sum(count > 2 for count in edge_counts.values())),
    }


def _face_adjacency(faces):
    keys = {_face_key(face) for face in faces}
    edge_faces: dict[tuple[str, str], list[tuple[str, str, str]]] = {}
    for face in faces:
        key = _face_key(face)
        for edge in _face_edges(face):
            edge_faces.setdefault(edge, []).append(key)
    out = {key: set() for key in keys}
    for rows in edge_faces.values():
        for a in rows:
            for b in rows:
                if a != b:
                    out[a].add(b)
    return out


def _expand_face_set(seeds, adjacency, hops: int):
    seen, frontier = set(seeds), set(seeds)
    for _ in range(int(hops)):
        nxt = set()
        for face in frontier:
            nxt.update(adjacency.get(face, ()))
        nxt.difference_update(seen)
        seen.update(nxt)
        frontier = nxt
    return seen


def _connected_key_clusters(keys, adjacency):
    unseen = set(keys)
    out = []
    while unseen:
        seed = min(unseen)
        unseen.remove(seed)
        stack, comp = [seed], {seed}
        while stack:
            cur = stack.pop()
            for nxt in sorted(adjacency.get(cur, ())):
                if nxt in unseen and nxt in keys:
                    unseen.remove(nxt)
                    comp.add(nxt)
                    stack.append(nxt)
        out.append(comp)
    out.sort(key=lambda comp: min(comp))
    return out


def _boundary_cycle(faces, xy_by_id):
    edge_count: dict[tuple[str, str], int] = {}
    for face in faces:
        for edge in _face_edges(face):
            edge_count[edge] = edge_count.get(edge, 0) + 1
    boundary = [edge for edge, count in edge_count.items() if count == 1]
    if len(boundary) < 3:
        return None
    neighbors: dict[str, list[str]] = {}
    for a, b in boundary:
        neighbors.setdefault(a, []).append(b)
        neighbors.setdefault(b, []).append(a)
    if any(len(rows) != 2 for rows in neighbors.values()):
        return None
    start = min(neighbors)
    cycle, prev, cur = [start], None, start
    while True:
        choices = sorted(v for v in neighbors[cur] if v != prev)
        if not choices:
            return None
        nxt = choices[0]
        if nxt == start:
            break
        if nxt in cycle:
            return None
        cycle.append(nxt)
        prev, cur = cur, nxt
        if len(cycle) > len(neighbors):
            return None
    if len(cycle) != len(neighbors):
        return None
    area2 = 0.0
    for i, vid in enumerate(cycle):
        a, b = xy_by_id[vid], xy_by_id[cycle[(i + 1) % len(cycle)]]
        area2 += float(a[0]) * float(b[1]) - float(b[0]) * float(a[1])
    if area2 < 0.0:
        cycle = [cycle[0]] + list(reversed(cycle[1:]))
    return cycle


def _baseline_state(candidate: MeshDiscretizationCandidateIR) -> WorkingState:
    vertices = {str(v.candidate_vertex_id): v for v in candidate.vertices}
    xy = {str(v.candidate_vertex_id): tuple(map(float, v.metadata["raster_xy"])) for v in candidate.vertices}
    return WorkingState(vertices, [tuple(map(str, f)) for f in candidate.faces], xy)


def _source_sid(vertex: MeshVertexCandidate) -> str:
    coeff = tuple(vertex.support_binding.coefficients)
    if vertex.support_binding.mode != "IDENTITY_SURFACE_NODE" or len(coeff) != 1 or abs(float(coeff[0][1]) - 1.0) > 1.0e-12:
        raise QualificationError("BASELINE_VERTEX_NOT_IDENTITY_BOUND")
    return str(coeff[0][0])


def _parent_maps(surface, *, view: int):
    kernel, parents = adaptive_v1._build_parent_triangles(surface, view=int(view))
    by_key = {_face_key(row.surface_ids): row for row in parents}
    bins, by_index = adaptive_v1._spatial_bins(parents)
    return kernel, parents, by_key, bins, by_index


def _baseline_face_parent_map(state, parent_by_key):
    out = {}
    for face in state.faces:
        key = _face_key(tuple(_source_sid(state.vertices[str(v)]) for v in face))
        if key not in parent_by_key:
            raise QualificationError(f"BASELINE_PARENT_MAP_MISSING:{key}")
        out[_face_key(face)] = parent_by_key[key]
    return out


def _binding_key(binding: SurfaceSupportBinding):
    return tuple((str(sid), round(float(w), 15)) for sid, w in binding.coefficients if float(w) > 1.0e-12)


def _existing_support_map(state):
    out = {}
    for vid, vertex in state.vertices.items():
        out.setdefault(_binding_key(vertex.support_binding), str(vid))
    return out


def _new_vertex(*, candidate_id, binding, P, xy, view, role, parent_index=None):
    return MeshVertexCandidate(
        str(candidate_id), tuple(map(float, P)), binding,
        metadata={
            "raster_xy": tuple(map(float, xy)),
            "generated_geometry": binding.mode == "LOCAL_CONVEX_INTERPOLATION",
            "baseline_preserving_patch": True,
            "patch_role": str(role),
            "parent_triangle_index": parent_index,
            "observed_view": int(view),
            "source_mesh_used": False,
        },
    )


def _sample_target(target, *, component_index, treatment, bins, by_index, nodes, surface_raster, view):
    requested, diag = adaptive_v1._sample_component_pixels(
        target,
        boundary_stride=int(treatment["boundary_stride"]),
        interior_spacing=int(treatment["interior_spacing"]),
    )
    out, seen, unmappable = [], set(), 0
    for x, y, role in requested:
        hit = adaptive_v1._map_point_to_parent(
            (float(x), float(y)), component_index=int(component_index), bins=bins, by_index=by_index
        )
        if hit is None:
            unmappable += 1
            continue
        parent, weights = hit
        binding = adaptive_v1._binding_from_parent(parent.surface_ids, weights, view=int(view), role=str(role))
        P, xy = adaptive_v1._derive_point(nodes, surface_raster, binding)
        key = (round(float(xy[0]), 10), round(float(xy[1]), 10))
        if key in seen:
            continue
        seen.add(key)
        out.append({"binding": binding, "P": tuple(map(float, P)), "xy": tuple(map(float, xy)), "role": str(role), "parent_index": int(parent.parent_index)})
    return out, {**diag, "mapped": len(out), "unmappable": int(unmappable)}


def _sample_parent_target(target, *, parent, treatment, nodes, surface_raster, view):
    requested, diag = adaptive_v1._sample_component_pixels(
        target,
        boundary_stride=int(treatment["boundary_stride"]),
        interior_spacing=int(treatment["interior_spacing"]),
    )
    out, seen, unmappable = [], set(), 0
    for x, y, role in requested:
        weights = adaptive_v1._barycentric_weights((float(x), float(y)), parent.xy)
        if weights is None:
            unmappable += 1
            continue
        binding = adaptive_v1._binding_from_parent(parent.surface_ids, weights, view=int(view), role=str(role))
        P, xy = adaptive_v1._derive_point(nodes, surface_raster, binding)
        key = (round(float(xy[0]), 10), round(float(xy[1]), 10))
        if key in seen:
            continue
        seen.add(key)
        out.append({"binding": binding, "P": tuple(map(float, P)), "xy": tuple(map(float, xy)), "role": str(role), "parent_index": int(parent.parent_index)})
    return out, {**diag, "mapped": len(out), "unmappable": int(unmappable)}


def _match_known_point(point, known_xy, *, eps: float = _EPS):
    px, py = map(float, point)
    best, best_dist = None, float("inf")
    for index, xy in enumerate(known_xy):
        dist = float(np.hypot(px - float(xy[0]), py - float(xy[1])))
        if dist < best_dist:
            best, best_dist = index, dist
    return best if best is not None and best_dist <= eps else None


def _run_local_cdt(boundary_xy, support_xy):
    result = triangulate_production_cdt(
        boundary_xy,
        support_points=support_xy,
        target_min_angle_deg=0.0,
        max_boundary_vertices=max(512, len(boundary_xy) + 64),
        max_support_vertices=max(2048, len(support_xy) + 128),
        max_constraint_recovery_iterations=128,
        max_quality_iterations=0,
        min_feature_spacing=1.0e-8,
    )
    if not bool(result.success):
        return None, f"CDT_FAIL:{result.reason}"
    known_xy = list(boundary_xy) + list(support_xy)
    index_to_known, generated = [], []
    for index, point in enumerate(result.vertices):
        matched = _match_known_point(point, known_xy)
        if matched is None:
            generated.append(int(index))
        index_to_known.append(matched)
    if generated:
        if int(getattr(result, "quality_insert_count", 0)) != 0 or int(getattr(result, "inserted_steiner_count", 0)) != 0:
            return None, "UNBOUND_QUALITY_STEINER"
        if len(generated) != int(result.constraint_split_count):
            return None, "CONSTRAINT_SPLIT_COUNT_MISMATCH"
        try:
            triangles = _contract_boundary_recovery_vertices(result, tuple(generated))
        except QualificationError as exc:
            return None, f"BOUNDARY_CONTRACTION_FAIL:{exc}"
    else:
        triangles = tuple(result.triangles)
    mapped = []
    for tri in triangles:
        ids = tuple(index_to_known[int(i)] for i in tri)
        if any(i is None for i in ids):
            return None, "UNBOUND_VERTEX_SURVIVED"
        if len(set(ids)) == 3:
            mapped.append(tuple(map(int, ids)))
    return mapped, None


def _patch_face_set(state, *, remove_keys, patch_vertices, patch_xy, patch_faces):
    faces = [f for f in state.faces if _face_key(f) not in remove_keys] + list(patch_faces)
    vertices, xy = dict(state.vertices), dict(state.xy_by_id)
    vertices.update(patch_vertices); xy.update(patch_xy)
    used = {str(v) for face in faces for v in face}
    return WorkingState({k: v for k, v in vertices.items() if k in used}, faces, {k: v for k, v in xy.items() if k in used})


def _add_faces(state, *, patch_vertices, patch_xy, patch_faces):
    faces = list(state.faces) + list(patch_faces)
    vertices, xy = dict(state.vertices), dict(state.xy_by_id)
    vertices.update(patch_vertices); xy.update(patch_xy)
    used = {str(v) for face in faces for v in face}
    return WorkingState({k: v for k, v in vertices.items() if k in used}, faces, {k: v for k, v in xy.items() if k in used})


def _build_patch_payload(state, *, boundary_ids, samples, view, treatment_name, stage, patch_index):
    boundary_xy = [state.xy_by_id[vid] for vid in boundary_ids]
    existing = _existing_support_map(state)
    records = []
    for row in samples:
        existing_id = existing.get(_binding_key(row["binding"]))
        if existing_id in boundary_ids:
            continue
        records.append({**row, "existing_id": existing_id})
    unique = []
    seen_xy = {(round(float(p[0]), 10), round(float(p[1]), 10)) for p in boundary_xy}
    for row in records:
        key = (round(float(row["xy"][0]), 10), round(float(row["xy"][1]), 10))
        if key in seen_xy:
            continue
        seen_xy.add(key); unique.append(row)
    records = unique
    tri_known, error = _run_local_cdt(boundary_xy, [row["xy"] for row in records])
    if error is not None:
        return None, error
    known_ids = list(boundary_ids)
    patch_vertices, patch_xy = {}, {}
    for rank, row in enumerate(records):
        vid = row["existing_id"]
        if vid is None:
            vid = f"MWB2BP:{view}:{treatment_name}:{stage}:{patch_index:05d}:{rank:05d}"
            patch_vertices[vid] = _new_vertex(
                candidate_id=vid, binding=row["binding"], P=row["P"], xy=row["xy"], view=view,
                role=f"{stage}:{row['role']}", parent_index=row["parent_index"]
            )
            patch_xy[vid] = tuple(map(float, row["xy"]))
        known_ids.append(str(vid))
    local_xy = {**state.xy_by_id, **patch_xy}
    faces, seen = [], set()
    for tri in tri_known:
        face = tuple(known_ids[int(i)] for i in tri)
        if len(set(face)) != 3:
            continue
        face = _oriented_face(face, local_xy)
        key = _face_key(face)
        if key not in seen:
            seen.add(key); faces.append(face)
    return (patch_vertices, patch_xy, faces), None


def _repair_quality_cavities(state, *, view, domain, treatment, parent_by_key, bins, by_index, nodes, surface_raster, authority):
    initial_bad = _quality_bad_faces(state)
    diagnostics = {
        "initial_bad_face_count": len(initial_bad), "attempted_cavities": 0, "accepted_cavities": 0,
        "rolled_back_cavities": 0, "rollback_reasons": {}, "mapped_samples": 0, "unmappable_samples": 0,
    }
    if not initial_bad:
        diagnostics["final_bad_face_count"] = 0
        return state, diagnostics
    original = {_face_key(face): face for face in state.faces}
    adjacency = _face_adjacency(state.faces)
    expanded = _expand_face_set(initial_bad, adjacency, int(treatment["repair_hops"]))
    clusters = _connected_key_clusters(expanded, adjacency)
    face_parent = _baseline_face_parent_map(state, parent_by_key)

    def rollback(reason):
        diagnostics["rolled_back_cavities"] += 1
        diagnostics["rollback_reasons"][reason] = diagnostics["rollback_reasons"].get(reason, 0) + 1

    current = state
    for patch_index, cluster in enumerate(clusters):
        diagnostics["attempted_cavities"] += 1
        cluster_faces = [original[key] for key in sorted(cluster)]
        cycle = _boundary_cycle(cluster_faces, current.xy_by_id)
        if cycle is None:
            rollback("NON_DISK_CAVITY"); continue
        components = {int(face_parent[key].component_index) for key in cluster}
        if len(components) != 1:
            rollback("MULTI_COMPONENT_CAVITY"); continue
        before_mask = _rasterize_faces(current.faces, current.xy_by_id, authority.shape)
        cavity_mask = _rasterize_faces(cluster_faces, current.xy_by_id, authority.shape)
        samples, sdiag = _sample_target(
            authority & cavity_mask,
            component_index=next(iter(components)), treatment=treatment, bins=bins, by_index=by_index,
            nodes=nodes, surface_raster=surface_raster, view=view,
        )
        diagnostics["mapped_samples"] += int(sdiag["mapped"]); diagnostics["unmappable_samples"] += int(sdiag["unmappable"])
        payload, error = _build_patch_payload(
            current, boundary_ids=cycle, samples=samples, view=view, treatment_name=str(treatment["name"]),
            stage="QUALITY", patch_index=patch_index,
        )
        if error is not None:
            rollback(error); continue
        patch_vertices, patch_xy, candidate_faces = payload
        local_xy = {**current.xy_by_id, **patch_xy}
        seam_edges = {_edge_key(cycle[i], cycle[(i + 1) % len(cycle)]) for i in range(len(cycle))}
        admitted, patch_edges, valid = [], set(), True
        for face in candidate_faces:
            pts = tuple(local_xy[str(v)] for v in face)
            if not _quality_ok_xy(pts) or not adaptive_v1._triangle_within_mask(pts, cavity_mask) or not domain.triangle_inside(pts):
                valid = False; break
            admitted.append(face); patch_edges.update(_face_edges(face))
        if not valid or not admitted:
            rollback("PATCH_FACE_GATE"); continue
        if not seam_edges.issubset(patch_edges):
            rollback("SEAM_EDGE_NOT_PRESERVED"); continue
        proposed = _patch_face_set(
            current, remove_keys=set(cluster), patch_vertices=patch_vertices, patch_xy=patch_xy, patch_faces=admitted
        )
        topo = _topology_counts(proposed.faces)
        if any(topo[k] != 0 for k in ("duplicate_faces", "degenerate_faces", "nonmanifold_edges")):
            rollback("TOPOLOGY_GATE"); continue
        after_mask = _rasterize_faces(proposed.faces, proposed.xy_by_id, authority.shape)
        if np.any(before_mask & ~after_mask):
            rollback("COVERAGE_REGRESSION"); continue
        if np.any(after_mask & ~authority):
            rollback("ALPHA_SPILL"); continue
        current = proposed; diagnostics["accepted_cavities"] += 1
    diagnostics["final_bad_face_count"] = len(_quality_bad_faces(current))
    return current, diagnostics


def _ensure_identity_vertex(state, *, sid, nodes, surface_raster, view, treatment_name, parent_index, corner_rank):
    binding = SurfaceSupportBinding(
        "IDENTITY_SURFACE_NODE", ((str(sid), 1.0),),
        metadata={"observed_view": int(view), "baseline_preserving_patch": True, "patch_role": "RECOVERY_PARENT_CORNER"},
    )
    existing = _existing_support_map(state).get(_binding_key(binding))
    if existing is not None:
        return existing, {}, {}
    P, xy = tuple(map(float, nodes[str(sid)].P)), tuple(map(float, surface_raster[str(sid)]))
    vid = f"MWB2BP:{view}:{treatment_name}:RECOVER:{parent_index:05d}:C{corner_rank}"
    return vid, {vid: _new_vertex(candidate_id=vid, binding=binding, P=P, xy=xy, view=view, role="RECOVERY_PARENT_CORNER", parent_index=parent_index)}, {vid: xy}


def _recover_residual_parents(state, *, view, domain, treatment, parents, baseline_parent_keys, nodes, surface_raster, authority):
    diagnostics = {
        "candidate_parent_count": 0, "attempted_parent_count": 0, "accepted_parent_count": 0,
        "added_face_count": 0, "added_vertex_count": 0, "mapped_samples": 0, "unmappable_samples": 0,
        "rollback_reasons": {},
    }
    def rollback(reason):
        diagnostics["rollback_reasons"][reason] = diagnostics["rollback_reasons"].get(reason, 0) + 1

    current = state
    current_mask = _rasterize_faces(current.faces, current.xy_by_id, authority.shape)
    recovery_parents = [p for p in parents if _face_key(p.surface_ids) not in baseline_parent_keys]
    diagnostics["candidate_parent_count"] = len(recovery_parents)
    for parent in recovery_parents:
        parent_mask = _mask_triangle(parent.xy, authority.shape)
        target = authority & parent_mask & ~current_mask
        if not np.any(target):
            continue
        diagnostics["attempted_parent_count"] += 1
        samples, sdiag = _sample_parent_target(
            target, parent=parent, treatment=treatment, nodes=nodes, surface_raster=surface_raster, view=view
        )
        diagnostics["mapped_samples"] += int(sdiag["mapped"]); diagnostics["unmappable_samples"] += int(sdiag["unmappable"])
        boundary_ids, corner_vertices, corner_xy = [], {}, {}
        temp = WorkingState(dict(current.vertices), list(current.faces), dict(current.xy_by_id))
        for rank, sid in enumerate(parent.surface_ids):
            vid, vv, xx = _ensure_identity_vertex(
                temp, sid=str(sid), nodes=nodes, surface_raster=surface_raster, view=view,
                treatment_name=str(treatment["name"]), parent_index=int(parent.parent_index), corner_rank=rank,
            )
            temp.vertices.update(vv); temp.xy_by_id.update(xx); corner_vertices.update(vv); corner_xy.update(xx); boundary_ids.append(str(vid))
        payload, error = _build_patch_payload(
            temp, boundary_ids=boundary_ids, samples=samples, view=view, treatment_name=str(treatment["name"]),
            stage="RECOVER", patch_index=int(parent.parent_index),
        )
        if error is not None:
            rollback(error); continue
        patch_vertices, patch_xy, candidate_faces = payload
        patch_vertices = {**corner_vertices, **patch_vertices}; patch_xy = {**corner_xy, **patch_xy}
        local_xy = {**current.xy_by_id, **patch_xy}
        admitted = []
        for face in candidate_faces:
            pts = tuple(local_xy[str(v)] for v in face)
            if _quality_ok_xy(pts) and adaptive_v1._triangle_within_mask(pts, parent_mask) and domain.triangle_inside(pts):
                admitted.append(face)
        if not admitted:
            rollback("NO_ADMITTED_RECOVERY_FACE"); continue
        proposed = _add_faces(current, patch_vertices=patch_vertices, patch_xy=patch_xy, patch_faces=admitted)
        topo = _topology_counts(proposed.faces)
        if any(topo[k] != 0 for k in ("duplicate_faces", "degenerate_faces", "nonmanifold_edges")):
            rollback("TOPOLOGY_GATE"); continue
        after_mask = _rasterize_faces(proposed.faces, proposed.xy_by_id, authority.shape)
        if np.any(current_mask & ~after_mask):
            rollback("COVERAGE_REGRESSION"); continue
        if np.any(after_mask & ~authority):
            rollback("ALPHA_SPILL"); continue
        if np.count_nonzero(after_mask & authority) <= np.count_nonzero(current_mask & authority):
            rollback("NO_COVERAGE_GAIN"); continue
        current, current_mask = proposed, after_mask
        diagnostics["accepted_parent_count"] += 1
        diagnostics["added_face_count"] += len(admitted)
        diagnostics["added_vertex_count"] += len(patch_vertices)
    return current, diagnostics


def _candidate_from_state(state, *, surface, view, camera_hash, domain, treatment, baseline_coverage, quality_diag, recovery_diag):
    used = sorted({str(v) for face in state.faces for v in face})
    vertices = tuple(state.vertices[vid] for vid in used)
    faces = tuple(_oriented_face(face, state.xy_by_id) for face in state.faces)
    edge_set = set()
    for face in faces:
        edge_set.update(_face_edges(face))
    edges = tuple(sorted(edge_set))
    triangles = tuple(tuple(state.xy_by_id[str(v)] for v in face) for face in faces)
    coverage = domain.coverage(triangles)
    residual = {
        **coverage,
        "vertex_count": len(vertices), "face_count": len(faces), "edge_count": len(edges),
        "local_convex_interpolation_vertex_count": int(sum(v.support_binding.mode == "LOCAL_CONVEX_INTERPOLATION" for v in vertices)),
        "baseline_source_alpha_recall": float(baseline_coverage["source_alpha_recall"]),
        "baseline_precision_inside_alpha": float(baseline_coverage["precision_inside_alpha"]),
        "baseline_largest_uncovered_component_fraction": float(baseline_coverage["largest_uncovered_component_fraction"]),
        "quality_repair": quality_diag, "residual_recovery": recovery_diag, "treatment": dict(treatment),
    }
    boundary = ({"kind": "EXACT_OBSERVATION_ALPHA_DOMAIN", "view_index": int(view), "mask_sha256": domain.mask_sha256, "source_alpha_sha256": domain.source_alpha_sha256, "width": int(domain.width), "height": int(domain.height)},)
    provisional = MeshDiscretizationCandidateIR(
        vertices=vertices, faces=faces, edges=edges,
        surface_binding_hash=str(surface.geometry_lineage_hash), view_index=int(view), camera_binding_hash=str(camera_hash),
        candidate_lineage_hash="", boundary_constraints=boundary,
        coverage_classification="OBSERVATION_DOMAIN_CDT_BASELINE_PRESERVING_ADAPTIVE_PATCH",
        solver_provenance={
            "solver": "BASELINE_PRESERVING_ADAPTIVE_PATCH_CDT_V1", "preregistration": PREREG_ID,
            "treatment": dict(treatment), "selection_rule": "FIRST_GLOBAL_TREATMENT_FULL_FROZEN_POLICY_PASS",
            "coverage_authority": "SEALED_BASELINE_CDT_EXACT_REPLAY", "quality_repair_operator": "LOCAL_ADAPTIVE_SUPPORT_CDT",
            "residual_recovery_operator": "ALPHA_REJECTED_SUPPORTED_PARENT_LOCAL_CDT",
            "generated_vertex_policy": "IDENTITY_OR_LOCAL_CONVEX_INTERPOLATION_ONLY", "quality_steiner_enabled": False,
            "cross_component_bridge_used": False, "unknown_relation_bridge_used": False,
            "source_mesh_used": False, "teacher_topology_used": False, "monotonic_coverage_required": True,
        },
        residual_report=residual,
        metadata={
            "producer": SCHEMA, "source_mesh_used": False, "teacher_topology_used": False, "observed_view": int(view),
            "surface_lineage_hash": str(surface.geometry_lineage_hash), "observation_mask_sha256": domain.mask_sha256,
            "source_alpha_sha256": domain.source_alpha_sha256, "target_view_winding": "CCW", "baseline_preserved": True,
        },
    )
    return replace(provisional, candidate_lineage_hash=mesh_candidate_lineage_hash(provisional))


def _build_hybrid_candidate(surface, baseline, *, view, camera_hash, domain, treatment):
    authority = np.frombuffer(domain.mask_bytes, dtype=np.uint8).reshape(domain.height, domain.width) != 0
    state = _baseline_state(baseline)
    baseline_mask = _rasterize_faces(state.faces, state.xy_by_id, authority.shape)
    baseline_coverage = dict(baseline.residual_report)
    _, parents, parent_by_key, bins, by_index = _parent_maps(surface, view=int(view))
    original_face_parent = _baseline_face_parent_map(state, parent_by_key)
    baseline_parent_keys = {_face_key(parent.surface_ids) for parent in original_face_parent.values()}
    nodes, surface_raster = adaptive_v1._surface_raster(surface, int(view))
    state, qdiag = _repair_quality_cavities(
        state, view=int(view), domain=domain, treatment=treatment, parent_by_key=parent_by_key,
        bins=bins, by_index=by_index, nodes=nodes, surface_raster=surface_raster, authority=authority,
    )
    state, rdiag = _recover_residual_parents(
        state, view=int(view), domain=domain, treatment=treatment, parents=parents,
        baseline_parent_keys=baseline_parent_keys, nodes=nodes, surface_raster=surface_raster, authority=authority,
    )
    final_mask = _rasterize_faces(state.faces, state.xy_by_id, authority.shape)
    if np.any(baseline_mask & ~final_mask):
        raise QualificationError("BASELINE_COVERAGE_MONOTONICITY_VIOLATION")
    if np.any(final_mask & ~authority):
        raise QualificationError("BASELINE_PRESERVING_ALPHA_SPILL")
    topo = _topology_counts(state.faces)
    if any(topo[k] != 0 for k in ("duplicate_faces", "degenerate_faces", "nonmanifold_edges")):
        raise QualificationError(f"BASELINE_PRESERVING_TOPOLOGY_FAIL:{topo}")
    candidate = _candidate_from_state(
        state, surface=surface, view=int(view), camera_hash=str(camera_hash), domain=domain, treatment=treatment,
        baseline_coverage=baseline_coverage, quality_diag=qdiag, recovery_diag=rdiag,
    )
    if float(candidate.residual_report["source_alpha_recall"]) + 1e-12 < float(baseline_coverage["source_alpha_recall"]):
        raise QualificationError("BASELINE_RECALL_REGRESSION")
    if float(candidate.residual_report["precision_inside_alpha"]) + 1e-12 < float(baseline_coverage["precision_inside_alpha"]):
        raise QualificationError("BASELINE_PRECISION_REGRESSION")
    return candidate


def _checker(width, height, cell=32):
    yy, xx = np.indices((height, width)); blocks = ((xx // cell) + (yy // cell)) % 2
    rgb = np.repeat(np.where(blocks[..., None] == 0, 54, 72).astype(np.uint8), 3, axis=2)
    return Image.fromarray(rgb).convert("RGBA")


def _mesh_mask(domain, mesh):
    from skimage.draw import polygon
    xy = {str(v.canonical_mesh_vertex_id): tuple(map(float, v.metadata["raster_xy"])) for v in mesh.vertices}
    out = np.zeros((domain.height, domain.width), dtype=bool)
    for face in mesh.faces:
        pts = [xy[str(x)] for x in face]
        rr, cc = polygon([p[1] for p in pts], [p[0] for p in pts], shape=out.shape); out[rr, cc] = True
    return out


def _mesh_only(source, domain, mesh):
    predicted = _mesh_mask(domain, mesh); rgba = np.asarray(source.convert("RGBA"), dtype=np.uint8).copy()
    rgba[..., 3] = np.where(predicted, rgba[..., 3], 0).astype(np.uint8)
    return Image.alpha_composite(_checker(source.width, source.height), Image.fromarray(rgba)).convert("RGB")


def _residual(source, domain, mesh):
    predicted = _mesh_mask(domain, mesh)
    authority = np.frombuffer(domain.mask_bytes, dtype=np.uint8).reshape(domain.height, domain.width) != 0
    missing = authority & ~predicted
    base = np.asarray(source.convert("RGBA"), dtype=np.uint8).copy(); base[..., :3] = (base[..., :3].astype(np.float32) * 0.25).astype(np.uint8)
    overlay = np.zeros_like(base); overlay[missing] = np.asarray([255, 25, 25, 235], dtype=np.uint8)
    return Image.alpha_composite(Image.fromarray(base), Image.fromarray(overlay)).convert("RGB")


def _label(image, text):
    out = image.convert("RGB"); draw = ImageDraw.Draw(out); draw.rectangle((0, 0, out.width, 44), fill=(0, 0, 0)); draw.text((10, 11), text, fill=(255, 255, 255)); return out


def _render(out, treatment_name, sources, domains, meshes, rows):
    tiles, names = [], []
    for view, mesh in enumerate(meshes):
        source = _label(sources[view], f"V{view} SOURCE")
        mesh_only = _label(_mesh_only(sources[view], domains[view], mesh), f"V{view} HYBRID REAL MESH | recall={rows[view]['source_alpha_recall']*100:.2f}% | faces={rows[view]['face_count']}")
        residual = _label(_residual(sources[view], domains[view], mesh), f"V{view} UNCOVERED RED | hole={rows[view]['largest_uncovered_component_fraction']*100:.3f}%")
        strip = Image.new("RGB", (3072, 1024)); strip.paste(source.resize((1024, 1024)), (0, 0)); strip.paste(mesh_only.resize((1024, 1024)), (1024, 0)); strip.paste(residual.resize((1024, 1024)), (2048, 0))
        name = f"V{view}_{treatment_name}_SOURCE_HYBRID_RESIDUAL.png"; strip.save(out / name); names.append(name); tiles.append(strip.resize((1536, 512), Image.Resampling.LANCZOS))
    contact = Image.new("RGB", (1536, 4096))
    for view, tile in enumerate(tiles): contact.paste(tile, (0, view * 512))
    contact_name = f"FIT2_{treatment_name}_BASELINE_PRESERVING_PATCH_8VIEW.png"; contact.save(out / contact_name)
    return contact_name, names


def run(args):
    surface, _tensor, gsa_replay = ceiling_v2._preflight_surface(args)
    sealed = json.loads(Path(__file__).with_name("MWB2_CDT_EXACT_RECLOSURE_REPORT_20260912.json").read_text(encoding="utf-8"))
    expected_by_view = {int(row["view"]): row for row in sealed["views"]}
    obs_paths = tuple(Path(p) for p in args.observations)
    ceiling_v1._load_cameras([Path(p) for p in args.cameras]); ceiling_v1._load_observation_masks(obs_paths)
    baselines, baseline_rows, domains, sources = [], [], [], []
    for view, obs_path in enumerate(obs_paths):
        with Image.open(obs_path) as im: source = im.convert("RGBA")
        authority = np.asarray(source, dtype=np.uint8)[..., 3] >= 8
        domain = ObservationRasterDomain.from_rows(authority.tolist(), view_index=view, source_alpha_sha256=ceiling_v1.OBSERVATION_SHA256[view])
        baseline = build_mwb2_observation_cdt_candidate(surface, view_index=view, camera_binding_hash=ceiling_v1.CAMERA_SHA256[view], observation_domain=domain)
        checks = ceiling_v1._exact_baseline_checks(baseline, expected_by_view[view])
        baseline_mesh = qualify_supported_mesh(surface, baseline, allowed_binding_modes=("IDENTITY_SURFACE_NODE",))
        baseline_full = evaluate_mesh_quality(
            coverage=dict(baseline.residual_report), raster_report=mesh_raster_quality_report(baseline_mesh, surface=surface, view_index=view), policy=FIT2_PRODUCT_MESH_QUALITY_POLICY_V1
        )
        baseline_rows.append({
            "view": view, "checks": checks, "source_alpha_recall": float(baseline.residual_report["source_alpha_recall"]),
            "precision_inside_alpha": float(baseline.residual_report["precision_inside_alpha"]), "alpha_iou": float(baseline.residual_report["alpha_iou"]),
            "largest_uncovered_component_fraction": float(baseline.residual_report["largest_uncovered_component_fraction"]), "face_count": len(baseline.faces),
            "strict_failure_invariants": list(baseline_full["failure_invariants"]), "min_raster_triangle_angle_deg": float(baseline_full["min_raster_triangle_angle_deg"]),
            "max_raster_triangle_aspect_ratio": float(baseline_full["max_raster_triangle_aspect_ratio"]),
        })
        baselines.append(baseline); domains.append(domain); sources.append(source)
    print("=" * 180, flush=True); print("BASELINE REPLAY: PASS V0..V7 exact sealed structural+raster metrics", flush=True); print("GSA exact lineage:", bool(gsa_replay["gsa_lineage_exact_match"]), flush=True); print("=" * 180, flush=True)

    treatment_rows, selected_name, selected_meshes, selected_rows = [], None, None, None
    last_name, last_meshes, last_rows = None, None, None
    for treatment in TREATMENTS:
        name = str(treatment["name"]); print("=" * 180, flush=True); print(f"TREATMENT {name} | repair_hops={treatment['repair_hops']} boundary_stride={treatment['boundary_stride']} interior_spacing={treatment['interior_spacing']}", flush=True); print("=" * 180, flush=True)
        rows, meshes = [], []
        for view in range(8):
            candidate = _build_hybrid_candidate(surface, baselines[view], view=view, camera_hash=ceiling_v1.CAMERA_SHA256[view], domain=domains[view], treatment=treatment)
            mesh = qualify_supported_mesh(surface, candidate)
            full = evaluate_mesh_quality(coverage=dict(candidate.residual_report), raster_report=mesh_raster_quality_report(mesh, surface=surface, view_index=view), policy=FIT2_PRODUCT_MESH_QUALITY_POLICY_V1)
            qdiag, rdiag = candidate.residual_report["quality_repair"], candidate.residual_report["residual_recovery"]
            row = {
                "view": view, "passed": bool(full["passed"]), "failure_invariants": list(full["failure_invariants"]),
                "vertex_count": len(mesh.vertices), "face_count": len(mesh.faces), "edge_count": len(mesh.edges),
                "local_convex_interpolation_vertex_count": int(mesh.qualification_report["local_convex_interpolation_vertex_count"]),
                "source_alpha_recall": float(full["source_alpha_recall"]), "precision_inside_alpha": float(full["precision_inside_alpha"]), "alpha_iou": float(full["alpha_iou"]),
                "largest_uncovered_component_fraction": float(full["largest_uncovered_component_fraction"]), "min_raster_triangle_angle_deg": float(full["min_raster_triangle_angle_deg"]),
                "max_raster_triangle_aspect_ratio": float(full["max_raster_triangle_aspect_ratio"]), "degenerate_faces": int(full["degenerate_faces"]),
                "duplicate_faces": int(full["duplicate_faces"]), "nonmanifold_edges": int(full["nonmanifold_edges"]),
                "candidate_lineage_hash": candidate.candidate_lineage_hash, "mesh_lineage_hash": mesh.mesh_lineage_hash,
                "baseline_source_alpha_recall": float(candidate.residual_report["baseline_source_alpha_recall"]), "quality_repair": qdiag, "residual_recovery": rdiag,
            }
            rows.append(row); meshes.append(mesh)
            print(f"V{view} {'PASS' if row['passed'] else 'FAIL'} baseline={row['baseline_source_alpha_recall']*100:7.3f}% final={row['source_alpha_recall']*100:7.3f}% precision={row['precision_inside_alpha']*100:7.3f}% IoU={row['alpha_iou']*100:7.3f}% hole={row['largest_uncovered_component_fraction']*100:6.3f}% angle={row['min_raster_triangle_angle_deg']:.3f} aspect={row['max_raster_triangle_aspect_ratio']:.2f} faces={row['face_count']:6d} Qpatch={qdiag['accepted_cavities']}/{qdiag['attempted_cavities']} Rpatch={rdiag['accepted_parent_count']}/{rdiag['attempted_parent_count']} bad={qdiag['initial_bad_face_count']}->{qdiag['final_bad_face_count']} fail={','.join(row['failure_invariants']) or '-'}", flush=True)
        global_pass = all(row["passed"] for row in rows)
        treatment_rows.append({**dict(treatment), "global_pass": bool(global_pass), "views": rows}); last_name, last_meshes, last_rows = name, tuple(meshes), list(rows)
        if global_pass:
            selected_name, selected_meshes, selected_rows = name, tuple(meshes), list(rows); print(f"SELECTED {name}: FIRST GLOBAL FULL FROZEN POLICY PASS", flush=True); break
        print(f"{name}: GLOBAL FAIL; continue per preregistration", flush=True)

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    render_name = selected_name if selected_name is not None else last_name; render_meshes = selected_meshes if selected_meshes is not None else last_meshes; render_rows = selected_rows if selected_rows is not None else last_rows
    contact_name, view_pngs = ("", [])
    if render_name and render_meshes and render_rows: contact_name, view_pngs = _render(out, render_name, sources, domains, render_meshes, render_rows)
    decision = f"PASS__FIRST_GLOBAL_FULL_FROZEN_MESH_POLICY__{selected_name}" if selected_name is not None else "FAIL__NO_PREREGISTERED_BASELINE_PRESERVING_PATCH_TREATMENT_PASSES_ALL_VIEWS"
    result = {
        "schema": SCHEMA, "status": "PASS__EXPERIMENT_COMPLETED", "decision": decision, "preregistration": PREREG_ID,
        "preregistered_treatments": [dict(row) for row in TREATMENTS], "selection_rule": "FIRST_GLOBAL_TREATMENT_FULL_FROZEN_POLICY_PASS",
        "baseline_replay_gate": "PASS__SEALED_STRUCTURAL_AND_RASTER_METRICS_V0_V7", "baseline_views": baseline_rows, "gsa_replay": gsa_replay,
        "gsa_authority_lineage_preserved_as_expected": ceiling_v2.EXPECTED_GSA_LINEAGE, "gsa_lineage_relabelled": False,
        "uniform_n_2_3_4_family_status": "FALSIFIED", "global_adaptive_reconstruction_family_status": "FALSIFIED_AS_GLOBAL_FINAL_SOLVER",
        "adaptive_local_patch_operator_status": "ACTIVE_DIAGNOSTIC", "coverage_authority": "SEALED_BASELINE_CDT_EXACT_REPLAY",
        "quality_repair_authority": "LOCAL_ADAPTIVE_SUPPORT_CDT", "monotonic_coverage_required": True,
        "cross_component_bridge_used": False, "unknown_relation_bridge_used": False, "source_mesh_used": False, "teacher_topology_used": False,
        "product_mesh_pass_claimed": False, "scope": "FULL_FROZEN_MESH_POLICY_PASS_DIAGNOSTIC__NOT_PRODUCT_PASS" if selected_name is not None else "DIAGNOSTIC_FAIL__NOT_PRODUCT_PASS",
        "selected_treatment": selected_name, "treatments": treatment_rows, "rendered_treatment": render_name, "contact_sheet_png": contact_name, "view_pngs": view_pngs,
    }
    result_path = out / "FIT2_BASELINE_PRESERVING_ADAPTIVE_PATCH_CDT_RESULT.json"; result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("=" * 180, flush=True); print("DECISION:", decision, flush=True); print("BASELINE COVERAGE AUTHORITY: PRESERVED", flush=True); print("GLOBAL ADAPTIVE RECONSTRUCTION: FALSIFIED", flush=True); print("ADAPTIVE LOCAL PATCH OPERATOR: ACTIVE", flush=True); print("GSA PRODUCT LINEAGE RELABELLED: FALSE", flush=True); print("PRODUCT_MESH_PASS: NOT CLAIMED", flush=True); print("RESULT:", result_path, flush=True)
    if contact_name: print("CONTACT:", out / contact_name, flush=True)
    print("=" * 180, flush=True)
    return result


def parse_args():
    p = argparse.ArgumentParser(); p.add_argument("--zero-surface", required=True); p.add_argument("--cameras", nargs=8, required=True); p.add_argument("--observations", nargs=8, required=True); p.add_argument("--output-dir", required=True); return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
