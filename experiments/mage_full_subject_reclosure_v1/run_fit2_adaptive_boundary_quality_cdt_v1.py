from __future__ import annotations

"""Preregistered FIT2 adaptive boundary + quality Delaunay emitted-mesh diagnostic.

Uniform global barycentric subdivision is falsified. This experiment instead remeshes
only the exact observation-alpha ∩ supported-kernel domain, using deterministic
support-derived raster samples and per-component Delaunay. Every emitted vertex is
bound to admitted S by IDENTITY_SURFACE_NODE or LOCAL_CONVEX_INTERPOLATION.

This is experiment authority only. It never relabels the frozen product GSA lineage.
"""

import argparse
from dataclasses import dataclass, replace
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.spatial import Delaunay, QhullError
from skimage.draw import polygon

from compiler.realsas_compiler_core.mesh.mesh_binding import (
    mesh_candidate_lineage_hash,
    qualify_supported_mesh,
)
from compiler.realsas_compiler_core.mesh.mwb2_cdt import (
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

import experiments.mage_full_subject_reclosure_v1.run_fit2_legal_steiner_ceiling_v1 as ceiling_v1
import experiments.mage_full_subject_reclosure_v1.run_fit2_legal_steiner_ceiling_v2 as ceiling_v2
import experiments.mage_full_subject_reclosure_v1.run_fit2_legal_steiner_uniform_recovery_v1 as uniform_v1


SCHEMA = "RealSaS.MageFIT2.AdaptiveBoundaryQualityDelaunayDiagnostic.v1"
PREREG_ID = "FIT2_ADAPTIVE_BOUNDARY_QUALITY_CDT_PREREG_20260913"
TREATMENTS = (
    {"name": "B4_G16", "boundary_stride": 4, "interior_spacing": 16},
    {"name": "B2_G12", "boundary_stride": 2, "interior_spacing": 12},
    {"name": "B1_G8", "boundary_stride": 1, "interior_spacing": 8},
)
_PARENT_BIN = 32
_BARY_EPS = 1.0e-7
_BINDING_EPS = 1.0e-10
_PARENT_LOCAL_HOPS = 2


@dataclass(frozen=True)
class ParentTriangle:
    parent_index: int
    component_index: int
    surface_ids: tuple[str, str, str]
    xy: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
    bbox: tuple[float, float, float, float]


@dataclass(frozen=True)
class SamplePoint:
    component_index: int
    sample_xy: tuple[float, float]
    raster_xy: tuple[float, float]
    parent_index: int
    binding: SurfaceSupportBinding
    P: tuple[float, float, float]
    sample_role: str


def _barycentric_weights(
    point: tuple[float, float],
    tri: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
) -> tuple[float, float, float] | None:
    """Return normalized barycentric weights when point is inside tri within tolerance."""
    px, py = map(float, point)
    (ax, ay), (bx, by), (cx, cy) = tri
    v0x, v0y = bx - ax, by - ay
    v1x, v1y = cx - ax, cy - ay
    v2x, v2y = px - ax, py - ay
    denom = v0x * v1y - v1x * v0y
    if abs(denom) <= 1.0e-14:
        return None
    wb = (v2x * v1y - v1x * v2y) / denom
    wc = (v0x * v2y - v2x * v0y) / denom
    wa = 1.0 - wb - wc
    raw = [float(wa), float(wb), float(wc)]
    if min(raw) < -_BARY_EPS or max(raw) > 1.0 + _BARY_EPS:
        return None
    clipped = [0.0 if abs(w) <= _BINDING_EPS else max(0.0, w) for w in raw]
    total = float(sum(clipped))
    if total <= 0.0:
        return None
    out = tuple(float(w / total) for w in clipped)
    if min(out) < 0.0 or abs(sum(out) - 1.0) > 1.0e-12:
        return None
    return out


def _binding_from_parent(
    surface_ids: tuple[str, str, str],
    weights: tuple[float, float, float],
    *,
    view: int,
    role: str,
) -> SurfaceSupportBinding:
    acc: dict[str, float] = {}
    for sid, weight in zip(surface_ids, weights):
        if float(weight) <= _BINDING_EPS:
            continue
        acc[str(sid)] = acc.get(str(sid), 0.0) + float(weight)
    total = float(sum(acc.values()))
    if total <= 0.0:
        raise QualificationError("ADAPTIVE_BINDING_EMPTY")
    coeff = tuple(
        (sid, float(value) / total)
        for sid, value in sorted(acc.items())
        if float(value) / total > _BINDING_EPS
    )
    norm = float(sum(w for _, w in coeff))
    if not coeff or abs(norm - 1.0) > 1.0e-9:
        raise QualificationError("ADAPTIVE_BINDING_SIMPLEX_DRIFT")
    if len(coeff) == 1:
        return SurfaceSupportBinding(
            "IDENTITY_SURFACE_NODE",
            ((coeff[0][0], 1.0),),
            metadata={
                "observed_view": int(view),
                "adaptive_boundary_quality_delaunay": True,
                "sample_role": str(role),
            },
        )
    return SurfaceSupportBinding(
        "LOCAL_CONVEX_INTERPOLATION",
        coeff,
        metadata={
            "observed_view": int(view),
            "adaptive_boundary_quality_delaunay": True,
            "sample_role": str(role),
        },
    )


def _derive_point(surface_nodes, surface_raster, binding: SurfaceSupportBinding):
    px = py = pz = rx = ry = 0.0
    for sid, coefficient in binding.coefficients:
        sid = str(sid)
        if sid not in surface_nodes or sid not in surface_raster:
            raise QualificationError(f"ADAPTIVE_SUPPORT_NOT_AVAILABLE:{sid}")
        c = float(coefficient)
        P = surface_nodes[sid].P
        xy = surface_raster[sid]
        px += c * float(P[0])
        py += c * float(P[1])
        pz += c * float(P[2])
        rx += c * float(xy[0])
        ry += c * float(xy[1])
    return (px, py, pz), (rx, ry)


def _quality_admissible(tri_xy) -> bool:
    policy = FIT2_PRODUCT_MESH_QUALITY_POLICY_V1
    area, angle, aspect = _triangle_metrics(*tri_xy)
    return bool(
        area > 1.0e-12
        and float(angle) >= float(policy.min_raster_triangle_angle_deg)
        and float(aspect) <= float(policy.max_raster_triangle_aspect_ratio)
    )


def _surface_raster(surface, view: int):
    nodes = {str(n.surface_id): n for n in surface.surface_nodes}
    raster = {}
    for sid, node in nodes.items():
        rows = [tuple(map(float, xy)) for v, xy in node.raster_bindings if int(v) == int(view)]
        if len(rows) > 1:
            raise QualificationError(f"ADAPTIVE_DUPLICATE_RASTER_BINDING:{view}:{sid}")
        if rows:
            raster[sid] = rows[0]
    return nodes, raster


def _build_parent_triangles(surface, *, view: int):
    kernel = uniform_v1._supported_kernel_id_triangles(surface, view_index=int(view))
    visible = kernel["visible"]
    rows: list[ParentTriangle] = []
    for parent_index, (component_index, tri_ids) in enumerate(kernel["triangles"]):
        xy = tuple(tuple(map(float, visible[sid])) for sid in tri_ids)
        xs = [p[0] for p in xy]
        ys = [p[1] for p in xy]
        rows.append(
            ParentTriangle(
                int(parent_index),
                int(component_index),
                tuple(map(str, tri_ids)),
                xy,
                (min(xs), min(ys), max(xs), max(ys)),
            )
        )
    return kernel, tuple(rows)


def _parent_adjacency(parents: tuple[ParentTriangle, ...]) -> dict[int, set[int]]:
    edge_map: dict[tuple[str, str], list[int]] = {}
    for row in parents:
        a, b, c = row.surface_ids
        for u, v in ((a, b), (b, c), (c, a)):
            edge_map.setdefault(tuple(sorted((u, v))), []).append(row.parent_index)
    neighbors = {row.parent_index: set() for row in parents}
    for ids in edge_map.values():
        for i in ids:
            for j in ids:
                if i != j:
                    neighbors[i].add(j)
    return neighbors


def _hop_neighborhood(neighbors: dict[int, set[int]], hops: int = _PARENT_LOCAL_HOPS):
    out = {}
    for seed in neighbors:
        seen = {seed}
        frontier = {seed}
        for _ in range(int(hops)):
            nxt = set()
            for cur in frontier:
                nxt.update(neighbors.get(cur, ()))
            nxt.difference_update(seen)
            seen.update(nxt)
            frontier = nxt
        out[seed] = seen
    return out


def _spatial_bins(parents: tuple[ParentTriangle, ...]):
    bins: dict[tuple[int, int, int], list[int]] = {}
    by_index = {row.parent_index: row for row in parents}
    for row in parents:
        xmin, ymin, xmax, ymax = row.bbox
        bx0 = int(math.floor(xmin / _PARENT_BIN))
        bx1 = int(math.floor(xmax / _PARENT_BIN))
        by0 = int(math.floor(ymin / _PARENT_BIN))
        by1 = int(math.floor(ymax / _PARENT_BIN))
        for by in range(by0, by1 + 1):
            for bx in range(bx0, bx1 + 1):
                bins.setdefault((row.component_index, bx, by), []).append(row.parent_index)
    for key in bins:
        bins[key].sort()
    return bins, by_index


def _map_point_to_parent(
    point: tuple[float, float],
    *,
    component_index: int,
    bins,
    by_index,
):
    x, y = map(float, point)
    bx = int(math.floor(x / _PARENT_BIN))
    by = int(math.floor(y / _PARENT_BIN))
    candidate_ids = list(bins.get((int(component_index), bx, by), ()))
    if not candidate_ids:
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                candidate_ids.extend(
                    bins.get((int(component_index), bx + dx, by + dy), ())
                )
    candidate_ids = sorted(set(candidate_ids))
    best = None
    best_min = -float("inf")
    for parent_id in candidate_ids:
        parent = by_index[parent_id]
        if parent.component_index != int(component_index):
            continue
        xmin, ymin, xmax, ymax = parent.bbox
        if x < xmin - 1.0e-5 or x > xmax + 1.0e-5 or y < ymin - 1.0e-5 or y > ymax + 1.0e-5:
            continue
        weights = _barycentric_weights((x, y), parent.xy)
        if weights is None:
            continue
        score = min(weights)
        if best is None or score > best_min + 1.0e-15 or (
            abs(score - best_min) <= 1.0e-15 and parent.parent_index < best[0].parent_index
        ):
            best = (parent, weights)
            best_min = score
    return best


def _component_masks(
    parents: tuple[ParentTriangle, ...],
    *,
    shape: tuple[int, int],
):
    masks: dict[int, np.ndarray] = {}
    for row in parents:
        mask = masks.setdefault(row.component_index, np.zeros(shape, dtype=bool))
        rr, cc = polygon(
            [p[1] for p in row.xy],
            [p[0] for p in row.xy],
            shape=shape,
        )
        mask[rr, cc] = True
    return masks


def _boundary_mask(target: np.ndarray) -> np.ndarray:
    structure = np.asarray([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    eroded = ndimage.binary_erosion(target, structure=structure, border_value=0)
    return np.asarray(target & ~eroded, dtype=bool)


def _sample_component_pixels(
    target: np.ndarray,
    *,
    boundary_stride: int,
    interior_spacing: int,
) -> tuple[list[tuple[int, int, str]], dict]:
    boundary = _boundary_mask(target)
    ys, xs = np.nonzero(boundary)
    samples: dict[tuple[int, int], str] = {}

    stride = int(boundary_stride)
    if stride <= 1:
        for y, x in zip(ys.tolist(), xs.tolist()):
            samples[(int(x), int(y))] = "ALPHA_SUPPORT_BOUNDARY"
    else:
        occupied = {}
        for y, x in zip(ys.tolist(), xs.tolist()):
            cell = (int(y) // stride, int(x) // stride)
            occupied.setdefault(cell, (int(x), int(y)))
        for x, y in occupied.values():
            samples[(x, y)] = "ALPHA_SUPPORT_BOUNDARY"

    spacing = int(interior_spacing)
    h, w = target.shape
    offset = spacing // 2
    for y in range(offset, h, spacing):
        for x in range(offset, w, spacing):
            if bool(target[y, x]):
                samples.setdefault((int(x), int(y)), "INTERIOR_GRID")

    labels, count = ndimage.label(
        target,
        structure=np.asarray([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8),
    )
    added_component_seeds = 0
    for label_id in range(1, int(count) + 1):
        yy, xx = np.nonzero(labels == label_id)
        if len(xx) == 0:
            continue
        component_points = [(int(x), int(y)) for y, x in zip(yy.tolist(), xx.tolist())]
        if not any(p in samples for p in component_points):
            mid = component_points[len(component_points) // 2]
            samples[mid] = "TARGET_COMPONENT_SEED"
            added_component_seeds += 1

    rows = [(x, y, role) for (x, y), role in sorted(samples.items(), key=lambda kv: (kv[0][1], kv[0][0]))]
    return rows, {
        "target_pixel_count": int(np.count_nonzero(target)),
        "boundary_pixel_count": int(np.count_nonzero(boundary)),
        "sample_count_requested": len(rows),
        "target_component_count": int(count),
        "added_component_seed_count": int(added_component_seeds),
    }


def _component_triangle_local(parent_ids, hop_sets) -> bool:
    ids = tuple(map(int, parent_ids))
    for a in ids:
        allowed = hop_sets.get(a, {a})
        if any(b not in allowed for b in ids):
            return False
    return True


def _triangle_within_mask(tri_xy, mask: np.ndarray) -> bool:
    rr, cc = polygon(
        [float(p[1]) for p in tri_xy],
        [float(p[0]) for p in tri_xy],
        shape=mask.shape,
    )
    return bool(len(rr)) and bool(np.all(mask[rr, cc]))


def _make_candidate(
    surface,
    *,
    view: int,
    camera_hash: str,
    domain: ObservationRasterDomain,
    treatment: dict,
):
    kernel, parents = _build_parent_triangles(surface, view=int(view))
    if not parents:
        raise QualificationError("ADAPTIVE_NO_SUPPORTED_PARENT_TRIANGLES")
    nodes, surface_raster = _surface_raster(surface, int(view))
    parent_neighbors = _parent_adjacency(parents)
    hop_sets = _hop_neighborhood(parent_neighbors)
    bins, by_index = _spatial_bins(parents)

    authority = np.frombuffer(domain.mask_bytes, dtype=np.uint8).reshape(
        domain.height, domain.width
    ) != 0
    component_masks = _component_masks(parents, shape=authority.shape)

    sample_points: list[SamplePoint] = []
    component_point_indices: dict[int, list[int]] = {}
    sample_diagnostics = {}
    unmappable_total = 0

    for component_index in sorted(component_masks):
        target = authority & component_masks[component_index]
        requested, diag = _sample_component_pixels(
            target,
            boundary_stride=int(treatment["boundary_stride"]),
            interior_spacing=int(treatment["interior_spacing"]),
        )
        mapped = 0
        unmappable = 0
        seen_raster = set()
        for x, y, role in requested:
            hit = _map_point_to_parent(
                (float(x), float(y)),
                component_index=int(component_index),
                bins=bins,
                by_index=by_index,
            )
            if hit is None:
                unmappable += 1
                continue
            parent, weights = hit
            binding = _binding_from_parent(
                parent.surface_ids,
                weights,
                view=int(view),
                role=role,
            )
            P, raster_xy = _derive_point(nodes, surface_raster, binding)
            key = (round(float(raster_xy[0]), 9), round(float(raster_xy[1]), 9))
            if key in seen_raster:
                continue
            seen_raster.add(key)
            point = SamplePoint(
                int(component_index),
                (float(x), float(y)),
                tuple(map(float, raster_xy)),
                int(parent.parent_index),
                binding,
                tuple(map(float, P)),
                str(role),
            )
            idx = len(sample_points)
            sample_points.append(point)
            component_point_indices.setdefault(int(component_index), []).append(idx)
            mapped += 1
        unmappable_total += int(unmappable)
        sample_diagnostics[str(component_index)] = {
            **diag,
            "mapped_sample_count": int(mapped),
            "unmappable_sample_count": int(unmappable),
        }

    accepted_faces: list[tuple[int, int, int]] = []
    rejection = {
        "qhull_component_failures": 0,
        "delaunay_simplex_count": 0,
        "degenerate": 0,
        "parent_locality": 0,
        "quality": 0,
        "support_union": 0,
        "alpha": 0,
    }

    for component_index in sorted(component_point_indices):
        global_ids = component_point_indices[component_index]
        if len(global_ids) < 3:
            continue
        coords = np.asarray([sample_points[i].raster_xy for i in global_ids], dtype=np.float64)
        if np.linalg.matrix_rank(coords - coords.mean(axis=0, keepdims=True)) < 2:
            continue
        try:
            tri = Delaunay(coords, qhull_options="Qbb Qc Qz Q12")
        except QhullError:
            rejection["qhull_component_failures"] += 1
            continue
        component_mask = component_masks[int(component_index)]
        seen_face = set()
        for simplex in np.asarray(tri.simplices, dtype=np.int64):
            if len(simplex) != 3 or np.any(simplex < 0) or np.any(simplex >= len(global_ids)):
                continue
            ids = tuple(global_ids[int(i)] for i in simplex.tolist())
            if len(set(ids)) != 3:
                rejection["degenerate"] += 1
                continue
            pts = tuple(sample_points[i].raster_xy for i in ids)
            area, _, _ = _triangle_metrics(*pts)
            if area <= 1.0e-12:
                rejection["degenerate"] += 1
                continue
            parents_for_face = tuple(sample_points[i].parent_index for i in ids)
            if not _component_triangle_local(parents_for_face, hop_sets):
                rejection["parent_locality"] += 1
                continue
            if not _quality_admissible(pts):
                rejection["quality"] += 1
                continue
            if not _triangle_within_mask(pts, component_mask):
                rejection["support_union"] += 1
                continue
            if not domain.triangle_inside(pts):
                rejection["alpha"] += 1
                continue
            ax, ay = pts[0]
            bx, by = pts[1]
            cx, cy = pts[2]
            area2 = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
            face = ids if area2 > 0.0 else (ids[0], ids[2], ids[1])
            key = tuple(sorted(face))
            if key in seen_face:
                continue
            seen_face.add(key)
            accepted_faces.append(face)
        rejection["delaunay_simplex_count"] += int(len(tri.simplices))

    if not accepted_faces:
        raise QualificationError("ADAPTIVE_NO_ADMITTED_DELAUNAY_FACE")

    used_global = sorted({i for face in accepted_faces for i in face})
    candidate_id = {
        idx: f"MWB2AQD:{view}:{treatment['name']}:{rank:06d}"
        for rank, idx in enumerate(used_global)
    }

    vertices = tuple(
        MeshVertexCandidate(
            candidate_id[idx],
            sample_points[idx].P,
            sample_points[idx].binding,
            metadata={
                "raster_xy": sample_points[idx].raster_xy,
                "sample_xy": sample_points[idx].sample_xy,
                "sample_role": sample_points[idx].sample_role,
                "parent_triangle_index": sample_points[idx].parent_index,
                "component_index": sample_points[idx].component_index,
                "generated_geometry": sample_points[idx].binding.mode
                == "LOCAL_CONVEX_INTERPOLATION",
                "source_mesh_used": False,
            },
        )
        for idx in used_global
    )

    face_ids = tuple(
        tuple(candidate_id[idx] for idx in face)
        for face in accepted_faces
        if all(idx in candidate_id for idx in face)
    )
    edge_set = set()
    for a, b, c in face_ids:
        edge_set.add(tuple(sorted((a, b))))
        edge_set.add(tuple(sorted((b, c))))
        edge_set.add(tuple(sorted((c, a))))
    edges = tuple(sorted(edge_set))

    xy_by_id = {
        candidate_id[idx]: sample_points[idx].raster_xy
        for idx in used_global
    }
    triangles = tuple(
        (
            xy_by_id[str(face[0])],
            xy_by_id[str(face[1])],
            xy_by_id[str(face[2])],
        )
        for face in face_ids
    )
    coverage = domain.coverage(triangles)
    local_count = sum(
        v.support_binding.mode == "LOCAL_CONVEX_INTERPOLATION" for v in vertices
    )

    boundary = (
        {
            "kind": "EXACT_OBSERVATION_ALPHA_DOMAIN",
            "view_index": int(view),
            "mask_sha256": domain.mask_sha256,
            "source_alpha_sha256": domain.source_alpha_sha256,
            "width": int(domain.width),
            "height": int(domain.height),
        },
    )
    residual = {
        **coverage,
        "vertex_count": len(vertices),
        "face_count": len(face_ids),
        "edge_count": len(edges),
        "local_convex_interpolation_vertex_count": int(local_count),
        "treatment": dict(treatment),
        "sample_diagnostics": sample_diagnostics,
        "unmappable_sample_count": int(unmappable_total),
        "rejection_counts": dict(rejection),
        "pre_alpha_kernel_face_count": len(parents),
        "kernel_component_count": int(kernel["component_count"]),
        "kernel_triangle_count": int(kernel["kernel_triangle_count"]),
        "post_contraction_triangle_count": int(kernel["post_contraction_triangle_count"]),
        "contracted_boundary_recovery_vertex_count": int(
            kernel["contracted_boundary_recovery_vertex_count"]
        ),
    }

    provisional = MeshDiscretizationCandidateIR(
        vertices=vertices,
        faces=face_ids,
        edges=edges,
        surface_binding_hash=str(surface.geometry_lineage_hash),
        view_index=int(view),
        camera_binding_hash=str(camera_hash),
        candidate_lineage_hash="",
        boundary_constraints=boundary,
        coverage_classification="OBSERVATION_DOMAIN_CDT_ADAPTIVE_BOUNDARY_QUALITY_DELAUNAY",
        solver_provenance={
            "solver": "CURRENT_CDT_SUPPORT_UNION_PLUS_SCIPY_DELAUNAY_V1",
            "preregistration": PREREG_ID,
            "treatment": dict(treatment),
            "selection_rule": "FIRST_GLOBAL_TREATMENT_FULL_FROZEN_POLICY_PASS",
            "generated_vertex_policy": "LOCAL_CONVEX_INTERPOLATION_ONLY",
            "parent_locality_hops": int(_PARENT_LOCAL_HOPS),
            "per_face_quality_prefilter_is_frozen_policy": True,
            "cross_component_bridge_used": False,
            "unknown_relation_bridge_used": False,
            "source_mesh_used": False,
            "teacher_topology_used": False,
        },
        residual_report=residual,
        metadata={
            "producer": SCHEMA,
            "source_mesh_used": False,
            "teacher_topology_used": False,
            "observed_view": int(view),
            "treatment_name": str(treatment["name"]),
            "surface_lineage_hash": str(surface.geometry_lineage_hash),
            "observation_mask_sha256": domain.mask_sha256,
            "source_alpha_sha256": domain.source_alpha_sha256,
            "target_view_winding": "CCW",
        },
    )
    return replace(
        provisional,
        candidate_lineage_hash=mesh_candidate_lineage_hash(provisional),
    )


def _checker(width: int, height: int, cell: int = 32):
    yy, xx = np.indices((height, width))
    blocks = ((xx // cell) + (yy // cell)) % 2
    rgb = np.where(blocks[..., None] == 0, 54, 72).astype(np.uint8)
    rgb = np.repeat(rgb, 3, axis=2)
    return Image.fromarray(rgb, "RGB").convert("RGBA")


def _mesh_mask(domain: ObservationRasterDomain, mesh) -> np.ndarray:
    xy = {
        str(v.canonical_mesh_vertex_id): tuple(map(float, v.metadata["raster_xy"]))
        for v in mesh.vertices
    }
    predicted = np.zeros((domain.height, domain.width), dtype=bool)
    for face in mesh.faces:
        pts = [xy[str(x)] for x in face]
        rr, cc = polygon(
            [p[1] for p in pts],
            [p[0] for p in pts],
            shape=predicted.shape,
        )
        predicted[rr, cc] = True
    return predicted


def _mesh_only(source: Image.Image, domain: ObservationRasterDomain, mesh):
    predicted = _mesh_mask(domain, mesh)
    rgba = np.asarray(source.convert("RGBA"), dtype=np.uint8).copy()
    rgba[..., 3] = np.where(predicted, rgba[..., 3], 0).astype(np.uint8)
    return Image.alpha_composite(
        _checker(source.width, source.height),
        Image.fromarray(rgba, "RGBA"),
    ).convert("RGB")


def _residual(source: Image.Image, domain: ObservationRasterDomain, mesh, *, view: int):
    predicted = _mesh_mask(domain, mesh)
    authority = np.frombuffer(domain.mask_bytes, dtype=np.uint8).reshape(
        domain.height, domain.width
    ) != 0
    missing = authority & ~predicted
    base = np.asarray(source.convert("RGBA"), dtype=np.uint8).copy()
    base[..., :3] = (base[..., :3].astype(np.float32) * 0.25).astype(np.uint8)
    overlay = np.zeros_like(base)
    overlay[missing] = np.asarray([255, 25, 25, 235], dtype=np.uint8)
    return Image.alpha_composite(
        Image.fromarray(base, "RGBA"),
        Image.fromarray(overlay, "RGBA"),
    ).convert("RGB")


def _label(image: Image.Image, text: str):
    out = image.convert("RGB")
    draw = ImageDraw.Draw(out)
    draw.rectangle((0, 0, out.width, 44), fill=(0, 0, 0))
    draw.text((10, 11), text, fill=(255, 255, 255))
    return out


def _render_treatment(out: Path, treatment_name: str, sources, domains, meshes, rows):
    tiles = []
    names = []
    for view, mesh in enumerate(meshes):
        source = _label(sources[view], f"V{view} SOURCE")
        mesh_only = _label(
            _mesh_only(sources[view], domains[view], mesh),
            f"V{view} {treatment_name} REAL MESH | recall={rows[view]['source_alpha_recall']*100:.2f}% | faces={rows[view]['face_count']}",
        )
        residual = _label(
            _residual(sources[view], domains[view], mesh, view=view),
            f"V{view} UNCOVERED RED | hole={rows[view]['largest_uncovered_component_fraction']*100:.3f}%",
        )
        strip = Image.new("RGB", (3072, 1024))
        strip.paste(source.resize((1024, 1024)), (0, 0))
        strip.paste(mesh_only.resize((1024, 1024)), (1024, 0))
        strip.paste(residual.resize((1024, 1024)), (2048, 0))
        name = f"V{view}_{treatment_name}_SOURCE_MESH_RESIDUAL.png"
        strip.save(out / name)
        names.append(name)
        tiles.append(strip.resize((1536, 512), Image.Resampling.LANCZOS))

    contact = Image.new("RGB", (1536, 4096))
    for view, tile in enumerate(tiles):
        contact.paste(tile, (0, view * 512))
    contact_name = f"FIT2_{treatment_name}_ADAPTIVE_BOUNDARY_QUALITY_8VIEW.png"
    contact.save(out / contact_name)
    return contact_name, names


def run(args):
    surface, tensor, gsa_replay = ceiling_v2._preflight_surface(args)

    sealed_path = Path(__file__).with_name("MWB2_CDT_EXACT_RECLOSURE_REPORT_20260912.json")
    sealed = json.loads(sealed_path.read_text(encoding="utf-8"))
    expected_by_view = {int(row["view"]): row for row in sealed["views"]}

    camera_paths = tuple(Path(p) for p in args.cameras)
    obs_paths = tuple(Path(p) for p in args.observations)
    ceiling_v1._load_cameras(list(camera_paths))
    ceiling_v1._load_observation_masks(obs_paths)

    baseline_rows = []
    domains = []
    sources = []

    for view, obs_path in enumerate(obs_paths):
        with Image.open(obs_path) as im:
            source = im.convert("RGBA")
        authority = np.asarray(source, dtype=np.uint8)[..., 3] >= 8
        domain = ObservationRasterDomain.from_rows(
            authority.tolist(),
            view_index=view,
            source_alpha_sha256=ceiling_v1.OBSERVATION_SHA256[view],
        )
        baseline = build_mwb2_observation_cdt_candidate(
            surface,
            view_index=view,
            camera_binding_hash=ceiling_v1.CAMERA_SHA256[view],
            observation_domain=domain,
        )
        checks = ceiling_v1._exact_baseline_checks(baseline, expected_by_view[view])
        baseline_rows.append(
            {
                "view": int(view),
                "checks": checks,
                "source_alpha_recall": float(
                    baseline.residual_report["source_alpha_recall"]
                ),
                "precision_inside_alpha": float(
                    baseline.residual_report["precision_inside_alpha"]
                ),
                "face_count": len(baseline.faces),
            }
        )
        domains.append(domain)
        sources.append(source)

    print("=" * 160, flush=True)
    print(
        "BASELINE REPLAY: PASS V0..V7 exact sealed structural+raster metrics",
        flush=True,
    )
    print("GSA exact lineage:", bool(gsa_replay["gsa_lineage_exact_match"]), flush=True)
    print("=" * 160, flush=True)

    treatment_rows = []
    selected_name = None
    selected_meshes = None
    selected_rows = None
    last_meshes = None
    last_rows = None
    last_name = None

    for treatment in TREATMENTS:
        name = str(treatment["name"])
        print("=" * 160, flush=True)
        print(
            f"TREATMENT {name} | boundary_stride={treatment['boundary_stride']} "
            f"interior_spacing={treatment['interior_spacing']}",
            flush=True,
        )
        print("=" * 160, flush=True)

        per_view = []
        meshes = []
        for view in range(8):
            candidate = _make_candidate(
                surface,
                view=view,
                camera_hash=ceiling_v1.CAMERA_SHA256[view],
                domain=domains[view],
                treatment=treatment,
            )
            mesh = qualify_supported_mesh(surface, candidate)
            raster_quality = mesh_raster_quality_report(
                mesh,
                surface=surface,
                view_index=view,
            )
            full = evaluate_mesh_quality(
                coverage=dict(candidate.residual_report),
                raster_report=raster_quality,
                policy=FIT2_PRODUCT_MESH_QUALITY_POLICY_V1,
            )
            rejection = dict(candidate.residual_report["rejection_counts"])
            row = {
                "view": int(view),
                "passed": bool(full["passed"]),
                "failure_invariants": list(full["failure_invariants"]),
                "vertex_count": len(mesh.vertices),
                "face_count": len(mesh.faces),
                "edge_count": len(mesh.edges),
                "local_convex_interpolation_vertex_count": int(
                    mesh.qualification_report[
                        "local_convex_interpolation_vertex_count"
                    ]
                ),
                "unmappable_sample_count": int(
                    candidate.residual_report["unmappable_sample_count"]
                ),
                "rejection_counts": rejection,
                "source_alpha_recall": float(full["source_alpha_recall"]),
                "precision_inside_alpha": float(full["precision_inside_alpha"]),
                "alpha_iou": float(full["alpha_iou"]),
                "largest_uncovered_component_fraction": float(
                    full["largest_uncovered_component_fraction"]
                ),
                "min_raster_triangle_angle_deg": float(
                    full["min_raster_triangle_angle_deg"]
                ),
                "max_raster_triangle_aspect_ratio": float(
                    full["max_raster_triangle_aspect_ratio"]
                ),
                "degenerate_faces": int(full["degenerate_faces"]),
                "duplicate_faces": int(full["duplicate_faces"]),
                "nonmanifold_edges": int(full["nonmanifold_edges"]),
                "candidate_lineage_hash": candidate.candidate_lineage_hash,
                "mesh_lineage_hash": mesh.mesh_lineage_hash,
                "sample_diagnostics": candidate.residual_report[
                    "sample_diagnostics"
                ],
            }
            per_view.append(row)
            meshes.append(mesh)
            print(
                f"V{view} {'PASS' if row['passed'] else 'FAIL'} "
                f"verts={row['vertex_count']:6d} faces={row['face_count']:7d} "
                f"local={row['local_convex_interpolation_vertex_count']:6d} "
                f"unmap={row['unmappable_sample_count']:4d} "
                f"recall={row['source_alpha_recall']*100:7.3f}% "
                f"precision={row['precision_inside_alpha']*100:7.3f}% "
                f"IoU={row['alpha_iou']*100:7.3f}% "
                f"hole={row['largest_uncovered_component_fraction']*100:6.3f}% "
                f"angle={row['min_raster_triangle_angle_deg']:.3f} "
                f"aspect={row['max_raster_triangle_aspect_ratio']:.2f} "
                f"reject[q={rejection['quality']},a={rejection['alpha']},"
                f"s={rejection['support_union']},l={rejection['parent_locality']}] "
                f"fail={','.join(row['failure_invariants']) or '-'}",
                flush=True,
            )

        global_pass = all(row["passed"] for row in per_view)
        treatment_rows.append(
            {
                "name": name,
                "boundary_stride": int(treatment["boundary_stride"]),
                "interior_spacing": int(treatment["interior_spacing"]),
                "global_pass": bool(global_pass),
                "views": per_view,
            }
        )
        last_meshes = tuple(meshes)
        last_rows = list(per_view)
        last_name = name
        if global_pass:
            selected_name = name
            selected_meshes = tuple(meshes)
            selected_rows = list(per_view)
            print(
                f"SELECTED {name}: FIRST GLOBAL FULL FROZEN POLICY PASS",
                flush=True,
            )
            break
        print(f"{name}: GLOBAL FAIL; continue per preregistration", flush=True)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    render_name = selected_name if selected_name is not None else last_name
    render_meshes = selected_meshes if selected_meshes is not None else last_meshes
    render_rows = selected_rows if selected_rows is not None else last_rows
    contact_name = ""
    view_pngs = []
    if render_name and render_meshes and render_rows:
        contact_name, view_pngs = _render_treatment(
            out,
            render_name,
            sources,
            domains,
            render_meshes,
            render_rows,
        )

    decision = (
        f"PASS__FIRST_GLOBAL_FULL_FROZEN_MESH_POLICY__{selected_name}"
        if selected_name is not None
        else "FAIL__NO_PREREGISTERED_ADAPTIVE_BOUNDARY_QUALITY_TREATMENT_PASSES_ALL_VIEWS"
    )
    result = {
        "schema": SCHEMA,
        "status": "PASS__EXPERIMENT_COMPLETED",
        "decision": decision,
        "preregistration": PREREG_ID,
        "preregistered_treatments": [dict(row) for row in TREATMENTS],
        "selection_rule": "FIRST_GLOBAL_TREATMENT_FULL_FROZEN_POLICY_PASS",
        "selected_treatment": selected_name,
        "baseline_replay_gate": "PASS__SEALED_STRUCTURAL_AND_RASTER_METRICS_V0_V7",
        "baseline_views": baseline_rows,
        "gsa_replay": gsa_replay,
        "gsa_authority_lineage_preserved_as_expected": ceiling_v2.EXPECTED_GSA_LINEAGE,
        "gsa_lineage_relabelled": False,
        "scope": (
            "FULL_FROZEN_MESH_POLICY_PASS_DIAGNOSTIC__NOT_PRODUCT_PASS"
            if selected_name is not None
            else "DIAGNOSTIC_FAIL__NOT_PRODUCT_PASS"
        ),
        "product_mesh_pass_claimed": False,
        "cross_component_bridge_used": False,
        "unknown_relation_bridge_used": False,
        "source_mesh_used": False,
        "teacher_topology_used": False,
        "uniform_n_2_3_4_family_status": "FALSIFIED",
        "treatments": treatment_rows,
        "rendered_treatment": render_name,
        "contact_sheet_png": contact_name,
        "view_pngs": view_pngs,
    }
    result_path = out / "FIT2_ADAPTIVE_BOUNDARY_QUALITY_CDT_RESULT.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("=" * 160, flush=True)
    print("DECISION:", decision, flush=True)
    print("UNIFORM n={2,3,4}: FALSIFIED", flush=True)
    print("GSA PRODUCT LINEAGE RELABELLED: FALSE", flush=True)
    print("PRODUCT_MESH_PASS: NOT CLAIMED", flush=True)
    print("RESULT:", result_path, flush=True)
    if contact_name:
        print("CONTACT:", out / contact_name, flush=True)
    print("=" * 160, flush=True)
    return result


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--zero-surface", required=True)
    p.add_argument("--cameras", nargs=8, required=True)
    p.add_argument("--observations", nargs=8, required=True)
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
