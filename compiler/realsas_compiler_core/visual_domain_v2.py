"""Compile one continuous, source-owned deformation field per connected chart.

Safe canonical surface samples constrain a positive graph Laplacian. Remaining
vertices share the harmonic extension, rather than independently extrapolating
unrelated mechanical faces. The same frozen operator transports camera depth.
Nothing here modifies the mechanical mesh, skeleton, weights or motion.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree

from .camera_geometry_v2 import project_points_xyz_v3
from .types import QualificationError

OPERATOR_ID = "SOURCE_CHART_HARMONIC_CANONICAL_FIELD_V2"
DEPTH_CONTRACT = "CANONICAL_CAMERA_DEPTH_ASCENDING__UNRESOLVED_TIES_FAIL_V1"
POLICY = {
    "schema": "RealSaS.VisualDeformationOperatorPolicy.v2",
    "operator_id": OPERATOR_ID,
    "anchor_grid_px": 32,
    "anchor_barycentric_extrapolation_authorized": False,
    "connected_chart_harmonic_extension": True,
    "unseeded_island_support": "NEAREST_SAFE_SAMPLE_IN_SAME_QUALIFIED_REGION",
    "maximum_unseeded_island_support_distance_px": 64,
    "rest_source_coordinates_preserved": True,
    "depth_ownership_contract": DEPTH_CONTRACT,
    "depth_tie_epsilon": 1e-9,
    "maximum_fragment_layers": 32,
    "minimum_signed_area_ratio": 0.05,
    "maximum_jacobian_condition": 16.0,
    "runtime_binding_solve_authorized": False,
    "dynamic_quality_authority": "STAGE45",
}


def chart_graph(points, faces, regions):
    p = np.asarray(points, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    r = np.asarray(regions, dtype=np.int32)
    if (p.ndim != 2 or p.shape[1] != 2 or not np.isfinite(p).all()
            or f.ndim != 2 or f.shape[1] != 3 or not len(f)
            or np.any(f < 0) or np.any(f >= len(p)) or r.shape != (len(p),)
            or np.any(r < 0) or np.any(r[f] != r[f[:, :1]])):
        raise QualificationError("VISUAL_DOMAIN_TOPOLOGY_INVALID")
    edges = np.unique(np.sort(np.concatenate((f[:, [0, 1]], f[:, [1, 2]],
                                              f[:, [2, 0]])), axis=1), axis=0)
    lengths = np.linalg.norm(p[edges[:, 0]] - p[edges[:, 1]], axis=1)
    if np.any(lengths <= 1e-12):
        raise QualificationError("VISUAL_DOMAIN_ZERO_REST_EDGE")
    weights = 1.0 / lengths
    graph = coo_matrix((np.r_[weights, weights],
                        (np.r_[edges[:, 0], edges[:, 1]],
                         np.r_[edges[:, 1], edges[:, 0]])),
                       shape=(len(p), len(p))).tocsr()
    count, domains = connected_components(graph, directed=False)
    if np.any(np.asarray(graph.sum(axis=1)).ravel() == 0):
        raise QualificationError("VISUAL_DOMAIN_UNUSED_VERTEX")
    return graph, domains.astype(np.int32), count


def build_domain_binding(*, points_source_xy, visual_faces, vertex_region_id,
                         seed_region_labels, owner_face_index,
                         mechanical_positions_xyz, mechanical_faces, camera):
    p = np.asarray(points_source_xy, dtype=np.float64)
    regions = np.asarray(vertex_region_id, dtype=np.int32)
    graph, domains, count = chart_graph(p, visual_faces, regions)
    seeds = np.asarray(seed_region_labels, dtype=np.int32)
    owners = np.asarray(owner_face_index, dtype=np.int64)
    mf = np.asarray(mechanical_faces, dtype=np.int64)
    xyz = np.asarray(mechanical_positions_xyz, dtype=np.float64)
    if seeds.shape != owners.shape or seeds.ndim != 2:
        raise QualificationError("VISUAL_DOMAIN_SEED_SHAPE_INVALID")
    projected = project_points_xyz_v3(xyz, camera)
    anchors = []
    ancestry = []
    barycentric = []
    distances = []
    for region in np.unique(regions):
        vertices = np.flatnonzero(regions == region)
        ys, xs = np.nonzero((seeds == region) & (owners >= 0) & (owners < len(mf)))
        if not len(xs):
            raise QualificationError("VISUAL_DOMAIN_SAFE_SEED_EMPTY:" + str(region))
        # Camera coordinates address pixel centres; source texel coordinates
        # address integer indices. Never evaluate a barycentric row outside M.
        sample_xy = np.column_stack((xs + 0.5, ys + 0.5))
        source_xy = sample_xy - 0.5
        _, nearest = cKDTree(p[vertices]).query(source_xy)
        sample_domain = domains[vertices[nearest]]
        for domain in np.unique(domains[vertices]):
            local_vertices = vertices[domains[vertices] == domain]
            candidates = np.flatnonzero(sample_domain == domain)
            if not len(candidates):
                # A source-alpha island can share a qualified canonical region
                # while its nearest seeds were assigned to the larger island.
                # Transport displacement from an actual safe sample; keep the
                # island's source rest offset and never extrapolate on M.
                distance, nearest_sample = cKDTree(source_xy).query(p[local_vertices])
                nearest_vertex = int(np.argmin(distance))
                if float(distance[nearest_vertex]) > POLICY["maximum_unseeded_island_support_distance_px"]:
                    raise QualificationError("VISUAL_DOMAIN_UNANCHORED_COMPONENT:" + str(domain))
                candidates = np.asarray([nearest_sample[nearest_vertex]], dtype=int)
            # One canonical sample per occupied source grid cell. Resolve
            # multiple samples mapped to a vertex by spatial distance only.
            bins = np.floor(source_xy[candidates] / POLICY["anchor_grid_px"]).astype(int)
            selected = {}
            for ci, cell in zip(candidates, bins):
                centre = (cell + 0.5) * POLICY["anchor_grid_px"]
                key = tuple(cell)
                rank = (float(np.linalg.norm(source_xy[ci] - centre)), int(ci))
                if key not in selected or rank < selected[key][0]:
                    selected[key] = (rank, int(ci))
            chosen = np.array([selected[k][1] for k in sorted(selected)], dtype=int)
            d, near = cKDTree(p[local_vertices]).query(source_xy[chosen])
            vertex_samples = {}
            for ci, distance, ni in zip(chosen, d, near):
                vertex = int(local_vertices[ni])
                rank = (float(distance), int(ci))
                if vertex not in vertex_samples or rank < vertex_samples[vertex]:
                    vertex_samples[vertex] = rank
            for vertex in sorted(vertex_samples):
                distance, ci = vertex_samples[vertex]
                indices = mf[owners[ys[ci], xs[ci]]]
                triangle = projected[indices, :2]
                matrix = (triangle[1:] - triangle[0]).T
                if abs(np.linalg.det(matrix)) <= 1e-12:
                    raise QualificationError("VISUAL_DOMAIN_DEGENERATE_CANONICAL_ANCHOR")
                q = np.linalg.solve(matrix, sample_xy[ci] - triangle[0])
                bary = np.r_[1.0 - q.sum(), q]
                if np.min(bary) < -1e-7 or np.max(bary) > 1 + 1e-7:
                    raise QualificationError("VISUAL_DOMAIN_ANCHOR_OUTSIDE_CANONICAL_FACE")
                bary = np.clip(bary, 0, 1)
                bary /= bary.sum()
                anchors.append(vertex)
                ancestry.append(indices)
                barycentric.append(bary)
                distances.append(distance)
    binding = {
        "rest_positions": p,
        "vertex_region_id": regions,
        "domain_id": domains,
        "anchor_vertex": np.asarray(anchors, dtype=np.int64),
        "anchor_mechanical_vertices": np.asarray(ancestry, dtype=np.int64),
        "anchor_barycentric": np.asarray(barycentric, dtype=np.float64),
        "anchor_source_distance_px": np.asarray(distances, dtype=np.float64),
        "mechanical_rest_xyz": xyz,
    }
    validate_domain_binding(binding, visual_faces)
    return binding


def validate_domain_binding(binding, visual_faces):
    graph, domains, count = chart_graph(binding["rest_positions"], visual_faces,
                                       binding["vertex_region_id"])
    a = np.asarray(binding["anchor_vertex"], dtype=np.int64)
    ids = np.asarray(binding["anchor_mechanical_vertices"], dtype=np.int64)
    b = np.asarray(binding["anchor_barycentric"], dtype=np.float64)
    xyz = np.asarray(binding["mechanical_rest_xyz"], dtype=np.float64)
    if (not np.array_equal(domains, binding["domain_id"])
            or not len(a) or len(np.unique(a)) != len(a)
            or np.any(a < 0) or np.any(a >= len(domains))
            or ids.shape != (len(a), 3) or b.shape != ids.shape
            or xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all()
            or np.any(ids < 0) or np.any(ids >= len(xyz))
            or not np.isfinite(b).all() or np.any(b < 0) or np.any(b > 1)
            or not np.allclose(b.sum(axis=1), 1, atol=1e-12, rtol=0)
            or set(domains[a]) != set(range(count))):
        raise QualificationError("VISUAL_DOMAIN_OPERATOR_INVALID")
    return graph


def evaluate_domain_binding(binding, *, visual_faces, posed_mechanical_positions_xyz,
                            camera):
    graph = validate_domain_binding(binding, visual_faces)
    rest = np.asarray(binding["rest_positions"], dtype=np.float64)
    a = np.asarray(binding["anchor_vertex"], dtype=np.int64)
    ids = np.asarray(binding["anchor_mechanical_vertices"], dtype=np.int64)
    b = np.asarray(binding["anchor_barycentric"], dtype=np.float64)
    xyz = np.asarray(posed_mechanical_positions_xyz, dtype=np.float64)
    if xyz.shape != np.asarray(binding["mechanical_rest_xyz"]).shape or not np.isfinite(xyz).all():
        raise QualificationError("VISUAL_DOMAIN_MOTION_WITNESS_INVALID")
    rest_camera = project_points_xyz_v3(binding["mechanical_rest_xyz"], camera)
    posed_camera = project_points_xyz_v3(xyz, camera)
    reference = np.sum(rest_camera[ids] * b[:, :, None], axis=1)
    sample = np.sum(posed_camera[ids] * b[:, :, None], axis=1)
    sample[:, :2] -= reference[:, :2]
    field = np.zeros((len(rest), 3), dtype=np.float64)
    field[a] = sample
    free = np.setdiff1d(np.arange(len(rest)), a)
    laplacian = diags(np.asarray(graph.sum(axis=1)).ravel()) - graph
    if len(free):
        field[free] = spsolve(laplacian[free][:, free].tocsc(),
                             -(laplacian[free][:, a] @ sample))
    field[:, :2] += rest
    if not np.isfinite(field).all():
        raise QualificationError("VISUAL_DOMAIN_FIELD_NONFINITE")
    return field


def domain_binding_from_arrays(arrays, view_index):
    prefix = f"view_{view_index}_domain_"
    return {name[len(prefix):]: np.asarray(value) for name, value in arrays.items()
            if name.startswith(prefix)}


def presentation_condition_metrics(rest, posed, faces):
    r, p, f = np.asarray(rest), np.asarray(posed), np.asarray(faces, dtype=np.int64)
    if r.shape != p.shape or r.ndim != 2 or r.shape[1] != 2:
        raise QualificationError("VISUAL_PRESENTATION_METRIC_SHAPE_INVALID")
    if not np.isfinite(r).all() or not np.isfinite(p).all():
        raise QualificationError("VISUAL_PRESENTATION_NONFINITE")
    rb = np.stack((r[f[:, 1]] - r[f[:, 0]], r[f[:, 2]] - r[f[:, 0]]), axis=2)
    pb = np.stack((p[f[:, 1]] - p[f[:, 0]], p[f[:, 2]] - p[f[:, 0]]), axis=2)
    if np.any(np.abs(np.linalg.det(rb)) <= 1e-12):
        raise QualificationError("VISUAL_PRESENTATION_REST_DEGENERATE")
    jac = pb @ np.linalg.inv(rb)
    area = np.linalg.det(jac)
    singular = np.linalg.svd(jac, compute_uv=False)
    condition = np.divide(singular[:, 0], singular[:, 1],
                          out=np.full(len(f), np.inf), where=singular[:, 1] > 1e-12)
    return {
        "area_collapse_count": int(np.count_nonzero(area < POLICY["minimum_signed_area_ratio"])),
        "condition_failure_count": int(np.count_nonzero(condition > POLICY["maximum_jacobian_condition"])),
        "minimum_signed_area_ratio": float(np.min(area)),
        "maximum_jacobian_condition": float(np.max(condition)) if np.isfinite(condition).all() else "INFINITE",
    }
