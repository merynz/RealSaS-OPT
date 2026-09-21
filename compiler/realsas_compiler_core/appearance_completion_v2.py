from __future__ import annotations

"""Topology-aware, source-constrained local appearance completion for CAA V2."""

import numpy as np

from .appearance_authority_v2 import CAA_PROVENANCE
from .appearance_color_v2 import (
    premultiplied_linear_to_straight_srgb_u8,
    straight_srgb_rgba_u8_to_premultiplied_linear,
)
from .types import QualificationError


def triangle_lattice_neighbor_pairs(
    tile_resolution: int,
) -> tuple[tuple[int, int], ...]:
    resolution = int(tile_resolution)
    if resolution < 4:
        raise QualificationError("CAA_COMPLETION_TILE_RESOLUTION_INVALID")
    index = {}
    cursor = 0
    for j in range(resolution):
        for i in range(resolution - j):
            index[(i, j)] = cursor
            cursor += 1
    pairs = set()
    for (i, j), a in index.items():
        for di, dj in ((1, 0), (0, 1), (1, -1)):
            other = index.get((i + di, j + dj))
            if other is not None:
                pairs.add((min(a, other), max(a, other)))
    return tuple(sorted(pairs))


def surface_sample_neighbors(
    *,
    positions: np.ndarray,
    face_count: int,
    tile_resolution: int,
) -> tuple[tuple[int, ...], ...]:
    points = np.asarray(positions, dtype=np.float64)
    per_face = int(tile_resolution) * (int(tile_resolution) + 1) // 2
    if points.shape != (int(face_count) * per_face, 3):
        raise QualificationError("CAA_SURFACE_GRAPH_SAMPLE_ACCOUNTING_DRIFT")
    neighbors = [set() for _ in range(len(points))]
    for face_index in range(int(face_count)):
        base = face_index * per_face
        for a, b in triangle_lattice_neighbor_pairs(tile_resolution):
            aa = base + a
            bb = base + b
            neighbors[aa].add(bb)
            neighbors[bb].add(aa)

    buckets: dict[tuple[int, int, int], list[int]] = {}
    scale = 1.0e8
    for sample_index, point in enumerate(points):
        key = tuple(int(round(float(value) * scale)) for value in point)
        buckets.setdefault(key, []).append(sample_index)
    for indices in buckets.values():
        if len(indices) < 2:
            continue
        for i, a in enumerate(indices):
            for b in indices[i + 1 :]:
                neighbors[a].add(b)
                neighbors[b].add(a)
    return tuple(tuple(sorted(row)) for row in neighbors)


def bounded_surface_harmonic_fill(
    *,
    rgba: np.ndarray,
    provenance: np.ndarray,
    source_view: np.ndarray,
    missing: np.ndarray,
    sample_component: tuple[str, ...],
    neighbors: tuple[tuple[int, ...], ...],
    observed_mask: np.ndarray | None = None,
    max_region_samples: int,
    max_graph_hops: int,
) -> dict:
    missing = np.asarray(missing, dtype=bool)
    if rgba.shape != (len(missing), 4):
        raise QualificationError("CAA_HARMONIC_RGBA_SHAPE_INVALID")
    if provenance.shape != (len(missing),) or source_view.shape != (len(missing),):
        raise QualificationError("CAA_HARMONIC_PROVENANCE_SHAPE_INVALID")
    if len(sample_component) != len(missing) or len(neighbors) != len(missing):
        raise QualificationError("CAA_HARMONIC_GRAPH_SHAPE_INVALID")
    if max_region_samples <= 0 or max_graph_hops <= 0:
        raise QualificationError("CAA_HARMONIC_POLICY_INVALID")

    observed = (
        ~missing
        if observed_mask is None
        else np.asarray(observed_mask, dtype=bool).copy()
    )
    if observed.shape != missing.shape or np.any(observed & missing):
        raise QualificationError("CAA_HARMONIC_OBSERVED_MASK_INVALID")
    seen = np.zeros(len(missing), dtype=bool)
    region_count = 0
    max_region_seen = 0
    max_hops_seen = 0

    for seed in np.flatnonzero(missing):
        seed = int(seed)
        if seen[seed]:
            continue
        component_id = str(sample_component[seed])
        stack = [seed]
        seen[seed] = True
        region = []
        while stack:
            node = stack.pop()
            region.append(node)
            for nxt in neighbors[node]:
                if (
                    missing[nxt]
                    and not seen[nxt]
                    and str(sample_component[nxt]) == component_id
                ):
                    seen[nxt] = True
                    stack.append(int(nxt))
        region = tuple(sorted(region))
        region_count += 1
        max_region_seen = max(max_region_seen, len(region))
        if len(region) > int(max_region_samples):
            raise QualificationError("CAA_HARMONIC_REGION_TOO_LARGE")

        region_set = set(region)
        boundary = {
            int(nxt)
            for node in region
            for nxt in neighbors[node]
            if observed[nxt] and str(sample_component[nxt]) == component_id
        }
        if not boundary:
            raise QualificationError("CAA_HARMONIC_REGION_WITHOUT_SOURCE_BOUNDARY")

        hop = {node: None for node in region}
        frontier = []
        for node in region:
            if any(nxt in boundary for nxt in neighbors[node]):
                hop[node] = 1
                frontier.append(node)
        cursor = 0
        while cursor < len(frontier):
            node = frontier[cursor]
            cursor += 1
            current = int(hop[node])
            for nxt in neighbors[node]:
                if nxt in region_set and hop[nxt] is None:
                    hop[nxt] = current + 1
                    frontier.append(int(nxt))
        if any(value is None for value in hop.values()):
            raise QualificationError("CAA_HARMONIC_REGION_GRAPH_DISCONNECTED")
        region_hops = max(int(value) for value in hop.values())
        max_hops_seen = max(max_hops_seen, region_hops)
        if region_hops > int(max_graph_hops):
            raise QualificationError("CAA_HARMONIC_REGION_TOO_DEEP")

        local_index = {node: i for i, node in enumerate(region)}
        size = len(region)
        matrix = np.zeros((size, size), dtype=np.float64)
        rhs = np.zeros((size, 4), dtype=np.float64)
        for node in region:
            row = local_index[node]
            valid_neighbors = [
                int(nxt)
                for nxt in neighbors[node]
                if str(sample_component[nxt]) == component_id
                and (nxt in region_set or observed[nxt])
            ]
            if not valid_neighbors:
                raise QualificationError("CAA_HARMONIC_NODE_WITHOUT_NEIGHBOR")
            matrix[row, row] = float(len(valid_neighbors))
            for nxt in valid_neighbors:
                if nxt in region_set:
                    matrix[row, local_index[nxt]] -= 1.0
                else:
                    rhs[row] += straight_srgb_rgba_u8_to_premultiplied_linear(
                        rgba[nxt][None, :]
                    )[0]
        try:
            solved = np.linalg.solve(matrix, rhs)
        except np.linalg.LinAlgError as exc:
            raise QualificationError("CAA_HARMONIC_SOLVE_SINGULAR") from exc
        solved = np.clip(solved, 0.0, 1.0)
        solved[:, :3] = np.minimum(solved[:, :3], solved[:, 3:4])
        encoded = premultiplied_linear_to_straight_srgb_u8(solved)
        for local, node in enumerate(region):
            rgba[node] = encoded[local]
            provenance[node] = CAA_PROVENANCE["COMPILED_LOCAL_HARMONIC"]
            source_view[node] = -2
            missing[node] = False
            observed[node] = True

    if np.any(missing):
        raise QualificationError("CAA_HARMONIC_TOTALITY_FAILURE")
    return {
        "region_count": int(region_count),
        "maximum_region_samples": int(max_region_seen),
        "maximum_graph_hops": int(max_hops_seen),
    }
