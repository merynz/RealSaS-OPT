from __future__ import annotations

"""Topology-aware, source-constrained local appearance completion for CAA V2."""

from dataclasses import dataclass

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


@dataclass(frozen=True)
class SurfaceSampleGraph:
    """Compact CSR graph plus one copy of each undirected surface edge."""

    offsets: np.ndarray
    indices: np.ndarray
    edge_a: np.ndarray
    edge_b: np.ndarray

    def __len__(self) -> int:
        return int(len(self.offsets) - 1)

    def __getitem__(self, key):
        if isinstance(key, slice):
            start, stop, step = key.indices(len(self))
            return tuple(self[index] for index in range(start, stop, step))
        index = int(key)
        if index < 0:
            index += len(self)
        if not (0 <= index < len(self)):
            raise IndexError(index)
        begin = int(self.offsets[index])
        end = int(self.offsets[index + 1])
        return self.indices[begin:end]

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]

    @property
    def edge_count(self) -> int:
        return int(len(self.edge_a))

    @property
    def storage_nbytes(self) -> int:
        return int(
            self.offsets.nbytes
            + self.indices.nbytes
            + self.edge_a.nbytes
            + self.edge_b.nbytes
        )


def _surface_graph_from_edge_key_chunks(
    *,
    node_count: int,
    edge_key_chunks: list[np.ndarray],
) -> SurfaceSampleGraph:
    count = int(node_count)
    if count <= 0 or count >= (1 << 32):
        raise QualificationError("CAA_SURFACE_GRAPH_NODE_COUNT_UNSUPPORTED")
    chunks = [
        np.asarray(chunk, dtype=np.uint64).reshape(-1)
        for chunk in edge_key_chunks
        if np.asarray(chunk).size
    ]
    if not chunks:
        raise QualificationError("CAA_SURFACE_GRAPH_EDGE_SET_EMPTY")
    keys = np.unique(np.concatenate(chunks, axis=0))
    low_mask = np.uint64(0xFFFFFFFF)
    edge_a = (keys >> np.uint64(32)).astype(np.int32, copy=False)
    edge_b = (keys & low_mask).astype(np.int32, copy=False)
    if (
        np.any(edge_a < 0)
        or np.any(edge_b < 0)
        or np.any(edge_a >= count)
        or np.any(edge_b >= count)
        or np.any(edge_a == edge_b)
    ):
        raise QualificationError("CAA_SURFACE_GRAPH_EDGE_INVALID")

    edge_count = len(keys)
    directed = np.empty(edge_count * 2, dtype=np.uint64)
    directed[:edge_count] = keys
    directed[edge_count:] = (
        np.left_shift(keys & low_mask, np.uint64(32))
        | (keys >> np.uint64(32))
    )
    del keys
    directed.sort()

    sources = (directed >> np.uint64(32)).astype(np.int64, copy=False)
    counts = np.bincount(sources, minlength=count)
    if counts.shape != (count,):
        raise QualificationError("CAA_SURFACE_GRAPH_DEGREE_ACCOUNTING_DRIFT")
    offsets = np.empty(count + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, dtype=np.int64, out=offsets[1:])
    indices = (directed & low_mask).astype(np.int32, copy=False)
    del directed, sources, counts

    graph = SurfaceSampleGraph(
        offsets=offsets,
        indices=indices,
        edge_a=edge_a,
        edge_b=edge_b,
    )
    if int(graph.offsets[-1]) != 2 * graph.edge_count:
        raise QualificationError("CAA_SURFACE_GRAPH_DIRECTED_EDGE_ACCOUNTING_DRIFT")
    if np.any(graph.offsets[1:] <= graph.offsets[:-1]):
        raise QualificationError("CAA_SURFACE_GRAPH_ISOLATED_SAMPLE")
    return graph


def surface_sample_neighbors(
    *,
    positions: np.ndarray,
    face_count: int,
    tile_resolution: int | None = None,
    face_sample_offsets: np.ndarray | None = None,
    face_tile_resolutions: np.ndarray | None = None,
    face_vertex_ids: tuple[tuple[str, str, str], ...] | None = None,
) -> SurfaceSampleGraph:
    """Build the canonical surface graph in compact CSR form.

    Product mode is topology-parametric: each face keeps its own triangular
    lattice resolution and true shared vertices/edges are coupled from exact
    face topology, never from Euclidean-nearest surface proximity.
    """
    points = np.asarray(positions, dtype=np.float64)
    faces = int(face_count)
    if points.ndim != 2 or points.shape[1] != 3 or faces <= 0:
        raise QualificationError("CAA_SURFACE_GRAPH_INPUT_INVALID")

    adaptive = face_sample_offsets is not None or face_tile_resolutions is not None
    if adaptive:
        if face_sample_offsets is None or face_tile_resolutions is None:
            raise QualificationError("CAA_SURFACE_GRAPH_ADAPTIVE_LAYOUT_INCOMPLETE")
        offsets = np.asarray(face_sample_offsets, dtype=np.int64)
        resolutions = np.asarray(face_tile_resolutions, dtype=np.int32)
        if (
            offsets.shape != (faces + 1,)
            or resolutions.shape != (faces,)
            or offsets[0] != 0
            or offsets[-1] != len(points)
            or np.any(offsets[1:] <= offsets[:-1])
            or np.any(resolutions < 4)
        ):
            raise QualificationError("CAA_SURFACE_GRAPH_ADAPTIVE_LAYOUT_INVALID")
        expected = (
            resolutions.astype(np.int64)
            * (resolutions.astype(np.int64) + 1)
            // 2
        )
        if not np.array_equal(offsets[1:] - offsets[:-1], expected):
            raise QualificationError(
                "CAA_SURFACE_GRAPH_ADAPTIVE_SAMPLE_ACCOUNTING_DRIFT"
            )
    else:
        if tile_resolution is None:
            raise QualificationError("CAA_SURFACE_GRAPH_TILE_RESOLUTION_MISSING")
        resolution = int(tile_resolution)
        per_face = resolution * (resolution + 1) // 2
        if resolution < 4 or points.shape != (faces * per_face, 3):
            raise QualificationError("CAA_SURFACE_GRAPH_SAMPLE_ACCOUNTING_DRIFT")
        offsets = np.arange(faces + 1, dtype=np.int64) * per_face
        resolutions = np.full((faces,), resolution, dtype=np.int32)

    # Local triangular-lattice edges are vectorized by resolution and encoded
    # as packed uint64 undirected keys. This avoids millions of Python sets.
    edge_key_chunks: list[np.ndarray] = []
    pair_cache: dict[int, np.ndarray] = {}
    for resolution in sorted(set(map(int, resolutions.tolist()))):
        raw_pairs = triangle_lattice_neighbor_pairs(resolution)
        pairs = np.asarray(raw_pairs, dtype=np.uint64)
        if pairs.ndim != 2 or pairs.shape[1] != 2 or len(pairs) <= 0:
            raise QualificationError("CAA_SURFACE_GRAPH_LOCAL_PATTERN_INVALID")
        pair_cache[resolution] = pairs
        face_ids = np.flatnonzero(resolutions == resolution).astype(np.int64)
        max_edges_per_chunk = 500_000
        faces_per_chunk = max(1, max_edges_per_chunk // len(pairs))
        for cursor in range(0, len(face_ids), faces_per_chunk):
            ids = face_ids[cursor : cursor + faces_per_chunk]
            bases = offsets[ids].astype(np.uint64)[:, None]
            aa = bases + pairs[None, :, 0]
            bb = bases + pairs[None, :, 1]
            lo = np.minimum(aa, bb).reshape(-1)
            hi = np.maximum(aa, bb).reshape(-1)
            edge_key_chunks.append(
                np.left_shift(lo, np.uint64(32)) | hi
            )

    pending_cross: list[int] = []

    def flush_cross() -> None:
        nonlocal pending_cross
        if pending_cross:
            edge_key_chunks.append(np.asarray(pending_cross, dtype=np.uint64))
            pending_cross = []

    def add_cross_edge(a: int, b: int) -> None:
        aa = int(a)
        bb = int(b)
        if aa == bb:
            raise QualificationError("CAA_SURFACE_GRAPH_SELF_EDGE")
        lo = min(aa, bb)
        hi = max(aa, bb)
        if lo < 0 or hi >= len(points):
            raise QualificationError("CAA_SURFACE_GRAPH_CROSS_EDGE_OUT_OF_RANGE")
        pending_cross.append((lo << 32) | hi)
        if len(pending_cross) >= 250_000:
            flush_cross()

    if face_vertex_ids is not None:
        topology = tuple(tuple(map(str, row)) for row in face_vertex_ids)
        if len(topology) != faces or any(len(row) != 3 for row in topology):
            raise QualificationError("CAA_SURFACE_GRAPH_FACE_TOPOLOGY_INVALID")
        if any(len(set(row)) != 3 for row in topology):
            raise QualificationError("CAA_SURFACE_GRAPH_FACE_TOPOLOGY_DEGENERATE")

        bary_cache: dict[int, np.ndarray] = {}

        def local_barycentric(resolution: int) -> np.ndarray:
            cached = bary_cache.get(int(resolution))
            if cached is not None:
                return cached
            denominator = float(int(resolution) - 1)
            rows = []
            for jj in range(int(resolution)):
                for ii in range(int(resolution) - jj):
                    u = float(ii) / denominator
                    v = float(jj) / denominator
                    rows.append((1.0 - u - v, u, v))
            value = np.asarray(rows, dtype=np.float64)
            bary_cache[int(resolution)] = value
            return value

        def corner_sample(face_index: int, vertex_id: str) -> int:
            row = topology[face_index]
            corner = row.index(vertex_id)
            bary = local_barycentric(int(resolutions[face_index]))
            local = int(np.argmax(bary[:, corner]))
            if float(bary[local, corner]) < 1.0 - 1e-12:
                raise QualificationError("CAA_SURFACE_GRAPH_CORNER_SAMPLE_MISSING")
            return int(offsets[face_index]) + local

        vertex_faces: dict[str, list[int]] = {}
        for face_index, row in enumerate(topology):
            for vertex_id in row:
                vertex_faces.setdefault(vertex_id, []).append(face_index)
        for vertex_id, incident in vertex_faces.items():
            unique_incident = sorted(set(incident))
            corner_indices = [
                corner_sample(face_index, vertex_id)
                for face_index in unique_incident
            ]
            for i, a in enumerate(corner_indices):
                for b in corner_indices[i + 1 :]:
                    add_cross_edge(a, b)

        edge_faces: dict[tuple[str, str], list[int]] = {}
        for face_index, row in enumerate(topology):
            for a, b in (
                (row[0], row[1]),
                (row[1], row[2]),
                (row[2], row[0]),
            ):
                edge = tuple(sorted((a, b)))
                edge_faces.setdefault(edge, []).append(face_index)

        edge_local_cache: dict[
            tuple[int, int, int],
            tuple[np.ndarray, np.ndarray],
        ] = {}

        def edge_samples(
            face_index: int,
            edge: tuple[str, str],
        ) -> tuple[np.ndarray, np.ndarray]:
            row = topology[face_index]
            try:
                corner_a = row.index(edge[0])
                corner_b = row.index(edge[1])
            except ValueError as exc:
                raise QualificationError(
                    "CAA_SURFACE_GRAPH_EDGE_TOPOLOGY_DRIFT"
                ) from exc
            cache_key = (
                int(resolutions[face_index]),
                int(corner_a),
                int(corner_b),
            )
            cached = edge_local_cache.get(cache_key)
            if cached is None:
                bary = local_barycentric(cache_key[0])
                nonshared = ({0, 1, 2} - {corner_a, corner_b}).pop()
                local_ids = np.flatnonzero(
                    np.abs(bary[:, nonshared]) <= 1e-12
                ).astype(np.int64)
                t = bary[local_ids, corner_b].astype(np.float64)
                order = np.argsort(t, kind="stable")
                cached = (t[order], local_ids[order])
                edge_local_cache[cache_key] = cached
            t, local_ids = cached
            return t, int(offsets[face_index]) + local_ids

        def bracket_indices(
            values: np.ndarray,
            indices: np.ndarray,
            t: float,
        ) -> tuple[int, ...]:
            if len(values) == 0:
                raise QualificationError("CAA_SURFACE_GRAPH_SHARED_EDGE_EMPTY")
            right = int(np.searchsorted(values, float(t), side="left"))
            if right < len(values) and abs(float(values[right]) - float(t)) <= 1e-12:
                return (int(indices[right]),)
            if right > 0 and abs(float(values[right - 1]) - float(t)) <= 1e-12:
                return (int(indices[right - 1]),)
            picks = []
            if right < len(values):
                picks.append(int(indices[right]))
            if right > 0:
                picks.append(int(indices[right - 1]))
            return tuple(sorted(set(picks)))

        for edge, incident in edge_faces.items():
            unique_incident = sorted(set(incident))
            if len(unique_incident) < 2:
                continue
            for i, face_a in enumerate(unique_incident):
                t_a, idx_a = edge_samples(face_a, edge)
                for face_b in unique_incident[i + 1 :]:
                    t_b, idx_b = edge_samples(face_b, edge)
                    for t, a in zip(t_a, idx_a):
                        for b in bracket_indices(t_b, idx_b, float(t)):
                            add_cross_edge(int(a), int(b))
                    for t, b in zip(t_b, idx_b):
                        for a in bracket_indices(t_a, idx_a, float(t)):
                            add_cross_edge(int(a), int(b))
    else:
        # Backwards-compatible isolated-test fallback. Shipping product CAA
        # always supplies exact face topology.
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
                    add_cross_edge(a, b)

    flush_cross()
    return _surface_graph_from_edge_key_chunks(
        node_count=len(points),
        edge_key_chunks=edge_key_chunks,
    )

def bounded_surface_harmonic_fill(
    *,
    rgba: np.ndarray,
    provenance: np.ndarray,
    source_view: np.ndarray,
    missing: np.ndarray,
    sample_component: tuple[str, ...],
    neighbors: SurfaceSampleGraph,
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
