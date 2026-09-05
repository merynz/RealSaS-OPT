from __future__ import annotations

from collections import deque
from dataclasses import replace

import numpy as np

from compiler.realsas_compiler_core.types import RiggingSurfaceIR
from models.geppetto.v2.geppetto_conditioning_v2 import (
    FEATURE_CONTRACT_V2,
    GeppettoConditioningAdapterV2,
)


RASTER_FEATURES = (
    "raster_mean_x",
    "raster_mean_y",
    "raster_std_x",
    "raster_std_y",
)
RASTER_COLUMNS = tuple(FEATURE_CONTRACT_V2.index(x) for x in RASTER_FEATURES)


def strip_raster_bindings_v1(surface: RiggingSurfaceIR) -> RiggingSurfaceIR:
    """Reproduce the B1-family compact-boundary omission without changing geometry."""
    nodes = tuple(replace(n, raster_bindings=()) for n in surface.surface_nodes)
    meta = dict(surface.metadata)
    meta["boundary_ablation_raster_policy"] = "STRIPPED_TO_MATCH_B1_COMPACT_WITNESS"
    return replace(surface, surface_nodes=nodes, metadata=meta)


def relation_edge_index_v1(surface: RiggingSurfaceIR) -> tuple[np.ndarray, tuple[str, ...]]:
    """Return undirected GSA relation edges in Geppetto's sorted-surface-id row order."""
    ids = tuple(sorted(str(n.surface_id) for n in surface.surface_nodes))
    row = {sid: i for i, sid in enumerate(ids)}
    edges = set()
    for rel in surface.local_relations:
        a = row.get(str(rel.a_surface_id))
        b = row.get(str(rel.b_surface_id))
        if a is None or b is None:
            raise ValueError("local relation references unknown surface id")
        if a == b:
            continue
        edges.add(tuple(sorted((a, b))))
    arr = np.asarray(sorted(edges), dtype=np.int64)
    if arr.size == 0:
        arr = np.zeros((0, 2), dtype=np.int64)
    return arr, ids


def gsa_graph_neighbor_index_v1(
    surface: RiggingSurfaceIR,
    *,
    k: int = 16,
) -> tuple[np.ndarray, dict]:
    """Build fixed-cardinality neighbors ranked by GSA graph geodesic locality.

    Cardinality matches current Geppetto local attention: self + (k-1) external
    neighbors. The intervention changes only the locality metric:
    Euclidean-KNN -> shortest-hop GSA relation graph, with Euclidean distance and
    sorted surface_id as deterministic within-hop tie breakers.
    """
    if k < 2:
        raise ValueError("k must be >=2")
    edges, ids = relation_edge_index_v1(surface)
    nodes_by_id = {str(n.surface_id): n for n in surface.surface_nodes}
    P = np.asarray([nodes_by_id[sid].P for sid in ids], dtype=np.float64)
    n = len(ids)
    if k > n:
        raise ValueError("k exceeds surface node count")
    adjacency = [[] for _ in range(n)]
    for a, b in edges.tolist():
        adjacency[a].append(b)
        adjacency[b].append(a)
    for row in adjacency:
        row.sort()

    out = np.empty((n, k), dtype=np.int64)
    disconnected_rows = 0
    max_hop_used = 0
    for src in range(n):
        hop = np.full(n, -1, dtype=np.int64)
        hop[src] = 0
        q = deque([src])
        while q:
            a = q.popleft()
            for b in adjacency[a]:
                if hop[b] < 0:
                    hop[b] = hop[a] + 1
                    q.append(b)
        if np.any(hop < 0):
            disconnected_rows += 1

        dist = np.linalg.norm(P - P[src][None, :], axis=1)
        candidates = list(range(n))
        candidates.sort(
            key=lambda j: (
                0 if j == src else 1,
                10**9 if hop[j] < 0 else int(hop[j]),
                float(dist[j]),
                j,
            )
        )
        selected = candidates[:k]
        out[src] = np.asarray(selected, dtype=np.int64)
        finite_hops = [int(hop[j]) for j in selected if hop[j] >= 0]
        if finite_hops:
            max_hop_used = max(max_hop_used, max(finite_hops))

    telemetry = {
        "schema": "RealSaS.GSA.GraphNeighborIndex.v1",
        "node_count": n,
        "undirected_relation_edge_count": int(len(edges)),
        "k": int(k),
        "self_included": True,
        "disconnected_source_row_count": int(disconnected_rows),
        "max_selected_graph_hop": int(max_hop_used),
        "ranking": "SELF__GRAPH_HOP__EUCLIDEAN_DISTANCE__SORTED_SURFACE_ID",
    }
    return out, telemetry


def conditioning_boundary_report_v1(
    production_surface: RiggingSurfaceIR,
    compact_like_surface: RiggingSurfaceIR | None = None,
) -> dict:
    """Measure exactly what the B1 compact boundary removes from 24D conditioning."""
    adapter = GeppettoConditioningAdapterV2()
    full = adapter([production_surface])
    compact_surface = compact_like_surface or strip_raster_bindings_v1(production_surface)
    compact = adapter([compact_surface])

    if full.surface_ids != compact.surface_ids:
        raise ValueError("surface row identity drift between conditioning arms")
    if not np.allclose(full.positions_normalized, compact.positions_normalized, atol=0, rtol=0):
        raise ValueError("surface position drift between conditioning arms")

    delta = full.features - compact.features
    raster = full.features[0, :, RASTER_COLUMNS]
    nonraster_cols = tuple(i for i in range(full.features.shape[-1]) if i not in RASTER_COLUMNS)
    nonraster_delta = delta[0, :, nonraster_cols]

    graph_idx, graph_tel = gsa_graph_neighbor_index_v1(production_surface, k=16)
    return {
        "schema": "RealSaS.GSA.GeppettoConditioningBoundaryReport.v1",
        "feature_contract": FEATURE_CONTRACT_V2,
        "raster_columns": RASTER_COLUMNS,
        "raster_feature_names": RASTER_FEATURES,
        "raster_channel_mean": raster.mean(axis=0).astype(float).tolist(),
        "raster_channel_std": raster.std(axis=0).astype(float).tolist(),
        "raster_channel_nonzero_fraction": (np.abs(raster) > 1e-12).mean(axis=0).astype(float).tolist(),
        "max_abs_nonraster_feature_delta": float(np.abs(nonraster_delta).max(initial=0.0)),
        "max_abs_raster_feature_delta": float(np.abs(delta[0, :, RASTER_COLUMNS]).max(initial=0.0)),
        "gsa_graph_neighbor_telemetry": graph_tel,
        "gsa_graph_neighbor_index_shape": list(graph_idx.shape),
    }


__all__ = [
    "RASTER_FEATURES",
    "RASTER_COLUMNS",
    "strip_raster_bindings_v1",
    "relation_edge_index_v1",
    "gsa_graph_neighbor_index_v1",
    "conditioning_boundary_report_v1",
]
