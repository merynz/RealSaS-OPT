from __future__ import annotations

"""Deterministic topology-bound variational appearance completion.

This module is a research/qualification primitive, not implicit product
authority. It minimizes a weighted graph Dirichlet energy with immutable source
constraints:

    E(X) = 1/2 * sum_(i,j) w_ij ||X_i - X_j||^2

Only unknown nodes are solved. Known/source nodes are exact Dirichlet boundary
conditions. Graph edges are canonical-surface topology edges; no Euclidean
cross-sheet nearest-neighbor transfer is introduced.
"""

from dataclasses import dataclass
import heapq

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import LinearOperator, cg

from .appearance_completion_v2 import SurfaceSampleGraph
from .types import QualificationError


@dataclass(frozen=True)
class VariationalCompletionStats:
    node_count: int
    known_count: int
    unknown_count: int
    edge_count: int
    positive_edge_length_median: float
    zero_length_topology_edge_count: int
    minimum_edge_weight: float
    maximum_edge_weight: float
    channel_iterations: tuple[int, ...]
    channel_relative_residuals: tuple[float, ...]
    guide_weight: float = 0.0
    guide_mode: str = "NONE"

    def to_dict(self) -> dict:
        return {
            "node_count": int(self.node_count),
            "known_count": int(self.known_count),
            "unknown_count": int(self.unknown_count),
            "edge_count": int(self.edge_count),
            "positive_edge_length_median": float(
                self.positive_edge_length_median
            ),
            "zero_length_topology_edge_count": int(
                self.zero_length_topology_edge_count
            ),
            "minimum_edge_weight": float(self.minimum_edge_weight),
            "maximum_edge_weight": float(self.maximum_edge_weight),
            "channel_iterations": list(map(int, self.channel_iterations)),
            "channel_relative_residuals": list(
                map(float, self.channel_relative_residuals)
            ),
            "source_constraints_exact": True,
            "cross_component_edges_consumed": False,
            "objective": "WEIGHTED_GRAPH_DIRICHLET_ENERGY",
            "solver": "SCIPY_CONJUGATE_GRADIENT_JACOBI_PRECONDITIONED",
            "guide_weight": float(self.guide_weight),
            "guide_mode": str(self.guide_mode),
        }


def _edge_weights(
    positions: np.ndarray,
    edge_a: np.ndarray,
    edge_b: np.ndarray,
    *,
    zero_length_weight: float,
    minimum_weight: float,
    maximum_weight: float,
) -> tuple[np.ndarray, float, int]:
    points = np.asarray(positions, dtype=np.float64)
    delta = points[edge_a] - points[edge_b]
    length = np.linalg.norm(delta, axis=1)
    if not np.isfinite(length).all():
        raise QualificationError("CAA_VARIATIONAL_EDGE_LENGTH_NONFINITE")
    positive = length > 1.0e-12
    if not np.any(positive):
        raise QualificationError("CAA_VARIATIONAL_NO_POSITIVE_EDGE_LENGTH")
    scale = float(np.median(length[positive]))
    if not np.isfinite(scale) or scale <= 0.0:
        raise QualificationError("CAA_VARIATIONAL_EDGE_SCALE_INVALID")

    normalized = length / scale
    weight = np.empty(len(length), dtype=np.float64)
    weight[~positive] = float(zero_length_weight)
    weight[positive] = 1.0 / np.maximum(normalized[positive], 1.0e-6)
    weight = np.clip(
        weight,
        float(minimum_weight),
        float(maximum_weight),
    )
    if not np.isfinite(weight).all() or np.any(weight <= 0.0):
        raise QualificationError("CAA_VARIATIONAL_EDGE_WEIGHT_INVALID")
    return weight, scale, int(np.count_nonzero(~positive))


def geodesic_source_guidance(
    *,
    values: np.ndarray,
    known_mask: np.ndarray,
    sample_component: np.ndarray,
    positions: np.ndarray,
    graph: SurfaceSampleGraph,
) -> tuple[np.ndarray, np.ndarray]:
    """Propagate the nearest same-component source value over surface geodesics.

    The returned donor id is a source node index. Exact distance ties resolve
    to the lower source node index, making the guidance byte-stable.
    """
    source=np.asarray(values,dtype=np.float64)
    known=np.asarray(known_mask,dtype=bool)
    component=np.asarray(sample_component,dtype=np.int32)
    points=np.asarray(positions,dtype=np.float64)
    node_count=len(known)
    if (
        source.ndim!=2
        or source.shape[0]!=node_count
        or component.shape!=(node_count,)
        or points.shape!=(node_count,3)
        or len(graph)!=node_count
    ):
        raise QualificationError("CAA_GEODESIC_GUIDANCE_SHAPE_INVALID")
    seeds=np.flatnonzero(known).astype(np.int64)
    if len(seeds)==0:
        raise QualificationError("CAA_GEODESIC_GUIDANCE_NO_SOURCE")

    edge_a=np.asarray(graph.edge_a,dtype=np.int64)
    edge_b=np.asarray(graph.edge_b,dtype=np.int64)
    lengths=np.linalg.norm(points[edge_a]-points[edge_b],axis=1)
    positive=lengths>1.0e-12
    if not np.any(positive):
        raise QualificationError("CAA_GEODESIC_GUIDANCE_NO_POSITIVE_EDGE")
    scale=float(np.median(lengths[positive]))
    if not np.isfinite(scale) or scale<=0.0:
        raise QualificationError("CAA_GEODESIC_GUIDANCE_SCALE_INVALID")

    distance=np.full(node_count,np.inf,dtype=np.float64)
    donor=np.full(node_count,-1,dtype=np.int64)
    heap=[]
    for seed in seeds.tolist():
        distance[seed]=0.0
        donor[seed]=seed
        heapq.heappush(heap,(0.0,int(seed),int(seed)))

    while heap:
        dist,source_id,node=heapq.heappop(heap)
        if dist>distance[node]+1.0e-15:
            continue
        if abs(dist-distance[node])<=1.0e-15 and source_id!=donor[node]:
            continue
        cid=int(component[node])
        for raw in graph[node]:
            nxt=int(raw)
            if int(component[nxt])!=cid:
                continue
            step=float(np.linalg.norm(points[node]-points[nxt]))/scale
            step=max(step,1.0e-6)
            candidate=dist+step
            better=candidate<distance[nxt]-1.0e-15
            tied=abs(candidate-distance[nxt])<=1.0e-15
            if better or (tied and (donor[nxt]<0 or source_id<donor[nxt])):
                distance[nxt]=candidate
                donor[nxt]=source_id
                heapq.heappush(heap,(candidate,source_id,nxt))

    if np.any(donor<0) or not np.isfinite(distance).all():
        raise QualificationError("CAA_GEODESIC_GUIDANCE_UNREACHED_NODE")
    guidance=source[donor].copy()
    if not np.array_equal(guidance[known],source[known]):
        raise QualificationError("CAA_GEODESIC_GUIDANCE_SOURCE_DRIFT")
    return guidance,donor


def solve_weighted_surface_dirichlet(
    *,
    values: np.ndarray,
    known_mask: np.ndarray,
    sample_component: np.ndarray,
    positions: np.ndarray,
    graph: SurfaceSampleGraph,
    zero_length_weight: float = 32.0,
    minimum_weight: float = 0.25,
    maximum_weight: float = 32.0,
    rtol: float = 1.0e-9,
    atol: float = 1.0e-11,
    maxiter: int = 2048,
    guide_values: np.ndarray | None = None,
    guide_weight: float = 0.0,
    guide_mode: str = "NONE",
) -> tuple[np.ndarray, VariationalCompletionStats]:
    """Fill unknown graph nodes by minimizing weighted Dirichlet energy."""
    source = np.asarray(values, dtype=np.float64)
    known = np.asarray(known_mask, dtype=bool)
    component = np.asarray(sample_component, dtype=np.int32)
    points = np.asarray(positions, dtype=np.float64)

    if source.ndim != 2 or source.shape[0] == 0:
        raise QualificationError("CAA_VARIATIONAL_VALUE_SHAPE_INVALID")
    node_count, channel_count = source.shape
    if (
        known.shape != (node_count,)
        or component.shape != (node_count,)
        or points.shape != (node_count, 3)
        or len(graph) != node_count
    ):
        raise QualificationError("CAA_VARIATIONAL_INPUT_SHAPE_DRIFT")
    if not np.isfinite(source[known]).all() or not np.isfinite(points).all():
        raise QualificationError("CAA_VARIATIONAL_INPUT_NONFINITE")
    if not (0.0 < rtol < 1.0) or atol < 0.0 or maxiter <= 0:
        raise QualificationError("CAA_VARIATIONAL_SOLVER_POLICY_INVALID")
    lam=float(guide_weight)
    if not np.isfinite(lam) or lam<0.0:
        raise QualificationError("CAA_VARIATIONAL_GUIDE_WEIGHT_INVALID")
    guide=None
    if lam>0.0:
        if guide_values is None:
            raise QualificationError("CAA_VARIATIONAL_GUIDE_MISSING")
        guide=np.asarray(guide_values,dtype=np.float64)
        if guide.shape!=source.shape or not np.isfinite(guide).all():
            raise QualificationError("CAA_VARIATIONAL_GUIDE_INVALID")

    unknown = ~known
    unknown_count = int(np.count_nonzero(unknown))
    if unknown_count == 0:
        stats = VariationalCompletionStats(
            node_count=node_count,
            known_count=node_count,
            unknown_count=0,
            edge_count=graph.edge_count,
            positive_edge_length_median=0.0,
            zero_length_topology_edge_count=0,
            minimum_edge_weight=0.0,
            maximum_edge_weight=0.0,
            channel_iterations=tuple(0 for _ in range(channel_count)),
            channel_relative_residuals=tuple(0.0 for _ in range(channel_count)),
            guide_weight=lam,
            guide_mode=str(guide_mode),
        )
        return source.copy(), stats

    component_ids = np.unique(component)
    for cid in component_ids:
        take = component == int(cid)
        if np.any(take & unknown) and not np.any(take & known):
            raise QualificationError(
                "CAA_VARIATIONAL_COMPONENT_WITHOUT_SOURCE_CONSTRAINT"
            )

    edge_a = np.asarray(graph.edge_a, dtype=np.int64)
    edge_b = np.asarray(graph.edge_b, dtype=np.int64)
    same_component = component[edge_a] == component[edge_b]
    edge_a = edge_a[same_component]
    edge_b = edge_b[same_component]
    if len(edge_a) == 0:
        raise QualificationError("CAA_VARIATIONAL_EDGE_SET_EMPTY")
    weights, median_length, zero_count = _edge_weights(
        points,
        edge_a,
        edge_b,
        zero_length_weight=zero_length_weight,
        minimum_weight=minimum_weight,
        maximum_weight=maximum_weight,
    )

    unknown_nodes = np.flatnonzero(unknown).astype(np.int64)
    local = np.full(node_count, -1, dtype=np.int64)
    local[unknown_nodes] = np.arange(unknown_count, dtype=np.int64)

    diagonal = np.zeros(unknown_count, dtype=np.float64)
    rhs = np.zeros((unknown_count, channel_count), dtype=np.float64)
    rows = []
    cols = []
    data = []

    def consume(src_nodes, dst_nodes, edge_weight):
        src_unknown = unknown[src_nodes]
        if not np.any(src_unknown):
            return
        src_take = src_nodes[src_unknown]
        dst_take = dst_nodes[src_unknown]
        w_take = edge_weight[src_unknown]
        row = local[src_take]
        np.add.at(diagonal, row, w_take)

        dst_unknown = unknown[dst_take]
        if np.any(dst_unknown):
            rows.append(row[dst_unknown])
            cols.append(local[dst_take[dst_unknown]])
            data.append(-w_take[dst_unknown])
        if np.any(~dst_unknown):
            rr = row[~dst_unknown]
            vv = dst_take[~dst_unknown]
            ww = w_take[~dst_unknown]
            for channel in range(channel_count):
                np.add.at(
                    rhs[:, channel],
                    rr,
                    ww * source[vv, channel],
                )

    consume(edge_a, edge_b, weights)
    consume(edge_b, edge_a, weights)

    if lam>0.0:
        diagonal += lam
        rhs += lam * guide[unknown_nodes]

    if np.any(diagonal <= 0.0) or not np.isfinite(diagonal).all():
        raise QualificationError("CAA_VARIATIONAL_UNKNOWN_NODE_UNCONSTRAINED")

    matrix = coo_matrix(
        (
            np.concatenate([diagonal] + data),
            (
                np.concatenate(
                    [np.arange(unknown_count, dtype=np.int64)] + rows
                ),
                np.concatenate(
                    [np.arange(unknown_count, dtype=np.int64)] + cols
                ),
            ),
        ),
        shape=(unknown_count, unknown_count),
        dtype=np.float64,
    ).tocsr()
    matrix.sum_duplicates()

    inv_diag = 1.0 / matrix.diagonal()
    preconditioner = LinearOperator(
        shape=matrix.shape,
        matvec=lambda x: inv_diag * x,
        dtype=np.float64,
    )

    component_mean = {}
    for cid in component_ids:
        anchors = (component == int(cid)) & known
        if np.any(anchors):
            component_mean[int(cid)] = np.mean(
                source[anchors],
                axis=0,
                dtype=np.float64,
            )
    x0 = np.empty((unknown_count, channel_count), dtype=np.float64)
    unknown_component = component[unknown_nodes]
    for cid, mean in component_mean.items():
        x0[unknown_component == cid] = mean

    solved = np.empty_like(x0)
    iterations = []
    residuals = []
    for channel in range(channel_count):
        count = 0

        def callback(_x):
            nonlocal count
            count += 1

        answer, info = cg(
            matrix,
            rhs[:, channel],
            x0=x0[:, channel],
            rtol=float(rtol),
            atol=float(atol),
            maxiter=int(maxiter),
            M=preconditioner,
            callback=callback,
        )
        if info != 0 or not np.isfinite(answer).all():
            raise QualificationError(
                f"CAA_VARIATIONAL_SOLVE_NOT_CONVERGED:{channel}:{info}"
            )
        residual = matrix @ answer - rhs[:, channel]
        denominator = max(
            float(np.linalg.norm(rhs[:, channel])),
            1.0e-15,
        )
        relative = float(np.linalg.norm(residual) / denominator)
        if not np.isfinite(relative):
            raise QualificationError(
                "CAA_VARIATIONAL_RESIDUAL_NONFINITE"
            )
        solved[:, channel] = answer
        iterations.append(int(count))
        residuals.append(relative)

    output = source.copy()
    output[unknown_nodes] = solved
    if not np.array_equal(output[known], source[known]):
        raise QualificationError("CAA_VARIATIONAL_SOURCE_CONSTRAINT_MUTATED")

    stats = VariationalCompletionStats(
        node_count=node_count,
        known_count=int(np.count_nonzero(known)),
        unknown_count=unknown_count,
        edge_count=int(len(edge_a)),
        positive_edge_length_median=median_length,
        zero_length_topology_edge_count=zero_count,
        minimum_edge_weight=float(np.min(weights)),
        maximum_edge_weight=float(np.max(weights)),
        channel_iterations=tuple(iterations),
        channel_relative_residuals=tuple(residuals),
        guide_weight=lam,
        guide_mode=str(guide_mode),
    )
    return output, stats
