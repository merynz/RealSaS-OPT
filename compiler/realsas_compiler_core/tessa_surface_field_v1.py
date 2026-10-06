from __future__ import annotations

"""Compiler-side material support field for TESSA learned mesh proposals.

This module intentionally does NOT promote learned TESSA geometry to canonical
product geometry.  It solves a narrower contract: every generated mesh vertex
gets a deterministic, component-safe convex support over RiggingSurfaceIR nodes
so downstream skin/mechanical fields can be queried continuously instead of by
teacher/source vertex index.

Geometry authority and material/mechanical support authority are deliberately
separate here.  The legacy QualifiedMesh G1 court may still reject the learned
vertex position; that rejection must not be hidden by fabricating support
coefficients or by moving the learned point during material binding.
"""

from dataclasses import asdict, dataclass, field, replace
import math
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from .hashing import content_sha256
from .product_authority_v1 import MechanicalPartitionIR, validate_mechanical_partition
from .types import QualificationError, RiggingSurfaceIR, SurfaceSupportBinding

Json = dict[str, Any]
_EPS = 1.0e-12


@dataclass(frozen=True)
class TESSAMaterialSupportRowIR:
    proposal_vertex_id: str
    decoded_component_index: int
    mechanical_component_id: str
    coefficients: tuple[tuple[str, float], ...]
    nearest_surface_distance: float
    seed_surface_id: str
    support_graph_hops: int
    metadata: Json = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class TESSAMaterialSupportFieldIR:
    rows: tuple[TESSAMaterialSupportRowIR, ...]
    surface_binding_hash: str
    partition_binding_hash: str
    topology_sequence_hash: str
    field_lineage_hash: str
    schema_version: str = "RealSaS.TESSAMaterialSupportFieldIR.v1"
    metadata: Json = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def tessa_material_support_field_lineage_hash(value: TESSAMaterialSupportFieldIR) -> str:
    payload = value.to_dict()
    payload.pop("field_lineage_hash", None)
    return content_sha256(payload)


def _owner_map(partition: MechanicalPartitionIR) -> dict[str, str]:
    return {
        str(sid): str(component.component_id)
        for component in partition.components
        for sid in component.surface_ids
    }


def _graph(surface: RiggingSurfaceIR) -> dict[str, tuple[str, ...]]:
    out: dict[str, set[str]] = {str(node.surface_id): set() for node in surface.surface_nodes}
    for relation in surface.local_relations:
        a = str(relation.a_surface_id)
        b = str(relation.b_surface_id)
        if a not in out or b not in out or a == b:
            raise QualificationError("TESSA_SUPPORT_RELATION_ENDPOINT_INVALID")
        # Material support follows admitted structural adjacency only. Unknown
        # relation semantics are handled by MechanicalPartition; no new edge is
        # invented here.
        out[a].add(b)
        out[b].add(a)
    return {sid: tuple(sorted(nbs)) for sid, nbs in out.items()}


def _component_rows(
    surface: RiggingSurfaceIR,
    partition: MechanicalPartitionIR,
) -> tuple[dict[str, np.ndarray], dict[str, tuple[str, ...]], dict[str, cKDTree]]:
    node_by_id = {str(node.surface_id): node for node in surface.surface_nodes}
    ids_by_component = {
        str(component.component_id): tuple(sorted(map(str, component.surface_ids)))
        for component in partition.components
    }
    points: dict[str, np.ndarray] = {}
    trees: dict[str, cKDTree] = {}
    for cid, ids in ids_by_component.items():
        arr = np.asarray([node_by_id[sid].P for sid in ids], dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 3 or not np.isfinite(arr).all():
            raise QualificationError("TESSA_SUPPORT_COMPONENT_POINTS_INVALID")
        points[cid] = arr
        trees[cid] = cKDTree(arr)
    return points, ids_by_component, trees


def _assign_decoded_components(
    vertices_world: np.ndarray,
    decoded_component_indices: np.ndarray,
    *,
    component_points: dict[str, np.ndarray],
    component_trees: dict[str, cKDTree],
) -> tuple[dict[int, str], dict[int, dict]]:
    assignments: dict[int, str] = {}
    evidence: dict[int, dict] = {}
    for decoded_ci in sorted(set(map(int, decoded_component_indices.tolist()))):
        mask = decoded_component_indices == decoded_ci
        pts = vertices_world[mask]
        if len(pts) == 0:
            raise QualificationError("TESSA_SUPPORT_EMPTY_DECODED_COMPONENT")
        scores = []
        for cid in sorted(component_trees):
            distances, _ = component_trees[cid].query(pts, k=1)
            distances = np.asarray(distances, dtype=np.float64).reshape(-1)
            scores.append((
                float(np.median(distances)),
                float(np.percentile(distances, 95)),
                str(cid),
                int(len(component_points[cid])),
            ))
        scores.sort(key=lambda row: (row[0], row[1], row[2]))
        winner = scores[0]
        assignments[decoded_ci] = str(winner[2])
        evidence[decoded_ci] = {
            "mechanical_component_id": str(winner[2]),
            "median_nearest_distance": float(winner[0]),
            "p95_nearest_distance": float(winner[1]),
            "mechanical_component_surface_node_count": int(winner[3]),
            "runner_up_median_nearest_distance": (
                float(scores[1][0]) if len(scores) > 1 else None
            ),
        }
    return assignments, evidence


def _graph_local_support_ids(
    *,
    seed_sid: str,
    query: np.ndarray,
    graph: dict[str, tuple[str, ...]],
    node_position: dict[str, np.ndarray],
    owner: dict[str, str],
    component_id: str,
    max_support_nodes: int,
    max_graph_hops: int,
) -> tuple[tuple[str, ...], int]:
    seen = {str(seed_sid)}
    frontier = [str(seed_sid)]
    candidates = [str(seed_sid)]
    hops_used = 0
    for hop in range(1, int(max_graph_hops) + 1):
        next_frontier = []
        for sid in sorted(frontier):
            for nb in graph.get(sid, ()):
                if nb in seen or owner.get(nb) != component_id:
                    continue
                seen.add(nb)
                next_frontier.append(nb)
                candidates.append(nb)
        frontier = next_frontier
        hops_used = hop
        if len(candidates) >= int(max_support_nodes) or not frontier:
            break

    # Rank only nodes reached through the structural graph. Euclidean distance
    # chooses among graph-admissible neighbors; it never creates adjacency.
    candidates = sorted(
        set(candidates),
        key=lambda sid: (
            float(np.linalg.norm(node_position[sid] - query)),
            sid,
        ),
    )[: int(max_support_nodes)]
    if not candidates:
        raise QualificationError("TESSA_SUPPORT_GRAPH_LOCAL_NEIGHBORHOOD_EMPTY")
    return tuple(candidates), int(hops_used)


def _convex_weights(
    query: np.ndarray,
    support_ids: tuple[str, ...],
    node_position: dict[str, np.ndarray],
    *,
    inverse_distance_power: float,
) -> tuple[tuple[tuple[str, float], ...], float]:
    points = np.asarray([node_position[sid] for sid in support_ids], dtype=np.float64)
    distances = np.linalg.norm(points - query[None, :], axis=1)
    nearest = float(distances.min())
    exact = np.flatnonzero(distances <= _EPS)
    if len(exact):
        sid = support_ids[int(exact[0])]
        return ((sid, 1.0),), nearest

    power = float(inverse_distance_power)
    if not math.isfinite(power) or power <= 0.0:
        raise QualificationError("TESSA_SUPPORT_INVERSE_DISTANCE_POWER_INVALID")
    weights = np.power(np.maximum(distances, _EPS), -power)
    total = float(weights.sum())
    if not math.isfinite(total) or total <= 0.0:
        raise QualificationError("TESSA_SUPPORT_WEIGHT_NORMALIZATION_INVALID")
    weights = weights / total
    rows = tuple(
        (str(sid), float(weight))
        for sid, weight in sorted(
            zip(support_ids, weights.tolist()), key=lambda row: row[0]
        )
        if float(weight) > 1.0e-14
    )
    if not rows:
        raise QualificationError("TESSA_SUPPORT_WEIGHT_ROWS_EMPTY")
    norm = sum(weight for _, weight in rows)
    if abs(norm - 1.0) > 1.0e-9:
        # Dropping numerical dust must not silently break the simplex.
        if not math.isfinite(norm) or norm <= 0.0:
            raise QualificationError("TESSA_SUPPORT_WEIGHT_SIMPLEX_INVALID")
        rows = tuple((sid, weight / norm) for sid, weight in rows)
    return rows, nearest


def validate_tessa_material_support_field_v1(
    value: TESSAMaterialSupportFieldIR,
    *,
    surface: RiggingSurfaceIR,
    partition: MechanicalPartitionIR,
    expected_vertex_ids: tuple[str, ...] | None = None,
) -> None:
    validate_mechanical_partition(partition, surface)
    if value.surface_binding_hash != surface.geometry_lineage_hash:
        raise QualificationError("TESSA_SUPPORT_SURFACE_LINEAGE_MISMATCH")
    if value.partition_binding_hash != partition.partition_lineage_hash:
        raise QualificationError("TESSA_SUPPORT_PARTITION_LINEAGE_MISMATCH")
    if not value.topology_sequence_hash:
        raise QualificationError("TESSA_SUPPORT_TOPOLOGY_SEQUENCE_HASH_MISSING")

    known = {str(node.surface_id) for node in surface.surface_nodes}
    owner = _owner_map(partition)
    component_ids = {str(component.component_id) for component in partition.components}
    seen = set()
    for row in value.rows:
        if not row.proposal_vertex_id or row.proposal_vertex_id in seen:
            raise QualificationError("TESSA_SUPPORT_VERTEX_ID_INVALID")
        seen.add(row.proposal_vertex_id)
        if row.mechanical_component_id not in component_ids:
            raise QualificationError("TESSA_SUPPORT_COMPONENT_INVALID")
        if not row.coefficients:
            raise QualificationError("TESSA_SUPPORT_COEFFICIENTS_EMPTY")
        total = 0.0
        support_seen = set()
        for sid, weight in row.coefficients:
            if sid in support_seen or sid not in known:
                raise QualificationError("TESSA_SUPPORT_SURFACE_ID_INVALID")
            if owner.get(sid) != row.mechanical_component_id:
                raise QualificationError("TESSA_SUPPORT_CROSS_COMPONENT_COEFFICIENT")
            if not math.isfinite(float(weight)) or float(weight) < 0.0:
                raise QualificationError("TESSA_SUPPORT_COEFFICIENT_INVALID")
            support_seen.add(sid)
            total += float(weight)
        if abs(total - 1.0) > 1.0e-9:
            raise QualificationError("TESSA_SUPPORT_SIMPLEX_INVALID")
        if (
            not math.isfinite(float(row.nearest_surface_distance))
            or float(row.nearest_surface_distance) < 0.0
        ):
            raise QualificationError("TESSA_SUPPORT_DISTANCE_INVALID")
    if expected_vertex_ids is not None and seen != set(map(str, expected_vertex_ids)):
        raise QualificationError("TESSA_SUPPORT_VERTEX_ACCOUNTING_INCOMPLETE")
    if value.field_lineage_hash != tessa_material_support_field_lineage_hash(value):
        raise QualificationError("TESSA_SUPPORT_FIELD_LINEAGE_HASH_MISMATCH")


def build_tessa_material_support_field_v1(
    *,
    vertices_world: np.ndarray,
    decoded_component_indices: np.ndarray,
    surface: RiggingSurfaceIR,
    partition: MechanicalPartitionIR,
    topology_sequence_hash: str,
    proposal_vertex_ids: tuple[str, ...] | None = None,
    max_support_nodes: int = 4,
    max_graph_hops: int = 2,
    inverse_distance_power: float = 2.0,
) -> TESSAMaterialSupportFieldIR:
    """Bind TESSA vertices to a continuous surface/material field.

    The returned coefficients are for material/mechanical field sampling (skin,
    deformation consequence, downstream correspondence). They are NOT a claim
    that the learned vertex position equals the convexly lifted GSA position.
    """
    validate_mechanical_partition(partition, surface)
    vertices = np.asarray(vertices_world, dtype=np.float64)
    components = np.asarray(decoded_component_indices, dtype=np.int64).reshape(-1)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise QualificationError("TESSA_SUPPORT_VERTEX_ARRAY_INVALID")
    if len(vertices) == 0 or len(components) != len(vertices):
        raise QualificationError("TESSA_SUPPORT_COMPONENT_ARRAY_INVALID")
    if int(max_support_nodes) < 1:
        raise QualificationError("TESSA_SUPPORT_MAX_NODES_INVALID")
    if int(max_graph_hops) < 0:
        raise QualificationError("TESSA_SUPPORT_MAX_HOPS_INVALID")
    if not topology_sequence_hash:
        raise QualificationError("TESSA_SUPPORT_TOPOLOGY_SEQUENCE_HASH_MISSING")

    if proposal_vertex_ids is None:
        proposal_vertex_ids = tuple(f"TESSA_V1:{i:08d}" for i in range(len(vertices)))
    if len(proposal_vertex_ids) != len(vertices) or len(set(proposal_vertex_ids)) != len(vertices):
        raise QualificationError("TESSA_SUPPORT_PROPOSAL_VERTEX_IDS_INVALID")

    node_by_id = {str(node.surface_id): node for node in surface.surface_nodes}
    node_position = {
        sid: np.asarray(node.P, dtype=np.float64) for sid, node in node_by_id.items()
    }
    owner = _owner_map(partition)
    graph = _graph(surface)
    component_points, ids_by_component, trees = _component_rows(surface, partition)
    assignments, assignment_evidence = _assign_decoded_components(
        vertices,
        components,
        component_points=component_points,
        component_trees=trees,
    )

    rows = []
    nearest_distances = []
    support_counts = []
    for index, (proposal_id, point, decoded_ci) in enumerate(
        zip(proposal_vertex_ids, vertices, components.tolist())
    ):
        cid = assignments[int(decoded_ci)]
        ids = ids_by_component[cid]
        _, seed_local = trees[cid].query(point, k=1)
        seed_sid = ids[int(seed_local)]
        local_ids, hops_used = _graph_local_support_ids(
            seed_sid=seed_sid,
            query=point,
            graph=graph,
            node_position=node_position,
            owner=owner,
            component_id=cid,
            max_support_nodes=int(max_support_nodes),
            max_graph_hops=int(max_graph_hops),
        )
        coefficients, nearest = _convex_weights(
            point,
            local_ids,
            node_position,
            inverse_distance_power=float(inverse_distance_power),
        )
        nearest_distances.append(float(nearest))
        support_counts.append(len(coefficients))
        rows.append(
            TESSAMaterialSupportRowIR(
                proposal_vertex_id=str(proposal_id),
                decoded_component_index=int(decoded_ci),
                mechanical_component_id=str(cid),
                coefficients=coefficients,
                nearest_surface_distance=float(nearest),
                seed_surface_id=str(seed_sid),
                support_graph_hops=int(hops_used),
                metadata={
                    "authority_class": "MATERIAL_SUPPORT_PROPOSAL",
                    "geometry_position_derived_from_support": False,
                    "teacher_vertex_index_used": False,
                    "surface_field_semantics": "CONTINUOUS_CONVEX_GSA_SUPPORT",
                    "row_index": int(index),
                },
            )
        )

    covered = sorted({row.mechanical_component_id for row in rows})
    all_components = sorted(str(component.component_id) for component in partition.components)
    distances_np = np.asarray(nearest_distances, dtype=np.float64)
    provisional = TESSAMaterialSupportFieldIR(
        rows=tuple(rows),
        surface_binding_hash=str(surface.geometry_lineage_hash),
        partition_binding_hash=str(partition.partition_lineage_hash),
        topology_sequence_hash=str(topology_sequence_hash),
        field_lineage_hash="",
        metadata={
            "authority_class": "MATERIAL_SUPPORT_PROPOSAL",
            "product_geometry_authority_claimed": False,
            "teacher_vertex_index_used": False,
            "binding_method": "DECODED_COMPONENT_TO_STRUCTURAL_COMPONENT__GRAPH_LOCAL_IDW_V1",
            "max_support_nodes": int(max_support_nodes),
            "max_graph_hops": int(max_graph_hops),
            "inverse_distance_power": float(inverse_distance_power),
            "covered_mechanical_component_ids": covered,
            "uncovered_mechanical_component_ids": [
                cid for cid in all_components if cid not in set(covered)
            ],
            "decoded_component_assignment": {
                str(k): assignment_evidence[k] for k in sorted(assignment_evidence)
            },
            "nearest_surface_distance": {
                "min": float(distances_np.min()),
                "median": float(np.median(distances_np)),
                "p95": float(np.percentile(distances_np, 95)),
                "max": float(distances_np.max()),
            },
            "support_count": {
                "min": int(min(support_counts)),
                "median": float(np.median(np.asarray(support_counts))),
                "max": int(max(support_counts)),
            },
        },
    )
    value = replace(
        provisional,
        field_lineage_hash=tessa_material_support_field_lineage_hash(provisional),
    )
    validate_tessa_material_support_field_v1(
        value,
        surface=surface,
        partition=partition,
        expected_vertex_ids=tuple(proposal_vertex_ids),
    )
    return value


def material_support_binding_v1(row: TESSAMaterialSupportRowIR) -> SurfaceSupportBinding:
    """Adapter for consumers that explicitly request material-field coefficients.

    Do not use this adapter as geometric support authority without a separate
    geometry qualification court.
    """
    mode = (
        "IDENTITY_SURFACE_NODE"
        if len(row.coefficients) == 1
        and abs(float(row.coefficients[0][1]) - 1.0) <= 1.0e-12
        else "LOCAL_CONVEX_INTERPOLATION"
    )
    return SurfaceSupportBinding(
        mode=mode,
        coefficients=tuple(row.coefficients),
        metadata={
            "authority_class": "MATERIAL_SUPPORT_ONLY",
            "geometry_position_derived_from_support": False,
            "teacher_vertex_index_used": False,
            "mechanical_component_id": str(row.mechanical_component_id),
            "source": "TESSA_MATERIAL_SUPPORT_FIELD_V1",
        },
    )
