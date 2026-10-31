from __future__ import annotations

"""Independent, label-invariant face-partition diagnostics.

Component IDs are addresses, not topology truth. Courts must compare the induced
partition of faces rather than requiring producer and court to assign identical
integer labels. This module also keeps three relations distinct:

* vertex-connected: faces may meet only at a point;
* surface-edge-connected: any shared full edge connects faces, including a
  non-manifold edge with incidence > 2;
* manifold-edge-connected: only a two-face edge connects faces.

The relations answer different questions and must not be silently conflated.
"""

from collections import defaultdict
import numpy as np


def _faces(value) -> np.ndarray:
    faces = np.asarray(value, dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1:] != (3,) or not len(faces) or np.any(faces < 0):
        raise ValueError("TOPOLOGY_PARTITION_TRIANGULATED_FACES_REQUIRED")
    return faces


def _components_from_graph(graph: list[set[int]]) -> np.ndarray:
    labels = np.full(len(graph), -1, dtype=np.int32)
    component = 0
    for start in range(len(graph)):
        if labels[start] >= 0:
            continue
        labels[start] = component
        stack = [start]
        while stack:
            current = stack.pop()
            for neighbor in graph[current]:
                if labels[neighbor] < 0:
                    labels[neighbor] = component
                    stack.append(neighbor)
        component += 1
    return labels


def vertex_connected_face_partition(faces) -> np.ndarray:
    faces = _faces(faces)
    vertex_faces: dict[int, list[int]] = defaultdict(list)
    for face_id, tri in enumerate(faces):
        for vertex in tri:
            vertex_faces[int(vertex)].append(face_id)
    graph = [set() for _ in range(len(faces))]
    for rows in vertex_faces.values():
        for i, left in enumerate(rows):
            for right in rows[i + 1 :]:
                graph[left].add(right)
                graph[right].add(left)
    return _components_from_graph(graph)


def edge_connected_face_partition(faces, *, manifold_edges_only: bool = False) -> np.ndarray:
    faces = _faces(faces)
    edge_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
    for face_id, tri in enumerate(faces):
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            edge_faces[tuple(sorted((int(a), int(b))))].append(face_id)
    graph = [set() for _ in range(len(faces))]
    for rows in edge_faces.values():
        if manifold_edges_only and len(rows) != 2:
            continue
        for i, left in enumerate(rows):
            for right in rows[i + 1 :]:
                graph[left].add(right)
                graph[right].add(left)
    return _components_from_graph(graph)


def canonicalize_partition(labels) -> np.ndarray:
    """Relabel by first face occurrence so numeric component IDs carry no meaning."""
    labels = np.asarray(labels, dtype=np.int64)
    if labels.ndim != 1 or np.any(labels < 0):
        raise ValueError("TOPOLOGY_PARTITION_LABEL_LAYOUT")
    mapping: dict[int, int] = {}
    out = np.empty(len(labels), dtype=np.int32)
    next_id = 0
    for face_id, raw in enumerate(labels.tolist()):
        raw = int(raw)
        if raw not in mapping:
            mapping[raw] = next_id
            next_id += 1
        out[face_id] = mapping[raw]
    return out


def partitions_equivalent(left, right) -> bool:
    left = np.asarray(left)
    right = np.asarray(right)
    if left.shape != right.shape:
        return False
    return bool(np.array_equal(canonicalize_partition(left), canonicalize_partition(right)))


def partition_refines(finer, coarser) -> bool:
    """True when every finer component is wholly contained in one coarser component."""
    finer = np.asarray(finer, dtype=np.int64)
    coarser = np.asarray(coarser, dtype=np.int64)
    if finer.shape != coarser.shape or finer.ndim != 1:
        return False
    for label in np.unique(finer):
        parents = np.unique(coarser[finer == label])
        if len(parents) != 1:
            return False
    return True


def topology_partition_diagnostics(faces, producer_face_component_id=None) -> dict:
    faces = _faces(faces)
    vertex = vertex_connected_face_partition(faces)
    surface_edge = edge_connected_face_partition(faces, manifold_edges_only=False)
    manifold_edge = edge_connected_face_partition(faces, manifold_edges_only=True)
    payload = {
        "vertex_connected_component_count": int(len(np.unique(vertex))),
        "surface_edge_connected_component_count": int(len(np.unique(surface_edge))),
        "manifold_edge_connected_component_count": int(len(np.unique(manifold_edge))),
        "vertex_vs_surface_edge_partition_equivalent": partitions_equivalent(vertex, surface_edge),
        "surface_edge_vs_manifold_edge_partition_equivalent": partitions_equivalent(
            surface_edge, manifold_edge
        ),
        "manifold_edge_refines_surface_edge": partition_refines(manifold_edge, surface_edge),
    }
    if producer_face_component_id is not None:
        producer = np.asarray(producer_face_component_id, dtype=np.int64)
        if producer.shape != (len(faces),):
            raise ValueError("TOPOLOGY_PARTITION_PRODUCER_LABEL_LAYOUT")
        payload.update(
            {
                "producer_component_count": int(len(np.unique(producer))),
                "producer_vs_vertex_partition_equivalent": partitions_equivalent(producer, vertex),
                "producer_vs_surface_edge_partition_equivalent": partitions_equivalent(
                    producer, surface_edge
                ),
                "producer_vs_manifold_edge_partition_equivalent": partitions_equivalent(
                    producer, manifold_edge
                ),
                "producer_raw_ids_equal_surface_edge_ids": bool(
                    np.array_equal(producer.astype(np.int32), surface_edge)
                ),
                "producer_raw_ids_equal_manifold_edge_ids": bool(
                    np.array_equal(producer.astype(np.int32), manifold_edge)
                ),
            }
        )
    return payload
