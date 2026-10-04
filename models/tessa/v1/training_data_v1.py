from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import Iterable

import numpy as np

from .model_v1 import TESSAConfigV1
from .tokenization_v1 import TESSATeacherSequenceV1, encode_adjacent_teacher_mesh_v1


@dataclass(frozen=True)
class TESSATeacherChartV1:
    component_index: int
    chart_index: int
    source_face_indices: tuple[int, ...]
    source_vertex_indices: tuple[int, ...]
    vertices_normalized: np.ndarray
    faces_local: np.ndarray
    sequence: TESSATeacherSequenceV1
    schema_version: str = "RealSaS.TESSATeacherChart.v1"


def _validate_mesh(vertices: np.ndarray, faces: np.ndarray, cfg: TESSAConfigV1) -> tuple[np.ndarray, np.ndarray]:
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    if v.ndim != 2 or v.shape[1] != 3 or not np.isfinite(v).all():
        raise ValueError("TESSA_TEACHER_VERTICES_INVALID")
    if f.ndim != 2 or f.shape[1] != 3 or len(f) == 0:
        raise ValueError("TESSA_TEACHER_FACES_INVALID")
    if np.any(f < 0) or np.any(f >= len(v)):
        raise ValueError("TESSA_TEACHER_FACE_INDEX_INVALID")
    if len(v) > cfg.max_vertices or len(f) > cfg.max_faces:
        raise ValueError("TESSA_TEACHER_MESH_OUT_OF_SCALE_POLICY")
    if np.any(v < -0.5000001) or np.any(v > 0.5000001):
        raise ValueError("TESSA_TEACHER_VERTICES_NOT_NORMALIZED")
    return v, f


def _face_neighbors(faces: np.ndarray) -> tuple[tuple[int, ...], ...]:
    edge_to_faces: dict[tuple[int, int], list[int]] = {}
    for fi, (a, b, c) in enumerate(faces.tolist()):
        for u, w in ((a, b), (b, c), (c, a)):
            edge_to_faces.setdefault(tuple(sorted((int(u), int(w)))), []).append(fi)
    neighbors: list[set[int]] = [set() for _ in range(len(faces))]
    for incident in edge_to_faces.values():
        ordered = sorted(set(map(int, incident)))
        for i in ordered:
            neighbors[i].update(j for j in ordered if j != i)
    return tuple(tuple(sorted(x)) for x in neighbors)


def connected_face_components_v1(faces: np.ndarray) -> tuple[tuple[int, ...], ...]:
    f = np.asarray(faces, dtype=np.int64)
    neighbors = _face_neighbors(f)
    remaining = set(range(len(f)))
    components: list[tuple[int, ...]] = []
    while remaining:
        seed = min(remaining)
        q: deque[int] = deque([seed])
        remaining.remove(seed)
        comp: list[int] = []
        while q:
            fi = q.popleft()
            comp.append(fi)
            for nb in neighbors[fi]:
                if nb in remaining:
                    remaining.remove(nb)
                    q.append(nb)
        components.append(tuple(sorted(comp)))
    return tuple(components)


def deterministic_face_charts_v1(
    faces: np.ndarray,
    *,
    max_faces_per_chart: int,
) -> tuple[tuple[int, int, tuple[int, ...]], ...]:
    """Partition each face-connected component into deterministic bounded BFS charts.

    This is a training serialization/scaling device only. It is not product mesh
    authority and it does not alter the teacher face set.
    """
    if max_faces_per_chart < 1:
        raise ValueError("TESSA_MAX_FACES_PER_CHART_NONPOSITIVE")
    f = np.asarray(faces, dtype=np.int64)
    neighbors = _face_neighbors(f)
    components = connected_face_components_v1(f)
    out: list[tuple[int, int, tuple[int, ...]]] = []
    for ci, comp in enumerate(components):
        unassigned = set(comp)
        chart_index = 0
        while unassigned:
            seed = min(unassigned)
            q: deque[int] = deque([seed])
            queued = {seed}
            chart: list[int] = []
            while q and len(chart) < max_faces_per_chart:
                fi = q.popleft()
                if fi not in unassigned:
                    continue
                unassigned.remove(fi)
                chart.append(fi)
                for nb in neighbors[fi]:
                    if nb in unassigned and nb not in queued:
                        queued.add(nb)
                        q.append(nb)
            out.append((ci, chart_index, tuple(chart)))
            chart_index += 1
    return tuple(out)


def build_teacher_charts_v1(
    vertices_normalized: np.ndarray,
    faces: np.ndarray,
    *,
    cfg: TESSAConfigV1 | None = None,
) -> tuple[TESSATeacherChartV1, ...]:
    cfg = cfg or TESSAConfigV1()
    cfg.scaling_policy()
    v, f = _validate_mesh(vertices_normalized, faces, cfg)
    chart_specs = deterministic_face_charts_v1(f, max_faces_per_chart=cfg.max_faces_per_chart)
    charts: list[TESSATeacherChartV1] = []
    seen_faces: list[int] = []
    for component_index, chart_index, face_ids in chart_specs:
        seen_faces.extend(face_ids)
        face_block = f[np.asarray(face_ids, dtype=np.int64)]
        vertex_ids = tuple(sorted(set(map(int, face_block.reshape(-1).tolist()))))
        remap = {source_vi: local_vi for local_vi, source_vi in enumerate(vertex_ids)}
        local_faces = np.asarray(
            [[remap[int(a)], remap[int(b)], remap[int(c)]] for a, b, c in face_block.tolist()],
            dtype=np.int64,
        )
        local_vertices = v[np.asarray(vertex_ids, dtype=np.int64)]
        sequence = encode_adjacent_teacher_mesh_v1(
            local_vertices,
            local_faces,
            component_id=f"component_{component_index}",
            chart_id=f"component_{component_index}_chart_{chart_index}",
            cfg=cfg,
        )
        charts.append(
            TESSATeacherChartV1(
                component_index=component_index,
                chart_index=chart_index,
                source_face_indices=tuple(face_ids),
                source_vertex_indices=vertex_ids,
                vertices_normalized=local_vertices,
                faces_local=local_faces,
                sequence=sequence,
            )
        )
    if sorted(seen_faces) != list(range(len(f))):
        raise AssertionError("TESSA_TEACHER_CHART_FACE_COVERAGE_DRIFT")
    if len(seen_faces) != len(set(seen_faces)):
        raise AssertionError("TESSA_TEACHER_CHART_FACE_DUPLICATION")
    return tuple(charts)
