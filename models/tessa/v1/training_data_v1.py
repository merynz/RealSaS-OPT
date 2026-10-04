from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .model_v1 import TESSAConfigV1
from .tokenization_v1 import (
    TESSATeacherSequenceV1,
    encode_adjacent_teacher_mesh_v1,
    quantize_tessa_xyz_v1,
)


@dataclass(frozen=True)
class TESSATeacherAssetSequenceV1:
    token_ids: tuple[int, ...]
    face_count: int
    vertex_count: int
    component_count: int
    restart_count: int
    source_component_face_indices: tuple[tuple[int, ...], ...]
    schema_version: str = "RealSaS.TESSATeacherAssetSequence.v1"


@dataclass(frozen=True)
class TESSATrainingWindowV1:
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    sequence_position_offset: int
    target_input_start: int
    target_input_end: int
    schema_version: str = "RealSaS.TESSATrainingWindow.v1"


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
    if f.ndim != 2 or f.shape[1] != 3:
        raise ValueError("TESSA_COMPONENT_FACE_SHAPE_INVALID")
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


def _local_component_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_ids: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    face_block = faces[np.asarray(face_ids, dtype=np.int64)]
    vertex_ids = tuple(sorted(set(map(int, face_block.reshape(-1).tolist()))))
    remap = {source_vi: local_vi for local_vi, source_vi in enumerate(vertex_ids)}
    local_faces = np.asarray(
        [[remap[int(a)], remap[int(b)], remap[int(c)]] for a, b, c in face_block.tolist()],
        dtype=np.int64,
    )
    local_vertices = vertices[np.asarray(vertex_ids, dtype=np.int64)]
    return local_vertices, local_faces, vertex_ids


def _reject_quantized_vertex_collisions(vertices: np.ndarray, cfg: TESSAConfigV1, component_index: int) -> None:
    q = quantize_tessa_xyz_v1(vertices, cfg.coordinate_bins)
    keys = [tuple(map(int, row)) for row in q.tolist()]
    if len(set(keys)) != len(keys):
        raise ValueError(f"TESSA_TEACHER_QUANTIZED_VERTEX_COLLISION:component={component_index}")


def build_teacher_asset_sequence_v1(
    vertices_normalized: np.ndarray,
    faces: np.ndarray,
    *,
    cfg: TESSAConfigV1 | None = None,
) -> TESSATeacherAssetSequenceV1:
    """Serialize the entire teacher topology without arbitrary chart cuts.

    True face-connected source components receive separate COMPONENT blocks.
    Artificial training windows are created later and never alter connectivity.
    Quantized vertex collisions within one connected component fail closed
    because the V1 coordinate grammar could not represent their identity exactly.
    """
    cfg = cfg or TESSAConfigV1()
    cfg.scaling_policy()
    v, f = _validate_mesh(vertices_normalized, faces, cfg)
    components = connected_face_components_v1(f)
    if not components:
        raise ValueError("TESSA_TEACHER_COMPONENTS_EMPTY")

    tokens: list[int] = [cfg.BOS]
    restart_total = 0
    seen_faces: list[int] = []
    for ci, face_ids in enumerate(components):
        local_v, local_f, _ = _local_component_mesh(v, f, face_ids)
        _reject_quantized_vertex_collisions(local_v, cfg, ci)
        seq = encode_adjacent_teacher_mesh_v1(
            local_v,
            local_f,
            component_id=f"source_component_{ci}",
            chart_id="global_component_sequence",
            cfg=cfg,
        )
        # Strip per-component BOS/EOS only. COMPONENT/CHART delimiters remain
        # explicit in the one global asset sequence.
        tokens.extend(seq.token_ids[1:-1])
        restart_total += int(seq.restart_count)
        seen_faces.extend(face_ids)

    tokens.append(cfg.EOS)
    if sorted(seen_faces) != list(range(len(f))):
        raise AssertionError("TESSA_TEACHER_ASSET_FACE_COVERAGE_DRIFT")
    if len(seen_faces) != len(set(seen_faces)):
        raise AssertionError("TESSA_TEACHER_ASSET_FACE_DUPLICATION")

    return TESSATeacherAssetSequenceV1(
        token_ids=tuple(tokens),
        face_count=int(len(f)),
        vertex_count=int(len(v)),
        component_count=int(len(components)),
        restart_count=int(restart_total),
        source_component_face_indices=tuple(components),
    )


def build_truncated_teacher_windows_v1(
    sequence: TESSATeacherAssetSequenceV1,
    *,
    cfg: TESSAConfigV1 | None = None,
    target_tokens: int = 4096,
) -> tuple[TESSATrainingWindowV1, ...]:
    """Create constant-memory LM windows over one global topology sequence.

    Each target range is scored exactly once. A trailing context prefix from the
    same global sequence is prepended, but its labels are PAD/ignored. Position
    offsets preserve the same wrapped phase as full-sequence inference.
    """
    cfg = cfg or TESSAConfigV1()
    cfg.scaling_policy()
    ids = tuple(map(int, sequence.token_ids))
    if len(ids) < 2 or ids[0] != cfg.BOS or ids[-1] != cfg.EOS:
        raise ValueError("TESSA_TEACHER_ASSET_SEQUENCE_BOUNDARY_INVALID")
    if target_tokens < 1:
        raise ValueError("TESSA_TARGET_TOKENS_NONPOSITIVE")
    input_count = len(ids) - 1
    windows: list[TESSATrainingWindowV1] = []
    scored_positions: list[int] = []
    for target_start in range(0, input_count, int(target_tokens)):
        target_end = min(input_count, target_start + int(target_tokens))
        context_start = max(0, target_start - int(cfg.local_attention_window))
        segment = ids[context_start : target_end + 1]
        input_ids = tuple(segment[:-1])
        labels = list(segment[1:])
        prefix = target_start - context_start
        for i in range(prefix):
            labels[i] = cfg.PAD
        windows.append(
            TESSATrainingWindowV1(
                input_ids=input_ids,
                labels=tuple(labels),
                sequence_position_offset=int(context_start),
                target_input_start=int(target_start),
                target_input_end=int(target_end),
            )
        )
        scored_positions.extend(range(target_start, target_end))
    if scored_positions != list(range(input_count)):
        raise AssertionError("TESSA_TRUNCATED_WINDOW_SCORE_COVERAGE_DRIFT")
    return tuple(windows)


# Retained only as a diagnostic partition helper. Canonical TESSA training does
# not decode independently generated charts because that would sever global
# connectivity at artificial chart boundaries.
def deterministic_face_charts_v1(
    faces: np.ndarray,
    *,
    max_faces_per_chart: int,
) -> tuple[tuple[int, int, tuple[int, ...]], ...]:
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
