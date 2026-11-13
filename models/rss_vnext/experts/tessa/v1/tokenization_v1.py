from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model_v1 import TESSAConfigV1


@dataclass(frozen=True)
class TESSATeacherSequenceV1:
    token_ids: tuple[int, ...]
    face_count: int
    restart_count: int
    component_id: str
    chart_id: str
    schema_version: str = "RealSaS.TESSATeacherSequence.v1"


def quantize_tessa_xyz_v1(xyz: np.ndarray, bins: int) -> np.ndarray:
    """Quantize normalized RealSaS object coordinates in [-0.5,0.5]."""
    x = np.asarray(xyz, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 3 or not np.isfinite(x).all():
        raise ValueError("TESSA_TOKENIZER_XYZ_INVALID")
    if np.any(x < -0.5000001) or np.any(x > 0.5000001):
        raise ValueError("TESSA_TOKENIZER_XYZ_OUT_OF_NORMALIZED_RANGE")
    u = np.clip(x + 0.5, 0.0, 1.0 - np.finfo(np.float64).eps)
    return np.floor(u * int(bins)).astype(np.int64)


def dequantize_tessa_xyz_v1(q: np.ndarray, bins: int) -> np.ndarray:
    q = np.asarray(q, dtype=np.int64)
    if q.ndim != 2 or q.shape[1] != 3 or np.any(q < 0) or np.any(q >= bins):
        raise ValueError("TESSA_TOKENIZER_QUANTIZED_XYZ_INVALID")
    return (q.astype(np.float64) + 0.5) / float(bins) - 0.5


def _emit_xyz(tokens: list[int], qxyz: np.ndarray, vertex_index: int) -> None:
    tokens.extend(map(int, qxyz[int(vertex_index)].tolist()))


def _face_key(face: tuple[int, int, int]) -> tuple[int, int, int]:
    return tuple(sorted(map(int, face)))


def encode_adjacent_teacher_mesh_v1(
    vertices_normalized: np.ndarray,
    faces: np.ndarray,
    *,
    component_id: str,
    chart_id: str,
    cfg: TESSAConfigV1 | None = None,
) -> TESSATeacherSequenceV1:
    """Clean-room adjacent-face serialization for supervised TESSA training.

    A seed face emits 9 coordinate tokens.  While an unvisited face shares the
    current trailing edge, only the new vertex (3 coordinates) is emitted.
    Otherwise FACE_BREAK starts a new strip.  This is an encoding prior, not
    topology authority; decoded proposals must still be support-bound and
    qualified by the Compiler.
    """
    cfg = cfg or TESSAConfigV1()
    cfg.scaling_policy()
    v = np.asarray(vertices_normalized, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    if f.ndim != 2 or f.shape[1] != 3 or len(f) == 0:
        raise ValueError("TESSA_TOKENIZER_FACE_SHAPE_INVALID")
    if np.any(f < 0) or np.any(f >= len(v)):
        raise ValueError("TESSA_TOKENIZER_FACE_INDEX_INVALID")
    if len(f) > cfg.max_faces:
        raise ValueError("TESSA_TOKENIZER_FACE_BUDGET_EXCEEDED")
    if len(v) > cfg.max_vertices:
        raise ValueError("TESSA_TOKENIZER_VERTEX_BUDGET_EXCEEDED")

    q = quantize_tessa_xyz_v1(v, cfg.coordinate_bins)
    faces_t = [tuple(map(int, row)) for row in f.tolist()]
    if len({_face_key(face) for face in faces_t}) != len(faces_t):
        raise ValueError("TESSA_TOKENIZER_DUPLICATE_FACE")

    edge_to_faces: dict[tuple[int, int], list[int]] = {}
    for fi, face in enumerate(faces_t):
        a, b, c = face
        for u, w in ((a, b), (b, c), (c, a)):
            key = tuple(sorted((u, w)))
            edge_to_faces.setdefault(key, []).append(fi)

    unvisited = set(range(len(faces_t)))
    tokens = [cfg.BOS, cfg.COMPONENT_BEGIN, cfg.CHART_BEGIN]
    restart_count = 0
    trailing: tuple[int, int] | None = None

    while unvisited:
        next_fi = None
        new_vertex = None
        if trailing is not None:
            candidates = sorted(edge_to_faces.get(tuple(sorted(trailing)), ()))
            for fi in candidates:
                if fi not in unvisited:
                    continue
                face = faces_t[fi]
                remaining = [x for x in face if x not in trailing]
                if len(remaining) == 1:
                    next_fi = fi
                    new_vertex = int(remaining[0])
                    break

        if next_fi is not None and new_vertex is not None:
            _emit_xyz(tokens, q, new_vertex)
            unvisited.remove(next_fi)
            trailing = (trailing[1], new_vertex)
            continue

        # Deterministic strip restart.  Preserve source face order inside the
        # seed; winding is retained as teacher evidence, not inferred from IDs.
        next_fi = min(unvisited, key=lambda i: (_face_key(faces_t[i]), i))
        face = faces_t[next_fi]
        tokens.append(cfg.FACE_BREAK)
        for vi in face:
            _emit_xyz(tokens, q, vi)
        trailing = (face[1], face[2])
        unvisited.remove(next_fi)
        restart_count += 1

    tokens.extend((cfg.CHART_END, cfg.COMPONENT_END, cfg.EOS))
    return TESSATeacherSequenceV1(
        token_ids=tuple(tokens),
        face_count=len(faces_t),
        restart_count=restart_count,
        component_id=str(component_id),
        chart_id=str(chart_id),
    )


def adjacent_sequence_token_upper_bound_v1(face_count: int, cfg: TESSAConfigV1 | None = None) -> int:
    """Worst case when every face starts a new strip: specials + 10F tokens."""
    cfg = cfg or TESSAConfigV1()
    if face_count < 0 or face_count > cfg.max_faces:
        raise ValueError("TESSA_FACE_COUNT_OUT_OF_POLICY")
    return 6 + 10 * int(face_count)
