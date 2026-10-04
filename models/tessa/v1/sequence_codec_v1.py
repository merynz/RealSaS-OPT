from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model_v1 import TESSAConfigV1
from .tokenization_v1 import dequantize_tessa_xyz_v1


@dataclass(frozen=True)
class TESSADecodedMeshV1:
    vertices_normalized: np.ndarray
    faces: np.ndarray
    vertex_component_indices: tuple[int, ...]
    face_component_indices: tuple[int, ...]
    component_count: int
    restart_count: int
    orientation_qualified: bool = False
    schema_version: str = "RealSaS.TESSADecodedMesh.v1"


def _is_coordinate(token: int, cfg: TESSAConfigV1) -> bool:
    return 0 <= int(token) < cfg.coordinate_bins


def _read_qvertex(tokens: tuple[int, ...], i: int, cfg: TESSAConfigV1) -> tuple[tuple[int, int, int], int]:
    if i + 3 > len(tokens):
        raise ValueError("TESSA_DECODE_TRUNCATED_VERTEX")
    xyz = tuple(map(int, tokens[i : i + 3]))
    if len(xyz) != 3 or not all(_is_coordinate(t, cfg) for t in xyz):
        raise ValueError("TESSA_DECODE_VERTEX_TOKEN_INVALID")
    return xyz, i + 3


def decode_tessa_asset_sequence_v1(
    token_ids: tuple[int, ...] | list[int] | np.ndarray,
    *,
    cfg: TESSAConfigV1 | None = None,
) -> TESSADecodedMeshV1:
    """Decode the strict V1 topology grammar into indexed geometry.

    Coordinate identity is welded only inside one explicit COMPONENT block.
    This avoids the global exact-coordinate merge used by some artist-mesh
    generators, which would be unsafe for RealSaS near-contact/disconnected
    surfaces. Winding is intentionally not claimed by the autoregressive grammar;
    Compiler support/normal evidence must orient and qualify faces later.
    """
    cfg = cfg or TESSAConfigV1()
    cfg.scaling_policy()
    raw = tuple(map(int, np.asarray(token_ids, dtype=np.int64).reshape(-1).tolist()))
    if not raw or raw[0] != cfg.BOS:
        raise ValueError("TESSA_DECODE_BOS_MISSING")
    try:
        eos = raw.index(cfg.EOS, 1)
    except ValueError as exc:
        raise ValueError("TESSA_DECODE_EOS_MISSING") from exc
    trailing = raw[eos + 1 :]
    if any(t != cfg.PAD for t in trailing):
        raise ValueError("TESSA_DECODE_NONPAD_AFTER_EOS")
    tokens = raw[: eos + 1]

    q_vertices: list[tuple[int, int, int]] = []
    faces: list[tuple[int, int, int]] = []
    vertex_components: list[int] = []
    face_components: list[int] = []
    component_count = 0
    restart_count = 0
    i = 1

    while i < len(tokens) - 1:
        if tokens[i] != cfg.COMPONENT_BEGIN:
            raise ValueError("TESSA_DECODE_COMPONENT_BEGIN_EXPECTED")
        component_index = component_count
        component_count += 1
        i += 1
        component_vertex_map: dict[tuple[int, int, int], int] = {}
        component_face_count = 0

        def vertex_index(q: tuple[int, int, int]) -> int:
            if q in component_vertex_map:
                return component_vertex_map[q]
            idx = len(q_vertices)
            component_vertex_map[q] = idx
            q_vertices.append(q)
            vertex_components.append(component_index)
            return idx

        while True:
            if i >= len(tokens) - 1:
                raise ValueError("TESSA_DECODE_COMPONENT_UNTERMINATED")
            if tokens[i] == cfg.COMPONENT_END:
                if component_face_count < 1:
                    raise ValueError("TESSA_DECODE_EMPTY_COMPONENT")
                i += 1
                break
            if tokens[i] != cfg.CHART_BEGIN:
                raise ValueError("TESSA_DECODE_CHART_BEGIN_EXPECTED")
            i += 1
            strip_trailing: tuple[tuple[int, int, int], tuple[int, int, int]] | None = None
            chart_face_count = 0

            while True:
                if i >= len(tokens) - 1:
                    raise ValueError("TESSA_DECODE_CHART_UNTERMINATED")
                token = tokens[i]
                if token == cfg.CHART_END:
                    if chart_face_count < 1:
                        raise ValueError("TESSA_DECODE_EMPTY_CHART")
                    i += 1
                    break
                if token == cfg.FACE_BREAK:
                    i += 1
                    q0, i = _read_qvertex(tokens, i, cfg)
                    q1, i = _read_qvertex(tokens, i, cfg)
                    q2, i = _read_qvertex(tokens, i, cfg)
                    qs = (q0, q1, q2)
                    restart_count += 1
                elif _is_coordinate(token, cfg):
                    if strip_trailing is None:
                        raise ValueError("TESSA_DECODE_ADJACENT_WITHOUT_SEED")
                    q2, i = _read_qvertex(tokens, i, cfg)
                    qs = (strip_trailing[0], strip_trailing[1], q2)
                else:
                    raise ValueError(f"TESSA_DECODE_UNEXPECTED_TOKEN:{token}")

                indices = tuple(vertex_index(q) for q in qs)
                if len(set(indices)) != 3:
                    raise ValueError("TESSA_DECODE_DEGENERATE_FACE_AFTER_COMPONENT_WELD")
                face_key = tuple(sorted(indices))
                if any(tuple(sorted(existing)) == face_key for existing in faces if all(vertex_components[x] == component_index for x in existing)):
                    raise ValueError("TESSA_DECODE_DUPLICATE_FACE")
                faces.append(indices)
                face_components.append(component_index)
                component_face_count += 1
                chart_face_count += 1
                strip_trailing = (qs[1], qs[2])

    if tokens[-1] != cfg.EOS:
        raise ValueError("TESSA_DECODE_EOS_POSITION_INVALID")
    if component_count < 1 or not faces:
        raise ValueError("TESSA_DECODE_EMPTY_ASSET")
    if len(faces) > cfg.max_faces or len(q_vertices) > cfg.max_vertices:
        raise ValueError("TESSA_DECODE_SCALE_POLICY_EXCEEDED")

    q_array = np.asarray(q_vertices, dtype=np.int64)
    vertices = dequantize_tessa_xyz_v1(q_array, cfg.coordinate_bins)
    return TESSADecodedMeshV1(
        vertices_normalized=vertices,
        faces=np.asarray(faces, dtype=np.int64),
        vertex_component_indices=tuple(vertex_components),
        face_component_indices=tuple(face_components),
        component_count=int(component_count),
        restart_count=int(restart_count),
        orientation_qualified=False,
    )
