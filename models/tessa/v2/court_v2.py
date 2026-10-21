from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

import numpy as np

from models.tessa.v1.model_v1 import TESSAConfigV1
from models.tessa.v1.sequence_codec_v1 import decode_tessa_asset_sequence_v1
from .config_v2 import TESSAStructuralizerConfigV2


@dataclass(frozen=True)
class TESSADecodedTopologyCourtV2:
    schema: str
    grammar_valid: bool
    decode_failure: str | None
    vertex_count: int
    face_count: int
    declared_component_count: int
    geometric_component_count: int
    restart_count: int
    unique_edge_count: int
    boundary_edge_count: int
    boundary_loop_count: int
    open_boundary_chain_count: int
    nonmanifold_edge_count: int
    zero_area_face_count: int
    euler_characteristic: int | None
    finite_geometry: bool
    within_scale_policy: bool
    scientific_pass_claimed: bool = False
    training_authorized: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _codec_config(config: TESSAStructuralizerConfigV2) -> TESSAConfigV1:
    config.validate()
    return TESSAConfigV1(
        d_model=config.d_model,
        n_heads=config.n_heads,
        surface_layers=0,
        decoder_layers=config.decoder_layers,
        mlp_ratio=config.mlp_ratio,
        dropout=config.dropout,
        coordinate_bins=config.coordinate_bins,
        surface_latent_count=config.total_conditioning_tokens,
        local_attention_window=config.local_attention_window,
        query_chunk_size=config.query_chunk_size,
        max_faces=config.max_faces,
        max_vertices=config.max_vertices,
    )


def _face_components(faces: np.ndarray) -> int:
    if len(faces) == 0:
        return 0
    edge_to_faces: dict[tuple[int, int], list[int]] = {}
    for fi, (a, b, c) in enumerate(faces.tolist()):
        for u, v in ((a, b), (b, c), (c, a)):
            edge_to_faces.setdefault(tuple(sorted((int(u), int(v)))), []).append(fi)
    adjacency: list[set[int]] = [set() for _ in range(len(faces))]
    for incident in edge_to_faces.values():
        for i in incident:
            adjacency[i].update(j for j in incident if j != i)
    remaining = set(range(len(faces)))
    count = 0
    while remaining:
        count += 1
        stack = [remaining.pop()]
        while stack:
            fi = stack.pop()
            for nb in adjacency[fi]:
                if nb in remaining:
                    remaining.remove(nb)
                    stack.append(nb)
    return count


def _boundary_structure(
    edge_incidence: dict[tuple[int, int], int],
) -> tuple[int, int, int]:
    boundary = [edge for edge, count in edge_incidence.items() if count == 1]
    if not boundary:
        return 0, 0, 0
    graph: dict[int, set[int]] = {}
    for a, b in boundary:
        graph.setdefault(a, set()).add(b)
        graph.setdefault(b, set()).add(a)
    remaining = set(graph)
    loops = 0
    chains = 0
    while remaining:
        seed = remaining.pop()
        stack = [seed]
        vertices = [seed]
        while stack:
            v = stack.pop()
            for nb in graph[v]:
                if nb in remaining:
                    remaining.remove(nb)
                    vertices.append(nb)
                    stack.append(nb)
        if all(len(graph[v]) == 2 for v in vertices):
            loops += 1
        else:
            chains += 1
    return len(boundary), loops, chains


def evaluate_decoded_topology_v2(
    token_ids: Iterable[int] | np.ndarray,
    *,
    config: TESSAStructuralizerConfigV2 = TESSAStructuralizerConfigV2(),
) -> TESSADecodedTopologyCourtV2:
    """Evaluate a *free-running or supplied* topology sequence independently of loss.

    This court never promotes a model and never authorizes training.  It answers
    only whether a token sequence decodes to coherent indexed geometry and emits
    topology diagnostics that can later receive preregistered scientific gates.
    """

    config.validate()
    try:
        decoded = decode_tessa_asset_sequence_v1(
            tuple(map(int, np.asarray(tuple(token_ids), dtype=np.int64).reshape(-1).tolist())),
            cfg=_codec_config(config),
        )
    except Exception as exc:
        return TESSADecodedTopologyCourtV2(
            schema="RealSaS.TESSADecodedTopologyCourt.v2",
            grammar_valid=False,
            decode_failure=f"{type(exc).__name__}:{exc}",
            vertex_count=0,
            face_count=0,
            declared_component_count=0,
            geometric_component_count=0,
            restart_count=0,
            unique_edge_count=0,
            boundary_edge_count=0,
            boundary_loop_count=0,
            open_boundary_chain_count=0,
            nonmanifold_edge_count=0,
            zero_area_face_count=0,
            euler_characteristic=None,
            finite_geometry=False,
            within_scale_policy=False,
        )

    vertices = np.asarray(decoded.vertices_normalized, dtype=np.float64)
    faces = np.asarray(decoded.faces, dtype=np.int64)
    edge_incidence: dict[tuple[int, int], int] = {}
    for a, b, c in faces.tolist():
        for u, v in ((a, b), (b, c), (c, a)):
            key = tuple(sorted((int(u), int(v))))
            edge_incidence[key] = edge_incidence.get(key, 0) + 1

    boundary_edges, boundary_loops, open_chains = _boundary_structure(edge_incidence)
    nonmanifold = sum(1 for count in edge_incidence.values() if count > 2)
    tri = vertices[faces]
    area2 = np.linalg.norm(
        np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]),
        axis=1,
    )
    zero_area = int(np.sum(area2 <= 1.0e-12))
    finite = bool(np.isfinite(vertices).all())
    within = (
        len(vertices) <= config.max_vertices
        and len(faces) <= config.max_faces
        and finite
        and (not len(vertices) or (np.min(vertices) >= -0.500001 and np.max(vertices) <= 0.500001))
    )
    geometric_components = _face_components(faces)
    euler = int(len(vertices) - len(edge_incidence) + len(faces))

    return TESSADecodedTopologyCourtV2(
        schema="RealSaS.TESSADecodedTopologyCourt.v2",
        grammar_valid=True,
        decode_failure=None,
        vertex_count=int(len(vertices)),
        face_count=int(len(faces)),
        declared_component_count=int(decoded.component_count),
        geometric_component_count=int(geometric_components),
        restart_count=int(decoded.restart_count),
        unique_edge_count=int(len(edge_incidence)),
        boundary_edge_count=int(boundary_edges),
        boundary_loop_count=int(boundary_loops),
        open_boundary_chain_count=int(open_chains),
        nonmanifold_edge_count=int(nonmanifold),
        zero_area_face_count=int(zero_area),
        euler_characteristic=euler,
        finite_geometry=finite,
        within_scale_policy=bool(within),
    )
