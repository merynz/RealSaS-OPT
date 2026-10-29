"""Teacher-blind grammar-constrained free-running topology generation for Truth139.

The grammar is public model syntax, not teacher topology.  It restricts generation to the canonical
TESSA coordinate/special-token language while leaving all geometry/topology choices to the model.
Generated sequences are independently decoded and validated before any Compiler consequence court.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from models.tessa.v1.tokenization_v1 import dequantize_tessa_xyz_v1
from models.tessa.v2.config_v2 import TESSAStructuralizerConfigV2
from .dual_stream_incremental_v31 import DualStreamIncrementalDecoderV31
from .dual_stream_v31 import TESSADualStreamStructuralizerV31


@dataclass(frozen=True)
class GeneratedTopologyMeshV1:
    token_ids: tuple[int, ...]
    vertices_normalized: np.ndarray
    faces: np.ndarray
    component_count: int
    face_break_count: int
    status: str


class TESSAGrammarStateV1:
    def __init__(self, config: TESSAStructuralizerConfigV2):
        config.validate()
        self.cfg = config
        self.phase = "EXPECT_COMPONENT_BEGIN"
        self.coordinate_remaining = 0
        self.coordinate_mode: str | None = None
        self.components = 0
        self.face_breaks = 0

    def allowed(self) -> tuple[tuple[int, int] | None, tuple[int, ...]]:
        """Return optional inclusive coordinate range and allowed special tokens."""
        if self.phase == "EXPECT_COMPONENT_BEGIN":
            return None, (self.cfg.COMPONENT_BEGIN,)
        if self.phase == "EXPECT_CHART_BEGIN":
            return None, (self.cfg.CHART_BEGIN,)
        if self.phase == "EXPECT_FACE_BREAK":
            return None, (self.cfg.FACE_BREAK,)
        if self.phase == "COORDINATES":
            return (0, self.cfg.coordinate_bins - 1), ()
        if self.phase == "STRIP_OR_END":
            return (0, self.cfg.coordinate_bins - 1), (self.cfg.FACE_BREAK, self.cfg.CHART_END)
        if self.phase == "EXPECT_COMPONENT_END":
            return None, (self.cfg.COMPONENT_END,)
        if self.phase == "NEXT_COMPONENT_OR_EOS":
            return None, (self.cfg.COMPONENT_BEGIN, self.cfg.EOS)
        if self.phase == "DONE":
            return None, ()
        raise AssertionError("TESSA_GENERATION_GRAMMAR_UNKNOWN_PHASE")

    def consume(self, token: int) -> None:
        cfg = self.cfg
        if self.phase == "EXPECT_COMPONENT_BEGIN":
            if token != cfg.COMPONENT_BEGIN:
                raise ValueError("TESSA_GENERATION_EXPECT_COMPONENT_BEGIN")
            self.components += 1
            self.phase = "EXPECT_CHART_BEGIN"
            return
        if self.phase == "EXPECT_CHART_BEGIN":
            if token != cfg.CHART_BEGIN:
                raise ValueError("TESSA_GENERATION_EXPECT_CHART_BEGIN")
            self.phase = "EXPECT_FACE_BREAK"
            return
        if self.phase == "EXPECT_FACE_BREAK":
            if token != cfg.FACE_BREAK:
                raise ValueError("TESSA_GENERATION_EXPECT_FACE_BREAK")
            self.face_breaks += 1
            self.coordinate_mode = "SEED"
            self.coordinate_remaining = 9
            self.phase = "COORDINATES"
            return
        if self.phase == "COORDINATES":
            if not 0 <= token < cfg.coordinate_bins:
                raise ValueError("TESSA_GENERATION_EXPECT_COORDINATE")
            self.coordinate_remaining -= 1
            if self.coordinate_remaining < 0:
                raise AssertionError("TESSA_GENERATION_COORDINATE_UNDERFLOW")
            if self.coordinate_remaining == 0:
                self.phase = "STRIP_OR_END"
                self.coordinate_mode = None
            return
        if self.phase == "STRIP_OR_END":
            if 0 <= token < cfg.coordinate_bins:
                self.coordinate_mode = "CONTINUATION"
                self.coordinate_remaining = 2
                self.phase = "COORDINATES"
                return
            if token == cfg.FACE_BREAK:
                self.face_breaks += 1
                self.coordinate_mode = "SEED"
                self.coordinate_remaining = 9
                self.phase = "COORDINATES"
                return
            if token == cfg.CHART_END:
                self.phase = "EXPECT_COMPONENT_END"
                return
            raise ValueError("TESSA_GENERATION_STRIP_TOKEN_INVALID")
        if self.phase == "EXPECT_COMPONENT_END":
            if token != cfg.COMPONENT_END:
                raise ValueError("TESSA_GENERATION_EXPECT_COMPONENT_END")
            self.phase = "NEXT_COMPONENT_OR_EOS"
            return
        if self.phase == "NEXT_COMPONENT_OR_EOS":
            if token == cfg.COMPONENT_BEGIN:
                self.components += 1
                self.phase = "EXPECT_CHART_BEGIN"
                return
            if token == cfg.EOS:
                self.phase = "DONE"
                return
            raise ValueError("TESSA_GENERATION_EXPECT_COMPONENT_OR_EOS")
        raise ValueError("TESSA_GENERATION_TOKEN_AFTER_DONE")


def _masked_argmax(logits: torch.Tensor, state: TESSAGrammarStateV1) -> int:
    if logits.shape != (1, state.cfg.vocab_size) or not bool(torch.isfinite(logits).all()):
        raise ValueError("TESSA_GENERATION_LOGITS_INVALID")
    coordinate_range, specials = state.allowed()
    mask = torch.full_like(logits, -float("inf"))
    if coordinate_range is not None:
        lo, hi = coordinate_range
        mask[:, lo : hi + 1] = logits[:, lo : hi + 1]
    for token in specials:
        mask[:, int(token)] = logits[:, int(token)]
    if not bool(torch.isfinite(mask).any()):
        raise RuntimeError("TESSA_GENERATION_NO_ALLOWED_TOKEN")
    return int(torch.argmax(mask, dim=-1).item())


def decode_generated_topology_v1(
    token_ids: tuple[int, ...] | list[int],
    *,
    config: TESSAStructuralizerConfigV2,
) -> GeneratedTopologyMeshV1:
    ids = tuple(map(int, token_ids))
    if len(ids) < 2 or ids[0] != config.BOS or ids[-1] != config.EOS:
        raise ValueError("TESSA_GENERATED_SEQUENCE_BOUNDARY_INVALID")
    state = TESSAGrammarStateV1(config)
    vertices: list[np.ndarray] = []
    faces: list[tuple[int, int, int]] = []
    component_vertex_map: dict[tuple[int, int, int], int] | None = None
    trailing: tuple[int, int] | None = None
    coordinate_buffer: list[int] = []
    coordinate_mode: str | None = None

    def vertex_from_q(q: tuple[int, int, int]) -> int:
        nonlocal component_vertex_map
        if component_vertex_map is None:
            raise AssertionError("TESSA_GENERATED_COMPONENT_MAP_MISSING")
        if q in component_vertex_map:
            return component_vertex_map[q]
        xyz = dequantize_tessa_xyz_v1(np.asarray([q], dtype=np.int64), config.coordinate_bins)[0]
        index = len(vertices)
        vertices.append(xyz)
        component_vertex_map[q] = index
        if len(vertices) > config.max_vertices:
            raise ValueError("TESSA_GENERATED_VERTEX_BUDGET_EXCEEDED")
        return index

    for token in ids[1:]:
        prior_phase = state.phase
        if prior_phase == "EXPECT_COMPONENT_BEGIN" or (prior_phase == "NEXT_COMPONENT_OR_EOS" and token == config.COMPONENT_BEGIN):
            component_vertex_map = {}
            trailing = None
        if prior_phase == "EXPECT_FACE_BREAK" or (prior_phase == "STRIP_OR_END" and token == config.FACE_BREAK):
            coordinate_mode = "SEED"
            coordinate_buffer = []
        elif prior_phase == "STRIP_OR_END" and 0 <= token < config.coordinate_bins:
            coordinate_mode = "CONTINUATION"
            coordinate_buffer = [token]
        elif prior_phase == "COORDINATES":
            coordinate_buffer.append(token)

        state.consume(token)

        if prior_phase == "COORDINATES" and state.phase == "STRIP_OR_END":
            if coordinate_mode == "SEED":
                if len(coordinate_buffer) != 9:
                    raise AssertionError("TESSA_GENERATED_SEED_COORDINATE_COUNT")
                qrows = [tuple(coordinate_buffer[i : i + 3]) for i in (0, 3, 6)]
                face = tuple(vertex_from_q(q) for q in qrows)
                if len(set(face)) != 3:
                    raise ValueError("TESSA_GENERATED_DEGENERATE_SEED_FACE")
                faces.append(face)
                trailing = (face[1], face[2])
            elif coordinate_mode == "CONTINUATION":
                if len(coordinate_buffer) != 3 or trailing is None:
                    raise AssertionError("TESSA_GENERATED_CONTINUATION_STATE")
                new_vertex = vertex_from_q(tuple(coordinate_buffer))
                face = (trailing[0], trailing[1], new_vertex)
                if len(set(face)) != 3:
                    raise ValueError("TESSA_GENERATED_DEGENERATE_CONTINUATION_FACE")
                faces.append(face)
                trailing = (trailing[1], new_vertex)
            else:
                raise AssertionError("TESSA_GENERATED_COORDINATE_MODE_MISSING")
            if len(faces) > config.max_faces:
                raise ValueError("TESSA_GENERATED_FACE_BUDGET_EXCEEDED")
            coordinate_buffer = []
            coordinate_mode = None
        if prior_phase == "STRIP_OR_END" and token == config.CHART_END:
            trailing = None

    if state.phase != "DONE" or not faces:
        raise ValueError("TESSA_GENERATED_SEQUENCE_INCOMPLETE_OR_EMPTY")
    face_array = np.asarray(faces, dtype=np.int64)
    keys = [tuple(sorted(map(int, face))) for face in faces]
    if len(set(keys)) != len(keys):
        raise ValueError("TESSA_GENERATED_DUPLICATE_FACE")
    vertex_array = np.asarray(vertices, dtype=np.float32)
    if vertex_array.ndim != 2 or vertex_array.shape[1] != 3 or not np.isfinite(vertex_array).all():
        raise ValueError("TESSA_GENERATED_VERTEX_ARRAY_INVALID")
    return GeneratedTopologyMeshV1(
        token_ids=ids,
        vertices_normalized=vertex_array,
        faces=face_array,
        component_count=state.components,
        face_break_count=state.face_breaks,
        status="PASS_GRAMMAR_AND_MESH_DECODE",
    )


@torch.inference_mode()
def generate_topology_v1(
    *,
    model: TESSADualStreamStructuralizerV31,
    scene_source: torch.Tensor,
    surface_source: torch.Tensor,
    maximum_tokens: int = 131072,
) -> GeneratedTopologyMeshV1:
    if model.training:
        raise ValueError("TESSA_GENERATION_REQUIRES_EVAL_MODEL")
    if int(maximum_tokens) < 8:
        raise ValueError("TESSA_GENERATION_TOKEN_BUDGET_TOO_SMALL")
    memory = model.encode_memory(scene_source=scene_source, surface_source=surface_source)
    decoder = DualStreamIncrementalDecoderV31(model, memory)
    state = TESSAGrammarStateV1(model.config)
    tokens = [model.config.BOS]
    logits = decoder.step(model.config.BOS)
    while len(tokens) < int(maximum_tokens):
        token = _masked_argmax(logits, state)
        tokens.append(token)
        state.consume(token)
        if state.phase == "DONE":
            return decode_generated_topology_v1(tokens, config=model.config)
        logits = decoder.step(token)
    raise RuntimeError("TESSA_GENERATION_TOKEN_BUDGET_EXHAUSTED")
