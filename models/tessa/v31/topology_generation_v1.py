"""Teacher-blind grammar-constrained free-running topology generation for Truth139.

The grammar is public model syntax, not teacher topology. It restricts generation to the canonical
TESSA coordinate/special-token language while leaving geometry/topology choices to the model.
Final decoding is delegated to the existing canonical TESSA sequence codec used by ShapeProposal
v2, so the research evaluator cannot silently diverge from the product mesh boundary.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from models.tessa.v1.model_v1 import TESSAConfigV1
from models.tessa.v1.sequence_codec_v1 import decode_tessa_asset_sequence_v1
from models.tessa.v2.config_v2 import TESSAStructuralizerConfigV2
from .dual_stream_incremental_v31 import DualStreamIncrementalDecoderV31
from .dual_stream_v31 import TESSADualStreamStructuralizerV31


@dataclass(frozen=True)
class GeneratedTopologyMeshV1:
    token_ids: tuple[int, ...]
    vertices_normalized: np.ndarray
    faces: np.ndarray
    vertex_component_indices: tuple[int, ...]
    face_component_indices: tuple[int, ...]
    component_count: int
    restart_count: int
    status: str


def _codec_config(config: TESSAStructuralizerConfigV2) -> TESSAConfigV1:
    return TESSAConfigV1(
        coordinate_bins=config.coordinate_bins,
        max_faces=config.max_faces,
        max_vertices=config.max_vertices,
        local_attention_window=config.local_attention_window,
        query_chunk_size=config.query_chunk_size,
        surface_latent_count=config.surface_memory_tokens,
    )


class TESSAGrammarStateV1:
    """Strict subset of the canonical grammar used by the current one-chart-per-component encoder."""

    def __init__(self, config: TESSAStructuralizerConfigV2):
        config.validate()
        self.cfg = config
        self.phase = "EXPECT_COMPONENT_BEGIN"
        self.coordinate_remaining = 0
        self.components = 0
        self.restarts = 0

    def allowed(self) -> tuple[tuple[int, int] | None, tuple[int, ...]]:
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
            self.restarts += 1
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
            return
        if self.phase == "STRIP_OR_END":
            if 0 <= token < cfg.coordinate_bins:
                # The first coordinate of a strip continuation has just been consumed; exactly two
                # more coordinate tokens must follow before another structural decision is legal.
                self.coordinate_remaining = 2
                self.phase = "COORDINATES"
                return
            if token == cfg.FACE_BREAK:
                self.restarts += 1
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
    masked = torch.full_like(logits, -float("inf"))
    if coordinate_range is not None:
        lo, hi = coordinate_range
        masked[:, lo : hi + 1] = logits[:, lo : hi + 1]
    for token in specials:
        masked[:, int(token)] = logits[:, int(token)]
    if not bool(torch.isfinite(masked).any()):
        raise RuntimeError("TESSA_GENERATION_NO_ALLOWED_TOKEN")
    return int(torch.argmax(masked, dim=-1).item())


def decode_generated_topology_v1(
    token_ids: tuple[int, ...] | list[int],
    *,
    config: TESSAStructuralizerConfigV2,
) -> GeneratedTopologyMeshV1:
    ids = tuple(map(int, token_ids))
    decoded = decode_tessa_asset_sequence_v1(ids, cfg=_codec_config(config))
    return GeneratedTopologyMeshV1(
        token_ids=ids,
        vertices_normalized=np.asarray(decoded.vertices_normalized, dtype=np.float32),
        faces=np.asarray(decoded.faces, dtype=np.int64),
        vertex_component_indices=tuple(map(int, decoded.vertex_component_indices)),
        face_component_indices=tuple(map(int, decoded.face_component_indices)),
        component_count=int(decoded.component_count),
        restart_count=int(decoded.restart_count),
        status="PASS_CANONICAL_TESSA_SEQUENCE_CODEC",
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
