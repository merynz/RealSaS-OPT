from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TESSAStructuralizerConfigV2:
    """Reference-capacity TESSA V2 structuralization configuration.

    V2 is intentionally not a reduced diagnostic model.  The decoder width/head
    class follows the public 1k-wide artist-mesh transformer family while
    retaining RealSaS' finer 1024-bin canonical coordinate lattice and fail-
    closed 32k admission ceiling.

    `max_faces`/`max_vertices` are scale-policy ceilings, not claims that a
    trained checkpoint has demonstrated that scale.
    """

    d_model: int = 1024
    n_heads: int = 16
    decoder_layers: int = 24
    mlp_ratio: int = 4
    dropout: float = 0.0

    coordinate_bins: int = 1024
    max_faces: int = 32768
    max_vertices: int = 32768

    scene_memory_tokens: int = 512
    surface_memory_tokens: int = 512
    memory_source_width: int = 512
    memory_refinement_layers: int = 4

    local_attention_window: int = 4096
    query_chunk_size: int = 256
    gradient_checkpointing: bool = True

    @property
    def special_token_count(self) -> int:
        # Preserve the V1 grammar IDs for exact tokenizer/codec A/B until a
        # separately versioned topology grammar proves better.
        return 8

    @property
    def vocab_size(self) -> int:
        return self.coordinate_bins + self.special_token_count

    @property
    def BOS(self) -> int:
        return self.coordinate_bins

    @property
    def COMPONENT_BEGIN(self) -> int:
        return self.coordinate_bins + 1

    @property
    def CHART_BEGIN(self) -> int:
        return self.coordinate_bins + 2

    @property
    def FACE_BREAK(self) -> int:
        return self.coordinate_bins + 3

    @property
    def CHART_END(self) -> int:
        return self.coordinate_bins + 4

    @property
    def COMPONENT_END(self) -> int:
        return self.coordinate_bins + 5

    @property
    def EOS(self) -> int:
        return self.coordinate_bins + 6

    @property
    def PAD(self) -> int:
        return self.coordinate_bins + 7

    @property
    def total_conditioning_tokens(self) -> int:
        return self.scene_memory_tokens + self.surface_memory_tokens

    @property
    def feedforward_dim(self) -> int:
        return self.d_model * self.mlp_ratio

    def validate(self) -> None:
        positive = (
            self.d_model,
            self.n_heads,
            self.decoder_layers,
            self.mlp_ratio,
            self.coordinate_bins,
            self.max_faces,
            self.max_vertices,
            self.scene_memory_tokens,
            self.surface_memory_tokens,
            self.memory_source_width,
            self.local_attention_window,
            self.query_chunk_size,
        )
        if min(positive) <= 0:
            raise ValueError("TESSA_V2_CONFIG_NONPOSITIVE")
        if self.d_model % self.n_heads:
            raise ValueError("TESSA_V2_D_MODEL_NOT_DIVISIBLE_BY_HEADS")
        if self.memory_refinement_layers < 0:
            raise ValueError("TESSA_V2_MEMORY_REFINEMENT_NEGATIVE")
        if not (0.0 <= float(self.dropout) < 1.0):
            raise ValueError("TESSA_V2_DROPOUT_INVALID")
        if self.coordinate_bins < 128:
            raise ValueError("TESSA_V2_COORDINATE_LATTICE_TOO_SMALL")
        if self.local_attention_window < self.query_chunk_size:
            raise ValueError("TESSA_V2_LOCAL_WINDOW_SMALLER_THAN_QUERY_CHUNK")
        if self.special_token_count != 8:
            raise ValueError("TESSA_V2_V1_GRAMMAR_COMPATIBILITY_DRIFT")
