from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TESSAStructuralizerConfigV3:
    """Matched-capacity tri-stream structuralization configuration.

    V3 exists to test whether role-free presentation evidence helps topology
    reconstruction.  It intentionally preserves the V2 decoder width/depth,
    coordinate vocabulary and total 1024 conditioning-token budget.  The budget
    is redistributed across scene, surface/mechanics and presentation streams so
    a later A/B cannot win merely by receiving more conditioning tokens.

    This is source architecture only.  It does not authorize training.
    """

    d_model: int = 1024
    n_heads: int = 16
    decoder_layers: int = 24
    mlp_ratio: int = 4
    dropout: float = 0.0

    coordinate_bins: int = 1024
    max_faces: int = 32768
    max_vertices: int = 32768

    scene_memory_tokens: int = 384
    surface_memory_tokens: int = 384
    presentation_memory_tokens: int = 256
    memory_source_width: int = 512
    memory_refinement_layers: int = 4

    local_attention_window: int = 4096
    query_chunk_size: int = 256
    gradient_checkpointing: bool = True

    @property
    def special_token_count(self) -> int:
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
        return (
            self.scene_memory_tokens
            + self.surface_memory_tokens
            + self.presentation_memory_tokens
        )

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
            self.presentation_memory_tokens,
            self.memory_source_width,
            self.local_attention_window,
            self.query_chunk_size,
        )
        if min(positive) <= 0:
            raise ValueError("TESSA_V3_CONFIG_NONPOSITIVE")
        if self.d_model % self.n_heads:
            raise ValueError("TESSA_V3_D_MODEL_NOT_DIVISIBLE_BY_HEADS")
        if self.memory_refinement_layers < 0:
            raise ValueError("TESSA_V3_MEMORY_REFINEMENT_NEGATIVE")
        if not (0.0 <= float(self.dropout) < 1.0):
            raise ValueError("TESSA_V3_DROPOUT_INVALID")
        if self.coordinate_bins < 128:
            raise ValueError("TESSA_V3_COORDINATE_LATTICE_TOO_SMALL")
        if self.local_attention_window < self.query_chunk_size:
            raise ValueError("TESSA_V3_LOCAL_WINDOW_SMALLER_THAN_QUERY_CHUNK")
        if self.special_token_count != 8:
            raise ValueError("TESSA_V3_V1_GRAMMAR_COMPATIBILITY_DRIFT")
        if self.total_conditioning_tokens != 1024:
            raise ValueError("TESSA_V3_MATCHED_MEMORY_BUDGET_DRIFT")
