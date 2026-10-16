from __future__ import annotations

from dataclasses import dataclass

from models.tessa.v1.model_v1 import TESSAConfigV1


@dataclass(frozen=True)
class IRISVNextConfigV1:
    """Balanced single-scene configuration for the post-demo IRIS line.

    DINO remains an external frozen 384-D image-token producer.  This config owns
    the trainable shared scene trunk and evidence/structure heads only.
    """

    image_token_dim: int = 384
    camera_feature_dim: int = 13
    scene_dim: int = 512
    camera_embed_dim: int = 128
    plane_size: int = 32
    plane_channels: int = 96
    field_hidden_dim: int = 384
    field_residual_blocks: int = 6
    local_feature_dim: int = 384
    attention_heads: int = 8
    scene_layers: int = 6
    feedforward_multiplier: int = 4
    dropout: float = 0.0

    topology_width: int = 512
    topology_heads: int = 8
    topology_decoder_layers: int = 8
    topology_memory_tokens: int = 256
    topology_refinement_layers: int = 2
    topology_source_width: int = 256

    appearance_descriptor_dim: int = 64
    relation_embedding_dim: int = 64
    maximum_extraction_delta: float = 0.03

    # Training uses a bounded dense FlexiCubes court.  Product inference may use
    # a separately sealed coarse-to-fine resolution policy; this value is not a
    # runtime/mobile mesh-density promise.
    extraction_training_resolution: int = 24
    surface_witness_count: int = 1024
    flexicubes_weight_scale: float = 0.99

    def validate(self) -> None:
        positive = (
            self.image_token_dim,
            self.camera_feature_dim,
            self.scene_dim,
            self.camera_embed_dim,
            self.plane_size,
            self.plane_channels,
            self.field_hidden_dim,
            self.field_residual_blocks,
            self.local_feature_dim,
            self.attention_heads,
            self.scene_layers,
            self.feedforward_multiplier,
            self.topology_width,
            self.topology_heads,
            self.topology_decoder_layers,
            self.topology_memory_tokens,
            self.topology_source_width,
            self.appearance_descriptor_dim,
            self.relation_embedding_dim,
            self.extraction_training_resolution,
            self.surface_witness_count,
        )
        if min(positive) <= 0:
            raise ValueError("IRIS_VNEXT_CONFIG_DIMENSION_INVALID")
        if self.scene_dim % self.attention_heads:
            raise ValueError("IRIS_VNEXT_SCENE_HEAD_WIDTH_MISMATCH")
        if self.topology_width % self.topology_heads:
            raise ValueError("IRIS_VNEXT_TOPOLOGY_HEAD_WIDTH_MISMATCH")
        if self.topology_refinement_layers < 0:
            raise ValueError("IRIS_VNEXT_TOPOLOGY_REFINEMENT_NEGATIVE")
        if not (0.0 <= float(self.dropout) < 1.0):
            raise ValueError("IRIS_VNEXT_DROPOUT_INVALID")
        if not (0.0 < float(self.maximum_extraction_delta) <= 0.25):
            raise ValueError("IRIS_VNEXT_EXTRACTION_DELTA_INVALID")
        if self.extraction_training_resolution < 2:
            raise ValueError("IRIS_VNEXT_EXTRACTION_RESOLUTION_INVALID")
        if not (0.0 < float(self.flexicubes_weight_scale) < 1.0):
            raise ValueError("IRIS_VNEXT_FLEXICUBES_WEIGHT_SCALE_INVALID")

    def tessa_decoder_config(self) -> TESSAConfigV1:
        self.validate()
        return TESSAConfigV1(
            d_model=self.topology_width,
            n_heads=self.topology_heads,
            surface_layers=0,
            decoder_layers=self.topology_decoder_layers,
            mlp_ratio=self.feedforward_multiplier,
            dropout=self.dropout,
            surface_latent_count=self.topology_memory_tokens,
        )
