from __future__ import annotations

"""RSS.vNext neural execution contracts.

One model, specialized trainable experts. Two observed views only.
This code creates *proposals*. Canonical geometry, controller and drawable
authority remain with the bounded, source-grounded Compiler.
"""
from dataclasses import dataclass, field

import torch

from models.iris.vnext.contracts_v1 import IRISVNextConfigV1
from models.tessa.v3.config_v3 import TESSAStructuralizerConfigV3
from models.axis.rig_program_model_v6 import AXISRigProgramConfigV6
from .semantic_geometry_prior import RSSvNextSemanticRoleSchema


@dataclass(frozen=True)
class RSSvNextConfig:
    iris: IRISVNextConfigV1 = field(default_factory=IRISVNextConfigV1)
    tessa: TESSAStructuralizerConfigV3 = field(default_factory=TESSAStructuralizerConfigV3)

    # This is the actual product observation contract. No external 8-view path.
    observed_views: int = 2
    # Total observations + predictions (NOT count of added images):
    # 0=disabled; 6=>2 source+4 predictions; 8=>2 source+6 predictions.
    constructed_total_views: int = 0
    query_grid_side: int = 8  # geometry probes, not source/teacher mesh vertices
    region_slots: int = 32   # anonymous region hypotheses; unknown explicitly represented
    # Explicit renderer-trainable 2D/3D presentation roles, NOT morphology or rig.
    # Same four role order as StdGEN is the baseline; six requires six real
    # source-renderable role definitions and a new schema ID.
    semantic_role_names: tuple[str, ...] = (
        "secondary_form", "focal_detail", "primary_form", "outer_covering",
    )
    semantic_role_schema_id: str = "RSS.vNext.SemanticPresentationRoles.FourWayResearch.v1"
    semantic_evidence_channels: int = 4  # must agree with above schema
    affordance_classes: int = 9
    graphics_channels: int = 5  # explicit RGB logits (3) + density + uncertainty
    relation_width: int = 256
    control_hidden_dim: int = 384
    control_layers: int = 8
    control_heads: int = 8
    control_max_seq: int = 1024
    max_pair_queries: int = 2048
    skin_tokens_per_joint: int = 4
    skin_fsq_levels: tuple[int, ...] = (5, 5, 5, 5)
    skin_hidden_dim: int = 256
    # Research ablation: old dot product vs per-vertex attention to all
    # predicted causal SkinTokens. Select by qualified unseen data, not hype.
    skin_decoder_mode: str = "DOT_PRODUCT_BASELINE"
    skin_query_heads: int = 8
    skin_query_chunk: int = 1024
    topology_decode_cap: int = 128

    def validate(self) -> None:
        self.iris.validate()
        self.tessa.validate()
        self.semantic_role_schema().validate()
        if self.graphics_channels != 5:
            raise ValueError("RSS_VNEXT_GRAPHICS_FIELD_REQUIRES_FIVE_TYPED_CHANNELS")
        if self.semantic_evidence_channels != len(self.semantic_role_names):
            raise ValueError("RSS_VNEXT_SEMANTIC_ROLE_FIELD_WIDTH_DRIFT")
        if self.observed_views != 2:
            raise ValueError("RSS_VNEXT_REQUIRES_EXACTLY_FRONT_BACK")
        if self.constructed_total_views not in (0, 6, 8):
            raise ValueError("RSS_VNEXT_TOTAL_VIEW_COUNT_INVALID")
        if not (2 <= self.query_grid_side <= 32):
            raise ValueError("RSS_VNEXT_QUERY_GRID_SIZE_INVALID")
        if min(
            self.region_slots, self.semantic_evidence_channels,
            self.affordance_classes, self.graphics_channels,
            self.relation_width, self.control_hidden_dim, self.control_layers,
            self.control_heads, self.control_max_seq, self.max_pair_queries,
            self.skin_tokens_per_joint, self.skin_hidden_dim, self.topology_decode_cap,
        ) <= 0:
            raise ValueError("RSS_VNEXT_DIMENSION_INVALID")
        if self.control_hidden_dim % self.control_heads:
            raise ValueError("RSS_VNEXT_CONTROL_HEAD_DIVISIBILITY")
        if self.skin_decoder_mode not in (
            "DOT_PRODUCT_BASELINE", "POINT_QUERY_CROSS_ATTENTION",
        ):
            raise ValueError("RSS_VNEXT_SKIN_DECODER_MODE_INVALID")
        if (
            self.skin_query_heads < 1 or self.skin_hidden_dim % self.skin_query_heads
            or self.skin_query_chunk < 1 or self.skin_query_chunk > 8192
        ):
            raise ValueError("RSS_VNEXT_SKIN_QUERY_CONFIG_INVALID")
        if any(level < 2 for level in self.skin_fsq_levels):
            raise ValueError("RSS_VNEXT_FSQ_LEVEL_INVALID")
        if self.iris.camera_feature_dim != 13:
            raise ValueError("RSS_VNEXT_CAMERA_CONTRACT_DRIFT")

    def semantic_role_schema(self) -> RSSvNextSemanticRoleSchema:
        return RSSvNextSemanticRoleSchema(
            roles=self.semantic_role_names, schema_id=self.semantic_role_schema_id,
        )

    def axis_config(self) -> AXISRigProgramConfigV6:
        self.validate()
        vocab = 1
        for level in self.skin_fsq_levels:
            vocab *= level
        return AXISRigProgramConfigV6(
            scene_dim=self.tessa.d_model,
            hidden_dim=self.control_hidden_dim,
            layers=self.control_layers,
            heads=self.control_heads,
            max_sequence_length=self.control_max_seq,
            skin_vocab_size=vocab,
        )

    @classmethod
    def tiny_for_test(cls) -> "RSSvNextConfig":
        return cls(
            iris=IRISVNextConfigV1(
                image_token_dim=32, scene_dim=32, camera_embed_dim=16,
                plane_size=2, plane_channels=8, field_hidden_dim=32,
                field_residual_blocks=2, local_feature_dim=32,
                attention_heads=4, scene_layers=1,
                topology_width=32, topology_heads=4,
                topology_decoder_layers=1, topology_memory_tokens=8,
                topology_source_width=16, extraction_training_resolution=4,
                surface_witness_count=16,
            ),
            tessa=TESSAStructuralizerConfigV3(
                d_model=32, n_heads=4, decoder_layers=2,
                coordinate_bins=128, scene_memory_tokens=16,
                surface_memory_tokens=32, presentation_memory_tokens=16,
                conditioning_token_budget=64, memory_source_width=16,
                memory_refinement_layers=0, local_attention_window=128,
                query_chunk_size=16, gradient_checkpointing=False,
            ),
            query_grid_side=2, region_slots=4, semantic_evidence_channels=4, affordance_classes=5,
            graphics_channels=5, relation_width=32,
            control_hidden_dim=32, control_layers=2,
            control_heads=4, control_max_seq=64, skin_hidden_dim=32,
            topology_decode_cap=16,
        )


@dataclass
class RSSvNextSceneState:
    """Learned state and provenance kept together, with no teacher asset fields."""
    observed_rgba: torch.Tensor
    camera_features: torch.Tensor
    observed_image_tokens: torch.Tensor
    conditioned_image_tokens: torch.Tensor  # original source tokens remain accessible
    source_semantic_patch_logits: torch.Tensor  # [B,2,T,K] from image encoder
    source_semantic_log_uncertainty: torch.Tensor
    constructed_camera_features: torch.Tensor | None
    constructed_camera_angles: torch.Tensor | None
    scene_triplanes: torch.Tensor
    query_xyz: torch.Tensor
    local_features: torch.Tensor
    geometry: dict[str, torch.Tensor]
    semantic_logits: torch.Tensor  # slot membership, including UNKNOWN
    semantic_evidence_logits: torch.Tensor  # non-authoritative field estimates
    presentation_role_logits: torch.Tensor  # StdGEN-style K-way 3D semantics
    presentation_role_probabilities: torch.Tensor
    affordance_logits: torch.Tensor
    graphics_logits: torch.Tensor
    region_slot_features: torch.Tensor
    region_relation_tokens: torch.Tensor  # outgoing+incoming graph context per region
    region_relation_logits: torch.Tensor
    region_relation_valid_mask: torch.Tensor
    source_view_authority: tuple[str, str] = ("SOURCE_OBSERVED_FRONT", "SOURCE_OBSERVED_BACK")
    aux_view_authority: str = "LEARNED_PREDICTED_ONLY"


@dataclass
class RSSvNextOutput:
    state: RSSvNextSceneState
    structural_memory: torch.Tensor
    structural_logits: torch.Tensor | None
    continuous_surface_proposal: torch.Tensor
    structural_quality_logits: torch.Tensor
    relation_logits: torch.Tensor | None
    auxiliary_view_latents: torch.Tensor | None
    auxiliary_view_log_uncertainty: torch.Tensor | None
    control_logits: dict[str, torch.Tensor] | None
    skin_weight_proposal: torch.Tensor | None
    skin_joint_valid_mask: torch.Tensor | None
    qualification_status: str = "UNQUALIFIED_NEURAL_PROPOSAL"
    mesh_candidate: object | None = None  # typed RSSvNextStructuralMeshCandidate; no circular import
    dynamic_mesh_candidate: object | None = None  # typed RSSvNextDynamicRefinementOutput
    flexicubes_surface_evidence: object | None = None  # typed noncanonical surface witness
    constructed_views: object | None = None  # predicted RGB/alpha/normal, not observed source


__all__ = [
    "RSSvNextConfig", "RSSvNextSceneState", "RSSvNextOutput",
]
