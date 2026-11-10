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


@dataclass(frozen=True)
class RSSvNextConfig:
    iris: IRISVNextConfigV1 = field(default_factory=IRISVNextConfigV1)
    tessa: TESSAStructuralizerConfigV3 = field(default_factory=TESSAStructuralizerConfigV3)

    # This is the actual product observation contract. No external 8-view path.
    observed_views: int = 2
    aux_constructed_view_latents: int = 0  # 0, 6, 8: hypotheses, NEVER observed views
    query_grid_side: int = 8  # geometry probes, not source/teacher mesh vertices
    region_slots: int = 32   # anonymous region hypotheses; unknown explicitly represented
    semantic_evidence_channels: int = 8  # independent masked evidence axes, NOT fixed bodypart classes
    affordance_classes: int = 9
    graphics_channels: int = 5
    relation_width: int = 256
    control_hidden_dim: int = 384
    control_layers: int = 8
    control_heads: int = 8
    control_max_seq: int = 1024
    max_pair_queries: int = 2048
    skin_tokens_per_joint: int = 4
    skin_fsq_levels: tuple[int, ...] = (5, 5, 5, 5)
    skin_hidden_dim: int = 256
    topology_decode_cap: int = 128

    def validate(self) -> None:
        self.iris.validate()
        self.tessa.validate()
        if self.observed_views != 2:
            raise ValueError("RSS_VNEXT_REQUIRES_EXACTLY_FRONT_BACK")
        if self.aux_constructed_view_latents not in (0, 6, 8):
            raise ValueError("RSS_VNEXT_AUX_VIEW_COUNT_INVALID")
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
        if any(level < 2 for level in self.skin_fsq_levels):
            raise ValueError("RSS_VNEXT_FSQ_LEVEL_INVALID")
        if self.iris.camera_feature_dim != 13:
            raise ValueError("RSS_VNEXT_CAMERA_CONTRACT_DRIFT")

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
            query_grid_side=2, region_slots=4, semantic_evidence_channels=5, affordance_classes=5,
            graphics_channels=4, relation_width=32,
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
    scene_triplanes: torch.Tensor
    query_xyz: torch.Tensor
    local_features: torch.Tensor
    geometry: dict[str, torch.Tensor]
    semantic_logits: torch.Tensor  # slot membership, including UNKNOWN
    semantic_evidence_logits: torch.Tensor  # independent 3D field evidence axes
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


__all__ = [
    "RSSvNextConfig", "RSSvNextSceneState", "RSSvNextOutput",
]
