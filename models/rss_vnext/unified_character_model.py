from __future__ import annotations

"""RSS.vNext: shared neural character computation graph (proposal-only).

All experts read the SAME live scene state. This is not a serial set of
independent scene interpreters. The bounded Compiler, not this model, alone
qualifies canonical mesh, rig, drawable and runtime-control authority.

FRONT/BACK are the ONLY source observations. Auxiliary 6/8 latent hypotheses
are explicitly predicted and are never appended to the observed image bank.
"""

import math
from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from models.iris.vnext.scene_core_v1 import IRISVNextSceneCoreV1
from models.tessa.v31.model_v31 import TESSAStructuralizerV31
from models.axis.rig_program_model_v6 import (
    AXISRigProgramModelV6,
    AXISSequenceV6,
    RigTokenKindV6,
)
from models.mira.skin_fsq_codec_v6 import MIRASkinFSQCodecV6, MIRASkinFSQCodecConfig
from models.tessa.v2.generation_v2 import TESSAGrammarStateV2
from .animation_ready_mesh import RSSvNextAnimationReadyMesh, RSSvNextStructuralMeshCandidate
from .tessa_incremental_decode_v1 import decode_next_token_v1
from .dynamic_refinement import RSSvNextDynamicRefinement, RSSvNextDynamicRefinementConfig
from .semantic_structure import RSSvNextGenericRegionStructure
from .early_semantic_field import RSSvNextEarlySemanticField
from .source_ray_evidence import sample_front_back_source_rays
from .skin_query_field import RSSvNextJointSkinQueryField
from .multiview_construction import (
    canonical_novel_camera_ring, RSSvNextViewFieldRenderer,
)
from .semantic_geometry_prior import (
    RSSvNextSemanticLayerSpec, semantic_equivalent_signed_field,
)
from .flexicubes_surface_evidence import (
    RSSvNextFlexiCubesEvidenceBridge,
    RSSvNextFlexiCubesSurfaceEvidence,
)
from .motion_aware_topology_learning import (
    RSSvNextMotionCourt, RSSvNextTopologyPreferenceConfig,
    RSSvNextTopologyPreferenceResult,
    topology_candidate_log_probs, source_relative_topology_preference_loss,
)

from .contracts import RSSvNextConfig, RSSvNextOutput, RSSvNextSceneState


class RSSvNextFrozenDINOv2(nn.Module):
    """Production-oriented local-only vision token adapter.

    Requires transformers + locally cached pretrained DINOv2-S. It never
    downloads weights or substitutes randomly initialized features. Alpha
    remains intact in the source carrier; only the encoder's input is matted.
    """

    def __init__(self, model_path: str, *, expected_dim: int = 384) -> None:
        super().__init__()
        try:
            from transformers import AutoModel
        except ImportError as exc:
            raise RuntimeError("RSS_VNEXT_TRANSFORMERS_REQUIRED") from exc
        self.backbone = AutoModel.from_pretrained(model_path, local_files_only=True)
        hidden = int(self.backbone.config.hidden_size)
        if hidden != expected_dim:
            raise ValueError("RSS_VNEXT_DINO_HIDDEN_WIDTH_DRIFT")
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None],
            persistent=False,
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None],
            persistent=False,
        )
        self.backbone.requires_grad_(False)
        self.backbone.eval()
        self.expected_dim = expected_dim

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, rgba: Tensor) -> tuple[Tensor, Tensor]:
        if rgba.ndim != 5 or rgba.shape[1:3] != (2,4):
            raise ValueError("RSS_VNEXT_DINO_EXPECTS_TWO_RGBA")
        return self.forward_variable_views(rgba)

    def forward_variable_views(self, rgba: Tensor) -> tuple[Tensor, Tensor]:
        if rgba.ndim != 5 or rgba.shape[1] not in (1,2,6) or rgba.shape[2] != 4:
            raise ValueError("RSS_VNEXT_REALSAS_DINO_EXPECT_ONE_TWO_OR_SIX_RGBA")
        b, v, _, h, w = rgba.shape
        if h % 14 or w % 14:
            raise ValueError("RSS_VNEXT_DINO_PATCH_DIVISIBILITY")
        rgb = rgba[:, :, :3]
        alpha = rgba[:, :, 3:4]
        matted = rgb * alpha + (1.0 - alpha)
        x = ((matted.reshape(b * v, 3, h, w) - self.mean) / self.std)
        with torch.no_grad():
            tokens = self.backbone(pixel_values=x).last_hidden_state[:, 1:]
        t = tokens.shape[1]
        if t != (h // 14) * (w // 14):
            raise ValueError("RSS_VNEXT_DINO_PATCH_COUNT_DRIFT")
        xy = _xy_grid(h // 14, w // 14, rgba.device, rgba.dtype)
        return tokens.reshape(b, v, t, self.expected_dim), xy[None, None].expand(b, v, -1, -1)


def _xy_grid(h: int, w: int, device, dtype) -> Tensor:
    yy = (torch.arange(h, device=device, dtype=dtype) + 0.5) * (2.0 / h) - 1
    xx = (torch.arange(w, device=device, dtype=dtype) + 0.5) * (2.0 / w) - 1
    gy, gx = torch.meshgrid(yy, xx, indexing="ij")
    return torch.stack((gx, gy), dim=-1).reshape(h * w, 2)


class RSSvNextPatchEncoderForTests(nn.Module):
    """Small trainable smoke-test encoder; NOT a validated product backbone."""

    def __init__(self, output_dim: int = 32, patch_size: int = 4):
        super().__init__()
        self.patch = nn.Conv2d(4, output_dim, kernel_size=patch_size, stride=patch_size)
        self.patch_size = patch_size

    def forward(self, rgba: Tensor) -> tuple[Tensor, Tensor]:
        if rgba.ndim != 5 or rgba.shape[1:3] != (2,4):
            raise ValueError("RSS_VNEXT_TEST_PATCH_INPUT_INVALID")
        return self.forward_variable_views(rgba)

    def forward_variable_views(self, rgba: Tensor) -> tuple[Tensor, Tensor]:
        if rgba.ndim != 5 or rgba.shape[1] not in (1,2,6) or rgba.shape[2] != 4:
            raise ValueError("RSS_VNEXT_REALSAS_PATCH_VIEW_COUNT_INVALID")
        b, v, c, h, w = rgba.shape
        if h % self.patch_size or w % self.patch_size:
            raise ValueError("RSS_VNEXT_TEST_PATCH_INPUT_INVALID")
        x = self.patch(rgba.reshape(b * v, c, h, w))
        hp, wp = x.shape[-2:]
        tokens = x.flatten(2).transpose(1, 2).reshape(b, v, hp * wp, -1)
        return tokens, _xy_grid(hp, wp, rgba.device, rgba.dtype)[None, None].expand(b, v, -1, -1)


class RSSvNextAuxiliaryViewLatents(nn.Module):
    """Predicted latent hypotheses, never an additional source-image view."""

    def __init__(self, input_dim: int, count: int, heads: int):
        super().__init__()
        if count not in (4, 6):
            raise ValueError("RSS_VNEXT_AUX_LATENT_COUNT_INVALID")
        self.queries = nn.Parameter(torch.randn(count, input_dim) / math.sqrt(input_dim))
        self.attn = nn.MultiheadAttention(input_dim, heads, batch_first=True)
        self.target_camera_embed = nn.Sequential(
            nn.LayerNorm(13), nn.Linear(13, input_dim),
        )
        self.norm = nn.LayerNorm(input_dim)
        self.uncertainty = nn.Linear(input_dim, 1)

    def forward(
        self, observed_tokens: Tensor, target_cameras: Tensor,
    ) -> tuple[Tensor, Tensor]:
        b, v, t, d = observed_tokens.shape
        if v != 2:
            raise ValueError("RSS_VNEXT_AUX_REQUIRES_TWO_OBSERVATIONS")
        if target_cameras.shape != (b, self.queries.shape[0], 13):
            raise ValueError("RSS_VNEXT_AUX_TARGET_CAMERA_REQUIRED")
        source = observed_tokens.reshape(b, 2 * t, d)
        # Each latent is camera-conditioned. Predicted latents never become
        # observed images or privileged rays.
        q = self.queries[None].expand(b, -1, -1) + self.target_camera_embed(
            target_cameras,
        )
        out, _ = self.attn(q, source, source, need_weights=False)
        latents = self.norm(q + out)
        return latents, self.uncertainty(latents).clamp(-8, 4)


class RSSvNextSkinCoupler(nn.Module):
    """Differentiable causal FSQ expectation coupled to local surface H.

    Joint and SkinCode positions are read from the SAME AXIS causal sequence.
    For predicting a skin code, use logits at the PRECEDING token position:
    consuming the label at its own input position would leak teacher codes.
    """

    def __init__(self, cfg: RSSvNextConfig):
        super().__init__()
        self.cfg = cfg
        self.codec = MIRASkinFSQCodecV6(MIRASkinFSQCodecConfig(
            vertex_feature_dim=cfg.iris.local_feature_dim + 3,
            joint_feature_dim=cfg.control_hidden_dim,
            hidden_dim=cfg.skin_hidden_dim,
            tokens_per_joint=cfg.skin_tokens_per_joint,
            levels=cfg.skin_fsq_levels,
        ))
        vocab = self.codec.quantizer.vocab_size
        with torch.no_grad():
            book = self.codec.quantizer.token_ids_to_codes(torch.arange(vocab))
        self.register_buffer("codebook", book, persistent=False)
        self.spatial_decoder = RSSvNextJointSkinQueryField(
            vertex_feature_dim=cfg.iris.local_feature_dim,
            joint_feature_dim=cfg.control_hidden_dim,
            code_dim=self.codec.config.latent_dim,
            tokens_per_joint=cfg.skin_tokens_per_joint,
            hidden_dim=cfg.skin_hidden_dim,
            heads=cfg.skin_query_heads,
            max_query_chunk=cfg.skin_query_chunk,
        )

    def forward(
        self,
        local_features: Tensor,
        vertex_xyz: Tensor,
        axis_hidden: Tensor,
        skin_code_logits: Tensor,
        sequence: AXISSequenceV6,
        *,
        vertex_normals: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if local_features.shape[:2] != vertex_xyz.shape[:2]:
            raise ValueError("RSS_VNEXT_SKIN_VERTEX_DOMAIN_DRIFT")
        if axis_hidden.shape[:2] != sequence.kind_ids.shape:
            raise ValueError("RSS_VNEXT_SKIN_CONTROL_DOMAIN_DRIFT")
        b, n, _ = local_features.shape
        joint_rows = []
        code_rows = []
        t = self.cfg.skin_tokens_per_joint
        for bi in range(b):
            kinds = sequence.kind_ids[bi]
            valid = sequence.attention_mask[bi].bool()
            joint_pos = torch.where(valid & (kinds == int(RigTokenKindV6.JOINT)))[0]
            skin_pos = torch.where(valid & (kinds == int(RigTokenKindV6.SKIN_CODE)))[0]
            j = int(joint_pos.numel())
            if j < 1 or int(skin_pos.numel()) != j * t or bool((skin_pos == 0).any()):
                raise ValueError("RSS_VNEXT_SKIN_CAUSAL_GRAMMAR_INCOMPLETE")
            if bool((joint_pos == 0).any()):
                raise ValueError("RSS_VNEXT_SKIN_JOINT_PREDICTION_POSITION_INVALID")
            # Predict each joint from the preceding causal state, never the
            # current ground-truth joint embedding.
            joint_rows.append(axis_hidden[bi, joint_pos - 1])
            # Prediction at k-1 targets code at k: never use current GT token.
            code_rows.append(skin_code_logits[bi, skin_pos - 1].reshape(j, t, -1))

        max_j = max(x.shape[0] for x in joint_rows)
        weight_rows = []
        mask_rows = []
        vfeat = torch.cat((local_features, vertex_xyz), dim=-1)
        vertex_state = (
            self.codec.vertex_decode(vfeat)
            if self.cfg.skin_decoder_mode == "DOT_PRODUCT_BASELINE" else None
        )
        for bi, (jstate, codes) in enumerate(zip(joint_rows, code_rows)):
            probabilities = torch.softmax(codes.float(), dim=-1)
            # Expected FSQ codes from *causal logits at previous positions*;
            # no current teacher SKIN_CODE embedding is a decoder input.
            expected = (probabilities @ self.codebook.float()).to(dtype=jstate.dtype)
            if self.cfg.skin_decoder_mode == "POINT_QUERY_CROSS_ATTENTION":
                mask = torch.ones(
                    (1, jstate.shape[0]), device=jstate.device, dtype=torch.bool,
                )
                queried = self.spatial_decoder(
                    local_features[bi:bi+1], vertex_xyz[bi:bi+1],
                    jstate[None],
                    expected.reshape(
                        1, jstate.shape[0], self.cfg.skin_tokens_per_joint,
                        self.codec.config.latent_dim,
                    ),
                    mask,
                    vertex_normals=(
                        None if vertex_normals is None
                        else vertex_normals[bi:bi+1]
                    ),
                )
                weights = queried.weights[0]
            else:
                assert vertex_state is not None
                joint_state = (
                    self.codec.joint_decode(jstate)
                    + self.codec.code_decode(expected.flatten(1))
                )
                logits = torch.matmul(
                    vertex_state[bi], joint_state.transpose(0, 1),
                ) / math.sqrt(self.codec.config.hidden_dim)
                logits = logits + self.codec.bias_head(joint_state).flatten()[None, :]
                weights = torch.softmax(logits, dim=-1)
            weight_rows.append(F.pad(weights, (0, max_j - jstate.shape[0])))
            mask_rows.append(torch.arange(max_j, device=jstate.device) < jstate.shape[0])
        joint_rows_padded = [
            F.pad(x, (0, 0, 0, max_j - x.shape[0])) for x in joint_rows
        ]
        return torch.stack(weight_rows), torch.stack(mask_rows), torch.stack(joint_rows_padded)

    def decode_discrete_skin_v1(
        self,
        local_features: Tensor,
        vertex_xyz: Tensor,
        joint_hidden: Tensor,
        skin_code_ids: Tensor,
        *,
        vertex_normals: Tensor | None = None,
    ) -> Tensor:
        """Replay the EXACT generated FSQ IDs with the embedded MIRA decoder.

        This is a separate executable candidate from the differentiable
        expected-code training path. It is not a parity claim, a source truth
        or canonical Skin authority. The decoded array has exact mesh-vertex
        and causal joint order; the unified checkpoint owns the decoder.
        """
        if local_features.ndim != 3:
            raise ValueError("RSS_VNEXT_DISCRETE_SKIN_VERTEX_RANK")
        b, v, h = local_features.shape
        if (
            vertex_xyz.shape != (b, v, 3)
            or h != self.cfg.iris.local_feature_dim
            or joint_hidden.ndim != 3
            or joint_hidden.shape[0] != b
            or joint_hidden.shape[2] != self.cfg.control_hidden_dim
            or not b or not v or not joint_hidden.shape[1]
        ):
            raise ValueError("RSS_VNEXT_DISCRETE_SKIN_DOMAIN_DRIFT")
        j = joint_hidden.shape[1]
        if (
            skin_code_ids.dtype != torch.long
            or skin_code_ids.shape != (b, j, self.cfg.skin_tokens_per_joint)
        ):
            raise ValueError("RSS_VNEXT_DISCRETE_SKIN_CODE_ADDRESS_DRIFT")
        if vertex_normals is not None and vertex_normals.shape != vertex_xyz.shape:
            raise ValueError("RSS_VNEXT_DISCRETE_SKIN_NORMAL_ADDRESS_DRIFT")
        tensors = (local_features, vertex_xyz, joint_hidden)
        if vertex_normals is not None:
            tensors += (vertex_normals,)
        if (
            len({x.device for x in tensors + (skin_code_ids,)}) != 1
            or len({x.dtype for x in tensors}) != 1
            or not all(bool(torch.isfinite(x).all()) for x in tensors)
        ):
            raise ValueError("RSS_VNEXT_DISCRETE_SKIN_NONFINITE_OR_DEVICE_DRIFT")
        # The quantizer checks token bounds; no clamping, re-encoding or
        # interpolation of discrete IDs is permitted at the Compiler boundary.
        codes = self.codec.quantizer.token_ids_to_codes(skin_code_ids).to(
            dtype=joint_hidden.dtype,
        )
        if self.cfg.skin_decoder_mode == "POINT_QUERY_CROSS_ATTENTION":
            if vertex_normals is None:
                raise ValueError("RSS_VNEXT_DISCRETE_SKIN_INDEXED_NORMALS_REQUIRED")
            mask = torch.ones((b, j), dtype=torch.bool, device=joint_hidden.device)
            weights = self.spatial_decoder(
                local_features, vertex_xyz, joint_hidden, codes, mask,
                vertex_normals=vertex_normals,
            ).weights
        else:
            weights = self.codec.decode_tokens(
                skin_code_ids,
                torch.cat((local_features, vertex_xyz), dim=-1),
                joint_hidden,
            )
        if (
            weights.shape != (b, v, j)
            or not bool(torch.isfinite(weights).all())
            or not bool(torch.allclose(
                weights.sum(dim=-1), torch.ones_like(weights[..., 0]),
                rtol=0.0, atol=1.e-5,
            ))
        ):
            raise RuntimeError("RSS_VNEXT_DISCRETE_SKIN_DECODER_RESULT_INVALID")
        return weights


class RSSvNextUnifiedCharacterModel(nn.Module):
    """One shared carrier and differentiably coupled expert computation graph."""

    architecture_id = "RSS.vNext.UnifiedCharacterCompiler.v1"

    def __init__(self, cfg: RSSvNextConfig, *, vision_encoder: nn.Module):
        super().__init__()
        cfg.validate()
        if vision_encoder is None:
            raise ValueError("RSS_VNEXT_IMAGE_ENCODER_REQUIRED")
        self.cfg = cfg
        self.vision = vision_encoder
        self.iris = IRISVNextSceneCoreV1(cfg.iris)
        h = cfg.iris.local_feature_dim
        # Semantic evidence is inferred BEFORE the 3D triplane is constructed;
        # learned priors act residually on image tokens, not as a replacement.
        self.early_semantics = RSSvNextEarlySemanticField(
            cfg.iris.image_token_dim, h, cfg.semantic_evidence_channels,
        )
        # Predicted normal maps inform 3D scene construction jointly with
        # predicted RGB; zero initialization preserves the legacy baseline.
        self.normal_condition_proj = nn.Linear(
            cfg.iris.image_token_dim, cfg.iris.image_token_dim, bias=False,
        )
        nn.init.zeros_(self.normal_condition_proj.weight)
        self.semantic_evidence_to_memory = nn.Linear(cfg.semantic_evidence_channels, h)
        # Shared semantic probabilities originate in H(x): same spatial
        # carrier for SDF, RGB/graphics, structural mesh and mechanics.
        self.presentation_role_head = nn.Sequential(
            nn.LayerNorm(h), nn.Linear(h, cfg.semantic_role_schema().count),
        )
        self.presentation_roles_to_memory = nn.Linear(
            cfg.semantic_role_schema().count, h, bias=False,
        )
        self.udf = nn.Linear(h, 1)  # always nonnegative, independent of sign authority
        self.sign_logit = nn.Linear(h, 1)  # a proposal, not global-sign qualification
        self.semantic = nn.Linear(h, cfg.region_slots + 1)  # final = UNKNOWN
        self.semantic_structure = RSSvNextGenericRegionStructure(
            h, cfg.region_slots, heads=cfg.iris.attention_heads,
        )
        self.semantic_to_memory = nn.Linear(cfg.region_slots + 1, h)
        # The full directed region graph participates in structural/mechanics
        # conditioning. Pool both INCOMING and OUTGOING relation evidence;
        # the unpooled raw relation logits remain in the common carrier.
        self.region_graph_to_memory = nn.Sequential(
            nn.LayerNorm(6), nn.Linear(6, h), nn.GELU(), nn.Linear(h, h),
        )
        self.affordance = nn.Linear(h, cfg.affordance_classes + 1)
        # Mechanics-first topology: the structural Transformer receives
        # learned deformation-affordance features BEFORE choosing face tokens.
        # These are inferred solely from the same two-image carrier H(x),
        # not from GT joints/skin/poses at student inference.
        self.articulation_prior = nn.Sequential(
            nn.LayerNorm(h), nn.Linear(h, h), nn.GELU(),
            nn.Linear(h, h),
        )
        self.affordance_to_memory = nn.Linear(cfg.affordance_classes + 1, h)
        # RealSaS-inspired queryable appearance field (not observed source truth):
        # RGB logits [0:3], density logit [3], log uncertainty residual [4].
        # The camera-aware view renderer consumes this SAME Graphics field.
        self.graphics = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, cfg.graphics_channels))
        self.presentation = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h), nn.GELU())
        self.graphics_to_memory = nn.Linear(cfg.graphics_channels, h, bias=False)
        self.surface_refine = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h), nn.GELU(), nn.Linear(h, 3))
        self.deformation_quality = nn.Linear(h, 4)
        self.relation = nn.Sequential(
            nn.LayerNorm(4 * h + 3),
            nn.Linear(4 * h + 3, cfg.relation_width),
            nn.GELU(), nn.Linear(cfg.relation_width, 4),
        )

        self.aux_views: Optional[RSSvNextAuxiliaryViewLatents] = None
        if cfg.constructed_total_views:
            self.aux_views = RSSvNextAuxiliaryViewLatents(
                cfg.iris.image_token_dim, cfg.constructed_total_views - 2,
                cfg.iris.attention_heads,
            )
            self.aux_to_scene = nn.Linear(cfg.iris.image_token_dim, cfg.iris.plane_channels)

        self.tessa = TESSAStructuralizerV31(
            scene_input_dim=cfg.iris.plane_channels,
            surface_input_dim=h,
            presentation_input_dim=h,
            config=cfg.tessa,
        )
        # Structural memory queries all raw local H without a reduced 7D carrier.
        self.control_cross = nn.MultiheadAttention(
            cfg.tessa.d_model, cfg.control_heads, kdim=h, vdim=h, batch_first=True,
        )
        self.control_norm = nn.LayerNorm(cfg.tessa.d_model)
        self.axis = AXISRigProgramModelV6(cfg.axis_config())
        self.mira = RSSvNextSkinCoupler(cfg)
        self.mesh_bridge = RSSvNextAnimationReadyMesh(self.iris, cfg.tessa)
        # FlexiCubes is a DIFFERENTIABLE SURFACE WITNESS for the structural
        # Transformer, not its final geometry/topology owner. It reuses IRIS H.
        self.fc_surface_bridge = RSSvNextFlexiCubesEvidenceBridge(
            self.iris, resolution=cfg.iris.extraction_training_resolution,
        )
        self.fc_surface_position = nn.Linear(3, h, bias=False)
        # Deliberate EXPERIMENTAL alternative, not RealSaS's upstream
        # multiclass diffusion. It renders from the SAME 2-source Z/H.
        self.constructed_view_renderer = RSSvNextViewFieldRenderer(
            h, hidden_dim=max(h, 32),
            ray_samples=16 if cfg.iris.scene_layers <= 1 else 48,
            ray_chunk=128,
        )
        self.dynamic_refine = RSSvNextDynamicRefinement(
            RSSvNextDynamicRefinementConfig(
                carrier_dim=cfg.iris.local_feature_dim,
                joint_dim=cfg.control_hidden_dim,
                width=cfg.relation_width,
                layers=3,
            )
        )
        self.register_buffer("_query_grid", self._make_query_grid(cfg.query_grid_side), persistent=False)

    @staticmethod
    def _make_query_grid(side: int) -> Tensor:
        axis = torch.linspace(-1, 1, side)
        xx, yy, zz = torch.meshgrid(axis, axis, axis, indexing="ij")
        return torch.stack((xx, yy, zz), dim=-1).reshape(1, -1, 3)

    def _observed_pixel_tokens_without_camera(self, rgba: Tensor):
        """Encode only original pixels, with no unmeasured camera features."""
        if rgba.ndim != 5 or rgba.shape[1] != 2 or rgba.shape[2] != 4:
            raise ValueError("RSS_VNEXT_ONLY_TWO_SOURCE_RGBA_VIEWS")
        if not torch.is_floating_point(rgba) or not bool(torch.isfinite(rgba).all()):
            raise ValueError("RSS_VNEXT_SOURCE_RGBA_NUMERIC")
        if bool((rgba < 0).any() or (rgba > 1).any()):
            raise ValueError("RSS_VNEXT_SOURCE_RGBA_RANGE")
        tokens, token_xy = self.vision(rgba)
        b = rgba.shape[0]
        if tokens.ndim != 4 or tokens.shape[:2] != (b, 2) or tokens.shape[-1] != self.cfg.iris.image_token_dim:
            raise ValueError("RSS_VNEXT_VISION_TOKEN_CONTRACT")
        if token_xy.shape != (*tokens.shape[:3], 2):
            raise ValueError("RSS_VNEXT_VISION_XY_CONTRACT")
        return tokens, token_xy, self.early_semantics(tokens)

    def _original_source_tokens(self, rgba: Tensor, camera_features: Tensor):
        if rgba.ndim != 5 or rgba.shape[1] != 2 or rgba.shape[2] != 4:
            raise ValueError("RSS_VNEXT_ONLY_TWO_SOURCE_RGBA_VIEWS")
        if not isinstance(camera_features, Tensor) or camera_features.shape != (
            rgba.shape[0], 2, self.cfg.iris.camera_feature_dim,
        ):
            raise ValueError("RSS_VNEXT_FRONT_BACK_CAMERA_SHAPE")
        if not bool(torch.isfinite(camera_features).all()):
            raise ValueError("RSS_VNEXT_CAMERA_NUMERIC")
        return self._observed_pixel_tokens_without_camera(rgba)

    def _assemble_scene_from_planes(
        self, *, rgba: Tensor, camera_features: Tensor | None, source_tokens: Tensor,
        source_early, planes: Tensor, novel_ring=None,
        authority: str = "DIRECT_OBSERVED_TWO_VIEWS",
        generation_checkpoint_sha256: str = "",
        generated_view_count: int = 0,
    ) -> RSSvNextSceneState:
        """Single H(x) construction consumed by training and product inference."""
        b = rgba.shape[0]
        xyz = self._query_grid.to(planes).expand(b, -1, -1)
        local = self.iris.query(planes, xyz)
        h = local["H"]
        local = {
            **local,
            "unsigned_distance": F.softplus(self.udf(h)).squeeze(-1),
            "sign_logit": self.sign_logit(h).squeeze(-1),
        }
        regions = self.semantic_structure(h)
        presentation_logits = self.presentation_role_head(h)
        relation_valid = regions.relation_valid_mask.unsqueeze(-1)
        relation_values = regions.relation_logits.masked_fill(~relation_valid, 0.0)
        outgoing_count = relation_valid.sum(dim=2).clamp_min(1)
        incoming_count = relation_valid.sum(dim=1).clamp_min(1)
        outgoing = relation_values.sum(dim=2) / outgoing_count
        incoming = relation_values.sum(dim=1) / incoming_count
        relation_tokens = (
            regions.region_slots
            + self.region_graph_to_memory(torch.cat((outgoing, incoming), dim=-1))
        )
        return RSSvNextSceneState(
            observed_rgba=rgba,
            camera_features=camera_features,
            source_view_authority=(
                ("SOURCE_OBSERVED_FRONT", "SOURCE_OBSERVED_BACK")
                if camera_features is not None else
                ("SOURCE_OBSERVED_UNCALIBRATED_0", "SOURCE_OBSERVED_UNCALIBRATED_1")
            ),
            source_camera_authority=(
                "ADMITTED_SOURCE_CAMERA_13D" if camera_features is not None
                else "UNKNOWN_NO_ORIGINAL_CAMERA_PROJECTION"
            ),
            observed_image_tokens=source_tokens,
            conditioned_image_tokens=source_early.conditioned_image_tokens,
            source_semantic_patch_logits=source_early.source_patch_logits,
            source_semantic_log_uncertainty=source_early.source_evidence_uncertainty,
            constructed_camera_features=(
                None if novel_ring is None else novel_ring.novel_cameras
            ),
            constructed_camera_angles=(
                None if novel_ring is None else novel_ring.novel_angles_rad
            ),
            scene_triplanes=planes,
            query_xyz=xyz,
            local_features=h,
            geometry=local,
            semantic_logits=self.semantic(h) + regions.membership_logits,
            semantic_evidence_logits=self.early_semantics.query_field(h),
            presentation_role_logits=presentation_logits,
            presentation_role_probabilities=torch.softmax(presentation_logits, dim=-1),
            affordance_logits=self.affordance(h),
            graphics_logits=self.graphics(h),
            region_slot_features=regions.region_slots,
            region_relation_tokens=relation_tokens,
            region_relation_logits=regions.relation_logits,
            region_relation_valid_mask=regions.relation_valid_mask,
            scene_conditioning_authority=authority,
            generating_checkpoint_sha256=generation_checkpoint_sha256,
            generated_view_count=generated_view_count,
        )

    def encode_observations(
        self, rgba: Tensor, camera_features: Tensor,
    ) -> tuple[RSSvNextSceneState, Tensor | None, Tensor | None]:
        tokens, token_xy, early = self._original_source_tokens(rgba, camera_features)
        planes = self.iris.encode_scene(
            early.conditioned_image_tokens, token_xy, camera_features,
        )
        aux = aux_unc = None
        novel_ring = None
        if self.aux_views is not None:
            novel_ring = canonical_novel_camera_ring(
                camera_features, self.cfg.constructed_total_views,
            )
            aux, aux_unc = self.aux_views(tokens, novel_ring.novel_cameras)
        state = self._assemble_scene_from_planes(
            rgba=rgba, camera_features=camera_features,
            source_tokens=tokens, source_early=early,
            planes=planes, novel_ring=novel_ring,
        )
        return state, aux, aux_unc

    def encode_generated_multiview(
        self, observed_rgba: Tensor, original_camera_features: Tensor | None,
        *, generated_rgba: Tensor, generated_cameras: Tensor,
        generation_checkpoint_sha256: str, source_binding_sha256: str,
        generated_normals: Tensor,
    ) -> RSSvNextSceneState:
        """Canonicalized six-view construction, not six original observations.

        The same scene H(x) and expert heads are used on both training and
        inference. No generated pixel overwrites either of the original views.
        """
        if self.aux_views is not None:
            raise ValueError("RSS_VNEXT_TWO_COMPETING_MULTIVIEW_CONSTRUCTORS")
        if original_camera_features is None:
            tokens, _, early = self._observed_pixel_tokens_without_camera(
                observed_rgba,
            )
        else:
            tokens, _, early = self._original_source_tokens(
                observed_rgba, original_camera_features,
            )
        from .realsas_scene_conditioning_v1 import (
            encode_realsas_conditioning_into_shared_H_v1,
            SIX_VIEW_PRIOR,
        )
        generated = encode_realsas_conditioning_into_shared_H_v1(
            self,
            conditioning_rgba=generated_rgba,
            conditioning_cameras=generated_cameras,
            conditioning_normals=generated_normals,
            source_binding_sha256=source_binding_sha256,
            original_observed_view_count=2,
            conditioning_authority=SIX_VIEW_PRIOR,
            generating_checkpoint_sha256=generation_checkpoint_sha256,
        )
        return self._assemble_scene_from_planes(
            rgba=observed_rgba, camera_features=original_camera_features,
            source_tokens=tokens, source_early=early,
            planes=generated.scene_triplanes,
            authority="GENERATED_SIX_VIEW_SHARED_SCENE",
            generation_checkpoint_sha256=generation_checkpoint_sha256,
            generated_view_count=6,
        )

    def construct_novel_views(
        self,
        state: RSSvNextSceneState,
        *,
        height: int,
        width: int,
        method: str,
    ):
        """Explicit non-RealSaS experimental path, never a source observation.

        Source-parity pre-reconstruction multi-view diffusion remains a
        separate research gate. This late Z/H renderer is an ablation only.
        """
        if method != "EXPERIMENTAL_POST_SCENE_H_RENDER":
            raise ValueError("RSS_VNEXT_MV_METHOD_NOT_IMPLEMENTED_OR_UNDECLARED")
        if state.constructed_camera_features is None:
            raise ValueError("RSS_VNEXT_MV_MUST_CONFIGURE_TOTAL_VIEW_COUNT")
        ring = canonical_novel_camera_ring(
            state.camera_features, self.cfg.constructed_total_views,
        )
        if not torch.equal(ring.novel_cameras, state.constructed_camera_features):
            raise ValueError("RSS_VNEXT_MV_CAMERA_STATE_DRIFT")
        return self.constructed_view_renderer(
            iris=self.iris, scene_triplanes=state.scene_triplanes,
            ring=ring, height=height, width=width,
            graphics_field=self.graphics,
        )

    def _structural_memory(
        self, state: RSSvNextSceneState, auxiliary: Tensor | None,
        flexicubes: RSSvNextFlexiCubesSurfaceEvidence | None = None,
    ) -> Tensor:
        planes = state.scene_triplanes
        b, p, c, h, w = planes.shape
        scene_src = planes.permute(0, 1, 3, 4, 2).reshape(b, p * h * w, c)
        if auxiliary is not None:
            scene_src = torch.cat((scene_src, self.aux_to_scene(auxiliary)), dim=1)
        surface_evidence = (
            state.local_features
            + self.articulation_prior(state.local_features)
            + self.affordance_to_memory(state.affordance_logits)
            + self.semantic_to_memory(state.semantic_logits)
            + self.semantic_evidence_to_memory(state.semantic_evidence_logits)
            + self.presentation_roles_to_memory(state.presentation_role_probabilities)
        )
        # Append region+relation tokens in the structural expert's surface
        # stream rather than discarding their identities after classification.
        surface_source = torch.cat(
            (surface_evidence, state.region_relation_tokens), dim=1,
        )
        if flexicubes is not None:
            if surface_source.shape[0] != 1:
                raise ValueError("RSS_VNEXT_FC_BONE_SCENE_BATCH_ONE_ONLY")
            # Cap attention tokens deterministically, without detaching the
            # underlying differentiable 3D surface or its local H(x).
            n = flexicubes.local_features.shape[1]
            cap = min(256, n)
            ix = torch.linspace(
                0, n - 1, cap, device=surface_source.device,
            ).long()
            witness_features = (
                flexicubes.local_features[:, ix]
                + self.fc_surface_position(flexicubes.xyz[:, ix])
                + self.presentation_roles_to_memory(
                    torch.softmax(flexicubes.presentation_role_logits[:, ix], dim=-1),
                )
            )
            surface_source = torch.cat((surface_source, witness_features), dim=1)
        return self.tessa.encode_memory(
            scene_source=scene_src,
            surface_source=surface_source,
            presentation_source=self.presentation(state.local_features) + self.graphics_to_memory(state.graphics_logits),
        )

    def _relation_logits(self, state: RSSvNextSceneState, pair_indices: Tensor | None) -> Tensor | None:
        if pair_indices is None:
            return None
        b, n, d = state.local_features.shape
        if pair_indices.ndim != 3 or pair_indices.shape[0] != b or pair_indices.shape[-1] != 2:
            raise ValueError("RSS_VNEXT_PAIR_INDEX_SHAPE")
        if pair_indices.shape[1] > self.cfg.max_pair_queries:
            raise ValueError("RSS_VNEXT_PAIR_RESOURCE_GUARD")
        if pair_indices.numel() and (int(pair_indices.min()) < 0 or int(pair_indices.max()) >= n):
            raise ValueError("RSS_VNEXT_PAIR_INDEX_RANGE")
        a = torch.gather(state.local_features, 1, pair_indices[:, :, 0:1].expand(-1, -1, d))
        z = torch.gather(state.local_features, 1, pair_indices[:, :, 1:2].expand(-1, -1, d))
        pa = torch.gather(state.query_xyz, 1, pair_indices[:, :, 0:1].expand(-1, -1, 3))
        pb = torch.gather(state.query_xyz, 1, pair_indices[:, :, 1:2].expand(-1, -1, 3))
        return self.relation(torch.cat((a, z, a - z, a * z, pa - pb), dim=-1))

    def _control_memory(
        self, state: RSSvNextSceneState, structural: Tensor,
        mesh: RSSvNextStructuralMeshCandidate | None = None,
    ) -> Tensor:
        # Mesh features remain addressed to the same live scene H(x).
        context = (
            torch.cat((state.local_features, state.region_relation_tokens), dim=1)
            if mesh is None
            else torch.cat((
                state.local_features, state.region_relation_tokens,
                mesh.local_features,
            ), dim=1)
        )
        fused, _ = self.control_cross(
            structural, context, context, need_weights=False,
        )
        return self.control_norm(structural + fused)

    def forward(
        self,
        rgba: Tensor,
        camera_features: Tensor,
        *,
        topology_prefix: Tensor | None = None,
        rig_sequence: AXISSequenceV6 | None = None,
        pair_indices: Tensor | None = None,
        mesh_teacher_tokens: Tensor | None = None,
        flexicubes_extractor=None,
        flexicubes_layer: RSSvNextSemanticLayerSpec | None = None,
    ) -> RSSvNextOutput:
        # No source 3D mesh, teacher joints, skin, or extra image view is accepted.
        state, auxiliary, aux_unc = self.encode_observations(rgba, camera_features)
        return self._forward_from_state(
            state, auxiliary, aux_unc,
            topology_prefix=topology_prefix, rig_sequence=rig_sequence,
            pair_indices=pair_indices, mesh_teacher_tokens=mesh_teacher_tokens,
            flexicubes_extractor=flexicubes_extractor, flexicubes_layer=flexicubes_layer,
        )

    def _forward_from_state(
        self, state: RSSvNextSceneState, auxiliary: Tensor | None, aux_unc: Tensor | None,
        *, topology_prefix: Tensor | None = None, rig_sequence: AXISSequenceV6 | None = None,
        pair_indices: Tensor | None = None, mesh_teacher_tokens: Tensor | None = None,
        flexicubes_extractor=None, flexicubes_layer: RSSvNextSemanticLayerSpec | None = None,
    ) -> RSSvNextOutput:
        surface_witness = None
        if (flexicubes_extractor is None) != (flexicubes_layer is None):
            raise ValueError("RSS_VNEXT_FC_OPERATOR_AND_LAYER_MUST_BE_TOGETHER")
        if flexicubes_extractor is not None:
            surface_witness = self.fc_surface_bridge(
                state.scene_triplanes,
                extractor=flexicubes_extractor,
                role_head=self.presentation_role_head,
                schema=self.cfg.semantic_role_schema(),
                layer=flexicubes_layer,
                field_policy="SIGNED_HYPOTHESIS_ONLY",
            )
        structural_memory = self._structural_memory(
            state, auxiliary, flexicubes=surface_witness,
        )
        topology_logits = None
        if topology_prefix is not None:
            topology_logits = self.tessa.decode_tokens(
                memory=structural_memory, input_ids=topology_prefix,
            ).logits

        mesh = None
        dynamic = None
        if mesh_teacher_tokens is not None:
            # Explicit training-only teacher-forcing: the model inference path
            # neither requires nor accepts privileged source topology.
            if not self.training:
                raise ValueError("RSS_VNEXT_TEACHER_MESH_FORBIDDEN_AT_INFERENCE")
            if state.scene_triplanes.shape[0] != 1:
                raise ValueError("RSS_VNEXT_RAGGED_MESH_BATCH_ONE_ONLY")
            mesh = self.build_mesh_candidate(
                state, mesh_teacher_tokens,
                source_authority="TEACHER_FORCED_TRAINING_ONLY",
            )

        control_logits = skin_weights = joint_mask = None
        if rig_sequence is not None:
            controls = self.axis(self._control_memory(state, structural_memory, mesh), rig_sequence)
            control_logits = controls
            kinds = rig_sequence.kind_ids
            if bool((kinds == int(RigTokenKindV6.SKIN_CODE)).any()):
                feat = state.local_features if mesh is None else mesh.local_features
                xyz = state.query_xyz + state.geometry["extraction_delta"] if mesh is None else mesh.rest_vertices_iris
                skin_weights, joint_mask, joint_states = self.mira(
                    feat, xyz,
                    controls["hidden"], controls["skin_code_logits"], rig_sequence,
                    vertex_normals=None if mesh is None else mesh.normals,
                )
                if mesh is not None:
                    # Same exact mesh vertices and joint order for rest/skin.
                    dynamic = self.dynamic_refine(mesh, skin_weights, joint_states)

        # Continuous scaffold, NOT a canonical mesh (no faces or qualified sign).
        surface = state.query_xyz + state.geometry["extraction_delta"]
        surface = surface + 0.03 * torch.tanh(self.surface_refine(state.local_features))
        return RSSvNextOutput(
            state=state,
            structural_memory=structural_memory,
            structural_logits=topology_logits,
            continuous_surface_proposal=surface,
            structural_quality_logits=self.deformation_quality(state.local_features),
            relation_logits=self._relation_logits(state, pair_indices),
            auxiliary_view_latents=auxiliary,
            auxiliary_view_log_uncertainty=aux_unc,
            control_logits=control_logits,
            skin_weight_proposal=skin_weights,
            skin_joint_valid_mask=joint_mask,
            mesh_candidate=mesh,
            dynamic_mesh_candidate=dynamic,
            flexicubes_surface_evidence=surface_witness,
        )

    def forward_generated_multiview(
        self, observed_rgba: Tensor, original_camera_features: Tensor | None,
        *, generated_rgba: Tensor, generated_cameras: Tensor,
        generated_normals: Tensor, generation_checkpoint_sha256: str,
        source_binding_sha256: str,
    ) -> RSSvNextOutput:
        state = self.encode_generated_multiview(
            observed_rgba, original_camera_features,
            generated_rgba=generated_rgba, generated_cameras=generated_cameras,
            generated_normals=generated_normals,
            generation_checkpoint_sha256=generation_checkpoint_sha256,
            source_binding_sha256=source_binding_sha256,
        )
        return self._forward_from_state(state, None, None)

    def query_semantic_equivalent_field(
        self,
        state: RSSvNextSceneState,
        layer: RSSvNextSemanticLayerSpec,
        *,
        signed_domain_qualified: bool,
    ) -> Tensor:
        """RealSaS-family geometry change under an explicit source role layer.

        This method cannot assert a solid from image evidence. An admission
        court must explicitly permit SDF use. Open/UDF meshes use component
        patch representation and must NOT call this signed-field method.
        """
        return semantic_equivalent_signed_field(
            state.geometry["field"],
            state.presentation_role_logits,
            schema=self.cfg.semantic_role_schema(), layer=layer,
            qualified_signed_domain=signed_domain_qualified,
        )

    def motion_aware_structural_objective(
        self,
        rgba: Tensor,
        camera_features: Tensor,
        *,
        candidate_tokens: Tensor,
        qualified_motion_court: RSSvNextMotionCourt,
        preference_config: RSSvNextTopologyPreferenceConfig = RSSvNextTopologyPreferenceConfig(),
    ) -> RSSvNextTopologyPreferenceResult:
        """Train causal face/vertex token choices using measured motion consequences.

        ALL source joint/skin/pose truth enters only `qualified_motion_court`
        as detached supervised costs. It is never used to form input scene Z.
        No post-hoc mesh edits, no privileged images, and no self-awarded PASS.
        """
        if not self.training:
            raise RuntimeError("RSS_VNEXT_MOTION_OBJECTIVE_TRAIN_ONLY")
        state, aux, _ = self.encode_observations(rgba, camera_features)
        memory = self._structural_memory(state, aux)
        logprobs = topology_candidate_log_probs(
            self.tessa, memory, candidate_tokens, config=self.cfg.tessa,
        )
        return source_relative_topology_preference_loss(
            logprobs, candidate_tokens, qualified_motion_court,
            config=preference_config,
        )

    def build_mesh_candidate(
        self,
        state: RSSvNextSceneState,
        topology_token_ids: Tensor | tuple[int, ...],
        *,
        source_authority: str = "LEARNED_PREDICTED_TOPOLOGY",
    ) -> RSSvNextStructuralMeshCandidate:
        mesh = self.mesh_bridge(
            state.scene_triplanes, topology_token_ids,
            refine=self.surface_refine,
            semantics=lambda local_h: (
                self.semantic(local_h)
                + self.semantic_structure.assign(
                    local_h, state.region_slot_features,
                )
            ),
            affordances=self.affordance, graphics=self.graphics,
            quality=self.deformation_quality, udf=self.udf,
            sign_logit=self.sign_logit,
            role_head=self.presentation_role_head,
            role_schema_id=self.cfg.semantic_role_schema_id,
            source_authority=source_authority,
        )
        # Requery immutable source rays instead of compressing all visual
        # evidence into semantic labels or assigning invented hidden textures.
        # Missing original camera means NO source-point projection:
        # generated virtual cameras cannot be relabeled as source rays.
        if state.camera_features is not None:
            mesh.source_ray_evidence = sample_front_back_source_rays(
                state.observed_rgba, state.camera_features,
                mesh.rest_vertices_iris,
            )
        mesh.validate()
        return mesh

    @torch.no_grad()
    def propose_teacher_free(
        self,
        rgba: Tensor,
        camera_features: Tensor | None,
        *,
        max_topology_tokens: int | None = None,
        generate_controls: bool = False,
        executable_constraint_operators: tuple[str, ...] | None = None,
        generated_rgba: Tensor | None = None,
        generated_cameras: Tensor | None = None,
        generated_normals: Tensor | None = None,
        generation_checkpoint_sha256: str | None = None,
        source_binding_sha256: str | None = None,
    ) -> dict:
        """Free-running constrained topology and mechanical proposal.

        No teacher geometry, rig or additional original views are accepted.
        Invalid/untrained topology remains ABSTAIN; never substitute a fake
        grid mesh to make the pipeline green.
        """
        if self.training:
            raise RuntimeError("RSS_VNEXT_TEACHER_FREE_REQUIRES_EVAL")
        if camera_features is None and generate_controls and (
            generated_rgba is None or generated_cameras is None
            or generated_normals is None or generation_checkpoint_sha256 is None
            or source_binding_sha256 is None
        ):
            raise RuntimeError("RSS_VNEXT_RAW_CONTROLS_REQUIRE_PINNED_GENERATED_SCENE")
        if generate_controls and executable_constraint_operators is None:
            raise ValueError("RSS_VNEXT_RIG_REQUIRES_COMPILER_CAPABILITIES")
        if not generate_controls and executable_constraint_operators is not None:
            raise ValueError("RSS_VNEXT_UNUSED_COMPILER_CAPABILITIES")
        if rgba.shape[0] != 1:
            raise ValueError("RSS_VNEXT_MESH_GENERATION_BATCH_ONE_ONLY")
        supplied = (
            generated_rgba, generated_cameras, generated_normals,
            generation_checkpoint_sha256, source_binding_sha256,
        )
        if any(v is not None for v in supplied):
            if any(v is None for v in supplied):
                raise ValueError("RSS_VNEXT_GENERATED_SCENE_INCOMPLETE")
            out = self.forward_generated_multiview(
                rgba, camera_features,
                generated_rgba=generated_rgba, generated_cameras=generated_cameras,
                generated_normals=generated_normals,
                generation_checkpoint_sha256=generation_checkpoint_sha256,
                source_binding_sha256=source_binding_sha256,
            )
        elif camera_features is None:
            raise RuntimeError("RSS_VNEXT_UNCALIBRATED_SOURCE_REQUIRES_GENERATED_MULTIVIEW")
        elif getattr(self, "construction_requires_generated_multiview", False):
            raise RuntimeError("RSS_VNEXT_CHECKPOINT_REQUIRES_CANONICAL_MULTIVIEW")
        else:
            out = self(rgba, camera_features)
        limit = min(max_topology_tokens or self.cfg.topology_decode_cap,
                    self.cfg.topology_decode_cap)
        if limit < 2:
            raise ValueError("RSS_VNEXT_DECODE_LIMIT_INVALID")
        ids = torch.tensor([[self.cfg.tessa.BOS]], device=rgba.device, dtype=torch.long)
        grammar = TESSAGrammarStateV2()
        failure = None
        incremental = None
        try:
            for _ in range(limit - 1):
                # V3.1 exact rolling causal KV state shares the teacher-forced
                # weights and semantics, avoiding O(prefix) full recomputation
                # at every token. No token enters the cache twice.
                logits_batch, incremental = decode_next_token_v1(
                    self.tessa, memory=out.structural_memory,
                    token=ids[:, -1:],
                    cache=incremental,
                )
                logits = logits_batch[0]
                allowed = grammar.allowed_token_mask(config=self.cfg.tessa, device=ids.device)
                # Existing V2 grammar allows adjacent-coordinate start before
                # the seed triangle; reject that path in this strict runner.
                if grammar.current_component_face_count == 0:
                    allowed[:self.cfg.tessa.coordinate_bins] = False
                if not bool(allowed.any()):
                    failure = "NO_LEGAL_TOPOLOGY_TOKEN"
                    break
                tok = int(logits.float().masked_fill(~allowed, float("-inf")).argmax().item())
                grammar.advance(tok, config=self.cfg.tessa)
                ids = torch.cat((ids, ids.new_tensor([[tok]])), dim=1)
                if grammar.complete:
                    break
        except (ValueError, RuntimeError) as exc:
            failure = f"TOPOLOGY_SEQUENCE_INVALID:{type(exc).__name__}:{exc}"
        mesh = None
        if grammar.complete and failure is None:
            try:
                mesh = self.build_mesh_candidate(out.state, ids[0])
            except (ValueError, RuntimeError) as exc:
                failure = f"TOPOLOGY_COURT_REJECTED:{type(exc).__name__}:{exc}"
        else:
            failure = failure or "TOPOLOGY_SEQUENCE_INCOMPLETE"

        result: dict = {
            "state": out.state,
            "structural_token_proposal": ids,
            "mesh_candidate": mesh,
            "mesh_status": "UNQUALIFIED_MESH_CANDIDATE" if mesh is not None else "ABSTAIN",
            "mesh_failure": failure,
            "qualification_status": "UNQUALIFIED_NEURAL_PROPOSAL",
            "observed_view_count": 2,
            "auxiliary_view_authority": "LEARNED_PREDICTED_ONLY",
        }
        if generate_controls:
            if mesh is None:
                # A legal failure to generate topology is an ABSTAIN receipt,
                # not an exception that discards the source-bound evidence.
                # Never fabricate a placeholder mesh just to decode controls.
                result["control_status"] = "ABSTAIN_NO_VALID_PREDICTED_MESH"
                return result
            memory = self._control_memory(out.state, out.structural_memory, mesh)
            events = self.axis.generate_events(
                memory, tokens_per_joint=self.cfg.skin_tokens_per_joint,
                max_steps=self.cfg.control_max_seq,
                allowed_constraint_operators=executable_constraint_operators,
            )
            result["control_event_proposal"] = events
            result["control_status"] = "UNQUALIFIED_AXISMIRA_CONTROL_CANDIDATE"
            seq = AXISSequenceV6.from_events(
                events, device=memory.device,
            )
            control_out = self.axis(memory, seq)
            if bool((seq.kind_ids == int(RigTokenKindV6.SKIN_CODE)).any()):
                weights, joint_valid, joints = self.mira(
                    mesh.local_features, mesh.rest_vertices_iris,
                    control_out["hidden"], control_out["skin_code_logits"], seq,
                    vertex_normals=mesh.normals,
                )
                result["dynamic_mesh_candidate"] = self.dynamic_refine(mesh, weights, joints)
                result["joint_valid_mask"] = joint_valid
                # SkinTokens-style product IDs must be decoded, not substituted
                # with the differentiable expected-code weights above. Preserve
                # BOTH outcomes and exact AXIS hidden joint states for the
                # Compiler's pinned same-checkpoint replay/parity court.
                n_joints = int(joint_valid[0].sum().item())
                code_positions = seq.kind_ids[0] == int(RigTokenKindV6.SKIN_CODE)
                token_ids = seq.skin_code_ids[0, code_positions].reshape(
                    1, n_joints, self.cfg.skin_tokens_per_joint,
                )
                hidden_joints = joints[:, :n_joints]
                hard_weights = self.mira.decode_discrete_skin_v1(
                    mesh.local_features, mesh.rest_vertices_iris,
                    hidden_joints, token_ids, vertex_normals=mesh.normals,
                )
                if hard_weights.shape != weights.shape:
                    raise RuntimeError("RSS_VNEXT_SKIN_SOFT_DISCRETE_JOINT_DRIFT")
                result["discrete_skin_weight_proposal"] = hard_weights
                result["skin_joint_hidden_states"] = hidden_joints
        return result


__all__ = [
    "RSSvNextFrozenDINOv2", "RSSvNextPatchEncoderForTests",
    "RSSvNextAuxiliaryViewLatents", "RSSvNextSkinCoupler",
    "RSSvNextUnifiedCharacterModel",
]
