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
from models.tessa.v3.model_v3 import TESSAStructuralizerV3
from models.axis.rig_program_model_v6 import (
    AXISRigProgramModelV6,
    AXISSequenceV6,
    RigTokenKindV6,
)
from models.mira.skin_fsq_codec_v6 import MIRASkinFSQCodecV6, MIRASkinFSQCodecConfig

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
        b, v, _, h, w = rgba.shape
        if v != 2 or rgba.shape[2] != 4:
            raise ValueError("RSS_VNEXT_DINO_EXPECTS_TWO_RGBA")
        if h % 14 or w % 14:
            raise ValueError("RSS_VNEXT_DINO_PATCH_DIVISIBILITY")
        rgb = rgba[:, :, :3]
        alpha = rgba[:, :, 3:4]
        matted = rgb * alpha + (1.0 - alpha)
        x = ((matted.reshape(b * 2, 3, h, w) - self.mean) / self.std)
        with torch.no_grad():
            tokens = self.backbone(pixel_values=x).last_hidden_state[:, 1:]
        t = tokens.shape[1]
        if t != (h // 14) * (w // 14):
            raise ValueError("RSS_VNEXT_DINO_PATCH_COUNT_DRIFT")
        xy = _xy_grid(h // 14, w // 14, rgba.device, rgba.dtype)
        return tokens.reshape(b, 2, t, self.expected_dim), xy[None, None].expand(b, 2, -1, -1)


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
        b, v, c, h, w = rgba.shape
        if v != 2 or c != 4 or h % self.patch_size or w % self.patch_size:
            raise ValueError("RSS_VNEXT_TEST_PATCH_INPUT_INVALID")
        x = self.patch(rgba.reshape(b * v, c, h, w))
        hp, wp = x.shape[-2:]
        tokens = x.flatten(2).transpose(1, 2).reshape(b, 2, hp * wp, -1)
        return tokens, _xy_grid(hp, wp, rgba.device, rgba.dtype)[None, None].expand(b, 2, -1, -1)


class RSSvNextAuxiliaryViewLatents(nn.Module):
    """Predicted latent hypotheses, never an additional source-image view."""

    def __init__(self, input_dim: int, count: int, heads: int):
        super().__init__()
        if count not in (6, 8):
            raise ValueError("RSS_VNEXT_AUX_LATENT_COUNT_INVALID")
        self.queries = nn.Parameter(torch.randn(count, input_dim) / math.sqrt(input_dim))
        self.attn = nn.MultiheadAttention(input_dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(input_dim)
        self.uncertainty = nn.Linear(input_dim, 1)

    def forward(self, observed_tokens: Tensor) -> tuple[Tensor, Tensor]:
        b, v, t, d = observed_tokens.shape
        if v != 2:
            raise ValueError("RSS_VNEXT_AUX_REQUIRES_TWO_OBSERVATIONS")
        source = observed_tokens.reshape(b, 2 * t, d)
        q = self.queries[None].expand(b, -1, -1)
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

    def forward(
        self,
        local_features: Tensor,
        vertex_xyz: Tensor,
        axis_hidden: Tensor,
        skin_code_logits: Tensor,
        sequence: AXISSequenceV6,
    ) -> tuple[Tensor, Tensor]:
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
            joint_rows.append(axis_hidden[bi, joint_pos])
            # Prediction at k-1 targets code at k: never use current GT token.
            code_rows.append(skin_code_logits[bi, skin_pos - 1].reshape(j, t, -1))

        max_j = max(x.shape[0] for x in joint_rows)
        weight_rows = []
        mask_rows = []
        vfeat = torch.cat((local_features, vertex_xyz), dim=-1)
        vertex_state = self.codec.vertex_decode(vfeat)
        for bi, (jstate, codes) in enumerate(zip(joint_rows, code_rows)):
            probabilities = torch.softmax(codes.float(), dim=-1)
            expected = probabilities @ self.codebook.float()
            expected = expected.to(dtype=jstate.dtype).flatten(1)
            joint_state = self.codec.joint_decode(jstate) + self.codec.code_decode(expected)
            logits = torch.matmul(vertex_state[bi], joint_state.transpose(0, 1))
            logits = logits / math.sqrt(self.codec.config.hidden_dim)
            logits = logits + self.codec.bias_head(joint_state).flatten()[None, :]
            weights = torch.softmax(logits, dim=-1)
            weight_rows.append(F.pad(weights, (0, max_j - jstate.shape[0])))
            mask_rows.append(torch.arange(max_j, device=jstate.device) < jstate.shape[0])
        return torch.stack(weight_rows), torch.stack(mask_rows)


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
        self.udf = nn.Linear(h, 1)  # always nonnegative, independent of sign authority
        self.sign_logit = nn.Linear(h, 1)  # a proposal, not global-sign qualification
        self.semantic = nn.Linear(h, cfg.region_slots + 1)  # final = UNKNOWN
        self.affordance = nn.Linear(h, cfg.affordance_classes + 1)
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
        if cfg.aux_constructed_view_latents:
            self.aux_views = RSSvNextAuxiliaryViewLatents(
                cfg.iris.image_token_dim, cfg.aux_constructed_view_latents,
                cfg.iris.attention_heads,
            )
            self.aux_to_scene = nn.Linear(cfg.iris.image_token_dim, cfg.iris.plane_channels)

        self.tessa = TESSAStructuralizerV3(
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
        self.register_buffer("_query_grid", self._make_query_grid(cfg.query_grid_side), persistent=False)

    @staticmethod
    def _make_query_grid(side: int) -> Tensor:
        axis = torch.linspace(-1, 1, side)
        xx, yy, zz = torch.meshgrid(axis, axis, axis, indexing="ij")
        return torch.stack((xx, yy, zz), dim=-1).reshape(1, -1, 3)

    def encode_observations(self, rgba: Tensor, camera_features: Tensor) -> tuple[RSSvNextSceneState, Tensor | None, Tensor | None]:
        if rgba.ndim != 5 or rgba.shape[1] != 2 or rgba.shape[2] != 4:
            raise ValueError("RSS_VNEXT_ONLY_TWO_SOURCE_RGBA_VIEWS")
        if camera_features.shape != (rgba.shape[0], 2, self.cfg.iris.camera_feature_dim):
            raise ValueError("RSS_VNEXT_FRONT_BACK_CAMERA_SHAPE")
        if not torch.is_floating_point(rgba) or not bool(torch.isfinite(rgba).all()):
            raise ValueError("RSS_VNEXT_SOURCE_RGBA_NUMERIC")
        if bool((rgba < 0).any() or (rgba > 1).any()):
            raise ValueError("RSS_VNEXT_SOURCE_RGBA_RANGE")
        if not bool(torch.isfinite(camera_features).all()):
            raise ValueError("RSS_VNEXT_CAMERA_NUMERIC")
        tokens, token_xy = self.vision(rgba)
        b = rgba.shape[0]
        if tokens.ndim != 4 or tokens.shape[:2] != (b, 2) or tokens.shape[-1] != self.cfg.iris.image_token_dim:
            raise ValueError("RSS_VNEXT_VISION_TOKEN_CONTRACT")
        if token_xy.shape != (*tokens.shape[:3], 2):
            raise ValueError("RSS_VNEXT_VISION_XY_CONTRACT")
        planes = self.iris.encode_scene(tokens, token_xy, camera_features)
        xyz = self._query_grid.to(planes).expand(b, -1, -1)
        local = self.iris.query(planes, xyz)
        h = local["H"]
        local = {
            **local,
            "unsigned_distance": F.softplus(self.udf(h)).squeeze(-1),
            "sign_logit": self.sign_logit(h).squeeze(-1),
        }
        aux = aux_unc = None
        if self.aux_views is not None:
            aux, aux_unc = self.aux_views(tokens)
        return (
            RSSvNextSceneState(
                observed_rgba=rgba,
                camera_features=camera_features,
                observed_image_tokens=tokens,
                scene_triplanes=planes,
                query_xyz=xyz,
                local_features=h,
                geometry=local,
                semantic_logits=self.semantic(h),
                affordance_logits=self.affordance(h),
                graphics_logits=self.graphics(h),
            ),
            aux, aux_unc,
        )

    def _structural_memory(self, state: RSSvNextSceneState, auxiliary: Tensor | None) -> Tensor:
        planes = state.scene_triplanes
        b, p, c, h, w = planes.shape
        scene_src = planes.permute(0, 1, 3, 4, 2).reshape(b, p * h * w, c)
        if auxiliary is not None:
            scene_src = torch.cat((scene_src, self.aux_to_scene(auxiliary)), dim=1)
        return self.tessa.encode_memory(
            scene_source=scene_src,
            surface_source=state.local_features,
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

    def _control_memory(self, state: RSSvNextSceneState, structural: Tensor) -> Tensor:
        fused, _ = self.control_cross(
            structural, state.local_features, state.local_features, need_weights=False,
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
    ) -> RSSvNextOutput:
        # No source 3D mesh, teacher joints, skin, or extra image view is accepted.
        state, auxiliary, aux_unc = self.encode_observations(rgba, camera_features)
        structural_memory = self._structural_memory(state, auxiliary)
        topology_logits = None
        if topology_prefix is not None:
            topology_logits = self.tessa.decode_tokens(
                memory=structural_memory, input_ids=topology_prefix,
            ).logits

        control_logits = skin_weights = joint_mask = None
        if rig_sequence is not None:
            controls = self.axis(self._control_memory(state, structural_memory), rig_sequence)
            control_logits = controls
            kinds = rig_sequence.kind_ids
            if bool((kinds == int(RigTokenKindV6.SKIN_CODE)).any()):
                skin_weights, joint_mask = self.mira(
                    state.local_features,
                    state.query_xyz + state.geometry["extraction_delta"],
                    controls["hidden"], controls["skin_code_logits"], rig_sequence,
                )

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
        )

    @torch.no_grad()
    def propose_teacher_free(
        self,
        rgba: Tensor,
        camera_features: Tensor,
        *,
        max_topology_tokens: int | None = None,
        generate_controls: bool = False,
    ) -> dict:
        """Never consumes topology/rig teacher tokens; still UNQUALIFIED."""
        if self.training:
            raise RuntimeError("RSS_VNEXT_TEACHER_FREE_REQUIRES_EVAL")
        out = self(rgba, camera_features)
        limit = min(max_topology_tokens or self.cfg.topology_decode_cap, self.cfg.topology_decode_cap)
        if limit < 2:
            raise ValueError("RSS_VNEXT_DECODE_LIMIT_INVALID")
        b = rgba.shape[0]
        ids = torch.full((b, 1), self.cfg.tessa.BOS, device=rgba.device, dtype=torch.long)
        done = torch.zeros(b, device=rgba.device, dtype=torch.bool)
        for _ in range(limit - 1):
            logits = self.tessa.decode_tokens(memory=out.structural_memory, input_ids=ids).logits[:, -1]
            next_ids = logits.argmax(dim=-1)
            next_ids = torch.where(done, self.cfg.tessa.EOS, next_ids)
            ids = torch.cat((ids, next_ids[:, None]), dim=1)
            done |= next_ids == self.cfg.tessa.EOS
            if bool(done.all()):
                break
        result = {
            "state": out.state,
            "structural_token_proposal": ids,
            "continuous_surface_proposal": out.continuous_surface_proposal,
            "qualification_status": "UNQUALIFIED_NEURAL_PROPOSAL",
            "observed_view_count": 2,
            "auxiliary_view_authority": "LEARNED_PREDICTED_ONLY",
        }
        if generate_controls:
            if b != 1:
                raise ValueError("RSS_VNEXT_AXIS_V6_GENERATE_BATCH_ONE_ONLY")
            memory = self._control_memory(out.state, out.structural_memory)
            events = self.axis.generate_events(
                memory, tokens_per_joint=self.cfg.skin_tokens_per_joint,
                max_steps=self.cfg.control_max_seq,
            )
            result["control_event_proposal"] = events
        return result


__all__ = [
    "RSSvNextFrozenDINOv2", "RSSvNextPatchEncoderForTests",
    "RSSvNextAuxiliaryViewLatents", "RSSvNextSkinCoupler",
    "RSSvNextUnifiedCharacterModel",
]
