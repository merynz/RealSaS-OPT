from __future__ import annotations

"""RSS.vNext unified neural training core — structural and motion pilot.

This is a real optimizer/gradient/checkpoint path, NOT the final Truth Foundry
schema. It consumes only admitted supervision, with explicit masks. Its
current task scope is structural CE + source-motion preference. Other heads
are not implicitly admitted or declared training-ready.
"""
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile

import torch
from torch import Tensor, nn

from models.tessa.v2.objective_v2 import (
    TESSAObjectiveWeightsV2, topology_sequence_objective_v2,
)
from .contracts import RSSvNextConfig
from .motion_aware_topology_learning import (
    RSSvNextMotionCourt, RSSvNextTopologyPreferenceConfig,
    topology_candidate_log_probs,
    source_relative_topology_preference_loss,
)
from .flexicubes_animation_supervision import (
    RSSvNextFCAnimationTeacher,
    RSSvNextFCAnimationLossConfig,
    source_animation_flexicubes_loss,
)
from .unified_character_model import RSSvNextUnifiedCharacterModel
from .early_semantic_field import masked_source_semantic_evidence_loss
from .semantic_geometry_prior import RSSvNextSemanticLayerSpec
from models.iris.vnext.flexicubes_v1 import symmetric_surface_chamfer_v1


@dataclass
class RSSvNextStructuralTrainingBatch:
    rgba: Tensor
    camera_features: Tensor
    topology_teacher_tokens: Tensor | None = None
    topology_valid_mask: Tensor | None = None
    motion_candidate_tokens: Tensor | None = None
    motion_court: RSSvNextMotionCourt | None = None
    # Provisional trainer-level evidence, NOT a frozen Truth Foundry schema.
    semantic_patch_target: Tensor | None = None  # [B,2,T,K]
    semantic_patch_mask: Tensor | None = None    # [B,2,T,K], per-axis qualified
    semantic_qualification_key: str = ""
    # Interim signed-surface training witness; not final foundry supervision.
    # Requires qualified source, mesh coordinate/camera bind and sample proof.
    flexicubes_extractor: object | None = None
    flexicubes_layer: RSSvNextSemanticLayerSpec | None = None
    flexicubes_reference_surface_xyz: Tensor | None = None  # [1,M,3]
    flexicubes_animation_teacher: RSSvNextFCAnimationTeacher | None = None
    flexicubes_signed_source_admission_key: str = ""
    source_qualification_key: str = ""
    admission_status: str = ""


@dataclass(frozen=True)
class RSSvNextTrainingWeights:
    topology: float = 1.0
    motion_preference: float = 0.2
    source_semantics: float = 0.2
    flexicubes_surface: float = 0.3
    flexicubes_animation: float = 0.0  # explicit opt-in on admitted source motion
    clip_gradient_norm: float = 1.0

    def validate(self):
        if min(
            self.topology, self.motion_preference, self.source_semantics,
            self.flexicubes_surface, self.flexicubes_animation,
        ) < 0:
            raise ValueError("RSS_VNEXT_TRAIN_NEGATIVE_OBJECTIVE_WEIGHT")
        if (
            self.topology + self.motion_preference
            + self.source_semantics + self.flexicubes_surface
            + self.flexicubes_animation <= 0
            or self.clip_gradient_norm <= 0
        ):
            raise ValueError("RSS_VNEXT_TRAIN_WEIGHT_POLICY_INVALID")


@dataclass
class RSSvNextTrainingResult:
    total_loss: Tensor
    topology_loss: Tensor | None
    motion_preference_loss: Tensor | None
    source_semantic_loss: Tensor | None
    flexicubes_surface_loss: Tensor | None
    flexicubes_animation_loss: Tensor | None
    eligible_animation_pose_count: int
    eligible_semantic_patch_labels: int
    eligible_topology_examples: int
    qualified_preference_pairs: int
    gradient_norm: Tensor | None = None


def architecture_digest(cfg: RSSvNextConfig) -> str:
    """Hash relevant model-shape config; mismatched checkpoint must not resume."""
    from dataclasses import asdict
    value = json.dumps(
        asdict(cfg), sort_keys=True, separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf8")
    return hashlib.sha256(value).hexdigest()


class RSSvNextStructuralTrainer:
    """Training-engine first slice: CE + consequence-driven discrete topology.

    One shared 2-view scene forward; the two objectives read the same latent
    memory and backpropagate through the same encoder. Teacher tokens and G3
    verdicts never enter image tokens or scene forward.
    """

    checkpoint_schema = "RealSaS.RSSvNext.StructuralTrainerCheckpoint.v1"

    def __init__(
        self,
        model: RSSvNextUnifiedCharacterModel,
        optimizer: torch.optim.Optimizer,
        *,
        dataset_manifest_sha256: str,
        weights: RSSvNextTrainingWeights = RSSvNextTrainingWeights(),
        preference: RSSvNextTopologyPreferenceConfig = RSSvNextTopologyPreferenceConfig(),
    ):
        if not dataset_manifest_sha256 or len(dataset_manifest_sha256) != 64:
            raise ValueError("RSS_VNEXT_TRAIN_DATASET_MANIFEST_SHA_REQUIRED")
        try:
            int(dataset_manifest_sha256, 16)
        except ValueError as exc:
            raise ValueError("RSS_VNEXT_TRAIN_DATASET_MANIFEST_NOT_HEX") from exc
        weights.validate()
        preference.validate()
        self.model = model
        self.optimizer = optimizer
        self.weights = weights
        self.preference = preference
        self.dataset_manifest_sha256 = dataset_manifest_sha256.lower()
        self.global_step = 0

    def objective(self, batch: RSSvNextStructuralTrainingBatch) -> RSSvNextTrainingResult:
        if batch.admission_status != "QUALIFIED_FOR_THIS_SUPERVISION_FAMILY":
            raise ValueError("RSS_VNEXT_TRAIN_ADMISSION_NOT_QUALIFIED")
        if not batch.source_qualification_key:
            raise ValueError("RSS_VNEXT_TRAIN_SOURCE_QUALIFICATION_KEY_MISSING")
        if not self.model.training:
            raise RuntimeError("RSS_VNEXT_TRAIN_REQUIRES_TRAIN_MODE")

        state, aux, _ = self.model.encode_observations(
            batch.rgba, batch.camera_features,
        )
        fc_surface = None
        fc_loss = None
        fc_anim_loss = None
        fc_anim_count = 0
        if any(x is not None for x in (
            batch.flexicubes_extractor, batch.flexicubes_layer,
            batch.flexicubes_reference_surface_xyz,
            batch.flexicubes_animation_teacher,
        )):
            if (
                not batch.flexicubes_signed_source_admission_key
                or batch.flexicubes_extractor is None
                or batch.flexicubes_layer is None
                or (
                    batch.flexicubes_reference_surface_xyz is None
                    and batch.flexicubes_animation_teacher is None
                )
            ):
                raise ValueError("RSS_VNEXT_FC_TRAIN_SIGNED_SOURCE_PROOF_REQUIRED")
            if state.scene_triplanes.shape[0] != 1:
                raise ValueError("RSS_VNEXT_FC_TRAIN_RAGGED_SINGLE_ASSET")
            target = batch.flexicubes_reference_surface_xyz
            if target is not None and (
                target.ndim != 3 or target.shape[0] != 1
                or target.shape[-1] != 3 or not 3 <= target.shape[1] <= 512
                or not bool(torch.isfinite(target).all())
                or target.device != state.scene_triplanes.device
            ):
                raise ValueError("RSS_VNEXT_FC_TRAIN_REFERENCE_SURFACE_INVALID")
            if (
                self.weights.flexicubes_surface > 0 and target is not None
            ) or (
                self.weights.flexicubes_animation > 0
                and batch.flexicubes_animation_teacher is not None
            ):
                fc_surface = self.model.fc_surface_bridge(
                    state.scene_triplanes,
                    extractor=batch.flexicubes_extractor,
                    role_head=self.model.presentation_role_head,
                    schema=self.model.cfg.semantic_role_schema(),
                    layer=batch.flexicubes_layer,
                    field_policy="SIGNED_HYPOTHESIS_ONLY",
                )
                n = fc_surface.xyz.shape[1]
                idx = torch.linspace(
                    0, n - 1, min(n, 512), device=target.device,
                ).long()
                # Only source samples are supervision. Neural/proposed FC
                # vertices enter the differentiable path back into H(x).
                if target is not None and self.weights.flexicubes_surface > 0:
                    fc_loss = (
                        symmetric_surface_chamfer_v1(
                            fc_surface.xyz[:, idx], target,
                        )
                        + 0.05 * fc_surface.regularizer.mean()
                    )
                if (
                    batch.flexicubes_animation_teacher is not None
                    and self.weights.flexicubes_animation > 0
                ):
                    if not self.model.training:
                        raise RuntimeError("RSS_VNEXT_FC_ANIMATION_TEACHER_TRAIN_ONLY")
                    result = source_animation_flexicubes_loss(
                        fc_surface, batch.flexicubes_animation_teacher,
                        config=RSSvNextFCAnimationLossConfig(),
                    )
                    fc_anim_loss = result.total
                    fc_anim_count = result.pose_count
        elif batch.flexicubes_signed_source_admission_key:
            raise ValueError("RSS_VNEXT_FC_TRAIN_ORPHAN_QUALIFICATION")
        memory = self.model._structural_memory(
            state, aux, flexicubes=fc_surface,
        )
        topology = None
        motion = None
        semantic = None
        n_semantic = 0
        if batch.semantic_patch_target is not None:
            if (batch.semantic_qualification_key == ""
                    or batch.semantic_patch_mask is None):
                raise ValueError("RSS_VNEXT_TRAIN_SEMANTIC_QUALIFICATION_REQUIRED")
            if self.weights.source_semantics > 0:
                semantic = masked_source_semantic_evidence_loss(
                    state.source_semantic_patch_logits,
                    batch.semantic_patch_target,
                    batch.semantic_patch_mask,
                )
                n_semantic = int(batch.semantic_patch_mask.sum().item())
        elif batch.semantic_patch_mask is not None:
            raise ValueError("RSS_VNEXT_TRAIN_SEMANTIC_ORPHAN_MASK")
        n_topo = 0
        n_pairs = 0
        if self.weights.topology > 0 and batch.topology_teacher_tokens is not None:
            tokens = batch.topology_teacher_tokens
            mask = batch.topology_valid_mask
            if tokens.ndim != 2 or tokens.shape[0] != memory.shape[0] or tokens.shape[1] < 3:
                raise ValueError("RSS_VNEXT_TRAIN_TOPOLOGY_TOKENS_SHAPE")
            if mask is None or mask.shape != (tokens.shape[0],) or mask.dtype != torch.bool:
                raise ValueError("RSS_VNEXT_TRAIN_TOPOLOGY_MASK_REQUIRED")
            if bool(mask.any()):
                selected = tokens[mask]
                if not bool((selected[:, 0] == self.model.cfg.tessa.BOS).all()):
                    raise ValueError("RSS_VNEXT_TRAIN_TOPOLOGY_BOS")
                logits = self.model.tessa.decode_tokens(
                    memory=memory[mask], input_ids=selected[:, :-1],
                ).logits
                topology = topology_sequence_objective_v2(
                    logits, selected[:, 1:],
                    config=self.model.cfg.tessa,
                    weights=TESSAObjectiveWeightsV2(),
                ).total
                n_topo = int(mask.sum())
        elif batch.topology_teacher_tokens is not None and batch.topology_valid_mask is None:
            raise ValueError("RSS_VNEXT_TRAIN_TOPOLOGY_MASK_REQUIRED")

        if self.weights.motion_preference > 0 and batch.motion_candidate_tokens is not None:
            if batch.motion_court is None:
                raise ValueError("RSS_VNEXT_TRAIN_MOTION_COURT_REQUIRED")
            if memory.shape[0] != 1:
                raise ValueError("RSS_VNEXT_TRAIN_MOTION_COURT_BATCH_ONE_ONLY")
            scores = topology_candidate_log_probs(
                self.model.tessa, memory, batch.motion_candidate_tokens,
                config=self.model.cfg.tessa,
            )
            preference = source_relative_topology_preference_loss(
                scores, batch.motion_candidate_tokens, batch.motion_court,
                config=self.preference,
            )
            motion = preference.loss
            n_pairs = preference.pair_count
        elif batch.motion_candidate_tokens is not None and batch.motion_court is None:
            raise ValueError("RSS_VNEXT_TRAIN_MOTION_COURT_REQUIRED")

        if (
            topology is None and motion is None and semantic is None
            and fc_loss is None and fc_anim_loss is None
        ):
            raise ValueError("RSS_VNEXT_TRAIN_NO_QUALIFIED_OBJECTIVE")
        total = sum(
            weight * loss
            for weight, loss in (
                (self.weights.topology, topology),
                (self.weights.motion_preference, motion),
                (self.weights.source_semantics, semantic),
                (self.weights.flexicubes_surface, fc_loss),
                (self.weights.flexicubes_animation, fc_anim_loss),
            )
            if loss is not None
        )
        if not bool(torch.isfinite(total)):
            raise RuntimeError("RSS_VNEXT_TRAIN_OBJECTIVE_NONFINITE")
        return RSSvNextTrainingResult(
            total_loss=total,
            topology_loss=topology,
            motion_preference_loss=motion,
            source_semantic_loss=semantic,
            flexicubes_surface_loss=fc_loss,
            flexicubes_animation_loss=fc_anim_loss,
            eligible_animation_pose_count=fc_anim_count,
            eligible_semantic_patch_labels=n_semantic,
            eligible_topology_examples=n_topo,
            qualified_preference_pairs=n_pairs,
        )

    def train_step(self, batch: RSSvNextStructuralTrainingBatch) -> RSSvNextTrainingResult:
        self.optimizer.zero_grad(set_to_none=True)
        result = self.objective(batch)
        result.total_loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(
            self.model.parameters(), self.weights.clip_gradient_norm,
            error_if_nonfinite=True,
        )
        self.optimizer.step()
        self.global_step += 1
        result.gradient_norm = grad_norm
        return result

    def save_checkpoint(self, path: str | Path, *, sampler_cursor: int) -> None:
        if sampler_cursor < 0:
            raise ValueError("RSS_VNEXT_TRAIN_SAMPLER_CURSOR_INVALID")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": self.checkpoint_schema,
            "architecture_id": self.model.architecture_id,
            "architecture_digest": architecture_digest(self.model.cfg),
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "global_step": self.global_step,
            "sampler_cursor": int(sampler_cursor),
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "torch_cpu_rng": torch.get_rng_state(),
            "torch_cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        }
        # Write and replace atomically on the same filesystem.
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix="rssckpt-", suffix=".pt", delete=False) as f:
            tmp = Path(f.name)
        try:
            torch.save(payload, tmp)
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)

    def load_checkpoint(self, path: str | Path) -> int:
        data = torch.load(
            Path(path), map_location="cpu", weights_only=True,
        )
        if data.get("schema") != self.checkpoint_schema:
            raise ValueError("RSS_VNEXT_TRAIN_CHECKPOINT_SCHEMA_DRIFT")
        if data.get("architecture_id") != self.model.architecture_id:
            raise ValueError("RSS_VNEXT_TRAIN_CHECKPOINT_ARCHITECTURE_ID_DRIFT")
        if data.get("architecture_digest") != architecture_digest(self.model.cfg):
            raise ValueError("RSS_VNEXT_TRAIN_CHECKPOINT_ARCHITECTURE_DIGEST_DRIFT")
        if data.get("dataset_manifest_sha256") != self.dataset_manifest_sha256:
            raise ValueError("RSS_VNEXT_TRAIN_CHECKPOINT_DATASET_MANIFEST_DRIFT")
        if int(data.get("global_step", -1)) < 0 or int(data.get("sampler_cursor", -1)) < 0:
            raise ValueError("RSS_VNEXT_TRAIN_CHECKPOINT_PROGRESS_INVALID")
        self.model.load_state_dict(data["model"], strict=True)
        self.optimizer.load_state_dict(data["optimizer"])
        self.global_step = int(data["global_step"])
        torch.set_rng_state(data["torch_cpu_rng"])
        if torch.cuda.is_available() and data.get("torch_cuda_rng"):
            torch.cuda.set_rng_state_all(data["torch_cuda_rng"])
        return int(data["sampler_cursor"])


__all__ = [
    "RSSvNextStructuralTrainingBatch", "RSSvNextTrainingWeights",
    "RSSvNextTrainingResult", "RSSvNextStructuralTrainer",
    "architecture_digest",
]
