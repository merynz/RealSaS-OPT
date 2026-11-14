from __future__ import annotations

"""RSS.vNext unified neural training core — structural and motion pilot.

This is a real optimizer/gradient/checkpoint path, NOT the final Truth Foundry
schema. It consumes only admitted supervision, with explicit masks. Its
current task scope includes structural topology, source-motion preference,
early source semantic patch evidence and explicitly qualified 8-way rendered
appearance supervision. Other heads are not silently declared training-ready.
"""
from dataclasses import dataclass, asdict
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Mapping
from types import MappingProxyType

import torch
from torch import Tensor, nn
from torch.nn import functional as F

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
from .attachment_face_correspondence_v1 import (
    source_qualified_socket_face_loss_v1, axis_sequence_digest_v1,
    mesh_topology_digest_v1,
)
from .early_semantic_field import (
    masked_source_semantic_evidence_loss,
    masked_semantic_expected_brier_risk_loss,
)
from .semantic_structure import (
    qualified_source_region_pair_loss,
    qualified_source_typed_relation_loss,
    TYPED_REGION_RELATION_CHANNELS_V1,
)
from .semantic_geometry_prior import (
    RSSvNextSemanticLayerSpec, masked_qualified_surface_role_loss,
)
from .multiview_construction import qualified_novel_view_loss
from .realsas_joint_slrm_render_v1 import (
    STAGE_1, STAGE_2, STAGE_3, RealSaSJointTargetV1,
    render_realsas_joint_from_shared_scene_v1,
    realsas_joint_stage_loss_v1,
)
from .eight_way_teacher_court_v1 import (
    RSSvNextEightWayTeacherReceiptV1,
    validate_eight_way_teacher_receipts_v1,
)
from models.iris.vnext.flexicubes_v1 import symmetric_surface_chamfer_v1
from models.axis.rig_program_model_v6 import (
    AXISSequenceV6, AXIS_V6_PRODUCT_ARCHITECTURE_ID,
    qualified_axis_rig_skin_sequence_loss_v6, RigTokenKindV6,
)
from compiler.realsas_compiler_core.rig_constraint_schema_v1 import RIG_CONSTRAINT_OPERATOR_IDS_V1


@dataclass
class RSSvNextStructuralTrainingBatch:
    rgba: Tensor
    camera_features: Tensor
    # Product reconstruction sees only original pixels + six generated
    # RGB/normal/camera hypotheses. Training source camera is teacher-only.
    product_generated_rgba: Tensor | None = None
    product_generated_normals: Tensor | None = None
    product_generated_cameras: Tensor | None = None
    product_source_binding_sha256: str = ""
    product_canonical_checkpoint_sha256: str = ""
    product_multiview_checkpoint_sha256: str = ""
    product_generation_checkpoint_sha256: str = ""
    product_conditioning_admission_key: str = ""
    topology_teacher_tokens: Tensor | None = None
    topology_valid_mask: Tensor | None = None
    motion_candidate_tokens: Tensor | None = None
    motion_court: RSSvNextMotionCourt | None = None
    # Provisional trainer-level evidence, NOT a frozen Truth Foundry schema.
    semantic_patch_target: Tensor | None = None  # [B,2,T,K]
    semantic_patch_mask: Tensor | None = None    # [B,2,T,K], per-axis qualified
    semantic_qualification_key: str = ""
    # Explicit 3D query-address source-surface truth, NOT patch projection or
    # pseudo-labels from the student. Unobserved/ambiguous locations stay masked.
    source_surface_role_indices: Tensor | None = None  # [B,N] int64
    source_surface_role_mask: Tensor | None = None     # [B,N] bool
    source_surface_role_admission_key: str = ""        # external trusted receipt
    # Exact H(x) query-address region co-membership (source-native or
    # source-rendered). This is NOT drawable/attachment/motion supervision.
    source_region_pair_indices: Tensor | None = None  # [B,P,2] long
    source_region_pair_is_same: Tensor | None = None  # [B,P] float 0/1; unknown masked
    source_region_pair_mask: Tensor | None = None     # [B,P] bool qualified
    source_region_pair_admission_key: str = ""
    # Three independent SOURCE-BOUND relation teachers over exact 3D query
    # pairs; these are NOT fixed anonymous slot labels. In particular, a part
    # segmentation cannot supply an attachment/contact/motion teacher.
    source_typed_relation_pair_indices: Tensor | None = None  # [B,P,2]
    source_typed_relation_target: Tensor | None = None         # [B,P,3]
    source_typed_relation_mask: Tensor | None = None           # [B,P,3] bool
    source_typed_relation_admission_key: str = ""
    # Exact H(x)-addressed visible RGB source witnesses. These are original
    # source-ray samples with independent visibility/occlusion admission,
    # never invented per-vertex material colors or novel-view pseudo-labels.
    source_graphics_rgb_target: Tensor | None = None  # [B,Q,3] float [0,1]
    source_graphics_view_directions: Tensor | None = None  # [B,Q,3] qualified source first-hit directions
    source_graphics_rgb_mask: Tensor | None = None    # [B,Q] bool
    source_graphics_rgb_admission_key: str = ""
    # IRIS query-address source geometry targets. UDF is admissible for an
    # open surface; inside/outside sign requires independently qualified
    # closed-solid topology and negative-inside sign convention.
    source_udf_target: Tensor | None = None            # [B,Q] nonnegative
    source_udf_mask: Tensor | None = None              # [B,Q] bool
    source_inside_sign_target: Tensor | None = None    # [B,Q] float 1=inside
    source_inside_sign_mask: Tensor | None = None      # [B,Q] bool
    source_field_admission_key: str = ""
    # SkinTokens-style causal joint+FSQ sequence, with independent teacher
    # admission. Control/constraint tokens additionally need intervention truth.
    axis_teacher_sequence: AXISSequenceV6 | None = None
    axis_teacher_admission_key: str = ""
    # Exact source-native HELD/HOLDER labels, independently reindexed to the
    # training TESSA teacher mesh; -1 UNKNOWN is masked, not a true negative.
    source_anchor_face_targets: Tensor | None = None
    source_anchor_face_mask: Tensor | None = None
    source_anchor_face_admission_key: str = ""
    # Only source-native, independently reindexed TESSA mesh labels may
    # condition product AXIS teacher forcing. Teacher remains training-only.
    # Independently source-rendered 6 novel views; FRONT/BACK remain the only
    # model observations. These are teacher targets, never encoder inputs.
    novel_view_rgba: Tensor | None = None       # [B,6,4,H,W]
    novel_view_normal: Tensor | None = None     # [B,6,3,H,W]
    novel_view_rgb_mask: Tensor | None = None   # [B,6,H,W], qualified
    novel_view_normal_mask: Tensor | None = None
    novel_view_provenance: str = ""              # Optional audit check; never grants authority
    novel_view_receipts: tuple[RSSvNextEightWayTeacherReceiptV1, ...] | None = None

    # the CVPR 2025 reference §3.2 joint geometry+RGB+semantic *ray-rendered* supervision.
    # Stage 1: full exterior. Stage 2: source 3D renderer peeled layers.
    # A signed mesh teacher MUST NOT be fabricated from these NeRF rays.
    realsas_joint_ray_xyz: Tensor | None = None        # [B,R,S,3] camera rays
    realsas_joint_ray_delta: Tensor | None = None      # [B,R,S] sample intervals
    realsas_joint_ray_depth: Tensor | None = None      # [B,R,S] ray coordinates
    realsas_joint_ray_direction: Tensor | None = None  # [B,R,3]
    realsas_joint_keep_roles: Tensor | None = None     # [B,R,K] bool
    realsas_joint_target: RealSaSJointTargetV1 | None = None
    realsas_joint_teacher_admission_key: str = ""

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
    semantic_risk: float = 0.0  # opt-in: expected masked source-patch Brier risk
    surface_roles: float = 0.0  # opt-in only with exact-source 3D address receipt
    region_pairs: float = 0.0  # opt-in only with source-bound pair receipt
    typed_relations: float = 0.0  # source-confirmed drawable/cut/directed attachment
    graphics_source_rgb: float = 0.0  # opt-in: qualified original visible rays
    source_udf: float = 0.0           # opt-in: source native unsigned distances
    source_inside_sign: float = 0.0   # opt-in: closed-solid signed occupancy
    axis_rig_skin_codes: float = 0.0  # opt-in: source rig and MIRA-codec-bound FSQ teacher
    anchor_face_correspondence: float = 0.0  # source-native directed face labels
    flexicubes_surface: float = 0.3
    flexicubes_animation: float = 0.0  # explicit opt-in on admitted source motion
    novel_view: float = 0.0             # opt-in: source-qualified 8-way appearance
    realsas_joint_slrm: float = 0.0      # opt-in: jointly rendered 3D scene teacher
    clip_gradient_norm: float = 1.0

    def validate(self):
        # NaN/Inf and booleans otherwise bypass comparison-based guards and
        # poison checkpoint reproducibility while the loss policy looks valid.
        if any(
            type(value) not in (float, int) or not math.isfinite(value)
            for value in asdict(self).values()
        ):
            raise ValueError("RSS_VNEXT_TRAIN_WEIGHT_POLICY_NONFINITE_OR_UNTYPED")
        if min(
            self.topology, self.motion_preference, self.source_semantics,
            self.semantic_risk,
            self.surface_roles, self.region_pairs, self.typed_relations,
            self.graphics_source_rgb,
            self.source_udf, self.source_inside_sign,
            self.axis_rig_skin_codes, self.anchor_face_correspondence, self.flexicubes_surface, self.flexicubes_animation,
            self.novel_view, self.realsas_joint_slrm,
        ) < 0:
            raise ValueError("RSS_VNEXT_TRAIN_NEGATIVE_OBJECTIVE_WEIGHT")
        if (
            self.topology + self.motion_preference
            + self.source_semantics + self.semantic_risk + self.surface_roles + self.region_pairs
            + self.typed_relations + self.graphics_source_rgb + self.source_udf + self.source_inside_sign
            + self.axis_rig_skin_codes + self.anchor_face_correspondence + self.flexicubes_surface
            + self.flexicubes_animation + self.novel_view
            + self.realsas_joint_slrm <= 0
            or self.clip_gradient_norm <= 0
        ):
            raise ValueError("RSS_VNEXT_TRAIN_WEIGHT_POLICY_INVALID")


@dataclass
class RSSvNextTrainingResult:
    total_loss: Tensor
    topology_loss: Tensor | None
    motion_preference_loss: Tensor | None
    source_semantic_loss: Tensor | None
    semantic_risk_loss: Tensor | None
    flexicubes_surface_loss: Tensor | None
    flexicubes_animation_loss: Tensor | None
    eligible_animation_pose_count: int
    eligible_semantic_patch_labels: int
    eligible_topology_examples: int
    qualified_preference_pairs: int
    novel_view_loss: Tensor | None = None
    eligible_novel_view_pixels: int = 0
    surface_role_loss: Tensor | None = None
    eligible_surface_role_points: int = 0
    region_pair_loss: Tensor | None = None
    eligible_region_pairs: int = 0
    typed_relation_loss: Tensor | None = None
    eligible_typed_relation_channels: tuple[int, int, int] = (0, 0, 0)
    graphics_source_rgb_loss: Tensor | None = None
    eligible_graphics_source_rgb_queries: int = 0
    source_udf_loss: Tensor | None = None
    source_inside_sign_loss: Tensor | None = None
    eligible_source_udf_queries: int = 0
    eligible_source_inside_sign_queries: int = 0
    axis_rig_skin_loss: Tensor | None = None
    eligible_axis_sequences: int = 0
    axis_training_conditioning: str = "NO_AXIS_OBJECTIVE"
    anchor_face_correspondence_loss: Tensor | None = None
    eligible_anchor_held_faces: int = 0
    eligible_anchor_holder_faces: int = 0
    eligible_anchor_negative_faces: int = 0
    realsas_joint_slrm_loss: Tensor | None = None
    eligible_realsas_joint_rays: int = 0
    gradient_norm: Tensor | None = None
    gradient_interference_report: dict | None = None


def axis_mesh_teacher_address_digest_v1(mesh) -> str:
    """Exact *discrete* TESSA index/face and canonical base-lattice identity.

    Rest vertices depend on current learned H(x), so hashing those would
    make source teacher admission dependent on student checkpoint weights.
    Only immutable decoded indices, component IDs and lattice XYZ are
    admissible for independent source-native face reindex qualification.
    """
    mesh.validate()
    return content_fields_sha256({
        "token_ids": hashlib.sha256(json.dumps(
            list(mesh.topology_token_ids), separators=(",", ":"),
        ).encode("utf8")).hexdigest(),
        "base_vertices_iris": source_surface_tensor_sha256(mesh.base_vertices_iris),
        "faces": source_surface_tensor_sha256(mesh.faces),
        "vertex_component_ids": source_surface_tensor_sha256(mesh.vertex_component_ids),
        "face_component_ids": source_surface_tensor_sha256(mesh.face_component_ids),
        "schema": "RealSaS.RSSvNext.AxisMatchedTessaTeacherMeshAddresses.v1",
    })


def axis_teacher_sequence_sha256_v6(teacher: AXISSequenceV6) -> str:
    """Hash all causal token labels/attributes; a batch cannot self-certify."""
    return content_fields_sha256({
        field_name: source_surface_tensor_sha256(getattr(teacher, field_name))
        for field_name in teacher.__dataclass_fields__
    })


def content_fields_sha256(fields: Mapping[str, str]) -> str:
    return hashlib.sha256(json.dumps(
        dict(fields), sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf8")).hexdigest()


def frozen_mira_codec_state_sha256_v6(model: RSSvNextUnifiedCharacterModel) -> str:
    """Exact FSQ codebook+decoder byte identity; no unpinned teacher code IDs."""
    return content_fields_sha256({
        key: source_surface_tensor_sha256(value)
        for key, value in model.mira.codec.state_dict().items()
    })


def architecture_digest(cfg: RSSvNextConfig) -> str:
    """Hash relevant model-shape config; mismatched checkpoint must not resume."""
    from dataclasses import asdict
    value = json.dumps(
        asdict(cfg), sort_keys=True, separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf8")
    return hashlib.sha256(value).hexdigest()


def source_surface_tensor_sha256(value: Tensor) -> str:
    """Stable query/teacher byte identity used by externally admitted receipts."""
    if not isinstance(value, Tensor):
        raise ValueError("RSS_VNEXT_SURFACE_ROLE_RECEIPT_TENSOR_REQUIRED")
    cpu = value.detach().contiguous().cpu()
    header = json.dumps({
        "dtype": str(cpu.dtype), "shape": list(cpu.shape),
    }, sort_keys=True, separators=(",", ":")).encode("utf8")
    return hashlib.sha256(header + bytes([0]) + cpu.numpy().tobytes()).hexdigest()


from .training_optimizer_ownership_v1 import qualify_rss_vnext_optimizer_ownership_v1
from .training_gradient_interference_v1 import (
    RSSvNextGradientCourtPolicyV1,
    audit_rss_vnext_gradient_interference_v1,
)


class RSSvNextStructuralTrainer:
    """Training-engine first slice: CE + consequence-driven discrete topology.

    One shared 2-view scene forward; the two objectives read the same latent
    memory and backpropagate through the same encoder. Teacher tokens and G3
    verdicts never enter image tokens or scene forward.
    """

    checkpoint_schema = "RealSaS.RSSvNext.StructuralTrainerCheckpoint.v2"

    def __init__(
        self,
        model: RSSvNextUnifiedCharacterModel,
        optimizer: torch.optim.Optimizer,
        *,
        dataset_manifest_sha256: str,
        weights: RSSvNextTrainingWeights = RSSvNextTrainingWeights(),
        preference: RSSvNextTopologyPreferenceConfig = RSSvNextTopologyPreferenceConfig(),
        gradient_court_policy: RSSvNextGradientCourtPolicyV1 = RSSvNextGradientCourtPolicyV1(),
        scene_conditioning_policy: str = "RESEARCH_CALIBRATED_TWO_VIEW",
        axis_training_policy: str = "RIG_ONLY_PRETRAIN",
        trusted_generated_conditioning_registry: Mapping[str, Mapping[str, object]] | None = None,
        trusted_render_registry: Mapping[str, Mapping[str, object]] | None = None,
        trusted_surface_role_registry: Mapping[str, Mapping[str, object]] | None = None,
        trusted_semantic_patch_registry: Mapping[str, Mapping[str, object]] | None = None,
        trusted_region_pair_registry: Mapping[str, Mapping[str, object]] | None = None,
        trusted_typed_relation_registry: Mapping[str, Mapping[str, object]] | None = None,
        trusted_axis_teacher_registry: Mapping[str, Mapping[str, object]] | None = None,
        trusted_anchor_face_registry: Mapping[str, Mapping[str, object]] | None = None,
        trusted_signed_source_registry: Mapping[str, Mapping[str, object]] | None = None,
        trusted_graphics_rgb_registry: Mapping[str, Mapping[str, object]] | None = None,
        trusted_source_field_registry: Mapping[str, Mapping[str, object]] | None = None,
        trusted_realsas_joint_scene_registry: Mapping[str, Mapping[str, object]] | None = None,
    ):
        if not dataset_manifest_sha256 or len(dataset_manifest_sha256) != 64:
            raise ValueError("RSS_VNEXT_TRAIN_DATASET_MANIFEST_SHA_REQUIRED")
        try:
            int(dataset_manifest_sha256, 16)
        except ValueError as exc:
            raise ValueError("RSS_VNEXT_TRAIN_DATASET_MANIFEST_NOT_HEX") from exc
        weights.validate()
        preference.validate()
        gradient_court_policy.validate()
        if scene_conditioning_policy not in (
            "RESEARCH_CALIBRATED_TWO_VIEW", "PRODUCT_GENERATED_SIX_VIEW",
        ):
            raise ValueError("RSS_VNEXT_TRAIN_SCENE_CONDITIONING_POLICY_INVALID")
        if (
            scene_conditioning_policy != "PRODUCT_GENERATED_SIX_VIEW"
            and getattr(model, "construction_requires_generated_multiview", False)
        ):
            raise ValueError("RSS_VNEXT_TRAIN_PRODUCT_CHECKPOINT_REQUIRES_GENERATED_SCENE")
        if (
            scene_conditioning_policy == "PRODUCT_GENERATED_SIX_VIEW"
            and model.cfg.constructed_total_views != 0
        ):
            raise ValueError("RSS_VNEXT_TRAIN_COMPETING_MULTIVIEW_CONSTRUCTORS")
        if axis_training_policy not in (
            "RIG_ONLY_PRETRAIN", "PRODUCT_MESH_MATCHED_REQUIRED",
        ):
            raise ValueError("RSS_VNEXT_AXIS_TRAINING_POLICY_UNKNOWN")
        if (
            scene_conditioning_policy == "PRODUCT_GENERATED_SIX_VIEW"
            and weights.axis_rig_skin_codes > 0
            and axis_training_policy != "PRODUCT_MESH_MATCHED_REQUIRED"
        ):
            raise ValueError("RSS_VNEXT_PRODUCT_AXIS_MESH_MATCHED_POLICY_REQUIRED")
        self.scene_conditioning_policy = scene_conditioning_policy
        self.axis_training_policy = axis_training_policy
        self.model = model
        self.optimizer = optimizer
        self.optimizer_ownership = qualify_rss_vnext_optimizer_ownership_v1(
            self.model, self.optimizer,
        )
        self.weights = weights
        self.preference = preference
        self.gradient_court_policy = gradient_court_policy
        # Freeze the full objective / topology-preference policy into the
        # resumption identity. A semantic-risk or motion weight change is a
        # different training run, not a transparent continuation of this one.
        self.objective_policy_sha256 = hashlib.sha256(json.dumps({
            "weights": asdict(weights),
            "preference": asdict(preference),
            "scene_conditioning_policy": self.scene_conditioning_policy,
            "axis_training_policy": self.axis_training_policy,
            "gradient_court_policy": asdict(gradient_court_policy),
        }, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf8")).hexdigest()
        self.dataset_manifest_sha256 = dataset_manifest_sha256.lower()
        # This launcher-owned registry binds the *actual generated* scene
        # conditioning, not independent source-rendered teacher imagery.
        # A bare checkpoint SHA in an untrusted batch is never sufficient.
        generated_snapshot = {
            str(key): dict(value)
            for key, value in (trusted_generated_conditioning_registry or {}).items()
        }
        self.trusted_generated_conditioning_registry = MappingProxyType({
            key: MappingProxyType(value)
            for key, value in generated_snapshot.items()
        })
        self.generated_conditioning_registry_sha256 = (
            hashlib.sha256(json.dumps(
                generated_snapshot, sort_keys=True, separators=(",", ":"),
                allow_nan=False,
            ).encode("utf8")).hexdigest()
            if generated_snapshot else ""
        )
        # The trusted training launcher supplies this, not the data batch.
        registry_snapshot = {
            str(key): dict(value)
            for key, value in (trusted_render_registry or {}).items()
        }
        self.trusted_render_registry = MappingProxyType({
            key: MappingProxyType(value)
            for key, value in registry_snapshot.items()
        })
        self.render_registry_sha256 = (
            hashlib.sha256(json.dumps(
                registry_snapshot, sort_keys=True,
                separators=(",", ":"), allow_nan=False,
            ).encode("utf8")).hexdigest()
            if self.trusted_render_registry else ""
        )
        # A batch cannot self-certify 3D semantic truth. The launcher's
        # separate foundry admission registry binds source, dataset, role
        # schema, query XYZ and exact label/mask bytes before gradients run.
        surface_snapshot = {
            str(key): dict(value)
            for key, value in (trusted_surface_role_registry or {}).items()
        }
        self.trusted_surface_role_registry = MappingProxyType({
            key: MappingProxyType(value) for key, value in surface_snapshot.items()
        })
        self.surface_role_registry_sha256 = (
            hashlib.sha256(json.dumps(
                surface_snapshot, sort_keys=True, separators=(",", ":"),
                allow_nan=False,
            ).encode("utf8")).hexdigest()
            if surface_snapshot else ""
        )
        # Early 2D semantic axes are not allowed to self-certify with a
        # user-supplied string. Bind exact observed rays/cameras and the
        # per-channel known/unknown teacher bytes at the training launcher.
        semantic_snapshot = {
            str(key): dict(value)
            for key, value in (trusted_semantic_patch_registry or {}).items()
        }
        self.trusted_semantic_patch_registry = MappingProxyType({
            key: MappingProxyType(value)
            for key, value in semantic_snapshot.items()
        })
        self.semantic_patch_registry_sha256 = (
            hashlib.sha256(json.dumps(
                semantic_snapshot, sort_keys=True, separators=(",", ":"),
                allow_nan=False,
            ).encode("utf8")).hexdigest() if semantic_snapshot else ""
        )
        # This registry is supplied by the trusted training launcher, not by
        # the batch. In particular, component IDs are *not* mechanically
        # reinterpreted as drawable, articulation or attachment ground truth.
        region_snapshot = {
            str(key): dict(value)
            for key, value in (trusted_region_pair_registry or {}).items()
        }
        self.trusted_region_pair_registry = MappingProxyType({
            key: MappingProxyType(value) for key, value in region_snapshot.items()
        })
        self.region_pair_registry_sha256 = (
            hashlib.sha256(json.dumps(
                region_snapshot, sort_keys=True, separators=(",", ":"),
                allow_nan=False,
            ).encode("utf8")).hexdigest()
            if region_snapshot else ""
        )
        # Source-space *typed* relation teaching has a separate registry;
        # it must not borrow co-region supervision and call it attachment.
        typed_relation_snapshot = {
            str(key): dict(value)
            for key, value in (trusted_typed_relation_registry or {}).items()
        }
        self.trusted_typed_relation_registry = MappingProxyType({
            key: MappingProxyType(value)
            for key, value in typed_relation_snapshot.items()
        })
        self.typed_relation_registry_sha256 = (
            hashlib.sha256(json.dumps(
                typed_relation_snapshot, sort_keys=True, separators=(",", ":"),
                allow_nan=False,
            ).encode("utf8")).hexdigest()
            if typed_relation_snapshot else ""
        )
        axis_snapshot = {
            str(key): dict(value)
            for key, value in (trusted_axis_teacher_registry or {}).items()
        }
        self.trusted_axis_teacher_registry = MappingProxyType({
            key: MappingProxyType(value) for key, value in axis_snapshot.items()
        })
        self.axis_teacher_registry_sha256 = (
            hashlib.sha256(json.dumps(
                axis_snapshot, sort_keys=True, separators=(",", ":"), allow_nan=False,
            ).encode("utf8")).hexdigest() if axis_snapshot else ""
        )
        # Separate source-native face-reindex truth: a joint token/region
        # segmentation cannot self-certify contact or true NO_FACE labels.
        anchor_snapshot = {
            str(key): dict(value)
            for key, value in (trusted_anchor_face_registry or {}).items()
        }
        self.trusted_anchor_face_registry = MappingProxyType({
            key: MappingProxyType(value) for key, value in anchor_snapshot.items()
        })
        self.anchor_face_registry_sha256 = (
            hashlib.sha256(json.dumps(
                anchor_snapshot, sort_keys=True, separators=(",", ":"),
                allow_nan=False,
            ).encode("utf8")).hexdigest() if anchor_snapshot else ""
        )
        signed_snapshot = {
            str(key): dict(value)
            for key, value in (trusted_signed_source_registry or {}).items()
        }
        self.trusted_signed_source_registry = MappingProxyType({
            key: MappingProxyType(value) for key, value in signed_snapshot.items()
        })
        self.signed_source_registry_sha256 = (
            hashlib.sha256(json.dumps(
                signed_snapshot, sort_keys=True, separators=(",", ":"),
                allow_nan=False,
            ).encode("utf8")).hexdigest() if signed_snapshot else ""
        )
        graphics_snapshot = {
            str(key): dict(value)
            for key, value in (trusted_graphics_rgb_registry or {}).items()
        }
        self.trusted_graphics_rgb_registry = MappingProxyType({
            key: MappingProxyType(value) for key, value in graphics_snapshot.items()
        })
        self.graphics_rgb_registry_sha256 = (
            hashlib.sha256(json.dumps(
                graphics_snapshot, sort_keys=True, separators=(",", ":"),
                allow_nan=False,
            ).encode("utf8")).hexdigest() if graphics_snapshot else ""
        )
        field_snapshot = {
            str(key): dict(value)
            for key, value in (trusted_source_field_registry or {}).items()
        }
        self.trusted_source_field_registry = MappingProxyType({
            key: MappingProxyType(value) for key, value in field_snapshot.items()
        })
        self.source_field_registry_sha256 = (
            hashlib.sha256(json.dumps(
                field_snapshot, sort_keys=True, separators=(",", ":"),
                allow_nan=False,
            ).encode("utf8")).hexdigest() if field_snapshot else ""
        )
        joint_snapshot = {
            str(key): dict(value)
            for key, value in (trusted_realsas_joint_scene_registry or {}).items()
        }
        self.trusted_realsas_joint_scene_registry = MappingProxyType({
            key: MappingProxyType(value) for key, value in joint_snapshot.items()
        })
        self.realsas_joint_registry_sha256 = (
            hashlib.sha256(json.dumps(
                joint_snapshot, sort_keys=True, separators=(",", ":"),
                allow_nan=False,
            ).encode("utf8")).hexdigest() if joint_snapshot else ""
        )
        self.global_step = 0

    def objective(self, batch: RSSvNextStructuralTrainingBatch) -> RSSvNextTrainingResult:
        if batch.admission_status != "QUALIFIED_FOR_THIS_SUPERVISION_FAMILY":
            raise ValueError("RSS_VNEXT_TRAIN_ADMISSION_NOT_QUALIFIED")
        if not batch.source_qualification_key:
            raise ValueError("RSS_VNEXT_TRAIN_SOURCE_QUALIFICATION_KEY_MISSING")
        if not self.model.training:
            raise RuntimeError("RSS_VNEXT_TRAIN_REQUIRES_TRAIN_MODE")

        product_fields = (
            batch.product_generated_rgba, batch.product_generated_normals,
            batch.product_generated_cameras, batch.product_source_binding_sha256,
            batch.product_canonical_checkpoint_sha256,
            batch.product_multiview_checkpoint_sha256,
            batch.product_generation_checkpoint_sha256,
            batch.product_conditioning_admission_key,
        )
        if self.scene_conditioning_policy == "PRODUCT_GENERATED_SIX_VIEW":
            if (
                any(x is None or x == "" for x in product_fields[3:])
                or any(x is None for x in product_fields[:3])
            ):
                raise ValueError("RSS_VNEXT_TRAIN_PRODUCT_GENERATED_SCENE_REQUIRED")
            generated, normals, cameras = product_fields[:3]
            if (
                not isinstance(generated, Tensor)
                or not isinstance(normals, Tensor)
                or not isinstance(cameras, Tensor)
                or batch.rgba.ndim != 5 or batch.rgba.shape[:3] != (1, 2, 4)
                or generated.ndim != 5 or generated.shape[:3] != (1, 6, 4)
                or normals.shape != (1, 6, 3, *generated.shape[-2:])
                or cameras.shape != (1, 6, 13)
                or any(t.device != batch.rgba.device for t in (generated, normals, cameras))
            ):
                raise ValueError("RSS_VNEXT_TRAIN_PRODUCT_GENERATED_SCENE_SHAPE")
            from .realsas_virtual_camera_policy_v1 import (
                target_canonical_and_six_cameras_v1,
            )
            # The generation stage uses precisely these six virtual cameras.
            # Model inputs must never receive measured source-camera labels.
            _, fixed = target_canonical_and_six_cameras_v1(
                1, device=str(cameras.device), dtype=cameras.dtype,
            )
            if not torch.allclose(cameras, fixed, atol=1.e-6, rtol=0):
                raise ValueError("RSS_VNEXT_TRAIN_GENERATOR_CAMERA_CONTRACT_DRIFT")
            expected = {
                "admission_status": "PINNED_GENERATOR_OUTPUT_NOT_SOURCE_TEACHER",
                "dataset_manifest_sha256": self.dataset_manifest_sha256,
                "source_qualification_key": batch.source_qualification_key,
                "source_observed_rgba_sha256": source_surface_tensor_sha256(batch.rgba),
                "original_camera_is_model_input": False,
                "generated_rgba_sha256": source_surface_tensor_sha256(generated),
                "generated_normals_sha256": source_surface_tensor_sha256(normals),
                "generated_cameras_sha256": source_surface_tensor_sha256(cameras),
                "source_binding_sha256": batch.product_source_binding_sha256,
                "canonical_stage_checkpoint_sha256": batch.product_canonical_checkpoint_sha256,
                "multiview_stage_checkpoint_sha256": batch.product_multiview_checkpoint_sha256,
                "combined_generation_checkpoint_sha256": batch.product_generation_checkpoint_sha256,
                "generated_views_are_observed_source": False,
            }
            if any(type(v) is not str or len(v) != 64 or any(
                c not in "0123456789abcdef" for c in v
            ) for v in (
                batch.product_source_binding_sha256,
                batch.product_canonical_checkpoint_sha256,
                batch.product_multiview_checkpoint_sha256,
                batch.product_generation_checkpoint_sha256,
            )):
                raise ValueError("RSS_VNEXT_TRAIN_GENERATOR_CHECKPOINT_SHA_DOMAIN")
            from .realsas_two_source_construction_v1 import (
                paired_generation_binding_sha256_v1,
            )
            expected_binding = paired_generation_binding_sha256_v1(
                canonical_checkpoint_sha256=batch.product_canonical_checkpoint_sha256,
                multiview_checkpoint_sha256=batch.product_multiview_checkpoint_sha256,
                source_binding_sha256=batch.product_source_binding_sha256,
            )
            if expected_binding != batch.product_generation_checkpoint_sha256:
                raise ValueError("RSS_VNEXT_TRAIN_GENERATOR_BINDING_RECOMPUTE_DRIFT")
            receipt = self.trusted_generated_conditioning_registry.get(
                batch.product_conditioning_admission_key
            )
            if receipt is None or any(receipt.get(k) != v for k,v in expected.items()):
                raise ValueError("RSS_VNEXT_TRAIN_GENERATED_SCENE_RECEIPT_DRIFT")
            state = self.model.encode_generated_multiview(
                batch.rgba, None,
                generated_rgba=generated, generated_cameras=cameras,
                generated_normals=normals,
                generation_checkpoint_sha256=batch.product_generation_checkpoint_sha256,
                source_binding_sha256=batch.product_source_binding_sha256,
            )
            if (
                state.camera_features is not None
                or state.scene_conditioning_authority != "GENERATED_SIX_VIEW_SHARED_SCENE"
                or state.source_camera_authority != "UNKNOWN_NO_ORIGINAL_CAMERA_PROJECTION"
            ):
                raise RuntimeError("RSS_VNEXT_TRAIN_GENERATED_SCENE_PRODUCT_AUTHORITY_DRIFT")
            aux = None
        else:
            if any(x is not None for x in product_fields[:3]) or any(product_fields[3:]):
                raise ValueError("RSS_VNEXT_TRAIN_GENERATED_SCENE_UNEXPECTED")
            state, aux, _ = self.model.encode_observations(
                batch.rgba, batch.camera_features,
            )
        joint_loss = None
        joint_rays = 0
        joint_fields = (
            batch.realsas_joint_ray_xyz, batch.realsas_joint_ray_delta,
            batch.realsas_joint_ray_depth, batch.realsas_joint_ray_direction,
            batch.realsas_joint_keep_roles, batch.realsas_joint_target,
        )
        if any(value is not None for value in joint_fields) or batch.realsas_joint_teacher_admission_key:
            if any(value is None for value in joint_fields) or not batch.realsas_joint_teacher_admission_key:
                raise ValueError("RSS_VNEXT_REALSAS_JOINT_TRAINING_TEACHER_INCOMPLETE")
            if self.weights.realsas_joint_slrm <= 0:
                raise ValueError("RSS_VNEXT_REALSAS_JOINT_TRAINING_WEIGHT_REQUIRED")
            xyz,delta,depth,direction,keep,target = joint_fields
            if type(target) is not RealSaSJointTargetV1:
                raise ValueError("RSS_VNEXT_REALSAS_JOINT_TARGET_TYPE_INVALID")
            if target.stage not in (STAGE_1,STAGE_2):
                raise ValueError("RSS_VNEXT_REALSAS_JOINT_FLEXICUBES_STAGE_MESH_RENDER_REQUIRED")
            k = self.model.cfg.semantic_role_schema().count
            b,r,s,_=xyz.shape if xyz.ndim==4 else (0,0,0,0)
            if (
                b!=batch.rgba.shape[0] or r<1 or s<2
                or delta.shape!=(b,r,s) or depth.shape!=(b,r,s)
                or direction.shape!=(b,r,3)
                or keep.shape!=(b,r,k) or keep.dtype!=torch.bool
                or (target.stage==STAGE_1 and not bool(keep.all()))
                or (target.stage==STAGE_2 and bool(keep.all()))
            ):
                raise ValueError("RSS_VNEXT_REALSAS_JOINT_RAY_OR_PEEL_STAGE_DOMAIN")
            # The independent renderer admission binds all tensors and the
            # original input identity. A batch string is NOT a teacher receipt.
            receipt=self.trusted_realsas_joint_scene_registry.get(
                batch.realsas_joint_teacher_admission_key
            )
            expected={
                "admission_status":"QUALIFIED_SOURCE_RENDERED_3D_JOINT_SCENE",
                "dataset_manifest_sha256":self.dataset_manifest_sha256,
                "source_qualification_key":batch.source_qualification_key,
                "observed_rgba_sha256":source_surface_tensor_sha256(batch.rgba),
                "observed_camera_sha256":source_surface_tensor_sha256(batch.camera_features),
                "ray_xyz_sha256":source_surface_tensor_sha256(xyz),
                "ray_delta_sha256":source_surface_tensor_sha256(delta),
                "ray_depth_sha256":source_surface_tensor_sha256(depth),
                "ray_direction_sha256":source_surface_tensor_sha256(direction),
                "keep_roles_sha256":source_surface_tensor_sha256(keep),
                "target_rgb_sha256":source_surface_tensor_sha256(target.rgb_white),
                "target_alpha_sha256":source_surface_tensor_sha256(target.alpha),
                "target_labels_sha256":source_surface_tensor_sha256(target.role_indices),
                "target_rgb_mask_sha256":source_surface_tensor_sha256(target.qualified_rgb_mask),
                "target_alpha_mask_sha256":source_surface_tensor_sha256(target.qualified_mask_mask),
                "target_role_mask_sha256":source_surface_tensor_sha256(target.qualified_role_mask),
                "semantic_role_schema_id":self.model.cfg.semantic_role_schema_id,
                "training_stage":target.stage,
                "semantic_layer_name":target.layer_name,
                "source_3d_truth_binding_sha256":target.source_3d_truth_binding_sha256,
                "generated_diffusion_truth":False,
                "independent_source_3d_render":True,
                "drawable_or_attachment_teacher":False,
            }
            if receipt is None or any(receipt.get(key)!=value for key,value in expected.items()):
                raise ValueError("RSS_VNEXT_REALSAS_JOINT_RENDER_TEACHER_RECEIPT_DRIFT")
            rendered=render_realsas_joint_from_shared_scene_v1(
                iris=self.model.iris,scene_triplanes=state.scene_triplanes,
                sample_xyz=xyz,ray_delta=delta,ray_depth=depth,
                ray_direction=direction,keep_roles=keep,
                shared_graphics_head=self.model.graphics,
                shared_semantic_role_head=self.model.presentation_role_head,
            )
            joint_parts=realsas_joint_stage_loss_v1(rendered,target)
            joint_loss=joint_parts["total"]
            joint_rays=int(target.qualified_rgb_mask.sum().item())
        # Source-rendered novel target is teacher-only: never append six
        # privileged RGBA/normal frames to the two-image scene encoder.
        novel_loss = None
        novel_count = 0
        novel_fields = (
            batch.novel_view_rgba, batch.novel_view_normal,
            batch.novel_view_rgb_mask, batch.novel_view_normal_mask,
        )
        novel_requested = (
            any(value is not None for value in novel_fields)
            or batch.novel_view_receipts is not None
            or bool(batch.novel_view_provenance)
        )
        if novel_requested:
            if not all(value is not None for value in novel_fields):
                raise ValueError("RSS_VNEXT_EIGHT_WAY_TEACHER_INCOMPLETE")
            derived_source_provenance = validate_eight_way_teacher_receipts_v1(
                observed_rgba=batch.rgba,
                source_cameras=batch.camera_features,
                novel_rgba=batch.novel_view_rgba,
                novel_normal=batch.novel_view_normal,
                rgb_mask=batch.novel_view_rgb_mask,
                normal_mask=batch.novel_view_normal_mask,
                receipts=batch.novel_view_receipts,
                trusted_registry=self.trusted_render_registry,
                dataset_manifest_sha256=self.dataset_manifest_sha256,
                source_qualification_key=batch.source_qualification_key,
            )
            if (
                batch.novel_view_provenance
                and batch.novel_view_provenance != derived_source_provenance
            ):
                raise ValueError("RSS_VNEXT_EIGHT_WAY_UNTRUSTED_PROVENANCE")
            if self.model.cfg.constructed_total_views != 8:
                raise ValueError("RSS_VNEXT_EIGHT_WAY_MODEL_CONFIGURATION_REQUIRED")
            if self.weights.novel_view <= 0:
                raise ValueError("RSS_VNEXT_EIGHT_WAY_OBJECTIVE_NOT_ENABLED")
            if batch.novel_view_rgba.ndim != 5 or batch.novel_view_rgba.shape[:3] != (batch.rgba.shape[0], 6, 4):
                raise ValueError("RSS_VNEXT_EIGHT_WAY_RGBA_TARGET_SHAPE")
            if batch.novel_view_rgba.shape[-1] < 2 or batch.novel_view_rgba.shape[-2] < 2:
                raise ValueError("RSS_VNEXT_EIGHT_WAY_TARGET_RESOLUTION")
            novel_prediction = self.model.construct_novel_views(
                state,
                height=batch.novel_view_rgba.shape[-2],
                width=batch.novel_view_rgba.shape[-1],
                method="EXPERIMENTAL_POST_SCENE_H_RENDER",
            )
            novel_objective = qualified_novel_view_loss(
                novel_prediction, batch.novel_view_rgba, batch.novel_view_normal,
                rgb_qualified_mask=batch.novel_view_rgb_mask,
                normal_qualified_mask=batch.novel_view_normal_mask,
                source_render_provenance=derived_source_provenance,
            )
            novel_loss = novel_objective["total"]
            novel_count = int(batch.novel_view_rgb_mask.sum().item())
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
            # RealSaS-style signed extraction is only meaningful in a
            # previously verified *closed signed solid* source domain.
            # A data-batch string must not grant source sign/closedness.
            signed = self.trusted_signed_source_registry.get(
                batch.flexicubes_signed_source_admission_key,
            )
            layer = batch.flexicubes_layer
            teacher = batch.flexicubes_animation_teacher
            if (
                signed is None
                or signed.get("admission_status") != "QUALIFIED_SOURCE_SIGNED_SOLID"
                or signed.get("source_component_domain") != "SINGLE_QUALIFIED_CLOSED_COMPONENT"
                or signed.get("dataset_manifest_sha256") != self.dataset_manifest_sha256
                or signed.get("source_qualification_key") != batch.source_qualification_key
                or signed.get("role_schema_id") != self.model.cfg.semantic_role_schema_id
                or signed.get("layer_name") != layer.name
                or signed.get("surface_target_sha256") != (
                    source_surface_tensor_sha256(batch.flexicubes_reference_surface_xyz)
                    if batch.flexicubes_reference_surface_xyz is not None else None
                )
                or signed.get("animation_source_asset_sha256") != (
                    teacher.source_asset_sha256 if teacher is not None else None
                )
                or signed.get("animation_source_rig_sha256") != (
                    teacher.source_rig_binding_sha256 if teacher is not None else None
                )
                or signed.get("animation_source_poses_sha256") != (
                    teacher.source_pose_suite_sha256 if teacher is not None else None
                )
            ):
                raise ValueError("RSS_VNEXT_FC_SOURCE_SIGNED_DOMAIN_RECEIPT_DRIFT")
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
                # Only source samples are supervision. Neural/proposed FC
                # vertices enter the differentiable path back into H(x).
                if target is not None and self.weights.flexicubes_surface > 0:
                    n = fc_surface.xyz.shape[1]
                    idx = torch.linspace(
                        0, n - 1, min(n, 512), device=target.device,
                    ).long()
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
        semantic_risk_loss = None
        n_semantic = 0
        semantic_requested = (
            batch.semantic_patch_target is not None
            or batch.semantic_patch_mask is not None
            or bool(batch.semantic_qualification_key)
        )
        if semantic_requested:
            if (not batch.semantic_qualification_key
                    or batch.semantic_patch_target is None
                    or batch.semantic_patch_mask is None):
                raise ValueError("RSS_VNEXT_TRAIN_SEMANTIC_QUALIFICATION_REQUIRED")
            if self.weights.source_semantics <= 0 and self.weights.semantic_risk <= 0:
                raise ValueError("RSS_VNEXT_TRAIN_SEMANTIC_OBJECTIVE_NOT_ENABLED")
            receipt = self.trusted_semantic_patch_registry.get(
                batch.semantic_qualification_key,
            )
            if receipt is None or receipt.get("admission_status") != "QUALIFIED_SOURCE_SEMANTIC_PATCH_AXES":
                raise ValueError("RSS_VNEXT_TRAIN_SEMANTIC_UNTRUSTED_ADMISSION")
            if (
                receipt.get("dataset_manifest_sha256") != self.dataset_manifest_sha256
                or receipt.get("source_qualification_key") != batch.source_qualification_key
                or receipt.get("observed_rgba_sha256") != source_surface_tensor_sha256(batch.rgba)
                or receipt.get("original_camera_sha256") != source_surface_tensor_sha256(batch.camera_features)
                or receipt.get("semantic_patch_target_sha256") != source_surface_tensor_sha256(batch.semantic_patch_target)
                or receipt.get("semantic_patch_mask_sha256") != source_surface_tensor_sha256(batch.semantic_patch_mask)
                or receipt.get("supervision_semantics") != "ORIGINAL_TWO_VIEW_MASKED_PATCH_AXES_NOT_3D_DRAWABLE"
            ):
                raise ValueError("RSS_VNEXT_TRAIN_SEMANTIC_RECEIPT_SOURCE_OR_LABEL_DRIFT")
            if (
                batch.semantic_patch_target.shape != state.source_semantic_patch_logits.shape
                or batch.semantic_patch_mask.shape != batch.semantic_patch_target.shape
                or batch.semantic_patch_mask.dtype != torch.bool
                or batch.semantic_patch_target.device != state.source_semantic_patch_logits.device
                or batch.semantic_patch_mask.device != state.source_semantic_patch_logits.device
                or not bool(batch.semantic_patch_mask.any())
            ):
                raise ValueError("RSS_VNEXT_TRAIN_SEMANTIC_TARGET_SHAPE_OR_MASK_INVALID")
            if self.weights.source_semantics > 0:
                semantic = masked_source_semantic_evidence_loss(
                    state.source_semantic_patch_logits,
                    batch.semantic_patch_target,
                    batch.semantic_patch_mask,
                )
            if self.weights.semantic_risk > 0:
                semantic_risk_loss = masked_semantic_expected_brier_risk_loss(
                    state.source_semantic_patch_logits,
                    state.source_semantic_log_uncertainty,
                    batch.semantic_patch_target,
                    batch.semantic_patch_mask,
                )
            n_semantic = int(batch.semantic_patch_mask.sum().item())

        # 3D presentation roles live on H(x), not on 2D source patches.
        # Only train this distinct head when a trusted, source-bound receipt
        # certifies the exact grid and labeled/masked teacher bytes.
        surface_role_loss = None
        n_surface_role_points = 0
        surface_fields = (
            batch.source_surface_role_indices,
            batch.source_surface_role_mask,
        )
        surface_requested = (
            any(x is not None for x in surface_fields)
            or bool(batch.source_surface_role_admission_key)
        )
        if surface_requested:
            if any(x is None for x in surface_fields):
                raise ValueError("RSS_VNEXT_SURFACE_ROLE_TEACHER_INCOMPLETE")
            if not batch.source_surface_role_admission_key:
                raise ValueError("RSS_VNEXT_SURFACE_ROLE_ADMISSION_KEY_REQUIRED")
            if self.weights.surface_roles <= 0:
                raise ValueError("RSS_VNEXT_SURFACE_ROLE_OBJECTIVE_NOT_ENABLED")
            receipt = self.trusted_surface_role_registry.get(
                batch.source_surface_role_admission_key
            )
            if receipt is None or receipt.get("admission_status") != "QUALIFIED_SOURCE_SURFACE_ROLE":
                raise ValueError("RSS_VNEXT_SURFACE_ROLE_UNTRUSTED_ADMISSION")
            if (
                receipt.get("dataset_manifest_sha256") != self.dataset_manifest_sha256
                or receipt.get("source_qualification_key") != batch.source_qualification_key
                or receipt.get("role_schema_id") != self.model.cfg.semantic_role_schema_id
                or receipt.get("query_xyz_sha256") != source_surface_tensor_sha256(state.query_xyz)
                or receipt.get("role_indices_sha256") != source_surface_tensor_sha256(batch.source_surface_role_indices)
                or receipt.get("role_mask_sha256") != source_surface_tensor_sha256(batch.source_surface_role_mask)
            ):
                raise ValueError("RSS_VNEXT_SURFACE_ROLE_RECEIPT_SOURCE_OR_ADDRESS_DRIFT")
            surface_role_loss = masked_qualified_surface_role_loss(
                state.presentation_role_logits,
                batch.source_surface_role_indices,
                batch.source_surface_role_mask,
                schema=self.model.cfg.semantic_role_schema(),
            )
            n_surface_role_points = int(batch.source_surface_role_mask.sum().item())
        region_pair_loss = None
        n_region_pairs = 0
        region_fields = (
            batch.source_region_pair_indices,
            batch.source_region_pair_is_same,
            batch.source_region_pair_mask,
        )
        region_requested = (
            any(x is not None for x in region_fields)
            or bool(batch.source_region_pair_admission_key)
        )
        if region_requested:
            if any(x is None for x in region_fields):
                raise ValueError("RSS_VNEXT_REGION_PAIR_TEACHER_INCOMPLETE")
            if not batch.source_region_pair_admission_key:
                raise ValueError("RSS_VNEXT_REGION_PAIR_ADMISSION_KEY_REQUIRED")
            if self.weights.region_pairs <= 0:
                raise ValueError("RSS_VNEXT_REGION_PAIR_OBJECTIVE_NOT_ENABLED")
            receipt = self.trusted_region_pair_registry.get(
                batch.source_region_pair_admission_key,
            )
            if receipt is None or receipt.get("admission_status") != "QUALIFIED_SOURCE_REGION_PAIR":
                raise ValueError("RSS_VNEXT_REGION_PAIR_UNTRUSTED_ADMISSION")
            if (
                receipt.get("dataset_manifest_sha256") != self.dataset_manifest_sha256
                or receipt.get("source_qualification_key") != batch.source_qualification_key
                or receipt.get("query_xyz_sha256") != source_surface_tensor_sha256(state.query_xyz)
                or receipt.get("pair_indices_sha256") != source_surface_tensor_sha256(batch.source_region_pair_indices)
                or receipt.get("pair_targets_sha256") != source_surface_tensor_sha256(batch.source_region_pair_is_same)
                or receipt.get("pair_mask_sha256") != source_surface_tensor_sha256(batch.source_region_pair_mask)
                or receipt.get("supervision_semantics") != "SOURCE_REGION_CO_MEMBERSHIP_NOT_DRAWABLE"
            ):
                raise ValueError("RSS_VNEXT_REGION_PAIR_RECEIPT_SOURCE_OR_ADDRESS_DRIFT")
            if batch.source_region_pair_indices.shape[1] > self.model.cfg.max_pair_queries:
                raise ValueError("RSS_VNEXT_REGION_PAIR_RESOURCE_LIMIT")
            region_pair_loss = qualified_source_region_pair_loss(
                state.semantic_logits,
                batch.source_region_pair_indices,
                batch.source_region_pair_is_same,
                batch.source_region_pair_mask,
            )
            n_region_pairs = int(batch.source_region_pair_mask.sum().item())
        # Source-bound typed pair relations, unlike anonymous slot co-membership,
        # have THREE distinct qualifications. A source semantic part ID cannot
        # masquerade as a drawable edit-consequence or mechanical attachment.
        typed_relation_loss = None
        typed_relation_counts = (0, 0, 0)
        typed_fields = (
            batch.source_typed_relation_pair_indices,
            batch.source_typed_relation_target,
            batch.source_typed_relation_mask,
        )
        typed_requested = (
            any(x is not None for x in typed_fields)
            or bool(batch.source_typed_relation_admission_key)
        )
        if typed_requested:
            if any(x is None for x in typed_fields):
                raise ValueError("RSS_VNEXT_TYPED_RELATION_TEACHER_INCOMPLETE")
            if not batch.source_typed_relation_admission_key:
                raise ValueError("RSS_VNEXT_TYPED_RELATION_ADMISSION_KEY_REQUIRED")
            if self.weights.typed_relations <= 0:
                raise ValueError("RSS_VNEXT_TYPED_RELATION_OBJECTIVE_NOT_ENABLED")
            receipt = self.trusted_typed_relation_registry.get(
                batch.source_typed_relation_admission_key,
            )
            if receipt is None or receipt.get("admission_status") != "QUALIFIED_SOURCE_TYPED_REGION_RELATIONS":
                raise ValueError("RSS_VNEXT_TYPED_RELATION_UNTRUSTED_ADMISSION")
            channels = receipt.get("channel_source_authority")
            verified_kinds = (
                "QUALIFIED_SOURCE_EDIT_DRAWABLE_CONSEQUENCE",
                "QUALIFIED_SOURCE_GEOMETRIC_SEPARABILITY",
                "QUALIFIED_SOURCE_CONTACT_AND_MOTION_INTERVENTION",
            )
            if (
                receipt.get("dataset_manifest_sha256") != self.dataset_manifest_sha256
                or receipt.get("source_qualification_key") != batch.source_qualification_key
                or receipt.get("query_xyz_sha256") != source_surface_tensor_sha256(state.query_xyz)
                or receipt.get("pair_indices_sha256") != source_surface_tensor_sha256(batch.source_typed_relation_pair_indices)
                or receipt.get("typed_targets_sha256") != source_surface_tensor_sha256(batch.source_typed_relation_target)
                or receipt.get("typed_mask_sha256") != source_surface_tensor_sha256(batch.source_typed_relation_mask)
                or receipt.get("original_rgba_sha256") != source_surface_tensor_sha256(batch.rgba)
                or receipt.get("original_camera_sha256") != source_surface_tensor_sha256(batch.camera_features)
                or receipt.get("supervision_semantics") != "SOURCE_QUERY_PAIR_TYPED_DRAWABLE_CUT_DIRECTED_ATTACHMENT_MASKED_V1"
                or not isinstance(channels, dict)
                or set(channels) != set(TYPED_REGION_RELATION_CHANNELS_V1)
                or batch.source_typed_relation_pair_indices.shape[1] > self.model.cfg.max_pair_queries
            ):
                raise ValueError("RSS_VNEXT_TYPED_RELATION_RECEIPT_SOURCE_OR_ADDRESS_DRIFT")
            for index, name in enumerate(TYPED_REGION_RELATION_CHANNELS_V1):
                is_admitted = bool(batch.source_typed_relation_mask[..., index].any())
                expected = verified_kinds[index] if is_admitted else "UNAVAILABLE"
                if channels[name] != expected:
                    raise ValueError("RSS_VNEXT_TYPED_RELATION_CHANNEL_AUTHORITY_INVALID:" + name)
            typed_relation_loss, typed_relation_counts = qualified_source_typed_relation_loss(
                state.semantic_logits,
                state.region_relation_logits,
                state.region_relation_valid_mask,
                batch.source_typed_relation_pair_indices,
                batch.source_typed_relation_target,
                batch.source_typed_relation_mask,
            )

        # Model geometry heads are learned separately. Source UDF remains
        # meaningful on open/uncertain surfaces; sign targets require a
        # closed-solid proof. Neither is made authoritative by a float label.
        source_udf_loss = None
        source_inside_sign_loss = None
        n_source_udf = 0
        n_source_inside_sign = 0
        field_items = (
            (batch.source_udf_target, batch.source_udf_mask),
            (batch.source_inside_sign_target, batch.source_inside_sign_mask),
        )
        field_requested = bool(batch.source_field_admission_key) or any(
            target is not None or mask is not None
            for target, mask in field_items
        )
        if field_requested:
            if not batch.source_field_admission_key or not any(
                target is not None for target, _ in field_items
            ):
                raise ValueError("RSS_VNEXT_SOURCE_FIELD_TEACHER_INCOMPLETE")
            if any((target is None) != (mask is None) for target, mask in field_items):
                raise ValueError("RSS_VNEXT_SOURCE_FIELD_TEACHER_INCOMPLETE")
            receipt = self.trusted_source_field_registry.get(
                batch.source_field_admission_key,
            )
            if (
                receipt is None
                or receipt.get("admission_status") != "QUALIFIED_SOURCE_QUERY_DISTANCE_FIELDS"
                or receipt.get("dataset_manifest_sha256") != self.dataset_manifest_sha256
                or receipt.get("source_qualification_key") != batch.source_qualification_key
                or receipt.get("query_xyz_sha256") != source_surface_tensor_sha256(state.query_xyz)
                or receipt.get("source_coordinate_system") != "IRIS_NORMALIZED_SOURCE_XYZ"
                or receipt.get("udf_target_sha256") != (
                    source_surface_tensor_sha256(batch.source_udf_target)
                    if batch.source_udf_target is not None else None
                )
                or receipt.get("udf_mask_sha256") != (
                    source_surface_tensor_sha256(batch.source_udf_mask)
                    if batch.source_udf_mask is not None else None
                )
                or receipt.get("inside_target_sha256") != (
                    source_surface_tensor_sha256(batch.source_inside_sign_target)
                    if batch.source_inside_sign_target is not None else None
                )
                or receipt.get("inside_mask_sha256") != (
                    source_surface_tensor_sha256(batch.source_inside_sign_mask)
                    if batch.source_inside_sign_mask is not None else None
                )
                or receipt.get("teacher_semantics") != "SOURCE_UDF_AND_OPTIONAL_NEGATIVE_INSIDE_SIGN"
                or receipt.get("signed_domain") not in (
                    "SINGLE_QUALIFIED_CLOSED_COMPONENT",
                    "OPEN_OR_UNKNOWN_UNSIGNED_ONLY",
                )
            ):
                raise ValueError("RSS_VNEXT_SOURCE_FIELD_RECEIPT_DRIFT")
            if (
                batch.source_inside_sign_target is not None
                and receipt.get("signed_domain") != "SINGLE_QUALIFIED_CLOSED_COMPONENT"
            ):
                raise ValueError("RSS_VNEXT_SOURCE_SIGN_CLOSED_DOMAIN_REQUIRED")
            expected_shape = state.geometry["unsigned_distance"].shape
            for target, mask in field_items:
                if target is None:
                    continue
                if (
                    target.shape != expected_shape or mask.shape != expected_shape
                    or not torch.is_floating_point(target)
                    or mask.dtype != torch.bool
                    or target.device != state.query_xyz.device
                    or mask.device != state.query_xyz.device
                    or not bool(mask.any())
                ):
                    raise ValueError("RSS_VNEXT_SOURCE_FIELD_TARGET_SHAPE_OR_MASK_INVALID")
                if not bool(torch.isfinite(target[mask]).all()):
                    raise ValueError("RSS_VNEXT_SOURCE_FIELD_TARGET_NONFINITE")
            if batch.source_udf_target is not None:
                if self.weights.source_udf <= 0:
                    raise ValueError("RSS_VNEXT_SOURCE_UDF_OBJECTIVE_NOT_ENABLED")
                target, mask = field_items[0]
                if bool((target[mask] < 0).any()):
                    raise ValueError("RSS_VNEXT_SOURCE_UDF_TARGET_NEGATIVE")
                source_udf_loss = F.smooth_l1_loss(
                    state.geometry["unsigned_distance"][mask], target[mask],
                )
                n_source_udf = int(mask.sum().item())
            if batch.source_inside_sign_target is not None:
                if self.weights.source_inside_sign <= 0:
                    raise ValueError("RSS_VNEXT_SOURCE_SIGN_OBJECTIVE_NOT_ENABLED")
                target, mask = field_items[1]
                if not bool(((target[mask] == 0) | (target[mask] == 1)).all()):
                    raise ValueError("RSS_VNEXT_SOURCE_SIGN_TARGET_NOT_BINARY")
                source_inside_sign_loss = F.binary_cross_entropy_with_logits(
                    state.geometry["sign_logit"][mask], target[mask],
                )
                n_source_inside_sign = int(mask.sum().item())

        # Graphics can learn source-visible RGB already emitted at H(x)
        # queries. This does NOT certify depth, density, occlusion, normals or
        # drawable identity. Each admitted query must be tied to one original
        # ray; no inferred view may enter this objective as source truth.
        graphics_rgb_loss = None
        n_graphics_rgb = 0
        rgb_fields = (
            batch.source_graphics_rgb_target,
            batch.source_graphics_view_directions,
            batch.source_graphics_rgb_mask,
        )
        rgb_requested = (
            any(value is not None for value in rgb_fields)
            or bool(batch.source_graphics_rgb_admission_key)
        )
        if rgb_requested:
            if any(value is None for value in rgb_fields):
                raise ValueError("RSS_VNEXT_GRAPHICS_RGB_SOURCE_TEACHER_INCOMPLETE")
            if self.weights.graphics_source_rgb <= 0:
                raise ValueError("RSS_VNEXT_GRAPHICS_RGB_OBJECTIVE_NOT_ENABLED")
            teacher_rgb, source_directions, source_mask = rgb_fields
            if (
                teacher_rgb.shape != (*state.graphics_logits.shape[:2], 3)
                or source_directions.shape != (*state.graphics_logits.shape[:2], 3)
                or source_mask.shape != state.graphics_logits.shape[:2]
                or source_mask.dtype != torch.bool
                or source_directions.device != state.graphics_logits.device
                or not bool(torch.isfinite(source_directions).all())
                or not bool(torch.allclose(
                    source_directions.norm(dim=-1)[source_mask],
                    torch.ones_like(source_directions.norm(dim=-1)[source_mask]),
                    atol=1e-3, rtol=0,
                ))
                or teacher_rgb.device != state.graphics_logits.device
                or source_mask.device != state.graphics_logits.device
                or not bool(source_mask.any())
                or not bool(torch.isfinite(teacher_rgb).all())
                or bool(((teacher_rgb < 0) | (teacher_rgb > 1)).any())
            ):
                raise ValueError("RSS_VNEXT_GRAPHICS_RGB_SOURCE_TARGET_INVALID")
            receipt = self.trusted_graphics_rgb_registry.get(
                batch.source_graphics_rgb_admission_key,
            )
            if (
                receipt is None
                or receipt.get("admission_status") != "QUALIFIED_SOURCE_VISIBLE_RGB_QUERIES"
                or receipt.get("dataset_manifest_sha256") != self.dataset_manifest_sha256
                or receipt.get("source_qualification_key") != batch.source_qualification_key
                or receipt.get("query_xyz_sha256") != source_surface_tensor_sha256(state.query_xyz)
                or receipt.get("rgb_target_sha256") != source_surface_tensor_sha256(teacher_rgb)
                or receipt.get("source_ray_direction_sha256") != source_surface_tensor_sha256(source_directions)
                or receipt.get("rgb_mask_sha256") != source_surface_tensor_sha256(source_mask)
                or receipt.get("original_rgba_sha256") != source_surface_tensor_sha256(batch.rgba)
                or receipt.get("original_camera_sha256") != source_surface_tensor_sha256(batch.camera_features)
                or receipt.get("teacher_semantics") != "ORIGINAL_OPAQUE_FIRST_HIT_RGB_NOT_TEXTURE_OR_DEPTH"
                or receipt.get("visibility_mechanism") != "SOURCE_ZBUFFER_FIRST_HIT_OPAQUE"
            ):
                raise ValueError("RSS_VNEXT_GRAPHICS_RGB_SOURCE_RECEIPT_DRIFT")
            # The Graphics head's RGB channels are additive logits, not
            # standalone RGB! Train through the SAME local ray-direction
            # decoder used by the experimental renderer's sampled field.
            # First-hit opaque source evidence is required: translucent
            # multi-hit source-over cannot be supervised by one RGB target.
            ray_latent = self.model.constructed_view_renderer.core(
                torch.cat((state.local_features, source_directions), dim=-1)
            )
            predicted_rgb = torch.sigmoid(
                self.model.constructed_view_renderer.rgb(ray_latent)
                + state.graphics_logits[..., :3]
            )
            graphics_rgb_loss = F.smooth_l1_loss(
                predicted_rgb[source_mask], teacher_rgb[source_mask],
            )
            n_graphics_rgb = int(source_mask.sum().item())
        axis_loss = None
        n_axis = 0
        axis_conditioning = "NO_AXIS_OBJECTIVE"
        matched_axis_mesh = None
        matched_axis_memory = None
        axis_requested = batch.axis_teacher_sequence is not None or bool(batch.axis_teacher_admission_key)
        if axis_requested:
            if batch.axis_teacher_sequence is None or not batch.axis_teacher_admission_key:
                raise ValueError("RSS_VNEXT_AXIS_TEACHER_INCOMPLETE")
            if self.weights.axis_rig_skin_codes <= 0:
                raise ValueError("RSS_VNEXT_AXIS_TEACHER_OBJECTIVE_NOT_ENABLED")
            teacher = batch.axis_teacher_sequence
            receipt = self.trusted_axis_teacher_registry.get(batch.axis_teacher_admission_key)
            if receipt is None or receipt.get("admission_status") != "QUALIFIED_SOURCE_RIG_SKIN_TOKEN_TEACHER":
                raise ValueError("RSS_VNEXT_AXIS_TEACHER_UNTRUSTED")
            control_qualified = (
                receipt.get("intervention_qualification_status")
                == "QUALIFIED_SOURCE_CONTROL_CONSEQUENCE"
                and isinstance(receipt.get("source_intervention_sha256"), str)
                and len(receipt.get("source_intervention_sha256", "")) == 64
            )
            if (
                receipt.get("dataset_manifest_sha256") != self.dataset_manifest_sha256
                or receipt.get("source_qualification_key") != batch.source_qualification_key
                or receipt.get("architecture_id") != AXIS_V6_PRODUCT_ARCHITECTURE_ID
                or receipt.get("teacher_sequence_sha256") != axis_teacher_sequence_sha256_v6(teacher)
                or receipt.get("mira_codec_state_sha256") != frozen_mira_codec_state_sha256_v6(self.model)
                or receipt.get("teacher_semantics") != "SOURCE_RIG_AND_PINNED_FSQ_NOT_INFERRED_CONTROLS"
                or any(parameter.requires_grad for parameter in self.model.mira.codec.parameters())
            ):
                raise ValueError("RSS_VNEXT_AXIS_TEACHER_SOURCE_CODEC_OR_SEQUENCE_DRIFT")
            if self.axis_training_policy == "PRODUCT_MESH_MATCHED_REQUIRED":
                # Inference conditions AXIS on TESSA's actual candidate mesh.
                # Train on the exact decoded/indexed topology, not an
                # unaddressed IRIS-only memory or a second teacher scene.
                tokens = batch.topology_teacher_tokens
                mask = batch.topology_valid_mask
                if (
                    tokens is None or mask is None
                    or tokens.ndim != 2 or tokens.shape[0] != 1
                    or tokens.dtype != torch.long
                    or mask.shape != (1,) or mask.dtype != torch.bool
                    or not bool(mask[0])
                    or state.scene_triplanes.shape[0] != 1
                ):
                    raise ValueError("RSS_VNEXT_AXIS_PRODUCT_MATCHED_TESSA_MESH_REQUIRED")
                if (
                    receipt.get("mesh_teacher_semantics")
                    != "SOURCE_NATIVE_TESSA_TEACHER_TOPOLOGY_FOR_AXIS_PRODUCT"
                    or receipt.get("mesh_teacher_source_reindex_proven") is not True
                    or receipt.get("mesh_teacher_tokens_sha256")
                    != source_surface_tensor_sha256(tokens)
                    or receipt.get("mesh_teacher_valid_mask_sha256")
                    != source_surface_tensor_sha256(mask)
                    or receipt.get("observed_rgba_sha256")
                    != source_surface_tensor_sha256(batch.rgba)
                    or receipt.get("observed_camera_sha256")
                    != source_surface_tensor_sha256(batch.camera_features)
                ):
                    raise ValueError("RSS_VNEXT_AXIS_PRODUCT_TESSA_SOURCE_BINDING_DRIFT")
                matched_axis_mesh = self.model.build_mesh_candidate(
                    state, tokens[0],
                    source_authority="TEACHER_FORCED_TRAINING_ONLY",
                )
                if (
                    receipt.get("mesh_teacher_topology_digest")
                    != axis_mesh_teacher_address_digest_v1(matched_axis_mesh)
                ):
                    raise ValueError("RSS_VNEXT_AXIS_PRODUCT_TESSA_FACE_REINDEX_DRIFT")
                matched_axis_memory = self.model._control_memory(
                    state, memory, matched_axis_mesh,
                )
                axis_conditioning = "PRODUCT_MESH_MATCHED_TESSA_H"
            else:
                # Rig-only can pretrain causal grammar, but it must not
                # masquerade as product-context learning or label a socket
                # whose held/holder mesh is unavailable.
                if bool((teacher.kind_ids == int(RigTokenKindV6.ANCHOR)).any()):
                    raise ValueError("RSS_VNEXT_AXIS_RIG_ONLY_ANCHOR_REQUIRES_MESH")
                matched_axis_memory = self.model._control_memory(state, memory)
                axis_conditioning = "RIG_ONLY_PRETRAIN_NOT_PRODUCT_PARITY"
            axis_loss, _axis_components = qualified_axis_rig_skin_sequence_loss_v6(
                self.model.axis,
                matched_axis_memory,
                teacher,
                tokens_per_joint=self.model.cfg.skin_tokens_per_joint,
                executable_constraint_operators=RIG_CONSTRAINT_OPERATOR_IDS_V1,
                control_intervention_qualified=control_qualified,
            )
            n_axis = int(teacher.kind_ids.shape[0])
        anchor_face_loss = None
        eligible_anchor_held = eligible_anchor_holder = eligible_anchor_negative = 0
        anchor_requested = (
            batch.source_anchor_face_targets is not None
            or batch.source_anchor_face_mask is not None
            or bool(batch.source_anchor_face_admission_key)
        )
        if anchor_requested:
            if (
                batch.source_anchor_face_targets is None
                or batch.source_anchor_face_mask is None
                or not batch.source_anchor_face_admission_key
                or batch.axis_teacher_sequence is None
                or not batch.axis_teacher_admission_key
                or batch.topology_teacher_tokens is None
                or batch.topology_valid_mask is None
            ):
                raise ValueError("RSS_VNEXT_SOCKET_FACE_TRAIN_TEACHER_INCOMPLETE")
            if self.weights.anchor_face_correspondence <= 0:
                raise ValueError("RSS_VNEXT_SOCKET_FACE_TRAIN_WEIGHT_REQUIRED")
            if self.axis_training_policy != "PRODUCT_MESH_MATCHED_REQUIRED":
                raise ValueError("RSS_VNEXT_SOCKET_FACE_MESH_MATCHED_PRODUCT_POLICY_REQUIRED")
            if matched_axis_mesh is None or matched_axis_memory is None:
                raise ValueError("RSS_VNEXT_SOCKET_FACE_MATCHED_AXIS_MEMORY_MISSING")
            source_registry = self.trusted_anchor_face_registry
            row = source_registry.get(batch.source_anchor_face_admission_key)
            axis_row = self.trusted_axis_teacher_registry.get(
                batch.axis_teacher_admission_key,
            )
            teacher = batch.axis_teacher_sequence
            if (
                row is None
                or row.get("admission_status")
                != "QUALIFIED_SOURCE_NATIVE_ATTACHMENT_FACE_REINDEX"
                or row.get("source_qualification_key") != batch.source_qualification_key
                or row.get("dataset_manifest_sha256") != self.dataset_manifest_sha256
                or row.get("observed_rgba_sha256") != source_surface_tensor_sha256(batch.rgba)
                or row.get("observed_camera_sha256") != source_surface_tensor_sha256(batch.camera_features)
                or axis_row is None
                or axis_row.get("admission_status") != "QUALIFIED_SOURCE_RIG_SKIN_TOKEN_TEACHER"
                or axis_row.get("dataset_manifest_sha256") != self.dataset_manifest_sha256
                or axis_row.get("source_qualification_key") != batch.source_qualification_key
                or axis_row.get("architecture_id") != AXIS_V6_PRODUCT_ARCHITECTURE_ID
                or axis_row.get("teacher_sequence_sha256") != axis_teacher_sequence_sha256_v6(teacher)
                or row.get("axis_teacher_sequence_sha256") != axis_sequence_digest_v1(teacher)
            ):
                raise ValueError("RSS_VNEXT_SOCKET_FACE_SOURCE_OR_AXIS_TEACHER_RECEIPT_DRIFT")
            # Reuse THE SAME mesh object and causal memory as the AXIS
            # teacher-forced objective; do not build a second source scene.
            control_hidden = self.model.axis(
                matched_axis_memory, teacher,
            )["hidden"]
            predicted = self.model.anchor_face_correspondence(
                mesh=matched_axis_mesh, axis_hidden=control_hidden,
                axis_sequence=teacher,
                region_relation_logits=state.region_relation_logits,
            )
            face_objective = source_qualified_socket_face_loss_v1(
                predicted,
                source_face_targets=batch.source_anchor_face_targets,
                source_face_mask=batch.source_anchor_face_mask,
                source_asset_sha256=row["source_asset_sha256"],
                dataset_manifest_sha256=self.dataset_manifest_sha256,
                source_admission_key=batch.source_anchor_face_admission_key,
                trusted_registry=source_registry,
            )
            anchor_face_loss = face_objective.loss
            eligible_anchor_held = face_objective.eligible_held
            eligible_anchor_holder = face_objective.eligible_holder
            eligible_anchor_negative = face_objective.eligible_negative
        elif self.weights.anchor_face_correspondence > 0 and axis_requested:
            # No source-native attachment labels: opt-in weight cannot
            # turn absent contact evidence into a synthetic negative.
            raise ValueError("RSS_VNEXT_SOCKET_FACE_SOURCE_LABELS_REQUIRED")
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
            and semantic_risk_loss is None
            and fc_loss is None and fc_anim_loss is None
            and novel_loss is None and surface_role_loss is None
            and region_pair_loss is None
            and typed_relation_loss is None
            and graphics_rgb_loss is None and axis_loss is None
            and anchor_face_loss is None
            and source_udf_loss is None and source_inside_sign_loss is None
            and joint_loss is None
        ):
            raise ValueError("RSS_VNEXT_TRAIN_NO_QUALIFIED_OBJECTIVE")
        total = sum(
            weight * loss
            for weight, loss in (
                (self.weights.topology, topology),
                (self.weights.motion_preference, motion),
                (self.weights.source_semantics, semantic),
                (self.weights.semantic_risk, semantic_risk_loss),
                (self.weights.surface_roles, surface_role_loss),
                (self.weights.region_pairs, region_pair_loss),
                (self.weights.typed_relations, typed_relation_loss),
                (self.weights.graphics_source_rgb, graphics_rgb_loss),
                (self.weights.source_udf, source_udf_loss),
                (self.weights.source_inside_sign, source_inside_sign_loss),
                (self.weights.axis_rig_skin_codes, axis_loss),
                (self.weights.anchor_face_correspondence, anchor_face_loss),
                (self.weights.flexicubes_surface, fc_loss),
                (self.weights.flexicubes_animation, fc_anim_loss),
                (self.weights.novel_view, novel_loss),
                (self.weights.realsas_joint_slrm, joint_loss),
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
            semantic_risk_loss=semantic_risk_loss,
            flexicubes_surface_loss=fc_loss,
            flexicubes_animation_loss=fc_anim_loss,
            eligible_animation_pose_count=fc_anim_count,
            eligible_semantic_patch_labels=n_semantic,
            eligible_topology_examples=n_topo,
            qualified_preference_pairs=n_pairs,
            novel_view_loss=novel_loss,
            eligible_novel_view_pixels=novel_count,
            surface_role_loss=surface_role_loss,
            eligible_surface_role_points=n_surface_role_points,
            region_pair_loss=region_pair_loss,
            eligible_region_pairs=n_region_pairs,
            typed_relation_loss=typed_relation_loss,
            eligible_typed_relation_channels=typed_relation_counts,
            graphics_source_rgb_loss=graphics_rgb_loss,
            eligible_graphics_source_rgb_queries=n_graphics_rgb,
            source_udf_loss=source_udf_loss,
            source_inside_sign_loss=source_inside_sign_loss,
            eligible_source_udf_queries=n_source_udf,
            eligible_source_inside_sign_queries=n_source_inside_sign,
            axis_rig_skin_loss=axis_loss,
            eligible_axis_sequences=n_axis,
            axis_training_conditioning=axis_conditioning,
            anchor_face_correspondence_loss=anchor_face_loss,
            eligible_anchor_held_faces=eligible_anchor_held,
            eligible_anchor_holder_faces=eligible_anchor_holder,
            eligible_anchor_negative_faces=eligible_anchor_negative,
            realsas_joint_slrm_loss=joint_loss,
            eligible_realsas_joint_rays=joint_rays,
        )

    def _measure_gradient_conflicts(self, result: RSSvNextTrainingResult) -> dict:
        # The same independent qualified terms that compose total_loss.
        # A diagnostic never reweights, backpropagates, or rewrites .grad.
        candidates = (
            ("topology", result.topology_loss, self.weights.topology),
            ("motion_preference", result.motion_preference_loss, self.weights.motion_preference),
            ("source_semantics", result.source_semantic_loss, self.weights.source_semantics),
            ("semantic_risk", result.semantic_risk_loss, self.weights.semantic_risk),
            ("surface_roles", result.surface_role_loss, self.weights.surface_roles),
            ("region_pairs", result.region_pair_loss, self.weights.region_pairs),
            ("typed_relations", result.typed_relation_loss, self.weights.typed_relations),
            ("graphics_source_rgb", result.graphics_source_rgb_loss, self.weights.graphics_source_rgb),
            ("source_udf", result.source_udf_loss, self.weights.source_udf),
            ("source_inside_sign", result.source_inside_sign_loss, self.weights.source_inside_sign),
            ("axis_rig_skin_codes", result.axis_rig_skin_loss, self.weights.axis_rig_skin_codes),
            ("anchor_face_correspondence", result.anchor_face_correspondence_loss, self.weights.anchor_face_correspondence),
            ("flexicubes_surface", result.flexicubes_surface_loss, self.weights.flexicubes_surface),
            ("flexicubes_animation", result.flexicubes_animation_loss, self.weights.flexicubes_animation),
            ("novel_view", result.novel_view_loss, self.weights.novel_view),
            ("realsas_joint_slrm", result.realsas_joint_slrm_loss, self.weights.realsas_joint_slrm),
        )
        report = audit_rss_vnext_gradient_interference_v1(
            named_parameters=tuple(self.model.named_parameters()),
            loss_terms={
                key: (loss, float(weight))
                for key, loss, weight in candidates
                if loss is not None and weight > 0
            },
            policy=self.gradient_court_policy,
        )
        report["global_step_before_update"] = self.global_step
        report["dataset_manifest_sha256"] = self.dataset_manifest_sha256
        report["objective_policy_sha256"] = self.objective_policy_sha256
        return report

    def _assert_optimizer_ownership(self) -> None:
        current = qualify_rss_vnext_optimizer_ownership_v1(
            self.model, self.optimizer,
        )
        if current.binding_sha256 != self.optimizer_ownership.binding_sha256:
            raise RuntimeError("RSS_VNEXT_TRAIN_OPTIMIZER_PHASE_IDENTITY_DRIFT")

    def train_step(self, batch: RSSvNextStructuralTrainingBatch) -> RSSvNextTrainingResult:
        self._assert_optimizer_ownership()
        # Phase changes may leave stale frozen tensors from a former
        # optimizer. Never let old .grad affect the new stage's clipping.
        for parameter in self.model.parameters():
            if not parameter.requires_grad:
                parameter.grad = None
        self.optimizer.zero_grad(set_to_none=True)
        result = self.objective(batch)
        if (
            self.gradient_court_policy.every_n_steps > 0
            and self.global_step % self.gradient_court_policy.every_n_steps == 0
        ):
            result.gradient_interference_report = self._measure_gradient_conflicts(result)
        result.total_loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(
            (p for p in self.model.parameters() if p.requires_grad),
            self.weights.clip_gradient_norm,
            error_if_nonfinite=True,
        )
        self.optimizer.step()
        self.global_step += 1
        result.gradient_norm = grad_norm
        return result

    def save_checkpoint(self, path: str | Path, *, sampler_cursor: int) -> None:
        if sampler_cursor < 0:
            raise ValueError("RSS_VNEXT_TRAIN_SAMPLER_CURSOR_INVALID")
        self._assert_optimizer_ownership()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": self.checkpoint_schema,
            "architecture_id": self.model.architecture_id,
            "architecture_digest": architecture_digest(self.model.cfg),
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "objective_policy_sha256": self.objective_policy_sha256,
            "optimizer_ownership_binding_sha256": self.optimizer_ownership.binding_sha256,
            "render_registry_sha256": self.render_registry_sha256,
            "generated_conditioning_registry_sha256": self.generated_conditioning_registry_sha256,
            "surface_role_registry_sha256": self.surface_role_registry_sha256,
            "semantic_patch_registry_sha256": self.semantic_patch_registry_sha256,
            "region_pair_registry_sha256": self.region_pair_registry_sha256,
            "typed_relation_registry_sha256": self.typed_relation_registry_sha256,
            "axis_teacher_registry_sha256": self.axis_teacher_registry_sha256,
            "anchor_face_registry_sha256": self.anchor_face_registry_sha256,
            "signed_source_registry_sha256": self.signed_source_registry_sha256,
            "graphics_rgb_registry_sha256": self.graphics_rgb_registry_sha256,
            "source_field_registry_sha256": self.source_field_registry_sha256,
            "realsas_joint_registry_sha256": self.realsas_joint_registry_sha256,
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
        if data.get("objective_policy_sha256") != self.objective_policy_sha256:
            raise ValueError("RSS_VNEXT_TRAIN_CHECKPOINT_OBJECTIVE_POLICY_DRIFT")
        self._assert_optimizer_ownership()
        if data.get("optimizer_ownership_binding_sha256") != self.optimizer_ownership.binding_sha256:
            raise ValueError("RSS_VNEXT_TRAIN_CHECKPOINT_OPTIMIZER_OWNERSHIP_DRIFT")
        if data.get("render_registry_sha256", "") != self.render_registry_sha256:
            raise ValueError("RSS_VNEXT_TRAIN_CHECKPOINT_RENDER_REGISTRY_DRIFT")
        if data.get("generated_conditioning_registry_sha256", "") != self.generated_conditioning_registry_sha256:
            raise ValueError("RSS_VNEXT_TRAIN_CHECKPOINT_GENERATOR_REGISTRY_DRIFT")
        if data.get("surface_role_registry_sha256", "") != self.surface_role_registry_sha256:
            raise ValueError("RSS_VNEXT_TRAIN_CHECKPOINT_SURFACE_ROLE_REGISTRY_DRIFT")
        if data.get("semantic_patch_registry_sha256", "") != self.semantic_patch_registry_sha256:
            raise ValueError("RSS_VNEXT_TRAIN_CHECKPOINT_SEMANTIC_PATCH_REGISTRY_DRIFT")
        if data.get("region_pair_registry_sha256", "") != self.region_pair_registry_sha256:
            raise ValueError("RSS_VNEXT_TRAIN_CHECKPOINT_REGION_PAIR_REGISTRY_DRIFT")
        if data.get("typed_relation_registry_sha256", "") != self.typed_relation_registry_sha256:
            raise ValueError("RSS_VNEXT_TRAIN_CHECKPOINT_TYPED_RELATION_REGISTRY_DRIFT")
        if data.get("axis_teacher_registry_sha256", "") != self.axis_teacher_registry_sha256:
            raise ValueError("RSS_VNEXT_TRAIN_CHECKPOINT_AXIS_TEACHER_REGISTRY_DRIFT")
        if data.get("anchor_face_registry_sha256", "") != self.anchor_face_registry_sha256:
            raise ValueError("RSS_VNEXT_TRAIN_CHECKPOINT_ANCHOR_FACE_REGISTRY_DRIFT")
        if data.get("signed_source_registry_sha256", "") != self.signed_source_registry_sha256:
            raise ValueError("RSS_VNEXT_TRAIN_CHECKPOINT_SIGNED_SOURCE_REGISTRY_DRIFT")
        if data.get("graphics_rgb_registry_sha256", "") != self.graphics_rgb_registry_sha256:
            raise ValueError("RSS_VNEXT_TRAIN_CHECKPOINT_GRAPHICS_RGB_REGISTRY_DRIFT")
        if data.get("realsas_joint_registry_sha256", "") != self.realsas_joint_registry_sha256:
            raise ValueError("RSS_VNEXT_TRAIN_CHECKPOINT_REALSAS_JOINT_REGISTRY_DRIFT")
        if data.get("source_field_registry_sha256", "") != self.source_field_registry_sha256:
            raise ValueError("RSS_VNEXT_TRAIN_CHECKPOINT_SOURCE_FIELD_REGISTRY_DRIFT")
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
    "architecture_digest", "source_surface_tensor_sha256",
    "axis_teacher_sequence_sha256_v6", "frozen_mira_codec_state_sha256_v6",
    "axis_mesh_teacher_address_digest_v1",
]
