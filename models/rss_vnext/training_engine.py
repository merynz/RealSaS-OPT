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
from .training_seeded_tessa_v1 import (
    RSSvNextSeededTESSATrainerV1, SeededTESSALossWeightsV1,
    SeededTESSASourceTeacherV1, SCHEMA as SEEDED_TESSA_SCHEMA,
    PHASE as SEEDED_TESSA_PHASE,
)
from .seeded_surface_sampling_v1 import SeededSurfaceSamplingPolicyV1
from .layered_iris_seed_v1 import (
    make_layered_iris_seed_v1, layer_namespace_digest_v1,
)
from .seeded_layer_source_court_v1 import (
    SOURCE_LAYER_AUTHORITY,SOURCE_LAYER_DERIVATION,
)
from .training_seeded_tessa_v1 import _digest as seeded_source_tensor_digest_v1
from .tessa_seeded_surface_v1 import tessa_seed_from_flexicubes_v1
from .component_conditioned_udf_v1 import (
    SourceNativeComponentUDFFitV1,
    source_native_component_udf_objective_v1,
)
from .native_face_layer_truth_v1 import tensor_sha256_v1 as native_tensor_sha256_v1
from .product_graphics_supervision_v1 import (
    product_source_graphics_loss_v1,
    product_source_joint_ray_loss_v1,
)
from .product_semantic_supervision_v1 import (
    ProductSemanticWeightsV1,
    product_source_semantic_objective_v1,
)
from .product_axis_mira_seeded_training_v1 import (
    ProductSeededAXISMIRAWeightsV1,
    product_seeded_axis_mira_objective_v1,
)


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
    source_topology_admission_key: str = ""
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
    source_signed_sdf_target: Tensor | None = None
    source_signed_sdf_mask: Tensor | None = None
    source_inside_sign_target: Tensor | None = None    # [B,Q] float 1=inside
    source_inside_sign_mask: Tensor | None = None      # [B,Q] bool
    source_field_admission_key: str = ""
    # SkinTokens-style causal joint+FSQ sequence, with independent teacher
    # admission. Control/constraint tokens additionally need intervention truth.
    axis_teacher_sequence: AXISSequenceV6 | None = None
    axis_teacher_admission_key: str = ""
    # Externally admitted native joint -> mechanical chain partition. -1=UNKNOWN.
    axis_chain_joint_ids: Tensor | None = None  # [J] aligned to native AXIS order
    axis_chain_vertex_mask: Tensor | None = None  # [1,V] exact final M* vertex order
    axis_chain_teacher_admission_key: str = ""
    # Independent source-native, LBS-qualified pose surface samples in the
    # exact matched TESSA mesh/AXIS joint domain; never generated-view truth.
    axis_pose_joint_skin_matrices: Tensor | None = None  # [1,P,J,4,4]
    axis_pose_rest_surface_samples: Tensor | None = None  # [S,3]
    axis_pose_surface_samples: Tensor | None = None  # [P,S,3]
    axis_pose_valid_mask: Tensor | None = None  # [P] bool
    axis_pose_admission_key: str = ""
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
    # Quality-stage refinement on the same 100x100x150 grid as production.
    # Explicitly opt-in; source court and memory profile must be qualified.
    flexicubes_high_resolution_training: bool = False
    flexicubes_reference_surface_xyz: Tensor | None = None  # [1,M,3]
    flexicubes_animation_teacher: RSSvNextFCAnimationTeacher | None = None
    flexicubes_signed_source_admission_key: str = ""
    # Production TRAINING seed is independent native-source 3D teacher only.
    # The EXACT student H(x) is still computed from the observed two RGBA
    # frames + pinned GENERATED six-view conditioning, never the teacher.
    seeded_tessa_source_teacher: SeededTESSASourceTeacherV1 | None = None
    seeded_tessa_source_face_budget: int = 0
    # Source-native component-specific unsigned fields, never the global
    # nearest-surface distance or a semantic "cape" pseudo-label.
    component_udf_source_teacher: SourceNativeComponentUDFFitV1 | None = None
    # Source-qualified signed shells, never infer garment presence merely
    # from four-way presentation role. No imaginary/open cloth shell.
    flexicubes_layers: tuple[RSSvNextSemanticLayerSpec,...] | None = None
    flexicubes_signed_layer_admission_keys: tuple[str,...] = ()
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
    source_signed_sdf: float = 0.0    # opt-in: exact boundary + sign for qualified solid
    axis_rig_skin_codes: float = 0.0  # opt-in: source rig and MIRA-codec-bound FSQ teacher
    axis_posed_surface: float = 0.0  # opt-in: source motion consequence through TESSA+AXIS+MIRA
    axis_dynamic_posed_surface: float = 0.0  # opt-in: same qualified source pose supervises live dynamic refinement
    anchor_face_correspondence: float = 0.0  # source-native directed face labels
    flexicubes_surface: float = 0.3
    flexicubes_animation: float = 0.0  # explicit opt-in on admitted source motion
    # Source-bound differentiable surface alignment; not a pseudo-label.
    tessa_iris_surface_alignment: float = 0.0
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
            self.source_udf, self.source_inside_sign, self.source_signed_sdf,
            self.axis_rig_skin_codes, self.axis_posed_surface, self.axis_dynamic_posed_surface, self.anchor_face_correspondence, self.flexicubes_surface, self.flexicubes_animation,
            self.tessa_iris_surface_alignment,
            self.novel_view, self.realsas_joint_slrm,
        ) < 0:
            raise ValueError("RSS_VNEXT_TRAIN_NEGATIVE_OBJECTIVE_WEIGHT")
        if (
            self.topology + self.motion_preference
            + self.source_semantics + self.semantic_risk + self.surface_roles + self.region_pairs
            + self.typed_relations + self.graphics_source_rgb + self.source_udf + self.source_inside_sign
            + self.source_signed_sdf
            + self.axis_rig_skin_codes + self.axis_posed_surface + self.axis_dynamic_posed_surface + self.anchor_face_correspondence + self.flexicubes_surface
            + self.flexicubes_animation + self.novel_view
            + self.tessa_iris_surface_alignment
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
    source_signed_sdf_loss: Tensor | None = None
    eligible_source_udf_queries: int = 0
    eligible_source_inside_sign_queries: int = 0
    eligible_source_signed_sdf_queries: int = 0
    axis_rig_skin_loss: Tensor | None = None
    axis_posed_surface_loss: Tensor | None = None
    axis_dynamic_posed_surface_loss: Tensor | None = None
    eligible_axis_dynamic_pose_count: int = 0
    eligible_axis_pose_count: int = 0
    eligible_axis_sequences: int = 0
    axis_training_conditioning: str = "NO_AXIS_OBJECTIVE"
    anchor_face_correspondence_loss: Tensor | None = None
    eligible_anchor_held_faces: int = 0
    eligible_anchor_holder_faces: int = 0
    eligible_anchor_negative_faces: int = 0
    tessa_iris_surface_alignment_loss: Tensor | None = None
    realsas_joint_slrm_loss: Tensor | None = None
    eligible_realsas_joint_rays: int = 0
    gradient_norm: Tensor | None = None
    gradient_interference_report: dict | None = None
    seeded_tessa_loss: Tensor | None = None
    seeded_tessa_loss_parts: dict[str, Tensor] | None = None
    seeded_tessa_source_point_count: int = 0
    seeded_tessa_sampled_source_point_count: int = 0
    seeded_tessa_sampled_student_vertex_count: int = 0
    seeded_tessa_sampled_quality_face_count: int = 0
    seeded_tessa_qualified_layer_count: int = 0
    component_udf_loss: Tensor | None = None
    component_udf_loss_parts: dict[str, Tensor] | None = None
    eligible_component_udf_source_components: int = 0
    product_graphics_joint_loss_parts: dict[str, Tensor] | None = None
    product_semantic_loss: Tensor | None = None
    product_semantic_loss_parts: dict[str, Tensor] | None = None
    product_axis_mira_loss: Tensor | None = None
    product_axis_mira_parts: dict[str, Tensor] | None = None


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
    # NumPy does not expose torch.bfloat16, but source receipts need
    # bitwise-stable BF16 hashes on A100 without a float32 cast (which
    # changes teacher authority). A flattened byte view preserves the
    # original dtype/shape in header and exact storage bits in payload.
    raw = cpu.reshape(-1).view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(header + bytes([0]) + raw).hexdigest()


from .training_optimizer_ownership_v1 import qualify_rss_vnext_optimizer_ownership_v1
from .training_phase_contract_v1 import RSSvNextStructuralTrainingPhaseContractV1
from .training_topology_source_admission_v1 import validate_source_topology_teacher_v1
from .training_axis_skin_warmup_v1 import (
    RSSvNextAxisSkinCodeWarmupV1, apply_rss_vnext_axis_skin_code_warmup_v1,
)
from .training_gradient_interference_v1 import (
    RSSvNextGradientCourtPolicyV1,
    audit_rss_vnext_gradient_interference_v1,
)
from .training_axis_posed_surface_v1 import qualified_unified_posed_surface_objective_v1


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
        phase_contract: RSSvNextStructuralTrainingPhaseContractV1 | None = None,
        axis_skin_code_warmup: RSSvNextAxisSkinCodeWarmupV1 = RSSvNextAxisSkinCodeWarmupV1(),
        scene_conditioning_policy: str = "RESEARCH_CALIBRATED_TWO_VIEW",
        axis_training_policy: str = "RIG_ONLY_PRETRAIN",
        trusted_generated_conditioning_registry: Mapping[str, Mapping[str, object]] | None = None,
        trusted_render_registry: Mapping[str, Mapping[str, object]] | None = None,
        trusted_surface_role_registry: Mapping[str, Mapping[str, object]] | None = None,
        trusted_topology_teacher_registry: Mapping[str, Mapping[str, object]] | None = None,
        trusted_semantic_patch_registry: Mapping[str, Mapping[str, object]] | None = None,
        trusted_region_pair_registry: Mapping[str, Mapping[str, object]] | None = None,
        trusted_typed_relation_registry: Mapping[str, Mapping[str, object]] | None = None,
        trusted_axis_teacher_registry: Mapping[str, Mapping[str, object]] | None = None,
        trusted_chain_teacher_registry: Mapping[str, Mapping[str, object]] | None = None,
        trusted_axis_pose_registry: Mapping[str, Mapping[str, object]] | None = None,
        trusted_anchor_face_registry: Mapping[str, Mapping[str, object]] | None = None,
        trusted_signed_source_registry: Mapping[str, Mapping[str, object]] | None = None,
        trusted_graphics_rgb_registry: Mapping[str, Mapping[str, object]] | None = None,
        trusted_source_field_registry: Mapping[str, Mapping[str, object]] | None = None,
        trusted_realsas_joint_scene_registry: Mapping[str, Mapping[str, object]] | None = None,
        # Explicit product-v1 admission: legacy token CE and product seeded
        # supervision can never be activated within the same optimizer phase.
        training_profile: str = "LEGACY_STRUCTURAL_RESEARCH",
        seeded_tessa_weights: SeededTESSALossWeightsV1 | None = None,
        seeded_tessa_sampling_policy: SeededSurfaceSamplingPolicyV1 = SeededSurfaceSamplingPolicyV1(),
        trusted_seeded_tessa_registry: Mapping[str, Mapping[str, object]] | None = None,
        # Open-source-native-only stage may train the same H without assuming
        # the asset is a closed signed solid. For signed solids, joint UDF
        # evidence can also accompany the existing seeded TESSA stage.
        component_udf_weight: float = 0.0,
        trusted_component_udf_registry: Mapping[str, Mapping[str, object]] | None = None,
        # Optional source-native appearance and semantic losses, same H(x).
        product_graphics_rgb_weight: float = 0.0,
        product_joint_ray_weight: float = 0.0,
        product_semantic_weights: ProductSemanticWeightsV1 = ProductSemanticWeightsV1(),
        product_axis_mira_weights: ProductSeededAXISMIRAWeightsV1 = ProductSeededAXISMIRAWeightsV1(),
    ):
        if not dataset_manifest_sha256 or len(dataset_manifest_sha256) != 64:
            raise ValueError("RSS_VNEXT_TRAIN_DATASET_MANIFEST_SHA_REQUIRED")
        try:
            int(dataset_manifest_sha256, 16)
        except ValueError as exc:
            raise ValueError("RSS_VNEXT_TRAIN_DATASET_MANIFEST_NOT_HEX") from exc
        if training_profile not in (
            "LEGACY_STRUCTURAL_RESEARCH",
            "PRODUCT_SEEDED_IRIS_TESSA_V1",
            "PRODUCT_COMPONENT_UDF_V1",
        ):
            raise ValueError("RSS_VNEXT_TRAIN_UNKNOWN_GEOMETRY_PROFILE")
        seeded_mode = training_profile == "PRODUCT_SEEDED_IRIS_TESSA_V1"
        component_mode = training_profile == "PRODUCT_COMPONENT_UDF_V1"
        if type(component_udf_weight) not in (float,int):
            raise ValueError("RSS_VNEXT_PRODUCT_COMPONENT_UDF_PHASE_POLICY")
        component_enabled = component_mode or (
            seeded_mode and component_udf_weight > 0.
        )
        if (
            type(component_udf_weight) not in (float,int)
            or not math.isfinite(component_udf_weight)
            or component_udf_weight < 0
            or (
                component_mode and component_udf_weight <= 0
            )
            or (component_enabled and not trusted_component_udf_registry)
            or (
                not component_enabled
                and (component_udf_weight != 0 or trusted_component_udf_registry)
            )
        ):
            raise ValueError("RSS_VNEXT_PRODUCT_COMPONENT_UDF_PHASE_POLICY")
        # Unlike global research losses, these opt-in terms must actually
        # reach the seeded product objective and the SAME optimizer.
        if not isinstance(product_semantic_weights,ProductSemanticWeightsV1):
            raise ValueError("RSS_VNEXT_PRODUCT_SEMANTIC_PHASE_POLICY")
        product_semantic_weights.validate()
        semantic_active=any(v>0 for v in asdict(product_semantic_weights).values())
        if (
            semantic_active and (
                not seeded_mode
                or (product_semantic_weights.patch_axes>0
                    and not trusted_semantic_patch_registry)
                or (product_semantic_weights.patch_risk>0
                    and not trusted_semantic_patch_registry)
                or (product_semantic_weights.surface_roles>0
                    and not trusted_surface_role_registry)
                or (product_semantic_weights.region_pairs>0
                    and not trusted_region_pair_registry)
                or (product_semantic_weights.typed_relations>0
                    and not trusted_typed_relation_registry)
            )
        ):
            raise ValueError("RSS_VNEXT_PRODUCT_SEMANTIC_PHASE_POLICY")
        if (
            any(
                type(w) not in (float,int) or not math.isfinite(w) or w<0
                for w in (product_graphics_rgb_weight,product_joint_ray_weight)
            )
            or (not seeded_mode and
                (product_graphics_rgb_weight!=0 or product_joint_ray_weight!=0))
            or (product_graphics_rgb_weight>0
                and not trusted_graphics_rgb_registry)
            or (product_joint_ray_weight>0
                and not trusted_realsas_joint_scene_registry)
        ):
            raise ValueError("RSS_VNEXT_PRODUCT_GRAPHICS_PHASE_POLICY_INVALID")
        if not isinstance(product_axis_mira_weights,ProductSeededAXISMIRAWeightsV1):
            raise ValueError("RSS_VNEXT_PRODUCT_SEEDED_AXIS_MIRA_PHASE_POLICY")
        product_axis_mira_weights.validate()
        if product_axis_mira_weights.chain_skin>0 and not trusted_chain_teacher_registry:
            raise ValueError("RSS_VNEXT_CHAIN_SKIN_REQUIRES_SOURCE_NATIVE_REGISTRY")
        if product_axis_mira_weights.enabled and (
            not seeded_mode or not trusted_axis_teacher_registry
            or axis_training_policy!="PRODUCT_MESH_MATCHED_REQUIRED"
        ):
            raise ValueError("RSS_VNEXT_PRODUCT_SEEDED_AXIS_MIRA_PHASE_POLICY")
        seeded_tessa_sampling_policy.validate()
        if not seeded_mode and seeded_tessa_sampling_policy != SeededSurfaceSamplingPolicyV1():
            raise ValueError("RSS_VNEXT_LEGACY_SEEDED_SAMPLING_POLICY_FORBIDDEN")
        if seeded_mode:
            if (
                seeded_tessa_weights is None
                or not trusted_seeded_tessa_registry
                or scene_conditioning_policy != "PRODUCT_GENERATED_SIX_VIEW"
                or model.cfg.tessa_surface_requirement
                    != "REQUIRE_SIGNED_IRIS_SURFACE"
                or phase_contract is not None
                or gradient_court_policy.every_n_steps != 0
                or weights != RSSvNextTrainingWeights()
            ):
                raise ValueError("RSS_VNEXT_PRODUCT_SEEDED_PHASE_POLICY_INVALID")
            seeded_tessa_weights.validate()
            if (
                not any(
                    p.requires_grad for p in model.tessa_seeded_head.parameters()
                )
                or not any(p.requires_grad for p in model.iris.parameters())
            ):
                raise ValueError("RSS_VNEXT_PRODUCT_SEEDED_GRADIENT_OWNERS_FROZEN")
        elif seeded_tessa_weights is not None or trusted_seeded_tessa_registry:
            raise ValueError("RSS_VNEXT_LEGACY_MAY_NOT_ACCEPT_SEEDED_TEACHER")
        if component_mode and (
            scene_conditioning_policy != "PRODUCT_GENERATED_SIX_VIEW"
            or phase_contract is not None
            or gradient_court_policy.every_n_steps != 0
            or weights != RSSvNextTrainingWeights()
        ):
            raise ValueError("RSS_VNEXT_COMPONENT_UDF_SOURCE_ONLY_PHASE_REQUIRED")
        weights.validate()
        if seeded_mode or component_mode:
            # Legacy topology/CE/motion coefficients must be zero in the
            # *effective policy and checkpoint*, not just skipped at runtime.
            # The seeded loss family has its OWN strictly positive weights.
            weights = RSSvNextTrainingWeights(**{
                k: (seeded_tessa_weights.clip_gradient_norm
                    if seeded_mode and k == "clip_gradient_norm"
                    else (1.0 if k == "clip_gradient_norm" else 0.0))
                for k in asdict(weights)
            })
        preference.validate()
        gradient_court_policy.validate()
        axis_skin_code_warmup.validate()
        if axis_skin_code_warmup.enabled and weights.axis_rig_skin_codes <= 0:
            raise ValueError("RSS_VNEXT_AXIS_SKIN_WARMUP_AXIS_LOSS_REQUIRED")
        if weights.axis_dynamic_posed_surface > 0 and (
            weights.axis_posed_surface <= 0
            or weights.axis_rig_skin_codes <= 0
            or axis_training_policy != "PRODUCT_MESH_MATCHED_REQUIRED"
        ):
            raise ValueError("RSS_VNEXT_DYNAMIC_POSE_REQUIRES_QUALIFIED_AXIS_AND_STATIC_POSE")
        if phase_contract is not None:
            phase_contract.validate(weights=weights)
            phase_contract.assert_trainable_owners(model)
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
        self.training_profile = training_profile
        self.seeded_tessa_weights = seeded_tessa_weights
        self.component_udf_weight = float(component_udf_weight)
        self.component_udf_enabled = component_enabled
        self.product_graphics_rgb_weight = float(product_graphics_rgb_weight)
        self.product_joint_ray_weight = float(product_joint_ray_weight)
        self.product_semantic_weights = product_semantic_weights
        self.product_semantic_active = semantic_active
        self.product_axis_mira_weights = product_axis_mira_weights
        component_snapshot = {
            str(k): dict(v)
            for k,v in (trusted_component_udf_registry or {}).items()
        }
        self.trusted_component_udf_registry = MappingProxyType({
            k:MappingProxyType(v) for k,v in component_snapshot.items()
        })
        self.component_udf_registry_sha256 = (
            hashlib.sha256(json.dumps(
                component_snapshot,sort_keys=True,
                separators=(",",":"),allow_nan=False,
            ).encode()).hexdigest() if component_enabled else ""
        )
        self.scene_conditioning_policy = scene_conditioning_policy
        self.axis_training_policy = axis_training_policy
        self.model = model
        self.optimizer = optimizer
        self.optimizer_ownership = qualify_rss_vnext_optimizer_ownership_v1(
            self.model, self.optimizer,
        )
        self.seeded_tessa_driver = (
            RSSvNextSeededTESSATrainerV1(
                model, optimizer, weights=seeded_tessa_weights,
                trusted_source_registry=trusted_seeded_tessa_registry,
                sampling_policy=seeded_tessa_sampling_policy,
            )
            if seeded_mode else None
        )
        self.seeded_tessa_registry_sha256 = (
            hashlib.sha256(json.dumps(
                {str(k):dict(v)
                 for k,v in trusted_seeded_tessa_registry.items()},
                sort_keys=True,separators=(",",":"),allow_nan=False,
            ).encode()).hexdigest()
            if seeded_mode else ""
        )
        self.weights = weights
        self.preference = preference
        self.gradient_court_policy = gradient_court_policy
        self.phase_contract = phase_contract
        self.axis_skin_code_warmup = axis_skin_code_warmup
        # Training/inference mesh ABI is versioned and CHECKPOINT-BOUND.
        # Legacy token supervision can never resume as seeded geometry.
        # Freeze the full objective / topology-preference policy into the
        # resumption identity. A semantic-risk or motion weight change is a
        # different training run, not a transparent continuation of this one.
        self.objective_policy_sha256 = hashlib.sha256(json.dumps({
            # A disabled new source family retains older v2 checkpoint
            # identities; enabling it creates a distinct objective policy.
            "training_profile": self.training_profile,
            **({
                "seeded_tessa_policy_sha256":
                    self.seeded_tessa_driver.policy_sha256,
                "seeded_tessa_registry_sha256":
                    self.seeded_tessa_registry_sha256,
                "seeded_tessa_source_schema": SEEDED_TESSA_SCHEMA,
                "seeded_tessa_phase": SEEDED_TESSA_PHASE,
            } if seeded_mode else {}),
            **({
                "component_udf_weight":self.component_udf_weight,
                "component_udf_registry_sha256":
                    self.component_udf_registry_sha256,
                "component_udf_teacher_schema":
                    "RealSaS.RSSvNext.SourceNativeComponentUDFAssignment.v1",
            } if component_enabled else {}),
            **({
                "product_semantic_weights":asdict(self.product_semantic_weights),
                "product_semantic_requires_source_native":True,
            } if self.product_semantic_active else {}),
            **({
                "product_axis_mira_weights":asdict(self.product_axis_mira_weights),
                "requires_exact_native_mstar_face_pose_and_pinned_codec":True,
            } if self.product_axis_mira_weights.enabled else {}),
            **({
                "product_graphics_rgb_weight":self.product_graphics_rgb_weight,
                "product_joint_ray_weight":self.product_joint_ray_weight,
                "source_graphics_qualified_required":True,
                "joint_ray_qualified_source_not_generated_required":True,
            } if self.product_graphics_rgb_weight>0
                or self.product_joint_ray_weight>0 else {}),
            "weights": {k: v for k, v in asdict(weights).items()
                        if not (
                            k in ("axis_posed_surface", "axis_dynamic_posed_surface", "tessa_iris_surface_alignment")
                            and v == 0.
                        )},
            "preference": asdict(preference),
            "scene_conditioning_policy": self.scene_conditioning_policy,
            "axis_training_policy": self.axis_training_policy,
            "gradient_court_policy": asdict(gradient_court_policy),
            **({"phase_contract": asdict(phase_contract)} if phase_contract is not None else {}),
            **({"axis_skin_code_warmup": asdict(axis_skin_code_warmup)} if axis_skin_code_warmup.enabled else {}),
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
        chain_snapshot = {
            str(key): dict(value)
            for key, value in (trusted_chain_teacher_registry or {}).items()
        }
        self.trusted_chain_teacher_registry = MappingProxyType({
            key: MappingProxyType(value) for key, value in chain_snapshot.items()
        })
        self.chain_teacher_registry_sha256 = (
            hashlib.sha256(json.dumps(
                chain_snapshot, sort_keys=True, separators=(",", ":"), allow_nan=False,
            ).encode("utf8")).hexdigest() if chain_snapshot else ""
        )
        pose_snapshot = {
            str(key): dict(value)
            for key, value in (trusted_axis_pose_registry or {}).items()
        }
        self.trusted_axis_pose_registry = MappingProxyType({
            key: MappingProxyType(value) for key, value in pose_snapshot.items()
        })
        self.axis_pose_registry_sha256 = (
            hashlib.sha256(json.dumps(
                pose_snapshot, sort_keys=True, separators=(",", ":"),
                allow_nan=False,
            ).encode("utf8")).hexdigest() if pose_snapshot else ""
        )
        # Trust only the launcher-owned source 3D indexed-mesh topology ledger.
        topology_snapshot = {
            str(key): dict(value)
            for key, value in (trusted_topology_teacher_registry or {}).items()
        }
        self.trusted_topology_teacher_registry = MappingProxyType({
            key: MappingProxyType(value) for key, value in topology_snapshot.items()
        })
        self.topology_teacher_registry_sha256 = (
            hashlib.sha256(json.dumps(
                topology_snapshot, sort_keys=True, separators=(",", ":"),
                allow_nan=False,
            ).encode("utf8")).hexdigest() if topology_snapshot else ""
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
        self.phase_handoff_source_model_sha256 = ""

    def objective(self, batch: RSSvNextStructuralTrainingBatch) -> RSSvNextTrainingResult:
        if batch.admission_status != "QUALIFIED_FOR_THIS_SUPERVISION_FAMILY":
            raise ValueError("RSS_VNEXT_TRAIN_ADMISSION_NOT_QUALIFIED")
        if not batch.source_qualification_key:
            raise ValueError("RSS_VNEXT_TRAIN_SOURCE_QUALIFICATION_KEY_MISSING")
        if not self.model.training:
            raise RuntimeError("RSS_VNEXT_TRAIN_REQUIRES_TRAIN_MODE")
        # Production phases must bind source-native topology token bytes to
        # an independently qualified asset, even for TESSA-only training.
        if self.phase_contract is not None and (
            batch.topology_teacher_tokens is not None
            or batch.topology_valid_mask is not None
            or bool(batch.source_topology_admission_key)
        ):
            validate_source_topology_teacher_v1(
                token_ids=batch.topology_teacher_tokens,
                valid_mask=batch.topology_valid_mask,
                admission_key=batch.source_topology_admission_key,
                trusted_registry=self.trusted_topology_teacher_registry,
                dataset_manifest_sha256=self.dataset_manifest_sha256,
                source_qualification_key=batch.source_qualification_key,
                original_rgba=batch.rgba,
                input_camera_features=batch.camera_features,
                tensor_digest=source_surface_tensor_sha256,
            )

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
        if self.training_profile == "PRODUCT_SEEDED_IRIS_TESSA_V1":
            return self._objective_seeded_product_v1(batch, state)
        if self.training_profile == "PRODUCT_COMPONENT_UDF_V1":
            self._reject_extra_component_udf_source_fields_v1(batch)
            out=self._component_udf_objective_v1(batch,state)
            return RSSvNextTrainingResult(
                total_loss=self.component_udf_weight*out["total"],
                topology_loss=None,motion_preference_loss=None,
                source_semantic_loss=None,semantic_risk_loss=None,
                flexicubes_surface_loss=None,flexicubes_animation_loss=None,
                eligible_animation_pose_count=0,
                eligible_semantic_patch_labels=0,
                eligible_topology_examples=0,qualified_preference_pairs=0,
                component_udf_loss=out["total"],
                component_udf_loss_parts={
                    k:v for k,v in out.items() if k in
                    ("field","kind","unmatched_unknown")
                },
                eligible_component_udf_source_components=int(
                    out["qualified_source_component_count"].item()
                ),
            )
        if batch.seeded_tessa_source_teacher is not None or (
            batch.seeded_tessa_source_face_budget != 0
        ):
            raise ValueError("RSS_VNEXT_LEGACY_SEEDED_SOURCE_FORBIDDEN")
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
        # The production structural decoder must be conditioned on the
        # predicted IRIS surface at BOTH training and free-running inference.
        # Source mesh tokens remain teacher-only; this is not a substitute
        # for independently qualified closed-solid source provenance.
        tessa_surface_needed = (
            self.model.cfg.tessa_surface_requirement == "REQUIRE_SIGNED_IRIS_SURFACE"
            and (
                (self.weights.topology > 0 and batch.topology_teacher_tokens is not None)
                or (self.weights.motion_preference > 0 and batch.motion_candidate_tokens is not None)
                or (
                    self.axis_training_policy == "PRODUCT_MESH_MATCHED_REQUIRED"
                    and batch.axis_teacher_sequence is not None
                )
            )
        )
        if tessa_surface_needed and (
            batch.flexicubes_extractor is None
            or batch.flexicubes_layer is None
        ):
            raise ValueError("RSS_VNEXT_TRAIN_TESSA_IRIS_SURFACE_REQUIRED")
        if type(batch.flexicubes_high_resolution_training) is not bool:
            raise ValueError("RSS_VNEXT_TRAIN_HIGHRES_PROFILE_TYPE")
        if batch.flexicubes_high_resolution_training and (
            batch.flexicubes_extractor is None
            or batch.flexicubes_layer is None
            or batch.flexicubes_reference_surface_xyz is None
            or self.weights.flexicubes_surface <= 0
            or not batch.flexicubes_signed_source_admission_key
        ):
            raise ValueError("RSS_VNEXT_TRAIN_HIGHRES_SOURCE_AND_SURFACE_REQUIRED")
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
                    and not tessa_surface_needed
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
                or signed.get("source_component_domain") not in (
                    "SINGLE_QUALIFIED_CLOSED_COMPONENT",
                    "EXPLICIT_QUALIFIED_CLOSED_ASSEMBLY",
                )
                or (
                    signed.get("source_component_domain")
                        == "EXPLICIT_QUALIFIED_CLOSED_ASSEMBLY"
                    and (
                        signed.get("assembly_boolean_operations_qualified") is not True
                        or signed.get("sdf_distance_scope")
                            != "EXACT_BOUNDARY_OF_THE_SAME_QUALIFIED_SOLID"
                        or signed.get("sdf_sign_scope")
                            != "GLOBAL_NEGATIVE_INSIDE_POSITIVE_OUTSIDE"
                    )
                )
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
                tessa_surface_needed
                or (self.weights.flexicubes_surface > 0 and target is not None)
                or (
                    self.weights.flexicubes_animation > 0
                    and batch.flexicubes_animation_teacher is not None
                )
            ):
                fc_surface = self.model.fc_surface_bridge(
                    state.scene_triplanes,
                    extractor=batch.flexicubes_extractor,
                    role_head=self.model.presentation_role_head,
                    schema=self.model.cfg.semantic_role_schema(),
                    layer=batch.flexicubes_layer,
                    field_policy="SIGNED_HYPOTHESIS_ONLY",
                    high_resolution_training=batch.flexicubes_high_resolution_training,
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
        if tessa_surface_needed and fc_surface is None:
            raise RuntimeError("RSS_VNEXT_TRAIN_TESSA_IRIS_SURFACE_NOT_EXTRACTED")

        tessa_iris_alignment = None
        if self.weights.tessa_iris_surface_alignment > 0:
            tokens = batch.topology_teacher_tokens
            mask = batch.topology_valid_mask
            receipt = self.trusted_signed_source_registry.get(
                batch.flexicubes_signed_source_admission_key,
            )
            # Only a source-matched *teacher topology* and independently
            # admitted reference surface can supervise a TESSA/IRIS tether.
            # Predicted FC geometry by itself must never act as source truth.
            if (
                fc_surface is None
                or fc_loss is None
                or self.weights.flexicubes_surface <= 0
                or tokens is None or mask is None
                or tokens.ndim != 2 or tokens.shape[0] != 1
                or tokens.dtype != torch.long
                or mask.shape != (1,) or mask.dtype != torch.bool
                or not bool(mask[0])
                or receipt is None
                or receipt.get("tessa_source_topology_sha256")
                    != source_surface_tensor_sha256(tokens)
            ):
                raise ValueError("RSS_VNEXT_TRAIN_TESSA_IRIS_ALIGNMENT_SOURCE_REQUIRED")
            # Continuous TESSA rest-surface correction receives gradients.
            # Discrete face decisions remain trained by topology CE and the
            # qualified posed preference court, never by a fake Chamfer grad.
            qualified_mesh = self.model.build_mesh_candidate(
                state, tokens[0],
                source_authority="TEACHER_FORCED_TRAINING_ONLY",
            )
            tessa_xyz = qualified_mesh.rest_vertices_iris
            iris_xyz = fc_surface.xyz
            ti = torch.linspace(
                0, tessa_xyz.shape[1] - 1, min(512, tessa_xyz.shape[1]),
                device=tessa_xyz.device,
            ).long()
            ii = torch.linspace(
                0, iris_xyz.shape[1] - 1, min(512, iris_xyz.shape[1]),
                device=iris_xyz.device,
            ).long()
            tessa_iris_alignment = symmetric_surface_chamfer_v1(
                tessa_xyz[:, ti], iris_xyz[:, ii],
            )
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
        source_signed_sdf_loss = None
        n_source_udf = 0
        n_source_inside_sign = 0
        n_source_signed_sdf = 0
        field_items = (
            (batch.source_udf_target, batch.source_udf_mask),
            (batch.source_inside_sign_target, batch.source_inside_sign_mask),
            (batch.source_signed_sdf_target, batch.source_signed_sdf_mask),
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
                or receipt.get("teacher_semantics") not in (
                    "SOURCE_UDF_AND_OPTIONAL_NEGATIVE_INSIDE_SIGN",
                    "SOURCE_EXACT_SIGNED_SOLID_BOUNDARY_DISTANCE_V1",
                )
                or receipt.get("signed_domain") not in (
                    "SINGLE_QUALIFIED_CLOSED_COMPONENT",
                    "EXPLICIT_QUALIFIED_CLOSED_ASSEMBLY",
                    "OPEN_OR_UNKNOWN_UNSIGNED_ONLY",
                )
                or receipt.get("sdf_target_sha256") != (
                    source_surface_tensor_sha256(batch.source_signed_sdf_target)
                    if batch.source_signed_sdf_target is not None else None
                )
                or receipt.get("sdf_mask_sha256") != (
                    source_surface_tensor_sha256(batch.source_signed_sdf_mask)
                    if batch.source_signed_sdf_mask is not None else None
                )
            ):
                raise ValueError("RSS_VNEXT_SOURCE_FIELD_RECEIPT_DRIFT")
            if (
                batch.source_inside_sign_target is not None
                and receipt.get("signed_domain") not in (
                    "SINGLE_QUALIFIED_CLOSED_COMPONENT",
                    "EXPLICIT_QUALIFIED_CLOSED_ASSEMBLY",
                )
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
            if batch.source_signed_sdf_target is not None:
                # All source targets refer to exactly the SAME normalized
                # coordinates and same boundary. Reject contradictory labels
                # even when their individual tensor hashes are valid.
                sdf_target, sdf_mask = field_items[2]
                if batch.source_udf_target is not None:
                    udf_target, udf_mask = field_items[0]
                    both = sdf_mask & udf_mask
                    if bool(both.any()) and not bool(torch.allclose(
                        sdf_target[both].abs(), udf_target[both],
                        rtol=1.e-3, atol=1.e-3,
                    )):
                        raise ValueError("RSS_VNEXT_SOURCE_SDF_UDF_TEACHER_CONFLICT")
                if batch.source_inside_sign_target is not None:
                    sign_target, sign_mask = field_items[1]
                    # A true zero level carries no unambiguous inside label.
                    both = sdf_mask & sign_mask & (sdf_target.abs() > 1.e-5)
                    if bool(both.any()) and not bool(torch.equal(
                        sign_target[both] == 1., sdf_target[both] < 0.,
                    )):
                        raise ValueError("RSS_VNEXT_SOURCE_SDF_SIGN_TEACHER_CONFLICT")
                # A direct SDF teacher is not recoverable from unrelated
                # component-local UDF + occupancy. Its distance must be to
                # the *same exact explicit solid/assembly boundary* used
                # for the global inside/outside sign.
                if (
                    self.weights.source_signed_sdf <= 0
                    or receipt.get("teacher_semantics")
                        != "SOURCE_EXACT_SIGNED_SOLID_BOUNDARY_DISTANCE_V1"
                    or receipt.get("signed_domain") not in (
                        "SINGLE_QUALIFIED_CLOSED_COMPONENT",
                        "EXPLICIT_QUALIFIED_CLOSED_ASSEMBLY",
                    )
                    or receipt.get("sdf_distance_scope")
                        != "EXACT_BOUNDARY_OF_THE_SAME_QUALIFIED_SOLID"
                    or receipt.get("sdf_sign_scope")
                        != "GLOBAL_NEGATIVE_INSIDE_POSITIVE_OUTSIDE"
                    or receipt.get("assembly_boolean_operations_qualified") is not True
                ):
                    raise ValueError("RSS_VNEXT_SOURCE_SDF_EXACT_SOLID_PROOF_REQUIRED")
                sdf_target, sdf_mask = field_items[2]
                # Full SDF is predicted by IRIS's *actual* FlexiCubes field;
                # do not train a shadow UDF/sign head only to leave this field
                # unconstrained at inference.
                source_signed_sdf_loss = F.smooth_l1_loss(
                    state.geometry["field"][sdf_mask], sdf_target[sdf_mask],
                )
                n_source_signed_sdf = int(sdf_mask.sum().item())
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
        axis_pose_loss = None
        axis_dynamic_pose_loss = None
        axis_pose_count = 0
        axis_dynamic_pose_count = 0
        pose_fields = (
            batch.axis_pose_joint_skin_matrices,
            batch.axis_pose_rest_surface_samples,
            batch.axis_pose_surface_samples,
            batch.axis_pose_valid_mask,
        )
        pose_requested = any(x is not None for x in pose_fields) or bool(batch.axis_pose_admission_key)
        if pose_requested:
            if (
                self.weights.axis_posed_surface <= 0
                or any(x is None for x in pose_fields)
                or not batch.axis_pose_admission_key
                or self.axis_training_policy != "PRODUCT_MESH_MATCHED_REQUIRED"
                or batch.axis_teacher_sequence is None
            ):
                raise ValueError("RSS_VNEXT_UNIFIED_POSE_SUPERVISION_INCOMPLETE")
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
            axis_loss = apply_rss_vnext_axis_skin_code_warmup_v1(
                raw_axis_loss=axis_loss,
                raw_axis_loss_parts=_axis_components,
                policy=self.axis_skin_code_warmup,
                step=self.global_step,
            )
            n_axis = int(teacher.kind_ids.shape[0])
            if pose_requested:
                if matched_axis_mesh is None or matched_axis_memory is None:
                    raise ValueError("RSS_VNEXT_UNIFIED_POSE_MATCHED_TESSA_REQUIRED")
                controls = self.model.axis(matched_axis_memory, teacher)
                predicted_skin, _, predicted_joint_features = self.model.mira(
                    matched_axis_mesh.local_features,
                    matched_axis_mesh.rest_vertices_iris,
                    controls["hidden"], controls["skin_code_logits"], teacher,
                    vertex_normals=matched_axis_mesh.normals,
                )
                expected_pose = {
                    "admission_status": "QUALIFIED_SOURCE_NATIVE_AXIS_TESSA_POSED_SURFACE",
                    "dataset_manifest_sha256": self.dataset_manifest_sha256,
                    "source_qualification_key": batch.source_qualification_key,
                    "source_binding_sha256": batch.product_source_binding_sha256,
                    "observed_rgba_sha256": source_surface_tensor_sha256(batch.rgba),
                    "observed_camera_sha256": source_surface_tensor_sha256(batch.camera_features),
                    "mesh_teacher_topology_digest": axis_mesh_teacher_address_digest_v1(matched_axis_mesh),
                    "axis_teacher_sequence_sha256": axis_teacher_sequence_sha256_v6(teacher),
                    "source_joint_skin_matrices_sha256": source_surface_tensor_sha256(pose_fields[0]),
                    "source_rest_surface_samples_sha256": source_surface_tensor_sha256(pose_fields[1]),
                    "source_posed_surface_samples_sha256": source_surface_tensor_sha256(pose_fields[2]),
                    "source_pose_valid_mask_sha256": source_surface_tensor_sha256(pose_fields[3]),
                    "pose_witness_source": "INDEPENDENT_SOURCE_NATIVE_LBS_MESH",
                    "generated_view_is_motion_teacher": False,
                    "reindexed_tessa_vertices_and_axis_joints": True,
                    "normalized_geometry_domain": "IRIS_NORMALIZED_SOURCE_XYZ",
                }
                axis_pose_loss, axis_pose_count, _posed_only = (
                    qualified_unified_posed_surface_objective_v1(
                        mesh=matched_axis_mesh, predicted_skin=predicted_skin,
                        joint_skin_matrices=pose_fields[0],
                        source_rest_surface_samples=pose_fields[1],
                        source_posed_surface_samples=pose_fields[2],
                        valid_pose_mask=pose_fields[3],
                        receipt=self.trusted_axis_pose_registry.get(batch.axis_pose_admission_key),
                        expected_receipt=expected_pose,
                    )
                )
                if self.weights.axis_dynamic_posed_surface > 0:
                    # This is the *same* source-qualified TESSA mesh, AXIS
                    # causal joint order and pinned MIRA hard-token skin.
                    # The dynamic head receives real gradients from source
                    # rest/posed surfaces, not a synthetic ready/not-ready bit.
                    dynamic = self.model.dynamic_refine(
                        matched_axis_mesh, predicted_skin,
                        predicted_joint_features,
                    )
                    axis_dynamic_pose_loss, axis_dynamic_pose_count, _ = (
                        qualified_unified_posed_surface_objective_v1(
                            mesh=matched_axis_mesh,
                            predicted_skin=dynamic.weights,
                            refined_vertices=dynamic.vertices,
                            joint_skin_matrices=pose_fields[0],
                            source_rest_surface_samples=pose_fields[1],
                            source_posed_surface_samples=pose_fields[2],
                            valid_pose_mask=pose_fields[3],
                            receipt=self.trusted_axis_pose_registry.get(
                                batch.axis_pose_admission_key,
                            ),
                            expected_receipt=expected_pose,
                        )
                    )
                    if axis_dynamic_pose_count != axis_pose_count:
                        raise RuntimeError("RSS_VNEXT_DYNAMIC_POSE_SOURCE_COVERAGE_DRIFT")
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
            and tessa_iris_alignment is None
            and novel_loss is None and surface_role_loss is None
            and region_pair_loss is None
            and typed_relation_loss is None
            and graphics_rgb_loss is None and axis_loss is None
            and axis_pose_loss is None and axis_dynamic_pose_loss is None
            and anchor_face_loss is None
            and source_udf_loss is None and source_inside_sign_loss is None
            and source_signed_sdf_loss is None
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
                (self.weights.source_signed_sdf, source_signed_sdf_loss),
                (self.weights.axis_rig_skin_codes, axis_loss),
                (self.weights.axis_posed_surface, axis_pose_loss),
                (self.weights.axis_dynamic_posed_surface, axis_dynamic_pose_loss),
                (self.weights.anchor_face_correspondence, anchor_face_loss),
                (self.weights.flexicubes_surface, fc_loss),
                (self.weights.flexicubes_animation, fc_anim_loss),
                (self.weights.tessa_iris_surface_alignment, tessa_iris_alignment),
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
            tessa_iris_surface_alignment_loss=tessa_iris_alignment,
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
            source_signed_sdf_loss=source_signed_sdf_loss,
            eligible_source_udf_queries=n_source_udf,
            eligible_source_inside_sign_queries=n_source_inside_sign,
            eligible_source_signed_sdf_queries=n_source_signed_sdf,
            axis_rig_skin_loss=axis_loss,
            axis_posed_surface_loss=axis_pose_loss,
            axis_dynamic_posed_surface_loss=axis_dynamic_pose_loss,
            eligible_axis_pose_count=axis_pose_count,
            eligible_axis_dynamic_pose_count=axis_dynamic_pose_count,
            eligible_axis_sequences=n_axis,
            axis_training_conditioning=axis_conditioning,
            anchor_face_correspondence_loss=anchor_face_loss,
            eligible_anchor_held_faces=eligible_anchor_held,
            eligible_anchor_holder_faces=eligible_anchor_holder,
            eligible_anchor_negative_faces=eligible_anchor_negative,
            realsas_joint_slrm_loss=joint_loss,
            eligible_realsas_joint_rays=joint_rays,
        )


    def _reject_extra_component_udf_source_fields_v1(
        self,batch:RSSvNextStructuralTrainingBatch,
    ) -> None:
        allowed={
            "rgba","camera_features",
            "product_generated_rgba","product_generated_normals",
            "product_generated_cameras",
            "product_source_binding_sha256",
            "product_canonical_checkpoint_sha256",
            "product_multiview_checkpoint_sha256",
            "product_generation_checkpoint_sha256",
            "product_conditioning_admission_key",
            "component_udf_source_teacher",
            "source_qualification_key","admission_status",
        }
        unexpected=[]
        for key,value in vars(batch).items():
            if key in allowed or value is None:
                continue
            if ((type(value) is str and not value)
                or (type(value) is bool and value is False)
                or (type(value) is int and value == 0)
                or (type(value) is tuple and not value)):
                continue
            unexpected.append(key)
        if unexpected:
            raise ValueError(
                "RSS_VNEXT_PRODUCT_COMPONENT_UDF_UNCONSUMED_TEACHER:"
                +",".join(sorted(unexpected))
            )

    def _component_udf_objective_v1(
        self,batch:RSSvNextStructuralTrainingBatch,state,
    ) -> dict:
        """One generated-six scene H(x), native component UDF loss only.

        Never use the source component class, source mesh or source rig
        as a student neural input. Hungarian matches the *predicted* slots
        to the independently qualified native component set at loss time.
        """
        teacher=batch.component_udf_source_teacher
        if not self.component_udf_enabled or not isinstance(
            teacher,SourceNativeComponentUDFFitV1
        ):
            raise ValueError("RSS_VNEXT_PRODUCT_COMPONENT_UDF_TEACHER_REQUIRED")
        teacher.validate()
        receipt=self.trusted_component_udf_registry.get(
            teacher.source_qualification_key,
        )
        expected={
            "dataset_manifest_sha256":self.dataset_manifest_sha256,
            "source_qualification_key":batch.source_qualification_key,
            "observed_rgba_sha256":
                source_surface_tensor_sha256(batch.rgba),
            "observed_camera_sha256":
                source_surface_tensor_sha256(batch.camera_features),
            "source_binding_sha256":
                batch.product_source_binding_sha256,
            "combined_generation_checkpoint_sha256":
                batch.product_generation_checkpoint_sha256,
            "source_model_observation_count":2,
            "source_native_component_asset_sha256":
                teacher.source_asset_sha256,
        }
        if (
            state.scene_conditioning_authority
                != "GENERATED_SIX_VIEW_SHARED_SCENE"
            or state.camera_features is not None
            or not isinstance(receipt,Mapping)
            or any(receipt.get(k)!=v for k,v in expected.items())
            or teacher.original_query_xyz.device != state.scene_triplanes.device
            or teacher.original_query_xyz.dtype != state.scene_triplanes.dtype
            or len(teacher.original_query_xyz)>8192
            or state.scene_triplanes.shape[0]!=1
        ):
            raise ValueError("RSS_VNEXT_PRODUCT_COMPONENT_UDF_SCENE_RECEIPT_DRIFT")
        xyz=teacher.original_query_xyz[None]
        h=self.model.iris.query(state.scene_triplanes,xyz)["H"]
        distances=self.model.component_udf.query_all(
            h,state.region_slot_features,
        )
        topology_kind=self.model.component_udf.topology_kind_logits(
            state.region_slot_features,
        )
        return source_native_component_udf_objective_v1(
            predicted_distances=distances,
            predicted_kind_logits=topology_kind,
            teacher=teacher,
            trusted_source_registry=self.trusted_component_udf_registry,
            source_query_xyz=xyz[0],
        )

    def _objective_seeded_product_v1(
        self, batch: RSSvNextStructuralTrainingBatch, state,
    ) -> RSSvNextTrainingResult:
        """Production gradient route: ORIGINAL two sources -> six generated
        conditioning views -> ONE IRIS surface -> seeded TESSA M*.

        All source-native teacher tensors stay strictly on the right-hand
        side of loss computation, never inside the learned scene decoder.
        Signed-domain proof and native-label hashes are launcher-bound.
        """
        # Reject unconsumed teacher fields: no silently ignored source loss,
        # alternate topology, contact pseudo-label, or auxiliary eight-way
        # target may be smuggled into a nominally final seeded v1 batch.
        allowed = {
            "rgba", "camera_features",
            "product_generated_rgba", "product_generated_normals",
            "product_generated_cameras",
            "product_source_binding_sha256",
            "product_canonical_checkpoint_sha256",
            "product_multiview_checkpoint_sha256",
            "product_generation_checkpoint_sha256",
            "product_conditioning_admission_key",
            "flexicubes_extractor", "flexicubes_layer",
            "flexicubes_layers", "flexicubes_signed_layer_admission_keys",
            "flexicubes_high_resolution_training",
            "flexicubes_reference_surface_xyz",
            "flexicubes_signed_source_admission_key",
            "seeded_tessa_source_teacher", "seeded_tessa_source_face_budget",
            "source_qualification_key", "admission_status",
            *({"component_udf_source_teacher"}
               if self.component_udf_enabled else set()),
            *({
                "axis_teacher_sequence","axis_teacher_admission_key",
            } if self.product_axis_mira_weights.enabled else set()),
            *({
                "semantic_patch_target","semantic_patch_mask",
                "semantic_qualification_key",
            } if self.product_semantic_weights.patch_axes>0
                or self.product_semantic_weights.patch_risk>0 else set()),
            *({
                "source_surface_role_indices",
                "source_surface_role_mask",
                "source_surface_role_admission_key",
            } if self.product_semantic_weights.surface_roles>0 else set()),
            *({
                "source_region_pair_indices",
                "source_region_pair_is_same",
                "source_region_pair_mask",
                "source_region_pair_admission_key",
            } if self.product_semantic_weights.region_pairs>0 else set()),
            *({
                "source_typed_relation_pair_indices",
                "source_typed_relation_target",
                "source_typed_relation_mask",
                "source_typed_relation_admission_key",
            } if self.product_semantic_weights.typed_relations>0 else set()),
            *({
                "source_graphics_rgb_target",
                "source_graphics_view_directions",
                "source_graphics_rgb_mask",
                "source_graphics_rgb_admission_key",
            } if self.product_graphics_rgb_weight>0 else set()),
            *({
                "realsas_joint_ray_xyz","realsas_joint_ray_delta",
                "realsas_joint_ray_depth","realsas_joint_ray_direction",
                "realsas_joint_keep_roles","realsas_joint_target",
                "realsas_joint_teacher_admission_key",
            } if self.product_joint_ray_weight>0 else set()),
        }
        unexpected = []
        for name, value in vars(batch).items():
            if name in allowed or value is None:
                continue
            if (
                (type(value) is str and not value)
                or (type(value) is bool and value is False)
                or (type(value) is int and value == 0)
                or (type(value) is tuple and not value)
            ):
                continue
            unexpected.append(name)
        if unexpected:
            raise ValueError(
                "RSS_VNEXT_PRODUCT_SEEDED_UNCONSUMED_SOURCE_TEACHER:"
                + ",".join(sorted(unexpected))
            )
        teacher = batch.seeded_tessa_source_teacher
        driver = self.seeded_tessa_driver
        if driver is None or teacher is None:
            raise ValueError("RSS_VNEXT_PRODUCT_SEEDED_TEACHER_REQUIRED")
        if (
            state.camera_features is not None
            or state.scene_conditioning_authority
                != "GENERATED_SIX_VIEW_SHARED_SCENE"
            or batch.topology_teacher_tokens is not None
            or batch.topology_valid_mask is not None
            or batch.motion_candidate_tokens is not None
            or batch.motion_court is not None
            or (batch.axis_teacher_sequence is not None
                and not self.product_axis_mira_weights.enabled)
            or batch.source_topology_admission_key
            or batch.flexicubes_extractor is None
            or batch.flexicubes_layer is None
            or batch.flexicubes_animation_teacher is not None
            or type(batch.flexicubes_high_resolution_training) is not bool
            or batch.flexicubes_reference_surface_xyz is None
        ):
            raise ValueError("RSS_VNEXT_PRODUCT_SEEDED_NO_LEGACY_TEACHER_OR_SURFACE")
        # Source 3D geometry is LOSS-ONLY. Named source layers are native
        # annotations and signed-domain admissions, NOT model predictions.
        # The student receives only two original images + six generated
        # conditioning views in exactly one H(x).
        target = batch.flexicubes_reference_surface_xyz
        layers = (
            batch.flexicubes_layers
            if batch.flexicubes_layers is not None
            else (batch.flexicubes_layer,)
        )
        layered = batch.flexicubes_layers is not None
        schema = self.model.cfg.semantic_role_schema()
        if (
            not isinstance(layers,tuple)
            or not all(isinstance(v,RSSvNextSemanticLayerSpec) for v in layers)
            or not len(layers) >= 1
            or layers[0] != batch.flexicubes_layer
            or target.shape != (1, teacher.source_xyz.shape[0], 3)
            or not torch.equal(target[0], teacher.source_xyz)
            or state.scene_triplanes.shape[0] != 1
        ):
            raise ValueError("RSS_VNEXT_PRODUCT_SEEDED_SOURCE_TARGET_OR_LAYER_DRIFT")
        if layered:
            if (
                not 2 <= len(layers) <= schema.count
                or not isinstance(batch.flexicubes_signed_layer_admission_keys,tuple)
                or len(batch.flexicubes_signed_layer_admission_keys)!=len(layers)
                or len(set(batch.flexicubes_signed_layer_admission_keys))!=len(layers)
                or batch.flexicubes_signed_layer_admission_keys[0]
                    != batch.flexicubes_signed_source_admission_key
                or teacher.source_layer_ids is None
                or teacher.source_component_ids is None
                or teacher.source_layer_ids.shape!=(teacher.source_xyz.shape[0],)
                or teacher.source_layer_ids.dtype!=torch.long
                or bool((teacher.source_layer_ids<0).any())
                or int(teacher.source_layer_ids.max()) >= len(layers)
            ):
                raise ValueError("RSS_VNEXT_PRODUCT_SEEDED_LAYER_SOURCE_ADMISSION_REQUIRED")
            claimed=set()
            for spec in layers:
                spec.validate(schema)
                if claimed.intersection(spec.keep_roles):
                    raise ValueError("RSS_VNEXT_PRODUCT_SEEDED_LAYER_ROLE_OVERLAP")
                claimed.update(spec.keep_roles)
            namespace=layer_namespace_digest_v1(
                tuple(layer.name for layer in layers),
                tuple(layer.keep_roles for layer in layers),
            )
            keys=batch.flexicubes_signed_layer_admission_keys
        else:
            if (
                batch.flexicubes_signed_layer_admission_keys
                or batch.flexicubes_layer.name!="all"
                or tuple(batch.flexicubes_layer.keep_roles)!=tuple(schema.roles)
                or teacher.source_layer_ids is not None
            ):
                raise ValueError("RSS_VNEXT_PRODUCT_SEEDED_LAYER_SOURCE_TARGET_DRIFT")
            namespace=""
            keys=(batch.flexicubes_signed_source_admission_key,)
        if any(not isinstance(k,str) or not k for k in keys):
            raise ValueError("RSS_VNEXT_PRODUCT_SEEDED_LAYER_SIGNED_KEYS_INVALID")
        for layer_index,(layer,key) in enumerate(zip(layers,keys)):
            signed=self.trusted_signed_source_registry.get(key)
            native_part=(
                teacher.source_xyz[
                    teacher.source_layer_ids==layer_index
                ][None] if layered else target
            )
            if (
                native_part.shape[1] < 3
                or signed is None
                or signed.get("admission_status")
                    != "QUALIFIED_SOURCE_SIGNED_SOLID"
                or signed.get("source_component_domain") not in (
                    "SINGLE_QUALIFIED_CLOSED_COMPONENT",
                    "EXPLICIT_QUALIFIED_CLOSED_ASSEMBLY",
                )
                or signed.get("dataset_manifest_sha256")
                    != self.dataset_manifest_sha256
                or signed.get("source_qualification_key")
                    != batch.source_qualification_key
                or signed.get("role_schema_id")
                    != self.model.cfg.semantic_role_schema_id
                or signed.get("layer_name")!=layer.name
                or signed.get("surface_target_sha256")
                    != source_surface_tensor_sha256(native_part)
                or (
                    signed.get("source_component_domain")
                        == "EXPLICIT_QUALIFIED_CLOSED_ASSEMBLY"
                    and (
                        signed.get("assembly_boolean_operations_qualified") is not True
                        or signed.get("sdf_distance_scope")
                            != "EXACT_BOUNDARY_OF_THE_SAME_QUALIFIED_SOLID"
                        or signed.get("sdf_sign_scope")
                            != "GLOBAL_NEGATIVE_INSIDE_POSITIVE_OUTSIDE"
                    )
                )
            ):
                raise ValueError("RSS_VNEXT_PRODUCT_SEEDED_SIGNED_SOURCE_NOT_ADMITTED")
            if layered and any(
                signed.get(k)!=v for k,v in {
                    "source_layer_partition_authority":SOURCE_LAYER_AUTHORITY,
                    "source_layer_partition_derivation":SOURCE_LAYER_DERIVATION,
                    "native_source_layer_partition_sha256":
                        seeded_source_tensor_digest_v1(teacher.source_layer_ids),
                    "source_layer_namespace_sha256":namespace,
                    "source_layer_native_asset_sha256":
                        teacher.source_asset_sha256,
                    "native_source_layer_index":layer_index,
                    "source_layer_independently_audited":True,
                    "source_label_student_checkpoint_dependency":False,
                }.items()
            ):
                raise ValueError("RSS_VNEXT_PRODUCT_SEEDED_NATIVE_SIGNED_LAYER_RECEIPT_DRIFT")
        receipt = driver.trusted_registry.get(
            teacher.source_qualification_key,
        )
        if receipt is None or any(
            receipt.get(key) != value for key,value in {
                "dataset_manifest_sha256":self.dataset_manifest_sha256,
                "source_qualification_key":batch.source_qualification_key,
                "observed_rgba_sha256":
                    source_surface_tensor_sha256(batch.rgba),
                "observed_camera_sha256":
                    source_surface_tensor_sha256(batch.camera_features),
                "source_binding_sha256":
                    batch.product_source_binding_sha256,
                "combined_generation_checkpoint_sha256":
                    batch.product_generation_checkpoint_sha256,
                "signed_source_admission_key":
                    batch.flexicubes_signed_source_admission_key,
                "semantic_layer_name":batch.flexicubes_layer.name,
                **({
                    "signed_layer_admission_keys":list(keys),
                    "source_layer_namespace_sha256":namespace,
                } if layered else {}),
            }.items()
        ):
            raise ValueError("RSS_VNEXT_PRODUCT_SEEDED_SOURCE_IMAGE_OR_SCENE_BINDING")
        surface = self.model.fc_surface_bridge(
            state.scene_triplanes,
            extractor=batch.flexicubes_extractor,
            role_head=self.model.presentation_role_head,
            schema=self.model.cfg.semantic_role_schema(),
            layer=batch.flexicubes_layer,
            field_policy="SIGNED_HYPOTHESIS_ONLY",
            high_resolution_training=batch.flexicubes_high_resolution_training,
        )
        if layered:
            from .layered_iris_seed_v1 import make_layered_iris_seed_v1
            evidence=[(layers[0],surface)]
            for spec in layers[1:]:
                shell=self.model.fc_surface_bridge(
                    state.scene_triplanes,
                    extractor=batch.flexicubes_extractor,
                    role_head=self.model.presentation_role_head,
                    schema=schema,
                    layer=spec,field_policy="SIGNED_HYPOTHESIS_ONLY",
                    high_resolution_training=batch.flexicubes_high_resolution_training,
                )
                evidence.append((spec,shell))
            seed=make_layered_iris_seed_v1(
                evidence=tuple(evidence),schema=schema,
                scene_triplanes=state.scene_triplanes,
            ).seed
        else:
            seed = tessa_seed_from_flexicubes_v1(
                witness=surface,scene_triplanes=state.scene_triplanes,
            )
        candidate = self.model.build_seeded_mechanics_candidate_v1(
            state, seed, adaptive_refinement=True,
        )
        mesh = candidate["mesh_candidate"]
        if (
            candidate["mesh_status"] != "UNQUALIFIED_SINGLE_SURFACE_CANDIDATE"
            or mesh.seeded_surface is None
        ):
            raise RuntimeError("RSS_VNEXT_PRODUCT_SEEDED_STUDENT_MESH_INVALID")
        losses = driver.objective(
            mesh, teacher,
            source_face_budget=batch.seeded_tessa_source_face_budget,
            sampler_step=self.global_step,
        )
        axis_mira=product_seeded_axis_mira_objective_v1(
            model=self.model,state=state,mesh=mesh,
            teacher=teacher,batch=batch,
            weights=self.product_axis_mira_weights,
            dataset_manifest_sha256=self.dataset_manifest_sha256,
            trusted_axis_teacher_registry=self.trusted_axis_teacher_registry,
            trusted_native_pose_registry=driver.trusted_registry,
            trusted_chain_teacher_registry=self.trusted_chain_teacher_registry,
            tensor_digest=source_surface_tensor_sha256,
            axis_sequence_digest=axis_teacher_sequence_sha256_v6,
            frozen_codec_digest=frozen_mira_codec_state_sha256_v6,
        )
        component = (
            self._component_udf_objective_v1(batch,state)
            if self.component_udf_enabled else None
        )
        rgb_loss,n_rgb=product_source_graphics_loss_v1(
            model=self.model,state=state,
            source_rgb=batch.source_graphics_rgb_target,
            source_directions=batch.source_graphics_view_directions,
            source_mask=batch.source_graphics_rgb_mask,
            admission_key=batch.source_graphics_rgb_admission_key,
            receipt_registry=self.trusted_graphics_rgb_registry,
            dataset_manifest_sha256=self.dataset_manifest_sha256,
            source_qualification_key=batch.source_qualification_key,
            source_asset_sha256=teacher.source_asset_sha256,
            original_rgba=batch.rgba,
            original_cameras=batch.camera_features,
            generated_source_binding_sha256=batch.product_source_binding_sha256,
            generation_checkpoint_sha256=batch.product_generation_checkpoint_sha256,
            tensor_digest=source_surface_tensor_sha256,
        )
        ray_loss,n_rays,ray_parts=product_source_joint_ray_loss_v1(
            model=self.model,state=state,
            ray_xyz=batch.realsas_joint_ray_xyz,
            ray_delta=batch.realsas_joint_ray_delta,
            ray_depth=batch.realsas_joint_ray_depth,
            ray_direction=batch.realsas_joint_ray_direction,
            keep_roles=batch.realsas_joint_keep_roles,
            target=batch.realsas_joint_target,
            admission_key=batch.realsas_joint_teacher_admission_key,
            receipt_registry=self.trusted_realsas_joint_scene_registry,
            dataset_manifest_sha256=self.dataset_manifest_sha256,
            source_qualification_key=batch.source_qualification_key,
            source_asset_sha256=teacher.source_asset_sha256,
            original_rgba=batch.rgba,
            original_cameras=batch.camera_features,
            generated_source_binding_sha256=batch.product_source_binding_sha256,
            generation_checkpoint_sha256=batch.product_generation_checkpoint_sha256,
            tensor_digest=source_surface_tensor_sha256,
        )
        if (rgb_loss is not None and self.product_graphics_rgb_weight<=0
            or ray_loss is not None and self.product_joint_ray_weight<=0):
            raise ValueError("RSS_VNEXT_PRODUCT_GRAPHICS_SUPERVISION_NOT_ENABLED")
        if (
            (self.product_graphics_rgb_weight>0
             or self.product_joint_ray_weight>0)
            and rgb_loss is None and ray_loss is None
        ):
            raise ValueError("RSS_VNEXT_PRODUCT_GRAPHICS_ENABLED_NO_SOURCE_TRUTH")
        semantic=product_source_semantic_objective_v1(
            model=self.model,state=state,batch=batch,
            weights=self.product_semantic_weights,
            manifest_sha256=self.dataset_manifest_sha256,
            trusted_patch_registry=self.trusted_semantic_patch_registry,
            trusted_role_registry=self.trusted_surface_role_registry,
            trusted_region_registry=self.trusted_region_pair_registry,
            trusted_typed_registry=self.trusted_typed_relation_registry,
            tensor_digest=source_surface_tensor_sha256,
        )
        joint_total = losses["total"] + (
            self.component_udf_weight * component["total"]
            if component is not None else losses["total"].new_zeros(())
        )
        if axis_mira.total is not None:
            joint_total=joint_total+axis_mira.total
        if semantic.total is not None:
            joint_total=joint_total+semantic.total
        if rgb_loss is not None:
            joint_total=joint_total+self.product_graphics_rgb_weight*rgb_loss
        if ray_loss is not None:
            joint_total=joint_total+self.product_joint_ray_weight*ray_loss
        return RSSvNextTrainingResult(
            total_loss=joint_total,
            topology_loss=None,motion_preference_loss=None,
            source_semantic_loss=semantic.patch,
            semantic_risk_loss=semantic.patch_risk,
            flexicubes_surface_loss=None,flexicubes_animation_loss=None,
            eligible_animation_pose_count=0,
            eligible_semantic_patch_labels=semantic.patch_axes_count,
            eligible_topology_examples=0,
            qualified_preference_pairs=0,
            seeded_tessa_loss=losses["total"],
            product_axis_mira_loss=axis_mira.total,
            product_axis_mira_parts={
                k:v for k,v in (
                    ("axis",axis_mira.axis),("skin",axis_mira.skin),
                    ("posed",axis_mira.posed),("dynamic",axis_mira.dynamic),
                    ("hard_skin",axis_mira.hard_skin),
                    ("chain_skin",axis_mira.chain_skin),
                ) if v is not None
            },
            axis_rig_skin_loss=axis_mira.axis,
            axis_posed_surface_loss=axis_mira.posed,
            axis_dynamic_posed_surface_loss=axis_mira.dynamic,
            eligible_axis_sequences=(1 if axis_mira.axis is not None else 0),
            eligible_axis_pose_count=axis_mira.eligible_pose_faces,
            eligible_axis_dynamic_pose_count=(
                axis_mira.eligible_pose_faces if axis_mira.dynamic is not None else 0
            ),
            graphics_source_rgb_loss=rgb_loss,
            eligible_graphics_source_rgb_queries=n_rgb,
            realsas_joint_slrm_loss=ray_loss,
            eligible_realsas_joint_rays=n_rays,
            product_graphics_joint_loss_parts=ray_parts,
            product_semantic_loss=semantic.total,
            product_semantic_loss_parts={
                key:value for key,value in (
                    ("patch",semantic.patch),
                    ("patch_risk",semantic.patch_risk),
                    ("roles",semantic.roles),
                    ("region",semantic.region),
                    ("typed",semantic.typed),
                ) if value is not None
            },
            surface_role_loss=semantic.roles,
            region_pair_loss=semantic.region,
            typed_relation_loss=semantic.typed,
            eligible_surface_role_points=semantic.role_count,
            eligible_region_pairs=semantic.region_count,
            eligible_typed_relation_channels=semantic.typed_counts,
            component_udf_loss=(
                component["total"] if component is not None else None
            ),
            component_udf_loss_parts=({
                k:v for k,v in component.items() if
                k in ("field","kind","unmatched_unknown")
            } if component is not None else None),
            eligible_component_udf_source_components=(
                int(component["qualified_source_component_count"].item())
                if component is not None else 0
            ),
            seeded_tessa_loss_parts={
                k:v for k,v in losses.items()
                if k not in (
                    "total","qualified_source_points","sampled_source_points",
                    "sampled_student_vertices","sampled_quality_faces",
                    "sampling_step","qualified_source_layer_count",
                )
            },
            seeded_tessa_source_point_count=int(
                losses["qualified_source_points"].item()
            ),
            seeded_tessa_sampled_source_point_count=int(
                losses["sampled_source_points"].item()
            ),
            seeded_tessa_sampled_student_vertex_count=int(
                losses["sampled_student_vertices"].item()
            ),
            seeded_tessa_sampled_quality_face_count=int(
                losses["sampled_quality_faces"].item()
            ),
            seeded_tessa_qualified_layer_count=int(
                losses["qualified_source_layer_count"].item()
            ),
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
            ("source_signed_sdf", result.source_signed_sdf_loss, self.weights.source_signed_sdf),
            ("axis_rig_skin_codes", result.axis_rig_skin_loss, self.weights.axis_rig_skin_codes),
            ("axis_posed_surface", result.axis_posed_surface_loss, self.weights.axis_posed_surface),
            ("axis_dynamic_posed_surface", result.axis_dynamic_posed_surface_loss, self.weights.axis_dynamic_posed_surface),
            ("anchor_face_correspondence", result.anchor_face_correspondence_loss, self.weights.anchor_face_correspondence),
            ("flexicubes_surface", result.flexicubes_surface_loss, self.weights.flexicubes_surface),
            ("flexicubes_animation", result.flexicubes_animation_loss, self.weights.flexicubes_animation),
            ("tessa_iris_surface_alignment", result.tessa_iris_surface_alignment_loss, self.weights.tessa_iris_surface_alignment),
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
        if self.phase_contract is not None:
            self.phase_contract.assert_trainable_owners(self.model)
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
        if self.phase_contract is not None:
            self.phase_contract.assert_qualified_result(result, weights=self.weights)
        if (
            self.gradient_court_policy.every_n_steps > 0
            and self.global_step % self.gradient_court_policy.every_n_steps == 0
        ):
            result.gradient_interference_report = self._measure_gradient_conflicts(result)
        result.total_loss.backward()
        if self.training_profile in (
            "PRODUCT_SEEDED_IRIS_TESSA_V1","PRODUCT_COMPONENT_UDF_V1",
        ):
            required=[
                ("IRIS",self.model.iris),
            ]
            if self.training_profile == "PRODUCT_SEEDED_IRIS_TESSA_V1":
                required.append(("TESSA",self.model.tessa_seeded_head))
            if self.component_udf_enabled:
                required.extend((
                    ("COMPONENT_UDF",self.model.component_udf),
                    ("REGION_STRUCTURE",self.model.semantic_structure),
                ))
            if (
                result.graphics_source_rgb_loss is not None
                or result.realsas_joint_slrm_loss is not None
            ):
                required.append(("GRAPHICS",self.model.graphics))
            if result.graphics_source_rgb_loss is not None:
                required.extend((
                    ("SOURCE_RGB_VIEW_CORE",self.model.constructed_view_renderer.core),
                    ("SOURCE_RGB_VIEW_HEAD",self.model.constructed_view_renderer.rgb),
                ))
            if result.product_axis_mira_loss is not None:
                required.append(("AXIS",self.model.axis))
                required.append(("AXIS_FACE_CONTEXT",self.model.mesh_face_to_control))
                if (
                    result.product_axis_mira_parts
                    and any(v is not None for k,v in result.product_axis_mira_parts.items()
                            if k!="axis")
                ):
                    required.append(("MIRA",self.model.mira))
            if result.product_semantic_loss is not None:
                parts=result.product_semantic_loss_parts or {}
                if parts.get("patch") is not None or parts.get("patch_risk") is not None:
                    required.append(("SEMANTIC_PATCH",self.model.early_semantics))
                if parts.get("roles") is not None:
                    required.append(("SEMANTIC_ROLE",self.model.presentation_role_head))
                if parts.get("region") is not None or parts.get("typed") is not None:
                    required.extend((
                        ("SEMANTIC_REGION",self.model.semantic_structure),
                        ("SEMANTIC_REGION_HEAD",self.model.semantic),
                    ))
            for label,module in required:
                trained = [p for p in module.parameters() if p.requires_grad]
                if not trained or not any(p.grad is not None for p in trained):
                    raise RuntimeError(
                        "RSS_VNEXT_PRODUCT_SEEDED_GRADIENT_OWNER_MISSING:"+label
                    )
        if self.phase_contract is not None:
            self.phase_contract.assert_required_gradients(self.model)
        grad_norm = nn.utils.clip_grad_norm_(
            (p for p in self.model.parameters() if p.requires_grad),
            (
                self.seeded_tessa_weights.clip_gradient_norm
                if self.training_profile == "PRODUCT_SEEDED_IRIS_TESSA_V1"
                else self.weights.clip_gradient_norm
            ),
            error_if_nonfinite=True,
        )
        self.optimizer.step()
        self.global_step += 1
        result.gradient_norm = grad_norm
        return result

    def export_phase_handoff(
        self, path: str | Path, *, target_kind: str = "STRUCTURAL",
    ):
        """Export a one-way weights-only boundary, NEVER an optimizer resume."""
        from .training_phase_handoff_v1 import (
            export_rss_vnext_training_phase_handoff_v1,
        )
        self._assert_optimizer_ownership()
        return export_rss_vnext_training_phase_handoff_v1(
            path, model=self.model, source_kind="STRUCTURAL",
            target_kind=target_kind, source_global_step=self.global_step,
            dataset_manifest_sha256=self.dataset_manifest_sha256,
            source_objective_policy_sha256=self.objective_policy_sha256,
            source_optimizer_binding_sha256=self.optimizer_ownership.binding_sha256,
            architecture_id=self.model.architecture_id,
            architecture_digest=architecture_digest(self.model.cfg),
        )

    def import_phase_handoff(
        self, path: str | Path, *,
        expected_source_kind: str,
        expected_source_objective_policy_sha256: str,
        expected_source_optimizer_binding_sha256: str,
    ):
        """Begin a NEW structural phase using an already-fresh optimizer."""
        from .training_phase_handoff_v1 import (
            import_rss_vnext_training_phase_handoff_v1,
        )
        self._assert_optimizer_ownership()
        if self.global_step != 0:
            raise ValueError("RSS_VNEXT_PHASE_HANDOFF_DESTINATION_ALREADY_TRAINED")
        receipt = import_rss_vnext_training_phase_handoff_v1(
            path, model=self.model, optimizer=self.optimizer,
            target_kind="STRUCTURAL", expected_source_kind=expected_source_kind,
            expected_source_objective_policy_sha256=expected_source_objective_policy_sha256,
            expected_source_optimizer_binding_sha256=expected_source_optimizer_binding_sha256,
            dataset_manifest_sha256=self.dataset_manifest_sha256,
            architecture_id=self.model.architecture_id,
            architecture_digest=architecture_digest(self.model.cfg),
        )
        self.phase_handoff_source_model_sha256 = receipt.model_state_sha256
        return receipt

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
            **({
                "training_profile":self.training_profile,
                "seeded_tessa_registry_sha256":
                    self.seeded_tessa_registry_sha256,
                "seeded_tessa_policy_sha256":
                    self.seeded_tessa_driver.policy_sha256,
            } if self.training_profile == "PRODUCT_SEEDED_IRIS_TESSA_V1"
              else {}),
            **({
                "component_udf_registry_sha256":
                    self.component_udf_registry_sha256,
                "component_udf_weight":self.component_udf_weight,
                "component_udf_training_profile":self.training_profile,
            } if self.component_udf_enabled else {}),
            "optimizer_ownership_binding_sha256": self.optimizer_ownership.binding_sha256,
            "phase_handoff_source_model_sha256": self.phase_handoff_source_model_sha256,
            "render_registry_sha256": self.render_registry_sha256,
            "generated_conditioning_registry_sha256": self.generated_conditioning_registry_sha256,
            "surface_role_registry_sha256": self.surface_role_registry_sha256,
            "semantic_patch_registry_sha256": self.semantic_patch_registry_sha256,
            "region_pair_registry_sha256": self.region_pair_registry_sha256,
            "typed_relation_registry_sha256": self.typed_relation_registry_sha256,
            "axis_teacher_registry_sha256": self.axis_teacher_registry_sha256,
            "chain_teacher_registry_sha256": self.chain_teacher_registry_sha256,
            "topology_teacher_registry_sha256": self.topology_teacher_registry_sha256,
            "axis_pose_registry_sha256": self.axis_pose_registry_sha256,
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
        if self.training_profile == "PRODUCT_SEEDED_IRIS_TESSA_V1" and (
            data.get("training_profile") != self.training_profile
            or data.get("seeded_tessa_registry_sha256")
                != self.seeded_tessa_registry_sha256
            or data.get("seeded_tessa_policy_sha256")
                != self.seeded_tessa_driver.policy_sha256
        ):
            raise ValueError("RSS_VNEXT_TRAIN_SEEDED_CHECKPOINT_REGISTRY_OR_PHASE_DRIFT")
        if self.component_udf_enabled and (
            data.get("component_udf_registry_sha256")
                != self.component_udf_registry_sha256
            or data.get("component_udf_weight")
                != self.component_udf_weight
            or data.get("component_udf_training_profile")
                != self.training_profile
        ):
            raise ValueError("RSS_VNEXT_COMPONENT_UDF_CHECKPOINT_PHASE_OR_REGISTRY_DRIFT")
        if data.get("phase_handoff_source_model_sha256", "") != self.phase_handoff_source_model_sha256:
            raise ValueError("RSS_VNEXT_TRAIN_CHECKPOINT_PHASE_ORIGIN_DRIFT")
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
        if data.get("topology_teacher_registry_sha256", "") != self.topology_teacher_registry_sha256:
            raise ValueError("RSS_VNEXT_TRAIN_CHECKPOINT_TOPOLOGY_TEACHER_REGISTRY_DRIFT")
        if data.get("axis_teacher_registry_sha256", "") != self.axis_teacher_registry_sha256:
            raise ValueError("RSS_VNEXT_TRAIN_CHECKPOINT_AXIS_TEACHER_REGISTRY_DRIFT")
        if data.get("chain_teacher_registry_sha256", "") != self.chain_teacher_registry_sha256:
            raise ValueError("RSS_VNEXT_TRAIN_CHECKPOINT_CHAIN_TEACHER_REGISTRY_DRIFT")
        if data.get("axis_pose_registry_sha256", "") != self.axis_pose_registry_sha256:
            raise ValueError("RSS_VNEXT_TRAIN_CHECKPOINT_AXIS_POSE_REGISTRY_DRIFT")
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
