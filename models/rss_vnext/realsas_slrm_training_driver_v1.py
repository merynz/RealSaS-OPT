from __future__ import annotations

"""First-party S-LRM clean-room training driver with 1/6-view neural conditioning.

References: the CVPR 2025 multi-view reference §3.2, InstantNeRF/InstantMesh S-LRM.
Stage1/2: jointly render RGB, opacity and semantic posterior through the
SAME IRIS H(x). Stage3: separately extract pinned FlexiCubes surface from
the SAME scene and use PyTorch3D's compiled CUDA raster (CPU: BUSTED-owned
reference raster) for RGB, opacity, semantics, normals, depth and deformation.
Camera/teacher
render authority stays with independently admitted source-3D data. An
optional second observed image never becomes six 'source' observations.

IMPORTANT: not a proof of pretrained InstantNeRF initialization, LoRA
stage policy, full multiview refinement or published-reference-level image quality.
"""

from dataclasses import dataclass,asdict
from hashlib import sha256
import json
from pathlib import Path
import os
import tempfile
from types import MappingProxyType
from typing import Callable,Mapping

import torch
from torch import Tensor,nn

from .realsas_scene_conditioning_v1 import (
    encode_realsas_conditioning_into_shared_H_v1,
    CANONICAL_REFERENCE,SIX_VIEW_PRIOR,ONE_MEASURED_SOURCE,
)
from .realsas_joint_slrm_render_v1 import (
    STAGE_1,STAGE_2,STAGE_3,RealSaSJointTargetV1,
    render_realsas_joint_from_shared_scene_v1,realsas_joint_stage_loss_v1,
)
from .realsas_flexicubes_mesh_render_v1 import (
    RASTER_ID as STAGE3_CPU_REFERENCE_ID,
    PYTORCH3D_RASTER_ID,
    stage3_default_raster_backend_v1,
    render_realsas_layered_flexicubes_mesh_v1,
)
from .unified_character_model import RSSvNextUnifiedCharacterModel
from .training_optimizer_ownership_v1 import qualify_rss_vnext_optimizer_ownership_v1
from .semantic_geometry_prior import RSSvNextSemanticLayerSpec
from .realsas_observed_source_ray_tether_v1 import (
    CAMERA_ABI as ORIGINAL_RAY_CAMERA_ABI,
    RSSvNextOriginalObservedRayBatchV1,
    construct_original_observed_ray_samples_v1,
    original_observed_source_photometric_loss_v1,
)


def _sha(v: object) -> bool:
    return type(v) is str and len(v)==64 and all(x in "0123456789abcdef" for x in v)


def _digest(t: Tensor) -> str:
    if type(t) is not Tensor and not isinstance(t,Tensor):
        raise ValueError("RSS_VNEXT_REALSAS_SLRM_TENSOR_RECEIPT_REQUIRED")
    cpu=t.detach().contiguous().cpu()
    metadata=json.dumps({"dtype":str(cpu.dtype),"shape":list(cpu.shape)},
        sort_keys=True,separators=(",",":")).encode()
    return sha256(metadata+b"\0"+cpu.numpy().tobytes()).hexdigest()


@dataclass(frozen=True)
class RealSaSSLRMBatchV1:
    conditioning_rgba: Tensor
    conditioning_cameras: Tensor
    conditioning_authority: str
    source_binding_sha256: str
    original_observed_view_count: int
    generating_checkpoint_sha256: str | None
    ray_xyz: Tensor             # [B,R,S,3] qualified camera ray sampling policy
    ray_deltas: Tensor          # [B,R,S]
    ray_depths: Tensor          # [B,R,S]
    ray_directions: Tensor      # [B,R,3]
    keep_roles: Tensor          # [B,R,K]
    source_3d_render_target: RealSaSJointTargetV1
    admitted_3d_render_key: str
    stage3_flexicubes_extractor: object | None = None
    stage3_semantic_layer: object | None = None
    stage3_camera_features: Tensor | None = None
    stage3_hw: tuple[int,int] | None = None
    stage3_target_normal: Tensor | None = None
    stage3_target_depth: Tensor | None = None
    stage3_normal_mask: Tensor | None = None
    stage3_depth_mask: Tensor | None = None
    conditioning_normals: Tensor | None = None  # learned 6-view normal conditioning
    # Optional original pixels: source camera geometry must be independently
    # qualified. The six constructed views are NEVER their replacement.
    source_observed_rays: RSSvNextOriginalObservedRayBatchV1 | None = None
    source_observed_admission_key: str = ""


@dataclass(frozen=True)
class RealSaSSLRMTrainingLossWeightsV1:
    rgb: float=1.
    alpha: float=1.
    semantic: float=1.
    lpips: float=1.
    mesh_normal: float=1.
    mesh_depth: float=1.
    flexicubes_deformation: float=1.
    observed_source_rgb: float=0.
    observed_source_alpha: float=0.

    def validate(self) -> None:
        import math
        for value in asdict(self).values():
            if type(value) not in (float,int) or not math.isfinite(value) or value<0:
                raise ValueError("RSS_VNEXT_REALSAS_SLRM_LOSS_WEIGHT_DOMAIN")
        if self.rgb+self.alpha+self.semantic+self.lpips<=0:
            raise ValueError("RSS_VNEXT_REALSAS_SLRM_NO_ACTIVE_LOSS")


REALSAS_SIX_VIEW_TRAINING_PROFILE="REALSAS_2025_SIX_PREDICTED_RGB_SLRM_BASELINE"
RESEARCH_ONE_VIEW_TRAINING_PROFILE="BUSTED_DIRECT_SINGLE_VIEW_SLRM_ABLATION_NOT_REALSAS"


class RealSaSSLRMTrainerV1:
    """Actual optimizer step, stages 1/2 with 1/6 source-conditioned H(x).

    The caller's independently attested 3D render registry is required for
    each minibatch. It binds original source, generated input, cameras,
    camera rays, peeled role mask, RGB, alpha and semantic GT.
    """
    schema="RealSaS.RSSvNext.RealSaSSLRMTrainer.v5"

    def __init__(
        self, model: RSSvNextUnifiedCharacterModel, optimizer: torch.optim.Optimizer,
        *, dataset_manifest_sha256: str,
        trusted_source_3d_render_registry: Mapping[str,Mapping[str,object]],
        trusted_original_observed_ray_registry: Mapping[str,Mapping[str,object]] | None = None,
        lpips_loss_fn: Callable[[Tensor,Tensor],Tensor] | None = None,
        weights: RealSaSSLRMTrainingLossWeightsV1 = RealSaSSLRMTrainingLossWeightsV1(),
        clip_gradient_norm: float=1.,
        input_profile: str=REALSAS_SIX_VIEW_TRAINING_PROFILE,
    ) -> None:
        weights.validate()
        if input_profile not in (
            REALSAS_SIX_VIEW_TRAINING_PROFILE,RESEARCH_ONE_VIEW_TRAINING_PROFILE,
        ):
            raise ValueError("RSS_VNEXT_REALSAS_SLRM_TRAINING_PROFILE_INVALID")
        if not _sha(dataset_manifest_sha256) or not isinstance(model,RSSvNextUnifiedCharacterModel):
            raise ValueError("RSS_VNEXT_REALSAS_SLRM_TRAINER_CONFIG")
        if clip_gradient_norm<=0:
            raise ValueError("RSS_VNEXT_REALSAS_SLRM_GRAD_CLIP_INVALID")
        self.model=model;self.optimizer=optimizer;self.weights=weights
        self.optimizer_ownership=qualify_rss_vnext_optimizer_ownership_v1(
            self.model,self.optimizer,
        )
        self.input_profile=input_profile
        self.stage3_raster_device_type=next(model.parameters()).device.type
        if self.stage3_raster_device_type not in ("cpu","cuda"):
            raise ValueError("RSS_VNEXT_SLRM_STAGE3_RASTER_DEVICE_UNSUPPORTED")
        backend=stage3_default_raster_backend_v1(
            torch.device(self.stage3_raster_device_type),
        )
        self.stage3_raster_backend_id=(
            PYTORCH3D_RASTER_ID if self.stage3_raster_device_type=="cuda"
            else STAGE3_CPU_REFERENCE_ID
        )
        if backend is None:
            raise ValueError("RSS_VNEXT_SLRM_STAGE3_RASTER_BACKEND_UNAVAILABLE")
        self.lpips_loss_fn=lpips_loss_fn
        if weights.lpips>0 and lpips_loss_fn is None:
            raise ValueError("RSS_VNEXT_REALSAS_SLRM_LPIPS_REQUIRED_NOT_MSE_SURROGATE")
        self.manifest_sha256=dataset_manifest_sha256
        snapshot={str(key):dict(value) for key,value in trusted_source_3d_render_registry.items()}
        if not snapshot:
            raise ValueError("RSS_VNEXT_REALSAS_SLRM_SOURCE_RENDER_REGISTRY_REQUIRED")
        self.trusted_source_3d_render_registry=MappingProxyType({
            k:MappingProxyType(v) for k,v in snapshot.items()
        })
        self.registry_sha256=sha256(json.dumps(
            snapshot,sort_keys=True,separators=(",",":"),allow_nan=False,
        ).encode()).hexdigest()
        original_snapshot={
            str(key):dict(value)
            for key,value in (trusted_original_observed_ray_registry or {}).items()
        }
        self.trusted_original_observed_ray_registry=MappingProxyType({
            k:MappingProxyType(v) for k,v in original_snapshot.items()
        })
        self.original_observed_ray_registry_sha256=sha256(json.dumps(
            original_snapshot,sort_keys=True,separators=(",",":"),allow_nan=False,
        ).encode()).hexdigest()
        if (
            (weights.observed_source_rgb>0 or weights.observed_source_alpha>0)
            and not original_snapshot
        ):
            raise ValueError("RSS_VNEXT_REALSAS_SLRM_ORIGINAL_SOURCE_REGISTRY_REQUIRED")
        self.policy_sha256=sha256(json.dumps(
            {"weights":asdict(weights),"input_profile":input_profile,
             "stage3_raster_backend":self.stage3_raster_backend_id,
             "stage3_device":self.stage3_raster_device_type},
            sort_keys=True,separators=(",",":"),
        ).encode()).hexdigest()
        self.clip_gradient_norm=clip_gradient_norm
        self.global_step=0

    def _verify_receipt(self,batch:RealSaSSLRMBatchV1) -> None:
        t=batch.source_3d_render_target
        # Published CVPR 2025 baseline: ONE input -> canonicalization ->
        # SIX generated orthographic RGBs -> S-LRM. Direct one-view
        # S-LRM is BUSTED-specific ablation, never advertised as baseline.
        input_views=batch.conditioning_rgba.shape[1] if batch.conditioning_rgba.ndim==5 else -1
        if (
            (self.input_profile==REALSAS_SIX_VIEW_TRAINING_PROFILE and (
                input_views!=6 or batch.conditioning_authority!=SIX_VIEW_PRIOR
            )) or
            (self.input_profile==RESEARCH_ONE_VIEW_TRAINING_PROFILE and (
                input_views!=1 or batch.conditioning_authority not in (
                    CANONICAL_REFERENCE,ONE_MEASURED_SOURCE,
                )
            ))
        ):
            raise ValueError("RSS_VNEXT_REALSAS_SLRM_SIX_VIEW_BASELINE_VS_ONE_VIEW_ABLATION")
        # Six generated RGB views without their normal-map partner are not
        # the joint RGB+normal construction prescribed for this product
        # profile. Omitting normals silently leaves normal_condition_proj
        # without the input it was designed to learn. The direct one-view
        # research profile has no generated normal teacher.
        normals=batch.conditioning_normals
        if self.input_profile==REALSAS_SIX_VIEW_TRAINING_PROFILE:
            if (
                not isinstance(normals,Tensor)
                or normals.shape!=(
                    batch.conditioning_rgba.shape[0],6,3,
                    *batch.conditioning_rgba.shape[-2:],
                )
                or normals.dtype!=batch.conditioning_rgba.dtype
                or normals.device!=batch.conditioning_rgba.device
                or not bool(torch.isfinite(normals).all())
                or bool(((normals < -1.0001) | (normals > 1.0001)).any())
            ):
                raise ValueError("RSS_VNEXT_REALSAS_SLRM_SIX_VIEW_NORMAL_CONDITION_REQUIRED")
        elif normals is not None:
            raise ValueError("RSS_VNEXT_REALSAS_SLRM_ONE_VIEW_NORMAL_CONDITION_FORBIDDEN")
        if type(t) is not RealSaSJointTargetV1 or not batch.admitted_3d_render_key:
            raise ValueError("RSS_VNEXT_REALSAS_SLRM_SOURCE_TARGET_REQUIRED")
        if t.stage not in (STAGE_1,STAGE_2,STAGE_3):
            raise ValueError("RSS_VNEXT_REALSAS_SLRM_UNKNOWN_STAGE")
        inputs={
            "admission_status":"QUALIFIED_SOURCE_3D_RENDER_MULTILAYER",
            "slrm_training_input_profile":self.input_profile,
            "dataset_manifest_sha256":self.manifest_sha256,
            "original_source_binding_sha256":batch.source_binding_sha256,
            "original_observed_view_count":batch.original_observed_view_count,
            "conditioning_authority":batch.conditioning_authority,
            "conditioning_checkpoint_sha256":batch.generating_checkpoint_sha256,
            "conditioning_rgba_sha256":_digest(batch.conditioning_rgba),
            # The normal map is a live component of the SAME H(x) encoder.
            # Its bytes must be receipt-bound just like RGB and cameras,
            # otherwise source teacher records would survive conditioning
            # changes and wrongly claim a stationary shared scene.
            "conditioning_normals_sha256":(
                _digest(normals) if normals is not None else None
            ),
            "conditioning_cameras_sha256":_digest(batch.conditioning_cameras),
            "ray_xyz_sha256":_digest(batch.ray_xyz),
            "ray_deltas_sha256":_digest(batch.ray_deltas),
            "ray_depths_sha256":_digest(batch.ray_depths),
            "ray_directions_sha256":_digest(batch.ray_directions),
            "keep_roles_sha256":_digest(batch.keep_roles),
            "target_rgb_sha256":_digest(t.rgb_white),
            "target_alpha_sha256":_digest(t.alpha),
            "target_role_sha256":_digest(t.role_indices),
            "target_rgb_mask_sha256":_digest(t.qualified_rgb_mask),
            "target_alpha_mask_sha256":_digest(t.qualified_mask_mask),
            "target_role_mask_sha256":_digest(t.qualified_role_mask),
            "training_stage":t.stage,
            "semantic_layer_name":t.layer_name,
            "source_3d_truth_binding_sha256":t.source_3d_truth_binding_sha256,
            "generated_2d_as_3d_teacher":False,
            "source_rights_independently_admitted":True,
            "role_visibility_from_source_3d":True,
        }
        if t.stage==STAGE_3:
            stage3=(
                batch.stage3_flexicubes_extractor,batch.stage3_semantic_layer,
                batch.stage3_camera_features,batch.stage3_hw,
                batch.stage3_target_normal,batch.stage3_target_depth,
                batch.stage3_normal_mask,batch.stage3_depth_mask,
            )
            if any(x is None for x in stage3):
                raise ValueError("RSS_VNEXT_REALSAS_SLRM_STAGE3_MESH_RENDER_REQUIRED")
            layer=batch.stage3_semantic_layer
            if getattr(layer,"name",None)!=t.layer_name:
                raise ValueError("RSS_VNEXT_REALSAS_SLRM_STAGE3_SEMANTIC_LAYER_DRIFT")
            schema=self.model.cfg.semantic_role_schema()
            if not hasattr(layer,"validate") or not hasattr(layer,"keep_roles"):
                raise ValueError("RSS_VNEXT_REALSAS_SLRM_STAGE3_ROLE_LAYER_CONTRACT")
            layer.validate(schema)
            expected_keep=torch.tensor(
                [role in layer.keep_roles for role in schema.roles],
                device=batch.keep_roles.device,dtype=torch.bool,
            )[None,None,:].expand_as(batch.keep_roles)
            if not torch.equal(batch.keep_roles,expected_keep):
                raise ValueError("RSS_VNEXT_REALSAS_SLRM_STAGE3_RENDER_ROLE_PEEL_MISMATCH")
            camera=batch.stage3_camera_features
            normal=batch.stage3_target_normal
            depth=batch.stage3_target_depth
            nmask=batch.stage3_normal_mask
            dmask=batch.stage3_depth_mask
            hw=batch.stage3_hw
            if (
                type(hw) is not tuple or len(hw)!=2
                or any(type(x) is not int or x<8 or x>1024 for x in hw)
                or not isinstance(camera,Tensor) or camera.ndim!=3
                or camera.shape[0]!=1 or camera.shape[1] not in (1,6)
                or camera.shape[2]!=13
                or hw[0]*hw[1]*camera.shape[1]!=t.rgb_white.shape[1]
                or normal.shape!=(*t.rgb_white.shape[:2],3)
                or depth.shape!=(*t.rgb_white.shape[:2],1)
                or nmask.shape!=t.alpha.shape[:2]
                or dmask.shape!=t.alpha.shape[:2]
                or nmask.dtype!=torch.bool or dmask.dtype!=torch.bool
                or not bool(nmask.any()) or not bool(dmask.any())
                or not all(bool(torch.isfinite(x).all()) for x in (camera,normal,depth))
                or not bool((camera[:,:,12]>0).all())
                or not bool((t.alpha[...,0][nmask]>0).all())
                or not bool((t.alpha[...,0][dmask]>0).all())
                or not bool((depth[...,0][dmask]>=0).all())
            ):
                raise ValueError("RSS_VNEXT_REALSAS_SLRM_STAGE3_SOURCE_RENDER_DOMAIN")
            # Stage3 target normal/depth must have been independently
            # source-rendered IN THE SAME WORLD/CAMERA convention as mesh.
            inputs.update({
                "independently_qualified_closed_signed_source":True,
                "stage3_source_camera_sha256":_digest(camera),
                "stage3_normal_target_sha256":_digest(normal),
                "stage3_depth_target_sha256":_digest(depth),
                "stage3_normal_mask_sha256":_digest(nmask),
                "stage3_depth_mask_sha256":_digest(dmask),
                "stage3_render_height":hw[0],"stage3_render_width":hw[1],
                "stage3_world_normal_convention":"SOURCE_GEOMETRIC_NORMAL_WORLD_XYZ",
                "stage3_metric_depth_convention":"SOURCE_CAMERA_FORWARD_FIRST_HIT_WORLD_UNITS",
                "stage3_visibility_convention":"SOURCE_3D_ZBUFFER_VISIBLE_ONLY",
                "stage3_semantic_role_schema_id":self.model.cfg.semantic_role_schema_id,
                "stage3_layer_keep_roles":list(layer.keep_roles),
            })
        elif any(x is not None for x in (
            batch.stage3_flexicubes_extractor,batch.stage3_semantic_layer,
            batch.stage3_camera_features,batch.stage3_hw,
            batch.stage3_target_normal,batch.stage3_target_depth,
            batch.stage3_normal_mask,batch.stage3_depth_mask,
        )):
            raise ValueError("RSS_VNEXT_REALSAS_SLRM_MESH_TEACHER_ON_NERF_STAGE")
        receipt=self.trusted_source_3d_render_registry.get(batch.admitted_3d_render_key)
        if receipt is None or any(receipt.get(key)!=value for key,value in inputs.items()):
            raise ValueError("RSS_VNEXT_REALSAS_SLRM_RENDER_RECEIPT_DRIFT")
        if t.stage==STAGE_1 and not bool(batch.keep_roles.all()):
            raise ValueError("RSS_VNEXT_REALSAS_SLRM_STAGE1_MUST_KEEP_ALL_ROLES")
        if t.stage in (STAGE_2,STAGE_3) and bool(batch.keep_roles.all()):
            raise ValueError("RSS_VNEXT_REALSAS_SLRM_PEELED_STAGE_MUST_REMOVE_OUTER_ROLE")
        # Samples have no source 3D mesh at inference. Teacher arrays are
        # compared with independently verified bytes; *only* this step reads
        # privileged rendering target data.
        if t.generated_completion_used_as_truth:
            raise ValueError("RSS_VNEXT_REALSAS_SLRM_GENERATED_TEACHER_FORBIDDEN")
        source=batch.source_observed_rays
        key=batch.source_observed_admission_key
        source_enabled=(
            self.weights.observed_source_rgb>0 or
            self.weights.observed_source_alpha>0
        )
        if (source is not None or key) and not source_enabled:
            raise ValueError("RSS_VNEXT_REALSAS_SLRM_ORIGINAL_SOURCE_LOSS_NOT_ENABLED")
        if source_enabled:
            if (
                type(source) is not RSSvNextOriginalObservedRayBatchV1
                or not key
                or self.input_profile!=REALSAS_SIX_VIEW_TRAINING_PROFILE
                or batch.original_observed_view_count != 2
                or t.stage not in (STAGE_1,STAGE_2,STAGE_3)
            ):
                raise ValueError("RSS_VNEXT_REALSAS_SLRM_ORIGINAL_SOURCE_CALIBRATED_PAIR_REQUIRED")
            r=self.trusted_original_observed_ray_registry.get(key)
            fields={
                "admission_status":"QUALIFIED_NATIVE_TWO_SOURCE_CAMERA_PHOTOMETRY",
                "dataset_manifest_sha256":self.manifest_sha256,
                "original_source_binding_sha256":batch.source_binding_sha256,
                "original_observed_view_count":2,
                "original_linear_rgba_sha256":_digest(source.original_linear_rgba),
                "original_camera_13d_sha256":_digest(source.original_camera_13d),
                "source_pixel_indices_sha256":_digest(source.source_pixel_indices),
                "ray_depth_sha256":_digest(source.ray_depth),
                "ray_delta_sha256":_digest(source.ray_delta),
                "source_rgb_mask_sha256":_digest(source.qualified_rgb_mask),
                "source_alpha_mask_sha256":_digest(source.qualified_alpha_mask),
                "original_camera_abi":ORIGINAL_RAY_CAMERA_ABI,
                "original_cameras_independently_calibrated":True,
                "original_source_linear_rgb_straight_alpha":True,
                "source_pixels_are_native_not_generated":True,
                "source_rays_supply_3d_depth_truth":False,
                "generated_completion_used_as_source_photometry":False,
            }
            if r is None or any(r.get(k)!=v for k,v in fields.items()):
                raise ValueError("RSS_VNEXT_REALSAS_SLRM_ORIGINAL_SOURCE_RECEIPT_DRIFT")
            # Camera consistency and two-source pixel coverage are asserted
            # *after* cryptographic source binding, before student forward.
            construct_original_observed_ray_samples_v1(source)

    def objective(self,batch:RealSaSSLRMBatchV1) -> dict[str,Tensor]:
        if not self.model.training:
            raise RuntimeError("RSS_VNEXT_REALSAS_SLRM_REQUIRES_MODEL_TRAIN_MODE")
        self._verify_receipt(batch)
        state=encode_realsas_conditioning_into_shared_H_v1(
            self.model,conditioning_rgba=batch.conditioning_rgba,
            conditioning_cameras=batch.conditioning_cameras,
            source_binding_sha256=batch.source_binding_sha256,
            original_observed_view_count=batch.original_observed_view_count,
            conditioning_authority=batch.conditioning_authority,
            generating_checkpoint_sha256=batch.generating_checkpoint_sha256,
            conditioning_normals=batch.conditioning_normals,
        )
        teacher=batch.source_3d_render_target
        # Stage3 has its OWN mesh raster objective: never consume NeRF RGB,
        # role, alpha or depth as a stand-in for mesh rendering. Save memory.
        pred=None
        if teacher.stage in (STAGE_1,STAGE_2):
            pred=render_realsas_joint_from_shared_scene_v1(
                iris=self.model.iris,scene_triplanes=state.scene_triplanes,
                sample_xyz=batch.ray_xyz,ray_delta=batch.ray_deltas,
                ray_depth=batch.ray_depths,ray_direction=batch.ray_directions,
                keep_roles=batch.keep_roles,
                shared_graphics_head=self.model.graphics,
                shared_semantic_role_head=self.model.presentation_role_head,
            )
        mesh=None
        extra={}
        if teacher.stage==STAGE_3:
            # A resumed CPU/reference objective is NOT silently equivalent
            # to the compiled PyTorch3D CUDA stage. Device and raster
            # identities are checkpoint-pinned at trainer construction.
            if (
                next(self.model.parameters()).device.type
                != self.stage3_raster_device_type
            ):
                raise RuntimeError("RSS_VNEXT_SLRM_STAGE3_RASTER_ENVIRONMENT_DRIFT")
            if any(x is None for x in (
                batch.stage3_flexicubes_extractor,batch.stage3_semantic_layer,
                batch.stage3_camera_features,batch.stage3_hw,
                batch.stage3_target_normal,batch.stage3_target_depth,
                batch.stage3_normal_mask,batch.stage3_depth_mask,
            )):
                raise ValueError("RSS_VNEXT_REALSAS_SLRM_STAGE3_MESH_RENDER_REQUIRED")
            # Signed-solid status is an independently admitted teacher
            # qualification in the receipt, NOT an inference claim.
            signed=self.trusted_source_3d_render_registry[
                batch.admitted_3d_render_key
            ].get("independently_qualified_closed_signed_source")
            if signed is not True:
                raise ValueError("RSS_VNEXT_REALSAS_SLRM_STAGE3_SOURCE_SIGN_NOT_ADMITTED")
            layer=batch.stage3_semantic_layer
            if layer.name!=teacher.layer_name:
                raise ValueError("RSS_VNEXT_REALSAS_SLRM_STAGE3_SEMANTIC_LAYER_DRIFT")
            witness=self.model.fc_surface_bridge(
                state.scene_triplanes,extractor=batch.stage3_flexicubes_extractor,
                role_head=self.model.presentation_role_head,
                schema=self.model.cfg.semantic_role_schema(),
                layer=layer,field_policy="SIGNED_HYPOTHESIS_ONLY",
            )
            scene_digest=_digest(state.scene_triplanes)
            mesh=render_realsas_layered_flexicubes_mesh_v1(
                witness=witness,iris=self.model.iris,
                scene_triplanes=state.scene_triplanes,
                graphics_head=self.model.graphics,
                semantic_role_head=self.model.presentation_role_head,
                cameras=batch.stage3_camera_features,
                height=batch.stage3_hw[0],width=batch.stage3_hw[1],
                source_scene_prediction_sha256=scene_digest,
            )
            if mesh.rasterizer_identity!=self.stage3_raster_backend_id:
                raise RuntimeError("RSS_VNEXT_SLRM_STAGE3_RASTER_BACKEND_DRIFT")
            extra=dict(
                mesh_raster=mesh,target_normal=batch.stage3_target_normal,
                target_depth=batch.stage3_target_depth,
                qualified_normal_mask=batch.stage3_normal_mask,
                qualified_depth_mask=batch.stage3_depth_mask,
            )
        parts=realsas_joint_stage_loss_v1(
            pred,teacher,lpips_loss_fn=self.lpips_loss_fn,**extra,
        )
        if batch.source_observed_rays is not None:
            # Full source photographs are rendered with ALL roles, even
            # while the independent 3D teacher peels outer layers in Stage2.
            # The source target is never a guessed peeled image or 3D point.
            observed=construct_original_observed_ray_samples_v1(
                batch.source_observed_rays,
            )
            if teacher.stage==STAGE_3:
                # Stage3 must preserve original FRONT/BACK via an ACTUAL
                # FlexiCubes mesh raster, never an auxiliary NeRF ray proxy.
                # It is an all-role surface extracted from the SAME H(x);
                # the independent source photo contributes color/alpha only.
                rgba=batch.source_observed_rays.original_linear_rgba
                hh,ww=rgba.shape[-2:]
                if (
                    rgba.shape[0]!=1 or hh>1024 or ww>1024
                    or rgba.shape[1]!=2
                ):
                    raise ValueError("RSS_VNEXT_REALSAS_STAGE3_ORIGINAL_SOURCE_RASTER_DOMAIN")
                schema=self.model.cfg.semantic_role_schema()
                source_layer=RSSvNextSemanticLayerSpec(
                    "original_source_all_visible_roles",schema.roles,
                )
                full_witness=self.model.fc_surface_bridge(
                    state.scene_triplanes,
                    extractor=batch.stage3_flexicubes_extractor,
                    role_head=self.model.presentation_role_head,
                    schema=schema,layer=source_layer,
                    field_policy="SIGNED_HYPOTHESIS_ONLY",
                )
                full_mesh=render_realsas_layered_flexicubes_mesh_v1(
                    witness=full_witness,iris=self.model.iris,
                    scene_triplanes=state.scene_triplanes,
                    graphics_head=self.model.graphics,
                    semantic_role_head=self.model.presentation_role_head,
                    cameras=batch.source_observed_rays.original_camera_13d.detach(),
                    height=hh,width=ww,
                    source_scene_prediction_sha256=_digest(state.scene_triplanes),
                )
                if full_mesh.rasterizer_identity!=self.stage3_raster_backend_id:
                    raise RuntimeError("RSS_VNEXT_SLRM_ORIGINAL_SOURCE_RASTER_BACKEND_DRIFT")
                addr=batch.source_observed_rays.source_pixel_indices
                flat=(addr[...,0]*hh*ww+addr[...,1]*ww+addr[...,2]).long()
                selected_rgb=torch.gather(
                    full_mesh.rgb_white_composite,1,
                    flat[...,None].expand(-1,-1,3),
                )
                selected_alpha=torch.gather(
                    full_mesh.alpha,1,flat[...,None],
                )
            else:
                visible_all_roles=torch.ones(
                    (*observed.ray_direction.shape[:2],
                     self.model.cfg.semantic_role_schema().count),
                    dtype=torch.bool,device=observed.ray_direction.device,
                )
                source_prediction=render_realsas_joint_from_shared_scene_v1(
                    iris=self.model.iris,scene_triplanes=state.scene_triplanes,
                    sample_xyz=observed.sample_xyz,
                    ray_delta=observed.ray_delta,
                    ray_depth=observed.ray_depth,
                    ray_direction=observed.ray_direction,
                    keep_roles=visible_all_roles,
                    shared_graphics_head=self.model.graphics,
                    shared_semantic_role_head=self.model.presentation_role_head,
                )
                selected_rgb=source_prediction.rgb_white_composite
                selected_alpha=source_prediction.alpha
            parts.update(original_observed_source_photometric_loss_v1(
                selected_rgb,selected_alpha,observed,
            ))
        weighted={
            "rgb_mse":self.weights.rgb,"mask_mse":self.weights.alpha,
            "semantic_ce":self.weights.semantic,"lpips":self.weights.lpips,
            "normal_cosine":self.weights.mesh_normal,"depth_l1":self.weights.mesh_depth,
            "flexicubes_deformation":self.weights.flexicubes_deformation,
            "observed_source_rgb_mse":self.weights.observed_source_rgb,
            "observed_source_alpha_mse":self.weights.observed_source_alpha,
        }
        total=sum(weighted[k]*v for k,v in parts.items() if k in weighted)
        if not bool(torch.isfinite(total)):
            raise ValueError("RSS_VNEXT_REALSAS_SLRM_NONFINITE_LOSS")
        return {**parts,"total":total}

    def _assert_optimizer_ownership(self)->None:
        current=qualify_rss_vnext_optimizer_ownership_v1(
            self.model,self.optimizer,
        )
        if current.binding_sha256!=self.optimizer_ownership.binding_sha256:
            raise RuntimeError("RSS_VNEXT_REALSAS_SLRM_OPTIMIZER_PHASE_DRIFT")

    def train_step(self,batch:RealSaSSLRMBatchV1) -> dict[str,Tensor]:
        self._assert_optimizer_ownership()
        for p in self.model.parameters():
            if not p.requires_grad:
                p.grad=None
        self.optimizer.zero_grad(set_to_none=True)
        parts=self.objective(batch)
        parts["total"].backward()
        torch.nn.utils.clip_grad_norm_(
            (p for p in self.model.parameters() if p.requires_grad),
            self.clip_gradient_norm,error_if_nonfinite=True,
        )
        self.optimizer.step()
        self.global_step+=1
        return parts

    def save_checkpoint(self,path:str|Path,*,sampler_cursor:int)->None:
        if sampler_cursor<0:raise ValueError("RSS_VNEXT_REALSAS_SLRM_BAD_SAMPLER_CURSOR")
        self._assert_optimizer_ownership()
        filename=Path(path);filename.parent.mkdir(parents=True,exist_ok=True)
        payload={
            "schema":self.schema,"dataset_manifest_sha256":self.manifest_sha256,
            "teacher_registry_sha256":self.registry_sha256,
            "original_source_registry_sha256":self.original_observed_ray_registry_sha256,
            "objective_policy_sha256":self.policy_sha256,
            "stage3_raster_backend_identity":self.stage3_raster_backend_id,
            "stage3_raster_device_type":self.stage3_raster_device_type,
            "optimizer_ownership_binding_sha256":self.optimizer_ownership.binding_sha256,
            "global_step":self.global_step,"sampler_cursor":sampler_cursor,
            "model":self.model.state_dict(),
            "optimizer":self.optimizer.state_dict(),
            "torch_cpu_rng":torch.get_rng_state(),
        }
        with tempfile.NamedTemporaryFile(dir=filename.parent,prefix="slrm-",suffix=".pt",delete=False) as f:
            tmp=Path(f.name)
        try:
            torch.save(payload,tmp)
            os.replace(tmp,filename)
        finally:tmp.unlink(missing_ok=True)

    def load_checkpoint(self,path:str|Path)->int:
        data=torch.load(Path(path),map_location="cpu",weights_only=True)
        if data.get("schema")!=self.schema:
            raise ValueError("RSS_VNEXT_REALSAS_SLRM_CHECKPOINT_SCHEMA")
        if data.get("dataset_manifest_sha256")!=self.manifest_sha256:
            raise ValueError("RSS_VNEXT_REALSAS_SLRM_CHECKPOINT_CORPUS_DRIFT")
        if data.get("teacher_registry_sha256")!=self.registry_sha256:
            raise ValueError("RSS_VNEXT_REALSAS_SLRM_CHECKPOINT_RENDER_TRUTH_DRIFT")
        if data.get("original_source_registry_sha256")!=self.original_observed_ray_registry_sha256:
            raise ValueError("RSS_VNEXT_REALSAS_SLRM_CHECKPOINT_ORIGINAL_SOURCE_DRIFT")
        if data.get("objective_policy_sha256")!=self.policy_sha256:
            raise ValueError("RSS_VNEXT_REALSAS_SLRM_CHECKPOINT_OBJECTIVE_POLICY_DRIFT")
        if (
            data.get("stage3_raster_backend_identity")!=self.stage3_raster_backend_id
            or data.get("stage3_raster_device_type")!=self.stage3_raster_device_type
        ):
            raise ValueError("RSS_VNEXT_REALSAS_SLRM_CHECKPOINT_STAGE3_RASTER_DRIFT")
        self._assert_optimizer_ownership()
        if data.get("optimizer_ownership_binding_sha256")!=self.optimizer_ownership.binding_sha256:
            raise ValueError("RSS_VNEXT_REALSAS_SLRM_CHECKPOINT_OPTIMIZER_OWNERSHIP_DRIFT")
        self.model.load_state_dict(data["model"],strict=True)
        self.optimizer.load_state_dict(data["optimizer"])
        self.global_step=int(data["global_step"])
        torch.set_rng_state(data["torch_cpu_rng"])
        return int(data["sampler_cursor"])


__all__=["RealSaSSLRMBatchV1","RealSaSSLRMTrainingLossWeightsV1",
    "RealSaSSLRMTrainerV1","REALSAS_SIX_VIEW_TRAINING_PROFILE",
    "RESEARCH_ONE_VIEW_TRAINING_PROFILE"]
