from __future__ import annotations

"""Train the EXACT embedded MIRA FSQ codec used by RSS.vNext product.skin.

A source-native rig/skin codec trained with old 7D/25D CanonicalMesh/Rig
features is NOT the RSS shared-H checkpoint decoder. This source-teacher
bootstrap uses TESSA's *exact* reindexed mesh and preceding causal AXIS joint
states to train model.mira.codec itself, without running another MIRA model.

Only DOT_PRODUCT_BASELINE is eligible here: POINT_QUERY_CROSS_ATTENTION uses
an additional spatial decoder which must have its own source-supervised loss.
This module teaches no unknown source pixels, contact, or 2D draw order.
"""
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping
import json
import os
import tempfile

import torch
from torch import Tensor
import torch.nn.functional as F

from .animation_ready_mesh import posed_lbs_from_joint_matrices

from models.axis.rig_program_model_v6 import AXISSequenceV6, RigTokenKindV6
from .contracts import RSSvNextSceneState
from .unified_character_model import RSSvNextUnifiedCharacterModel
from .training_engine import (
    architecture_digest, axis_mesh_teacher_address_digest_v1,
    axis_teacher_sequence_sha256_v6, source_surface_tensor_sha256,
)

SCHEMA = "RealSaS.RSSvNext.EmbeddedMIRASourceCodecTrainer.v2"
_TEACHER = "QUALIFIED_SOURCE_MIRA_DENSE_SKIN_TESSA_AXIS_JOINT_REINDEX"


@dataclass(frozen=True)
class RSSvNextEmbeddedMIRACodecBatchV1:
    scene: RSSvNextSceneState
    structural_memory: Tensor
    topology_teacher_tokens: Tensor      # [1,T] source reindexed TESSA
    axis_teacher_sequence: AXISSequenceV6
    source_dense_weights: Tensor          # [1,V,J], full source-observed simplex
    source_binding_sha256: str
    source_qualification_key: str
    admission_key: str
    # Optional independent Blender-native POSED source mesh supervision.
    # All are exact source-asset/TESSA-vertex/AXIS-joint aligned teacher data.
    source_rest_vertices: Tensor | None = None        # [1,V,3]
    source_joint_skin_matrices: Tensor | None = None  # [1,P,J,4,4]
    source_posed_vertices: Tensor | None = None       # [1,P,V,3]
    source_pose_vertex_mask: Tensor | None = None     # [1,P,V] bool
    source_pose_admission_key: str = ""


@dataclass(frozen=True)
class RSSvNextEmbeddedMIRACodecResultV1:
    loss: Tensor
    reconstruction_loss: Tensor
    quantization_loss: Tensor
    token_ids: Tensor
    mesh_address_digest: str
    hard_decoder_replay_max_abs_error: float
    source_pose_deformation_loss: Tensor | None = None
    qualified_pose_vertex_count: int = 0
    hard_token_pose_replay_max_abs_error: float | None = None
    sparse_zero_influence_error: float | None = None
    sparse_nonzero_influence_error: float | None = None
    source_qualified: bool = False
    product_skin_qualified: bool = False
    generated_skin_program_qualified: bool = False


def _hex(v: object) -> bool:
    return type(v) is str and len(v) == 64 and all(
        c in "0123456789abcdef" for c in v
    )


def _digest_json(value: object) -> str:
    return sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


class RSSvNextEmbeddedMIRASourceCodecTrainerV1:
    """Codec-first phase, then freeze and tokenize teacher skin for AXIS.

    The trainer never exposes source weights, source 3D mesh, or AXIS labels to
    inference. Underlying source admission is independently signed by Foundry
    / trusted training launcher, not derived from student predictions.
    """

    def __init__(
        self,
        model: RSSvNextUnifiedCharacterModel,
        optimizer: torch.optim.Optimizer,
        *,
        dataset_manifest_sha256: str,
        trusted_source_skin_registry: Mapping[str, Mapping[str, Any]],
        trusted_source_pose_registry: Mapping[str, Mapping[str, Any]] | None = None,
        pose_consequence_weight: float = 0.,
    ):
        if (
            type(model) is not RSSvNextUnifiedCharacterModel
            or not isinstance(optimizer, torch.optim.Optimizer)
            or not _hex(dataset_manifest_sha256)
            or model.cfg.skin_decoder_mode != "DOT_PRODUCT_BASELINE"
            or not model.training
            or not any(p.requires_grad for p in model.mira.codec.parameters())
            or not trusted_source_skin_registry
        ):
            raise ValueError("RSS_VNEXT_EMBEDDED_MIRA_CODEC_TRAINER_POLICY")
        # Motion supervision is a distinct source-native family. Its weight
        # is bound to the training checkpoint and NEVER silently enabled by
        # merely receiving motion-shaped arrays from a generated scene.
        if (
            type(pose_consequence_weight) not in (int,float)
            or not __import__("math").isfinite(pose_consequence_weight)
            or pose_consequence_weight<0
        ):
            raise ValueError("RSS_VNEXT_EMBEDDED_MIRA_POSE_LOSS_POLICY")
        # A codec bootstrap optimizer must cover every trainable embedded
        # codec parameter, and MUST NOT update IRIS/TESSA/AXIS or the
        # untrained point-query decoder under an unrelated phase.
        codec_ids={id(p) for p in model.mira.codec.parameters() if p.requires_grad}
        optimizer_ids=[
            id(p) for group in optimizer.param_groups for p in group["params"]
        ]
        if (
            not codec_ids or len(optimizer_ids)!=len(set(optimizer_ids))
            or set(optimizer_ids)!=codec_ids
        ):
            raise ValueError("RSS_VNEXT_EMBEDDED_MIRA_CODEC_OPTIMIZER_SCOPE_DRIFT")
        self.model = model
        self.optimizer = optimizer
        self.dataset_manifest_sha256 = dataset_manifest_sha256
        snapshot = {
            str(k): dict(v) for k,v in trusted_source_skin_registry.items()
        }
        self.trusted_registry = MappingProxyType({
            k: MappingProxyType(v) for k,v in snapshot.items()
        })
        self.registry_sha256 = _digest_json(snapshot)
        pose_snapshot={
            str(k):dict(v) for k,v in (trusted_source_pose_registry or {}).items()
        }
        self.trusted_source_pose_registry=MappingProxyType({
            k:MappingProxyType(v) for k,v in pose_snapshot.items()
        })
        self.pose_registry_sha256=_digest_json(pose_snapshot)
        self.pose_consequence_weight=float(pose_consequence_weight)
        self.config_sha256 = architecture_digest(model.cfg)
        self.step = 0

    def objective(self, batch: RSSvNextEmbeddedMIRACodecBatchV1):
        model=self.model
        if not model.training or not model.mira.codec.training:
            raise RuntimeError("RSS_VNEXT_EMBEDDED_MIRA_CODEC_REQUIRES_TRAIN")
        if (
            type(batch) is not RSSvNextEmbeddedMIRACodecBatchV1
            or not _hex(batch.source_binding_sha256)
            or not batch.source_qualification_key
            or not batch.admission_key
            or type(batch.scene) is not RSSvNextSceneState
            or type(batch.axis_teacher_sequence) is not AXISSequenceV6
            or not isinstance(batch.topology_teacher_tokens, Tensor)
            or not isinstance(batch.source_dense_weights, Tensor)
        ):
            raise ValueError("RSS_VNEXT_EMBEDDED_MIRA_SOURCE_BATCH_ABI")
        state=batch.scene
        if (
            state.scene_conditioning_authority!="GENERATED_SIX_VIEW_SHARED_SCENE"
            or state.source_camera_authority!="UNKNOWN_NO_ORIGINAL_CAMERA_PROJECTION"
            or state.camera_features is not None
            or state.generated_view_count!=6
            or not _hex(state.generating_checkpoint_sha256)
            or batch.topology_teacher_tokens.ndim!=2
            or batch.topology_teacher_tokens.shape[0]!=1
            or batch.topology_teacher_tokens.dtype!=torch.long
            or batch.structural_memory.ndim!=3
            or batch.structural_memory.shape[0]!=1
            or batch.structural_memory.shape[-1]!=model.cfg.tessa.d_model
        ):
            raise ValueError("RSS_VNEXT_EMBEDDED_MIRA_GENERATED_PRODUCT_SCENE_REQUIRED")
        sequence=batch.axis_teacher_sequence
        sequence.validate(model.axis.config)
        kinds=sequence.kind_ids[0]
        valid=sequence.attention_mask[0].bool()
        joints=torch.where(valid & (kinds==int(RigTokenKindV6.JOINT)))[0]
        skins=torch.where(valid & (kinds==int(RigTokenKindV6.SKIN_CODE)))[0]
        n_joints=int(joints.numel())
        if (
            n_joints<2
            or bool((joints==0).any())
            or len(skins)!=n_joints*model.cfg.skin_tokens_per_joint
            or (len(skins) and int(skins.min())<=int(joints.max()))
        ):
            raise ValueError("RSS_VNEXT_EMBEDDED_MIRA_CAUSAL_JOINT_SKIN_DOMAIN")
        # Reject altered topology/AXIS bytes BEFORE student mesh decoding.
        # A forged batch must not invoke a variable-cost grammar from a
        # launcher-unadmitted source tensor.
        pre_receipt=self.trusted_registry.get(batch.admission_key)
        if (
            pre_receipt is None
            or pre_receipt.get("admission_status")!=_TEACHER
            or pre_receipt.get("dataset_manifest_sha256")!=self.dataset_manifest_sha256
            or pre_receipt.get("source_binding_sha256")!=batch.source_binding_sha256
            or pre_receipt.get("source_qualification_key")!=batch.source_qualification_key
            or pre_receipt.get("scene_generator_binding_sha256")!=state.generating_checkpoint_sha256
            or pre_receipt.get("mesh_teacher_tokens_sha256")!=source_surface_tensor_sha256(batch.topology_teacher_tokens)
            or pre_receipt.get("axis_teacher_sequence_sha256")!=axis_teacher_sequence_sha256_v6(sequence)
        ):
            raise ValueError("RSS_VNEXT_EMBEDDED_MIRA_SOURCE_ADMISSION_DRIFT")
        # Source rig/skin must be reindexed to the SAME discrete TESSA mesh.
        # Student rest refinement is a hypothesis; no source-native topology
        # is smuggled into inference. Detach shared state in codec-first phase.
        with torch.no_grad():
            mesh=model.build_mesh_candidate(
                state,batch.topology_teacher_tokens[0],
                source_authority="TEACHER_FORCED_TRAINING_ONLY",
            )
            memory=model._control_memory(state,batch.structural_memory,mesh)
            axis_logits=model.axis(memory,sequence)
            joints_hidden=axis_logits["hidden"][:,joints-1].detach()
            local=mesh.local_features.detach()
            rest=mesh.rest_vertices_iris.detach()
            normals=mesh.normals.detach()
        digest=axis_mesh_teacher_address_digest_v1(mesh)
        weights=batch.source_dense_weights
        v=rest.shape[1]
        if (
            weights.shape!=(1,v,n_joints)
            or not torch.is_floating_point(weights)
            or weights.device!=local.device
            or not bool(torch.isfinite(weights).all())
            or bool((weights<0).any()) or bool((weights>1).any())
            or not bool(torch.allclose(
                weights.sum(-1), torch.ones_like(weights[...,0]),
                atol=1e-6,rtol=0.,
            ))
        ):
            raise ValueError("RSS_VNEXT_EMBEDDED_MIRA_SOURCE_WEIGHT_SIMPLEX_DOMAIN")
        expected={
            "admission_status":_TEACHER,
            "dataset_manifest_sha256":self.dataset_manifest_sha256,
            "source_binding_sha256":batch.source_binding_sha256,
            "source_qualification_key":batch.source_qualification_key,
            "scene_generator_binding_sha256":state.generating_checkpoint_sha256,
            "mesh_teacher_tokens_sha256":source_surface_tensor_sha256(batch.topology_teacher_tokens),
            "mesh_teacher_address_digest":digest,
            "axis_teacher_sequence_sha256":axis_teacher_sequence_sha256_v6(sequence),
            "source_dense_weights_sha256":source_surface_tensor_sha256(weights),
            "source_mesh_face_reindex_proven":True,
            "source_rig_joint_reindex_proven":True,
            "full_source_weight_coverage":True,
            "teacher_semantics":"SOURCE_NATIVE_DENSE_SKIN_NOT_INFERRED_WEIGHTS",
            "codec_feature_abi":"IRIS_LOCAL_H_PLUS_REST_XYZ__AXIS_CAUSAL_PRE_JOINT_H",
            "skin_decoder_mode":"DOT_PRODUCT_BASELINE",
        }
        receipt=self.trusted_registry.get(batch.admission_key)
        if receipt is None or any(receipt.get(k)!=v for k,v in expected.items()):
            raise ValueError("RSS_VNEXT_EMBEDDED_MIRA_SOURCE_ADMISSION_DRIFT")
        features=torch.cat((local,rest),dim=-1)
        codec=model.mira.codec
        if (
            features.shape[-1]!=codec.config.vertex_feature_dim
            or joints_hidden.shape[-1]!=codec.config.joint_feature_dim
        ):
            raise RuntimeError("RSS_VNEXT_EMBEDDED_MIRA_PRODUCT_CODEC_FEATURE_ABI")
        fit=codec(weights,features,joints_hidden)
        # ABI parity must hold for the deployed hard-token decoder. Neither
        # expected-code soft approximation nor a second independently trained
        # MIRA decoder can substitute for exact product hard IDs.
        with torch.no_grad():
            hard=model.mira.decode_discrete_skin_v1(
                local,rest,joints_hidden,fit["token_ids"],
                vertex_normals=normals,
            )
            discrepancy=float((hard-fit["reconstructed_weights"]).abs().max().item())
        if discrepancy>1.e-5:
            raise RuntimeError("RSS_VNEXT_EMBEDDED_MIRA_HARD_RUNTIME_REPLAY_DRIFT")
        pose_deform_loss=None
        hard_pose_discrepancy=None
        pose_count=0
        pose_fields=(
            batch.source_rest_vertices,batch.source_joint_skin_matrices,
            batch.source_posed_vertices,batch.source_pose_vertex_mask,
        )
        pose_requested=any(value is not None for value in pose_fields) or bool(
            batch.source_pose_admission_key
        )
        if pose_requested or self.pose_consequence_weight>0:
            if (
                self.pose_consequence_weight<=0
                or any(value is None for value in pose_fields)
                or not batch.source_pose_admission_key
            ):
                raise ValueError("RSS_VNEXT_EMBEDDED_MIRA_POSE_SUPERVISION_INCOMPLETE")
            rest_gt,mats_gt,posed_gt,valid_gt=pose_fields
            if (
                not all(isinstance(value,Tensor) for value in pose_fields)
                or rest_gt.shape!=(1,v,3)
                or mats_gt.ndim!=5
                or mats_gt.shape[0]!=1
                or mats_gt.shape[2:]!=(n_joints,4,4)
                or mats_gt.shape[1]<1
                or mats_gt.shape[1]>64
                or posed_gt.shape!=(1,mats_gt.shape[1],v,3)
                or valid_gt.shape!=(1,mats_gt.shape[1],v)
                or valid_gt.dtype!=torch.bool
                or not bool(valid_gt.any())
                or any(value.device!=weights.device for value in pose_fields)
                or any(not bool(torch.isfinite(value).all())
                       for value in (rest_gt,mats_gt,posed_gt))
                or not bool(torch.allclose(
                    mats_gt[...,3,:],
                    torch.tensor([0.,0.,0.,1.],device=mats_gt.device,dtype=mats_gt.dtype)
                        .expand_as(mats_gt[...,3,:]),
                    atol=1e-5,rtol=0.,
                ))
            ):
                raise ValueError("RSS_VNEXT_EMBEDDED_MIRA_SOURCE_POSE_DOMAIN")
            pose_row=self.trusted_source_pose_registry.get(
                batch.source_pose_admission_key
            )
            expected_pose={
                "admission_status":"QUALIFIED_NATIVE_POSED_LBS_VERTEX_TEACHER",
                "dataset_manifest_sha256":self.dataset_manifest_sha256,
                "source_qualification_key":batch.source_qualification_key,
                "source_binding_sha256":batch.source_binding_sha256,
                "scene_generator_binding_sha256":state.generating_checkpoint_sha256,
                "mesh_teacher_address_digest":digest,
                "axis_teacher_sequence_sha256":axis_teacher_sequence_sha256_v6(sequence),
                "source_rest_vertices_sha256":source_surface_tensor_sha256(rest_gt),
                "source_joint_skin_matrices_sha256":source_surface_tensor_sha256(mats_gt),
                "source_posed_vertices_sha256":source_surface_tensor_sha256(posed_gt),
                "source_pose_vertex_mask_sha256":source_surface_tensor_sha256(valid_gt),
                "reindexed_tessa_vertices_and_axis_joints":True,
                "pose_witness_source":"INDEPENDENT_SOURCE_NATIVE_ANIMATED_MESH",
                "generated_view_is_motion_teacher":False,
            }
            if pose_row is None or any(
                pose_row.get(key)!=value for key,value in expected_pose.items()
            ):
                raise ValueError("RSS_VNEXT_EMBEDDED_MIRA_POSE_RECEIPT_DRIFT")
            predicted_pose=posed_lbs_from_joint_matrices(
                rest_gt,fit["reconstructed_weights"],mats_gt,
            )
            pose_point_errors=F.smooth_l1_loss(
                predicted_pose,posed_gt,reduction="none",
            ).mean(dim=-1)
            pose_deform_loss=pose_point_errors[valid_gt].mean()
            pose_count=int(valid_gt.sum().item())
            with torch.no_grad():
                hard_posed=posed_lbs_from_joint_matrices(
                    rest_gt,hard,mats_gt,
                )
                hard_pose_discrepancy=float(
                    (hard_posed-predicted_pose.detach()).abs().max().item()
                )
            if hard_pose_discrepancy>1.e-5:
                raise RuntimeError("RSS_VNEXT_EMBEDDED_MIRA_POSE_TOKEN_REPLAY_DRIFT")
        with torch.no_grad():
            error=(fit["reconstructed_weights"].detach()-weights).abs()
            zero=weights<1.e-5
            nonzero=~zero
            zero_error=float(error[zero].mean().item()) if bool(zero.any()) else None
            nonzero_error=float(error[nonzero].mean().item()) if bool(nonzero.any()) else None
        return RSSvNextEmbeddedMIRACodecResultV1(
            loss=fit["loss"] + (
                self.pose_consequence_weight*pose_deform_loss
                if pose_deform_loss is not None else 0.
            ),reconstruction_loss=fit["reconstruction_loss"],
            quantization_loss=fit["quantization_loss"],token_ids=fit["token_ids"],
            mesh_address_digest=digest,
            hard_decoder_replay_max_abs_error=discrepancy,
            source_pose_deformation_loss=pose_deform_loss,
            qualified_pose_vertex_count=pose_count,
            hard_token_pose_replay_max_abs_error=hard_pose_discrepancy,
            sparse_zero_influence_error=zero_error,
            sparse_nonzero_influence_error=nonzero_error,
        )

    def train_step(self, batch: RSSvNextEmbeddedMIRACodecBatchV1):
        self.optimizer.zero_grad(set_to_none=True)
        result=self.objective(batch)
        result.loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.mira.codec.parameters(),1.,error_if_nonfinite=True,
        )
        self.optimizer.step()
        self.step+=1
        return result

    def save_checkpoint(self, path: str | Path) -> None:
        """Resume codec-first source training, not export a ProductRelease."""
        target=Path(path)
        target.parent.mkdir(parents=True,exist_ok=True)
        payload={
            "schema":SCHEMA,
            "dataset_manifest_sha256":self.dataset_manifest_sha256,
            "registry_sha256":self.registry_sha256,
            "pose_registry_sha256":self.pose_registry_sha256,
            "pose_consequence_weight":self.pose_consequence_weight,
            "model_config_sha256":self.config_sha256,
            "step":self.step,
            "codec_state_dict":self.model.mira.codec.state_dict(),
            "optimizer":self.optimizer.state_dict(),
            "torch_cpu_rng":torch.get_rng_state(),
        }
        with tempfile.NamedTemporaryFile(
            dir=target.parent,prefix="miracodec-",suffix=".pt",delete=False,
        ) as stream:
            tmp=Path(stream.name)
        try:
            torch.save(payload,tmp)
            os.replace(tmp,target)
        finally:
            tmp.unlink(missing_ok=True)

    def load_checkpoint(self, path: str | Path) -> None:
        payload=torch.load(Path(path),map_location="cpu",weights_only=True)
        if (
            payload.get("schema")!=SCHEMA
            or payload.get("dataset_manifest_sha256")!=self.dataset_manifest_sha256
            or payload.get("registry_sha256")!=self.registry_sha256
            or payload.get("pose_registry_sha256")!=self.pose_registry_sha256
            or payload.get("pose_consequence_weight")!=self.pose_consequence_weight
            or payload.get("model_config_sha256")!=self.config_sha256
            or type(payload.get("step")) is not int or payload["step"]<0
        ):
            raise ValueError("RSS_VNEXT_EMBEDDED_MIRA_SOURCE_CODEC_RESUME_DRIFT")
        self.model.mira.codec.load_state_dict(payload["codec_state_dict"],strict=True)
        self.optimizer.load_state_dict(payload["optimizer"])
        self.step=payload["step"]
        torch.set_rng_state(payload["torch_cpu_rng"])


__all__ = [
    "RSSvNextEmbeddedMIRACodecBatchV1",
    "RSSvNextEmbeddedMIRACodecResultV1",
    "RSSvNextEmbeddedMIRASourceCodecTrainerV1",
]
