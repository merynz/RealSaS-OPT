from __future__ import annotations

"""Source-verified, stationary-codebook AXIS SkinTokens handoff.

A trained MIRA FSQ vocabulary cannot be taught to AXIS as fixed labels while
its own encoding features (IRIS/TESSA vertex features and AXIS causal JOINT
hidden) keep changing. This stage freezes the entire representation producer
and the product MIRA codec, then learns ONLY the AXIS causal SkinTokens head
while freezing the entire token embedding table.

The stage has no authority for source rig/control reconstruction, editing,
attachment or release. It requires independently trusted source-native weights
reindexed to the exact TESSA face/vertex and AXIS joint ordering.
"""
from dataclasses import dataclass
from hashlib import sha256
from math import isfinite
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping
import json
import os
import tempfile

import torch
from torch import Tensor
import torch.nn.functional as F

from models.axis.rig_program_model_v6 import (
    AXISRigProgramGrammarV6Product, AXISSequenceV6, RigGrammarStateV6,
    RigTokenKindV6,
)
from .contracts import RSSvNextSceneState
from .training_engine import (
    architecture_digest, axis_mesh_teacher_address_digest_v1,
    axis_teacher_sequence_sha256_v6, frozen_mira_codec_state_sha256_v6,
    source_surface_tensor_sha256,
)
from .training_optimizer_ownership_v1 import (
    qualify_rss_vnext_optimizer_ownership_v1,
)
from .unified_character_model import RSSvNextUnifiedCharacterModel

SCHEMA = "RealSaS.RSSvNext.StationaryAXISFSQTransfer.v1"
_ADMISSION = "QUALIFIED_NATIVE_SOURCE_FROZEN_AXIS_MIRA_SKIN_CODE_HANDOFF"
# Important: axis.skin_embedding[0] is present in *all* BOS/JOINT
# prefixes because the model embeds -1 (non-skin) as index 0. Updating
# even the null embedding row changes AXIS pre-JOINT hidden, then changes
# the FSQ labels we are trying to hold stationary. Therefore do NOT train
# any skin_embedding row here. Train only the classifier head.
_STAGE_TRAINABLE = ("axis.skin_code_head.",)


def frozen_rss_vnext_axis_token_upstream_state_sha256_v1(
    model: RSSvNextUnifiedCharacterModel,
) -> str:
    """One-time exact upstream weight+buffer identity, excludes trained head.

    Check at stage admission, checkpoint save and resume. Expensive for large
    A100 models by design, but never rehashes the full model every batch.
    """
    state=model.state_dict()
    fields = {
        name: source_surface_tensor_sha256(tensor)
        for name,tensor in state.items()
        if not name.startswith(_STAGE_TRAINABLE)
    }
    if not fields:
        raise RuntimeError("RSS_VNEXT_STATIONARY_TOKEN_NO_FROZEN_SOURCE_STATE")
    return _json_sha(fields)


def _sha(v: object) -> bool:
    return type(v) is str and len(v) == 64 and all(
        c in "0123456789abcdef" for c in v
    )


def _json_sha(v: object) -> str:
    return sha256(json.dumps(
        v, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


@dataclass(frozen=True)
class RSSvNextStationarySkinTokenBatchV1:
    scene: RSSvNextSceneState
    structural_memory: Tensor
    topology_teacher_tokens: Tensor
    axis_teacher_sequence: AXISSequenceV6
    source_dense_weights: Tensor
    source_binding_sha256: str
    source_qualification_key: str
    admission_key: str


@dataclass(frozen=True)
class RSSvNextStationarySkinTokenResultV1:
    skin_code_ce: Tensor
    target_code_accuracy: float
    target_code_count: int
    independent_source_codec_weight_mae: float
    hard_codec_runtime_replay_max_error: float
    mesh_address_digest: str
    source_qualified_for_token_supervision: bool = True
    trained_skin_program_release_qualified: bool = False


class RSSvNextStationaryAXISSkinTokenTrainerV1:
    def __init__(
        self,
        model: RSSvNextUnifiedCharacterModel,
        optimizer: torch.optim.Optimizer,
        *,
        dataset_manifest_sha256: str,
        upstream_frozen_checkpoint_sha256: str,
        trusted_source_bridge_registry: Mapping[str, Mapping[str, Any]],
    ):
        if (
            type(model) is not RSSvNextUnifiedCharacterModel
            or not isinstance(optimizer, torch.optim.Optimizer)
            or not _sha(dataset_manifest_sha256)
            or not _sha(upstream_frozen_checkpoint_sha256)
            or model.cfg.skin_decoder_mode != "DOT_PRODUCT_BASELINE"
            or not trusted_source_bridge_registry
            or model.training
            or model.axis.training
            or model.mira.codec.training
        ):
            raise ValueError("RSS_VNEXT_STATIONARY_TOKEN_STAGE_POLICY_INVALID")
        self.model = model
        self.optimizer = optimizer
        self.dataset_manifest_sha256 = dataset_manifest_sha256
        self.upstream_frozen_checkpoint_sha256 = upstream_frozen_checkpoint_sha256
        registry = {
            str(key): dict(value)
            for key, value in trusted_source_bridge_registry.items()
        }
        self.registry = MappingProxyType({
            key: MappingProxyType(value) for key, value in registry.items()
        })
        self.registry_sha256 = _json_sha(registry)
        self.config_sha256 = architecture_digest(model.cfg)
        self.codec_sha256 = frozen_mira_codec_state_sha256_v6(model)
        self.frozen_state_sha256 = (
            frozen_rss_vnext_axis_token_upstream_state_sha256_v1(model)
        )
        self._frozen_parameter_versions = tuple(
            (name,param._version)
            for name,param in model.named_parameters()
            if not name.startswith(_STAGE_TRAINABLE)
        )
        self.optimizer_ownership = self._assert_scope()
        self.step = 0

    def _assert_scope(self):
        trainable = tuple(
            name for name, param in self.model.named_parameters()
            if param.requires_grad
        )
        permitted = tuple(
            name for name, _ in self.model.named_parameters()
            if name.startswith(_STAGE_TRAINABLE)
        )
        if (
            not permitted or set(trainable) != set(permitted)
            or self.model.training or self.model.axis.training
            or self.model.mira.codec.training
        ):
            raise ValueError("RSS_VNEXT_STATIONARY_TOKEN_UPSTREAM_MUST_BE_FROZEN")
        # PyTorch parameter version counters catch in-place mutations
        # of frozen upstream modules during this stage without rehashing
        # multi-GB tensors on EVERY optimizer step.
        versions = tuple(
            (name,param._version)
            for name,param in self.model.named_parameters()
            if not name.startswith(_STAGE_TRAINABLE)
        )
        if versions != self._frozen_parameter_versions:
            raise RuntimeError("RSS_VNEXT_STATIONARY_TOKEN_UPSTREAM_PARAMETER_MUTATED")
        owner = qualify_rss_vnext_optimizer_ownership_v1(
            self.model, self.optimizer,
        )
        if hasattr(self, "optimizer_ownership") and (
            owner.binding_sha256 != self.optimizer_ownership.binding_sha256
        ):
            raise RuntimeError("RSS_VNEXT_STATIONARY_TOKEN_OPTIMIZER_PHASE_DRIFT")
        if frozen_mira_codec_state_sha256_v6(self.model) != self.codec_sha256:
            raise RuntimeError("RSS_VNEXT_STATIONARY_TOKEN_FSQ_CODEC_STATE_DRIFT")
        return owner

    @staticmethod
    def _code_positions(teacher: AXISSequenceV6, tokens_per_joint: int) -> tuple[Tensor, Tensor]:
        if teacher.kind_ids.ndim != 2 or teacher.kind_ids.shape[0] != 1:
            raise ValueError("RSS_VNEXT_STATIONARY_TOKEN_BATCH_ONE_REQUIRED")
        kind = teacher.kind_ids[0]
        if not bool(teacher.attention_mask.all()):
            raise ValueError("RSS_VNEXT_STATIONARY_TOKEN_UNPADDED_SEQUENCE_REQUIRED")
        state = RigGrammarStateV6()
        for raw in kind.tolist():
            event = RigTokenKindV6(raw)
            # A source rig/skin program with control actions needs a separate
            # independently qualified intervention; this stage is skin-only.
            if event in (
                RigTokenKindV6.CONTROL, RigTokenKindV6.CONTROL_TARGET,
                RigTokenKindV6.CHAIN, RigTokenKindV6.CONSTRAINT,
                RigTokenKindV6.CONSTRAINT_INPUT,
                RigTokenKindV6.CONSTRAINT_OUTPUT, RigTokenKindV6.ANCHOR,
            ):
                raise ValueError("RSS_VNEXT_STATIONARY_TOKEN_CONTROL_TEACHER_FORBIDDEN")
            state = AXISRigProgramGrammarV6Product.advance(
                state, event, tokens_per_joint=tokens_per_joint,
            )
        if state.section.name != "DONE":
            raise ValueError("RSS_VNEXT_STATIONARY_TOKEN_TEACHER_NOT_TERMINATED")
        joints = torch.where(kind == int(RigTokenKindV6.JOINT))[0]
        skins = torch.where(kind == int(RigTokenKindV6.SKIN_CODE))[0]
        if (
            len(joints)<2 or bool((joints==0).any())
            or len(skins)!=len(joints)*tokens_per_joint
            or int(skins.min())<=int(joints.max())
        ):
            raise ValueError("RSS_VNEXT_STATIONARY_TOKEN_CAUSAL_JOINT_DOMAIN")
        return joints, skins

    def objective(self, batch: RSSvNextStationarySkinTokenBatchV1) -> RSSvNextStationarySkinTokenResultV1:
        self._assert_scope()
        if (
            type(batch) is not RSSvNextStationarySkinTokenBatchV1
            or not _sha(batch.source_binding_sha256)
            or not batch.source_qualification_key
            or not batch.admission_key
            or type(batch.scene) is not RSSvNextSceneState
            or type(batch.axis_teacher_sequence) is not AXISSequenceV6
            or not isinstance(batch.topology_teacher_tokens, Tensor)
            or not isinstance(batch.structural_memory, Tensor)
            or not isinstance(batch.source_dense_weights, Tensor)
        ):
            raise ValueError("RSS_VNEXT_STATIONARY_TOKEN_BATCH_INVALID")
        state=batch.scene
        teacher=batch.axis_teacher_sequence
        teacher.validate(self.model.axis.config)
        joints, skins = self._code_positions(
            teacher, self.model.cfg.skin_tokens_per_joint,
        )
        if (
            state.scene_conditioning_authority != "GENERATED_SIX_VIEW_SHARED_SCENE"
            or state.source_camera_authority != "UNKNOWN_NO_ORIGINAL_CAMERA_PROJECTION"
            or state.camera_features is not None
            or state.generated_view_count != 6
            or not _sha(state.generating_checkpoint_sha256)
            or batch.topology_teacher_tokens.ndim != 2
            or batch.topology_teacher_tokens.shape[0] != 1
            or batch.topology_teacher_tokens.dtype != torch.long
            or batch.structural_memory.shape[0] != 1
            or batch.structural_memory.ndim != 3
            or batch.structural_memory.shape[-1] != self.model.cfg.tessa.d_model
            or batch.topology_teacher_tokens.device != batch.structural_memory.device
            or teacher.kind_ids.device != batch.structural_memory.device
        ):
            raise ValueError("RSS_VNEXT_STATIONARY_TOKEN_SIX_VIEW_PRODUCT_ABI")
        pre=self.registry.get(batch.admission_key)
        seq_digest=axis_teacher_sequence_sha256_v6(teacher)
        if (
            pre is None
            or pre.get("admission_status") != _ADMISSION
            or pre.get("dataset_manifest_sha256") != self.dataset_manifest_sha256
            or pre.get("source_qualification_key") != batch.source_qualification_key
            or pre.get("source_binding_sha256") != batch.source_binding_sha256
            or pre.get("upstream_frozen_checkpoint_sha256") != self.upstream_frozen_checkpoint_sha256
            or pre.get("upstream_frozen_model_state_sha256") != self.frozen_state_sha256
            or pre.get("mira_codec_state_sha256") != self.codec_sha256
            or pre.get("scene_generator_binding_sha256") != state.generating_checkpoint_sha256
            or pre.get("mesh_teacher_tokens_sha256") != source_surface_tensor_sha256(batch.topology_teacher_tokens)
            or pre.get("axis_teacher_sequence_sha256") != seq_digest
            or pre.get("source_dense_weights_sha256") != source_surface_tensor_sha256(batch.source_dense_weights)
            or pre.get("structural_memory_sha256") != source_surface_tensor_sha256(batch.structural_memory)
            or pre.get("source_mesh_face_reindex_proven") is not True
            or pre.get("source_rig_joint_reindex_proven") is not True
            or pre.get("source_codec_hard_reconstruction_qualified") is not True
            or pre.get("generated_view_is_skin_teacher") is not False
            or pre.get("teacher_semantics") != "EXACT_FROZEN_FSQ_SOURCE_SKIN_TOKENS"
        ):
            raise ValueError("RSS_VNEXT_STATIONARY_TOKEN_SOURCE_RECEIPT_DRIFT")

        # All conditions are stationary. Teacher labels are verified BEFORE
        # any CE and are never computed from student predictions during
        # a train step. No source-weight gradients or teacher side inputs enter
        # the product neural inference path.
        with torch.no_grad():
            mesh = self.model.build_mesh_candidate(
                state,batch.topology_teacher_tokens[0],
                source_authority="TEACHER_FORCED_TRAINING_ONLY",
            )
            digest=axis_mesh_teacher_address_digest_v1(mesh)
            if pre.get("mesh_teacher_address_digest") != digest:
                raise ValueError("RSS_VNEXT_STATIONARY_TOKEN_MESH_ADDRESS_DRIFT")
            memory=self.model._control_memory(
                state,batch.structural_memory,mesh,
            ).detach()
            hidden=self.model.axis(memory,teacher)["hidden"]
            joint_hidden=hidden[:,joints-1].detach()
            vertex_features=torch.cat(
                (mesh.local_features.detach(),mesh.rest_vertices_iris.detach()),
                dim=-1,
            )
            source=batch.source_dense_weights.detach()
            count=mesh.rest_vertices_iris.shape[1]
            if (
                source.shape != (1,count,len(joints))
                or not torch.is_floating_point(source)
                or source.device != vertex_features.device
                or not bool(torch.isfinite(source).all())
                or bool((source<0).any()) or bool((source>1).any())
                or not bool(torch.allclose(
                    source.sum(-1),torch.ones_like(source[...,0]),
                    atol=1.e-6,rtol=0.,
                ))
            ):
                raise ValueError("RSS_VNEXT_STATIONARY_TOKEN_SOURCE_WEIGHTS_DOMAIN")
            ids, _, _ = self.model.mira.codec.encode_tokens(
                source, vertex_features, joint_hidden,
            )
            expected = teacher.skin_code_ids[0,skins].reshape(
                1,len(joints),self.model.cfg.skin_tokens_per_joint,
            )
            if not torch.equal(ids,expected):
                raise ValueError("RSS_VNEXT_STATIONARY_TOKEN_LABELS_NOT_FROZEN_CODEC_CODES")
            hard=self.model.mira.decode_discrete_skin_v1(
                mesh.local_features.detach(),
                mesh.rest_vertices_iris.detach(),
                joint_hidden, ids,
                vertex_normals=mesh.normals.detach(),
            )
            hard_direct=self.model.mira.codec.decode_tokens(
                ids,vertex_features,joint_hidden,
            )
            max_error=float((hard-hard_direct).abs().max().item())
            if max_error>1e-5:
                raise RuntimeError("RSS_VNEXT_STATIONARY_TOKEN_PRODUCT_DECODER_DRIFT")
            mae=float((hard-source).abs().mean().item())
            threshold=pre.get("max_admitted_source_codec_weight_mae")
            if (
                type(threshold) not in (float,int) or not isfinite(threshold)
                or threshold < 0 or threshold > 1 or mae > threshold
            ):
                raise ValueError("RSS_VNEXT_STATIONARY_TOKEN_CODEC_QUALITY_NOT_ADMITTED")
        # Only AXIS's SkinTokens classifier/embedding can receive gradient.
        # Source weights and source 3D geometry are not classifier inputs.
        logits=self.model.axis(memory,teacher)["skin_code_logits"][:,:-1]
        positions=(teacher.kind_ids[:,1:] == int(RigTokenKindV6.SKIN_CODE))
        labels=teacher.skin_code_ids[:,1:][positions]
        if labels.numel()!=len(skins):
            raise RuntimeError("RSS_VNEXT_STATIONARY_TOKEN_TARGET_ALIGNMENT_DRIFT")
        selected=logits[positions]
        ce=F.cross_entropy(selected,labels)
        accuracy=float((selected.argmax(-1)==labels).float().mean().detach().item())
        if not bool(torch.isfinite(ce)):
            raise RuntimeError("RSS_VNEXT_STATIONARY_TOKEN_NONFINITE_CE")
        return RSSvNextStationarySkinTokenResultV1(
            skin_code_ce=ce,target_code_accuracy=accuracy,
            target_code_count=int(labels.numel()),
            independent_source_codec_weight_mae=mae,
            hard_codec_runtime_replay_max_error=max_error,
            mesh_address_digest=digest,
        )

    def train_step(self, batch: RSSvNextStationarySkinTokenBatchV1) -> RSSvNextStationarySkinTokenResultV1:
        self._assert_scope()
        self.optimizer.zero_grad(set_to_none=True)
        result=self.objective(batch)
        result.skin_code_ce.backward()
        torch.nn.utils.clip_grad_norm_(
            (p for p in self.model.parameters() if p.requires_grad),
            1.,error_if_nonfinite=True,
        )
        self.optimizer.step()
        self.step+=1
        return result

    def _assert_frozen_state_identity(self) -> None:
        if (
            frozen_rss_vnext_axis_token_upstream_state_sha256_v1(self.model)
            != self.frozen_state_sha256
        ):
            raise RuntimeError("RSS_VNEXT_STATIONARY_TOKEN_UPSTREAM_STATE_DRIFT")

    def save_checkpoint(self, path: str | Path) -> None:
        self._assert_scope()
        self._assert_frozen_state_identity()
        target=Path(path)
        target.parent.mkdir(parents=True,exist_ok=True)
        payload={
            "schema":SCHEMA,
            "dataset_manifest_sha256":self.dataset_manifest_sha256,
            "registry_sha256":self.registry_sha256,
            "architecture_sha256":self.config_sha256,
            "frozen_upstream_checkpoint_sha256":self.upstream_frozen_checkpoint_sha256,
            "frozen_upstream_actual_model_state_sha256":self.frozen_state_sha256,
            "frozen_codec_state_sha256":self.codec_sha256,
            "optimizer_ownership_sha256":self.optimizer_ownership.binding_sha256,
            "step":self.step,
            "skin_code_head":self.model.axis.skin_code_head.state_dict(),
            "optimizer":self.optimizer.state_dict(),
            "torch_cpu_rng":torch.get_rng_state(),
        }
        with tempfile.NamedTemporaryFile(
            dir=target.parent,prefix="axis-skin-",suffix=".pt",delete=False,
        ) as stream:
            tmp=Path(stream.name)
        try:
            torch.save(payload,tmp)
            os.replace(tmp,target)
        finally:
            tmp.unlink(missing_ok=True)

    def load_checkpoint(self, path: str | Path) -> None:
        self._assert_scope()
        self._assert_frozen_state_identity()
        p=torch.load(Path(path),map_location="cpu",weights_only=True)
        if (
            p.get("schema")!=SCHEMA
            or p.get("dataset_manifest_sha256")!=self.dataset_manifest_sha256
            or p.get("registry_sha256")!=self.registry_sha256
            or p.get("architecture_sha256")!=self.config_sha256
            or p.get("frozen_upstream_checkpoint_sha256")!=self.upstream_frozen_checkpoint_sha256
            or p.get("frozen_upstream_actual_model_state_sha256")!=self.frozen_state_sha256
            or p.get("frozen_codec_state_sha256")!=self.codec_sha256
            or p.get("optimizer_ownership_sha256")!=self.optimizer_ownership.binding_sha256
            or type(p.get("step")) is not int or p["step"]<0
        ):
            raise ValueError("RSS_VNEXT_STATIONARY_TOKEN_RESUME_POLICY_DRIFT")
        self.model.axis.skin_code_head.load_state_dict(p["skin_code_head"],strict=True)
        self.optimizer.load_state_dict(p["optimizer"])
        self.step=p["step"]
        torch.set_rng_state(p["torch_cpu_rng"])


__all__=[
    "SCHEMA","RSSvNextStationarySkinTokenBatchV1",
    "RSSvNextStationarySkinTokenResultV1",
    "RSSvNextStationaryAXISSkinTokenTrainerV1",
    "frozen_rss_vnext_axis_token_upstream_state_sha256_v1",
]
