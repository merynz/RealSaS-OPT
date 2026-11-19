from __future__ import annotations

"""Qualified AXIS+MIRA consequences on the EXACT product seeded M*.

The student sees one H(x), six generated conditioning hypotheses, and the
same TESSA indexed mesh as inference; native rig, skin and poses are LOSS ONLY.
The MIRA codec is source-trained/frozen beforehand, not silently trained from
student-induced token identities. Actual G3/G3B and clips are Compiler gates,
not properties conferred by a differentiable deformation loss.
"""

from dataclasses import asdict, dataclass
import math
from typing import Mapping

import torch
from torch import Tensor
import torch.nn.functional as F

from models.axis.rig_program_model_v6 import (
    AXISSequenceV6, AXIS_V6_PRODUCT_ARCHITECTURE_ID,
    qualified_axis_rig_skin_sequence_loss_v6, RigTokenKindV6,
)
from compiler.realsas_compiler_core.rig_constraint_schema_v1 import (
    RIG_CONSTRAINT_OPERATOR_IDS_V1,
)
from .training_seeded_tessa_v1 import SeededTESSASourceTeacherV1
from .seeded_mstar_teacher_address_v1 import (
    seeded_mstar_teacher_address_sha256_v1,
)
from .native_pose_transport_v1 import (
    validate_source_native_pose_candidate_teacher_v1,
)
from .seeded_pose_consequence_objective_v1 import (
    exact_m_qualified_pose_strain_v1,
)
from .animation_ready_mesh import posed_lbs_from_joint_matrices
from .qualified_chain_skin_v1 import qualified_source_chain_skin_loss_v1
from .native_skin_transport_v1 import SourceNativeCandidateSkinTransportV1
from .axis_mechanical_chain_partition_v1 import assert_qualified_chain_partition_matches_axis_v1
from .hierarchical_skin_field_v1 import axis_joint_group_ids_v1


@dataclass(frozen=True)
class ProductSeededAXISMIRAWeightsV1:
    axis_program: float = 0.
    source_skin: float = 0.
    posed_surface: float = 0.
    dynamic_surface: float = 0.
    hard_skin: float = 0.
    chain_skin: float = 0.  # source-qualified joint->chain partition; no named ontology

    def validate(self) -> None:
        if any(type(v) not in (float,int) or not math.isfinite(v) or v<0
               for v in asdict(self).values()):
            raise ValueError("RSS_VNEXT_PRODUCT_SEEDED_AXIS_MIRA_WEIGHT_POLICY")
        if (
            (self.source_skin>0 or self.posed_surface>0
             or self.dynamic_surface>0 or self.hard_skin>0 or self.chain_skin>0)
            and self.axis_program<=0
        ) or (self.dynamic_surface>0 and self.posed_surface<=0):
            raise ValueError("RSS_VNEXT_PRODUCT_SEEDED_AXIS_MIRA_PHASE_ORDER")

    @property
    def enabled(self) -> bool:
        return any(v>0 for v in asdict(self).values())


@dataclass(frozen=True)
class ProductSeededAXISMIRAResultV1:
    total: Tensor | None
    axis: Tensor | None
    skin: Tensor | None
    posed: Tensor | None
    dynamic: Tensor | None
    hard_skin: Tensor | None
    eligible_pose_faces: int
    chain_skin: Tensor | None = None
    eligible_chain_vertices: int = 0
    authority: str = "SOURCE_NATIVE_RIG_POSE_LOSS_ONLY_NOT_MODEL_RELEASE"


def product_seeded_axis_mira_objective_v1(
    *,model,state,mesh,teacher: SeededTESSASourceTeacherV1,
    structural_memory: Tensor,
    batch,
    weights: ProductSeededAXISMIRAWeightsV1,
    dataset_manifest_sha256: str,
    trusted_axis_teacher_registry: Mapping[str,Mapping[str,object]],
    trusted_native_pose_registry: Mapping[str,Mapping[str,object]],
    trusted_native_skin_registry: Mapping[str,Mapping[str,object]],
    trusted_chain_teacher_registry: Mapping[str,Mapping[str,object]],
    tensor_digest,
    axis_sequence_digest,
    frozen_codec_digest,
) -> ProductSeededAXISMIRAResultV1:
    weights.validate()
    chain_fields=(
        batch.axis_chain_joint_ids,batch.axis_chain_vertex_mask,
        batch.axis_chain_teacher_admission_key,
    )
    if weights.chain_skin<=0 and any(x is not None and x!='' if isinstance(x,str) else x is not None for x in chain_fields):
        raise ValueError("RSS_VNEXT_CHAIN_TEACHER_DISABLED_BUT_PRESENT")
    if not weights.enabled:
        if batch.axis_teacher_sequence is not None or batch.axis_teacher_admission_key:
            raise ValueError("RSS_VNEXT_SEEDED_AXIS_UNCONSUMED_SOURCE_TRUTH")
        return ProductSeededAXISMIRAResultV1(None,None,None,None,None,None,0)
    seq=batch.axis_teacher_sequence
    key=batch.axis_teacher_admission_key
    if (
        not isinstance(seq,AXISSequenceV6) or not key
        or state.scene_conditioning_authority!="GENERATED_SIX_VIEW_SHARED_SCENE"
        or mesh.seeded_surface is None
        or mesh.seeded_surface.original_vertex_count<3
    ):
        raise ValueError("RSS_VNEXT_PRODUCT_SEEDED_AXIS_SOURCE_REQUIRED")
    row=trusted_axis_teacher_registry.get(key)
    expected={
        "admission_status":"QUALIFIED_SOURCE_RIG_SKIN_TOKEN_TEACHER",
        "dataset_manifest_sha256":dataset_manifest_sha256,
        "source_qualification_key":batch.source_qualification_key,
        "architecture_id":AXIS_V6_PRODUCT_ARCHITECTURE_ID,
        "teacher_sequence_sha256":axis_sequence_digest(seq),
        "mira_codec_state_sha256":frozen_codec_digest(model),
        "source_asset_sha256":teacher.source_asset_sha256,
        "observed_rgba_sha256":tensor_digest(batch.rgba),
        "observed_camera_sha256":tensor_digest(batch.camera_features),
        "source_binding_sha256":batch.product_source_binding_sha256,
        "combined_generation_checkpoint_sha256":
            batch.product_generation_checkpoint_sha256,
        "source_model_observation_count":2,
        "source_teacher_is_loss_only":True,
        "mesh_context":"SAME_SEEDED_TESSA_MSTAR_NOT_LEGACY_TOKEN_MESH",
        "teacher_semantics":"SOURCE_RIG_AND_PINNED_FSQ_NOT_INFERRED_CONTROLS",
    }
    if not isinstance(row,Mapping) or any(row.get(k)!=v for k,v in expected.items()):
        raise ValueError("RSS_VNEXT_PRODUCT_SEEDED_AXIS_RECEIPT_DRIFT")
    if any(p.requires_grad for p in model.mira.codec.parameters()):
        raise ValueError("RSS_VNEXT_PRODUCT_SEEDED_MIRA_CODEC_NOT_FROZEN")
    intervention=(
        row.get("intervention_qualification_status")
            =="QUALIFIED_SOURCE_CONTROL_CONSEQUENCE"
        and isinstance(row.get("source_intervention_sha256"),str)
        and len(row["source_intervention_sha256"])==64
    )
    # Mechanical part grouping is a stronger, independently sourced label
    # than either source-native CHAIN paths or inferred skeleton hierarchy.
    # A bare "WING_L" bone name or missing part graph is NEVER a teacher.
    part_status=row.get("mechanical_part_graph_status")
    part_qualified=(
        part_status=="QUALIFIED_SOURCE_NATIVE_MULTI_CHAIN_PART_PROGRAM"
        and row.get("mechanical_part_teacher_sequence_sha256")
            ==axis_sequence_digest(seq)
        and row.get("mechanical_part_not_bone_name_heuristic") is True
        and isinstance(row.get("mechanical_part_source_evidence_sha256"),str)
        and len(row["mechanical_part_source_evidence_sha256"])==64
        and weights.chain_skin>0
        and model.cfg.axis_mechanical_part_policy
            == "EMIT_SOURCE_QUALIFIED_PART_GRAPH_V1"
    )
    if part_status is not None and not part_qualified:
        raise ValueError("RSS_VNEXT_PART_GRAPH_SOURCE_RECEIPT_UNQUALIFIED")
    # Never recompute a weaker H here: TESSA and AXIS must see the same
    # constructed source + native FlexiCubes witness memory as inference.
    if (
        not isinstance(structural_memory,Tensor)
        or structural_memory.ndim!=3 or structural_memory.shape[0]!=1
        or structural_memory.shape[-1]!=model.cfg.tessa.d_model
        or structural_memory.device!=mesh.rest_vertices_iris.device
        or not bool(torch.isfinite(structural_memory).all())
    ):
        raise ValueError("RSS_VNEXT_SEEDED_AXIS_STRUCTURAL_MEMORY_INVALID")
    memory=model._control_memory(state,structural_memory,mesh)
    axis_loss,axis_parts=qualified_axis_rig_skin_sequence_loss_v6(
        model.axis,memory,seq,
        tokens_per_joint=model.cfg.skin_tokens_per_joint,
        executable_constraint_operators=RIG_CONSTRAINT_OPERATOR_IDS_V1,
        control_intervention_qualified=intervention,
        # Non-control CHAIN events are teachable from independently admitted
        # source joint-parent and skin partitions, not from control motion.
        # Exact registry, M* and causal chain replay are checked below before
        # this objective may return to the optimizer.
        structural_chains_qualified=weights.chain_skin>0,
        mechanical_part_graph_qualified=part_qualified,
        # Native rig/skin alone never proves the *absence* of authored
        # controls, constraints, or auxiliary chains.
        control_absence_qualified=(
            row.get("source_authored_control_graph_status")
                == "VERIFIED_NO_AUTHORED_CONTROLS"
        ),
        chain_absence_qualified=(
            row.get("source_auxiliary_chain_status")
                == "VERIFIED_NO_AUXILIARY_CHAINS"
        ) and weights.chain_skin<=0,
        # Bind-local frame matrices do not imply observed physical DOF/limits.
        frame_dof_qualified=(
            row.get("source_joint_dof_qualification_status")
                == "QUALIFIED_SOURCE_JOINT_DOF_LIMITS"
        ),
    )
    total=weights.axis_program*axis_loss
    skin_loss=pose_loss=dynamic_loss=hard_loss=chain_loss=None
    pose_count=0
    chain_count=0
    needs_skin=any(v>0 for v in (
        weights.source_skin,weights.posed_surface,
        weights.dynamic_surface,weights.hard_skin,weights.chain_skin,
    ))
    if needs_skin:
        # M* can move continuously under H(x); its discrete face, component,
        # layer and original-parent identity MUST be exactly teacher-pinned.
        active_mstar_topology_address_sha256=seeded_mstar_teacher_address_sha256_v1(
            mesh.seeded_surface,
        )
        if not bool((seq.kind_ids==int(RigTokenKindV6.SKIN_CODE)).any()):
            raise ValueError("RSS_VNEXT_PRODUCT_SEEDED_AXIS_SKIN_TOKEN_REQUIRED")
        if teacher.native_face_layer_samples is None:
            raise ValueError("RSS_VNEXT_PRODUCT_SEEDED_AXIS_SOURCE_FACE_REQUIRED")
        requires_pose=weights.posed_surface>0 or weights.dynamic_surface>0
        pose_support=(
            teacher.native_pose_transport is not None
            and teacher.native_dynamic_importance is not None
            and teacher.native_dynamic_importance.source_absolute_g3_within_limits
            and all(v is not None for v in (
                teacher.source_candidate_skin,teacher.source_candidate_posed,
                teacher.source_pose_matrices,teacher.source_pose_mask,
            ))
        )
        if requires_pose and not pose_support:
            raise ValueError("RSS_VNEXT_PRODUCT_SEEDED_AXIS_NATIVE_POSE_REQUIRED")
        if teacher.native_skin_transport is not None and teacher.native_pose_transport is not None:
            raise ValueError("RSS_VNEXT_PRODUCT_SEEDED_AXIS_COMPETING_NATIVE_SKIN_TEACHERS")
        if teacher.native_skin_transport is not None:
            if not isinstance(teacher.native_skin_transport,SourceNativeCandidateSkinTransportV1):
                raise ValueError("RSS_VNEXT_PRODUCT_SEEDED_AXIS_NATIVE_SKIN_TYPE_INVALID")
            if requires_pose:
                raise ValueError("RSS_VNEXT_PRODUCT_SEEDED_AXIS_SKIN_CANNOT_SUBSTITUTE_FOR_POSE")
            # This independent exact source-native face->candidate M* transport
            # needs no clip or fabricated rest-motion matrices. The qualifier
            # authenticates rights, native vertex skin and barycentric support.
            source_skin=teacher.native_skin_transport.replay(
                witness=teacher.native_face_layer_samples,
                candidate_faces=mesh.faces,
                candidate_rest=mesh.rest_vertices_iris,
                trusted_source_registry=trusted_native_skin_registry,
                dataset_manifest_sha256=dataset_manifest_sha256,
                expected_axis_teacher_sequence_sha256=axis_sequence_digest(seq),
                active_mstar_topology_address_sha256=active_mstar_topology_address_sha256,
            )
        elif pose_support:
            # Pose-qualified legacy path remains unchanged for source-animated
            # assets and is mandatory for posed/dynamic deformation objectives.
            source_receipt=trusted_native_pose_registry.get(
                teacher.source_qualification_key,
            )
            if not isinstance(source_receipt,Mapping):
                raise ValueError("RSS_VNEXT_PRODUCT_SEEDED_AXIS_NATIVE_POSE_RECEIPT")
            validate_source_native_pose_candidate_teacher_v1(
                support=teacher.native_pose_transport,
                native_face_witness=teacher.native_face_layer_samples,
                candidate_faces=mesh.faces,
                candidate_rest=mesh.rest_vertices_iris,
                active_mstar_topology_address_sha256=active_mstar_topology_address_sha256,
                source_candidate_skin=teacher.source_candidate_skin,
                source_candidate_posed=teacher.source_candidate_posed,
                source_pose_mask=teacher.source_pose_mask,
                source_pose_matrices=teacher.source_pose_matrices,
                teacher_source_asset_sha256=teacher.source_asset_sha256,
                teacher_receipt=source_receipt,
                trusted_source_registry=trusted_native_pose_registry,
            )
            source_skin=teacher.source_candidate_skin
        else:
            raise ValueError("RSS_VNEXT_PRODUCT_SEEDED_AXIS_NATIVE_SKIN_SUPPORT_REQUIRED")
        controls=model.axis(memory,seq)
        skin,valid,joint_state=model.mira(
            mesh.mechanics_local_features,mesh.rest_vertices_iris,
            controls["hidden"],controls["skin_code_logits"],seq,
            vertex_normals=mesh.normals,
        )
        if (
            skin.shape!=source_skin.shape
            or valid.shape[0]!=1
            or int(valid[0].sum())!=skin.shape[-1]
        ):
            raise ValueError("RSS_VNEXT_PRODUCT_SEEDED_MIRA_SKIN_ADDRESS_DRIFT")
        if weights.chain_skin>0:
            cids=batch.axis_chain_joint_ids
            vmask=batch.axis_chain_vertex_mask
            receipt=trusted_chain_teacher_registry.get(
                batch.axis_chain_teacher_admission_key,
            )
            if (
                not isinstance(cids,Tensor) or not isinstance(vmask,Tensor)
                or not isinstance(receipt,Mapping)
                or any(receipt.get(k)!=v for k,v in {
                    "admission_status":"QUALIFIED_SOURCE_NATIVE_JOINT_CHAIN_OWNERSHIP",
                    "dataset_manifest_sha256":dataset_manifest_sha256,
                    "source_qualification_key":batch.source_qualification_key,
                    "source_asset_sha256":teacher.source_asset_sha256,
                    "axis_teacher_sequence_sha256":axis_sequence_digest(seq),
                    "source_candidate_skin_sha256":tensor_digest(source_skin),
                    "joint_chain_ids_sha256":tensor_digest(cids),
                    "qualified_vertex_mask_sha256":tensor_digest(vmask),
                    "chain_supervision":"SOURCE_NATIVE_PARTITION_NOT_NAMED_BONE_HEURISTIC",
                }.items())
            ):
                raise ValueError("RSS_VNEXT_CHAIN_TEACHER_UNQUALIFIED_OR_DRIFT")
            # Native source ownership must agree with the *same* causal
            # AXIS chain path program the student learns. Source bone names,
            # part segments or teacher-only labels cannot substitute for a
            # valid inference-time CHAIN event graph.
            assert_qualified_chain_partition_matches_axis_v1(seq,cids)
            chain=qualified_source_chain_skin_loss_v1(
                predicted_skin=skin,source_skin=source_skin,
                joint_chain_ids=cids,qualified_vertex_mask=vmask,
            )
            chain_loss=chain.total
            chain_count=chain.eligible_vertex_count
            total=total+weights.chain_skin*chain_loss
        if weights.source_skin>0:
            skin_loss=F.smooth_l1_loss(skin,source_skin)
            total=total+weights.source_skin*skin_loss
        if weights.posed_surface>0:
            posed=posed_lbs_from_joint_matrices(
                mesh.rest_vertices_iris,skin,teacher.source_pose_matrices,
            )
            source_pose=teacher.source_candidate_posed
            mask=teacher.source_pose_mask
            position=F.smooth_l1_loss(
                posed[mask],source_pose[mask],
            )
            strain=exact_m_qualified_pose_strain_v1(
                rest_xyz=mesh.rest_vertices_iris,
                faces=mesh.faces,
                predicted_poses=posed,
                source_native_posed=source_pose,
                source_pose_mask=mask,
                source_candidate_skin=source_skin,
                scale=mesh.rest_vertices_iris.detach().new_tensor(1.),
            )
            pose_loss=position+strain["total"]
            pose_count=int(strain["qualified_face_pose_count"].item())
            total=total+weights.posed_surface*pose_loss
        if weights.dynamic_surface>0:
            dynamic=model.dynamic_refine(mesh,skin,joint_state)
            posed=posed_lbs_from_joint_matrices(
                dynamic.vertices,dynamic.weights,
                teacher.source_pose_matrices,
            )
            strain=exact_m_qualified_pose_strain_v1(
                rest_xyz=dynamic.vertices,
                faces=mesh.faces,predicted_poses=posed,
                source_native_posed=teacher.source_candidate_posed,
                source_pose_mask=teacher.source_pose_mask,
                source_candidate_skin=source_skin,
                scale=mesh.rest_vertices_iris.detach().new_tensor(1.),
            )
            dynamic_loss=strain["total"]
            total=total+weights.dynamic_surface*dynamic_loss
        if weights.hard_skin>0:
            # Discrete argument is explicitly the MODEL's argmax, never
            # the teacher code sequence. Gradients go to spatial decoder;
            # AXIS learns code logits from qualified causal teacher CE.
            codes=seq.kind_ids[0]==int(RigTokenKindV6.SKIN_CODE)
            pos=torch.where(codes)[0]
            joints=int(valid[0].sum())
            if bool((pos==0).any()) or len(pos)!=joints*model.cfg.skin_tokens_per_joint:
                raise ValueError("RSS_VNEXT_PRODUCT_SEEDED_MIRA_HARD_CODE_ADDRESS")
            ids=controls["skin_code_logits"][0,pos-1].argmax(dim=-1).reshape(
                1,joints,model.cfg.skin_tokens_per_joint,
            )
            hard=model.mira.decode_discrete_skin_v1(
                mesh.mechanics_local_features,mesh.rest_vertices_iris,
                joint_state[:,:joints],ids,vertex_normals=mesh.normals,
                joint_group_ids=axis_joint_group_ids_v1(seq)[None],
            )
            if hard.shape!=source_skin.shape:
                raise ValueError("RSS_VNEXT_PRODUCT_SEEDED_MIRA_HARD_SKIN_SHAPE")
            hard_loss=F.smooth_l1_loss(hard,source_skin)
            total=total+weights.hard_skin*hard_loss
    if not bool(torch.isfinite(total).all()):
        raise ValueError("RSS_VNEXT_PRODUCT_SEEDED_AXIS_MIRA_LOSS_NONFINITE")
    return ProductSeededAXISMIRAResultV1(
        total=total,axis=axis_loss,skin=skin_loss,
        posed=pose_loss,dynamic=dynamic_loss,hard_skin=hard_loss,
        eligible_pose_faces=pose_count,
        chain_skin=chain_loss,eligible_chain_vertices=chain_count,
    )


__all__=[
    "ProductSeededAXISMIRAWeightsV1",
    "ProductSeededAXISMIRAResultV1",
    "product_seeded_axis_mira_objective_v1",
]
