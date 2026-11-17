from __future__ import annotations

"""Production source-qualified Semantic objectives on the SAME IRIS H(x).

The original source patch labels, spatial presentation roles, co-region pairs
and typed relations are FOUR DIFFERENT kinds of evidence.  In particular:
 - 2D patch evidence is not a 3D drawable mask or attachment.
 - Source component co-membership cannot certify edit-cut or directed contact.
 - Source 3D role evidence must bind the exact H(x) grid query addresses.
 - The source is NEVER an extra neural observation (only 2 originals and six
   generated conditioning hypotheses are exposed to the student).
Missing or unqualified masks ABSTAIN. This is a training objective, not a
Compiler semantic/drawable authority certificate.
"""

from dataclasses import dataclass, asdict
import math
from typing import Mapping

import torch
from torch import Tensor
from .early_semantic_field import (
    masked_source_semantic_evidence_loss,
    masked_semantic_expected_brier_risk_loss,
)
from .semantic_geometry_prior import masked_qualified_surface_role_loss
from .semantic_structure import (
    qualified_source_region_pair_loss,
    qualified_source_typed_relation_loss,
    TYPED_REGION_RELATION_CHANNELS_V1,
)


@dataclass(frozen=True)
class ProductSemanticWeightsV1:
    patch_axes: float = 0.
    patch_risk: float = 0.
    surface_roles: float = 0.
    region_pairs: float = 0.
    typed_relations: float = 0.

    def validate(self)->None:
        if any(type(v) not in (int,float) or not math.isfinite(v) or v<0
               for v in asdict(self).values()):
            raise ValueError("RSS_VNEXT_PRODUCT_SEMANTIC_WEIGHT_POLICY")


@dataclass(frozen=True)
class ProductSemanticLossesV1:
    total: Tensor | None
    patch: Tensor | None
    patch_risk: Tensor | None
    roles: Tensor | None
    region: Tensor | None
    typed: Tensor | None
    patch_axes_count: int
    role_count: int
    region_count: int
    typed_counts: tuple[int,int,int]


def product_source_semantic_objective_v1(
    *,model,state,batch,
    weights:ProductSemanticWeightsV1,
    manifest_sha256:str,
    trusted_patch_registry:Mapping[str,Mapping[str,object]],
    trusted_role_registry:Mapping[str,Mapping[str,object]],
    trusted_region_registry:Mapping[str,Mapping[str,object]],
    trusted_typed_registry:Mapping[str,Mapping[str,object]],
    tensor_digest,
)->ProductSemanticLossesV1:
    weights.validate()
    out={"patch":None,"patch_risk":None,
         "roles":None,"region":None,"typed":None}
    counts={"patch":0,"roles":0,"region":0,"typed":(0,0,0)}
    source=state.scene_triplanes
    if (state.scene_conditioning_authority!="GENERATED_SIX_VIEW_SHARED_SCENE"
        or state.camera_features is not None or source.shape[0]!=1):
        raise ValueError("RSS_VNEXT_PRODUCT_SEMANTIC_SAME_H_REQUIRED")
    common={
        "dataset_manifest_sha256":manifest_sha256,
        "source_qualification_key":batch.source_qualification_key,
        "source_asset_sha256":batch.seeded_tessa_source_teacher.source_asset_sha256,
        "observed_rgba_sha256":tensor_digest(batch.rgba),
        "original_camera_sha256":tensor_digest(batch.camera_features),
        "source_binding_sha256":batch.product_source_binding_sha256,
        "combined_generation_checkpoint_sha256":
            batch.product_generation_checkpoint_sha256,
        "source_model_observation_count":2,
        "generated_views_as_source_truth":False,
    }
    def admit(registry,key,special,status,err):
        row=registry.get(key)
        expected={**common,"admission_status":status,**special}
        if not key or not isinstance(row,Mapping) or any(
            row.get(k)!=v for k,v in expected.items()
        ):
            raise ValueError(err)
    patch_fields=(batch.semantic_patch_target,batch.semantic_patch_mask)
    patch_requested=(
        any(x is not None for x in patch_fields)
        or bool(batch.semantic_qualification_key)
    )
    if patch_requested:
        if (
            any(x is None for x in patch_fields)
            or not weights.patch_axes and not weights.patch_risk
        ):
            raise ValueError("RSS_VNEXT_PRODUCT_SEMANTIC_PATCH_NOT_ENABLED_OR_INCOMPLETE")
        t,m=patch_fields
        if (
            t.shape!=state.source_semantic_patch_logits.shape
            or m.shape!=t.shape or m.dtype!=torch.bool
            or t.device!=source.device or m.device!=source.device
            or not bool(m.any())
            or not bool(torch.isfinite(t).all())
        ):
            raise ValueError("RSS_VNEXT_PRODUCT_SEMANTIC_PATCH_DOMAIN")
        admit(trusted_patch_registry,batch.semantic_qualification_key,{
            "semantic_patch_target_sha256":tensor_digest(t),
            "semantic_patch_mask_sha256":tensor_digest(m),
            "supervision_semantics":
                "ORIGINAL_TWO_VIEW_MASKED_PATCH_AXES_NOT_3D_DRAWABLE",
        },"QUALIFIED_SOURCE_SEMANTIC_PATCH_AXES",
        "RSS_VNEXT_PRODUCT_SEMANTIC_PATCH_RECEIPT_DRIFT")
        if weights.patch_axes>0:
            out["patch"]=masked_source_semantic_evidence_loss(
                state.source_semantic_patch_logits,t,m,
            )
        if weights.patch_risk>0:
            out["patch_risk"]=masked_semantic_expected_brier_risk_loss(
                state.source_semantic_patch_logits,
                state.source_semantic_log_uncertainty,t,m,
            )
        counts["patch"]=int(m.sum().item())
    role_fields=(
        batch.source_surface_role_indices,batch.source_surface_role_mask,
    )
    if (
        any(v is not None for v in role_fields)
        or bool(batch.source_surface_role_admission_key)
    ):
        if any(v is None for v in role_fields) or weights.surface_roles<=0:
            raise ValueError("RSS_VNEXT_PRODUCT_SEMANTIC_ROLE_NOT_ENABLED_OR_INCOMPLETE")
        t,m=role_fields
        admit(trusted_role_registry,batch.source_surface_role_admission_key,{
            "role_schema_id":model.cfg.semantic_role_schema_id,
            "query_xyz_sha256":tensor_digest(state.query_xyz),
            "role_indices_sha256":tensor_digest(t),
            "role_mask_sha256":tensor_digest(m),
        },"QUALIFIED_SOURCE_SURFACE_ROLE",
        "RSS_VNEXT_PRODUCT_SEMANTIC_ROLE_RECEIPT_DRIFT")
        out["roles"]=masked_qualified_surface_role_loss(
            state.presentation_role_logits,t,m,
            schema=model.cfg.semantic_role_schema(),
        )
        counts["roles"]=int(m.sum().item())
    region_fields=(
        batch.source_region_pair_indices,
        batch.source_region_pair_is_same,
        batch.source_region_pair_mask,
    )
    if any(x is not None for x in region_fields) or bool(
        batch.source_region_pair_admission_key
    ):
        if any(x is None for x in region_fields) or weights.region_pairs<=0:
            raise ValueError("RSS_VNEXT_PRODUCT_SEMANTIC_REGION_NOT_ENABLED_OR_INCOMPLETE")
        ids,t,m=region_fields
        admit(trusted_region_registry,batch.source_region_pair_admission_key,{
            "query_xyz_sha256":tensor_digest(state.query_xyz),
            "pair_indices_sha256":tensor_digest(ids),
            "pair_targets_sha256":tensor_digest(t),
            "pair_mask_sha256":tensor_digest(m),
            "supervision_semantics":"SOURCE_REGION_CO_MEMBERSHIP_NOT_DRAWABLE",
        },"QUALIFIED_SOURCE_REGION_PAIR",
        "RSS_VNEXT_PRODUCT_SEMANTIC_REGION_RECEIPT_DRIFT")
        if ids.shape[1]>model.cfg.max_pair_queries:
            raise ValueError("RSS_VNEXT_PRODUCT_SEMANTIC_REGION_BUDGET")
        out["region"]=qualified_source_region_pair_loss(
            state.semantic_logits,ids,t,m,
        )
        counts["region"]=int(m.sum().item())
    typed_fields=(
        batch.source_typed_relation_pair_indices,
        batch.source_typed_relation_target,
        batch.source_typed_relation_mask,
    )
    if any(x is not None for x in typed_fields) or bool(
        batch.source_typed_relation_admission_key
    ):
        if any(x is None for x in typed_fields) or weights.typed_relations<=0:
            raise ValueError("RSS_VNEXT_PRODUCT_SEMANTIC_TYPED_NOT_ENABLED_OR_INCOMPLETE")
        ids,t,m=typed_fields
        if ids.shape[1]>model.cfg.max_pair_queries:
            raise ValueError("RSS_VNEXT_PRODUCT_SEMANTIC_TYPED_BUDGET")
        row=trusted_typed_registry.get(batch.source_typed_relation_admission_key)
        channels=(
            row.get("channel_source_authority")
            if isinstance(row,Mapping) else None
        )
        approved=(
            "QUALIFIED_SOURCE_EDIT_DRAWABLE_CONSEQUENCE",
            "QUALIFIED_SOURCE_GEOMETRIC_SEPARABILITY",
            "QUALIFIED_SOURCE_CONTACT_AND_MOTION_INTERVENTION",
        )
        if not isinstance(channels,dict) or set(channels)!=set(
            TYPED_REGION_RELATION_CHANNELS_V1
        ) or any(
            channels[name]!=(approved[i] if bool(m[...,i].any())
                            else "UNAVAILABLE")
            for i,name in enumerate(TYPED_REGION_RELATION_CHANNELS_V1)
        ):
            raise ValueError("RSS_VNEXT_PRODUCT_SEMANTIC_TYPED_AUTHORITY_MISSING")
        admit(trusted_typed_registry,batch.source_typed_relation_admission_key,{
            "query_xyz_sha256":tensor_digest(state.query_xyz),
            "pair_indices_sha256":tensor_digest(ids),
            "typed_targets_sha256":tensor_digest(t),
            "typed_mask_sha256":tensor_digest(m),
            "supervision_semantics":
                "SOURCE_QUERY_PAIR_TYPED_DRAWABLE_CUT_DIRECTED_ATTACHMENT_MASKED_V1",
        },"QUALIFIED_SOURCE_TYPED_REGION_RELATIONS",
        "RSS_VNEXT_PRODUCT_SEMANTIC_TYPED_RECEIPT_DRIFT")
        out["typed"],counts["typed"]=qualified_source_typed_relation_loss(
            state.semantic_logits,
            state.region_relation_logits,
            state.region_relation_valid_mask,
            ids,t,m,
        )
    total=None
    for name,weight in (
        ("patch",weights.patch_axes),
        ("patch_risk",weights.patch_risk),
        ("roles",weights.surface_roles),
        ("region",weights.region_pairs),
        ("typed",weights.typed_relations),
    ):
        value=out[name]
        if value is not None and weight>0:
            term=weight*value
            total=term if total is None else total+term
    if total is None and any(v>0 for v in asdict(weights).values()):
        raise ValueError("RSS_VNEXT_PRODUCT_SEMANTIC_NO_QUALIFIED_TARGET")
    return ProductSemanticLossesV1(
        total=total,patch=out["patch"],patch_risk=out["patch_risk"],
        roles=out["roles"],region=out["region"],typed=out["typed"],
        patch_axes_count=counts["patch"],role_count=counts["roles"],
        region_count=counts["region"],typed_counts=counts["typed"],
    )


__all__=["ProductSemanticWeightsV1","ProductSemanticLossesV1",
         "product_source_semantic_objective_v1"]
