from __future__ import annotations

"""Product-seeded E1..E5 parameter and gradient ownership contract.

Unlike the legacy RSSvNextStructuralTrainingPhaseContractV1, this contract
supports the active seeded M* product objective. Each phase constructs a NEW
optimizer after parameter activation; stale optimizer moments cannot cross a
curriculum transition. Vision and the independently fitted MIRA FSQ codec
are never accidentally unfrozen.
"""

from dataclasses import dataclass, asdict
from hashlib import sha256
from typing import Any
import json
import torch
from torch import nn
from torch.optim import Optimizer

from .training_optimizer_ownership_v1 import (
    qualify_rss_vnext_optimizer_ownership_v1,
)

SCHEMA="RealSaS.RSSvNext.ProductSeededPhaseOwnership.v1"
PREFIXES_GEOMETRY=(
    "iris.","tessa_seeded_head.","surface_refine.",
    "udf.","sign_logit.","component_udf.",
    "deformation_quality.","fc_surface_position.",
    "fc_surface_incidence.","fc_face_geometry.",
)
PREFIXES_SEMANTIC_GRAPHICS=(
    "early_semantics.","normal_condition_proj.",
    "semantic_evidence_to_memory.","presentation_role_head.",
    "presentation_roles_to_memory.","semantic.","semantic_structure.",
    "semantic_to_memory.","region_graph_to_memory.","relation.",
    "graphics.","graphics_to_memory.","presentation.",
    "constructed_view_renderer.","affordance.","affordance_to_memory.",
)
PREFIXES_MECHANICS=(
    "axis.","mira.","mesh_face_to_control.",
    "control_cross.","control_norm.","surface_mechanics_context.",
    "anchor_face_correspondence.","open_pin_vertex.","articulation_prior.",
)
PREFIXES_POSED=("dynamic_refine.",)
PROTECTED=("vision.","mira.codec.")
PHASES={
    "MSTAR_GEOMETRY_BOOTSTRAP":PREFIXES_GEOMETRY,
    "IRIS_GEOMETRY_SLR_GRAPHICS":
        PREFIXES_GEOMETRY+PREFIXES_SEMANTIC_GRAPHICS,
    "AXIS_MIRA_MSTAR_BINDING":
        PREFIXES_GEOMETRY+PREFIXES_SEMANTIC_GRAPHICS+PREFIXES_MECHANICS,
    "SOURCE_POSED_CONSEQUENCE":
        PREFIXES_GEOMETRY+PREFIXES_SEMANTIC_GRAPHICS+
        PREFIXES_MECHANICS+PREFIXES_POSED,
    "FULL_SHARED_H_FINETUNE":
        PREFIXES_GEOMETRY+PREFIXES_SEMANTIC_GRAPHICS+
        PREFIXES_MECHANICS+PREFIXES_POSED+("tessa.","mesh_bridge."),
}
STAGE_REQUIRED_GRADIENTS={
    "MSTAR_GEOMETRY_BOOTSTRAP":("iris.","tessa_seeded_head."),
    "IRIS_GEOMETRY_SLR_GRAPHICS":(
        "iris.","tessa_seeded_head.","graphics.",
    ),
    "AXIS_MIRA_MSTAR_BINDING":(
        "iris.","tessa_seeded_head.","axis.","mira.",
    ),
    "SOURCE_POSED_CONSEQUENCE":(
        "iris.","tessa_seeded_head.","axis.","mira.",
    ),
    "FULL_SHARED_H_FINETUNE":(
        "iris.","tessa_seeded_head.","axis.","mira.","graphics.",
    ),
}


@dataclass(frozen=True)
class ProductSeededPhaseOwnerPolicyV1:
    stage:str
    allowed_prefixes:tuple[str,...]
    required_gradient_prefixes:tuple[str,...]
    protected_prefixes:tuple[str,...]=PROTECTED
    schema:str=SCHEMA

    @classmethod
    def standard(cls,stage:str)->"ProductSeededPhaseOwnerPolicyV1":
        if stage not in PHASES:
            raise ValueError("RSS_VNEXT_PRODUCT_PHASE_STAGE_UNRECOGNIZED")
        return cls(stage,PHASES[stage],STAGE_REQUIRED_GRADIENTS[stage])

    def validate(self)->None:
        if (
            self.schema!=SCHEMA or self.stage not in PHASES
            or type(self.allowed_prefixes) is not tuple
            or type(self.required_gradient_prefixes) is not tuple
            or type(self.protected_prefixes) is not tuple
            or not self.allowed_prefixes
            or self.protected_prefixes!=PROTECTED
            or len(set(self.allowed_prefixes))!=len(self.allowed_prefixes)
            or len(set(self.required_gradient_prefixes))
                !=len(self.required_gradient_prefixes)
            or self.allowed_prefixes!=PHASES[self.stage]
            or self.required_gradient_prefixes
                !=STAGE_REQUIRED_GRADIENTS[self.stage]
            or any(not isinstance(p,str) or not p.endswith(".")
                   for p in (*self.allowed_prefixes,
                             *self.required_gradient_prefixes))
        ):
            raise ValueError("RSS_VNEXT_PRODUCT_PHASE_OWNER_POLICY_INVALID")

    @property
    def digest(self)->str:
        self.validate()
        return sha256(json.dumps(asdict(self),sort_keys=True,
                                 separators=(",",":")).encode()).hexdigest()

    def _authorized(self,name:str)->bool:
        return (
            any(name.startswith(p) for p in self.allowed_prefixes)
            and not any(name.startswith(p) for p in self.protected_prefixes)
        )

    def activate_before_optimizer_v1(self,model:nn.Module)->tuple[str,...]:
        """Load prior phase state, then call this, then construct a fresh optimizer.

        Explicitly changes requires_grad; NEVER invoke with an already-live
        optimizer. The stage runner must verify a weights-only handoff first.
        """
        self.validate()
        if not isinstance(model,nn.Module):
            raise TypeError("RSS_VNEXT_PRODUCT_PHASE_MODEL_INVALID")
        named=list(model.named_parameters())
        if not named:
            raise ValueError("RSS_VNEXT_PRODUCT_PHASE_EMPTY_MODEL")
        owned={name for name,_ in named if self._authorized(name)}
        if not owned or any(
            not any(n.startswith(prefix) for n in owned)
            for prefix in self.required_gradient_prefixes
        ):
            raise ValueError("RSS_VNEXT_PRODUCT_PHASE_REQUIRED_OWNER_NOT_PRESENT")
        for name,parameter in named:
            parameter.requires_grad_(name in owned)
            parameter.grad=None
        self.assert_active_trainable_owners(model)
        return tuple(name for name,_ in named if name in owned)

    def assert_active_trainable_owners(self,model:nn.Module)->None:
        self.validate()
        if not isinstance(model,nn.Module):
            raise TypeError("RSS_VNEXT_PRODUCT_PHASE_MODEL_INVALID")
        live={n for n,p in model.named_parameters() if p.requires_grad}
        if not live:
            raise ValueError("RSS_VNEXT_PRODUCT_PHASE_NO_ACTIVE_OWNERS")
        if any(not self._authorized(n) for n in live):
            raise ValueError("RSS_VNEXT_PRODUCT_PHASE_UNAUTHORIZED_TRAINABLE_OWNER")
        if any(
            not any(n.startswith(prefix) for n in live)
            for prefix in self.required_gradient_prefixes
        ):
            raise ValueError("RSS_VNEXT_PRODUCT_PHASE_REQUIRED_OWNER_NOT_TRAINABLE")

    def assert_qualified_optimizer(self,model:nn.Module,optimizer:Optimizer)->str:
        self.assert_active_trainable_owners(model)
        binding=qualify_rss_vnext_optimizer_ownership_v1(model,optimizer)
        if optimizer.state:
            raise ValueError("RSS_VNEXT_PRODUCT_PHASE_OPTIMIZER_MOMENTS_NOT_FRESH")
        return binding.binding_sha256

    def assert_gradient_owners_after_backward(self,model:nn.Module)->None:
        """No missing graph edge for explicitly required stage families."""
        self.assert_active_trainable_owners(model)
        for prefix in self.required_gradient_prefixes:
            live=[p.grad for n,p in model.named_parameters()
                  if n.startswith(prefix) and p.requires_grad]
            if not any(grad is not None for grad in live):
                raise RuntimeError(
                    "RSS_VNEXT_PRODUCT_PHASE_EXPECTED_GRADIENT_MISSING:"+prefix
                )
            if any(
                grad is not None
                and (grad.is_sparse or not bool(torch.isfinite(grad).all()))
                for grad in live
            ):
                raise RuntimeError(
                    "RSS_VNEXT_PRODUCT_PHASE_GRADIENT_NONFINITE:"+prefix
                )


__all__=[
    "SCHEMA","PHASES","PROTECTED","STAGE_REQUIRED_GRADIENTS",
    "ProductSeededPhaseOwnerPolicyV1",
]
