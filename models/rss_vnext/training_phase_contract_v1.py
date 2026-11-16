from __future__ import annotations

"""Opt-in, fail-closed supervised training phase contract for RSS.vNext.

A trainer phase has explicit objective-family and parameter ownership. No
teacher is invented from model predictions and no absent supervised objective
is silently turned into a valid zero loss.
"""
from dataclasses import dataclass, asdict
from hashlib import sha256
import json
import torch
from torch import nn, Tensor

SCHEMA = "RealSaS.RSSvNext.StructuralTrainingPhaseContract.v1"

# Weight -> (loss field, qualified count); None = own admission gate.
LOSS_FIELDS = {
    "topology": ("topology_loss", "eligible_topology_examples"),
    "motion_preference": ("motion_preference_loss", "qualified_preference_pairs"),
    "source_semantics": ("source_semantic_loss", "eligible_semantic_patch_labels"),
    "semantic_risk": ("semantic_risk_loss", "eligible_semantic_patch_labels"),
    "surface_roles": ("surface_role_loss", "eligible_surface_role_points"),
    "region_pairs": ("region_pair_loss", "eligible_region_pairs"),
    "typed_relations": ("typed_relation_loss", "eligible_typed_relation_channels"),
    "graphics_source_rgb": ("graphics_source_rgb_loss", "eligible_graphics_source_rgb_queries"),
    "source_udf": ("source_udf_loss", "eligible_source_udf_queries"),
    "source_inside_sign": ("source_inside_sign_loss", "eligible_source_inside_sign_queries"),
    "source_signed_sdf": ("source_signed_sdf_loss", "eligible_source_signed_sdf_queries"),
    "axis_rig_skin_codes": ("axis_rig_skin_loss", "eligible_axis_sequences"),
    "axis_rig_grounding": ("axis_rig_grounding_loss", "eligible_axis_sequences"),
    "axis_posed_surface": ("axis_posed_surface_loss", "eligible_axis_pose_count"),
    "axis_dynamic_posed_surface": ("axis_dynamic_posed_surface_loss", "eligible_axis_dynamic_pose_count"),
    "anchor_face_correspondence": ("anchor_face_correspondence_loss", "eligible_anchor_held_faces"),
    "flexicubes_surface": ("flexicubes_surface_loss", None),
    "flexicubes_animation": ("flexicubes_animation_loss", "eligible_animation_pose_count"),
    "tessa_iris_surface_alignment": ("tessa_iris_surface_alignment_loss", None),
    "novel_view": ("novel_view_loss", "eligible_novel_view_pixels"),
    "realsas_joint_slrm": ("realsas_joint_slrm_loss", "eligible_realsas_joint_rays"),
}


@dataclass(frozen=True)
class RSSvNextStructuralTrainingPhaseContractV1:
    name: str
    trainable_parameter_prefixes: tuple[str, ...]
    allowed_positive_objectives: tuple[str, ...]
    required_any_qualified_loss_groups: tuple[tuple[str, ...], ...]
    # Explicit active owners must receive a differentiable gradient per step.
    required_gradient_parameter_prefixes: tuple[str, ...] = ()

    def validate(self, *, weights=None) -> None:
        if (
            type(self.name) is not str or not self.name or len(self.name)>96
            or type(self.trainable_parameter_prefixes) is not tuple
            or not self.trainable_parameter_prefixes
            or len(set(self.trainable_parameter_prefixes))!=len(self.trainable_parameter_prefixes)
            or any(type(p) is not str or not p.endswith(".") or p.startswith(".")
                   or ".." in p for p in self.trainable_parameter_prefixes)
            or type(self.allowed_positive_objectives) is not tuple
            or not self.allowed_positive_objectives
            or len(set(self.allowed_positive_objectives))!=len(self.allowed_positive_objectives)
            or any(x not in LOSS_FIELDS for x in self.allowed_positive_objectives)
            or type(self.required_any_qualified_loss_groups) is not tuple
            or not self.required_any_qualified_loss_groups
        ):
            raise ValueError("RSS_VNEXT_PHASE_CONTRACT_INVALID")
        if (
            type(self.required_gradient_parameter_prefixes) is not tuple
            or len(set(self.required_gradient_parameter_prefixes))
                != len(self.required_gradient_parameter_prefixes)
            or any(
                type(prefix) is not str or not prefix.endswith(".")
                or not any(prefix.startswith(owner)
                           for owner in self.trainable_parameter_prefixes)
                for prefix in self.required_gradient_parameter_prefixes
            )
        ):
            raise ValueError("RSS_VNEXT_PHASE_REQUIRED_GRADIENT_PREFIX_INVALID")
        for group in self.required_any_qualified_loss_groups:
            if (type(group) is not tuple or not group
                or len(set(group))!=len(group)
                or any(x not in self.allowed_positive_objectives for x in group)):
                raise ValueError("RSS_VNEXT_PHASE_REQUIRED_GROUP_INVALID")
        if weights is not None:
            weights.validate()
            active={x for x in LOSS_FIELDS if getattr(weights,x)>0}
            if not active.issubset(self.allowed_positive_objectives):
                raise ValueError("RSS_VNEXT_PHASE_UNAUTHORIZED_OBJECTIVE")
            if any(not active.intersection(g) for g in self.required_any_qualified_loss_groups):
                raise ValueError("RSS_VNEXT_PHASE_REQUIRED_OBJECTIVE_DISABLED")

    def digest(self) -> str:
        self.validate()
        return sha256(json.dumps(
            {"schema":SCHEMA,"contract":asdict(self)},sort_keys=True,
            separators=(",",":"),allow_nan=False,
        ).encode("utf8")).hexdigest()

    def _matches(self,name:str)->bool:
        return any(name.startswith(p) for p in self.trainable_parameter_prefixes)

    def assert_trainable_owners(self,model:nn.Module)->None:
        self.validate()
        if not isinstance(model,nn.Module):
            raise TypeError("RSS_VNEXT_PHASE_MODEL_INVALID")
        names=[n for n,p in model.named_parameters() if p.requires_grad]
        if not names:
            raise ValueError("RSS_VNEXT_PHASE_NO_TRAINABLE_PARAMETERS")
        foreign=[n for n in names if not self._matches(n)]
        if foreign:
            raise ValueError("RSS_VNEXT_PHASE_UNAUTHORIZED_TRAINABLE_OWNER:"+",".join(foreign[:3]))
        unbound=[p for p in self.trainable_parameter_prefixes
                 if not any(n.startswith(p) for n in names)]
        if unbound:
            raise ValueError("RSS_VNEXT_PHASE_UNBOUND_OWNER_PREFIX:"+",".join(unbound))

    def freeze_for_phase(self,model:nn.Module)->tuple[str,...]:
        """Only freezes, never unfreezes an upstream protected encoder."""
        self.validate()
        if not isinstance(model,nn.Module):
            raise TypeError("RSS_VNEXT_PHASE_MODEL_INVALID")
        for name,parameter in model.named_parameters():
            if not self._matches(name):
                parameter.requires_grad_(False)
                parameter.grad=None
        self.assert_trainable_owners(model)
        return tuple(n for n,p in model.named_parameters() if p.requires_grad)

    def assert_required_gradients(self, model: nn.Module) -> None:
        """After backward and BEFORE optimizer.step, reject an inactive head.

        Zero gradients can occur at convergence; absent gradients indicate an
        unconnected computation. The requirement is explicit and phase-local.
        """
        self.validate()
        self.assert_trainable_owners(model)
        for prefix in self.required_gradient_parameter_prefixes:
            params = [
                p for name, p in model.named_parameters()
                if name.startswith(prefix) and p.requires_grad
            ]
            if not params:
                raise ValueError("RSS_VNEXT_PHASE_REQUIRED_GRADIENT_OWNER_MISSING:"+prefix)
            live = [p.grad for p in params if p.grad is not None]
            if not live:
                raise ValueError("RSS_VNEXT_PHASE_REQUIRED_GRADIENT_MISSING:"+prefix)
            if any(
                grad.is_sparse or not bool(torch.isfinite(grad).all())
                for grad in live
            ):
                raise ValueError("RSS_VNEXT_PHASE_REQUIRED_GRADIENT_NONFINITE:"+prefix)

    def assert_qualified_result(self,result,*,weights)->None:
        """A zero-valued but qualified loss is valid; an absent loss is not."""
        self.validate(weights=weights)
        for group in self.required_any_qualified_loss_groups:
            accepted=False
            for name in group:
                if getattr(weights,name)<=0:
                    continue
                loss_field,coverage_field=LOSS_FIELDS[name]
                loss=getattr(result,loss_field,None)
                if loss is None:
                    continue
                if (not isinstance(loss,Tensor) or loss.numel()!=1
                    or not bool(torch.isfinite(loss).all())):
                    raise ValueError("RSS_VNEXT_PHASE_INVALID_SCALAR_LOSS")
                if coverage_field is not None:
                    count=getattr(result,coverage_field,None)
                    if type(count) is tuple:
                        count=sum(count)
                    if type(count) is not int or count<1:
                        continue
                accepted=True
                break
            if not accepted:
                raise ValueError("RSS_VNEXT_PHASE_REQUIRED_QUALIFIED_SUPERVISION_MISSING:"+",".join(group))


__all__=["SCHEMA","LOSS_FIELDS","RSSvNextStructuralTrainingPhaseContractV1"]
