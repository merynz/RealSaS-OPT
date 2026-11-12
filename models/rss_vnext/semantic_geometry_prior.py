from __future__ import annotations

"""StdGEN-family semantic-equivalent geometry, with a versioned K-way role schema.

The verified StdGEN 2025 implementation uses four roles, but the algebra is
independent of K. K=6 (or another justified K) is a schema choice, not a claim
that an unpublished StdGEN++ trainer used six classes.

Roles are SOURCE-RENDERABLE presentation/material strata, not joints, motion,
grips, attachments, or a human-only body-part taxonomy. The caller must
establish source-to-role authority before enabling any semantic loss.
"""
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class RSSvNextSemanticRoleSchema:
    roles: tuple[str, ...]
    schema_id: str

    def validate(self) -> None:
        if not self.schema_id or not 2 <= len(self.roles) <= 16:
            raise ValueError("RSS_VNEXT_ROLE_SCHEMA_ID_OR_CARDINALITY")
        if any(not x or not x.strip() or x.strip() != x for x in self.roles):
            raise ValueError("RSS_VNEXT_ROLE_SCHEMA_INVALID_NAME")
        if len(self.roles) != len(set(self.roles)):
            raise ValueError("RSS_VNEXT_ROLE_SCHEMA_DUPLICATE_ROLE")

    @property
    def count(self) -> int:
        self.validate()
        return len(self.roles)


@dataclass(frozen=True)
class RSSvNextSemanticLayerSpec:
    name: str
    keep_roles: tuple[str, ...]

    def validate(self, schema: RSSvNextSemanticRoleSchema) -> None:
        schema.validate()
        if not self.name or not self.keep_roles:
            raise ValueError("RSS_VNEXT_SEMANTIC_LAYER_EMPTY")
        if len(set(self.keep_roles)) != len(self.keep_roles):
            raise ValueError("RSS_VNEXT_SEMANTIC_LAYER_DUPLICATE_ROLE")
        if not set(self.keep_roles).issubset(set(schema.roles)):
            raise ValueError("RSS_VNEXT_SEMANTIC_LAYER_UNKNOWN_ROLE")


def stdgen_style_generic_schema() -> RSSvNextSemanticRoleSchema:
    """Same class index semantics as published 2025 StdGEN, generalized names.

    [0] secondary form (hair-like)
    [1] focal detail (face-like)
    [2] primary form (body-like)
    [3] outer covering (cloth-like)
    """
    return RSSvNextSemanticRoleSchema(
        roles=("secondary_form", "focal_detail", "primary_form", "outer_covering"),
        schema_id="RSS.vNext.SemanticPresentationRoles.FourWayResearch.v1",
    )


def stdgen_style_layers(schema: RSSvNextSemanticRoleSchema) -> tuple[RSSvNextSemanticLayerSpec, ...]:
    """Reproduce StdGEN's five extraction questions for the canonical four roles.

    This is a 4-way *baseline*, not a mandatory semantic policy for K=6.
    """
    if schema.roles != stdgen_style_generic_schema().roles:
        raise ValueError("RSS_VNEXT_STDGEN_FIVE_LEVELS_REQUIRE_FOUR_ROLE_BASELINE")
    r = schema.roles
    return (
        RSSvNextSemanticLayerSpec("all", r),
        RSSvNextSemanticLayerSpec("without_secondary", r[1:]),
        RSSvNextSemanticLayerSpec("without_secondary_or_covering", r[1:3]),
        RSSvNextSemanticLayerSpec("isolated_covering", (r[3],)),
        RSSvNextSemanticLayerSpec("isolated_secondary", (r[0],)),
    )


def semantic_equivalent_signed_field(
    signed_field: Tensor,
    semantic_logits: Tensor,
    *,
    schema: RSSvNextSemanticRoleSchema,
    layer: RSSvNextSemanticLayerSpec,
    qualified_signed_domain: bool,
) -> Tensor:
    """Generalization of published StdGEN semantic SDF levels to any K.

    For retained subset S:
      f_S(x) = max(f(x), max_{r not in S} p_r(x) - max_{r in S} p_r(x)).
    For all roles, f_S=f. For one role this matches the StdGEN++ stated
    semantic-equivalent SDF equation. No edge/attachment inference is claimed.

    Operation is permitted ONLY for a court-qualified signed solid domain.
    Open/modular/UDF surfaces must use explicit patch-based extraction.
    """
    schema.validate()
    layer.validate(schema)
    if not qualified_signed_domain:
        raise ValueError("RSS_VNEXT_SEMANTIC_SIGNED_DOMAIN_NOT_QUALIFIED")
    if signed_field.ndim != 2 or semantic_logits.shape != (*signed_field.shape, schema.count):
        raise ValueError("RSS_VNEXT_SEMANTIC_FIELD_SHAPE")
    if not (torch.isfinite(signed_field).all() and torch.isfinite(semantic_logits).all()):
        raise ValueError("RSS_VNEXT_SEMANTIC_FIELD_NONFINITE")
    keep = [schema.roles.index(name) for name in layer.keep_roles]
    excluded = [i for i in range(schema.count) if i not in keep]
    if not excluded:
        return signed_field
    # No hard argmax or detached class label: gradients flow to role logits
    # and shared geometry through the softmax-based suppression boundary.
    p = F.softmax(semantic_logits, dim=-1)
    inside = p[..., keep].amax(dim=-1)
    outside = p[..., excluded].amax(dim=-1)
    return torch.maximum(signed_field, outside - inside)


def all_semantic_equivalent_layers(
    signed_field: Tensor,
    semantic_logits: Tensor,
    *,
    schema: RSSvNextSemanticRoleSchema,
    layers: tuple[RSSvNextSemanticLayerSpec, ...],
    qualified_signed_domain: bool,
) -> dict[str, Tensor]:
    if len({x.name for x in layers}) != len(layers) or not layers:
        raise ValueError("RSS_VNEXT_SEMANTIC_LAYER_NAMES_NOT_UNIQUE")
    return {
        layer.name: semantic_equivalent_signed_field(
            signed_field, semantic_logits, schema=schema, layer=layer,
            qualified_signed_domain=qualified_signed_domain,
        )
        for layer in layers
    }


def masked_visible_surface_role_loss(
    semantic_logits: Tensor,
    role_indices: Tensor,
    qualified_visible_mask: Tensor,
    *,
    schema: RSSvNextSemanticRoleSchema,
) -> Tensor:
    """StdGEN-style K-way *visible surface* teacher, not arbitrary pixel labels.

    [B,2,T,K] predictions, [B,2,T] source-render role indices; mask true only
    for admitted, source-proven visible role pixels/patches. The source-role
    mapping and camera/material/mesh identity must be certified upstream.
    Unqualified/hidden regions are NOT treated as background or negative.
    """
    schema.validate()
    if semantic_logits.ndim != 4 or semantic_logits.shape[1] != 2 or semantic_logits.shape[-1] != schema.count:
        raise ValueError("RSS_VNEXT_SEMANTIC_VISIBLE_LOGIT_SHAPE")
    expected = semantic_logits.shape[:3]
    if role_indices.shape != expected or qualified_visible_mask.shape != expected:
        raise ValueError("RSS_VNEXT_SEMANTIC_VISIBLE_TEACHER_SHAPE")
    if qualified_visible_mask.dtype != torch.bool or role_indices.dtype != torch.long:
        raise ValueError("RSS_VNEXT_SEMANTIC_VISIBLE_TEACHER_DTYPE")
    if not bool(qualified_visible_mask.any()):
        raise ValueError("RSS_VNEXT_SEMANTIC_VISIBLE_NO_QUALIFIED_PIXELS")
    values = role_indices[qualified_visible_mask]
    if bool((values < 0).any()) or bool((values >= schema.count).any()):
        raise ValueError("RSS_VNEXT_SEMANTIC_VISIBLE_ROLE_OUT_OF_RANGE")
    if not bool(torch.isfinite(semantic_logits).all()):
        raise ValueError("RSS_VNEXT_SEMANTIC_VISIBLE_NONFINITE")
    return F.cross_entropy(semantic_logits[qualified_visible_mask].float(), values)


def masked_qualified_surface_role_loss(
    role_logits: Tensor,
    source_surface_role_indices: Tensor,
    qualified_surface_mask: Tensor,
    *,
    schema: RSSvNextSemanticRoleSchema,
) -> Tensor:
    """Teacher-only multiclass labels at exact 3D shared-carrier query addresses.

    Only a separately source-admitted surface sample can be supervised.
    Unobserved/open/ambiguous points are masked, never cast to background.
    This is not rig, drawable-boundary or visibility authority.
    """
    schema.validate()
    if role_logits.ndim != 3 or role_logits.shape[-1] != schema.count:
        raise ValueError("RSS_VNEXT_SURFACE_ROLE_LOGITS_SHAPE")
    if (
        source_surface_role_indices.shape != role_logits.shape[:2]
        or qualified_surface_mask.shape != role_logits.shape[:2]
    ):
        raise ValueError("RSS_VNEXT_SURFACE_ROLE_TARGET_ADDRESS_SHAPE")
    if (
        source_surface_role_indices.dtype != torch.long
        or qualified_surface_mask.dtype != torch.bool
    ):
        raise ValueError("RSS_VNEXT_SURFACE_ROLE_TARGET_DTYPE")
    if not bool(qualified_surface_mask.any()):
        raise ValueError("RSS_VNEXT_SURFACE_ROLE_NO_ADMITTED_POINTS")
    if not bool(torch.isfinite(role_logits).all()):
        raise ValueError("RSS_VNEXT_SURFACE_ROLE_LOGITS_NONFINITE")
    valid = source_surface_role_indices[qualified_surface_mask]
    if bool(((valid < 0) | (valid >= schema.count)).any()):
        raise ValueError("RSS_VNEXT_SURFACE_ROLE_LABEL_OUT_OF_RANGE")
    return F.cross_entropy(role_logits[qualified_surface_mask].float(), valid)


__all__ = [
    "RSSvNextSemanticRoleSchema", "RSSvNextSemanticLayerSpec",
    "stdgen_style_generic_schema", "stdgen_style_layers",
    "semantic_equivalent_signed_field", "all_semantic_equivalent_layers",
    "masked_visible_surface_role_loss", "masked_qualified_surface_role_loss",
]
