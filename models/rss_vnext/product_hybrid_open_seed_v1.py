from __future__ import annotations

"""One-H product inference bridge: signed IRIS shell + learned open UDF -> M0.

The *model* selects an OPEN component from its exchangeable region slots:
no source garment ID, native rig, reference mesh, teacher component index or
post-hoc nearest mesh welding is used to pick the open surface. The resulting
ONE indexed M0 is refined by the existing TESSA path before AXIS and MIRA.

This is source-free, optional and ABSTAIN-first. It does NOT qualify inferred
surface topology, material properties, source appearance or ProductRevision.
"""

from dataclasses import dataclass
import math

import torch
from torch import Tensor

from .component_conditioned_udf_v1 import KIND_OPEN, KIND_CLOSED, KIND_UNKNOWN
from .hybrid_signed_open_iris_seed_v1 import (
    HybridSignedOpenIRISSeedV1,
    join_hybrid_iris_seed_v1,
)
from .open_udf_surface_v1 import extract_open_udf_surface_v1
from .tessa_seeded_surface_v1 import TESSASeededSurfaceV1

POLICY_SCHEMA = "RealSaS.RSSvNext.ProductHybridOpenM0Policy.v1"
AUTHORITY = "ONE_SHARED_H_MODEL_PREDICTED_OPEN_COMPONENT_NOT_SOURCE_QUALIFIED"


@dataclass(frozen=True)
class ProductHybridOpenM0PolicyV1:
    """Bounded extraction with source-independent component selection."""

    open_layer_name: str = "open_component"
    open_presentation_roles: tuple[str, ...] = ("outer_covering",)
    resolution: tuple[int, int, int] = (24, 24, 24)
    query_chunk: int = 4096
    min_open_probability: float = 0.75
    min_kind_margin: float = 0.20
    min_slot_margin: float = 0.10
    max_output_vertices: int = 65536

    def validate(self) -> None:
        if (
            not isinstance(self.open_layer_name, str) or not self.open_layer_name
            or not isinstance(self.open_presentation_roles, tuple)
            or not self.open_presentation_roles
            or any(not isinstance(x, str) or not x
                   for x in self.open_presentation_roles)
            or len(set(self.open_presentation_roles))
                   != len(self.open_presentation_roles)
            or not isinstance(self.resolution, tuple) or len(self.resolution) != 3
            or any(type(x) is not int or not 8 <= x <= 96
                   for x in self.resolution)
            or math.prod(self.resolution) > 700000
            or type(self.query_chunk) is not int
            or not 128 <= self.query_chunk <= 65536
            or type(self.max_output_vertices) is not int
            or not 3 <= self.max_output_vertices <= 65536
            or any(
                type(x) not in (float, int) or not math.isfinite(x)
                or not 0 <= x <= 1
                for x in (
                    self.min_open_probability, self.min_kind_margin,
                    self.min_slot_margin,
                )
            )
        ):
            raise ValueError("RSS_VNEXT_HYBRID_OPEN_POLICY_INVALID")


@dataclass(frozen=True)
class ProductHybridOpenM0ProposalV1:
    hybrid: HybridSignedOpenIRISSeedV1
    predicted_open_slot_index: int
    open_probability: float
    kind_margin: float
    slot_margin: float
    source_authority: str = AUTHORITY
    product_admission_claimed: bool = False


def propose_product_hybrid_open_m0_v1(
    *,
    model,
    state,
    signed_seed: TESSASeededSurfaceV1,
    policy: ProductHybridOpenM0PolicyV1,
    signed_layer_name: str = "signed_solid",
    signed_presentation_roles: tuple[str, ...] = ("primary_form",),
) -> ProductHybridOpenM0ProposalV1:
    """Use exactly the six-conditioned scene H; do not infer from teacher IDs."""
    if type(policy) is not ProductHybridOpenM0PolicyV1:
        raise ValueError("RSS_VNEXT_HYBRID_OPEN_POLICY_TYPE")
    policy.validate()
    signed_seed.validate()
    if (
        state.scene_conditioning_authority != "GENERATED_SIX_VIEW_SHARED_SCENE"
        or state.camera_features is not None
        or state.scene_triplanes.ndim != 5
        or state.scene_triplanes.shape[0] != 1
        or signed_seed.xyz.device != state.scene_triplanes.device
        or signed_seed.xyz.dtype != state.scene_triplanes.dtype
    ):
        raise ValueError("RSS_VNEXT_HYBRID_REQUIRES_SAME_GENERATED_SIX_VIEW_H")
    valid_roles = set(model.cfg.semantic_role_schema().roles)
    if (
        type(signed_layer_name) is not str or not signed_layer_name
        or type(signed_presentation_roles) is not tuple
        or not signed_presentation_roles
        or any(type(role) is not str or role not in valid_roles
               for role in signed_presentation_roles)
        or set(signed_presentation_roles).intersection(
            policy.open_presentation_roles,
        )
    ):
        raise ValueError("RSS_VNEXT_HYBRID_SIGNED_LAYER_ROLE_POLICY")
    if not set(policy.open_presentation_roles).issubset(valid_roles):
        raise ValueError("RSS_VNEXT_HYBRID_UNKNOWN_PRESENTATION_ROLE")

    kind_logits = model.query_component_topology_kind_logits(state)
    if (
        not isinstance(kind_logits, Tensor)
        or kind_logits.ndim != 3
        or kind_logits.shape[0] != 1
        or kind_logits.shape[1] < 1
        or kind_logits.shape[2] != 3
        or kind_logits.device != state.scene_triplanes.device
        or kind_logits.dtype != state.scene_triplanes.dtype
        or not bool(torch.isfinite(kind_logits).all())
    ):
        raise ValueError("RSS_VNEXT_HYBRID_COMPONENT_KIND_LOGITS_INVALID")
    # A deterministic source-free slot selection; UNKNOWN is its own class,
    # not a negative label and not an implicit garment.
    prob = torch.softmax(kind_logits[0].float(), dim=-1).detach()
    open_probs = prob[:, KIND_OPEN]
    order = torch.argsort(open_probs, descending=True, stable=True)
    slot = int(order[0].item())
    p = float(open_probs[slot].item())
    kind_margin = p - max(
        float(prob[slot, KIND_CLOSED].item()),
        float(prob[slot, KIND_UNKNOWN].item()),
    )
    slot_margin = p - (
        float(open_probs[order[1]].item()) if len(order) > 1 else 0.0
    )
    if (
        p < policy.min_open_probability
        or kind_margin < policy.min_kind_margin
        or slot_margin < policy.min_slot_margin
    ):
        raise ValueError("RSS_VNEXT_HYBRID_OPEN_SLOT_ABSTAIN_UNCERTAIN")

    def query(points: Tensor) -> Tensor:
        # The same H(x) and learned region slot used in product inference,
        # not any source/GT garment mask. The extractor re-queries active
        # edges with gradients when called in a trainable context.
        out = model.query_component_unsigned_distance(
            state, points.unsqueeze(0), slot_index=slot,
        )
        if not isinstance(out, Tensor) or out.shape != (1, len(points)):
            raise ValueError("RSS_VNEXT_HYBRID_COMPONENT_FIELD_QUERY_DRIFT")
        return out[0]

    open_surface = extract_open_udf_surface_v1(
        scene_triplanes=state.scene_triplanes,
        query_unsigned_distance=query,
        resolution=policy.resolution,
        query_chunk=policy.query_chunk,
        max_output_vertices=policy.max_output_vertices,
    )
    hybrid = join_hybrid_iris_seed_v1(
        signed=signed_seed,
        open_udf=open_surface,
        scene_triplanes=state.scene_triplanes,
        open_layer_name=policy.open_layer_name,
        open_presentation_roles=policy.open_presentation_roles,
        signed_layer_name=signed_layer_name,
        signed_presentation_roles=signed_presentation_roles,
    )
    hybrid.validate()
    return ProductHybridOpenM0ProposalV1(
        hybrid=hybrid,
        predicted_open_slot_index=slot,
        open_probability=p,
        kind_margin=kind_margin,
        slot_margin=slot_margin,
    )


__all__ = [
    "POLICY_SCHEMA", "AUTHORITY",
    "ProductHybridOpenM0PolicyV1", "ProductHybridOpenM0ProposalV1",
    "propose_product_hybrid_open_m0_v1",
]
