from __future__ import annotations

"""Research-only TESSA/AXIS consumer ablation of shared scene support.

Three comparable consumers with ONE unmodified RSS.vNext checkpoint:
- PRODUCT_BASELINE: exact current `_structural_memory` output;
- FIXED_VOLUME_SUPPORT: same product streams + K dual-attended H(x) tokens;
- LEARNED_INTERIOR_SUPPORT: same K/width/modules, student-predicted coordinates.

Unlike a toy nearest-bone loss this probe enters the *actual* TESSA decoder
and the *actual* AXIS causal program. The research support module is never
attached to, exported from, or hidden inside the shipping RSS model. AXIS
prefixes here are causal evaluation prefixes, NOT privileged source rig input.
No output is evidence of trained accuracy, qualified topology or a socket.
"""
from dataclasses import dataclass

import torch
from torch import Tensor

from models.axis.rig_program_model_v6 import AXISSequenceV6
from .contracts import RSSvNextSceneState
from .dual_spatial_support_research_v1 import (
    RSSvNextDualSpatialSupportCandidateV1,
    RSSvNextDualSpatialSupportResearchV1,
)
from .unified_character_model import RSSvNextUnifiedCharacterModel

_MODES = (
    "PRODUCT_BASELINE",
    "FIXED_VOLUME_SUPPORT",
    "LEARNED_INTERIOR_SUPPORT",
)


@dataclass(frozen=True)
class RSSvNextDualSpatialConsumerProbeV1:
    conditioning_memory: Tensor
    topology_logits: Tensor
    axis_kind_logits: Tensor
    axis_anchor_parent_logits: Tensor
    support_candidate: RSSvNextDualSpatialSupportCandidateV1 | None
    scene_triplane_identity: int
    mode: str
    authority: str = "RESEARCH_ONLY_UNQUALIFIED_TESSA_AXIS_ABLATION"


def _product_surface_source(
    model: RSSvNextUnifiedCharacterModel,
    state: RSSvNextSceneState,
) -> Tensor:
    # Keep this exact formula in sync with
    # RSSvNextUnifiedCharacterModel._structural_memory(), without introducing
    # an uncontrolled new carrier for the production model.
    h = state.local_features
    return (
        h
        + model.articulation_prior(h)
        + model.affordance_to_memory(state.affordance_logits)
        + model.semantic_to_memory(state.semantic_logits)
        + model.semantic_evidence_to_memory(state.semantic_evidence_logits)
        + model.presentation_roles_to_memory(state.presentation_role_probabilities)
    )


def run_rss_vnext_dual_support_consumer_probe_v1(
    *, model: RSSvNextUnifiedCharacterModel,
    state: RSSvNextSceneState,
    research: RSSvNextDualSpatialSupportResearchV1,
    topology_prefix: Tensor,
    axis_prefix: AXISSequenceV6,
    mode: str,
) -> RSSvNextDualSpatialConsumerProbeV1:
    """Read real downstream topology/mechanics logits without changing RSS.

    The *same* causal prefix is supplied to all three A/B/C arms. The prefix
    can be source-supervised in a separate, admitted teacher-forcing trial;
    it is not a discovered skeleton and cannot qualify its own semantics.
    """
    if mode not in _MODES:
        raise ValueError("RSS_VNEXT_DUAL_CONSUMER_MODE_INVALID")
    if not isinstance(model, RSSvNextUnifiedCharacterModel):
        raise ValueError("RSS_VNEXT_DUAL_CONSUMER_MODEL_REQUIRED")
    if not isinstance(research, RSSvNextDualSpatialSupportResearchV1):
        raise ValueError("RSS_VNEXT_DUAL_CONSUMER_RESEARCH_REQUIRED")
    if research.field_width != model.cfg.iris.local_feature_dim:
        raise ValueError("RSS_VNEXT_DUAL_CONSUMER_CARRIER_WIDTH_DRIFT")
    if state.constructed_camera_features is not None:
        # The model's aux-view conditioning is a separate experimental axis.
        raise ValueError("RSS_VNEXT_DUAL_CONSUMER_AUX_VIEW_CONFOUNDED")
    planes = state.scene_triplanes
    b,p,c,h,w = planes.shape
    if (
        p != 3 or c != model.cfg.iris.plane_channels
        or b != 1 or research.support_count > state.query_xyz.shape[1]
        or topology_prefix.ndim != 2
        or topology_prefix.shape[0] != b
        or topology_prefix.shape[1] < 1
        or topology_prefix.dtype != torch.long
        or topology_prefix.device != planes.device
        or axis_prefix.kind_ids.shape[0] != b
        or axis_prefix.kind_ids.device != planes.device
        or state.local_features.shape[-1] != research.field_width
    ):
        raise ValueError("RSS_VNEXT_DUAL_CONSUMER_ADDRESS_OR_RESOURCE_DOMAIN")

    candidate: RSSvNextDualSpatialSupportCandidateV1 | None = None
    if mode == "PRODUCT_BASELINE":
        memory = model._structural_memory(state, auxiliary=None)
    else:
        candidate = research(
            iris=model.iris, scene_planes=planes,
            grid_xyz=state.query_xyz, grid_h=state.local_features,
            mode=mode,
        )
        # Preserve exact evidence and region identities from the product
        # structural stream; add the same K independent support tokens to
        # each dual branch. The transformer resampler output token budget
        # does not change, and presentation+scene streams remain unchanged.
        original_surface = _product_surface_source(model,state)
        enriched_surface = original_surface + (
            candidate.surface_tokens - state.local_features
        )
        surface_source = torch.cat(
            (enriched_surface,state.region_relation_tokens,
             candidate.interior_tokens),dim=1,
        )
        scene_source = planes.permute(0,1,3,4,2).reshape(b,p*h*w,c)
        presentation_source = (
            model.presentation(state.local_features)
            + model.graphics_to_memory(state.graphics_logits)
        )
        memory = model.tessa.encode_memory(
            scene_source=scene_source,
            surface_source=surface_source,
            presentation_source=presentation_source,
        )
    topology = model.tessa.decode_tokens(
        memory=memory,input_ids=topology_prefix,
    ).logits
    axis_memory = model._control_memory(state,memory,mesh=None)
    axis = model.axis(axis_memory,axis_prefix)
    return RSSvNextDualSpatialConsumerProbeV1(
        conditioning_memory=memory,
        topology_logits=topology,
        axis_kind_logits=axis["kind_logits"],
        axis_anchor_parent_logits=axis["anchor_parent_logits"],
        support_candidate=candidate,
        scene_triplane_identity=id(state.scene_triplanes),
        mode=mode,
    )


__all__=[
    "RSSvNextDualSpatialConsumerProbeV1",
    "run_rss_vnext_dual_support_consumer_probe_v1",
]
