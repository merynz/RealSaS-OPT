from __future__ import annotations

"""Canonical deterministic completion for surface samples unseen in every view.

Source-backed appearance remains directional A(p,v). Only samples with no
qualified source support in any input direction are eligible for the
view-independent canonical completion C(p).

This module is backend/challenger infrastructure. Calling it does not promote
any shipping policy.
"""

from dataclasses import dataclass

import numpy as np

from .appearance_color_v2 import (
    premultiplied_linear_to_straight_srgb_u8,
    straight_srgb_rgba_u8_to_premultiplied_linear,
)
from .appearance_completion_v2 import SurfaceSampleGraph
from .appearance_variational_completion_v1 import (
    VariationalCompletionStats,
    solve_weighted_surface_dirichlet,
)
from .types import QualificationError


@dataclass(frozen=True)
class CanonicalUnseenCompletion:
    rgba: np.ndarray
    globally_unseen_mask: np.ndarray
    anchor_mask: np.ndarray
    anchor_source_view: np.ndarray
    solver_stats: VariationalCompletionStats

    def metadata(self) -> dict:
        return {
            "mode": "ALL_VIEW_UNSEEN_CANONICAL_VARIATIONAL_V1",
            "globally_unseen_sample_count": int(
                np.count_nonzero(self.globally_unseen_mask)
            ),
            "anchor_sample_count": int(np.count_nonzero(self.anchor_mask)),
            "view_independent_completion": True,
            "source_backed_directional_appearance_mutated": False,
            "runtime_generation_required": False,
            "solver": self.solver_stats.to_dict(),
        }


def build_all_view_unseen_canonical_completion(
    *,
    direct_valid: np.ndarray,
    direct_rgba: np.ndarray,
    face_support_by_view: np.ndarray,
    sample_face_index: np.ndarray,
    sample_component_index: np.ndarray,
    sample_positions: np.ndarray,
    surface_graph: SurfaceSampleGraph,
) -> CanonicalUnseenCompletion:
    valid=np.asarray(direct_valid,dtype=bool)
    rgba=np.asarray(direct_rgba,dtype=np.uint8)
    support=np.asarray(face_support_by_view,dtype=np.float64)
    sample_face=np.asarray(sample_face_index,dtype=np.int32)
    component=np.asarray(sample_component_index,dtype=np.int32)
    positions=np.asarray(sample_positions,dtype=np.float64)

    if valid.ndim!=2 or valid.shape[0]!=8:
        raise QualificationError(
            "CAA_CANONICAL_COMPLETION_DIRECT_VALID_SHAPE_INVALID"
        )
    sample_count=valid.shape[1]
    if rgba.shape!=(8,sample_count,4):
        raise QualificationError(
            "CAA_CANONICAL_COMPLETION_DIRECT_RGBA_SHAPE_INVALID"
        )
    if (
        support.ndim!=2
        or support.shape[0]!=8
        or sample_face.shape!=(sample_count,)
        or component.shape!=(sample_count,)
        or positions.shape!=(sample_count,3)
        or len(surface_graph)!=sample_count
    ):
        raise QualificationError(
            "CAA_CANONICAL_COMPLETION_INPUT_SHAPE_DRIFT"
        )
    if (
        np.any(sample_face<0)
        or np.any(sample_face>=support.shape[1])
        or not np.isfinite(support).all()
        or not np.isfinite(positions).all()
    ):
        raise QualificationError(
            "CAA_CANONICAL_COMPLETION_INPUT_INVALID"
        )

    anchor_mask=np.any(valid,axis=0)
    globally_unseen=~anchor_mask
    if not np.any(anchor_mask):
        raise QualificationError(
            "CAA_CANONICAL_COMPLETION_NO_SOURCE_ANCHOR"
        )
    if not np.any(globally_unseen):
        empty_rgba=np.zeros((sample_count,4),dtype=np.uint8)
        empty_source=np.full(sample_count,-1,dtype=np.int16)
        pm=np.zeros((sample_count,4),dtype=np.float64)
        pm[anchor_mask]=straight_srgb_rgba_u8_to_premultiplied_linear(
            rgba[np.argmax(valid[:,anchor_mask],axis=0),np.flatnonzero(anchor_mask)]
        )
        solved,stats=solve_weighted_surface_dirichlet(
            values=pm,
            known_mask=anchor_mask,
            sample_component=component,
            positions=positions,
            graph=surface_graph,
        )
        if not np.array_equal(solved[anchor_mask],pm[anchor_mask]):
            raise QualificationError(
                "CAA_CANONICAL_COMPLETION_SOURCE_MUTATION"
            )
        return CanonicalUnseenCompletion(
            rgba=empty_rgba,
            globally_unseen_mask=globally_unseen,
            anchor_mask=anchor_mask,
            anchor_source_view=empty_source,
            solver_stats=stats,
        )

    best_view=np.full(sample_count,-1,dtype=np.int16)
    best_score=np.full(sample_count,-np.inf,dtype=np.float64)
    for view in range(8):
        score=support[view,sample_face]
        improve=valid[view]&(score>best_score+1.0e-12)
        if np.any(improve):
            best_score[improve]=score[improve]
            best_view[improve]=view
    if np.any(anchor_mask&(best_view<0)):
        raise QualificationError(
            "CAA_CANONICAL_COMPLETION_ANCHOR_DONOR_MISSING"
        )

    anchor_ids=np.flatnonzero(anchor_mask)
    donors=best_view[anchor_ids].astype(np.int64)
    anchor_rgba=rgba[donors,anchor_ids]
    pm=np.zeros((sample_count,4),dtype=np.float64)
    pm[anchor_ids]=straight_srgb_rgba_u8_to_premultiplied_linear(
        anchor_rgba
    )

    solved,stats=solve_weighted_surface_dirichlet(
        values=pm,
        known_mask=anchor_mask,
        sample_component=component,
        positions=positions,
        graph=surface_graph,
    )
    if not np.array_equal(solved[anchor_mask],pm[anchor_mask]):
        raise QualificationError(
            "CAA_CANONICAL_COMPLETION_SOURCE_MUTATION"
        )
    solved=np.clip(solved,0.0,1.0)
    solved[:,:3]=np.minimum(solved[:,:3],solved[:,3:4])
    encoded=premultiplied_linear_to_straight_srgb_u8(solved)

    output=np.zeros((sample_count,4),dtype=np.uint8)
    output[globally_unseen]=encoded[globally_unseen]
    if np.any(output[anchor_mask]!=0):
        raise QualificationError(
            "CAA_CANONICAL_COMPLETION_WROTE_SOURCE_BACKED_SAMPLE"
        )
    return CanonicalUnseenCompletion(
        rgba=output,
        globally_unseen_mask=globally_unseen,
        anchor_mask=anchor_mask,
        anchor_source_view=best_view,
        solver_stats=stats,
    )
