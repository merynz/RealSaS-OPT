from __future__ import annotations

"""Canonical deterministic completion for surface samples unseen in every view.

Source-backed appearance remains directional A(p,v). Only samples with no
qualified source support in any input direction are eligible for the
view-independent canonical completion C(p).

The solver operates on a bounded control lattice. A separate deterministic
piecewise-linear prolongation may evaluate that canonical field on denser
per-face CAA lattices. Calling these primitives does not itself promote a
shipping policy.
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
    solved_pm_linear: np.ndarray
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
            "field_storage": "PREMULTIPLIED_LINEAR_RGBA_FLOAT64",
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
        solved_pm_linear=np.asarray(solved,dtype=np.float64),
        globally_unseen_mask=globally_unseen,
        anchor_mask=anchor_mask,
        anchor_source_view=best_view,
        solver_stats=stats,
    )


def _triangle_lattice_offset(resolution: int, j: np.ndarray) -> np.ndarray:
    jj=np.asarray(j,dtype=np.int64)
    return jj*int(resolution)-(jj*(jj-1))//2


def _prolongation_stencil(
    *,
    target_resolution: int,
    control_resolution: int,
) -> tuple[np.ndarray,np.ndarray]:
    """Map one triangular face lattice onto a coarser triangular control field."""
    target=int(target_resolution)
    control=int(control_resolution)
    if target<4 or control<2:
        raise QualificationError("CAA_CANONICAL_PROLONGATION_RESOLUTION_INVALID")
    denom=float(target-1)
    control_denom=float(control-1)
    rows=[]
    for j in range(target):
        for i in range(target-j):
            u=float(i)/denom
            v=float(j)/denom
            a=u*control_denom
            b=v*control_denom
            p=min(int(np.floor(a+1.0e-12)),control-1)
            q=min(int(np.floor(b+1.0e-12)),control-1)
            x=float(a-p)
            y=float(b-q)
            if p+q>=control-1:
                # Boundary points belong to a control-lattice vertex/edge.
                if p+q>control-1:
                    raise QualificationError(
                        "CAA_CANONICAL_PROLONGATION_OUTSIDE_SIMPLEX"
                    )
                candidates=((p,q),(p,q),(p,q))
                weights=(1.0,0.0,0.0)
            elif x+y<=1.0+1.0e-12:
                candidates=((p,q),(p+1,q),(p,q+1))
                weights=(1.0-x-y,x,y)
            else:
                candidates=((p+1,q),(p,q+1),(p+1,q+1))
                weights=(1.0-y,1.0-x,x+y-1.0)
            local=[]
            for ci,cj in candidates:
                if ci<0 or cj<0 or ci+cj>=control:
                    raise QualificationError(
                        "CAA_CANONICAL_PROLONGATION_CONTROL_INDEX_INVALID"
                    )
                local.append(
                    int(_triangle_lattice_offset(
                        control,np.asarray([cj],dtype=np.int64)
                    )[0]+ci)
                )
            w=np.asarray(weights,dtype=np.float64)
            w[np.abs(w)<1.0e-14]=0.0
            if np.any(w< -1.0e-12) or abs(float(np.sum(w))-1.0)>1.0e-10:
                raise QualificationError(
                    "CAA_CANONICAL_PROLONGATION_WEIGHT_INVALID"
                )
            rows.append((tuple(local),tuple(map(float,w))))
    index=np.asarray([row[0] for row in rows],dtype=np.int64)
    weight=np.asarray([row[1] for row in rows],dtype=np.float64)
    return index,weight


def prolongate_control_pm_to_adaptive_faces(
    *,
    control_pm_linear: np.ndarray,
    face_tile_resolutions: np.ndarray,
    control_resolution: int=4,
) -> np.ndarray:
    """Evaluate a face-local control field on the exact adaptive CAA lattices.

    This operation never crosses face/component topology and never invents a
    second surface. It is ordinary piecewise-linear finite-element
    prolongation in barycentric coordinates.
    """
    field=np.asarray(control_pm_linear,dtype=np.float64)
    resolutions=np.asarray(face_tile_resolutions,dtype=np.int32)
    control=int(control_resolution)
    if resolutions.ndim!=1 or len(resolutions)<=0 or np.any(resolutions<4):
        raise QualificationError("CAA_CANONICAL_PROLONGATION_FACE_RESOLUTION_INVALID")
    control_per_face=control*(control+1)//2
    if field.shape!=(len(resolutions)*control_per_face,4):
        raise QualificationError("CAA_CANONICAL_PROLONGATION_CONTROL_FIELD_SHAPE")
    counts=(
        resolutions.astype(np.int64)
        *(resolutions.astype(np.int64)+1)//2
    )
    offsets=np.zeros((len(resolutions)+1,),dtype=np.int64)
    offsets[1:]=np.cumsum(counts,dtype=np.int64)
    out=np.empty((int(offsets[-1]),4),dtype=np.float64)
    cache:dict[int,tuple[np.ndarray,np.ndarray]]={}
    for face_index,resolution_raw in enumerate(resolutions):
        resolution=int(resolution_raw)
        stencil=cache.get(resolution)
        if stencil is None:
            stencil=_prolongation_stencil(
                target_resolution=resolution,
                control_resolution=control,
            )
            cache[resolution]=stencil
        index,weight=stencil
        c0=face_index*control_per_face
        control_face=field[c0:c0+control_per_face]
        values=control_face[index]
        dense=np.sum(values*weight[:,:,None],axis=1)
        d0=int(offsets[face_index])
        d1=int(offsets[face_index+1])
        if dense.shape!=(d1-d0,4):
            raise QualificationError(
                "CAA_CANONICAL_PROLONGATION_SAMPLE_ACCOUNTING_DRIFT"
            )
        out[d0:d1]=dense
    out=np.clip(out,0.0,1.0)
    out[:,:3]=np.minimum(out[:,:3],out[:,3:4])
    if not np.isfinite(out).all():
        raise QualificationError("CAA_CANONICAL_PROLONGATION_NONFINITE")
    return out
