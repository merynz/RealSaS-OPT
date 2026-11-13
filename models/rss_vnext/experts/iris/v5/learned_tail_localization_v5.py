from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from .indexed_sparse_tetra_decoder_v5 import (
    SparseRegularTetraPolicyV5,
    _carrier_chunk,
    fine_positions_from_gids_v5,
)


@dataclass(frozen=True)
class LearnedTailLocalizationPolicyV5:
    """Frozen, training-blind diagnostics for a selected learned V5 field."""

    top_residual_fraction: float = 0.01
    spatial_resolutions: tuple[int, ...] = (32, 64, 128)
    target_abs_band_edges: tuple[float, ...] = (
        2.0 / 1024.0,
        2.0 / 512.0,
        math.sqrt(3.0) * (2.0 / 512.0),
        2.0 * math.sqrt(3.0) * (2.0 / 512.0),
        0.03,
    )

    def validate(self) -> None:
        f=float(self.top_residual_fraction)
        if not math.isfinite(f) or not (0.0 < f <= 1.0):
            raise ValueError("top_residual_fraction must be in (0,1]")
        if not self.spatial_resolutions or any(int(r) <= 1 for r in self.spatial_resolutions):
            raise ValueError("spatial_resolutions must contain integers > 1")
        e=tuple(float(x) for x in self.target_abs_band_edges)
        if not e or any(not math.isfinite(x) or x <= 0.0 for x in e):
            raise ValueError("target_abs_band_edges must be finite and positive")
        if any(b <= a for a,b in zip(e,e[1:])):
            raise ValueError("target_abs_band_edges must be strictly increasing")


def _points(points: np.ndarray) -> np.ndarray:
    q=np.asarray(points,dtype=np.float32)
    if q.ndim != 2 or q.shape[1] != 3 or len(q) == 0:
        raise ValueError("points must be non-empty [N,3]")
    if not np.isfinite(q).all() or np.any(q < -1.000001) or np.any(q > 1.000001):
        raise ValueError("points must be finite and normalized to [-1,1]")
    return q


def target_abs_band_counts_v5(
    target: np.ndarray,
    *,
    edges: tuple[float, ...],
) -> list[dict[str, Any]]:
    t=np.abs(np.asarray(target,dtype=np.float64).reshape(-1))
    if len(t)==0 or not np.isfinite(t).all():
        raise ValueError("target must be finite and non-empty")
    e=np.asarray(edges,dtype=np.float64)
    if e.ndim!=1 or len(e)==0 or np.any(~np.isfinite(e)) or np.any(e<=0) or np.any(np.diff(e)<=0):
        raise ValueError("invalid target magnitude band edges")
    rows=[]
    lo=0.0
    for hi in e:
        mask=(t>=lo)&(t<hi)
        rows.append({
            "low_inclusive":float(lo),
            "high_exclusive":float(hi),
            "count":int(mask.sum()),
            "fraction":float(mask.mean()),
        })
        lo=float(hi)
    mask=t>=lo
    rows.append({
        "low_inclusive":float(lo),
        "high_exclusive":None,
        "count":int(mask.sum()),
        "fraction":float(mask.mean()),
    })
    return rows


def deterministic_top_residual_indices_v5(
    predicted: np.ndarray,
    target: np.ndarray,
    *,
    fraction: float,
) -> np.ndarray:
    p=np.asarray(predicted,dtype=np.float64).reshape(-1)
    t=np.asarray(target,dtype=np.float64).reshape(-1)
    if p.shape != t.shape or len(p)==0 or not np.isfinite(p).all() or not np.isfinite(t).all():
        raise ValueError("predicted and target must match, be finite, and be non-empty")
    f=float(fraction)
    if not math.isfinite(f) or not (0.0 < f <= 1.0):
        raise ValueError("fraction must be in (0,1]")
    n=max(1,int(math.ceil(len(p)*f)))
    err=np.abs(p-t)
    idx=np.arange(len(err),dtype=np.int64)
    order=np.lexsort((idx,-err))
    return idx[order[:n]]


def sign_disagreement_indices_v5(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    p=np.asarray(predicted,dtype=np.float64).reshape(-1)
    t=np.asarray(target,dtype=np.float64).reshape(-1)
    if p.shape != t.shape or len(p)==0 or not np.isfinite(p).all() or not np.isfinite(t).all():
        raise ValueError("predicted and target must match, be finite, and be non-empty")
    # Same exact convention as direct_fstar_metrics_v5.
    bad=(p>=0.0)!=(t>=0.0)
    return np.flatnonzero(bad).astype(np.int64)


def uniform_spatial_bin_ids_v5(points: np.ndarray, *, resolution: int) -> np.ndarray:
    q=_points(points).astype(np.float64)
    r=int(resolution)
    if r<=1:
        raise ValueError("resolution must be > 1")
    xyz=np.floor((np.clip(q,-1.0,1.0)+1.0)*0.5*r).astype(np.int64)
    xyz=np.clip(xyz,0,r-1)
    return ((xyz[:,0]*r+xyz[:,1])*r+xyz[:,2]).astype(np.int64)


def spatial_concentration_v5(
    points: np.ndarray,
    *,
    resolutions: tuple[int, ...]=(32,64,128),
) -> dict[str, Any]:
    q=_points(points)
    out={}
    for rr in resolutions:
        r=int(rr)
        ids=uniform_spatial_bin_ids_v5(q,resolution=r)
        _,counts=np.unique(ids,return_counts=True)
        counts=np.sort(counts.astype(np.int64))[::-1]
        probs=counts.astype(np.float64)/float(len(q))
        entropy=float(-(probs*np.log(np.maximum(probs,1e-300))).sum())
        norm_entropy=float(entropy/math.log(len(counts))) if len(counts)>1 else 0.0
        def top_share(k:int)->float:
            return float(counts[:min(k,len(counts))].sum()/len(q))
        out[str(r)]={
            "point_count":int(len(q)),
            "occupied_bin_count":int(len(counts)),
            "possible_bin_count":int(r**3),
            "occupied_fraction":float(len(counts)/(r**3)),
            "maximum_bin_count":int(counts[0]),
            "maximum_bin_share":float(counts[0]/len(q)),
            "top10_bin_share":top_share(10),
            "top100_bin_share":top_share(100),
            "normalized_occupied_bin_entropy":norm_entropy,
        }
    return out


def heldout_tail_report_v5(
    points: np.ndarray,
    target: np.ndarray,
    predicted: np.ndarray,
    *,
    policy: LearnedTailLocalizationPolicyV5=LearnedTailLocalizationPolicyV5(),
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    policy.validate()
    q=_points(points)
    t=np.asarray(target,dtype=np.float32).reshape(-1)
    p=np.asarray(predicted,dtype=np.float32).reshape(-1)
    if len(q)!=len(t) or p.shape!=t.shape or not np.isfinite(t).all() or not np.isfinite(p).all():
        raise ValueError("heldout arrays must match and be finite")
    residual=np.abs(p.astype(np.float64)-t.astype(np.float64))
    sign_idx=sign_disagreement_indices_v5(p,t)
    top_idx=deterministic_top_residual_indices_v5(p,t,fraction=policy.top_residual_fraction)
    report={
        "count":int(len(q)),
        "mae":float(residual.mean()),
        "p95_abs":float(np.quantile(residual,0.95)),
        "maximum_abs":float(residual.max()),
        "sign_disagreement_count":int(len(sign_idx)),
        "sign_disagreement_fraction":float(len(sign_idx)/len(q)),
        "top_residual_fraction":float(policy.top_residual_fraction),
        "top_residual_count":int(len(top_idx)),
        "sign_error_target_abs_bands":target_abs_band_counts_v5(t[sign_idx],edges=policy.target_abs_band_edges) if len(sign_idx) else [],
        "top_residual_target_abs_bands":target_abs_band_counts_v5(t[top_idx],edges=policy.target_abs_band_edges),
        "spatial_all":spatial_concentration_v5(q,resolutions=policy.spatial_resolutions),
        "spatial_sign_errors":spatial_concentration_v5(q[sign_idx],resolutions=policy.spatial_resolutions) if len(sign_idx) else {},
        "spatial_top_residual":spatial_concentration_v5(q[top_idx],resolutions=policy.spatial_resolutions),
    }
    arrays={
        "sign_error_indices":sign_idx,
        "top_residual_indices":top_idx,
        "abs_residual":residual.astype(np.float32),
    }
    return report,arrays


def boundary_crossing_tetra_localization_v5(
    refined_cells: np.ndarray,
    fine_gids: np.ndarray,
    fine_scalar: np.ndarray,
    boundary_parent: np.ndarray,
    *,
    policy: SparseRegularTetraPolicyV5,
) -> dict[str, np.ndarray]:
    """Return exact locations of sparse-shell boundary-crossing tetrahedra.

    This is diagnostic-only. It does not alter refinement, MT topology, or any
    qualification gate.
    """
    policy.validate()
    refined=np.asarray(refined_cells,dtype=np.int32)
    gids=np.asarray(fine_gids,dtype=np.int64).reshape(-1)
    scalar=np.asarray(fine_scalar,dtype=np.float32).reshape(-1)
    boundary=np.asarray(boundary_parent,dtype=bool).reshape(-1)
    if refined.ndim!=2 or refined.shape[1]!=3 or len(refined)==0:
        raise ValueError("refined_cells must be non-empty [N,3]")
    if len(boundary)!=len(refined):
        raise ValueError("boundary_parent must align with refined_cells")
    rows=[]
    for start in range(0,len(refined),int(policy.cell_chunk)):
        end=min(len(refined),start+int(policy.cell_chunk))
        tet_local,tet_scalar,codes=_carrier_chunk(
            refined,gids,scalar,start=start,end=end,policy=policy
        )
        valid=(codes>0)&(codes<15)
        parent_boundary=np.repeat(boundary[start:end],8*6)
        ids=np.flatnonzero(valid & parent_boundary)
        if len(ids)==0:
            continue
        parent_local=ids//48
        rem=ids%48
        subcube=rem//6
        tetra=rem%6
        parent_index=(start+parent_local).astype(np.int64)
        local_vertices=tet_local[ids].astype(np.int64)
        tet_gids=gids[local_vertices]
        tet_pos=fine_positions_from_gids_v5(
            tet_gids.reshape(-1),fine_cells=int(policy.fine_cells)
        ).reshape(-1,4,3)
        rows.append({
            "parent_index":parent_index,
            "parent_cell":refined[parent_index].astype(np.int32),
            "subcube_index":subcube.astype(np.int8),
            "tetra_index":tetra.astype(np.int8),
            "case_code":codes[ids].astype(np.int8),
            "tet_vertex_gids":tet_gids.astype(np.int64),
            "tet_vertex_scalar":tet_scalar[ids].astype(np.float32),
            "tet_center_normalized":tet_pos.mean(axis=1).astype(np.float32),
        })
    if not rows:
        return {
            "parent_index":np.empty((0,),dtype=np.int64),
            "parent_cell":np.empty((0,3),dtype=np.int32),
            "subcube_index":np.empty((0,),dtype=np.int8),
            "tetra_index":np.empty((0,),dtype=np.int8),
            "case_code":np.empty((0,),dtype=np.int8),
            "tet_vertex_gids":np.empty((0,4),dtype=np.int64),
            "tet_vertex_scalar":np.empty((0,4),dtype=np.float32),
            "tet_center_normalized":np.empty((0,3),dtype=np.float32),
        }
    keys=rows[0].keys()
    return {k:np.concatenate([r[k] for r in rows],axis=0) for k in keys}
