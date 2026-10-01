from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from .indexed_sparse_tetra_decoder_v5 import SparseRegularTetraPolicyV5


FACE_NAMES_V5 = ("X_NEG","X_POS","Y_NEG","Y_POS","Z_NEG","Z_POS")
FACE_OFFSETS_V5 = np.asarray(
    [[-1,0,0],[1,0,0],[0,-1,0],[0,1,0],[0,0,-1],[0,0,1]],
    dtype=np.int64,
)


@dataclass(frozen=True)
class ShellEscapeDiagnosticPolicyV5:
    """Diagnostic-only decomposition of conservative boundary-parent failures."""

    zero_epsilon: float = 0.0

    def validate(self) -> None:
        e=float(self.zero_epsilon)
        if not math.isfinite(e) or e < 0.0:
            raise ValueError("zero_epsilon must be finite and non-negative")


def refined_membership_keys_v5(
    refined_cells: np.ndarray,
    *,
    base_cells: int,
) -> np.ndarray:
    c=np.asarray(refined_cells,dtype=np.int64)
    b=int(base_cells)
    if c.ndim!=2 or c.shape[1]!=3 or len(c)==0:
        raise ValueError("refined_cells must be non-empty [N,3]")
    if np.any(c<0) or np.any(c>=b):
        raise ValueError("refined_cells outside base grid")
    keys=((c[:,0]*b+c[:,1])*b+c[:,2]).astype(np.int64)
    if np.any(np.diff(keys)<0):
        raise ValueError("refined_cells must be deterministic sorted order")
    if len(np.unique(keys))!=len(keys):
        raise ValueError("refined_cells contains duplicates")
    return keys


def missing_neighbor_faces_v5(
    refined_cells: np.ndarray,
    *,
    base_cells: int,
) -> np.ndarray:
    """[N,6] True where a base-cell face has no refined in-domain neighbor.

    Domain boundary is *not* called a missing neighbor because there is no omitted
    in-domain cell across that face.
    """

    c=np.asarray(refined_cells,dtype=np.int64)
    b=int(base_cells)
    keys=refined_membership_keys_v5(c,base_cells=b)
    out=np.zeros((len(c),6),dtype=bool)
    for fi,off in enumerate(FACE_OFFSETS_V5):
        nn=c+off
        in_domain=((nn>=0)&(nn<b)).all(axis=1)
        nkeys=((nn[:,0]*b+nn[:,1])*b+nn[:,2]).astype(np.int64)
        pos=np.searchsorted(keys,nkeys)
        present=np.zeros(len(c),dtype=bool)
        valid=in_domain&(pos<len(keys))
        ids=np.flatnonzero(valid)
        if len(ids):
            present[ids]=keys[pos[ids]]==nkeys[ids]
        out[:,fi]=in_domain & (~present)
    return out


def _fine_gid_from_ijk(ijk: np.ndarray, *, fine_cells: int) -> np.ndarray:
    p=np.asarray(ijk,dtype=np.int64)
    n=int(fine_cells)+1
    if p.ndim!=2 or p.shape[1]!=3:
        raise ValueError("ijk must be [N,3]")
    if np.any(p<0) or np.any(p>int(fine_cells)):
        raise ValueError("fine-grid ijk outside domain")
    return ((p[:,0]*n+p[:,1])*n+p[:,2]).astype(np.int64)


def parent_face_fine_ijk_v5(parent_cell: np.ndarray, face_index: int) -> np.ndarray:
    """Return the 3x3 fine-grid vertices on one base-cell face."""

    c=np.asarray(parent_cell,dtype=np.int64).reshape(3)
    fi=int(face_index)
    if fi<0 or fi>=6:
        raise ValueError("face_index must be in [0,5]")
    origin=2*c
    rows=[]
    if fi in (0,1):
        x=origin[0]+(0 if fi==0 else 2)
        for y in range(origin[1],origin[1]+3):
            for z in range(origin[2],origin[2]+3):
                rows.append((x,y,z))
    elif fi in (2,3):
        y=origin[1]+(0 if fi==2 else 2)
        for x in range(origin[0],origin[0]+3):
            for z in range(origin[2],origin[2]+3):
                rows.append((x,y,z))
    else:
        z=origin[2]+(0 if fi==4 else 2)
        for x in range(origin[0],origin[0]+3):
            for y in range(origin[1],origin[1]+3):
                rows.append((x,y,z))
    return np.asarray(rows,dtype=np.int64)


def face_zero_crossing_from_scalar_v5(
    values_3x3: np.ndarray,
    *,
    epsilon: float=0.0,
) -> bool:
    """Exact sign-mix test over the four fine squares covering one parent face.

    The MT boundary face is piecewise linear over a triangulation of these 2x2
    fine squares. Any square containing both non-positive and non-negative vertex
    values has a zero set on that square under the linear face interpolation.
    """

    v=np.asarray(values_3x3,dtype=np.float64)
    if v.shape!=(3,3) or not np.isfinite(v).all():
        raise ValueError("values_3x3 must be finite [3,3]")
    e=float(epsilon)
    if not math.isfinite(e) or e<0:
        raise ValueError("epsilon must be finite and non-negative")
    for i in range(2):
        for j in range(2):
            s=v[i:i+2,j:j+2]
            if float(np.min(s)) <= e and float(np.max(s)) >= -e:
                # Exclude a uniformly same-sign square when epsilon > 0.
                if (float(np.min(s)) < -e and float(np.max(s)) > e) or np.any(np.abs(s)<=e):
                    return True
    return False


def _reshape_face_values(face_index: int, values: np.ndarray) -> np.ndarray:
    v=np.asarray(values,dtype=np.float64).reshape(9)
    # parent_face_fine_ijk_v5 orders the two free axes in row-major order.
    return v.reshape(3,3)


def exact_missing_face_escape_report_v5(
    refined_cells: np.ndarray,
    fine_gids: np.ndarray,
    fine_scalar: np.ndarray,
    *,
    policy: SparseRegularTetraPolicyV5,
    diagnostic_policy: ShellEscapeDiagnosticPolicyV5=ShellEscapeDiagnosticPolicyV5(),
) -> tuple[dict[str,Any],dict[str,np.ndarray]]:
    """Separate conservative boundary-parent status from actual omitted-face zero crossing."""

    policy.validate(); diagnostic_policy.validate()
    refined=np.asarray(refined_cells,dtype=np.int32)
    gids=np.asarray(fine_gids,dtype=np.int64).reshape(-1)
    scalar=np.asarray(fine_scalar,dtype=np.float32).reshape(-1)
    if scalar.shape!=gids.shape or len(gids)==0 or np.any(np.diff(gids)<=0):
        raise ValueError("fine_gids/fine_scalar must be matching sorted vectors")

    missing=missing_neighbor_faces_v5(refined,base_cells=int(policy.base_cells))
    parent_ids=[]; face_ids=[]; neighbor_cells=[]; min_abs=[]; face_min=[]; face_max=[]
    for pi in np.flatnonzero(missing.any(axis=1)):
        parent=refined[pi].astype(np.int64)
        for fi in np.flatnonzero(missing[pi]):
            ijk=parent_face_fine_ijk_v5(parent,int(fi))
            fgids=_fine_gid_from_ijk(ijk,fine_cells=int(policy.fine_cells))
            pos=np.searchsorted(gids,fgids)
            if np.any(pos>=len(gids)) or np.any(gids[pos]!=fgids):
                raise RuntimeError("MISSING_FACE_FINE_VERTEX_LOOKUP_MISS")
            vals=scalar[pos]
            grid=_reshape_face_values(int(fi),vals)
            if face_zero_crossing_from_scalar_v5(
                grid,epsilon=float(diagnostic_policy.zero_epsilon)
            ):
                parent_ids.append(int(pi))
                face_ids.append(int(fi))
                neighbor_cells.append((parent+FACE_OFFSETS_V5[int(fi)]).astype(np.int32))
                min_abs.append(float(np.min(np.abs(vals))))
                face_min.append(float(np.min(vals)))
                face_max.append(float(np.max(vals)))

    parent_ids=np.asarray(parent_ids,dtype=np.int64)
    face_ids=np.asarray(face_ids,dtype=np.int8)
    neighbor_cells=np.asarray(neighbor_cells,dtype=np.int32).reshape(-1,3) if len(neighbor_cells) else np.empty((0,3),dtype=np.int32)
    escape_parent_unique=np.unique(parent_ids) if len(parent_ids) else np.empty((0,),dtype=np.int64)
    report={
        "refined_cell_count":int(len(refined)),
        "boundary_parent_cell_count_conservative":int(missing.any(axis=1).sum()),
        "missing_in_domain_neighbor_face_count":int(missing.sum()),
        "actual_missing_face_zero_crossing_count":int(len(face_ids)),
        "actual_escape_parent_cell_count":int(len(escape_parent_unique)),
        "conservative_boundary_parent_false_positive_count":int(
            missing.any(axis=1).sum()-len(escape_parent_unique)
        ),
        "actual_escape_fraction_of_conservative_boundary_parents":float(
            len(escape_parent_unique)/max(1,int(missing.any(axis=1).sum()))
        ),
        "face_name_counts":{
            FACE_NAMES_V5[i]:int(np.count_nonzero(face_ids==i)) for i in range(6)
        },
    }
    arrays={
        "missing_face_matrix":missing.astype(np.uint8),
        "escape_parent_indices":parent_ids,
        "escape_parent_cells":refined[parent_ids].astype(np.int32) if len(parent_ids) else np.empty((0,3),dtype=np.int32),
        "escape_face_indices":face_ids,
        "escape_neighbor_cells":neighbor_cells,
        "escape_face_min_abs":np.asarray(min_abs,dtype=np.float32),
        "escape_face_min":np.asarray(face_min,dtype=np.float32),
        "escape_face_max":np.asarray(face_max,dtype=np.float32),
    }
    return report,arrays


def coarse_cell_corner_values_v5(coarse_field: np.ndarray, cells: np.ndarray) -> np.ndarray:
    f=np.asarray(coarse_field)
    c=np.asarray(cells,dtype=np.int64)
    if c.ndim!=2 or c.shape[1]!=3:
        raise ValueError("cells must be [N,3]")
    out=np.empty((len(c),8),dtype=np.float32)
    offsets=np.asarray(
        [(0,0,0),(0,0,1),(0,1,0),(0,1,1),(1,0,0),(1,0,1),(1,1,0),(1,1,1)],
        dtype=np.int64,
    )
    for k,off in enumerate(offsets):
        p=c+off
        out[:,k]=f[p[:,0],p[:,1],p[:,2]]
    return out


def refine_rule_for_corner_values_v5(
    corner_values: np.ndarray,
    *,
    refine_band: float,
) -> np.ndarray:
    v=np.asarray(corner_values,dtype=np.float64)
    if v.ndim!=2 or v.shape[1]!=8 or not np.isfinite(v).all():
        raise ValueError("corner_values must be finite [N,8]")
    lo=v.min(axis=1); hi=v.max(axis=1); ma=np.abs(v).min(axis=1)
    return ((lo<=0.0)&(hi>=0.0)) | (ma<=float(refine_band))
