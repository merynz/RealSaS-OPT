from __future__ import annotations

"""Bounded continuous edge--edge tests for linear canonical trajectories.

Following the coplanarity-cubic candidate construction of Bridson, Fedkiw
and Anderson (SIGGRAPH 2002). This implementation is double-precision and
conservatively rejects identically coplanar/parallel ambiguous sweeps.
It is NOT exact-arithmetic CCD, nonlinear trajectory CCD or coplanar CCD.
"""

from dataclasses import dataclass
import math

import numpy as np

from .mechanics_xpbd_surface_self_contact_v1 import _closest_edges
from .types import QualificationError

Vec = np.ndarray


@dataclass(frozen=True)
class XPBDSweptEdgePairContactV1:
    cloth_edge_parameter: float
    body_edge_parameter: float
    normal: tuple[float,float,float]
    required_separation: float
    endpoint_penetration: float
    swept_hits: int


def _crossing_cubic_roots_v1(
    a0:Vec,b0:Vec,c0:Vec,d0:Vec,
    a1:Vec,b1:Vec,c1:Vec,d1:Vec,
) -> tuple[float,...]:
    """Roots of (a-c) dot ((b-a) cross (d-c)) for linear endpoints."""
    r0=a0-c0
    dr=(a1-a0)-(c1-c0)
    u0=b0-a0
    du=(b1-b0)-(a1-a0)
    v0=d0-c0
    dv=(d1-d0)-(c1-c0)
    cross0=np.cross(u0,v0)
    cross1=np.cross(du,v0)+np.cross(u0,dv)
    cross2=np.cross(du,dv)
    coeff=np.array((
        float(dr@cross2),
        float(r0@cross2+dr@cross1),
        float(r0@cross1+dr@cross0),
        float(r0@cross0),
    ),dtype=np.float64)
    if not np.isfinite(coeff).all():
        raise QualificationError("XPBD_BODY_EDGE_CCD_COEFFICIENT_NONFINITE")
    scale=float(np.max(np.abs(coeff)))
    if scale<=1e-13:
        raise QualificationError("XPBD_BODY_EDGE_COPLANAR_SWEEP_UNSUPPORTED")
    first=next((i for i,x in enumerate(coeff) if abs(x)>scale*1e-12),3)
    roots=np.roots(coeff[first:])
    hits=[]
    for root in roots:
        if abs(float(np.imag(root)))>1e-7:
            continue
        t=float(np.real(root))
        if not 1e-8<t<1.-1e-8:
            continue
        a=a0+t*(a1-a0)
        b=b0+t*(b1-b0)
        c=c0+t*(c1-c0)
        d=d0+t*(d1-d0)
        x,y,s,q=_closest_edges(a,b,c,d)
        length=max(1.,float(np.linalg.norm(b-a)),float(np.linalg.norm(d-c)))
        if float(np.linalg.norm(x-y))<=1e-7*length:
            hits.append(t)
    return tuple(sorted(hits))


def evaluate_swept_body_edge_pair_v1(
    cloth_start:tuple[Vec,Vec],
    cloth_end:tuple[Vec,Vec],
    body_start:tuple[Vec,Vec],
    body_end:tuple[Vec,Vec],
    *,thickness:float,check_sweeps:bool=True,
) -> XPBDSweptEdgePairContactV1 | None:
    if not math.isfinite(thickness) or thickness<=0.:
        raise QualificationError("XPBD_BODY_EDGE_THICKNESS_INVALID")
    arrays=(*cloth_start,*cloth_end,*body_start,*body_end)
    if any(np.asarray(v).shape!=(3,) or not np.isfinite(v).all() for v in arrays):
        raise QualificationError("XPBD_BODY_EDGE_VERTEX_NONFINITE")
    a0,b0=cloth_start
    a1,b1=cloth_end
    c0,d0=body_start
    c1,d1=body_end
    first,second,s,q=_closest_edges(a1,b1,c1,d1)
    difference=first-second
    distance=float(np.linalg.norm(difference))
    sweep_times=(
        _crossing_cubic_roots_v1(a0,b0,c0,d0,a1,b1,c1,d1)
        if check_sweeps else ()
    )
    if distance>=thickness and not sweep_times:
        return None
    cross_start=np.cross(b0-a0,d0-c0)
    cross_end=np.cross(b1-a1,d1-c1)
    norm_start=float(np.linalg.norm(cross_start))
    norm_end=float(np.linalg.norm(cross_end))
    if norm_end<=1e-12:
        # Do not invent a sided normal for coincident/parallel edges.
        if distance<=1e-12 or sweep_times:
            raise QualificationError("XPBD_BODY_EDGE_PARALLEL_NORMAL_UNSUPPORTED")
        normal=difference/distance
        needed=max(0.,thickness-distance)
    elif sweep_times:
        if norm_start<=1e-12:
            raise QualificationError("XPBD_BODY_EDGE_INITIAL_NORMAL_UNDEFINED")
        n0=cross_start/norm_start
        n1=cross_end/norm_end
        if float(n0@n1)<=0.:
            raise QualificationError("XPBD_BODY_EDGE_MOVING_NORMAL_FLIP")
        incoming=1. if float((a0-c0)@n0)>=0. else -1.
        normal=incoming*n1
        signed=float(difference@normal)
        needed=max(0.,thickness-signed)
    else:
        normal=difference/distance if distance>1e-12 else cross_end/norm_end
        needed=max(0.,thickness-distance)
    return XPBDSweptEdgePairContactV1(
        cloth_edge_parameter=float(s),
        body_edge_parameter=float(q),
        normal=tuple(map(float,normal)),
        required_separation=float(needed),
        endpoint_penetration=max(0.,thickness-distance),
        swept_hits=len(sweep_times),
    )
