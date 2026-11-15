from __future__ import annotations

"""Deterministic canonical-mesh surface self-contact for declared XPBD patches.

Two-sided vertex--triangle and non-adjacent edge--edge constraints use the
same addressed particles, inverse masses and fixed-tick XPBD lambda state.
This is discrete thickness contact, not continuous collision detection (CCD).
No PartVerse label, material coefficient or source annotation activates it.
"""

from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np

from .types import QualificationError

Vec3 = np.ndarray
Edge = tuple[str, str]
Face = tuple[str, str, str]


@dataclass(frozen=True)
class SurfaceSelfContactTopologyV1:
    faces: tuple[Face, ...]
    edges: tuple[Edge, ...]
    one_ring: dict[str, frozenset[str]]


@dataclass(frozen=True)
class SurfaceSelfContactStatsV1:
    projections: int
    candidate_pairs: int
    maximum_penetration: float


def prepare_surface_self_contact_topology_v1(
    faces: tuple[Face, ...],
    edges: tuple[tuple[str, str, float], ...],
) -> SurfaceSelfContactTopologyV1:
    neighbors: dict[str, set[str]] = {}
    for a, b, _ in edges:
        neighbors.setdefault(a, set()).add(b)
        neighbors.setdefault(b, set()).add(a)
    return SurfaceSelfContactTopologyV1(
        faces=faces,
        edges=tuple((a, b) for a, b, _ in edges),
        one_ring={v: frozenset(ns) for v, ns in neighbors.items()},
    )


def _closest_triangle(
    p: Vec3, a: Vec3, b: Vec3, c: Vec3,
) -> tuple[Vec3, tuple[float, float, float]]:
    """Clamped barycentric closest point, including edge and corner regions."""
    ab, ac = b - a, c - a
    if float(np.dot(np.cross(ab, ac), np.cross(ab, ac))) <= 1e-24:
        raise QualificationError("XPBD_SELF_CONTACT_DEGENERATE_MOVING_FACE")
    ap = p - a
    d1, d2 = float(ab @ ap), float(ac @ ap)
    if d1 <= 0. and d2 <= 0.:
        return a, (1., 0., 0.)
    bp = p - b
    d3, d4 = float(ab @ bp), float(ac @ bp)
    if d3 >= 0. and d4 <= d3:
        return b, (0., 1., 0.)
    vc = d1*d4 - d3*d2
    if vc <= 0. and d1 >= 0. and d3 <= 0.:
        t = d1/(d1-d3)
        return a+t*ab, (1.-t, t, 0.)
    cp = p - c
    d5, d6 = float(ab @ cp), float(ac @ cp)
    if d6 >= 0. and d5 <= d6:
        return c, (0., 0., 1.)
    vb = d5*d2 - d1*d6
    if vb <= 0. and d2 >= 0. and d6 <= 0.:
        t = d2/(d2-d6)
        return a+t*ac, (1.-t, 0., t)
    va = d3*d6 - d5*d4
    if va <= 0. and (d4-d3) >= 0. and (d5-d6) >= 0.:
        t = (d4-d3)/((d4-d3)+(d5-d6))
        return b+t*(c-b), (0., 1.-t, t)
    denom = va+vb+vc
    if abs(denom) <= 1e-20:
        raise QualificationError("XPBD_SELF_CONTACT_BARYCENTRIC_DEGENERATE")
    v, w = vb/denom, vc/denom
    return a+ab*v+ac*w, (1.-v-w, v, w)


def _closest_edges(
    a: Vec3, b: Vec3, c: Vec3, d: Vec3,
) -> tuple[Vec3, Vec3, float, float]:
    """Robust finite-segment closest points; deterministic parallel fallback."""
    u, v, w = b-a, d-c, a-c
    aa, bb, cc = float(u@u), float(u@v), float(v@v)
    dd, ee = float(u@w), float(v@w)
    if aa <= 1e-20 or cc <= 1e-20:
        raise QualificationError("XPBD_SELF_CONTACT_COLLAPSED_EDGE")
    D = aa*cc-bb*bb
    if D <= 1e-20:
        s=0.
        t=np.clip(ee/cc, 0., 1.)
    else:
        s=float(np.clip((bb*ee-cc*dd)/D, 0., 1.))
        t=float(np.clip((bb*s+ee)/cc, 0., 1.))
        s=float(np.clip((bb*t-dd)/aa, 0., 1.))
        t=float(np.clip((bb*s+ee)/cc, 0., 1.))
    return a+s*u, c+t*v, s, t


def _cells(lo: Vec3, hi: Vec3, size: float, limit: int):
    x = np.floor(lo / size).astype(np.int64)
    y = np.floor(hi / size).astype(np.int64)
    lengths=(y-x+1).tolist()
    count=int(lengths[0]*lengths[1]*lengths[2])
    if count > limit:
        raise QualificationError("XPBD_SELF_CONTACT_SPATIAL_RESOURCE_BOUND")
    for i in range(int(x[0]), int(y[0])+1):
        for j in range(int(x[1]), int(y[1])+1):
            for k in range(int(x[2]), int(y[2])+1):
                yield (i,j,k)


def _pairs(
    positions: Mapping[str, Vec3],
    topology: SurfaceSelfContactTopologyV1,
    thickness: float, cell_size: float,
    max_pairs: int, max_cells: int,
) -> tuple[list[tuple[str, int]], list[tuple[int, int]]]:
    face_grid: dict[tuple[int,int,int], list[int]] = {}
    edge_grid: dict[tuple[int,int,int], list[int]] = {}
    for fi, face in enumerate(topology.faces):
        p = np.stack([positions[v] for v in face])
        for cell in _cells(p.min(axis=0)-thickness, p.max(axis=0)+thickness,
                           cell_size, 1024):
            face_grid.setdefault(cell, []).append(fi)
            if len(face_grid)>max_cells:
                raise QualificationError("XPBD_SELF_CONTACT_SPATIAL_RESOURCE_BOUND")
    for ei, edge in enumerate(topology.edges):
        p = np.stack([positions[v] for v in edge])
        for cell in _cells(p.min(axis=0)-thickness, p.max(axis=0)+thickness,
                           cell_size, 1024):
            edge_grid.setdefault(cell, []).append(ei)
            if len(edge_grid)+len(face_grid)>max_cells:
                raise QualificationError("XPBD_SELF_CONTACT_SPATIAL_RESOURCE_BOUND")

    point_face: set[tuple[str,int]]=set()
    for vid in sorted(positions):
        cell=tuple(np.floor(positions[vid]/cell_size).astype(np.int64))
        for fi in face_grid.get(cell, ()):
            face=topology.faces[fi]
            # One-ring topology must not self-collide at its own seam.
            # This includes a vertex directly belonging to the target face.
            if any(v==vid or v in topology.one_ring.get(vid, ())
                   for v in face):
                continue
            point_face.add((vid,fi))
            if len(point_face)>max_pairs:
                raise QualificationError("XPBD_SELF_CONTACT_CANDIDATE_RESOURCE_BOUND")

    edge_pairs: set[tuple[int,int]]=set()
    for ei, edge in enumerate(topology.edges):
        p=np.stack([positions[v] for v in edge])
        for cell in _cells(p.min(axis=0)-thickness,p.max(axis=0)+thickness,
                           cell_size,1024):
            for other in edge_grid.get(cell,()):
                if other<=ei:
                    continue
                a,b=topology.edges[other]
                if any(x==y or y in topology.one_ring.get(x,())
                       for x in edge for y in (a,b)):
                    continue
                edge_pairs.add((ei,other))
                if len(edge_pairs)+len(point_face)>max_pairs:
                    raise QualificationError("XPBD_SELF_CONTACT_CANDIDATE_RESOURCE_BOUND")
    return sorted(point_face), sorted(edge_pairs)


def _normal(diff: Vec3, fallback: Vec3) -> tuple[Vec3,float]:
    dist=float(np.linalg.norm(diff))
    if dist > 1e-12:
        return diff/dist, dist
    n=float(np.linalg.norm(fallback))
    if n<=1e-12:
        raise QualificationError("XPBD_SELF_CONTACT_UNDEFINED_NORMAL")
    return fallback/n, 0.


def _apply_unilateral(
    positions: dict[str, Vec3], gradients: list[tuple[str, Vec3]],
    mass: Mapping[str,float], C: float, alpha: float,
    accumulated: float,
) -> tuple[float, bool]:
    denominator=alpha+sum(mass[k]*float(g@g) for k,g in gradients)
    if denominator<=1e-18:
        # Two fixed patches may geometrically intersect: do not pretend to
        # move an immovable surface. Final admission evaluates the residual.
        return accumulated, False
    delta=max(-accumulated, -(C+alpha*accumulated)/denominator)
    if delta==0.:
        return accumulated, False
    for key,g in gradients:
        positions[key]=positions[key]+mass[key]*delta*g
    return accumulated+delta, True


def evaluate_surface_self_contacts_v1(
    positions: dict[str, Vec3],
    mass: Mapping[str, float],
    topology: SurfaceSelfContactTopologyV1,
    *, thickness: float, cell_size: float,
    compliance: float = 0., step_seconds: float = 1./120.,
    max_pairs: int = 16384, max_cells: int = 200000,
    lambdas: dict[tuple[str, int | str, int | None], float] | None = None,
    project: bool = False,
) -> SurfaceSelfContactStatsV1:
    """One deterministic broadphase/narrowphase/XPBD contact iteration.

    On project=True modifies the shared canonical positions in-place. A lambda
    map belongs only to the current fixed physics substep and is not serialized.
    """
    if thickness<=0.:
        return SurfaceSelfContactStatsV1(0,0,0.)
    point_pairs,edge_pairs=_pairs(
        positions,topology,thickness,cell_size,max_pairs,max_cells,
    )
    max_penetration=0.
    projections=0
    alpha=compliance/(step_seconds*step_seconds)
    accum=lambdas if lambdas is not None else {}
    for vid,fi in point_pairs:
        a,b,c=topology.faces[fi]
        nearest,bary=_closest_triangle(positions[vid],positions[a],positions[b],positions[c])
        normal,distance=_normal(positions[vid]-nearest,
                                np.cross(positions[b]-positions[a],positions[c]-positions[a]))
        C=distance-thickness
        max_penetration=max(max_penetration,max(0.,-C))
        if project and C < 0.:
            gradients=[(vid,normal)]+[
                (key,-weight*normal) for key,weight in zip((a,b,c),bary)
            ]
            key=("VT",vid,fi)
            accum[key],changed=_apply_unilateral(
                positions,gradients,mass,C,alpha,accum.get(key,0.),
            )
            projections+=int(changed)
    for ei,ej in edge_pairs:
        a,b=topology.edges[ei]
        c,d=topology.edges[ej]
        first,second,s,t=_closest_edges(
            positions[a],positions[b],positions[c],positions[d],
        )
        # If two edge segments coincide, use a stable normal perpendicular
        # to the first edge, with deterministic fallback in axis order.
        u=positions[b]-positions[a]
        fallback=np.cross(u,np.array((1.,0.,0.)))
        if float(np.linalg.norm(fallback))<=1e-12:
            fallback=np.cross(u,np.array((0.,1.,0.)))
        normal,distance=_normal(first-second,fallback)
        C=distance-thickness
        max_penetration=max(max_penetration,max(0.,-C))
        if project and C < 0.:
            gradients=[
                (a,(1.-s)*normal),(b,s*normal),
                (c,-(1.-t)*normal),(d,-t*normal),
            ]
            key=("EE",ei,ej)
            accum[key],changed=_apply_unilateral(
                positions,gradients,mass,C,alpha,accum.get(key,0.),
            )
            projections+=int(changed)
    if not all(np.isfinite(v).all() for v in positions.values()):
        raise QualificationError("XPBD_SELF_CONTACT_NONFINITE")
    return SurfaceSelfContactStatsV1(
        projections=projections,
        candidate_pairs=len(point_pairs)+len(edge_pairs),
        maximum_penetration=max_penetration,
    )
