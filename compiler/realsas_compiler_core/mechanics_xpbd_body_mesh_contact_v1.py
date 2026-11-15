from __future__ import annotations

"""Addressed deformable-body contact for canonical XPBD cloth.

The collider is an *explicit source-face selection* of the SAME canonical
mesh, deformed by the authoritative primary MIRA/Mechanics pose. Its vertices
are kinematic during cloth solving: cloth cannot write body mesh positions.
Narrowphase resolves vertex/triangle thickness and tests continuous point/
moving-triangle coplanarity over LINEAR fixed-substep trajectories. This is
NOT general edge-edge CCD, coplanar overlap CCD, or full cloth collision.
"""

from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np

from .types import QualificationError
from .mechanics_xpbd_surface_self_contact_v1 import _closest_triangle

Face = tuple[str, str, str]
Vec = np.ndarray


@dataclass(frozen=True)
class XPBDBodyMeshColliderV1:
    collider_id: str
    source_faces: tuple[Face, ...]
    thickness: float = 0.005
    swept_vertex_triangle_ccd: bool = True


@dataclass(frozen=True)
class XPBDBodyMeshContactResultV1:
    projections: int
    swept_hits: int
    candidates: int
    max_penetration: float
    unresolved_sweeps: int


def qualify_xpbd_body_mesh_colliders_v1(
    mesh_faces: tuple[Face, ...],
    simulated_vertex_ids: tuple[str, ...],
    body_colliders: tuple[XPBDBodyMeshColliderV1, ...],
) -> tuple[str, ...]:
    """Reject foreign, rehashed, duplicated or overlapping collider ownership."""
    if not isinstance(body_colliders, tuple) or len(body_colliders)>8:
        raise QualificationError("XPBD_BODY_COLLIDER_RESOURCE_BOUND")
    source=set(mesh_faces)
    cloth=set(simulated_vertex_ids)
    ids=set()
    body_vertices=set()
    selected=set()
    for collider in body_colliders:
        if (type(collider) is not XPBDBodyMeshColliderV1
                or not isinstance(collider.collider_id,str)
                or not collider.collider_id or collider.collider_id in ids
                or not isinstance(collider.source_faces,tuple)
                or not collider.source_faces or len(collider.source_faces)>512
                or type(collider.swept_vertex_triangle_ccd) is not bool):
            raise QualificationError("XPBD_BODY_COLLIDER_DECLARATION_INVALID")
        ids.add(collider.collider_id)
        try:
            thickness=float(collider.thickness)
        except (ValueError,TypeError) as exc:
            raise QualificationError("XPBD_BODY_COLLIDER_THICKNESS_INVALID") from exc
        if not math.isfinite(thickness) or thickness<=0. or thickness>1.:
            raise QualificationError("XPBD_BODY_COLLIDER_THICKNESS_INVALID")
        if len(set(collider.source_faces))!=len(collider.source_faces):
            raise QualificationError("XPBD_BODY_COLLIDER_FACE_DUPLICATE")
        for face in collider.source_faces:
            if face not in source:
                raise QualificationError("XPBD_BODY_COLLIDER_FACE_NOT_CANONICAL")
            if any(vertex in cloth for vertex in face):
                raise QualificationError("XPBD_BODY_COLLIDER_DOUBLE_VERTEX_OWNERSHIP")
            if face in selected:
                raise QualificationError("XPBD_BODY_COLLIDER_FACE_DOUBLE_OWNERSHIP")
            selected.add(face)
            body_vertices.update(face)
    if len(selected)>1024 or len(body_vertices)>3072:
        raise QualificationError("XPBD_BODY_COLLIDER_RESOURCE_BOUND")
    return tuple(sorted(body_vertices))


def _unit_triangle_normal(vertices: tuple[Vec,Vec,Vec]) -> Vec:
    a,b,c=vertices
    cross=np.cross(b-a,c-a)
    n=float(np.linalg.norm(cross))
    if n<=1e-12 or not math.isfinite(n):
        raise QualificationError("XPBD_BODY_COLLIDER_DEGENERATE_POSED_FACE")
    return cross/n


def _plane_crossing_times(
    point0: Vec, point1: Vec,
    tri0: tuple[Vec,Vec,Vec],
    tri1: tuple[Vec,Vec,Vec],
) -> tuple[float,...]:
    """Exact cubic co-planarity roots for linear vertex trajectories.

    A valid root is retained only if the moving point is inside the moving
    triangle at that instant. Nearly identically coplanar trajectories are
    rejected conservatively; this module cannot claim coplanar CCD.
    """
    a0,b0,c0=tri0
    a1,b1,c1=tri1
    r0=point0-a0
    dr=(point1-point0)-(a1-a0)
    u0=b0-a0
    du=(b1-b0)-(a1-a0)
    v0=c0-a0
    dv=(c1-c0)-(a1-a0)
    cross0=np.cross(u0,v0)
    cross1=np.cross(du,v0)+np.cross(u0,dv)
    cross2=np.cross(du,dv)
    coeffs=np.array((
        float(dr@cross2),
        float(r0@cross2+dr@cross1),
        float(r0@cross1+dr@cross0),
        float(r0@cross0),
    ),dtype=np.float64)
    norm=float(np.max(np.abs(coeffs)))
    if norm<=1e-13:
        raise QualificationError("XPBD_BODY_COLLIDER_COPLANAR_SWEEP_UNSUPPORTED")
    start=next((i for i,x in enumerate(coeffs) if abs(x)>norm*1e-12),3)
    roots=np.roots(coeffs[start:])
    hits=[]
    for root in roots:
        if abs(float(np.imag(root)))>1e-7:
            continue
        t=float(np.real(root))
        if not 1e-8<t<1.-1e-8:
            continue
        p=point0+t*(point1-point0)
        a,b,c=(x+t*(y-x) for x,y in zip(tri0,tri1))
        nearest,_=_closest_triangle(p,a,b,c)
        scale=max(1.,float(np.linalg.norm(b-a)),float(np.linalg.norm(c-a)))
        if float(np.linalg.norm(p-nearest))<=1e-7*scale:
            hits.append(t)
    return tuple(sorted(hits))


def _candidate(
    p0:Vec,p1:Vec,tri0:tuple[Vec,Vec,Vec],tri1:tuple[Vec,Vec,Vec],
    margin:float,
) -> bool:
    pmin=np.minimum(p0,p1)-margin
    pmax=np.maximum(p0,p1)+margin
    vertices=np.stack((*tri0,*tri1))
    bmin=vertices.min(axis=0)-margin
    bmax=vertices.max(axis=0)+margin
    return bool(np.all(pmax>=bmin) and np.all(bmax>=pmin))


def resolve_xpbd_body_mesh_contact_v1(
    positions: dict[str,Vec],
    previous_positions: Mapping[str,Vec],
    body_at_start: Mapping[str,Vec],
    body_at_end: Mapping[str,Vec],
    dynamic_vertex_ids: tuple[str,...],
    colliders: tuple[XPBDBodyMeshColliderV1,...],
    *,
    max_candidates: int,
    project: bool,
    check_sweeps: bool = True,
) -> XPBDBodyMeshContactResultV1:
    """One deterministic iteration; body positions are immutable kinematic inputs.

    `previous_positions` refers to the beginning of THIS physics substep.
    The body at each endpoint follows linear interpolation between MIRA-skinned
    primary deformations. Moving-triangle roots catch full-step tunneling.
    """
    if type(max_candidates) is not int or max_candidates<1:
        raise QualificationError("XPBD_BODY_CONTACT_CANDIDATE_POLICY_INVALID")
    candidates=0
    projections=0
    hits=0
    unresolved=0
    maximum=0.
    for collider in colliders:
        for face in collider.source_faces:
            tri0=tuple(np.asarray(body_at_start[k],dtype=np.float64) for k in face)
            tri1=tuple(np.asarray(body_at_end[k],dtype=np.float64) for k in face)
            n0=_unit_triangle_normal(tri0)
            n1=_unit_triangle_normal(tri1)
            if float(n0@n1)<=0.:
                raise QualificationError("XPBD_BODY_COLLIDER_POSED_FACE_FLIPPED")
            for vid in dynamic_vertex_ids:
                p0=np.asarray(previous_positions[vid],dtype=np.float64)
                p1=np.asarray(positions[vid],dtype=np.float64)
                if not _candidate(p0,p1,tri0,tri1,collider.thickness):
                    continue
                candidates+=1
                if candidates>max_candidates:
                    raise QualificationError("XPBD_BODY_CONTACT_CANDIDATE_BOUND")
                closest,_=_closest_triangle(p1,*tri1)
                diff=p1-closest
                distance=float(np.linalg.norm(diff))
                signed=float(diff@n1)
                start_signed=float((p0-tri0[0])@n0)
                # An initial side is authoritative only when the point was
                # clearly separated from the old collider surface.
                start_side=1. if start_signed>=0. else -1.
                swept=()
                if check_sweeps and collider.swept_vertex_triangle_ccd:
                    swept=_plane_crossing_times(p0,p1,tri0,tri1)
                if swept:
                    hits+=1
                if distance>=collider.thickness and not swept:
                    continue
                maximum=max(maximum,max(0.,collider.thickness-distance))
                if swept:
                    # Preserve the incoming side. A tunneling point must not
                    # be pushed further out on the WRONG side of the body.
                    delta=max(0.,collider.thickness-start_side*signed)
                    correction=delta*start_side*n1
                elif distance>1e-12:
                    correction=(collider.thickness-distance)*diff/distance
                else:
                    correction=collider.thickness*start_side*n1
                if project and float(np.linalg.norm(correction))>1e-12:
                    positions[vid]=p1+correction
                    projections+=1
                elif swept and not project:
                    unresolved+=1
    if not all(np.isfinite(v).all() for v in positions.values()):
        raise QualificationError("XPBD_BODY_CONTACT_NONFINITE")
    return XPBDBodyMeshContactResultV1(
        projections=projections,swept_hits=hits,candidates=candidates,
        max_penetration=maximum,unresolved_sweeps=unresolved,
    )
