from __future__ import annotations

"""Addressed deformable-body contact for canonical XPBD cloth.

The collider is an *explicit source-face selection* of the SAME canonical
mesh, deformed by the authoritative primary MIRA/Mechanics pose. Its vertices
are kinematic during cloth solving: cloth cannot write body mesh positions.
Narrowphase resolves vertex/triangle thickness and tests continuous point/
moving-triangle coplanarity over LINEAR fixed-substep trajectories. This is
NOT general edge-edge CCD, coplanar overlap CCD, or full cloth collision.
"""

from dataclasses import asdict, dataclass
import math
from typing import Mapping

import numpy as np

from .types import QualificationError
from .mechanics_xpbd_surface_self_contact_v1 import _closest_triangle
from .hashing import content_sha256
from .mechanics_xpbd_swept_bvh_v1 import (
    SweptBVHV1, prepare_swept_bvh_v1, query_swept_bvh_v1,
    swept_aabb_from_endpoints_v1,
)
from .mechanics_xpbd_body_edge_ccd_v1 import evaluate_swept_body_edge_pair_v1

Face = tuple[str, str, str]
Vec = np.ndarray


@dataclass(frozen=True)
class XPBDBodyMeshColliderV1:
    collider_id: str
    source_faces: tuple[Face, ...]
    thickness: float = 0.005
    swept_vertex_triangle_ccd: bool = True
    excluded_cloth_vertex_ids: tuple[str, ...] = ()
    swept_edge_edge_ccd: bool = False


@dataclass(frozen=True)
class XPBDBodyMeshContactResultV1:
    projections: int
    swept_hits: int
    candidates: int
    max_penetration: float
    unresolved_sweeps: int
    broadphase_visited_nodes: int = 0
    broadphase_examined_primitives: int = 0


@dataclass(frozen=True)
class PreparedXPBDBodyMeshSweptBVHV1:
    """Bound to the exact canonical body trajectories and authored colliders.

    Prepared once per fixed physics substep and reused across all XPBD
    iterations. A stale/mismatched index is rejected by its content hash.
    """

    binding_hash: str
    collider_faces: tuple[tuple[int, Face], ...]
    face_bvh: SweptBVHV1
    collider_edges: tuple[tuple[int, tuple[str,str]], ...]
    edge_bvh: SweptBVHV1


@dataclass(frozen=True)
class XPBDBodyMeshEdgeContactResultV1:
    projections: int
    swept_hits: int
    candidates: int
    max_penetration: float
    unresolved_sweeps: int
    broadphase_examined_primitives: int


def _body_sweep_binding_v1(
    colliders: tuple[XPBDBodyMeshColliderV1, ...],
    body_at_start: Mapping[str, Vec],
    body_at_end: Mapping[str, Vec],
) -> str:
    identifiers=sorted({
        vid for c in colliders for face in c.source_faces for vid in face
    })
    rows=[]
    for vid in identifiers:
        try:
            start=np.asarray(body_at_start[vid],dtype=np.float64)
            end=np.asarray(body_at_end[vid],dtype=np.float64)
        except (KeyError,ValueError,TypeError) as exc:
            raise QualificationError("XPBD_BODY_SWEEP_VERTEX_COVERAGE_INVALID") from exc
        if (start.shape!=(3,) or end.shape!=(3,)
                or not np.isfinite(start).all() or not np.isfinite(end).all()):
            raise QualificationError("XPBD_BODY_SWEEP_VERTEX_NONFINITE")
        rows.append((vid,tuple(map(float,start)),tuple(map(float,end))))
    return content_sha256({
        "canonical_body_vertices": rows,
        "authored_colliders": [asdict(c) for c in colliders],
    })


def prepare_xpbd_body_mesh_swept_bvh_v1(
    colliders: tuple[XPBDBodyMeshColliderV1, ...],
    body_at_start: Mapping[str, Vec],
    body_at_end: Mapping[str, Vec],
) -> PreparedXPBDBodyMeshSweptBVHV1:
    """Construct body-face swept bounds once for the current fixed substep."""
    if not colliders:
        raise QualificationError("XPBD_BODY_SWEEP_BVH_EMPTY")
    signature=_body_sweep_binding_v1(colliders,body_at_start,body_at_end)
    records=[]
    bounds=[]
    edges=[]
    edge_bounds=[]
    for ci,collider in enumerate(colliders):
        seen_edges=set()
        for face in collider.source_faces:
            tri0=tuple(np.asarray(body_at_start[vid],dtype=np.float64) for vid in face)
            tri1=tuple(np.asarray(body_at_end[vid],dtype=np.float64) for vid in face)
            n0=_unit_triangle_normal(tri0)
            n1=_unit_triangle_normal(tri1)
            if float(n0@n1)<=0.:
                raise QualificationError("XPBD_BODY_COLLIDER_POSED_FACE_FLIPPED")
            records.append((ci,face))
            bounds.append(swept_aabb_from_endpoints_v1(
                tri0,tri1,margin=collider.thickness,
            ))
            for a,b in ((face[0],face[1]),(face[1],face[2]),(face[2],face[0])):
                edge=tuple(sorted((a,b)))
                if edge in seen_edges:
                    continue
                seen_edges.add(edge)
                edges.append((ci,edge))
                edge_bounds.append(swept_aabb_from_endpoints_v1(
                    tuple(np.asarray(body_at_start[vid],dtype=np.float64) for vid in edge),
                    tuple(np.asarray(body_at_end[vid],dtype=np.float64) for vid in edge),
                    margin=collider.thickness,
                ))
    return PreparedXPBDBodyMeshSweptBVHV1(
        signature,tuple(records),prepare_swept_bvh_v1(tuple(bounds)),
        tuple(edges),prepare_swept_bvh_v1(tuple(edge_bounds)),
    )


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
                or type(collider.swept_vertex_triangle_ccd) is not bool
                or type(collider.swept_edge_edge_ccd) is not bool):
            raise QualificationError("XPBD_BODY_COLLIDER_DECLARATION_INVALID")
        ids.add(collider.collider_id)
        try:
            thickness=float(collider.thickness)
        except (ValueError,TypeError) as exc:
            raise QualificationError("XPBD_BODY_COLLIDER_THICKNESS_INVALID") from exc
        if not math.isfinite(thickness) or thickness<=0. or thickness>1.:
            raise QualificationError("XPBD_BODY_COLLIDER_THICKNESS_INVALID")
        if (type(collider.excluded_cloth_vertex_ids) is not tuple
                or any(type(v) is not str or not v for v in collider.excluded_cloth_vertex_ids)
                or len(set(collider.excluded_cloth_vertex_ids))!=len(collider.excluded_cloth_vertex_ids)
                or not set(collider.excluded_cloth_vertex_ids)<=cloth):
            raise QualificationError("XPBD_BODY_COLLIDER_EXCLUSION_INVALID")
        if any(
            type(face) is not tuple or len(face)!=3
            or any(type(v) is not str for v in face)
            for face in collider.source_faces
        ):
            raise QualificationError("XPBD_BODY_COLLIDER_FACE_INVALID")
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
    prepared_bvh: PreparedXPBDBodyMeshSweptBVHV1 | None = None,
) -> XPBDBodyMeshContactResultV1:
    """One deterministic iteration; body positions are immutable kinematic inputs.

    `previous_positions` refers to the beginning of THIS physics substep.
    The body at each endpoint follows linear interpolation between MIRA-skinned
    primary deformations. Moving-triangle roots catch full-step tunneling.
    """
    if type(max_candidates) is not int or not 1<=max_candidates<=32768:
        raise QualificationError("XPBD_BODY_CONTACT_CANDIDATE_POLICY_INVALID")
    if prepared_bvh is None:
        prepared_bvh=prepare_xpbd_body_mesh_swept_bvh_v1(
            colliders,body_at_start,body_at_end,
        )
    elif (type(prepared_bvh) is not PreparedXPBDBodyMeshSweptBVHV1
            or prepared_bvh.binding_hash!=_body_sweep_binding_v1(
                colliders,body_at_start,body_at_end,
            )):
        raise QualificationError("XPBD_BODY_SWEEP_BVH_BINDING_DRIFT")
    candidates=0
    projections=0
    hits=0
    unresolved=0
    maximum=0.
    visited=0
    examined=0
    # One BVH query per cloth trajectory. Body faces are not scanned for
    # each cloth vertex; candidate pairs are then sorted in canonical face
    # order and cloth-ID order for the same deterministic solver schedule.
    by_face:dict[int,list[str]]={}
    for vid in dynamic_vertex_ids:
        p0=np.asarray(previous_positions[vid],dtype=np.float64)
        p1=np.asarray(positions[vid],dtype=np.float64)
        query=swept_aabb_from_endpoints_v1((p0,),(p1,),margin=0.)
        matches=query_swept_bvh_v1(
            prepared_bvh.face_bvh,query,max_candidates=max_candidates,
        )
        visited+=matches.visited_nodes
        examined+=matches.examined_leaf_primitives
        for face_index in matches.candidate_indices:
            by_face.setdefault(face_index,[]).append(vid)
    for face_index in sorted(by_face):
        ci,face=prepared_bvh.collider_faces[face_index]
        collider=colliders[ci]
        tri0=tuple(np.asarray(body_at_start[k],dtype=np.float64) for k in face)
        tri1=tuple(np.asarray(body_at_end[k],dtype=np.float64) for k in face)
        n0=_unit_triangle_normal(tri0)
        n1=_unit_triangle_normal(tri1)
        if float(n0@n1)<=0.:
            raise QualificationError("XPBD_BODY_COLLIDER_POSED_FACE_FLIPPED")
        for vid in by_face[face_index]:
                if vid in collider.excluded_cloth_vertex_ids:
                    continue
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
        broadphase_visited_nodes=visited,
        broadphase_examined_primitives=examined,
    )


def resolve_xpbd_body_mesh_edge_contact_v1(
    positions: dict[str,Vec],
    previous_positions: Mapping[str,Vec],
    body_at_start: Mapping[str,Vec],
    body_at_end: Mapping[str,Vec],
    cloth_edges: tuple[tuple[str,str,float], ...],
    inverse_masses: Mapping[str,float],
    colliders: tuple[XPBDBodyMeshColliderV1,...],
    *,
    max_candidates: int,
    project: bool,
    check_sweeps: bool = True,
    prepared_bvh: PreparedXPBDBodyMeshSweptBVHV1 | None = None,
) -> XPBDBodyMeshEdgeContactResultV1:
    """Two endpoint-weighted XPBD corrections against kinematic body edges.

    This opt-in court covers discrete edge proximity plus interior swept
    edge/edge coplanarity roots. The body owns its geometry and never moves.
    Unsolved crossings and non-resolvable pinned edge intersections fail
    the *caller*'s final residual admission rather than manufacturing PASS.
    """
    if type(max_candidates) is not int or not 1<=max_candidates<=32768:
        raise QualificationError("XPBD_BODY_EDGE_CANDIDATE_POLICY_INVALID")
    enabled=any(c.swept_edge_edge_ccd for c in colliders)
    if not enabled:
        return XPBDBodyMeshEdgeContactResultV1(0,0,0,0.,0,0)
    if prepared_bvh is None:
        prepared_bvh=prepare_xpbd_body_mesh_swept_bvh_v1(
            colliders,body_at_start,body_at_end,
        )
    elif (type(prepared_bvh) is not PreparedXPBDBodyMeshSweptBVHV1
            or prepared_bvh.binding_hash!=_body_sweep_binding_v1(
                colliders,body_at_start,body_at_end,
            )):
        raise QualificationError("XPBD_BODY_SWEEP_BVH_BINDING_DRIFT")
    candidates=0
    projections=0
    swept_hits=0
    unresolved=0
    maximum=0.
    examined=0
    # Query each addressed cloth edge's full linear swept AABB against
    # kinematic body-edge swept boxes. Order is stable by canonical edge
    # list, collider ID/order and body edge vertex IDs.
    for a,b,_rest in cloth_edges:
        p0=np.asarray(previous_positions[a],dtype=np.float64)
        q0=np.asarray(previous_positions[b],dtype=np.float64)
        p1=np.asarray(positions[a],dtype=np.float64)
        q1=np.asarray(positions[b],dtype=np.float64)
        query=swept_aabb_from_endpoints_v1((p0,q0),(p1,q1),margin=0.)
        matches=query_swept_bvh_v1(
            prepared_bvh.edge_bvh,query,max_candidates=max_candidates,
        )
        examined+=matches.examined_leaf_primitives
        for index in matches.candidate_indices:
            ci,body_edge=prepared_bvh.collider_edges[index]
            collider=colliders[ci]
            if not collider.swept_edge_edge_ccd:
                continue
            if (a in collider.excluded_cloth_vertex_ids
                    or b in collider.excluded_cloth_vertex_ids):
                continue
            candidates+=1
            if candidates>max_candidates:
                raise QualificationError("XPBD_BODY_EDGE_CANDIDATE_BOUND")
            # Re-read dynamic current endpoints after earlier pair contacts.
            p1=np.asarray(positions[a],dtype=np.float64)
            q1=np.asarray(positions[b],dtype=np.float64)
            c,d=body_edge
            contact=evaluate_swept_body_edge_pair_v1(
                (p0,q0),(p1,q1),
                (np.asarray(body_at_start[c],dtype=np.float64),
                 np.asarray(body_at_start[d],dtype=np.float64)),
                (np.asarray(body_at_end[c],dtype=np.float64),
                 np.asarray(body_at_end[d],dtype=np.float64)),
                thickness=collider.thickness,
                check_sweeps=check_sweeps,
            )
            if contact is None:
                continue
            swept_hits+=contact.swept_hits
            maximum=max(maximum,contact.endpoint_penetration)
            s=contact.cloth_edge_parameter
            wa=float(inverse_masses[a])
            wb=float(inverse_masses[b])
            normal=np.asarray(contact.normal,dtype=np.float64)
            denominator=wa*(1.-s)**2+wb*s*s
            if (project and denominator>1e-18
                    and contact.required_separation>1e-12):
                factor=contact.required_separation/denominator
                positions[a]=p1+wa*(1.-s)*factor*normal
                positions[b]=q1+wb*s*factor*normal
                projections+=1
            elif not project and contact.swept_hits:
                unresolved+=contact.swept_hits
    if any(not np.isfinite(x).all() for x in positions.values()):
        raise QualificationError("XPBD_BODY_EDGE_NONFINITE")
    return XPBDBodyMeshEdgeContactResultV1(
        projections=projections,swept_hits=swept_hits,
        candidates=candidates,max_penetration=maximum,
        unresolved_sweeps=unresolved,
        broadphase_examined_primitives=examined,
    )
