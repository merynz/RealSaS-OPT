from __future__ import annotations
"""Exact on-demand depth witness on the same posed canonical Mesh.

This is geometry-only; alpha, material, source first-hit, and scene-wide
visibility are NOT proven. No triangle ordering or re-meshing is introduced.
"""
from dataclasses import dataclass
import math
from typing import Sequence
from .hashing import content_sha256
from .types import QualificationError


@dataclass(frozen=True)
class PoseDepthPixelEvidenceV1:
    view_id: str
    graphics_hash: str
    pose_projection_hash: str
    screen_xy: tuple[float, float]
    intersection_depths_by_face: tuple[tuple[int, float], ...]
    geometrically_front_face_indices: tuple[int, ...]
    geometrically_front_fragment_ids: tuple[str, ...]
    state: str
    source_visibility_qualified: bool = False
    source_alpha_qualified: bool = False
    presentation_order_qualified: bool = False
    authority: str = "POSED_CANONICAL_GEOMETRY_DEPTH_ONLY"


def _barycentric(x, y, vertices):
    (ax, ay, _), (bx, by, _), (cx, cy, _) = vertices
    d = (by-cy)*(ax-cx) + (cx-bx)*(ay-cy)
    if abs(d) <= 1e-12:
        return None
    a = ((by-cy)*(x-cx)+(cx-bx)*(y-cy))/d
    b = ((cy-ay)*(x-cx)+(ax-cx)*(y-cy))/d
    c = 1.0-a-b
    return (a,b,c) if min(a,b,c) >= -1e-8 else None


def inspect_pose_depth_pixel_v1(
    *, view_id: str, graphics_hash: str,
    projected_faces: Sequence[Sequence[Sequence[float]]],
    face_fragment_ids: Sequence[str],
    screen_x: float, screen_y: float, depth_tolerance: float = 1e-6,
) -> PoseDepthPixelEvidenceV1:
    """Smaller camera-forward Z wins; ties ABSTAIN rather than face-order rule."""
    if (
        type(view_id) is not str or not view_id
        or type(graphics_hash) is not str or not graphics_hash
        or len(projected_faces) != len(face_fragment_ids)
        or not 1 <= len(projected_faces) <= 2_000_000
        or any(type(k) is not str or not k for k in face_fragment_ids)
        or any(type(v) not in (int,float) or not math.isfinite(v)
               for v in (screen_x,screen_y,depth_tolerance))
        or not 0 <= depth_tolerance <= .01
    ):
        raise QualificationError("POSE_DEPTH_QUERY_DOMAIN_INVALID")
    faces=[]
    for face in projected_faces:
        if len(face)!=3:
            raise QualificationError("POSE_DEPTH_TRIANGLE_DOMAIN_INVALID")
        vs=[]
        for vertex in face:
            if len(vertex)!=3 or any(
                type(v) not in (int,float) or not math.isfinite(v) for v in vertex
            ):
                raise QualificationError("POSE_DEPTH_VERTEX_NONFINITE_OR_INVALID")
            vs.append(tuple(float(v) for v in vertex))
        faces.append(tuple(vs))
    pose_hash=content_sha256({
        "schema":"RealSaS.PoseGraphicsProjection.v1",
        "view_id":view_id,"graphics_hash":graphics_hash,
        "triangles":faces,"face_fragment_ids":tuple(face_fragment_ids),
    })
    hits=[]
    for i,face in enumerate(faces):
        weights=_barycentric(float(screen_x),float(screen_y),face)
        if weights is None: continue
        depth=sum(w*v[2] for w,v in zip(weights,face))
        if not math.isfinite(depth):
            raise QualificationError("POSE_DEPTH_INTERPOLATED_NONFINITE")
        hits.append((i,float(depth)))
    hits.sort(key=lambda row:(row[1],row[0]))
    if not hits:
        front=()
        state="MISS"
    else:
        z=hits[0][1]
        front=tuple(i for i,d in hits if d-z<=depth_tolerance)
        state=("UNIQUE_GEOMETRIC_FRONT" if len(front)==1
               else "AMBIGUOUS_COPLANAR_FRONT")
    return PoseDepthPixelEvidenceV1(
        view_id=view_id,graphics_hash=graphics_hash,pose_projection_hash=pose_hash,
        screen_xy=(float(screen_x),float(screen_y)),
        intersection_depths_by_face=tuple(hits),
        geometrically_front_face_indices=front,
        geometrically_front_fragment_ids=tuple(sorted(set(face_fragment_ids[i] for i in front))),
        state=state,
    )


__all__=["PoseDepthPixelEvidenceV1","inspect_pose_depth_pixel_v1"]
