from __future__ import annotations

"""Deterministic rest-frame construction for canonical/source skeletons.

Frames are derived from geometry plus the frozen RealSaS object frame. They are
mechanical coordinates only: no semantic joint labels are inferred.
"""

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np

from .hashing import content_sha256
from .types import QualificationError


REALSAS_OBJECT_FRAME_V1="REALSAS_OBJECT_FRAME_V1"


@dataclass(frozen=True)
class DerivedJointFrameV1:
    joint_id:str
    parent_id:str|None
    rest_position:tuple[float,float,float]
    rotation_matrix:tuple[tuple[float,float,float],...]
    frame_hash:str


def _unit(value,code):
    v=np.asarray(value,dtype=np.float64)
    if v.shape!=(3,) or not np.isfinite(v).all():
        raise QualificationError(code)
    n=float(np.linalg.norm(v))
    if n<=1e-12:
        raise QualificationError(code)
    return v/n


def _frame_hash(jid,parent,pos,R):
    return content_sha256({
        "schema":"RealSaS.DerivedJointFrame.v1",
        "joint_id":str(jid),
        "parent_id":None if parent is None else str(parent),
        "rest_position":tuple(map(float,pos)),
        "rotation_matrix":tuple(tuple(map(float,row)) for row in np.asarray(R)),
    })


def object_basis_from_camera_set(cameras):
    rows=tuple(sorted(tuple(cameras),key=lambda c:int(c.view_index)))
    if len(rows)!=8 or tuple(int(c.view_index) for c in rows)!=tuple(range(8)):
        raise QualificationError("JOINT_FRAME_REQUIRES_EXACT_8_CAMERAS")
    v0=rows[0]
    right=_unit(v0.right,"JOINT_FRAME_CAMERA_RIGHT_INVALID")
    up=_unit(v0.screen_up,"JOINT_FRAME_CAMERA_UP_INVALID")
    forward=_unit(tuple(-float(x) for x in v0.forward),"JOINT_FRAME_CAMERA_FORWARD_INVALID")
    up=up-right*float(np.dot(up,right))
    up=_unit(up,"JOINT_FRAME_OBJECT_UP_DEGENERATE")
    f=np.cross(right,up)
    f=_unit(f,"JOINT_FRAME_OBJECT_FORWARD_DEGENERATE")
    if float(np.dot(f,forward))<0.0:
        f=-f
    right=np.cross(up,f)
    right=_unit(right,"JOINT_FRAME_OBJECT_RIGHT_DEGENERATE")
    return right,up,f


def _derive_frame_for_joint(*,parent,pos,by_pos,global_right,global_up,global_forward):
    if parent is None:
        return np.column_stack((global_right,global_up,global_forward))
    y=_unit(np.asarray(pos)-np.asarray(by_pos[parent]),"JOINT_FRAME_BONE_DEGENERATE")
    z=global_forward-y*float(np.dot(global_forward,y))
    if float(np.linalg.norm(z))<=1e-8:
        z=global_up-y*float(np.dot(global_up,y))
    if float(np.linalg.norm(z))<=1e-8:
        z=global_right-y*float(np.dot(global_right,y))
    z=_unit(z,"JOINT_FRAME_SECONDARY_AXIS_DEGENERATE")
    x=_unit(np.cross(y,z),"JOINT_FRAME_PRIMARY_AXIS_DEGENERATE")
    if float(np.dot(x,global_right))<0.0:
        x=-x; z=-z
    z=_unit(np.cross(x,y),"JOINT_FRAME_ORTHONORMALIZATION_FAIL")
    R=np.column_stack((x,y,z))
    if np.linalg.det(R)<0.999999:
        raise QualificationError("JOINT_FRAME_NOT_RIGHT_HANDED")
    return R


def derive_joint_frames_from_rows(
    rows:Iterable[Mapping],
    *,
    joint_id_key:str,
    parent_id_key:str,
    position_key:str,
    global_right=(1.0,0.0,0.0),
    global_up=(0.0,0.0,1.0),
    global_forward=(0.0,1.0,0.0),
):
    rows=tuple(dict(r) for r in rows)
    if not rows:
        raise QualificationError("JOINT_FRAME_SKELETON_EMPTY")
    ids=[str(r[joint_id_key]) for r in rows]
    if len(ids)!=len(set(ids)):
        raise QualificationError("JOINT_FRAME_JOINT_ID_DUPLICATE")
    by={str(r[joint_id_key]):r for r in rows}
    parent={}
    pos={}
    roots=[]
    for jid in ids:
        raw=by[jid]
        p=raw.get(parent_id_key)
        p=None if p in (None,"") else str(p)
        if p is not None and p not in by:
            raise QualificationError("JOINT_FRAME_PARENT_UNKNOWN")
        parent[jid]=p
        if p is None:
            roots.append(jid)
        xyz=np.asarray(raw[position_key],dtype=np.float64)
        if xyz.shape!=(3,) or not np.isfinite(xyz).all():
            raise QualificationError("JOINT_FRAME_POSITION_INVALID")
        pos[jid]=xyz
    if len(roots)!=1:
        raise QualificationError("JOINT_FRAME_EXACTLY_ONE_ROOT_REQUIRED")
    for jid in ids:
        seen=set()
        cur=jid
        while cur is not None:
            if cur in seen:
                raise QualificationError("JOINT_FRAME_SKELETON_CYCLE")
            seen.add(cur)
            cur=parent[cur]
    gr=_unit(global_right,"JOINT_FRAME_GLOBAL_RIGHT_INVALID")
    gu=np.asarray(global_up,dtype=np.float64)
    gu=gu-gr*float(np.dot(gu,gr))
    gu=_unit(gu,"JOINT_FRAME_GLOBAL_UP_INVALID")
    gf=_unit(np.cross(gr,gu),"JOINT_FRAME_GLOBAL_FORWARD_INVALID")
    desired=_unit(global_forward,"JOINT_FRAME_GLOBAL_FORWARD_INVALID")
    if float(np.dot(gf,desired))<0.0:
        gf=-gf
    gr=_unit(np.cross(gu,gf),"JOINT_FRAME_GLOBAL_RIGHT_INVALID")
    out={}
    for jid in sorted(ids):
        R=_derive_frame_for_joint(
            parent=parent[jid],pos=pos[jid],by_pos=pos,
            global_right=gr,global_up=gu,global_forward=gf,
        )
        fh=_frame_hash(jid,parent[jid],pos[jid],R)
        out[jid]=DerivedJointFrameV1(
            jid,parent[jid],tuple(map(float,pos[jid])),
            tuple(tuple(map(float,row)) for row in R),fh,
        )
    return out


def derive_joint_frames_from_skeleton(skeleton,*,cameras):
    right,up,forward=object_basis_from_camera_set(cameras)
    rows=tuple({
        "joint_id":j.canonical_joint_id,
        "parent_id":j.parent_canonical_id,
        "position":j.position,
    } for j in skeleton.joints)
    return derive_joint_frames_from_rows(
        rows,joint_id_key="joint_id",parent_id_key="parent_id",position_key="position",
        global_right=right,global_up=up,global_forward=forward,
    )


def frame_set_hash(frames)->str:
    return content_sha256({
        "schema":"RealSaS.DerivedJointFrameSet.v1",
        "frames":tuple((jid,frames[jid].frame_hash) for jid in sorted(frames)),
    })
