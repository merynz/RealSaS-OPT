"""Certify local oriented probes without promoting them to global solid truth.

The anchor must be the isolated nearest primitive. Equidistant competing surfaces
are masked, including the two-parallel-surface counterexample in the corpus audit.
Only qualified rows are returned; absence never becomes a numeric zero label.
"""
from __future__ import annotations
import numpy as np

SCHEMA = 'RealSaS.IsolatedLocalOrientedProbes.v1'


def certify(vertices, faces, anchor_face, anchor_barycentric, offsets, *, face_valid=None, separation_fraction=1e-8):
    import trimesh
    v=np.asarray(vertices,np.float64); f=np.asarray(faces)
    ids=np.asarray(anchor_face); bary=np.asarray(anchor_barycentric,np.float64); delta=np.asarray(offsets,np.float64)
    if v.ndim!=2 or v.shape[1:]!=(3,) or not len(v) or not np.isfinite(v).all():
        raise ValueError('LOCAL_PROBE_VERTEX_LAYOUT')
    if f.ndim!=2 or f.shape[1:]!=(3,) or not len(f) or f.dtype.kind not in 'iu' or np.any(f<0) or np.any(f>=len(v)):
        raise ValueError('LOCAL_PROBE_FACE_LAYOUT')
    if ids.ndim!=1 or ids.dtype.kind not in 'iu' or np.any(ids<0) or np.any(ids>=len(f)) or bary.shape!=(len(ids),3) or delta.shape!=(len(ids),):
        raise ValueError('LOCAL_PROBE_ANCHOR_LAYOUT')
    if not np.isfinite(bary).all() or not np.isfinite(delta).all() or np.any(bary<0) or not np.allclose(bary.sum(1),1.,atol=1e-12,rtol=0):
        raise ValueError('LOCAL_PROBE_ANCHOR_DOMAIN')
    fraction=float(separation_fraction)
    if not np.isfinite(fraction) or fraction<=0:
        raise ValueError('LOCAL_PROBE_SEPARATION_POLICY')
    scale=max(float(np.linalg.norm(np.ptp(v,axis=0))),1e-12); margin=scale*fraction
    tri=v[f]; cross=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]); length=np.linalg.norm(cross,axis=1)
    valid=length>1e-12*scale*scale
    if face_valid is not None:
        supplied=np.asarray(face_valid)
        if supplied.shape!=(len(f),) or supplied.dtype!=bool:raise ValueError('LOCAL_PROBE_VALIDITY_LAYOUT')
        valid &= supplied
    normal=np.divide(cross,length[:,None],out=np.zeros_like(cross),where=length[:,None]>0)
    anchors=np.sum(tri[ids]*bary[:,:,None],axis=1); queries=anchors+delta[:,None]*normal[ids]
    mesh=trimesh.Trimesh(vertices=v,faces=f,process=False)
    tree=mesh.triangles_tree
    keep=np.zeros(len(ids),bool); reasons=np.full(len(ids),'UNMEASURED',dtype='<U48')
    for i,(q,fi,d) in enumerate(zip(queries,ids,delta)):
        if not valid[fi] or abs(d)<=margin:
            reasons[i]='INVALID_ANCHOR_OR_UNRESOLVED_OFFSET';continue
        own=trimesh.triangles.closest_point(tri[fi:fi+1],q[None])[0]
        own_distance=float(np.linalg.norm(q-own))
        if not np.allclose(own,anchors[i],atol=margin,rtol=0) or abs(own_distance-abs(d))>margin:
            reasons[i]='ANCHOR_NOT_OWN_CLOSEST_POINT';continue
        # Every point within the required distance lies in this AABB. Filtering
        # triangle AABBs cannot omit a closer/tied competing source primitive.
        radius=abs(float(d))+margin
        candidates=np.asarray(list(tree.intersection(tuple(q-radius)+tuple(q+radius))),np.int64)
        candidates=candidates[candidates!=fi]
        if len(candidates):
            # Invalid surfaces still obstruct a sign certificate; they are not
            # silently removed merely because their own supervision is masked.
            points=trimesh.triangles.closest_point(tri[candidates],np.broadcast_to(q,(len(candidates),3)))
            distance=np.linalg.norm(points-q,axis=1)
            if not np.isfinite(distance).all() or np.any(distance<=abs(d)+margin):
                reasons[i]='COMPETING_OR_AMBIGUOUS_SURFACE';continue
        keep[i]=True;reasons[i]='CERTIFIED_ISOLATED_NEAREST_ANCHOR'
    selected=np.flatnonzero(keep).astype(np.int64)
    return {'schema':SCHEMA,'probe_index':selected,'query_canonical':queries[selected],
            'signed_local_distance':delta[selected],'anchor_face_index':ids[selected].astype(np.int64),
            'valid_mask':keep,'reason_codes':reasons,'separation_margin':margin,
            'scope':'LOCAL_ORIENTED_DISTANCE_ONLY','global_sign_authorized':False,'occupancy_authorized':False}
