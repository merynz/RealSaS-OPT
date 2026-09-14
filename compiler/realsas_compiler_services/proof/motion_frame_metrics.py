from __future__ import annotations

"""Evaluator-independent measurements over qualification-owned 2D frame bakes.

Loop closure and return-to-rest are distinct invariants. Historical probe clips
used both, but professional cyclic locomotion may intentionally start/end on a
non-rest planted pose. The frozen numerical return-to-rest threshold is preserved;
its applicability is now an explicit policy bit instead of being silently imposed
on every loop clip.
"""

import numpy as np

RESTORED_V05_POLICY_V1 = {
    "max_edge_stretch_ratio": 1.55,
    "max_area_change_ratio": 2.10,
    "max_loop_seam_error01": 0.025,
    "max_return_to_rest_error01": 0.025,
    "require_return_to_rest": True,
    "min_motion01": 1.0e-4,
    "max_flipped_triangles": 0,
}


def _area(a, b, c):
    return 0.5 * ((b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0]))


def _diag(points):
    p=np.asarray(points,dtype=np.float64); span=p.max(axis=0)-p.min(axis=0); return max(float(np.linalg.norm(span)),1e-12)


def measure_motion_bake_geometry(bake) -> dict:
    rest={mid:np.asarray(points,dtype=np.float64) for mid,points in bake.rest_mesh_vertices_by_id}
    tris={mid:tuple(rows) for mid,rows in bake.triangles_by_mesh_id}
    frames=[{mid:np.asarray(points,dtype=np.float64) for mid,points in f.mesh_vertices_by_id} for f in bake.frames]
    max_motion=max_edge=max_area=loop=return_rest=0.0; flips=0; first_flip={}; worst_edge={}; worst_tri={}
    for mid,r in rest.items():
        diag=_diag(r); edges=sorted({tuple(sorted((a,b))) for tri in tris.get(mid,()) for a,b in ((tri[0],tri[1]),(tri[1],tri[2]),(tri[2],tri[0]))})
        rest_edge={e:max(float(np.linalg.norm(r[e[1]]-r[e[0]])),1e-12) for e in edges}
        rest_area=[_area(r[a],r[b],r[c]) for a,b,c in tris.get(mid,())]
        for fi,fmap in enumerate(frames):
            p=fmap[mid]; max_motion=max(max_motion,float(np.linalg.norm(p-r,axis=1).max())/diag)
            for e in edges:
                ratio=float(np.linalg.norm(p[e[1]]-p[e[0]]))/rest_edge[e]; symmetric=max(ratio,1.0/max(ratio,1e-12))
                if symmetric>max_edge: max_edge=symmetric; worst_edge={"mesh_id":mid,"frame_index":fi,"edge":e,"ratio":symmetric}
            for ti,(a,b,c) in enumerate(tris.get(mid,())):
                ra=rest_area[ti]; pa=_area(p[a],p[b],p[c]); ratio=abs(pa)/max(abs(ra),1e-12); symmetric=max(ratio,1.0/max(ratio,1e-12))
                if symmetric>max_area: max_area=symmetric; worst_tri={"mesh_id":mid,"frame_index":fi,"triangle_index":ti,"ratio":symmetric}
                if ra*pa < 0:
                    flips+=1
                    if not first_flip: first_flip={"mesh_id":mid,"frame_index":fi,"triangle_index":ti}
        loop=max(loop,float(np.linalg.norm(frames[-1][mid]-frames[0][mid],axis=1).max())/diag if bake.loop else 0.0)
        return_rest=max(return_rest,float(np.linalg.norm(frames[-1][mid]-r,axis=1).max())/diag)
    return {"clip_id":bake.clip_id,"max_motion01":max_motion,"max_edge_stretch_ratio":max_edge or 1.0,"max_area_change_ratio":max_area or 1.0,"flipped_triangles":flips,"loop_seam_error01":loop,"return_to_rest_error01":return_rest,"first_flip":first_flip,"worst_edge":worst_edge,"worst_triangle":worst_tri,"requested_amplitude_evaluated":True,"safe_envelope_can_upgrade_failure":False}


def evaluate_motion_bake_metrics(measurements: dict, *, policy: dict | None = None) -> dict:
    p={**RESTORED_V05_POLICY_V1, **dict(policy or {})}; failures=[]
    if not isinstance(p.get("require_return_to_rest"), bool):
        raise ValueError("motion metric policy require_return_to_rest must be bool")
    if measurements["max_edge_stretch_ratio"]>float(p["max_edge_stretch_ratio"]): failures.append("bounded_edge_stretch")
    if measurements["max_area_change_ratio"]>float(p["max_area_change_ratio"]): failures.append("bounded_area_change")
    if int(measurements["flipped_triangles"])>int(p["max_flipped_triangles"]): failures.append("no_triangle_flip")
    if measurements["loop_seam_error01"]>float(p["max_loop_seam_error01"]): failures.append("loop_seam")
    if bool(p["require_return_to_rest"]) and measurements["return_to_rest_error01"]>float(p["max_return_to_rest_error01"]): failures.append("return_to_rest")
    if measurements["max_motion01"]<float(p["min_motion01"]): failures.append("required_mobility")
    return {
        **measurements,
        "policy":p,
        "return_to_rest_evaluated":bool(p["require_return_to_rest"]),
        "failure_invariants":failures,
        "passed":not failures,
    }
