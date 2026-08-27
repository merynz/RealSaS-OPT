#!/usr/bin/env python3
"""E0-a oracle vs E0-b deterministic persistence. E0-b API has no teacher identity."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import hashlib, math
import numpy as np
from e0_geometry import (VIEWS,NATIVE_RESOLUTION,RENDER_HALF_EXTENT,geometric_vertex_normals,reconstruct_surface,derive_view_local_normals,pixel_center_to_grid,grid_to_pixel_center,project_grid,camera_for_view,grid_to_nearest_pixel,raster_lookup_near,grid_distance_in_pixels)

@dataclass(frozen=True)
class ObservableView:
    view: int
    resolution: int
    pixel_linear_index: np.ndarray
    P: np.ndarray
    N_derived: np.ndarray
    N_valid: np.ndarray

    @property
    def grid(self):
        y = self.pixel_linear_index // self.resolution
        x = self.pixel_linear_index % self.resolution
        return pixel_center_to_grid(np.stack([x, y], axis=1), self.resolution)


@dataclass(frozen=True)
class AuthorityView:
    observable: ObservableView
    triangle_id: np.ndarray
    barycentric_uv: np.ndarray


def load_asset_authority(asset_dir: str | Path) -> tuple[dict, list[AuthorityView]]:
    asset_dir = Path(asset_dir)
    with np.load(asset_dir / "primary_geometry.npz", allow_pickle=False) as g:
        vertices = np.asarray(g["vertices"], np.float32)
        faces = np.asarray(g["faces"], np.int64)
    vertex_normals = geometric_vertex_normals(vertices, faces)
    views: list[AuthorityView] = []
    for v in range(VIEWS):
        rp = asset_dir / "renders" / f"V{v}" / "raster_authority.npz"
        with np.load(rp, allow_pickle=False) as z:
            required = {"pixel_linear_index", "triangle_id", "barycentric_uv", "resolution"}
            if not required.issubset(z.files):
                raise RuntimeError(f"{rp}: missing {sorted(required-set(z.files))}")
            pix = np.asarray(z["pixel_linear_index"], np.int64)
            tri = np.asarray(z["triangle_id"], np.int64)
            uv = np.asarray(z["barycentric_uv"], np.float32)
            rv = np.asarray(z["resolution"]).reshape(-1)
        if rv.size < 1 or not np.all(rv == NATIVE_RESOLUTION):
            raise RuntimeError(f"{rp}: native resolution drift {rv.tolist()}")
        if not (len(pix) == len(tri) == len(uv)):
            raise RuntimeError(f"{rp}: raster row mismatch")
        order = np.argsort(pix, kind="stable")
        pix, tri, uv = pix[order], tri[order], uv[order]
        P, _teacher_N_unused = reconstruct_surface(vertices, faces, vertex_normals, tri, uv)
        Nd, Nv = derive_view_local_normals(pix, P, NATIVE_RESOLUTION)
        obs = ObservableView(v, NATIVE_RESOLUTION, pix, P, Nd, Nv)
        views.append(AuthorityView(obs, tri, uv))
    geom = {"vertices": vertices, "faces": faces, "vertex_normals": vertex_normals}
    return geom, views


def deterministic_anchor_rows(authority_views: list[AuthorityView], asset_id: str, anchor_count: int = 512, pool_per_view: int = 2048):
    """P-only union sampler shared by E0-a/b.

    Teacher provenance does not participate. View-local rows are pooled deterministically and a
    common-frame farthest-point sampler spreads anchors over the actually visible union.
    """
    pts=[]; vv=[]; rr=[]
    for av in authority_views:
        n=len(av.observable.P)
        if n == 0:
            continue
        k=min(pool_per_view,n)
        seed=int(hashlib.sha256(f"E0|{asset_id}|V{av.observable.view}".encode()).hexdigest()[:16],16)&0x7fffffff
        ids=np.sort(np.random.default_rng(seed).choice(n,size=k,replace=False))
        pts.append(av.observable.P[ids]); vv.append(np.full(k,av.observable.view,np.int16)); rr.append(ids.astype(np.int64))
    P=np.concatenate(pts); V=np.concatenate(vv); R=np.concatenate(rr)
    if len(P)<anchor_count:
        raise RuntimeError(f"visible P pool too small: {len(P)} < {anchor_count}")
    # deterministic first point: farthest from pool centroid; then classic FPS.
    centroid=P.mean(axis=0)
    first=int(np.argmax(np.sum((P-centroid)**2,axis=1)))
    chosen=np.empty(anchor_count,np.int64); chosen[0]=first
    d2=np.sum((P-P[first])**2,axis=1)
    for i in range(1,anchor_count):
        j=int(np.argmax(d2)); chosen[i]=j
        d2=np.minimum(d2,np.sum((P-P[j])**2,axis=1))
    return V[chosen], R[chosen], P[chosen].astype(np.float32)


def oracle_match_row(
    anchor_P,
    anchor_triangle_id: int,
    anchor_barycentric_uv,
    target: AuthorityView,
    *,
    radius_px: int = 2,
    max_reprojection_error_px: float = 1.75,
    max_surface_error: float = 0.006,
):
    """E0-a only: teacher physical identity is the source (triangle,bary) carrier.

    The exact carrier generally projects between target pixel centers, so target raster rows are only
    *visibility witnesses*. A target row can witness the carrier only if the z-buffer winner is the
    same source triangle near the carrier's exact continuous projection. Barycentric identity remains
    attached to the anchor and is emitted only in E0-a.
    """
    obs = target.observable
    anchor_P = np.asarray(anchor_P, np.float32)
    _anchor_bary = np.asarray(anchor_barycentric_uv, np.float32)  # explicit identity handle; not a score
    if _anchor_bary.shape != (2,):
        raise ValueError(f"expected barycentric_uv shape (2,), got {_anchor_bary.shape}")
    grid = project_grid(anchor_P[None], camera_for_view(obs.view))[0]
    if np.any(np.abs(grid) > 1.0):
        return -1, float("inf"), float("inf"), grid
    x, y = grid_to_nearest_pixel(grid[None], obs.resolution)
    cand = raster_lookup_near(obs.pixel_linear_index, x, y, obs.resolution, radius_px)[0]
    cand = cand[cand >= 0]
    if not len(cand):
        return -1, float("inf"), float("inf"), grid
    # Teacher identity gate: a nearby different surface may not stand in for the physical carrier.
    cand = cand[target.triangle_id[cand] == int(anchor_triangle_id)]
    if not len(cand):
        return -1, float("inf"), float("inf"), grid
    reproj = grid_distance_in_pixels(target.observable.grid[cand], np.broadcast_to(grid, (len(cand), 2)), obs.resolution)
    surf = np.linalg.norm(obs.P[cand] - anchor_P[None], axis=1)
    valid = (reproj <= max_reprojection_error_px) & (surf <= max_surface_error)
    if not np.any(valid):
        return -1, float(np.min(surf)), float(np.min(reproj)), grid
    score = reproj + surf / max(max_surface_error, 1e-8)
    score[~valid] = np.inf
    j = int(np.argmin(score)); best = int(cand[j])
    return best, float(surf[j]), float(reproj[j]), grid


def derived_match_row(
    anchor_P,
    anchor_N,
    anchor_grid,
    source_view: int,
    target: ObservableView,
    *,
    radius_px: int = 4,
    max_common_frame_error: float = 0.006,
    min_abs_normal_cos: float = math.cos(math.radians(50.0)),
    max_reciprocal_error_px: float = 3.0,
):
    """E0-b match. This API has no teacher provenance fields by construction."""
    anchor_P=np.asarray(anchor_P,np.float32)
    anchor_N=np.asarray(anchor_N,np.float32)
    tg=project_grid(anchor_P[None],camera_for_view(target.view))[0]
    if np.any(np.abs(tg)>1.0): return -1,{"reason":"out_of_frame"}
    x,y=grid_to_nearest_pixel(tg[None],target.resolution)
    cand=raster_lookup_near(target.pixel_linear_index,x,y,target.resolution,radius_px)[0]
    cand=cand[cand>=0]
    if not len(cand): return -1,{"reason":"no_raster_candidate"}

    pd=np.linalg.norm(target.P[cand]-anchor_P[None],axis=1)
    srcNnorm=float(np.linalg.norm(anchor_N))
    tgtN=target.N_derived[cand]
    tgtNnorm=np.linalg.norm(tgtN,axis=1)
    nvalid=(srcNnorm>0.5)&target.N_valid[cand]&(tgtNnorm>0.5)
    ndot=np.zeros(len(cand),np.float32)
    if np.any(nvalid): ndot[nvalid]=np.abs(tgtN[nvalid]@anchor_N)/(tgtNnorm[nvalid]*srcNnorm+1e-8)

    back=project_grid(target.P[cand],camera_for_view(source_view))
    recip=grid_distance_in_pixels(back,np.broadcast_to(anchor_grid,back.shape),target.resolution)
    valid=(pd<=max_common_frame_error)&nvalid&(ndot>=min_abs_normal_cos)&(recip<=max_reciprocal_error_px)
    if not np.any(valid):
        return -1,{"reason":"gated","candidate_count":int(len(cand)),"min_P_error":float(pd.min()),"max_abs_normal_cos":float(ndot.max()),"min_reciprocal_px":float(recip.min())}
    score=(pd/max_common_frame_error)+(1.0-ndot)+(recip/max_reciprocal_error_px)
    score[~valid]=np.inf
    j=int(np.argmin(score)); row=int(cand[j])
    return row,{"reason":"match","P_error":float(pd[j]),"abs_normal_cos":float(ndot[j]),"reciprocal_px":float(recip[j]),"score":float(score[j])}



def _seed64(label: str) -> int:
    return int(hashlib.sha256(label.encode()).hexdigest()[:16], 16) & 0x7fffffffffffffff


def _face_normals(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tri = vertices[faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area2 = np.linalg.norm(cross, axis=1)
    normals = np.zeros_like(cross, dtype=np.float32)
    good = area2 > 1e-12
    normals[good] = (cross[good] / area2[good, None]).astype(np.float32)
    areas = (0.5 * area2).astype(np.float64)
    return normals, areas


def deterministic_full_mesh_surface(geom: dict, asset_id: str, count: int = 512, *, stream: str = "downstream") -> dict:
    """Area-uniform full-mesh surface ceiling in the canonical RealSaS object frame.

    This is E0-0 authority only. It deliberately does not apply a pretrained consumer's bbox
    normalization; consumer-native normalization is a later OOD-adapter treatment.
    """
    vertices = np.asarray(geom['vertices'], np.float32)
    faces = np.asarray(geom['faces'], np.int64)
    fn, areas = _face_normals(vertices, faces)
    valid_faces = np.flatnonzero(areas > 1e-12)
    if not len(valid_faces):
        raise RuntimeError(f'{asset_id}: no nondegenerate full-mesh faces')
    probs = areas[valid_faces] / areas[valid_faces].sum()
    rng = np.random.default_rng(_seed64(f'E0-0|{asset_id}|{stream}|{count}'))
    chosen = rng.choice(valid_faces, size=count, replace=True, p=probs)
    r = rng.random((count, 2), dtype=np.float64)
    su = np.sqrt(r[:, 0])
    w0 = 1.0 - su
    w1 = su * (1.0 - r[:, 1])
    w2 = su * r[:, 1]
    tri = vertices[faces[chosen]]
    P = (tri[:, 0] * w0[:, None] + tri[:, 1] * w1[:, None] + tri[:, 2] * w2[:, None]).astype(np.float32)
    N = fn[chosen].astype(np.float32)
    return {
        'P': P,
        'N_geometric': N,
        'N_valid': np.ones(count, np.uint8),
        'sampling': np.array('AREA_UNIFORM_FULL_MESH'),
        'canonical_frame_preserved': np.array(1, np.uint8),
    }


def _bbox_center_scale(P: np.ndarray) -> tuple[np.ndarray, float]:
    P = np.asarray(P, np.float32)
    center = (P.max(axis=0) + P.min(axis=0)) * 0.5
    scale = float(np.max(np.abs(P - center[None])))
    return center.astype(np.float32), scale


def _nearest_distances(query: np.ndarray, reference: np.ndarray, chunk: int = 1024) -> np.ndarray:
    query = np.asarray(query, np.float32); reference = np.asarray(reference, np.float32)
    if not len(query) or not len(reference):
        raise ValueError('nearest-distance sets must be non-empty')
    out = np.empty(len(query), np.float32)
    for s in range(0, len(query), chunk):
        q = query[s:s+chunk]
        d2 = np.sum((q[:, None, :] - reference[None, :, :]) ** 2, axis=2)
        out[s:s+len(q)] = np.sqrt(d2.min(axis=1))
    return out


def full_vs_observable_distribution_metrics(geom: dict, asset_id: str, observable_P: np.ndarray, *, dense_reference_count: int = 8192) -> dict:
    """Measure the full-surface -> observable-union gap without any pretrained consumer."""
    full = deterministic_full_mesh_surface(geom, asset_id, dense_reference_count, stream='coverage_reference')['P']
    observable_P = np.asarray(observable_P, np.float32)
    f2o = _nearest_distances(full, observable_P)
    o2f = _nearest_distances(observable_P, full)
    fc, fs = _bbox_center_scale(full); oc, os = _bbox_center_scale(observable_P)
    return {
        'schema': 'RealSaS.E0.FullVsObservableDistribution.v1',
        'dense_full_reference_count': int(dense_reference_count),
        'observable_point_count': int(len(observable_P)),
        'full_to_observable_nn_p50': float(np.quantile(f2o, .50)),
        'full_to_observable_nn_p90': float(np.quantile(f2o, .90)),
        'full_to_observable_nn_p95': float(np.quantile(f2o, .95)),
        'full_to_observable_nn_max': float(f2o.max()),
        'observable_to_full_nn_p95': float(np.quantile(o2f, .95)),
        'full_bbox_center': fc.tolist(),
        'observable_bbox_center': oc.tolist(),
        'bbox_center_shift_l2': float(np.linalg.norm(fc-oc)),
        'full_bbox_maxabs_scale': fs,
        'observable_bbox_maxabs_scale': os,
        'observable_to_full_scale_ratio': (os/fs) if fs > 0 else None,
        'consumer_native_normalization_applied': False,
    }

def build_e0_asset(asset_dir: str | Path, *, anchor_count: int = 512) -> tuple[dict, dict, dict]:
    asset_dir=Path(asset_dir); asset_id=asset_dir.name
    geom,auth=load_asset_authority(asset_dir)
    obs=[a.observable for a in auth]
    src_views,src_rows,anchorP=deterministic_anchor_rows(auth,asset_id,anchor_count=anchor_count)
    K=len(anchorP)
    source_grid=np.zeros((K,2),np.float32); source_N=np.zeros((K,3),np.float32)
    source_N_valid=np.zeros(K,bool)
    # Teacher physical identity is attached only after the P-only anchor sampler has frozen the rows.
    # It is never included in the E0-b common payload or passed to derived_match_row.
    source_triangle=np.zeros(K,np.int64); source_bary=np.zeros((K,2),np.float32)
    for i,(sv,sr) in enumerate(zip(src_views,src_rows)):
        av=auth[int(sv)]
        source_grid[i]=obs[int(sv)].grid[int(sr)]
        source_N[i]=obs[int(sv)].N_derived[int(sr)]
        source_N_valid[i]=obs[int(sv)].N_valid[int(sr)]
        source_triangle[i]=av.triangle_id[int(sr)]
        source_bary[i]=av.barycentric_uv[int(sr)]

    a_rows=np.full((K,VIEWS),-1,np.int64); b_rows=np.full((K,VIEWS),-1,np.int64)
    a_err=np.full((K,VIEWS),np.inf,np.float32); a_reproj=np.full((K,VIEWS),np.inf,np.float32)
    b_p=np.full((K,VIEWS),np.nan,np.float32); b_n=np.full((K,VIEWS),np.nan,np.float32); b_recip=np.full((K,VIEWS),np.nan,np.float32)
    for i in range(K):
        sv=int(src_views[i]); sr=int(src_rows[i]); a_rows[i,sv]=sr; b_rows[i,sv]=sr; a_err[i,sv]=0.0; a_reproj[i,sv]=0.0; b_p[i,sv]=0.0; b_n[i,sv]=1.0; b_recip[i,sv]=0.0
        for tv in range(VIEWS):
            if tv==sv: continue
            ar,ae,arp,_=oracle_match_row(anchorP[i],source_triangle[i],source_bary[i],auth[tv]); a_rows[i,tv]=ar; a_err[i,tv]=ae; a_reproj[i,tv]=arp
            br,info=derived_match_row(anchorP[i],source_N[i],source_grid[i],sv,obs[tv]); b_rows[i,tv]=br
            if br>=0:
                b_p[i,tv]=info['P_error']; b_n[i,tv]=info['abs_normal_cos']; b_recip[i,tv]=info['reciprocal_px']

    # Strict evaluator: a B match is correct only where E0-a says the physical point is visible and
    # the B-selected visible raster sample is within the same frozen 0.003 surface witness radius.
    a_support=a_rows>=0; b_support=b_rows>=0
    correct=np.zeros_like(b_support)
    for i in range(K):
        for tv in range(VIEWS):
            br=b_rows[i,tv]
            if br>=0 and a_support[i,tv]:
                correct[i,tv]=np.linalg.norm(obs[tv].P[br]-anchorP[i])<=0.003
    # exclude source self-edge from pair metrics
    nonself=np.ones((K,VIEWS),bool); nonself[np.arange(K),src_views.astype(np.int64)]=False
    tp=int(np.sum(correct&nonself)); pred=int(np.sum(b_support&nonself)); truth=int(np.sum(a_support&nonself))
    fp=pred-tp
    missed=int(np.sum(a_support&(~b_support)&nonself))
    wrong_visible=int(np.sum(a_support&b_support&(~correct)&nonself))
    precision=tp/pred if pred else 1.0; recall=tp/truth if truth else 1.0
    metrics={
      'schema':'RealSaS.E0.ObservableGeometryPersistenceEval.v1','asset_id':asset_id,'anchor_count':K,
      'oracle_supported_pairs':truth,'derived_predicted_pairs':pred,'true_positive_pairs':tp,'false_match_pairs':fp,'missed_visible_pairs':missed,'wrong_visible_match_pairs':wrong_visible,
      'precision':precision,'recall':recall,'f1':(2*precision*recall/(precision+recall)) if precision+recall else 0.0,
      'oracle_support_count_per_anchor':{'median':float(np.median(a_support.sum(1))),'p10':float(np.quantile(a_support.sum(1),.1)),'min':int(a_support.sum(1).min()),'max':int(a_support.sum(1).max())},
      'derived_support_count_per_anchor':{'median':float(np.median(b_support.sum(1))),'p10':float(np.quantile(b_support.sum(1),.1)),'min':int(b_support.sum(1).min()),'max':int(b_support.sum(1).max())},
      'derived_abstention_pair_rate':float(np.mean((~b_support)&nonself)),
      'derived_match_P_error_p95':float(np.nanquantile(b_p,0.95)) if np.isfinite(b_p).any() else None,
      'derived_reciprocal_px_p95':float(np.nanquantile(b_recip,0.95)) if np.isfinite(b_recip).any() else None,
      'derived_abs_normal_cos_p05':float(np.nanquantile(b_n,0.05)) if np.isfinite(b_n).any() else None,
      'oracle_identity_mode':'source_triangle_plus_barycentric_carrier__target_same_triangle_visibility_witness','teacher_identity_used_by_e0_b':False,'camera_json_consumed':False,'scientific_optimizer_steps':0,
    }
    common={
      'P':anchorP,'N_source_derived':source_N,'N_source_valid':source_N_valid.astype(np.uint8),
      'source_view':src_views.astype(np.int16),'source_row':src_rows,'source_grid':source_grid,
    }
    arm_a={**common,'teacher_source_triangle_id':source_triangle,'teacher_source_barycentric_uv':source_bary,'support_mask':a_support.astype(np.uint8),'matched_row':a_rows,'surface_error':a_err,'reprojection_error_px':a_reproj}
    arm_b={**common,'support_mask':b_support.astype(np.uint8),'matched_row':b_rows,'match_P_error':b_p,'match_abs_normal_cos':b_n,'reciprocal_px':b_recip}
    return arm_a,arm_b,metrics
