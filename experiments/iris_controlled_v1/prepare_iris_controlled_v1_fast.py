from __future__ import annotations
import argparse, json, hashlib, os, tempfile, time, threading
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from PIL import Image
from iris_geometry import (sha256_file, geometric_vertex_normals, reconstruct_surface,
                           pixel_to_grid, project_grid, grid_to_nearest_pixel,
                           raster_lookup_near, choose_visible_correspondence)

STYLES=('cel_clean_512.png','ink_cel_512.png')
VIEWS=8
CACHE_RES=256


def atomic_json(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    os.replace(tmp,path)


def atomic_npz_uncompressed(path,**arrs):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=path.name+'.',suffix='.npz',dir=path.parent); os.close(fd)
    try:
        np.savez(tmp,**arrs)
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def seed_for(asset,tag):
    return int(hashlib.sha256((asset+'|'+tag).encode()).hexdigest()[:16],16) & 0x7fffffff


def load_rgba_256(path):
    with Image.open(path) as im:
        im=im.convert('RGBA')
        if im.size!=(CACHE_RES,CACHE_RES):
            im=im.resize((CACHE_RES,CACHE_RES),resample=Image.Resampling.BILINEAR)
        return np.asarray(im,dtype=np.uint8)


def choose_visible_correspondence_fast(points,target_camera,target_ra,vertices,faces,vertex_normals,
                                       radius_px=3,max_surface_error=.003):
    """Vectorized equivalent of iris_geometry.choose_visible_correspondence.

    Candidate ordering is identical to raster_lookup_near. np.argmin keeps the first tie,
    matching the old strict-< loop update semantics.
    """
    points=np.asarray(points,np.float32)
    res=int(np.asarray(target_ra['resolution']).reshape(-1)[0])
    grid=project_grid(points,target_camera)
    x,y=grid_to_nearest_pixel(grid,res)
    in_frame=(np.abs(grid[:,0])<=1)&(np.abs(grid[:,1])<=1)
    cand,_=raster_lookup_near(target_ra['pixel_linear_index'],x,y,res,radius_px)
    n,w=cand.shape
    valid=in_frame[:,None]&(cand>=0)
    errmat=np.full((n,w),np.inf,np.float32)
    flat=np.flatnonzero(valid.ravel())
    if flat.size:
        qi=(flat//w).astype(np.int64)
        rows=cand.ravel()[flat]
        pp,_=reconstruct_surface(vertices,faces,vertex_normals,
                                 target_ra['triangle_id'][rows],
                                 target_ra['barycentric_uv'][rows])
        err=np.linalg.norm(pp-points[qi],axis=1).astype(np.float32)
        errmat.ravel()[flat]=err
    best_j=np.argmin(errmat,axis=1)
    best_err=errmat[np.arange(n),best_j]
    best_row=cand[np.arange(n),best_j].astype(np.int64)
    ok=best_err<=float(max_surface_error)
    best_row[~ok]=-1
    target_grid=np.zeros((n,2),np.float32)
    if np.any(ok):
        target_grid[ok]=pixel_to_grid(target_ra['pixel_linear_index'][best_row[ok]],res)
    return ok,target_grid,best_err,best_row


def correspondence_parity_check(root,record):
    aid=record['asset_id']; ar=root/'master'/'assets'/aid; gp=ar/'primary_geometry.npz'
    with np.load(gp,allow_pickle=False) as gz:
        vertices=gz['vertices'].astype(np.float32); faces=gz['faces'].astype(np.int32)
    gnorm=geometric_vertex_normals(vertices,faces); ras=[]; cams=[]
    for v in range(VIEWS):
        vd=ar/'renders'/f'V{v}'
        with np.load(vd/'raster_authority.npz',allow_pickle=False) as ra:
            ras.append({k:ra[k] for k in ra.files})
        cams.append(json.load(open(vd/'camera.json')))
    ra0=ras[0]; n=len(ra0['pixel_linear_index']); rng=np.random.default_rng(seed_for(aid,'fast_parity'))
    idx=rng.choice(n,size=min(24,n),replace=False)
    points,_=reconstruct_surface(vertices,faces,gnorm,ra0['triangle_id'][idx],ra0['barycentric_uv'][idx])
    max_err_diff=0.0; row_mismatch=0; ok_mismatch=0
    for v in range(VIEWS):
        a=choose_visible_correspondence(points,cams[v],ras[v],vertices,faces,gnorm,radius_px=3,max_surface_error=.003)
        b=choose_visible_correspondence_fast(points,cams[v],ras[v],vertices,faces,gnorm,radius_px=3,max_surface_error=.003)
        ok_mismatch += int(np.sum(a[0]!=b[0])); both=a[0]&b[0]
        if np.any(both):
            row_mismatch += int(np.sum(a[3][both]!=b[3][both]))
            max_err_diff=max(max_err_diff,float(np.max(np.abs(a[2][both]-b[2][both]))))
    result={'asset_id':aid,'ok_mismatch':ok_mismatch,'row_mismatch':row_mismatch,'max_err_diff':max_err_diff}
    if ok_mismatch or row_mismatch or max_err_diff>1e-6:
        raise RuntimeError('fast correspondence parity failed: '+json.dumps(result))
    return result


def build_asset_fast(root,record,out_dir,geom_samples=512,anchors_per_view=48,max_tracks=384,
                     track_radius_px=3,max_surface_error=.003):
    aid=record['asset_id']; ar=root/'master'/'assets'/aid
    cp=out_dir/'assets'/f'{aid}.npz'; mp=out_dir/'meta'/f'{aid}.json'
    if cp.exists() and mp.exists():
        try:
            meta=json.load(open(mp))
            if meta.get('cache_size_bytes')==cp.stat().st_size and meta.get('cache_sha256')==sha256_file(cp):
                return meta
        except Exception:
            pass

    t0=time.time(); gp=ar/'primary_geometry.npz'
    if not gp.exists(): raise FileNotFoundError(gp)
    with np.load(gp,allow_pickle=False) as gz:
        vertices=gz['vertices'].astype(np.float32); faces=gz['faces'].astype(np.int32)
    gnorm=geometric_vertex_normals(vertices,faces)

    ras=[]; cams=[]; images=np.empty((2,VIEWS,CACHE_RES,CACHE_RES,4),np.uint8)
    for v in range(VIEWS):
        vd=ar/'renders'/f'V{v}'; rp=vd/'raster_authority.npz'; cam_p=vd/'camera.json'
        if not rp.exists() or not cam_p.exists(): raise FileNotFoundError(f'{aid} V{v} raster/camera missing')
        with np.load(rp,allow_pickle=False) as ra:
            ras.append({k:ra[k] for k in ra.files})
        cams.append(json.load(open(cam_p)))
        for si,sn in enumerate(STYLES):
            ip=vd/sn
            if not ip.exists(): raise FileNotFoundError(ip)
            images[si,v]=load_rgba_256(ip)

    yaws=np.asarray([float(c['yaw_deg']) for c in cams],np.float32)
    if not np.allclose(yaws,np.arange(8,dtype=np.float32)*45.0,atol=1e-4):
        raise RuntimeError(f'{aid}: noncanonical controlled yaw sequence {yaws.tolist()}')

    geom_xy=np.zeros((VIEWS,geom_samples,2),np.float32)
    geom_p=np.zeros((VIEWS,geom_samples,3),np.float32)
    geom_n=np.zeros((VIEWS,geom_samples,3),np.float32)
    geom_mask=np.zeros((VIEWS,geom_samples),np.uint8)
    anchor_ps=[]; anchor_ns=[]
    for v,ra in enumerate(ras):
        n=len(ra['pixel_linear_index'])
        if n==0: raise RuntimeError(f'{aid}: empty raster V{v}')
        rng=np.random.default_rng(seed_for(aid,f'geom{v}'))
        take=min(geom_samples,n); idx=rng.choice(n,size=take,replace=False)
        p,nrm=reconstruct_surface(vertices,faces,gnorm,ra['triangle_id'][idx],ra['barycentric_uv'][idx])
        geom_xy[v,:take]=pixel_to_grid(ra['pixel_linear_index'][idx],int(np.asarray(ra['resolution']).reshape(-1)[0]))
        geom_p[v,:take]=p; geom_n[v,:take]=nrm; geom_mask[v,:take]=1
        at=min(anchors_per_view,n); ai=rng.choice(n,size=at,replace=False)
        ap,an=reconstruct_surface(vertices,faces,gnorm,ra['triangle_id'][ai],ra['barycentric_uv'][ai])
        anchor_ps.append(ap); anchor_ns.append(an)

    P=np.concatenate(anchor_ps,0); N=np.concatenate(anchor_ns,0)
    q=np.round(P/1e-4).astype(np.int64); _,uniq=np.unique(q,axis=0,return_index=True); uniq=np.sort(uniq); P=P[uniq]; N=N[uniq]
    T=len(P); track_xy=np.zeros((T,VIEWS,2),np.float32); track_vis=np.zeros((T,VIEWS),np.uint8); track_err=np.full((T,VIEWS),np.inf,np.float32)
    track_p_view=np.zeros((T,VIEWS,3),np.float32); track_n_view=np.zeros((T,VIEWS,3),np.float32)
    for v in range(VIEWS):
        ok,g,err,row=choose_visible_correspondence_fast(P,cams[v],ras[v],vertices,faces,gnorm,
                                                        radius_px=track_radius_px,max_surface_error=max_surface_error)
        track_vis[:,v]=ok.astype(np.uint8); track_xy[ok,v]=g[ok]; track_err[:,v]=err
        if np.any(ok):
            pp,nn=reconstruct_surface(vertices,faces,gnorm,ras[v]['triangle_id'][row[ok]],ras[v]['barycentric_uv'][row[ok]])
            track_p_view[ok,v]=pp; track_n_view[ok,v]=nn
    support=track_vis.sum(1); keep=np.flatnonzero(support>=2)
    if len(keep)==0: raise RuntimeError(f'{aid}: no cross-view tracks')
    if len(keep)>max_tracks:
        h=np.array([int(hashlib.sha256(np.asarray(P[i],np.float32).tobytes()).hexdigest()[:16],16) for i in keep],dtype=np.uint64)
        order=np.lexsort((h,-support[keep])); keep=keep[order[:max_tracks]]
    P=P[keep]; N=N[keep]; track_xy=track_xy[keep]; track_vis=track_vis[keep]; track_err=track_err[keep]; support=support[keep]
    track_p_view=track_p_view[keep]; track_n_view=track_n_view[keep]

    arrays=dict(images=images,yaw_deg=yaws,geom_xy=geom_xy,geom_p=geom_p,geom_n=geom_n,geom_mask=geom_mask,
                track_p=P.astype(np.float32),track_n=N.astype(np.float32),track_xy=track_xy,track_visible=track_vis,
                track_p_view=track_p_view,track_n_view=track_n_view,track_support=support.astype(np.uint8),track_surface_error=track_err,
                asset_id=np.asarray([aid]),split=np.asarray([record['split']]),cache_resolution=np.asarray([CACHE_RES],np.int32))
    atomic_npz_uncompressed(cp,**arrays)
    meta={'asset_id':aid,'split':record['split'],'cache_path':str(cp),'cache_sha256':sha256_file(cp),'cache_size_bytes':cp.stat().st_size,
          'geom_samples_valid':int(geom_mask.sum()),'track_count':int(len(P)),
          'track_support_hist':{str(k):int((support==k).sum()) for k in range(2,9)},'elapsed_sec':time.time()-t0}
    atomic_json(mp,meta)
    return meta


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',default='/content/drive/MyDrive/RealSaS_MASTER_CORPUS_1024_V3')
    ap.add_argument('--seed-manifest',required=True)
    ap.add_argument('--out',default='/content/IRIS_CONTROLLED_V1_FAST_CACHE')
    ap.add_argument('--splits',default='FIT,TUNE')
    ap.add_argument('--workers',type=int,default=8)
    ap.add_argument('--geom-samples',type=int,default=512); ap.add_argument('--anchors-per-view',type=int,default=48); ap.add_argument('--max-tracks',type=int,default=384)
    ap.add_argument('--limit',type=int,default=0)
    a=ap.parse_args(); root=Path(a.root); out=Path(a.out); out.mkdir(parents=True,exist_ok=True); (out/'assets').mkdir(exist_ok=True); (out/'meta').mkdir(exist_ok=True)
    seed=json.load(open(a.seed_manifest)); requested=[s.strip().upper() for s in a.splits.split(',') if s.strip()]
    sealed={'CAL','DEV','EXTERNAL_HOLDOUT'}
    bad=[s for s in requested if s in sealed]
    if bad: raise RuntimeError(f'FAIL CLOSED: fast prep is open-splits only, got {bad}')
    records=[r for r in seed['records'] if r['split'] in requested]
    if a.limit: records=records[:a.limit]
    workers=max(1,min(int(a.workers),16)); print(f'[fast-prep] records={len(records)} workers={workers} out={out}',flush=True)
    parity=correspondence_parity_check(root,records[0]); print('[fast-prep] correspondence parity PASS',json.dumps(parity),flush=True)
    rows=[]; failures=[]; t0=time.time(); lock=threading.Lock()
    def job(r): return build_asset_fast(root,r,out,a.geom_samples,a.anchors_per_view,a.max_tracks)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs={ex.submit(job,r):r for r in records}
        done=0
        for fut in as_completed(futs):
            r=futs[fut]; done+=1
            try: rows.append(fut.result())
            except Exception as e: failures.append({'asset_id':r['asset_id'],'error':repr(e)})
            if done%10==0 or done==len(records):
                elapsed=time.time()-t0; rate=done/max(elapsed,1e-9); eta=(len(records)-done)/max(rate,1e-9)
                med=float(np.median([x['track_count'] for x in rows])) if rows else 0.0
                print(f'[fast-prep] {done}/{len(records)} ok={len(rows)} fail={len(failures)} rate={rate:.3f} asset/s ETA_min={eta/60:.1f} tracks_median={med:.1f}',flush=True)
                atomic_json(out/'FAST_PREP_PROGRESS.json',{'schema':'RealSaS.IRISControlledV1.FastPrepProgress.v1','done':done,'ok':len(rows),'failures':failures[-20:],'elapsed_sec':elapsed,'rate_assets_per_sec':rate,'eta_sec':eta})
    rows.sort(key=lambda x:x['asset_id'])
    if failures:
        atomic_json(out/'FAST_PREP_FAILURES.json',{'failures':failures}); raise RuntimeError(f'fast prep failures={len(failures)}; see FAST_PREP_FAILURES.json')
    manifest={'schema':'RealSaS.IRISControlledV1.CacheManifest.FastV2.v1','date':'2026-08-23','root':str(root),'requested_splits':requested,
              'record_count':len(rows),'records':rows,'source_seed_sha256':sha256_file(a.seed_manifest),
              'settings':{'cache_resolution':CACHE_RES,'image_resize':'PIL_RGBA_BILINEAR_512_to_256','geom_samples':a.geom_samples,
                          'anchors_per_view':a.anchors_per_view,'max_tracks':a.max_tracks,'styles':list(STYLES),'views':VIEWS,
                          'normal_authority':'area_weighted_geometric_vertex_normals_from_vertices_faces',
                          'source_per_file_rehash':'DISABLED__PARENT_CORPUS_AUTHORITY_USED','cache_storage':'uncompressed_npz_local_ssd'},
              'correspondence_parity':parity,'elapsed_sec':time.time()-t0}
    mp=out/'IRIS_CONTROLLED_V1_CACHE_MANIFEST_FAST_V2.json'; atomic_json(mp,manifest)
    seal={'schema':'RealSaS.IRISControlledV1.CacheSeal.FastV2.v1','manifest':str(mp),'manifest_sha256':sha256_file(mp),'record_count':len(rows),
          'splits':dict(Counter(x['split'] for x in rows)),'cache_bytes':sum(x['cache_size_bytes'] for x in rows),'elapsed_sec':time.time()-t0}
    atomic_json(out/'IRIS_CONTROLLED_V1_CACHE_SEAL_FAST_V2.json',seal); print(json.dumps(seal,indent=2),flush=True)

if __name__=='__main__': main()
