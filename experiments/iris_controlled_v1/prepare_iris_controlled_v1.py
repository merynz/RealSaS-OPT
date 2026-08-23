from __future__ import annotations
import argparse, json, hashlib, os, tempfile, time
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image
from iris_geometry import (sha256_file, geometric_vertex_normals, reconstruct_surface,
                           pixel_to_grid, choose_visible_correspondence)

STYLES=('cel_clean_512.png','ink_cel_512.png')
VIEWS=8


def atomic_json(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    os.replace(tmp,path)


def atomic_npz(path,**arrs):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=path.name+'.',suffix='.npz',dir=path.parent); os.close(fd)
    try:
        np.savez_compressed(tmp,**arrs); os.replace(tmp,path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def seed_for(asset,tag):
    return int(hashlib.sha256((asset+'|'+tag).encode()).hexdigest()[:16],16) & 0x7fffffff


def load_rgba(path):
    im=Image.open(path).convert('RGBA')
    if im.size!=(512,512): raise RuntimeError(f'expected 512x512: {path} got {im.size}')
    return np.asarray(im,dtype=np.uint8)


def build_asset(root,record,out_path,geom_samples=512,anchors_per_view=48,max_tracks=384,
                track_radius_px=3,max_surface_error=.003):
    aid=record['asset_id']; ar=root/'master'/'assets'/aid
    gp=ar/'primary_geometry.npz'
    if not gp.exists(): raise FileNotFoundError(gp)
    gz=np.load(gp,allow_pickle=False); vertices=gz['vertices'].astype(np.float32); faces=gz['faces'].astype(np.int32)
    gnorm=geometric_vertex_normals(vertices,faces)
    ras=[]; cams=[]; images=np.empty((2,VIEWS,512,512,4),np.uint8); source_sha={str(gp.relative_to(root)):sha256_file(gp)}
    for v in range(VIEWS):
        vd=ar/'renders'/f'V{v}'
        rp=vd/'raster_authority.npz'; cp=vd/'camera.json'
        if not rp.exists() or not cp.exists(): raise FileNotFoundError(f'{aid} V{v} raster/camera missing')
        ra=np.load(rp,allow_pickle=False)
        ra_d={k:ra[k] for k in ra.files}; ras.append(ra_d); cams.append(json.load(open(cp)))
        source_sha[str(rp.relative_to(root))]=sha256_file(rp); source_sha[str(cp.relative_to(root))]=sha256_file(cp)
        for si,sn in enumerate(STYLES):
            ip=vd/sn
            if not ip.exists(): raise FileNotFoundError(ip)
            images[si,v]=load_rgba(ip); source_sha[str(ip.relative_to(root))]=sha256_file(ip)
    yaws=np.asarray([float(c['yaw_deg']) for c in cams],np.float32)
    if not np.allclose(yaws,np.arange(8,dtype=np.float32)*45.0,atol=1e-4):
        raise RuntimeError(f'{aid}: noncanonical controlled yaw sequence {yaws.tolist()}')

    # Dense geometry supervision samples, deterministic per asset/view.
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
        geom_xy[v,:take]=pixel_to_grid(ra['pixel_linear_index'][idx],int(ra['resolution'][0])); geom_p[v,:take]=p; geom_n[v,:take]=nrm; geom_mask[v,:take]=1
        at=min(anchors_per_view,n); ai=rng.choice(n,size=at,replace=False)
        ap,an=reconstruct_surface(vertices,faces,gnorm,ra['triangle_id'][ai],ra['barycentric_uv'][ai])
        anchor_ps.append(ap); anchor_ns.append(an)

    # Build exact multi-view surface tracks. A track is kept only when the same physical locus is observed in >=2 views.
    P=np.concatenate(anchor_ps,0); N=np.concatenate(anchor_ns,0)
    # Quantized dedupe prevents one large visible region from dominating the track pool.
    q=np.round(P/1e-4).astype(np.int64); _,uniq=np.unique(q,axis=0,return_index=True); uniq=np.sort(uniq); P=P[uniq]; N=N[uniq]
    T=len(P); track_xy=np.zeros((T,VIEWS,2),np.float32); track_vis=np.zeros((T,VIEWS),np.uint8); track_err=np.full((T,VIEWS),np.inf,np.float32)
    track_p_view=np.zeros((T,VIEWS,3),np.float32); track_n_view=np.zeros((T,VIEWS,3),np.float32)
    for v in range(VIEWS):
        ok,g,err,row=choose_visible_correspondence(P,cams[v],ras[v],vertices,faces,gnorm,
                                                    radius_px=track_radius_px,max_surface_error=max_surface_error)
        track_vis[:,v]=ok.astype(np.uint8); track_xy[ok,v]=g[ok]; track_err[:,v]=err
        if np.any(ok):
            pp,nn=reconstruct_surface(vertices,faces,gnorm,ras[v]['triangle_id'][row[ok]],ras[v]['barycentric_uv'][row[ok]])
            track_p_view[ok,v]=pp; track_n_view[ok,v]=nn
    support=track_vis.sum(1); keep=np.flatnonzero(support>=2)
    if len(keep)==0: raise RuntimeError(f'{aid}: no cross-view tracks')
    # Prefer high-support tracks, deterministic hash tiebreak from physical position.
    if len(keep)>max_tracks:
        h=np.array([int(hashlib.sha256(np.asarray(P[i],np.float32).tobytes()).hexdigest()[:16],16) for i in keep],dtype=np.uint64)
        order=np.lexsort((h,-support[keep])); keep=keep[order[:max_tracks]]
    P=P[keep]; N=N[keep]; track_xy=track_xy[keep]; track_vis=track_vis[keep]; track_err=track_err[keep]; support=support[keep]
    track_p_view=track_p_view[keep]; track_n_view=track_n_view[keep]

    arrays=dict(images=images,yaw_deg=yaws,geom_xy=geom_xy,geom_p=geom_p,geom_n=geom_n,geom_mask=geom_mask,
                track_p=P.astype(np.float32),track_n=N.astype(np.float32),track_xy=track_xy,track_visible=track_vis,
                track_p_view=track_p_view,track_n_view=track_n_view,track_support=support.astype(np.uint8),track_surface_error=track_err,
                asset_id=np.asarray([aid]),split=np.asarray([record['split']]))
    atomic_npz(out_path,**arrays)
    return {
        'asset_id':aid,'split':record['split'],'cache_path':str(out_path),'cache_sha256':sha256_file(out_path),
        'cache_size_bytes':out_path.stat().st_size,'source_files_sha256':source_sha,
        'geom_samples_valid':int(geom_mask.sum()),'track_count':int(len(P)),
        'track_support_hist':{str(k):int((support==k).sum()) for k in range(2,9)},
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',default='/content/drive/MyDrive/RealSaS_MASTER_CORPUS_1024_V3')
    ap.add_argument('--seed-manifest',required=True)
    ap.add_argument('--out',default=None)
    ap.add_argument('--splits',default='FIT,TUNE')
    ap.add_argument('--authorize-sealed',default='')
    ap.add_argument('--geom-samples',type=int,default=512); ap.add_argument('--anchors-per-view',type=int,default=48); ap.add_argument('--max-tracks',type=int,default=384)
    ap.add_argument('--limit',type=int,default=0); ap.add_argument('--resume',action='store_true',default=True)
    a=ap.parse_args()
    root=Path(a.root); out=Path(a.out) if a.out else root/'cache'/'IRIS_CONTROLLED_V1'
    seed=json.load(open(a.seed_manifest)); requested=[s.strip().upper() for s in a.splits.split(',') if s.strip()]
    sealed={'CAL','DEV','EXTERNAL_HOLDOUT'}; auth={s.strip().upper() for s in a.authorize_sealed.split(',') if s.strip()}
    bad=[s for s in requested if s in sealed and s not in auth]
    if bad: raise RuntimeError(f'FAIL CLOSED: sealed split requested without --authorize-sealed: {bad}')
    records=[r for r in seed['records'] if r['split'] in requested]
    if a.limit: records=records[:a.limit]
    (out/'assets').mkdir(parents=True,exist_ok=True); progress_path=out/'PREPARE_PROGRESS.json'
    done={}
    if progress_path.exists():
        try: done={x['asset_id']:x for x in json.load(open(progress_path)).get('records',[])}
        except Exception: done={}
    rows=[]; start=time.time()
    for i,r in enumerate(records,1):
        aid=r['asset_id']; cp=out/'assets'/f'{aid}.npz'
        if aid in done and cp.exists() and sha256_file(cp)==done[aid].get('cache_sha256'):
            row=done[aid]
        else:
            row=build_asset(root,r,cp,a.geom_samples,a.anchors_per_view,a.max_tracks)
        rows.append(row)
        if i%10==0 or i==len(records):
            atomic_json(progress_path,{'schema':'RealSaS.IRISControlledV1.PrepareProgress.v1','requested_splits':requested,'records':rows})
            print(f'[prepare] {i}/{len(records)} tracks_median={np.median([x["track_count"] for x in rows]):.1f}')
    manifest={'schema':'RealSaS.IRISControlledV1.CacheManifest.v1','date':'2026-08-23','root':str(root),'requested_splits':requested,
              'record_count':len(rows),'records':rows,'source_seed_sha256':sha256_file(a.seed_manifest),
              'settings':{'geom_samples':a.geom_samples,'anchors_per_view':a.anchors_per_view,'max_tracks':a.max_tracks,
                          'styles':list(STYLES),'views':VIEWS,'normal_authority':'area_weighted_geometric_vertex_normals_from_vertices_faces'},
              'elapsed_sec':time.time()-start}
    mp=out/'IRIS_CONTROLLED_V1_CACHE_MANIFEST.json'; atomic_json(mp,manifest)
    seal={'manifest':str(mp),'manifest_sha256':sha256_file(mp),'record_count':len(rows),'splits':dict(Counter(x['split'] for x in rows)),
          'cache_bytes':sum(x['cache_size_bytes'] for x in rows)}
    atomic_json(out/'IRIS_CONTROLLED_V1_CACHE_SEAL.json',seal)
    print(json.dumps(seal,indent=2))
if __name__=='__main__': main()
