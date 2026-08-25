from __future__ import annotations
import argparse, hashlib, json, os
from pathlib import Path
import numpy as np
from geometry import geometric_vertex_normals,reconstruct_surface,pixel_linear_to_grid,sha256_file
STYLES=('cel_clean','ink_cel')
def atomic_json(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n',encoding='utf-8'); os.replace(tmp,path)
def seed_for(aid,v): return int(hashlib.sha256(f'{aid}|pv5-depth|{v}'.encode()).hexdigest()[:16],16)&0x7fffffff
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--stage-manifest',required=True); ap.add_argument('--out',required=True); ap.add_argument('--samples-per-view',type=int,default=4096); a=ap.parse_args()
    sm=json.load(open(a.stage_manifest,encoding='utf-8')); rows=sm.get('records',[])
    if sm.get('record_count')!=16 or sm.get('asset_count')!=8 or sm.get('asset_style_cells')!=16 or sm.get('camera_json_consumed') is not False: raise RuntimeError('8x2 stage authority drift')
    assets=[]
    for r in rows:
        if r['asset_id'] not in assets: assets.append(r['asset_id'])
    out=Path(a.out); (out/'truth').mkdir(parents=True,exist_ok=True); S=int(a.samples_per_view); records=[]; truth_shas={}
    for aid in assets:
        arows=[r for r in rows if r['asset_id']==aid]
        if {r['style'] for r in arows}!=set(STYLES): raise RuntimeError(f'style coverage drift {aid}')
        ar=Path(arows[0]['cell_dir'])
        with np.load(ar/'primary_geometry.npz',allow_pickle=False) as z: vertices=np.asarray(z['vertices'],np.float32); faces=np.asarray(z['faces'],np.int32)
        vn=geometric_vertex_normals(vertices,faces); xy=np.zeros((8,S,2),np.float32); p=np.zeros((8,S,3),np.float32); mask=np.zeros((8,S),np.uint8)
        for v in range(8):
            with np.load(ar/'renders'/f'V{v}'/'raster_authority.npz',allow_pickle=False) as ra:
                n=len(ra['pixel_linear_index'])
                if n<S: raise RuntimeError(f'{aid} V{v} has only {n} visible rows; prereg requires {S}')
                rng=np.random.default_rng(seed_for(aid,v)); ids=np.sort(rng.choice(n,size=S,replace=False))
                pp,_=reconstruct_surface(vertices,faces,vn,ra['triangle_id'][ids],ra['barycentric_uv'][ids]); res=int(np.asarray(ra['resolution']).reshape(-1)[0])
                xy[v]=pixel_linear_to_grid(ra['pixel_linear_index'][ids],res); p[v]=pp; mask[v]=1
        truth=out/'truth'/f'{aid}.npz'; np.savez(truth,geom_xy=xy,geom_p=p,geom_mask=mask,yaw_deg=np.arange(8,dtype=np.float32)*45.0); tsha=sha256_file(truth); truth_shas[aid]=tsha
        for style in STYLES:
            r=next(x for x in arows if x['style']==style); records.append({'asset_id':aid,'style':style,'split':'FIT','cell_dir':r['cell_dir'],'truth_path':str(truth),'truth_sha256':tsha,'sheet_half_extent':float(r['sheet_half_extent'])})
    atomic_json(out/'CACHE_MANIFEST.json',{'schema':'RealSaS.IRISSinglePoseV2.PV5R256EightByTwoCacheManifest.v1','record_count':16,'asset_count':8,'asset_style_cells':16,'styles':list(STYLES),'samples_per_view':S,'shared_truth_loci_across_styles':True,'camera_json_consumed':False,'truth_sha256_by_asset':truth_shas,'records':records})
    print(f'[cache-r256-8x2] assets={len(assets)} cells={len(records)}',flush=True)
if __name__=='__main__': main()
