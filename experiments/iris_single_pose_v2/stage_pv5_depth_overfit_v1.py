from __future__ import annotations
import argparse, hashlib, json, os, shutil, tempfile
from pathlib import Path
import numpy as np
from PIL import Image

VIEWS=8; STYLES=('cel_clean','ink_cel'); YAWS=[float(i*45) for i in range(8)]

def sha256_file(path,chunk=8<<20):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(chunk),b''): h.update(b)
    return h.hexdigest()
def atomic_json(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n',encoding='utf-8'); os.replace(tmp,path)
def bbox_span(mask,axis):
    occ=mask.any(axis=0 if axis=='x' else 1); idx=np.flatnonzero(occ)
    if not len(idx): raise RuntimeError('blank alpha support')
    return int(idx[-1]-idx[0]+1)
def native_h_from_paths(paths):
    if len(paths)!=8: raise ValueError('need 8 ordered native views')
    alpha=[]
    for p in paths:
        with Image.open(p) as im:
            if im.mode!='RGBA' or im.size!=(1024,1024): raise RuntimeError(f'bad native {p}: {im.mode} {im.size}')
            alpha.append(np.asarray(im.getchannel('A'),np.uint8))
    fg=[a>=128 for a in alpha]; w0=bbox_span(fg[0],'x')/1024.0; w2=bbox_span(fg[2],'x')/1024.0; hz=max(bbox_span(m,'y')/1024.0 for m in fg)
    return float(0.5/max(w0,w2,hz))
def stage_asset(root,out,aid):
    src=root/'master'/'assets'/aid
    if not src.is_dir(): raise FileNotFoundError(src)
    final=out/'assets'/aid; tmp=Path(tempfile.mkdtemp(prefix=aid+'.partial.',dir=str(out/'assets')))
    try:
        with np.load(src/'primary_geometry.npz',allow_pickle=False) as z:
            if 'vertices' not in z.files or 'faces' not in z.files: raise RuntimeError(f'geometry missing {aid}')
            np.savez(tmp/'primary_geometry.npz',vertices=np.asarray(z['vertices'],np.float32),faces=np.asarray(z['faces'],np.int32))
        views=[]; native_hashes={}; raster_hashes={}; deriv_hashes={}
        for v in range(VIEWS):
            sv=src/'renders'/f'V{v}'; dv=tmp/'renders'/f'V{v}'; dv.mkdir(parents=True,exist_ok=True)
            ra=sv/'raster_authority.npz'; shutil.copy2(ra,dv/'raster_authority.npz'); raster_hashes[f'V{v}']=sha256_file(ra); style_meta={}
            for style in STYLES:
                native=sv/f'{style}.png'; deriv=sv/f'{style}_512.png'; native_hashes[f'V{v}/{style}']=sha256_file(native); deriv_hashes[f'V{v}/{style}']=sha256_file(deriv)
                with Image.open(deriv) as im:
                    if im.mode!='RGBA' or im.size!=(512,512): raise RuntimeError(f'bad derivative {deriv}')
                    im.resize((256,256),resample=Image.Resampling.BILINEAR).save(dv/f'{style}_input.png')
                style_meta[style]={'native_sha256':native_hashes[f'V{v}/{style}'],'derivative_512_sha256':deriv_hashes[f'V{v}/{style}'],'input_transform':'PIL_RGBA_BILINEAR_512_to_256'}
            views.append({'view':v,'yaw_deg':YAWS[v],'styles':style_meta})
        hs={style:native_h_from_paths([src/'renders'/f'V{v}'/f'{style}.png' for v in range(VIEWS)]) for style in STYLES}
        meta={'schema':'RealSaS.IRISSinglePoseV2.PV5DepthOverfitStageAsset.v1','asset_id':aid,'input_resolution':256,'native_scale_source_resolution':1024,'canonical_yaw_deg':YAWS,'sheet_half_extent_by_style':hs,'camera_json_consumed':False,'teacher_camera_half_extent_consumed':False,'geometry_sha256':sha256_file(src/'primary_geometry.npz'),'raster_sha256':raster_hashes,'native_rgba_sha256':native_hashes,'derivative_512_sha256':deriv_hashes,'views':views}; atomic_json(tmp/'STAGE.json',meta)
        if final.exists(): shutil.rmtree(final)
        os.replace(tmp,final); return meta
    finally:
        if tmp.exists(): shutil.rmtree(tmp,ignore_errors=True)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',required=True); ap.add_argument('--membership',required=True); ap.add_argument('--out',required=True); a=ap.parse_args(); root=Path(a.root); out=Path(a.out); (out/'assets').mkdir(parents=True,exist_ok=True); mem=json.load(open(a.membership,encoding='utf-8')); rows=mem['records']
    if len(rows)!=8 or any(r.get('split')!='FIT' for r in rows): raise RuntimeError('FIT-only 8-asset membership required')
    result=[]
    for i,r in enumerate(rows,1):
        m=stage_asset(root,out,r['asset_id']); result.append({'asset_id':r['asset_id'],'split':'FIT','asset_dir':str(out/'assets'/r['asset_id']),'sheet_half_extent_by_style':m['sheet_half_extent_by_style']}); print(f'[stage-pv5] {i}/8 {r["asset_id"]}',flush=True)
    atomic_json(out/'STAGE_MANIFEST.json',{'schema':'RealSaS.IRISSinglePoseV2.PV5DepthOverfitStageManifest.v1','record_count':8,'input_resolution':256,'camera_json_consumed':False,'records':result})
if __name__=='__main__': main()
