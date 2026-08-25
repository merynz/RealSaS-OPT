from __future__ import annotations
import argparse, hashlib, json, os, shutil, tempfile
from pathlib import Path
import numpy as np
from PIL import Image

VIEWS=8; YAWS=[float(i*45) for i in range(8)]; STYLES=('cel_clean','ink_cel')

def sha256_file(path,chunk=8<<20):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for x in iter(lambda:f.read(chunk),b''): h.update(x)
    return h.hexdigest()

def atomic_json(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n',encoding='utf-8'); os.replace(tmp,path)

def bbox_span(mask,axis):
    occ=mask.any(axis=0 if axis=='x' else 1); idx=np.flatnonzero(occ)
    if not len(idx): raise RuntimeError('blank alpha support')
    return int(idx[-1]-idx[0]+1)

def native_h(paths):
    alpha=[]
    for p in paths:
        with Image.open(p) as im:
            if im.mode!='RGBA' or im.size!=(1024,1024): raise RuntimeError(f'bad native {p}: {im.mode} {im.size}')
            alpha.append(np.asarray(im.getchannel('A'),np.uint8))
    fg=[a>=128 for a in alpha]
    w0=bbox_span(fg[0],'x')/1024.0; w2=bbox_span(fg[2],'x')/1024.0; hz=max(bbox_span(m,'y')/1024.0 for m in fg)
    return float(0.5/max(w0,w2,hz))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',required=True); ap.add_argument('--membership',required=True); ap.add_argument('--out',required=True); a=ap.parse_args()
    root=Path(a.root); out=Path(a.out); mem=json.load(open(a.membership,encoding='utf-8')); rows=mem.get('records',[])
    if mem.get('asset_count')!=8 or mem.get('asset_style_cells')!=16 or len(rows)!=16 or mem.get('post_result_selection') is not False: raise RuntimeError('exact frozen 8x2 membership required')
    assets=[]
    for r in rows:
        if r.get('split')!='FIT' or r.get('style') not in STYLES: raise RuntimeError('membership split/style drift')
        if r['asset_id'] not in assets: assets.append(r['asset_id'])
    if len(assets)!=8 or any({r['style'] for r in rows if r['asset_id']==aid}!=set(STYLES) for aid in assets): raise RuntimeError('8x2 coverage drift')
    (out/'cells').mkdir(parents=True,exist_ok=True); records=[]
    for aid in assets:
        src=root/'master'/'assets'/aid
        if not src.is_dir(): raise FileNotFoundError(src)
        gsha=sha256_file(src/'primary_geometry.npz'); asset_records=[]
        for style in STYLES:
            final=out/'cells'/f'{aid}__{style}'; tmp=Path(tempfile.mkdtemp(prefix=f'{aid}__{style}.partial.',dir=str(out/'cells')))
            try:
                with np.load(src/'primary_geometry.npz',allow_pickle=False) as z:
                    np.savez(tmp/'primary_geometry.npz',vertices=np.asarray(z['vertices'],np.float32),faces=np.asarray(z['faces'],np.int32))
                raster_hashes={}; native_hashes={}; deriv_hashes={}
                for v in range(VIEWS):
                    sv=src/'renders'/f'V{v}'; dv=tmp/'renders'/f'V{v}'; dv.mkdir(parents=True,exist_ok=True)
                    ra=sv/'raster_authority.npz'; shutil.copy2(ra,dv/'raster_authority.npz'); raster_hashes[f'V{v}']=sha256_file(ra)
                    native=sv/f'{style}.png'; deriv=sv/f'{style}_512.png'
                    if not native.is_file() or not deriv.is_file(): raise FileNotFoundError(f'missing {aid} {style} V{v}')
                    native_hashes[f'V{v}/{style}']=sha256_file(native); deriv_hashes[f'V{v}/{style}']=sha256_file(deriv)
                    with Image.open(deriv) as im:
                        if im.mode!='RGBA' or im.size!=(512,512): raise RuntimeError(f'bad derivative {deriv}')
                        im.resize((256,256),resample=Image.Resampling.BILINEAR).save(dv/f'{style}_input.png')
                h=native_h([src/'renders'/f'V{v}'/f'{style}.png' for v in range(VIEWS)])
                meta={'schema':'RealSaS.IRISSinglePoseV2.PV5R256EightByTwoStageCell.v1','asset_id':aid,'style':style,'split':'FIT','input_resolution':256,'sheet_half_extent':h,'camera_json_consumed':False,'teacher_camera_half_extent_consumed':False,'geometry_sha256':gsha,'raster_sha256':raster_hashes,'native_rgba_sha256':native_hashes,'derivative_512_sha256':deriv_hashes}
                atomic_json(tmp/'STAGE.json',meta)
                if final.exists(): shutil.rmtree(final)
                os.replace(tmp,final)
            finally:
                if tmp.exists(): shutil.rmtree(tmp,ignore_errors=True)
            rec={'asset_id':aid,'style':style,'split':'FIT','cell_dir':str(final),'sheet_half_extent':h,'geometry_sha256':gsha,'raster_sha256':raster_hashes}
            records.append(rec); asset_records.append(rec)
        if asset_records[0]['geometry_sha256']!=asset_records[1]['geometry_sha256'] or asset_records[0]['raster_sha256']!=asset_records[1]['raster_sha256']: raise RuntimeError(f'style truth authority drift {aid}')
    atomic_json(out/'STAGE_MANIFEST.json',{'schema':'RealSaS.IRISSinglePoseV2.PV5R256EightByTwoStageManifest.v1','record_count':16,'asset_count':8,'asset_style_cells':16,'styles':list(STYLES),'input_resolution':256,'camera_json_consumed':False,'records':records})
    print(f'[stage-r256-8x2] assets={len(assets)} cells={len(records)}',flush=True)
if __name__=='__main__': main()
