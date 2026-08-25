from __future__ import annotations
import argparse, hashlib, json, os, shutil, tempfile
from pathlib import Path
import numpy as np
from PIL import Image

VIEWS=8
YAWS=[float(i*45) for i in range(8)]
STYLES=('cel_clean','ink_cel')
AID='asset_36fb02305846592b1ecdf3d4'

def sha256_file(path,chunk=8<<20):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for x in iter(lambda:f.read(chunk),b''): h.update(x)
    return h.hexdigest()

def atomic_json(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    os.replace(tmp,path)

def bbox_span(mask,axis):
    occ=mask.any(axis=0 if axis=='x' else 1); idx=np.flatnonzero(occ)
    if not len(idx): raise RuntimeError('blank alpha support')
    return int(idx[-1]-idx[0]+1)

def native_h_from_paths(paths):
    if len(paths)!=8: raise ValueError('need 8 ordered native views')
    alpha=[]
    for p in paths:
        with Image.open(p) as im:
            if im.mode!='RGBA' or im.size!=(1024,1024):
                raise RuntimeError(f'bad native {p}: {im.mode} {im.size}')
            alpha.append(np.asarray(im.getchannel('A'),np.uint8))
    fg=[a>=128 for a in alpha]
    w0=bbox_span(fg[0],'x')/1024.0
    w2=bbox_span(fg[2],'x')/1024.0
    hz=max(bbox_span(m,'y')/1024.0 for m in fg)
    return float(0.5/max(w0,w2,hz))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',required=True)
    ap.add_argument('--membership',required=True)
    ap.add_argument('--out',required=True)
    a=ap.parse_args()
    root=Path(a.root); out=Path(a.out)
    mem=json.load(open(a.membership,encoding='utf-8'))
    rows=mem.get('records',[])
    if mem.get('asset_count')!=1 or mem.get('asset_style_cells')!=2 or len(rows)!=2:
        raise RuntimeError('exact 1-asset x 2-style membership required')
    if {r.get('asset_id') for r in rows}!={AID} or {r.get('style') for r in rows}!=set(STYLES):
        raise RuntimeError('frozen two-style membership drift')
    if any(r.get('split')!='FIT' for r in rows) or mem.get('post_result_selection') is not True:
        raise RuntimeError('hard-asset diagnostic membership split/selection drift')

    src=root/'master'/'assets'/AID
    if not src.is_dir(): raise FileNotFoundError(src)
    (out/'cells').mkdir(parents=True,exist_ok=True)
    records=[]
    geometry_sha=sha256_file(src/'primary_geometry.npz')

    for style in STYLES:
        final=out/'cells'/f'{AID}__{style}'
        tmp=Path(tempfile.mkdtemp(prefix=f'{AID}__{style}.partial.',dir=str(out/'cells')))
        try:
            with np.load(src/'primary_geometry.npz',allow_pickle=False) as z:
                if 'vertices' not in z.files or 'faces' not in z.files:
                    raise RuntimeError(f'geometry missing {AID}')
                np.savez(tmp/'primary_geometry.npz',
                         vertices=np.asarray(z['vertices'],np.float32),
                         faces=np.asarray(z['faces'],np.int32))
            raster_hashes={}; native_hashes={}; deriv_hashes={}; views=[]
            for v in range(VIEWS):
                sv=src/'renders'/f'V{v}'; dv=tmp/'renders'/f'V{v}'
                dv.mkdir(parents=True,exist_ok=True)
                ra=sv/'raster_authority.npz'
                shutil.copy2(ra,dv/'raster_authority.npz')
                raster_hashes[f'V{v}']=sha256_file(ra)
                native=sv/f'{style}.png'; deriv=sv/f'{style}_512.png'
                if not native.is_file() or not deriv.is_file():
                    raise FileNotFoundError(f'missing style render {style} V{v}')
                native_hashes[f'V{v}/{style}']=sha256_file(native)
                deriv_hashes[f'V{v}/{style}']=sha256_file(deriv)
                with Image.open(deriv) as im:
                    if im.mode!='RGBA' or im.size!=(512,512):
                        raise RuntimeError(f'bad derivative {deriv}: {im.mode} {im.size}')
                    im.resize((256,256),resample=Image.Resampling.BILINEAR).save(dv/f'{style}_input.png')
                views.append({'view':v,'yaw_deg':YAWS[v],'style':style,
                              'native_sha256':native_hashes[f'V{v}/{style}'],
                              'derivative_512_sha256':deriv_hashes[f'V{v}/{style}'],
                              'input_transform':'PIL_RGBA_BILINEAR_512_to_256'})
            h=native_h_from_paths([src/'renders'/f'V{v}'/f'{style}.png' for v in range(VIEWS)])
            meta={'schema':'RealSaS.IRISSinglePoseV2.PV5R256HardAssetTwoStyleStageCell.v1',
                  'asset_id':AID,'style':style,'split':'FIT','input_resolution':256,
                  'native_scale_source_resolution':1024,'canonical_yaw_deg':YAWS,
                  'sheet_half_extent':h,'camera_json_consumed':False,
                  'teacher_camera_half_extent_consumed':False,
                  'geometry_sha256':geometry_sha,'raster_sha256':raster_hashes,
                  'native_rgba_sha256':native_hashes,'derivative_512_sha256':deriv_hashes,
                  'views':views}
            atomic_json(tmp/'STAGE.json',meta)
            if final.exists(): shutil.rmtree(final)
            os.replace(tmp,final)
        finally:
            if tmp.exists(): shutil.rmtree(tmp,ignore_errors=True)
        records.append({'asset_id':AID,'style':style,'split':'FIT',
                        'cell_dir':str(final),'sheet_half_extent':h,
                        'geometry_sha256':geometry_sha,'raster_sha256':raster_hashes})

    if records[0]['geometry_sha256']!=records[1]['geometry_sha256']:
        raise RuntimeError('geometry drift across style cells')
    if records[0]['raster_sha256']!=records[1]['raster_sha256']:
        raise RuntimeError('raster authority drift across style cells')

    atomic_json(out/'STAGE_MANIFEST.json',
                {'schema':'RealSaS.IRISSinglePoseV2.PV5R256HardAssetTwoStyleStageManifest.v1',
                 'record_count':2,'asset_count':1,'asset_style_cells':2,
                 'styles':list(STYLES),'input_resolution':256,
                 'camera_json_consumed':False,'records':records})
    print(f'[stage-r256-twostyle] {AID} {STYLES}',flush=True)

if __name__=='__main__': main()
