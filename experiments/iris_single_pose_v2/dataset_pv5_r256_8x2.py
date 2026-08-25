from __future__ import annotations
import json
from pathlib import Path
import numpy as np, torch
from PIL import Image
from torch.utils.data import Dataset
STYLES=('cel_clean','ink_cel')
class PV5R256EightByTwoDataset(Dataset):
    def __init__(self,cache_manifest:str):
        m=json.load(open(cache_manifest,encoding='utf-8')); rows=m.get('records',[])
        if m.get('record_count')!=16 or m.get('asset_count')!=8 or m.get('asset_style_cells')!=16 or m.get('camera_json_consumed') is not False or m.get('shared_truth_loci_across_styles') is not True: raise RuntimeError('invalid R256 8x2 cache')
        assets=[]
        for r in rows:
            if r.get('split')!='FIT': raise RuntimeError('non-FIT record')
            if r['asset_id'] not in assets: assets.append(r['asset_id'])
        if len(assets)!=8: raise RuntimeError('asset count drift')
        self.rows=[]
        for aid in assets:
            ar=[r for r in rows if r['asset_id']==aid]
            if {r['style'] for r in ar}!=set(STYLES) or len({r['truth_sha256'] for r in ar})!=1: raise RuntimeError(f'asset style/truth drift {aid}')
            for s in STYLES: self.rows.append(next(r for r in ar if r['style']==s))
        self.assets=assets
    def __len__(self): return 16
    def __getitem__(self,index):
        r=self.rows[index]; ar=Path(r['cell_dir']); ims=[]
        for v in range(8):
            with Image.open(ar/'renders'/f'V{v}'/f'{r["style"]}_input.png') as im:
                if im.mode!='RGBA' or im.size!=(256,256): raise RuntimeError(f'image drift {r["asset_id"]} {r["style"]} V{v}')
                ims.append(np.asarray(im,np.uint8))
        images=torch.from_numpy(np.stack(ims).astype(np.float32)/255.0).permute(0,3,1,2).contiguous()
        with np.load(r['truth_path'],allow_pickle=False) as z:
            xy=torch.from_numpy(z['geom_xy'].astype(np.float32)); p=torch.from_numpy(z['geom_p'].astype(np.float32)); mask=torch.from_numpy(z['geom_mask'].astype(bool)); yaw=torch.from_numpy(z['yaw_deg'].astype(np.float32))
        return {'asset_id':r['asset_id'],'style':r['style'],'images':images,'yaw_deg':yaw,'sheet_half_extent':torch.tensor(float(r['sheet_half_extent']),dtype=torch.float32),'geom_xy':xy,'geom_p':p,'geom_mask':mask}
