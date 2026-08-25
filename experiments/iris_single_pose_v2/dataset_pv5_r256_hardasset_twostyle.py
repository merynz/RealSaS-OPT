from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

AID='asset_36fb02305846592b1ecdf3d4'
STYLES=('cel_clean','ink_cel')

class PV5R256HardAssetTwoStyleDataset(Dataset):
    def __init__(self,cache_manifest:str):
        m=json.load(open(cache_manifest,encoding='utf-8'))
        rows=m.get('records',[])
        if m.get('record_count')!=2 or m.get('asset_count')!=1 or m.get('asset_style_cells')!=2:
            raise RuntimeError('invalid R256 two-style cache')
        if m.get('camera_json_consumed') is not False or m.get('shared_truth_loci_across_styles') is not True:
            raise RuntimeError('two-style cache firewall/truth drift')
        if {r.get('asset_id') for r in rows}!={AID} or {r.get('style') for r in rows}!=set(STYLES):
            raise RuntimeError('two-style cache membership drift')
        if any(r.get('split')!='FIT' for r in rows):
            raise RuntimeError('non-FIT record')
        self.rows=[next(r for r in rows if r['style']==s) for s in STYLES]
        if len({r['truth_sha256'] for r in self.rows})!=1:
            raise RuntimeError('style cells do not share exact truth authority')
    def __len__(self): return 2
    def __getitem__(self,index):
        if index not in (0,1): raise IndexError(index)
        r=self.rows[index]; ar=Path(r['cell_dir']); style=r['style']; ims=[]
        for v in range(8):
            with Image.open(ar/'renders'/f'V{v}'/f'{style}_input.png') as im:
                if im.mode!='RGBA' or im.size!=(256,256):
                    raise RuntimeError(f'staged learner image drift {style} V{v}')
                ims.append(np.asarray(im,np.uint8))
        images=torch.from_numpy(np.stack(ims).astype(np.float32)/255.0).permute(0,3,1,2).contiguous()
        with np.load(r['truth_path'],allow_pickle=False) as z:
            xy=torch.from_numpy(z['geom_xy'].astype(np.float32))
            p=torch.from_numpy(z['geom_p'].astype(np.float32))
            mask=torch.from_numpy(z['geom_mask'].astype(bool))
            yaw=torch.from_numpy(z['yaw_deg'].astype(np.float32))
        return {'asset_id':r['asset_id'],'style':style,'images':images,'yaw_deg':yaw,
                'sheet_half_extent':torch.tensor(float(r['sheet_half_extent']),dtype=torch.float32),
                'geom_xy':xy,'geom_p':p,'geom_mask':mask}
