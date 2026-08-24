from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

STYLES=('cel_clean','ink_cel')

class PV5DepthOverfitDataset(Dataset):
    def __init__(self,cache_manifest:str):
        m=json.load(open(cache_manifest,encoding='utf-8'))
        if m.get('record_count')!=8 or m.get('camera_json_consumed') is not False: raise RuntimeError('invalid P-V5 overfit cache')
        self.rows=[]
        for r in m['records']:
            if r.get('split')!='FIT': raise RuntimeError('non-FIT row forbidden')
            for style in STYLES:self.rows.append((r,style))
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):
        r,style=self.rows[index]; ar=Path(r['asset_dir']); ims=[]
        for v in range(8):
            with Image.open(ar/'renders'/f'V{v}'/f'{style}_input.png') as im:
                if im.mode!='RGBA' or im.size!=(256,256): raise RuntimeError('staged learner image drift')
                ims.append(np.asarray(im,np.uint8))
        images=torch.from_numpy(np.stack(ims).astype(np.float32)/255.0).permute(0,3,1,2).contiguous()
        with np.load(r['truth_path'],allow_pickle=False) as z:
            xy=torch.from_numpy(z['geom_xy'].astype(np.float32)); p=torch.from_numpy(z['geom_p'].astype(np.float32)); mask=torch.from_numpy(z['geom_mask'].astype(bool)); yaw=torch.from_numpy(z['yaw_deg'].astype(np.float32))
        h=float(r['sheet_half_extent_by_style'][style])
        return {'asset_id':r['asset_id'],'style':style,'images':images,'yaw_deg':yaw,'sheet_half_extent':torch.tensor(h,dtype=torch.float32),'geom_xy':xy,'geom_p':p,'geom_mask':mask}
