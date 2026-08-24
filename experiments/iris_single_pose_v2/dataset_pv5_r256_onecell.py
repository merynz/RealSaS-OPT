from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

class PV5R256OneCellDataset(Dataset):
    def __init__(self,cache_manifest:str):
        m=json.load(open(cache_manifest,encoding='utf-8'))
        if m.get('record_count')!=1 or m.get('asset_style_cells')!=1 or m.get('camera_json_consumed') is not False: raise RuntimeError('invalid R256 one-cell cache')
        r=m['records'][0]
        if r.get('split')!='FIT' or r.get('style')!='cel_clean': raise RuntimeError('one-cell membership drift')
        self.r=r
    def __len__(self): return 1
    def __getitem__(self,index):
        if index!=0: raise IndexError(index)
        r=self.r; ar=Path(r['asset_dir']); style=r['style']; ims=[]
        for v in range(8):
            with Image.open(ar/'renders'/f'V{v}'/f'{style}_input.png') as im:
                if im.mode!='RGBA' or im.size!=(256,256): raise RuntimeError('staged learner image drift')
                ims.append(np.asarray(im,np.uint8))
        images=torch.from_numpy(np.stack(ims).astype(np.float32)/255.0).permute(0,3,1,2).contiguous()
        with np.load(r['truth_path'],allow_pickle=False) as z:
            xy=torch.from_numpy(z['geom_xy'].astype(np.float32)); p=torch.from_numpy(z['geom_p'].astype(np.float32)); mask=torch.from_numpy(z['geom_mask'].astype(bool)); yaw=torch.from_numpy(z['yaw_deg'].astype(np.float32))
        return {'asset_id':r['asset_id'],'style':style,'images':images,'yaw_deg':yaw,'sheet_half_extent':torch.tensor(float(r['sheet_half_extent']),dtype=torch.float32),'geom_xy':xy,'geom_p':p,'geom_mask':mask}
