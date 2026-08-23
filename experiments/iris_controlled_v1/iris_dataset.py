from __future__ import annotations
import json, hashlib
import numpy as np
import torch
from torch.utils.data import Dataset

def _seed(asset,epoch,index): return int(hashlib.sha256(f'{asset}|{epoch}|{index}'.encode()).hexdigest()[:16],16)&0x7fffffff

class IRISCacheDataset(Dataset):
    def __init__(self,cache_manifest,split='FIT',corr_samples=256,training=True):
        self.manifest=json.load(open(cache_manifest)); self.split=split.upper(); self.training=training; self.corr_samples=corr_samples; self.epoch=0
        self.rows=[r for r in self.manifest['records'] if r['split']==self.split]
        if not self.rows: raise RuntimeError(f'no cache rows for split {self.split}')
    def set_epoch(self,e): self.epoch=int(e)
    def __len__(self): return len(self.rows)
    def __getitem__(self,index):
        r=self.rows[index]; z=np.load(r['cache_path'],allow_pickle=False); aid=r['asset_id']; rng=np.random.default_rng(_seed(aid,self.epoch,index)); style=int(rng.integers(0,2)) if self.training else 0
        images=torch.from_numpy(z['images'][style].astype(np.float32)/255.).permute(0,3,1,2).contiguous()
        out={'asset_id':aid,'images':images,'yaw_deg':torch.from_numpy(z['yaw_deg'].astype(np.float32)),'geom_xy':torch.from_numpy(z['geom_xy'].astype(np.float32)),'geom_p':torch.from_numpy(z['geom_p'].astype(np.float32)),'geom_n':torch.from_numpy(z['geom_n'].astype(np.float32)),'geom_mask':torch.from_numpy(z['geom_mask'].astype(bool))}
        vis=z['track_visible'].astype(bool); xy=z['track_xy'].astype(np.float32); p=z['track_p'].astype(np.float32); valid=np.flatnonzero(vis.sum(1)>=2); M=self.corr_samples
        sv=np.zeros(M,np.int64); tv=np.zeros(M,np.int64); sx=np.zeros((M,2),np.float32); tx=np.zeros((M,2),np.float32); pp=np.zeros((M,3),np.float32); mask=np.zeros(M,bool)
        if len(valid):
            picks=rng.choice(valid,size=M,replace=len(valid)<M)
            for j,t in enumerate(picks):
                vv=np.flatnonzero(vis[t]); pair=rng.choice(vv,size=2,replace=False); sv[j],tv[j]=pair; sx[j]=xy[t,sv[j]]; tx[j]=xy[t,tv[j]]; pp[j]=p[t]; mask[j]=True
        out.update(corr_src_view=torch.from_numpy(sv),corr_tgt_view=torch.from_numpy(tv),corr_src_xy=torch.from_numpy(sx),corr_tgt_xy=torch.from_numpy(tx),corr_p=torch.from_numpy(pp),corr_mask=torch.from_numpy(mask)); return out
