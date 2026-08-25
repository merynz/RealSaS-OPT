from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

def _seed(asset,epoch,index):return int(hashlib.sha256(f"{asset}|{epoch}|{index}".encode()).hexdigest()[:16],16)&0x7fffffff
class IRISV2Dataset(Dataset):
 def __init__(self,cache_manifest,split="FIT",track_samples=256,style_mode="random"):
  self.manifest=json.load(open(cache_manifest,encoding="utf-8"));self.rows=[r for r in self.manifest["records"] if r["split"]==split.upper()];self.track_samples=int(track_samples);self.style_mode=style_mode;self.epoch=0
  if not self.rows:raise RuntimeError(f"no rows for {split}")
  if style_mode not in {"random","cel_clean","ink_cel"}:raise ValueError(style_mode)
 def set_epoch(self,e):self.epoch=int(e)
 def __len__(self):return len(self.rows)
 def _style(self,aid,index):
  if self.style_mode!="random":return self.style_mode
  return ("cel_clean","ink_cel")[int(np.random.default_rng(_seed(aid,self.epoch,index)).integers(0,2))]
 def __getitem__(self,index):
  r=self.rows[index];aid=r["asset_id"];style=self._style(aid,index);asset=Path(r["asset_dir"]);ims=[]
  for v in range(8):
   with Image.open(asset/"renders"/f"V{v}"/f"{style}_input.png") as im:ims.append(np.asarray(im.convert("RGBA"),np.uint8))
  images=torch.from_numpy(np.stack(ims).astype(np.float32)/255.).permute(0,3,1,2).contiguous()
  with np.load(r["truth_path"],allow_pickle=False) as z:
   out={"asset_id":aid,"style":style,"images":images,"yaw_deg":torch.from_numpy(z["yaw_deg"].astype(np.float32)),"geom_xy":torch.from_numpy(z["geom_xy"].astype(np.float32)),"geom_p":torch.from_numpy(z["geom_p"].astype(np.float32)),"geom_n":torch.from_numpy(z["geom_n"].astype(np.float32)),"geom_mask":torch.from_numpy(z["geom_mask"].astype(bool))};p=z["track_p"].astype(np.float32);xy=z["track_xy"].astype(np.float32);vis=z["track_visible"].astype(bool);n=len(p);m=self.track_samples;rng=np.random.default_rng(_seed(aid,self.epoch,index)^0x5A17);ids=rng.choice(n,size=min(m,n),replace=False);pp=np.zeros((m,3),np.float32);xx=np.zeros((m,8,2),np.float32);vv=np.zeros((m,8),bool);take=len(ids);pp[:take]=p[ids];xx[:take]=xy[ids];vv[:take]=vis[ids]
  out.update(track_p=torch.from_numpy(pp),track_xy=torch.from_numpy(xx),track_visible=torch.from_numpy(vv));return out
