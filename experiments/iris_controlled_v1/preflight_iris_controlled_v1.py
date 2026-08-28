from __future__ import annotations
import argparse,json,sys,hashlib,shutil,tempfile
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from prepare_iris_controlled_v1 import build_asset
from iris_model import IRISControlledV1,count_parameters
from iris_losses import total_loss


def sha(path):
    h=hashlib.sha256(Path(path).read_bytes()).hexdigest(); return h

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',required=True); ap.add_argument('--asset',required=True); ap.add_argument('--out',required=True); ap.add_argument('--split',default='FIT'); a=ap.parse_args()
    out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); cache=out.parent/(a.asset+'_preflight_cache.npz')
    r=build_asset(Path(a.root),{'asset_id':a.asset,'split':a.split},cache,geom_samples=128,anchors_per_view=24,max_tracks=128)
    z=np.load(cache,allow_pickle=False); images=torch.from_numpy(z['images'][0].astype(np.float32)/255).permute(0,3,1,2)[None]
    images=F.interpolate(images.reshape(8,4,512,512),size=(128,128),mode='bilinear',align_corners=False).reshape(1,8,4,128,128)
    K=64; batch={'images':images,'yaw_deg':torch.from_numpy(z['yaw_deg'].astype(np.float32))[None],
      'geom_xy':torch.from_numpy(z['geom_xy'][:,:K].astype(np.float32))[None],'geom_p':torch.from_numpy(z['geom_p'][:,:K].astype(np.float32))[None],
      'geom_n':torch.from_numpy(z['geom_n'][:,:K].astype(np.float32))[None],'geom_mask':torch.from_numpy(z['geom_mask'][:,:K].astype(bool))[None]}
    vis=z['track_visible'].astype(bool); xy=z['track_xy'].astype(np.float32); p=z['track_p'].astype(np.float32); M=min(64,len(p))
    sv=[];tv=[];sx=[];tx=[];pp=[]
    for t in range(M):
        vv=np.flatnonzero(vis[t]); sv.append(vv[0]); tv.append(vv[1]); sx.append(xy[t,vv[0]]); tx.append(xy[t,vv[1]]); pp.append(p[t])
    batch.update(corr_src_view=torch.tensor([sv]),corr_tgt_view=torch.tensor([tv]),corr_src_xy=torch.tensor(np.asarray([sx])),corr_tgt_xy=torch.tensor(np.asarray([tx])),corr_p=torch.tensor(np.asarray([pp])),corr_mask=torch.ones(1,M,dtype=torch.bool))
    model=IRISControlledV1(); pred=model(images,batch['yaw_deg']); parts=total_loss(pred,batch,epoch=5); parts['total'].backward(); grad=sum(float(x.grad.abs().sum()) for x in model.parameters() if x.grad is not None)
    result={'schema':'RealSaS.IRISControlledV1.RealAssetPreflight.v1','asset':a.asset,'cache_sha256':r['cache_sha256'],'track_count':r['track_count'],
      'track_support_hist':r['track_support_hist'],'parameters':count_parameters(model),'output_shapes':{k:list(v.shape) for k,v in pred.items()},
      'losses':{k:float(v.detach()) for k,v in parts.items()},'grad_abs_sum':grad,'pass':bool(np.isfinite(float(parts['total'].detach())) and grad>0)}
    out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n'); print(json.dumps(result,indent=2))
    if not result['pass']: raise RuntimeError('preflight failed')
if __name__=='__main__': main()
