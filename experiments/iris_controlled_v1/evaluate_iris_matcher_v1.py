from __future__ import annotations
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from iris_dataset import IRISCacheDataset
from iris_model import IRISControlledV1
from iris_losses import total_loss
from iris_matcher_v1 import build_view_nodes, build_all_pairs, evaluate_asset_matcher, aggregate_asset_metrics


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return h.hexdigest()


def move(batch,device):
    return {k:(v.to(device) if torch.is_tensor(v) else v) for k,v in batch.items()}


def eval_split(model, manifest, split, device):
    ds=IRISCacheDataset(manifest,split,256,False)
    loader=DataLoader(ds,batch_size=1,shuffle=False,num_workers=0)
    asset_rows=[]; loss_sums={}; n=0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            aid=batch['asset_id'][0]
            batch=move(batch,device)
            outputs=model(batch['images'],batch['yaw_deg'])
            parts=total_loss(outputs,batch,epoch=5,warmup_epochs=4)
            for k,v in parts.items(): loss_sums[k]=loss_sums.get(k,0.0)+float(v.detach())
            n+=1
            row=ds.rows[n-1]
            with np.load(row['cache_path'],allow_pickle=False) as z:
                nodes=build_view_nodes(outputs,z)
                pairs=build_all_pairs(nodes)
                m=evaluate_asset_matcher(nodes,pairs,z)
            m['asset_id']=aid; asset_rows.append(m)
            if n%10==0 or n==len(ds):
                print(f'[matcher-eval] {split} {n}/{len(ds)} composite_top4={aggregate_asset_metrics(asset_rows)["composite"]["top4"]:.4f}',flush=True)
    losses={k:v/max(n,1) for k,v in loss_sums.items()}
    return {'losses':losses,'matcher':aggregate_asset_metrics(asset_rows),'per_asset':asset_rows}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--cache-manifest',required=True); ap.add_argument('--checkpoint',required=True); ap.add_argument('--out',required=True)
    ap.add_argument('--splits',default='TUNE'); ap.add_argument('--device',default='cuda')
    a=ap.parse_args(); device=torch.device(a.device if a.device.startswith('cuda') and torch.cuda.is_available() else 'cpu')
    ck=torch.load(a.checkpoint,map_location=device); model=IRISControlledV1().to(device); model.load_state_dict(ck['model'])
    res={'schema':'RealSaS.IRISControlledV1.FullMatcherEval.v1','checkpoint':str(Path(a.checkpoint).resolve()),'checkpoint_sha256':sha256(a.checkpoint),
         'cache_manifest':str(Path(a.cache_manifest).resolve()),'cache_manifest_sha256':sha256(a.cache_manifest),'matcher':'IRIS_DETERMINISTIC_MATCHER_V1_NO_D3',
         'splits':{},'sealed_opened':False}
    for split in [x.strip().upper() for x in a.splits.split(',') if x.strip()]:
        if split not in ('FIT','TUNE'): raise RuntimeError(f'FAIL CLOSED: matcher eval open splits only, got {split}')
        res['splits'][split]=eval_split(model,a.cache_manifest,split,device)
    Path(a.out).write_text(json.dumps(res,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:v['matcher'] for k,v in res['splits'].items()},indent=2),flush=True)
if __name__=='__main__': main()
