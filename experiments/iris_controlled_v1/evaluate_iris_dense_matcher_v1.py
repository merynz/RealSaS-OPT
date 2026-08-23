from __future__ import annotations
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from iris_dataset import IRISCacheDataset
from iris_model import IRISControlledV1
from iris_losses import total_loss
from iris_dense_matcher_v1 import evaluate_dense_asset,aggregate

def sha(path):
 h=hashlib.sha256();
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--cache-manifest',required=True);ap.add_argument('--checkpoint',required=True);ap.add_argument('--out',required=True);ap.add_argument('--device',default='cuda');a=ap.parse_args();dev=torch.device(a.device if torch.cuda.is_available() else 'cpu')
 ck=torch.load(a.checkpoint,map_location=dev);model=IRISControlledV1().to(dev);model.load_state_dict(ck['model']);ds=IRISCacheDataset(a.cache_manifest,'TUNE',256,False);loader=DataLoader(ds,batch_size=1,shuffle=False,num_workers=0)
 rows=[];losses={};n=0;model.eval()
 with torch.no_grad():
  for batch in loader:
   aid=batch['asset_id'][0]
   for k,v in list(batch.items()):
    if torch.is_tensor(v):batch[k]=v.to(dev)
   out=model(batch['images'],batch['yaw_deg']);parts=total_loss(out,batch,epoch=5,warmup_epochs=4)
   for k,v in parts.items():losses[k]=losses.get(k,0.0)+float(v.detach())
   with np.load(ds.rows[n]['cache_path'],allow_pickle=False) as z:r=evaluate_dense_asset(aid,out,batch['images'],z)
   r['asset_id']=aid;rows.append(r);n+=1
   if n%5==0 or n==len(ds):print(f'[dense-matcher] TUNE {n}/{len(ds)} top4={aggregate(rows)["composite"]["top4"]:.4f} setcov={aggregate(rows)["qualification"]["output_truth_coverage"]:.4f}',flush=True)
 res={'schema':'RealSaS.IRISControlledV1.DenseMatcherEval.v1','candidate_domain':'alpha-supported 128x128 dense pixels; GT tracks only define evaluation queries/truth','checkpoint':str(Path(a.checkpoint).resolve()),'checkpoint_sha256':sha(a.checkpoint),'cache_manifest_sha256':sha(a.cache_manifest),'TUNE':{'losses':{k:v/max(n,1) for k,v in losses.items()},'matcher':aggregate(rows),'per_asset':rows},'sealed_opened':False,'D3_used':False}
 Path(a.out).write_text(json.dumps(res,indent=2,sort_keys=True)+'\n');print(json.dumps(res['TUNE']['matcher'],indent=2),flush=True)
if __name__=='__main__':main()
