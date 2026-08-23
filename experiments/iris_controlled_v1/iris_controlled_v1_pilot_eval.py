from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from iris_dataset import IRISCacheDataset
from iris_model import IRISControlledV1
from iris_losses import sample_pair_field,total_loss

def retrieval_topk(zs,zt,p,mask,ks=(1,4,8)):
    vals={k:[] for k in ks}; mrr=[]
    B,M,D=zs.shape
    for b in range(B):
        mb=mask[b].bool()
        if mb.sum()<2: continue
        a=F.normalize(zs[b,mb],dim=-1); c=F.normalize(zt[b,mb],dim=-1); pp=p[b,mb].float(); sim=a@c.T; dist=torch.cdist(pp,pp); pos=dist<=0.003; order=torch.argsort(sim,dim=1,descending=True)
        for i in range(order.shape[0]):
            hits=pos[i,order[i]]; idx=torch.nonzero(hits,as_tuple=False)
            if idx.numel()==0: continue
            rank=int(idx[0,0])+1
            for k in ks: vals[k].append(rank<=k)
            mrr.append(1.0/rank)
    out={f'top{k}':float(np.mean(vals[k])) if vals[k] else 0.0 for k in ks}; out['mrr']=float(np.mean(mrr)) if mrr else 0.0; out['queries']=len(mrr); return out

def evaluate(model,loader,device):
    model.eval(); sums={}; nb=0; coarse=[]; fine=[]
    with torch.no_grad():
        for batch in loader:
            for k,v in list(batch.items()):
                if torch.is_tensor(v): batch[k]=v.to(device)
            out=model(batch['images'],batch['yaw_deg']); parts=total_loss(out,batch,epoch=5,warmup_epochs=4)
            for k,v in parts.items(): sums[k]=sums.get(k,0.0)+float(v.detach())
            nb+=1
            sv=batch['corr_src_view'].long(); tv=batch['corr_tgt_view'].long(); sx=batch['corr_src_xy']; tx=batch['corr_tgt_xy']; p=batch['corr_p']; mask=batch['corr_mask'].bool()
            zcs=sample_pair_field(out['Z_coarse'],sv,sx); zct=sample_pair_field(out['Z_coarse'],tv,tx); zfs=sample_pair_field(out['Z_fine'],sv,sx); zft=sample_pair_field(out['Z_fine'],tv,tx)
            coarse.append(retrieval_topk(zcs,zct,p,mask)); fine.append(retrieval_topk(zfs,zft,p,mask))
    avg={k:v/max(nb,1) for k,v in sums.items()}
    def agg(rows):
        q=sum(r['queries'] for r in rows); return {k:(sum(r[k]*r['queries'] for r in rows)/max(q,1)) for k in ('top1','top4','top8','mrr')}|{'queries':q}
    return {'losses':avg,'coarse_retrieval':agg(coarse),'fine_retrieval':agg(fine)}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--cache-manifest',required=True); ap.add_argument('--checkpoint',required=True); ap.add_argument('--out',required=True); ap.add_argument('--device',default='cuda'); a=ap.parse_args(); device=torch.device(a.device if torch.cuda.is_available() else 'cpu')
    ck=torch.load(a.checkpoint,map_location=device); model=IRISControlledV1().to(device); model.load_state_dict(ck['model']); fit=IRISCacheDataset(a.cache_manifest,'FIT',256,False); tune=IRISCacheDataset(a.cache_manifest,'TUNE',256,False)
    res={'schema':'RealSaS.IRISControlledV1.OpenPilotEval.v1','checkpoint':str(a.checkpoint),'FIT':evaluate(model,DataLoader(fit,batch_size=1,num_workers=0),device),'TUNE':evaluate(model,DataLoader(tune,batch_size=1,num_workers=0),device),'sealed_opened':False}
    Path(a.out).write_text(json.dumps(res,indent=2,sort_keys=True)+'\n'); print(json.dumps(res,indent=2))
if __name__=='__main__': main()
