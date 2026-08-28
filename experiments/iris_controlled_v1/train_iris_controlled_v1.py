from __future__ import annotations
import argparse,json,os,random,time,hashlib
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from iris_dataset import IRISCacheDataset
from iris_model import IRISControlledV1,count_parameters
from iris_losses import total_loss


def atomic_json(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n'); os.replace(tmp,path)

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return h.hexdigest()

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def move(batch,device,input_resolution):
    out={}
    for k,v in batch.items(): out[k]=v.to(device,non_blocking=True) if torch.is_tensor(v) else v
    if input_resolution and out['images'].shape[-1]!=input_resolution:
        B,V,C,H,W=out['images'].shape
        x=F.interpolate(out['images'].reshape(B*V,C,H,W),size=(input_resolution,input_resolution),mode='bilinear',align_corners=False,antialias=True)
        out['images']=x.reshape(B,V,C,input_resolution,input_resolution)
    return out

def eval_split(model,loader,device,input_resolution,epoch):
    model.eval(); sums={}; n=0
    with torch.no_grad():
        for batch in loader:
            batch=move(batch,device,input_resolution); out=model(batch['images'],batch['yaw_deg']); parts=total_loss(out,batch,epoch,warmup_epochs=4)
            for k,v in parts.items(): sums[k]=sums.get(k,0.0)+float(v.detach())
            n+=1
    return {k:v/max(n,1) for k,v in sums.items()}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--cache-manifest',required=True)
    ap.add_argument('--out',required=True)
    ap.add_argument('--epochs',type=int,default=24); ap.add_argument('--batch-size',type=int,default=1)
    ap.add_argument('--lr',type=float,default=5e-5); ap.add_argument('--weight-decay',type=float,default=1e-4)
    ap.add_argument('--grad-clip',type=float,default=2.0); ap.add_argument('--input-resolution',type=int,default=256)
    ap.add_argument('--corr-samples',type=int,default=256); ap.add_argument('--workers',type=int,default=2); ap.add_argument('--seed',type=int,default=20260823)
    ap.add_argument('--device',default='cuda'); ap.add_argument('--resume',action='store_true')
    a=ap.parse_args(); seed_all(a.seed)
    if a.device.startswith('cuda') and not torch.cuda.is_available(): raise RuntimeError('CUDA training requested but unavailable')
    device=torch.device(a.device); outdir=Path(a.out); outdir.mkdir(parents=True,exist_ok=True)
    fit=IRISCacheDataset(a.cache_manifest,'FIT',a.corr_samples,True); tune=IRISCacheDataset(a.cache_manifest,'TUNE',a.corr_samples,False)
    model=IRISControlledV1().to(device); opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=a.weight_decay)
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda')
    start_epoch=0; best=float('inf'); hist=[]; last=outdir/'last.pt'
    if a.resume and last.exists():
        ck=torch.load(last,map_location=device); model.load_state_dict(ck['model']); opt.load_state_dict(ck['optimizer']); start_epoch=ck['epoch']+1; best=ck['best']; hist=ck.get('history',[])
    fit_loader=DataLoader(fit,batch_size=a.batch_size,shuffle=True,num_workers=a.workers,pin_memory=device.type=='cuda',persistent_workers=a.workers>0)
    tune_loader=DataLoader(tune,batch_size=1,shuffle=False,num_workers=a.workers,pin_memory=device.type=='cuda',persistent_workers=a.workers>0)
    meta={'schema':'RealSaS.IRISControlledV1.TrainingRun.v1','cache_manifest':str(Path(a.cache_manifest).resolve()),'cache_manifest_sha256':sha256(a.cache_manifest),
          'parameters':count_parameters(model),'args':vars(a),'sealed_splits_opened':False}
    atomic_json(outdir/'RUN_CONTRACT.json',meta)
    for epoch in range(start_epoch,a.epochs):
        fit.set_epoch(epoch); model.train(); sums={}; n=0; t=time.time()
        for batch in fit_loader:
            batch=move(batch,device,a.input_resolution); opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type,enabled=device.type=='cuda'):
                pred=model(batch['images'],batch['yaw_deg']); parts=total_loss(pred,batch,epoch,warmup_epochs=4); loss=parts['total']
            scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(),a.grad_clip); scaler.step(opt); scaler.update()
            for k,v in parts.items(): sums[k]=sums.get(k,0.0)+float(v.detach()); n+=1 if k=='total' else 0
        train={k:v/max(n,1) for k,v in sums.items()}; val=eval_split(model,tune_loader,device,a.input_resolution,epoch)
        row={'epoch':epoch,'train':train,'tune':val,'elapsed_sec':time.time()-t}; hist.append(row); score=val['total']
        ck={'epoch':epoch,'model':model.state_dict(),'optimizer':opt.state_dict(),'best':min(best,score),'history':hist,'contract':meta}
        torch.save(ck,last)
        if score<best: best=score; torch.save(ck,outdir/'best.pt')
        atomic_json(outdir/'history.json',{'schema':'RealSaS.IRISControlledV1.History.v1','best_tune_total':best,'epochs':hist})
        print(json.dumps(row))
    print(json.dumps({'status':'TRAINING_COMPLETE_OPEN_SPLITS_ONLY','best_tune_total':best,'sealed_splits_opened':False},indent=2))
if __name__=='__main__': main()
