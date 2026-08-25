from __future__ import annotations
import argparse,json,os,random,shutil
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset_pv5_depth_overfit import PV5DepthOverfitDataset
from evaluate_pv5_depth_overfit_v1 import evaluate
from model_pv5 import IRISSinglePoseV2PV5
from pv5_depth_objective import p_only_objective

SEED=20260824; EPOCHS=64; CANDIDATES=(16,32,48,64); LR=3e-4; WD=0.0
ALLOWED_TRAINABLE_PREFIXES=('encoder.','within.','cross.','context_fuse.','d8.','d4.','d2.','p_depth_head.')

def atomic_json(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n',encoding='utf-8'); os.replace(tmp,path)
def seed_all():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED); torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
def configure_p_only(model):
    for name,p in model.named_parameters(): p.requires_grad=any(name.startswith(x) for x in ALLOWED_TRAINABLE_PREFIXES)
    trainable=[n for n,p in model.named_parameters() if p.requires_grad]; forbidden=[n for n in trainable if not any(n.startswith(x) for x in ALLOWED_TRAINABLE_PREFIXES)]
    if forbidden: raise RuntimeError(f'forbidden trainable params: {forbidden}')
    for head in ('n_head','u_geo_head','coarse_head','fine_head'):
        if any(n.startswith(head+'.') and p.requires_grad for n,p in model.named_parameters()): raise RuntimeError(f'{head} must be frozen')
    return trainable
def main():
    ap=argparse.ArgumentParser(description='P-V5 FIT-only depth/P overfit; CUDA required; no TUNE path accepted'); ap.add_argument('--cache-manifest',required=True); ap.add_argument('--out-dir',required=True); a=ap.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError('CUDA GPU REQUIRED for P-V5 depth overfit training')
    seed_all(); device=torch.device('cuda'); out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True); (out/'eval').mkdir(exist_ok=True); (out/'checkpoints').mkdir(exist_ok=True); ds=PV5DepthOverfitDataset(a.cache_manifest)
    if len(ds)!=16: raise RuntimeError(f'expected 16 asset-style cells, got {len(ds)}')
    model=IRISSinglePoseV2PV5().to(device); trainable=configure_p_only(model); opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=LR,weight_decay=WD,betas=(.9,.95)); scaler=torch.cuda.amp.GradScaler(enabled=True)
    authority={'schema':'RealSaS.IRISSinglePoseV2.PV5DepthOverfitTrainingAuthority.v1','seed':SEED,'input_resolution':256,'epochs':EPOCHS,'cells_per_epoch':16,'planned_optimizer_steps':EPOCHS*16,'candidate_epochs':list(CANDIDATES),'optimizer':{'name':'AdamW','lr':LR,'betas':[.9,.95],'weight_decay':WD},'objective':'FP32 SmoothL1 on camera-forward depth only, beta=0.01','evaluation_authority':'full reconstructed canonical P Euclidean error','target_P_p95':.005,'membership':'8 frozen FIT_TRAIN sentinels x 2 styles','tune_consumed':False,'sealed_splits_opened':False,'camera_json_consumed':False,'trainable_parameters':trainable}; atomic_json(out/'TRAIN_AUTHORITY.json',authority); atomic_json(out/'eval'/'INIT.json',evaluate(model,a.cache_manifest,device))
    history=[]; candidates=[]; steps=0
    for epoch in range(1,EPOCHS+1):
        gen=torch.Generator(); gen.manual_seed(SEED+epoch); dl=DataLoader(ds,batch_size=1,shuffle=True,generator=gen,num_workers=0,pin_memory=True); model.train(); sums={}
        for batch in dl:
            images=batch['images'].to(device,non_blocking=True); yaw=batch['yaw_deg'].to(device,non_blocking=True); h=batch['sheet_half_extent'].to(device,non_blocking=True); xy=batch['geom_xy'].to(device,non_blocking=True); gp=batch['geom_p'].to(device,non_blocking=True); gm=batch['geom_mask'].to(device,non_blocking=True); packed={'geom_xy':xy,'geom_p':gp,'geom_mask':gm,'yaw_deg':yaw}; opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda',dtype=torch.float16): outputs=model(images,yaw,h)
            parts=p_only_objective(outputs,packed); scaler.scale(parts['total']).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0); scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); steps+=1
            for k,v in parts.items(): sums[k]=sums.get(k,0.0)+float(v.detach().cpu())
        rec={'epoch':epoch,'optimizer_steps':steps,'mean':{k:v/len(dl) for k,v in sums.items()}}; history.append(rec); atomic_json(out/'TRAIN_HISTORY.json',{'epochs':history}); print(f'[pv5-depth] epoch={epoch}/{EPOCHS} steps={steps} depth={rec["mean"]["depth_smooth_l1"]:.6g} p95={rec["mean"]["p_p95"]:.6g}',flush=True)
        if epoch in CANDIDATES:
            ev=evaluate(model,a.cache_manifest,device); ev.update(epoch=epoch,optimizer_steps=steps); atomic_json(out/'eval'/f'EPOCH_{epoch:03d}.json',ev); cp=out/'checkpoints'/f'epoch_{epoch:03d}.pt'; torch.save({'schema':'RealSaS.IRISSinglePoseV2.PV5DepthOverfitCheckpoint.v1','epoch':epoch,'optimizer_steps':steps,'model_state':{k:v.detach().cpu() for k,v in model.state_dict().items()}},cp); candidates.append({'epoch':epoch,'optimizer_steps':steps,'worst_cell_P_p95':ev['worst_cell_P_p95'],'all_cells_pass_0p005':ev['all_cells_pass_0p005'],'checkpoint':str(cp)}); print(json.dumps(candidates[-1],indent=2),flush=True)
    best=min(candidates,key=lambda x:(x['worst_cell_P_p95'],x['epoch'])); shutil.copy2(best['checkpoint'],out/'BEST_CHECKPOINT.pt'); best_eval=json.load(open(out/'eval'/f'EPOCH_{best["epoch"]:03d}.json',encoding='utf-8')); status='P_V5_DEPTH_OVERFIT_PASS' if best_eval['all_cells_pass_0p005'] else 'P_V5_DEPTH_OPTIMIZATION_INSUFFICIENT'
    decision={'schema':'RealSaS.IRISSinglePoseV2.PV5DepthOverfitDecision.v1','status':status,'selected_epoch':best['epoch'],'optimizer_steps':best['optimizer_steps'],'worst_cell_P_p95':best_eval['worst_cell_P_p95'],'global':best_eval['global'],'all_cells_pass_0p005':best_eval['all_cells_pass_0p005'],'threshold_P_p95':.005,'tune_consumed':False,'sealed_splits_opened':False,'camera_json_consumed':False,'next_policy':{'P_V5_DEPTH_OVERFIT_PASS':'authorize preregistration of broader P-V5 FIT/TUNE learner gate; no product claim','P_V5_DEPTH_OPTIMIZATION_INSUFFICIENT':'localize learner/optimizer/feature capacity; do not reopen representation or V5 formulation'}}; atomic_json(out/'P_V5_DEPTH_OVERFIT_DECISION.json',decision); print(json.dumps(decision,indent=2),flush=True)
if __name__=='__main__': main()
