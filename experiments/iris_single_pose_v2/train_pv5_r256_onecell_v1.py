from __future__ import annotations
import argparse, json, os, random, shutil
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset_pv5_r256_onecell import PV5R256OneCellDataset
from evaluate_pv5_r256_onecell_v1 import evaluate
from model_pv5_r256 import IRISSinglePoseV2PV5R256
from pv5_depth_objective import p_only_objective

SEED=20260824
STEPS=2048
CANDIDATES=(64,128,256,512,1024,2048)
LR=3e-4
WD=0.0
ALLOWED_TRAINABLE_PREFIXES=('encoder.','within.','cross.','context_fuse.','d8.','d4.','d2.','p_s1_stem.','p_s1_fuse.','p_depth_head.')

def atomic_json(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n',encoding='utf-8'); os.replace(tmp,path)

def seed_all():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED); torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True

def configure_p_only(model):
    for name,p in model.named_parameters(): p.requires_grad=any(name.startswith(x) for x in ALLOWED_TRAINABLE_PREFIXES)
    trainable=[n for n,p in model.named_parameters() if p.requires_grad]
    forbidden=[n for n in trainable if not any(n.startswith(x) for x in ALLOWED_TRAINABLE_PREFIXES)]
    if forbidden: raise RuntimeError(f'forbidden trainable params: {forbidden}')
    for head in ('n_head','u_geo_head','coarse_head','fine_head'):
        if any(n.startswith(head+'.') and p.requires_grad for n,p in model.named_parameters()): raise RuntimeError(f'{head} must be frozen')
    return trainable

def pack(batch,device):
    return {'images':batch['images'].to(device,non_blocking=True),'yaw_deg':batch['yaw_deg'].to(device,non_blocking=True),'sheet_half_extent':batch['sheet_half_extent'].to(device,non_blocking=True),'geom_xy':batch['geom_xy'].to(device,non_blocking=True),'geom_p':batch['geom_p'].to(device,non_blocking=True),'geom_mask':batch['geom_mask'].to(device,non_blocking=True)}

def main():
    ap=argparse.ArgumentParser(description='P-V5 R256 one-cell overfit; CUDA required; no TUNE/sealed arguments'); ap.add_argument('--cache-manifest',required=True); ap.add_argument('--out-dir',required=True); a=ap.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError('CUDA GPU REQUIRED for R256 one-cell training')
    seed_all(); device=torch.device('cuda'); out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True); (out/'eval').mkdir(exist_ok=True); (out/'checkpoints').mkdir(exist_ok=True)
    ds=PV5R256OneCellDataset(a.cache_manifest)
    if len(ds)!=1: raise RuntimeError('expected exactly one asset-style cell')
    dl=DataLoader(ds,batch_size=1,shuffle=False,num_workers=0,pin_memory=True); cpu_batch=next(iter(dl)); batch=pack(cpu_batch,device)
    model=IRISSinglePoseV2PV5R256().to(device); trainable=configure_p_only(model)
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=LR,weight_decay=WD,betas=(.9,.95)); scaler=torch.cuda.amp.GradScaler(enabled=True)
    authority={'schema':'RealSaS.IRISSinglePoseV2.PV5R256OneCellTrainingAuthority.v1','seed':SEED,'input_resolution':256,'output_field_hw':256,'planned_optimizer_steps':STEPS,'candidate_steps':list(CANDIDATES),'optimizer':{'name':'AdamW','lr':LR,'betas':[.9,.95],'weight_decay':WD},'objective':'FP32 SmoothL1 on camera-forward depth only, beta=0.01','evaluation_authority':'full reconstructed canonical P Euclidean error','target_P_p95':.005,'membership':'1 frozen FIT asset x cel_clean','asset_id':cpu_batch['asset_id'][0],'style':cpu_batch['style'][0],'augmentation':False,'tune_consumed':False,'sealed_splits_opened':False,'camera_json_consumed':False,'trainable_parameters':trainable}
    atomic_json(out/'TRAIN_AUTHORITY.json',authority); atomic_json(out/'eval'/'INIT.json',evaluate(model,a.cache_manifest,device))
    history=[]; candidates=[]
    for step in range(1,STEPS+1):
        model.train(); opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type='cuda',dtype=torch.float16): outputs=model(batch['images'],batch['yaw_deg'],batch['sheet_half_extent'])
        if tuple(outputs['P'].shape)!=(1,8,3,256,256): raise RuntimeError(f'R256 training field drift: {tuple(outputs["P"].shape)}')
        parts=p_only_objective(outputs,{'geom_xy':batch['geom_xy'],'geom_p':batch['geom_p'],'geom_mask':batch['geom_mask'],'yaw_deg':batch['yaw_deg']})
        scaler.scale(parts['total']).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0); scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
        if step==1 or step%16==0 or step in CANDIDATES:
            rec={'step':step,'depth_smooth_l1':float(parts['depth_smooth_l1'].detach().cpu()),'train_P_p95':float(parts['p_p95'].detach().cpu()),'train_P_mean':float(parts['p_mean'].detach().cpu())}; history.append(rec); atomic_json(out/'TRAIN_HISTORY.json',{'records':history})
            if step%64==0 or step in CANDIDATES: print(f'[r256-onecell] step={step}/{STEPS} depth={rec["depth_smooth_l1"]:.6g} p95={rec["train_P_p95"]:.6g}',flush=True)
        if step in CANDIDATES:
            ev=evaluate(model,a.cache_manifest,device); ev.update(optimizer_steps=step); atomic_json(out/'eval'/f'STEP_{step:04d}.json',ev); cp=out/'checkpoints'/f'step_{step:04d}.pt'; torch.save({'schema':'RealSaS.IRISSinglePoseV2.PV5R256OneCellCheckpoint.v1','optimizer_steps':step,'model_state':{k:v.detach().cpu() for k,v in model.state_dict().items()}},cp); candidates.append({'optimizer_steps':step,'P_p95':ev['P_p95'],'pass_0p005':ev['pass_0p005'],'checkpoint':str(cp)}); print(json.dumps(candidates[-1],indent=2),flush=True)
    best=min(candidates,key=lambda x:(x['P_p95'],x['optimizer_steps'])); shutil.copy2(best['checkpoint'],out/'BEST_CHECKPOINT.pt'); best_eval=json.load(open(out/'eval'/f'STEP_{best["optimizer_steps"]:04d}.json',encoding='utf-8'))
    status='P_V5_R256_ONE_CELL_OVERFIT_PASS' if best_eval['pass_0p005'] else 'P_V5_R256_ONE_CELL_OPTIMIZATION_INSUFFICIENT'
    decision={'schema':'RealSaS.IRISSinglePoseV2.PV5R256OneCellDecision.v1','status':status,'selected_step':best['optimizer_steps'],'optimizer_steps':STEPS,'selected_P_p95':best_eval['P_p95'],'threshold_P_p95':.005,'pass_0p005':best_eval['pass_0p005'],'output_field_hw':256,'asset_id':authority['asset_id'],'style':authority['style'],'tune_consumed':False,'sealed_splits_opened':False,'camera_json_consumed':False,'next_policy':{'P_V5_R256_ONE_CELL_OVERFIT_PASS':'preregister 1 asset x 2 styles R256 overfit only','P_V5_R256_ONE_CELL_OPTIMIZATION_INSUFFICIENT':'localize R256 learner/optimizer; do not reopen closed P ontology/V5 geometry/free-R256 representation without contradictory evidence'}[status]}
    atomic_json(out/'P_V5_R256_ONE_CELL_DECISION.json',decision); print(json.dumps(decision,indent=2,sort_keys=True),flush=True)

if __name__=='__main__': main()
