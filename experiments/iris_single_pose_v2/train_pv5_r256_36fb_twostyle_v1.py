from __future__ import annotations
import argparse, json, os, random, shutil
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset_pv5_r256_hardasset_twostyle import PV5R256HardAssetTwoStyleDataset
from evaluate_pv5_r256_hardasset_twostyle_v1 import evaluate
from model_pv5_r256 import IRISSinglePoseV2PV5R256
from pv5_depth_objective import p_only_objective

SEED=20260825
MAIN_STEPS=2048
TAIL_STEPS=2048
MAIN_LR=3e-4
TAIL_LR=3e-5
BETAS=(.9,.95)
WD=0.0
MAIN_EVAL=(512,1024,2048)
TAIL_EVAL=(64,128,256,512,1024,1536,2048)
THRESH=.005

ALLOWED_TRAINABLE_PREFIXES=(
    'encoder.','within.','cross.','context_fuse.','d8.','d4.','d2.',
    'p_s1_stem.','p_s1_fuse.','p_depth_head.'
)

def configure_p_only(model):
    for name,p in model.named_parameters():
        p.requires_grad=any(name.startswith(x) for x in ALLOWED_TRAINABLE_PREFIXES)
    trainable=[n for n,p in model.named_parameters() if p.requires_grad]
    forbidden=[n for n in trainable if not any(n.startswith(x) for x in ALLOWED_TRAINABLE_PREFIXES)]
    if forbidden: raise RuntimeError(f'forbidden trainable params: {forbidden}')
    for head in ('n_head','u_geo_head','coarse_head','fine_head'):
        if any(n.startswith(head+'.') and p.requires_grad for n,p in model.named_parameters()):
            raise RuntimeError(f'{head} must be frozen')
    return trainable

def atomic_json(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    os.replace(tmp,path)

def seed_all():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True

def pack(batch,device):
    return {k:(v.to(device,non_blocking=True) if torch.is_tensor(v) else v) for k,v in batch.items()}

def candidate_key(c):
    return (float(c['worst_cell_P_p95']),float(c['aggregate_P_p95']),int(c['total_optimizer_steps']))

def save_checkpoint(path,model,label,total_steps,phase,phase_step):
    torch.save({'schema':'RealSaS.IRISSinglePoseV2.PV5R256HardAssetTwoStyleCheckpoint.v1',
                'label':label,'total_optimizer_steps':int(total_steps),
                'phase':phase,'phase_step':int(phase_step),
                'model_state':{k:v.detach().cpu() for k,v in model.state_dict().items()}},path)

def run_phase(model,batch,device,out,cache_manifest,phase,steps,lr,eval_steps,total_offset,history,candidates):
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                          lr=lr,weight_decay=WD,betas=BETAS)
    scaler=torch.amp.GradScaler('cuda',enabled=True)
    for step in range(1,steps+1):
        model.train(); opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type='cuda',dtype=torch.float16):
            outputs=model(batch['images'],batch['yaw_deg'],batch['sheet_half_extent'])
        if tuple(outputs['P'].shape)!=(2,8,3,256,256):
            raise RuntimeError(f'R256 two-style training field drift: {tuple(outputs["P"].shape)}')
        parts=p_only_objective(outputs,{'geom_xy':batch['geom_xy'],'geom_p':batch['geom_p'],
                                        'geom_mask':batch['geom_mask'],'yaw_deg':batch['yaw_deg']})
        scaler.scale(parts['total']).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0)
        scaler.step(opt); scaler.update()
        total=total_offset+step
        if step==1 or step%16==0 or step in eval_steps:
            rec={'phase':phase,'phase_step':step,'total_optimizer_steps':total,
                 'lr':lr,'depth_smooth_l1':float(parts['depth_smooth_l1'].detach().cpu()),
                 'train_P_p95':float(parts['p_p95'].detach().cpu()),
                 'train_P_mean':float(parts['p_mean'].detach().cpu())}
            history.append(rec); atomic_json(out/'TRAIN_HISTORY.json',{'records':history})
        if step in eval_steps:
            ev=evaluate(model,cache_manifest,device)
            label=f'{phase}_{step:04d}'
            ev.update(label=label,phase=phase,phase_step=step,total_optimizer_steps=total,lr=lr)
            atomic_json(out/'eval'/f'{label}.json',ev)
            cp=out/'checkpoints'/f'{label}.pt'
            save_checkpoint(cp,model,label,total,phase,step)
            cand={'label':label,'phase':phase,'phase_step':step,'total_optimizer_steps':total,
                  'lr':lr,'worst_cell_P_p95':ev['worst_cell_P_p95'],
                  'aggregate_P_p95':ev['aggregate']['P_p95'],
                  'all_cells_pass_0p005':ev['all_cells_pass_0p005'],
                  'per_cell':ev['per_cell'],'checkpoint':str(cp)}
            candidates.append(cand)
            print(json.dumps(cand,indent=2),flush=True)

def main():
    ap=argparse.ArgumentParser(description='P-V5 R256 hard asset 36fb x 2 styles sufficiency gate')
    ap.add_argument('--cache-manifest',required=True)
    ap.add_argument('--out-dir',required=True)
    a=ap.parse_args()
    cache_manifest=str(a.cache_manifest)
    if not torch.cuda.is_available(): raise RuntimeError('CUDA GPU REQUIRED for R256 two-style training')
    seed_all(); device=torch.device('cuda')
    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    (out/'eval').mkdir(exist_ok=True); (out/'checkpoints').mkdir(exist_ok=True)

    ds=PV5R256HardAssetTwoStyleDataset(cache_manifest)
    if len(ds)!=2: raise RuntimeError('expected exactly two style cells')
    dl=DataLoader(ds,batch_size=2,shuffle=False,num_workers=0,pin_memory=True)
    cpu_batch=next(iter(dl)); batch=pack(cpu_batch,device)
    if list(cpu_batch['style'])!=['cel_clean','ink_cel']:
        raise RuntimeError(f'style batch order drift: {cpu_batch["style"]}')

    model=IRISSinglePoseV2PV5R256().to(device)
    trainable=configure_p_only(model)
    authority={'schema':'RealSaS.IRISSinglePoseV2.PV5R256HardAssetTwoStyleTrainingAuthority.v1',
               'seed':SEED,'fresh_model_initialization':True,'input_resolution':256,'output_field_hw':256,
               'membership':'1 frozen FIT asset x 2 styles, both cells in every optimizer step',
               'asset_id':cpu_batch['asset_id'][0],'styles':list(cpu_batch['style']),
               'batch_size_cells':2,'views_per_cell':8,'augmentation':False,
               'objective':'FP32 SmoothL1 on camera-forward depth only, beta=0.01; averaged over both style cells',
               'evaluation_authority':'cell-wise full reconstructed canonical P Euclidean p95; aggregate alone cannot pass',
               'target_P_p95_per_cell':THRESH,
               'schedule':[
                   {'phase':'MAIN','optimizer':'AdamW','fresh_moments':True,'lr':MAIN_LR,'betas':list(BETAS),'weight_decay':WD,'steps':MAIN_STEPS,'eval_steps':list(MAIN_EVAL)},
                   {'phase':'TAIL','optimizer':'AdamW','fresh_moments':True,'lr':TAIL_LR,'betas':list(BETAS),'weight_decay':WD,'steps':TAIL_STEPS,'eval_steps':list(TAIL_EVAL)}
               ],
               'tune_consumed':False,'sealed_splits_opened':False,'camera_json_consumed':False,
               'trainable_parameters':trainable}
    atomic_json(out/'TRAIN_AUTHORITY.json',authority)
    init=evaluate(model,cache_manifest,device)
    init.update(label='INIT',phase='INIT',phase_step=0,total_optimizer_steps=0,lr=None)
    atomic_json(out/'eval'/'INIT.json',init)

    history=[]; candidates=[]
    run_phase(model,batch,device,out,cache_manifest,'MAIN',MAIN_STEPS,MAIN_LR,MAIN_EVAL,0,history,candidates)
    run_phase(model,batch,device,out,cache_manifest,'TAIL',TAIL_STEPS,TAIL_LR,TAIL_EVAL,MAIN_STEPS,history,candidates)

    best=min(candidates,key=candidate_key)
    shutil.copy2(best['checkpoint'],out/'BEST_CHECKPOINT.pt')
    status='P_V5_R256_36FB_TWO_STYLE_SUFFICIENCY_PASS' if best['all_cells_pass_0p005'] else 'P_V5_R256_36FB_TWO_STYLE_OPTIMIZATION_INSUFFICIENT'
    decision={'schema':'RealSaS.IRISSinglePoseV2.PV5R256HardAssetTwoStyleDecision.v1',
              'status':status,'selected_label':best['label'],
              'selected_phase':best['phase'],'selected_phase_step':best['phase_step'],
              'selected_total_optimizer_steps':best['total_optimizer_steps'],
              'selected_worst_cell_P_p95':best['worst_cell_P_p95'],
              'selected_aggregate_P_p95':best['aggregate_P_p95'],
              'selected_per_cell':best['per_cell'],'threshold_P_p95_per_cell':THRESH,
              'all_cells_pass_0p005':best['all_cells_pass_0p005'],
              'planned_optimizer_steps_total':MAIN_STEPS+TAIL_STEPS,
              'main_steps':MAIN_STEPS,'tail_steps':TAIL_STEPS,
              'tune_consumed':False,'sealed_splits_opened':False,'camera_json_consumed':False,
              'next_policy':{
                  'P_V5_R256_36FB_TWO_STYLE_SUFFICIENCY_PASS':'freeze a fresh 8x2 V2 certification with evidence-backed adequate schedule; unseen-family remains closed',
                  'P_V5_R256_36FB_TWO_STYLE_OPTIMIZATION_INSUFFICIENT':'localize 36fb solo residual before any shared-capacity or PatchMatch claim; unseen-family remains closed'
              }[status],
              'candidates':[{k:v for k,v in c.items() if k!='checkpoint'} for c in candidates]}
    atomic_json(out/'P_V5_R256_36FB_TWO_STYLE_DECISION.json',decision)
    print(json.dumps(decision,indent=2,sort_keys=True),flush=True)

if __name__=='__main__':
    main()
