from __future__ import annotations
import argparse, json, os, random, shutil
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader
from dataset_pv5_r256_8x2 import PV5R256EightByTwoDataset
from evaluate_pv5_r256_8x2_v1 import evaluate
from model_pv5_r256 import IRISSinglePoseV2PV5R256
from pv5_depth_objective import p_only_objective
SEED=20260825; MAIN_STEPS=2048; TAIL_STEPS=512; MAIN_LR=3e-4; TAIL_LR=3e-5; BETAS=(.9,.95); WD=0.; MAIN_EVAL=(512,1024,2048); TAIL_EVAL=(64,128,256,512); THRESH=.005; MICRO=8; ACCUM=2
ALLOWED=('encoder.','within.','cross.','context_fuse.','d8.','d4.','d2.','p_s1_stem.','p_s1_fuse.','p_depth_head.')
def configure_p_only(model):
    for n,p in model.named_parameters(): p.requires_grad=any(n.startswith(x) for x in ALLOWED)
    tr=[n for n,p in model.named_parameters() if p.requires_grad]
    for head in ('n_head','u_geo_head','coarse_head','fine_head'):
        if any(n.startswith(head+'.') and p.requires_grad for n,p in model.named_parameters()): raise RuntimeError(f'{head} must be frozen')
    return tr
def atomic_json(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n',encoding='utf-8'); os.replace(tmp,path)
def seed_all():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED); torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
def slice_to(batch,s,e,device): return {k:(v[s:e].to(device,non_blocking=True) if torch.is_tensor(v) else v[s:e]) for k,v in batch.items()}
def candidate_key(c): return (float(c['worst_cell_P_p95']),float(c['aggregate_P_p95']),int(c['total_optimizer_steps']))
def save_cp(path,model,label,total,phase,phase_step): torch.save({'schema':'RealSaS.IRISSinglePoseV2.PV5R256EightByTwoCheckpoint.v1','label':label,'total_optimizer_steps':int(total),'phase':phase,'phase_step':int(phase_step),'model_state':{k:v.detach().cpu() for k,v in model.state_dict().items()}},path)
def run_phase(model,cpu_batch,device,out,cm,phase,steps,lr,eval_steps,offset,history,cands):
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=lr,weight_decay=WD,betas=BETAS); scaler=torch.amp.GradScaler('cuda',enabled=True)
    for step in range(1,steps+1):
        model.train(); opt.zero_grad(set_to_none=True); p95s=[]; means=[]; losses=[]
        for mi,(s,e) in enumerate(((0,8),(8,16))):
            b=slice_to(cpu_batch,s,e,device)
            with torch.autocast(device_type='cuda',dtype=torch.float16): outp=model(b['images'],b['yaw_deg'],b['sheet_half_extent'])
            if tuple(outp['P'].shape)!=(8,8,3,256,256): raise RuntimeError(f'8x2 microbatch field drift {tuple(outp["P"].shape)}')
            parts=p_only_objective(outp,{'geom_xy':b['geom_xy'],'geom_p':b['geom_p'],'geom_mask':b['geom_mask'],'yaw_deg':b['yaw_deg']})
            scaler.scale(parts['total']/ACCUM).backward(); losses.append(float(parts['depth_smooth_l1'].detach().cpu())); p95s.append(float(parts['p_p95'].detach().cpu())); means.append(float(parts['p_mean'].detach().cpu()))
            del b,outp,parts
        scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0); scaler.step(opt); scaler.update(); total=offset+step
        if step==1 or step%16==0 or step in eval_steps:
            rec={'phase':phase,'phase_step':step,'total_optimizer_steps':total,'lr':lr,'microbatch_cells':MICRO,'accumulation_steps':ACCUM,'depth_smooth_l1_micro_mean':float(np.mean(losses)),'train_P_p95_micro_max':max(p95s),'train_P_mean_micro_mean':float(np.mean(means))}; history.append(rec); atomic_json(out/'TRAIN_HISTORY.json',{'records':history})
        if step in eval_steps:
            ev=evaluate(model,cm,device); label=f'{phase}_{step:04d}'; ev.update(label=label,phase=phase,phase_step=step,total_optimizer_steps=total,lr=lr); atomic_json(out/'eval'/f'{label}.json',ev); cp=out/'checkpoints'/f'{label}.pt'; save_cp(cp,model,label,total,phase,step)
            cand={'label':label,'phase':phase,'phase_step':step,'total_optimizer_steps':total,'lr':lr,'worst_cell_P_p95':ev['worst_cell_P_p95'],'aggregate_P_p95':ev['aggregate']['P_p95'],'all_cells_pass_0p005':ev['all_cells_pass_0p005'],'pass_count':ev['pass_count'],'per_cell':ev['per_cell'],'checkpoint':str(cp)}; cands.append(cand); print(json.dumps({k:v for k,v in cand.items() if k not in ('per_cell','checkpoint')},indent=2),flush=True)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--cache-manifest',required=True); ap.add_argument('--out-dir',required=True); a=ap.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required'); seed_all(); device=torch.device('cuda'); out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True); (out/'eval').mkdir(exist_ok=True); (out/'checkpoints').mkdir(exist_ok=True)
    ds=PV5R256EightByTwoDataset(a.cache_manifest); dl=DataLoader(ds,batch_size=16,shuffle=False,num_workers=0,pin_memory=True); cpu=next(iter(dl))
    if len(cpu['asset_id'])!=16: raise RuntimeError('full 16-cell batch missing')
    model=IRISSinglePoseV2PV5R256().to(device); tr=configure_p_only(model)
    auth={'schema':'RealSaS.IRISSinglePoseV2.PV5R256EightByTwoTrainingAuthority.v1','seed':SEED,'fresh_model_initialization':True,'input_resolution':256,'output_field_hw':256,'membership':'8 frozen FIT assets x 2 styles; all 16 cells contribute before every optimizer update','batch_size_cells_effective':16,'microbatch_cells':MICRO,'gradient_accumulation_steps':ACCUM,'views_per_cell':8,'augmentation':False,'objective':'full 16-cell FP32 SmoothL1 camera-forward depth objective via exact equal-size microbatch accumulation','evaluation_authority':'16 cell-wise canonical P Euclidean p95; aggregate alone cannot pass','target_P_p95_per_cell':THRESH,'schedule':[{'phase':'MAIN','lr':MAIN_LR,'steps':MAIN_STEPS,'eval_steps':list(MAIN_EVAL),'optimizer':'AdamW','fresh_moments':True,'betas':list(BETAS),'weight_decay':WD},{'phase':'TAIL','lr':TAIL_LR,'steps':TAIL_STEPS,'eval_steps':list(TAIL_EVAL),'optimizer':'AdamW','fresh_moments':True,'betas':list(BETAS),'weight_decay':WD}],'tune_consumed':False,'sealed_splits_opened':False,'camera_json_consumed':False,'trainable_parameters':tr}; atomic_json(out/'TRAIN_AUTHORITY.json',auth)
    init=evaluate(model,a.cache_manifest,device); init.update(label='INIT',phase='INIT',phase_step=0,total_optimizer_steps=0,lr=None); atomic_json(out/'eval'/'INIT.json',init)
    hist=[]; cands=[]; run_phase(model,cpu,device,out,a.cache_manifest,'MAIN',MAIN_STEPS,MAIN_LR,MAIN_EVAL,0,hist,cands); run_phase(model,cpu,device,out,a.cache_manifest,'TAIL',TAIL_STEPS,TAIL_LR,TAIL_EVAL,MAIN_STEPS,hist,cands)
    best=min(cands,key=candidate_key); shutil.copy2(best['checkpoint'],out/'BEST_CHECKPOINT.pt'); status='P_V5_R256_8X2_JOINT_FIT_PASS' if best['all_cells_pass_0p005'] else 'P_V5_R256_8X2_JOINT_FIT_INSUFFICIENT'
    dec={'schema':'RealSaS.IRISSinglePoseV2.PV5R256EightByTwoDecision.v1','status':status,'selected_label':best['label'],'selected_phase':best['phase'],'selected_phase_step':best['phase_step'],'selected_total_optimizer_steps':best['total_optimizer_steps'],'selected_worst_cell_P_p95':best['worst_cell_P_p95'],'selected_aggregate_P_p95':best['aggregate_P_p95'],'selected_pass_count':best['pass_count'],'selected_per_cell':best['per_cell'],'threshold_P_p95_per_cell':THRESH,'all_cells_pass_0p005':best['all_cells_pass_0p005'],'planned_optimizer_steps_total':MAIN_STEPS+TAIL_STEPS,'main_steps':MAIN_STEPS,'tail_steps':TAIL_STEPS,'tune_consumed':False,'sealed_splits_opened':False,'camera_json_consumed':False,'next_policy':('preregister unseen-family generalization only' if status.endswith('_PASS') else 'remain at 8x2; localize cell/family failure pattern before architecture change or unseen-family evaluation'),'candidates':[{k:v for k,v in c.items() if k!='checkpoint'} for c in cands]}; atomic_json(out/'P_V5_R256_8X2_DECISION.json',dec); print(json.dumps({k:v for k,v in dec.items() if k!='candidates'},indent=2,sort_keys=True),flush=True)
if __name__=='__main__': main()
