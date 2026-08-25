from __future__ import annotations
import argparse, copy, json, os, random, shutil
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader
from dataset_pv5_r256_8x2 import PV5R256EightByTwoDataset
from evaluate_pv5_r256_8x2_v1 import evaluate
from model_pv5_r256 import IRISSinglePoseV2PV5R256
from pv5_depth_objective import p_only_objective

LR=3e-5
BETAS=(.9,.95)
WD=0.0
EXTRA_STEPS=1024
EVAL_STEPS=(128,256,512,768,1024)
THRESH=.005
MICRO=8
ACCUM=2
PARENT_TOTAL=6144
PARENT_TAIL_STEP=4096
PARENT_WORST=0.005129679851233959
PARENT_REPRO_TOL=5e-5
ALLOWED=('encoder.','within.','cross.','context_fuse.','d8.','d4.','d2.','p_s1_stem.','p_s1_fuse.','p_depth_head.')

def configure_p_only(model):
    for n,p in model.named_parameters(): p.requires_grad=any(n.startswith(x) for x in ALLOWED)
    tr=[n for n,p in model.named_parameters() if p.requires_grad]
    for head in ('n_head','u_geo_head','coarse_head','fine_head'):
        if any(n.startswith(head+'.') and p.requires_grad for n,p in model.named_parameters()):
            raise RuntimeError(f'{head} must be frozen')
    return tr

def atomic_json(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n',encoding='utf-8'); os.replace(tmp,path)

def slice_to(batch,s,e,device):
    return {k:(v[s:e].to(device,non_blocking=True) if torch.is_tensor(v) else v[s:e]) for k,v in batch.items()}

def candidate_key(c): return (float(c['worst_cell_P_p95']),float(c['aggregate_P_p95']),int(c['total_optimizer_steps']))

def _cpu_clone(obj):
    if torch.is_tensor(obj): return obj.detach().cpu().clone()
    if isinstance(obj,dict): return {k:_cpu_clone(v) for k,v in obj.items()}
    if isinstance(obj,list): return [_cpu_clone(v) for v in obj]
    if isinstance(obj,tuple): return tuple(_cpu_clone(v) for v in obj)
    return copy.deepcopy(obj)

def checkpoint_payload(model,opt,scaler,label,total,tail_phase_step,continuation_step):
    return {
        'schema':'RealSaS.IRISSinglePoseV2.PV5R256EightByTwoCheckpoint.v2',
        'label':label,
        'total_optimizer_steps':int(total),
        'phase':'TAIL_EXACT_CONTINUATION',
        'phase_step':int(tail_phase_step),
        'continuation_step':int(continuation_step),
        'model_state':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
        'optimizer_state':_cpu_clone(opt.state_dict()),
        'scaler_state':copy.deepcopy(scaler.state_dict()),
        'rng_state':{
            'python_random_state':random.getstate(),
            'numpy_random_state':np.random.get_state(),
            'torch_cpu_rng_state':torch.get_rng_state().cpu(),
            'torch_cuda_rng_state_all':[x.cpu() for x in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [],
        },
        'resume_semantics':'exact continuation of V2 TAIL optimizer/scaler/RNG state; no fresh moments',
    }

def save_cp(path,model,opt,scaler,label,total,tail_phase_step,continuation_step):
    path=Path(path); tmp=path.with_suffix(path.suffix+'.tmp')
    torch.save(checkpoint_payload(model,opt,scaler,label,total,tail_phase_step,continuation_step),tmp); os.replace(tmp,path)

def restore_rng(r):
    random.setstate(r['python_random_state'])
    np.random.set_state(r['numpy_random_state'])
    torch.set_rng_state(r['torch_cpu_rng_state'])
    if torch.cuda.is_available():
        states=r.get('torch_cuda_rng_state_all',[])
        if len(states)!=torch.cuda.device_count():
            raise RuntimeError(f'CUDA RNG device-count drift: checkpoint {len(states)} vs runtime {torch.cuda.device_count()}')
        torch.cuda.set_rng_state_all(states)

def per_cell_map(ev):
    return {(x['asset_id'],x['style']):float(x['P_p95']) for x in ev['per_cell']}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--cache-manifest',required=True)
    ap.add_argument('--parent-checkpoint',required=True)
    ap.add_argument('--parent-decision',required=True)
    ap.add_argument('--out-dir',required=True)
    a=ap.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    device=torch.device('cuda'); out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True); (out/'eval').mkdir(exist_ok=True); (out/'checkpoints').mkdir(exist_ok=True)

    ds=PV5R256EightByTwoDataset(a.cache_manifest); dl=DataLoader(ds,batch_size=16,shuffle=False,num_workers=0,pin_memory=True); cpu=next(iter(dl))
    if len(cpu['asset_id'])!=16: raise RuntimeError('full 16-cell batch missing')
    parent_dec=json.load(open(a.parent_decision,encoding='utf-8'))
    if parent_dec.get('status')!='P_V5_R256_8X2_V2_CERTIFICATION_INSUFFICIENT' or parent_dec.get('selected_label')!='TAIL_4096' or parent_dec.get('selected_total_optimizer_steps')!=PARENT_TOTAL or parent_dec.get('selected_pass_count')!=14:
        raise RuntimeError('parent decision identity drift')

    cp=torch.load(a.parent_checkpoint,map_location='cpu',weights_only=False)
    if cp.get('schema')!='RealSaS.IRISSinglePoseV2.PV5R256EightByTwoCheckpoint.v2' or cp.get('label')!='TAIL_4096' or cp.get('phase')!='TAIL' or cp.get('phase_step')!=PARENT_TAIL_STEP or cp.get('total_optimizer_steps')!=PARENT_TOTAL:
        raise RuntimeError('parent checkpoint identity drift')
    for key in ('model_state','optimizer_state','scaler_state','rng_state'):
        if key not in cp: raise RuntimeError(f'parent checkpoint missing {key}')

    model=IRISSinglePoseV2PV5R256().to(device); tr=configure_p_only(model); model.load_state_dict(cp['model_state'],strict=True)
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=LR,weight_decay=WD,betas=BETAS); opt.load_state_dict(cp['optimizer_state'])
    for g in opt.param_groups:
        if abs(float(g['lr'])-LR)>1e-12 or tuple(g['betas'])!=BETAS or abs(float(g['weight_decay'])-WD)>1e-12:
            raise RuntimeError(f'parent optimizer hyperparameter drift: {g}')
    scaler=torch.amp.GradScaler('cuda',enabled=True); scaler.load_state_dict(cp['scaler_state'])

    parent_ev=evaluate(model,a.cache_manifest,device)
    parent_ref={(x['asset_id'],x['style']):float(x['P_p95']) for x in parent_dec['selected_per_cell']}
    parent_now=per_cell_map(parent_ev)
    if set(parent_ref)!=set(parent_now): raise RuntimeError('parent re-eval cell identity drift')
    max_delta=max(abs(parent_now[k]-parent_ref[k]) for k in parent_ref)
    if parent_ev['pass_count']!=14 or abs(float(parent_ev['worst_cell_P_p95'])-PARENT_WORST)>PARENT_REPRO_TOL or max_delta>PARENT_REPRO_TOL:
        raise RuntimeError(f'parent re-eval drift before optimizer: pass={parent_ev["pass_count"]}, worst={parent_ev["worst_cell_P_p95"]}, max_cell_delta={max_delta}')
    parent_ev.update(label='PARENT_REPRO',phase='PARENT',phase_step=PARENT_TAIL_STEP,total_optimizer_steps=PARENT_TOTAL,lr=LR,max_abs_cell_P_p95_delta_vs_parent=max_delta,scientific_optimizer_steps=0)
    atomic_json(out/'eval'/'PARENT_REPRO.json',parent_ev)

    auth={
        'schema':'RealSaS.IRISSinglePoseV2.PV5R256EightByTwoExactContinuationTrainingAuthority.v1',
        'parent_label':'TAIL_4096','parent_total_optimizer_steps':PARENT_TOTAL,'parent_tail_phase_step':PARENT_TAIL_STEP,
        'checkpoint_reuse':'exact full-state resume','fresh_optimizer_moments':False,'optimizer_state_restored':True,'scaler_state_restored':True,'rng_state_restored':True,
        'lr':LR,'additional_steps':EXTRA_STEPS,'eval_steps':list(EVAL_STEPS),'run_full_schedule_even_if_pass_seen_early':True,
        'input_resolution':256,'output_field_hw':256,'batch_size_cells_effective':16,'microbatch_cells':MICRO,'gradient_accumulation_steps':ACCUM,'views_per_cell':8,
        'objective':'unchanged full 16-cell FP32 SmoothL1 camera-forward depth objective via exact equal-size microbatch accumulation',
        'evaluation_authority':'16 cell-wise canonical P Euclidean p95; aggregate alone cannot pass','target_P_p95_per_cell':THRESH,
        'parent_reproduction_max_abs_cell_P_p95_delta':max_delta,
        'patchmatch_admitted':False,'architecture_changed':False,'augmentation':False,'tune_consumed':False,'sealed_splits_opened':False,'camera_json_consumed':False,
        'trainable_parameters':tr,
    }
    atomic_json(out/'CONTINUATION_TRAIN_AUTHORITY.json',auth)

    restore_rng(cp['rng_state'])
    hist=[]; cands=[]
    for step in range(1,EXTRA_STEPS+1):
        model.train(); opt.zero_grad(set_to_none=True); p95s=[]; means=[]; losses=[]
        for s,e in ((0,8),(8,16)):
            b=slice_to(cpu,s,e,device)
            with torch.autocast(device_type='cuda',dtype=torch.float16): outp=model(b['images'],b['yaw_deg'],b['sheet_half_extent'])
            if tuple(outp['P'].shape)!=(8,8,3,256,256): raise RuntimeError(f'8x2 microbatch field drift {tuple(outp["P"].shape)}')
            parts=p_only_objective(outp,{'geom_xy':b['geom_xy'],'geom_p':b['geom_p'],'geom_mask':b['geom_mask'],'yaw_deg':b['yaw_deg']})
            scaler.scale(parts['total']/ACCUM).backward(); losses.append(float(parts['depth_smooth_l1'].detach().cpu())); p95s.append(float(parts['p_p95'].detach().cpu())); means.append(float(parts['p_mean'].detach().cpu()))
            del b,outp,parts
        scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0); scaler.step(opt); scaler.update()
        total=PARENT_TOTAL+step; tail_step=PARENT_TAIL_STEP+step
        if step==1 or step%16==0 or step in EVAL_STEPS:
            rec={'phase':'TAIL_EXACT_CONTINUATION','continuation_step':step,'tail_phase_step':tail_step,'total_optimizer_steps':total,'lr':LR,'microbatch_cells':MICRO,'accumulation_steps':ACCUM,'depth_smooth_l1_micro_mean':float(np.mean(losses)),'train_P_p95_micro_max':max(p95s),'train_P_mean_micro_mean':float(np.mean(means))}
            hist.append(rec); atomic_json(out/'CONTINUATION_HISTORY.json',{'records':hist})
        if step in EVAL_STEPS:
            ev=evaluate(model,a.cache_manifest,device); label=f'EXACT_CONT_{step:04d}'
            ev.update(label=label,phase='TAIL_EXACT_CONTINUATION',continuation_step=step,tail_phase_step=tail_step,total_optimizer_steps=total,lr=LR)
            atomic_json(out/'eval'/f'{label}.json',ev); cpout=out/'checkpoints'/f'{label}.pt'; save_cp(cpout,model,opt,scaler,label,total,tail_step,step)
            cand={'label':label,'continuation_step':step,'tail_phase_step':tail_step,'total_optimizer_steps':total,'lr':LR,'worst_cell_P_p95':ev['worst_cell_P_p95'],'aggregate_P_p95':ev['aggregate']['P_p95'],'all_cells_pass_0p005':ev['all_cells_pass_0p005'],'pass_count':ev['pass_count'],'per_cell':ev['per_cell'],'checkpoint':str(cpout)}
            cands.append(cand); print(json.dumps({k:v for k,v in cand.items() if k not in ('per_cell','checkpoint')},indent=2),flush=True)

    best=min(cands,key=candidate_key); shutil.copy2(best['checkpoint'],out/'BEST_CONTINUATION_CHECKPOINT.pt')
    status='P_V5_R256_8X2_V2_EXACT_CONTINUATION_PASS' if any(c['all_cells_pass_0p005'] for c in cands) else 'P_V5_R256_8X2_V2_EXACT_CONTINUATION_INSUFFICIENT'
    passing=[c for c in cands if c['all_cells_pass_0p005']]
    selected=min(passing,key=candidate_key) if passing else best
    if selected is not best: shutil.copy2(selected['checkpoint'],out/'BEST_CONTINUATION_CHECKPOINT.pt')
    dec={
        'schema':'RealSaS.IRISSinglePoseV2.PV5R256EightByTwoExactContinuationDecision.v1','status':status,
        'parent_status':'P_V5_R256_8X2_V2_CERTIFICATION_INSUFFICIENT','parent_label':'TAIL_4096','parent_total_optimizer_steps':PARENT_TOTAL,'parent_worst_cell_P_p95':PARENT_WORST,
        'fresh_optimizer_moments':False,'optimizer_state_restored':True,'scaler_state_restored':True,'rng_state_restored':True,
        'additional_steps':EXTRA_STEPS,'lr':LR,'selected_label':selected['label'],'selected_continuation_step':selected['continuation_step'],'selected_total_optimizer_steps':selected['total_optimizer_steps'],'selected_worst_cell_P_p95':selected['worst_cell_P_p95'],'selected_aggregate_P_p95':selected['aggregate_P_p95'],'selected_pass_count':selected['pass_count'],'selected_per_cell':selected['per_cell'],'all_cells_pass_0p005':selected['all_cells_pass_0p005'],'threshold_P_p95_per_cell':THRESH,
        'patchmatch_admitted':False,'architecture_changed':False,'tune_consumed':False,'sealed_splits_opened':False,'camera_json_consumed':False,
        'next_policy':('freeze/preregister unseen-family generalization before opening any unseen data' if status.endswith('_PASS') else 'remain at 8x2; stop blind budget extension and localize shared capacity/interference'),
        'candidates':[{k:v for k,v in c.items() if k!='checkpoint'} for c in cands],
    }
    atomic_json(out/'P_V5_R256_8X2_V2_EXACT_CONTINUATION_DECISION.json',dec)
    print(json.dumps({k:v for k,v in dec.items() if k!='candidates'},indent=2,sort_keys=True),flush=True)

if __name__=='__main__': main()
