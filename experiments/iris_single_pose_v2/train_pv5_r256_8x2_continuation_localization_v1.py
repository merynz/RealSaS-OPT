from __future__ import annotations
import argparse, hashlib, json, os, random, shutil
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader
from dataset_pv5_r256_8x2 import PV5R256EightByTwoDataset
from evaluate_pv5_r256_8x2_v1 import evaluate
from model_pv5_r256 import IRISSinglePoseV2PV5R256
from pv5_depth_objective import p_only_objective

SEED=20260825
STEPS=2048
LR=3e-5
BETAS=(.9,.95)
WD=0.
EVAL_STEPS=(128,256,512,1024,1536,2048)
THRESH=.005
MICRO=8
ACCUM=2
PARENT_TOTAL_STEPS=2560
PARENT_LABEL='TAIL_0512'
ORIGINAL_STATUS='P_V5_R256_8X2_JOINT_FIT_INSUFFICIENT'
ALLOWED=('encoder.','within.','cross.','context_fuse.','d8.','d4.','d2.','p_s1_stem.','p_s1_fuse.','p_depth_head.')

def sha256_file(path,chunk=8<<20):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for x in iter(lambda:f.read(chunk),b''): h.update(x)
    return h.hexdigest()

def configure_p_only(model):
    for n,p in model.named_parameters(): p.requires_grad=any(n.startswith(x) for x in ALLOWED)
    tr=[n for n,p in model.named_parameters() if p.requires_grad]
    for head in ('n_head','u_geo_head','coarse_head','fine_head'):
        if any(n.startswith(head+'.') and p.requires_grad for n,p in model.named_parameters()):
            raise RuntimeError(f'{head} must be frozen')
    return tr

def atomic_json(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    os.replace(tmp,path)

def seed_all():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True

def slice_to(batch,s,e,device):
    return {k:(v[s:e].to(device,non_blocking=True) if torch.is_tensor(v) else v[s:e]) for k,v in batch.items()}

def candidate_key(c):
    return (float(c['worst_cell_P_p95']),float(c['aggregate_P_p95']),int(c['continuation_optimizer_steps']))

def save_cp(path,model,label,cont_step,parent_sha):
    torch.save({
        'schema':'RealSaS.IRISSinglePoseV2.PV5R256EightByTwoContinuationCheckpoint.v1',
        'label':label,
        'continuation_optimizer_steps':int(cont_step),
        'total_optimizer_steps_including_parent':int(PARENT_TOTAL_STEPS+cont_step),
        'parent_checkpoint_sha256':parent_sha,
        'model_state':{k:v.detach().cpu() for k,v in model.state_dict().items()},
    },path)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--cache-manifest',required=True)
    ap.add_argument('--init-checkpoint',required=True)
    ap.add_argument('--expected-parent-sha256',required=True)
    ap.add_argument('--out-dir',required=True)
    a=ap.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    seed_all(); device=torch.device('cuda'); out=Path(a.out_dir)
    out.mkdir(parents=True,exist_ok=True); (out/'eval').mkdir(exist_ok=True); (out/'checkpoints').mkdir(exist_ok=True)

    parent=Path(a.init_checkpoint)
    got_sha=sha256_file(parent)
    if got_sha!=a.expected_parent_sha256:
        raise RuntimeError(f'parent checkpoint SHA drift {got_sha} != {a.expected_parent_sha256}')
    ck=torch.load(parent,map_location='cpu',weights_only=False)
    if ck.get('label')!=PARENT_LABEL or int(ck.get('total_optimizer_steps',-1))!=PARENT_TOTAL_STEPS:
        raise RuntimeError(f'parent checkpoint identity drift: {ck.get("label")} steps={ck.get("total_optimizer_steps")}')

    ds=PV5R256EightByTwoDataset(a.cache_manifest)
    dl=DataLoader(ds,batch_size=16,shuffle=False,num_workers=0,pin_memory=True)
    cpu=next(iter(dl))
    if len(cpu['asset_id'])!=16: raise RuntimeError('full 16-cell batch missing')
    model=IRISSinglePoseV2PV5R256().to(device)
    model.load_state_dict(ck['model_state'],strict=True)
    tr=configure_p_only(model)

    authority={
        'schema':'RealSaS.IRISSinglePoseV2.PV5R256EightByTwoContinuationAuthority.v1',
        'scientific_role':'LOCALIZATION_ONLY__DOES_NOT_REWRITE_ORIGINAL_8X2_FAIL',
        'original_8x2_status_immutable':ORIGINAL_STATUS,
        'parent_checkpoint_label':PARENT_LABEL,
        'parent_checkpoint_total_optimizer_steps':PARENT_TOTAL_STEPS,
        'parent_checkpoint_sha256':got_sha,
        'fresh_model_initialization':False,
        'fresh_optimizer_moments':True,
        'seed':SEED,
        'input_resolution':256,'output_field_hw':256,
        'membership':'same frozen 8 FIT assets x 2 styles = 16 cells',
        'microbatch_cells':MICRO,'gradient_accumulation_steps':ACCUM,'effective_cells_per_optimizer_step':16,
        'augmentation':False,
        'objective':'same full-16-cell SmoothL1 camera-forward depth objective',
        'lr':LR,'steps':STEPS,'optimizer':'AdamW','betas':list(BETAS),'weight_decay':WD,
        'eval_steps':list(EVAL_STEPS),
        'target_P_p95_per_cell':THRESH,
        'pass_interpretation':'additional shared low-LR optimization from the failed TAIL_0512 weights, with fresh AdamW moments, is sufficient; this does not isolate pure extra-step budget from optimizer-reset effects',
        'fail_interpretation':'additional shared low-LR optimization under this frozen continuation remains insufficient; shared capacity/interference becomes more plausible',
        'promotion_rule':'neither PASS nor FAIL directly authorizes unseen-family; PASS authorizes preregistration of a fresh 8x2 V2 certification run with an adequate schedule',
        'patchmatch_admitted':False,
        'architecture_changed':False,
        'tune_consumed':False,'sealed_splits_opened':False,'camera_json_consumed':False,
        'trainable_parameters':tr,
    }
    atomic_json(out/'CONTINUATION_AUTHORITY.json',authority)

    init=evaluate(model,a.cache_manifest,device)
    init.update(label='CONT_INIT',continuation_optimizer_steps=0,total_optimizer_steps_including_parent=PARENT_TOTAL_STEPS,lr=None)
    atomic_json(out/'eval'/'CONT_INIT.json',init)
    print(json.dumps({k:v for k,v in init.items() if k!='per_cell'},indent=2),flush=True)

    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=LR,weight_decay=WD,betas=BETAS)
    scaler=torch.amp.GradScaler('cuda',enabled=True)
    history=[]; cands=[]
    for step in range(1,STEPS+1):
        model.train(); opt.zero_grad(set_to_none=True); p95s=[]; means=[]; losses=[]
        for s,e in ((0,8),(8,16)):
            b=slice_to(cpu,s,e,device)
            with torch.autocast(device_type='cuda',dtype=torch.float16):
                outp=model(b['images'],b['yaw_deg'],b['sheet_half_extent'])
            if tuple(outp['P'].shape)!=(8,8,3,256,256): raise RuntimeError(f'continuation field drift {tuple(outp["P"].shape)}')
            parts=p_only_objective(outp,{'geom_xy':b['geom_xy'],'geom_p':b['geom_p'],'geom_mask':b['geom_mask'],'yaw_deg':b['yaw_deg']})
            scaler.scale(parts['total']/ACCUM).backward()
            losses.append(float(parts['depth_smooth_l1'].detach().cpu())); p95s.append(float(parts['p_p95'].detach().cpu())); means.append(float(parts['p_mean'].detach().cpu()))
            del b,outp,parts
        scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0); scaler.step(opt); scaler.update()
        if step==1 or step%16==0 or step in EVAL_STEPS:
            rec={'continuation_optimizer_steps':step,'total_optimizer_steps_including_parent':PARENT_TOTAL_STEPS+step,'lr':LR,'microbatch_cells':MICRO,'accumulation_steps':ACCUM,'depth_smooth_l1_micro_mean':float(np.mean(losses)),'train_P_p95_micro_max':max(p95s),'train_P_mean_micro_mean':float(np.mean(means))}
            history.append(rec); atomic_json(out/'CONTINUATION_HISTORY.json',{'records':history})
        if step in EVAL_STEPS:
            ev=evaluate(model,a.cache_manifest,device); label=f'CONT_{step:04d}'
            ev.update(label=label,continuation_optimizer_steps=step,total_optimizer_steps_including_parent=PARENT_TOTAL_STEPS+step,lr=LR)
            atomic_json(out/'eval'/f'{label}.json',ev)
            cp=out/'checkpoints'/f'{label}.pt'; save_cp(cp,model,label,step,got_sha)
            cand={'label':label,'continuation_optimizer_steps':step,'total_optimizer_steps_including_parent':PARENT_TOTAL_STEPS+step,'lr':LR,'worst_cell_P_p95':ev['worst_cell_P_p95'],'aggregate_P_p95':ev['aggregate']['P_p95'],'all_cells_pass_0p005':ev['all_cells_pass_0p005'],'pass_count':ev['pass_count'],'per_cell':ev['per_cell'],'checkpoint':str(cp)}
            cands.append(cand); print(json.dumps({k:v for k,v in cand.items() if k not in ('per_cell','checkpoint')},indent=2),flush=True)

    best=min(cands,key=candidate_key)
    shutil.copy2(best['checkpoint'],out/'BEST_CONTINUATION_CHECKPOINT.pt')
    status='P_V5_R256_8X2_ADDITIONAL_LOW_LR_OPTIMIZATION_SUFFICIENT' if best['all_cells_pass_0p005'] else 'P_V5_R256_8X2_ADDITIONAL_LOW_LR_OPTIMIZATION_NOT_SUFFICIENT'
    dec={
        'schema':'RealSaS.IRISSinglePoseV2.PV5R256EightByTwoContinuationDecision.v1',
        'status':status,
        'original_8x2_status_immutable':ORIGINAL_STATUS,
        'selected_label':best['label'],
        'selected_continuation_optimizer_steps':best['continuation_optimizer_steps'],
        'selected_total_optimizer_steps_including_parent':best['total_optimizer_steps_including_parent'],
        'selected_worst_cell_P_p95':best['worst_cell_P_p95'],
        'selected_aggregate_P_p95':best['aggregate_P_p95'],
        'selected_pass_count':best['pass_count'],
        'selected_per_cell':best['per_cell'],
        'threshold_P_p95_per_cell':THRESH,
        'all_cells_pass_0p005':best['all_cells_pass_0p005'],
        'continuation_lr':LR,'continuation_steps_planned':STEPS,'fresh_optimizer_moments':True,
        'patchmatch_admitted':False,'architecture_changed':False,
        'tune_consumed':False,'sealed_splits_opened':False,'camera_json_consumed':False,
        'next_policy':('preregister a fresh 8x2 V2 certification run with an evidence-backed adequate schedule; do not open unseen-family yet' if best['all_cells_pass_0p005'] else 'remain at 8x2; localize shared capacity/interference before architecture change; PatchMatch remains unjustified absent geometric residual localization'),
        'candidates':[{k:v for k,v in c.items() if k!='checkpoint'} for c in cands],
    }
    atomic_json(out/'P_V5_R256_8X2_CONTINUATION_DECISION.json',dec)
    print(json.dumps({k:v for k,v in dec.items() if k!='candidates'},indent=2,sort_keys=True),flush=True)

if __name__=='__main__': main()
