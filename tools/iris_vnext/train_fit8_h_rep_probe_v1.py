#!/usr/bin/env python3
from __future__ import annotations

"""Train equal-capacity H-REP-P0 control probes over exact sealed V8 features."""

import argparse, json, os, shutil, time
from pathlib import Path
from typing import Any

import torch

from models.iris.vnext.cohort_v1 import SUBJECT_IDS
from models.iris.vnext.h_rep_probe_v1 import PROBE_ARMS_V1, HRepresentationProbeV1, build_probe_input_v1, h_rep_probe_objective_v1
from tools.iris_vnext.h_rep_probe_runtime_v1 import cache_frozen_v8_probe_features_v1, load_h_rep_binding_v1
from tools.iris_vnext.materialize_fit8_direct_surface_v1 import git_head, load_json, sha256_file

SCHEMA="RealSaS.HRepP0TrainingResult.v1"; CHECKPOINT_SCHEMA="RealSaS.HRepP0Checkpoint.v1"
TOTAL_STEPS=1200; BATCH_SIZE=2048

def _atomic_torch(value:dict[str,Any],path:Path)->None:
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+'.tmp'); torch.save(value,tmp); os.replace(tmp,path)
def _atomic_json(value:dict[str,Any],path:Path)->None:
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n'); os.replace(tmp,path)
def _lr(step:int)->float:
    warm=60
    if step < warm: return 2e-4*float(step+1)/warm
    t=(step-warm)/max(1,TOTAL_STEPS-warm-1); return 2e-5+0.5*(2e-4-2e-5)*(1.0+__import__('math').cos(__import__('math').pi*t))

def train(args:argparse.Namespace)->dict[str,Any]:
    repo=Path(args.repo_root).resolve(); source_sha=str(args.expected_source_sha).lower()
    if git_head(repo)!=source_sha: raise RuntimeError('H_REP_P0_SOURCE_SHA_DRIFT')
    binding_path=Path(args.binding).resolve(); prereg_path=Path(args.preregistration).resolve(); dataset_receipt=Path(args.dataset_receipt).resolve()
    binding=load_h_rep_binding_v1(binding_path); prereg=load_json(prereg_path)
    if prereg.get('schema')!='RealSaS.HRepP0Preregistration.v1': raise ValueError('H_REP_P0_PREREG_SCHEMA_INVALID')
    binding_sha=sha256_file(binding_path); prereg_sha=sha256_file(prereg_path); dataset_sha=sha256_file(dataset_receipt)
    device=torch.device('cuda')
    if not torch.cuda.is_available(): raise RuntimeError('H_REP_P0_CUDA_REQUIRED')
    torch.manual_seed(26101990); torch.cuda.manual_seed_all(26101990)
    cache,scene_hash=cache_frozen_v8_probe_features_v1(v8_checkpoint=Path(args.v8_checkpoint),prepared_root=Path(args.prepared_root),prepared_manifest=Path(args.prepared_manifest),dataset_root=Path(args.dataset_root),dataset_receipt=dataset_receipt,binding=binding,split='train',device=device,query_chunk=int(args.query_chunk))
    models={arm:HRepresentationProbeV1(arm).to(device) for arm in PROBE_ARMS_V1}; opts={arm:torch.optim.AdamW(models[arm].parameters(),lr=2e-4,betas=(0.9,0.95),weight_decay=0.01) for arm in PROBE_ARMS_V1}
    output=Path(args.output_dir).resolve(); mirror=Path(args.checkpoint_mirror_dir).resolve(); output.mkdir(parents=True,exist_ok=True); mirror.mkdir(parents=True,exist_ok=True)
    local=output/'checkpoint_latest.pt'; remote=mirror/'checkpoint_latest.pt'; completed=0
    if not local.is_file() and remote.is_file(): shutil.copy2(remote,local)
    if local.is_file():
        cp=torch.load(local,map_location='cpu',weights_only=True)
        if cp.get('schema')!=CHECKPOINT_SCHEMA: raise RuntimeError('H_REP_P0_RESUME_SCHEMA_DRIFT')
        expected={'execution_source_sha':source_sha,'binding_sha256':binding_sha,'preregistration_sha256':prereg_sha,'dataset_receipt_sha256':dataset_sha,'v8_scene_core_sha256':scene_hash}
        for k,v in expected.items():
            if cp.get(k)!=v: raise RuntimeError('H_REP_P0_RESUME_AUTHORITY_DRIFT:'+k)
        for arm in PROBE_ARMS_V1: models[arm].load_state_dict(cp['models'][arm],strict=True); opts[arm].load_state_dict(cp['optimizers'][arm])
        completed=int(cp['completed_steps']); print('H_REP_P0_RESUME',completed,flush=True)
    started=time.perf_counter(); gens={s:torch.Generator(device='cpu').manual_seed(26104000+i) for i,s in enumerate(SUBJECT_IDS)}
    for step in range(completed,TOTAL_STEPS):
        subject=SUBJECT_IDS[step%len(SUBJECT_IDS)]; item=cache[subject]; n=len(item['xyz_iris']); idx=torch.randint(0,n,(min(BATCH_SIZE,n),),generator=gens[subject])
        batch={k:v[idx].to(device=device) for k,v in item.items() if k!='shuffle_index'}; shuffled=item['H'][item['shuffle_index'][idx]].to(device=device)
        lr=_lr(step); logs={}
        for arm in PROBE_ARMS_V1:
            for group in opts[arm].param_groups: group['lr']=lr
            opts[arm].zero_grad(set_to_none=True)
            x=build_probe_input_v1(arm,h=batch['H'],xyz_iris=batch['xyz_iris'],raw_field=batch['raw_field'],shuffled_h=shuffled if arm=='SHUFFLED_H_XYZ' else None)
            pred=models[arm](x); losses=h_rep_probe_objective_v1(pred,distance_iris=batch['distance_iris'],offset_to_reference_iris=batch['offset_to_reference_iris'],normal_unit=batch['normal_unit'],normal_valid=batch['normal_valid'].bool(),near_label=batch['near_label'].bool())
            if not bool(torch.isfinite(losses['total']).item()): raise RuntimeError('H_REP_P0_NONFINITE_LOSS:'+arm)
            losses['total'].backward(); gn=torch.nn.utils.clip_grad_norm_(models[arm].parameters(),1.0)
            if not bool(torch.isfinite(torch.as_tensor(gn)).item()): raise RuntimeError('H_REP_P0_NONFINITE_GRAD:'+arm)
            opts[arm].step(); logs[arm]=float(losses['total'].detach().item())
        done=step+1
        if done%20==0 or done in (1,TOTAL_STEPS): print('H_REP_P0_STEP',f'step={done}/{TOTAL_STEPS}',f'subject={subject}',f'lr={lr:.8g}',*(f'{a}={logs[a]:.6f}' for a in PROBE_ARMS_V1),flush=True)
        if done%200==0 or done==TOTAL_STEPS:
            cp={'schema':CHECKPOINT_SCHEMA,'completed_steps':done,'execution_source_sha':source_sha,'binding_sha256':binding_sha,'preregistration_sha256':prereg_sha,'dataset_receipt_sha256':dataset_sha,'v8_scene_core_sha256':scene_hash,'models':{a:models[a].state_dict() for a in PROBE_ARMS_V1},'optimizers':{a:opts[a].state_dict() for a in PROBE_ARMS_V1}}
            _atomic_torch(cp,local); tmp=remote.with_suffix('.pt.tmp'); shutil.copy2(local,tmp); os.replace(tmp,remote); _atomic_json({'schema':'RealSaS.HRepP0CheckpointPointer.v1','completed_steps':done,'checkpoint_sha256':sha256_file(remote),'execution_source_sha':source_sha},mirror/'LATEST.json')
            print('H_REP_P0_CHECKPOINT',done,sha256_file(remote)[:16],flush=True)
    result={'schema':SCHEMA,'status':'COMPLETED_H_REP_P0_EQUAL_CAPACITY_PROBES__AWAITING_INDEPENDENT_COURT','execution_source_sha':source_sha,'binding_sha256':binding_sha,'preregistration_sha256':prereg_sha,'dataset_receipt_sha256':dataset_sha,'v8_scene_core_sha256':scene_hash,'optimizer_steps_per_arm':TOTAL_STEPS,'arms':list(PROBE_ARMS_V1),'checkpoint_file':local.name,'checkpoint_sha256':sha256_file(local),'elapsed_seconds':time.perf_counter()-started,'mesh_extraction_invoked':False,'tessa_invoked':False,'scientific_pass_claimed':False,'next_action':'RUN_INDEPENDENT_H_REP_P0_HELDOUT_COURT'}
    _atomic_json(result,output/'H_REP_P0_TRAINING_RESULT.json'); return result

def main()->int:
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--repo-root',required=True); p.add_argument('--expected-source-sha',required=True); p.add_argument('--binding',default='canonical/H_REP_P0_INPUT_BINDING_V1.json'); p.add_argument('--preregistration',default='canonical/H_REP_P0_PREREGISTRATION_V1.json'); p.add_argument('--dataset-root',required=True); p.add_argument('--dataset-receipt',required=True); p.add_argument('--prepared-root',required=True); p.add_argument('--prepared-manifest',required=True); p.add_argument('--v8-checkpoint',required=True); p.add_argument('--output-dir',required=True); p.add_argument('--checkpoint-mirror-dir',required=True); p.add_argument('--query-chunk',type=int,default=8192); train(p.parse_args()); return 0
if __name__=='__main__': raise SystemExit(main())
