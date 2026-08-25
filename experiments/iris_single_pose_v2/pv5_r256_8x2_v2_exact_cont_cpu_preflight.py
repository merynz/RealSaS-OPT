from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import torch
from model_pv5_r256 import IRISSinglePoseV2PV5R256
from train_pv5_r256_8x2_v2_exact_cont_v1 import configure_p_only, LR, BETAS, WD

def sha(p):
    h=hashlib.sha256();
    with open(p,'rb') as f:
        for x in iter(lambda:f.read(8<<20),b''): h.update(x)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--parent-checkpoint',required=True); ap.add_argument('--out',required=True); a=ap.parse_args()
    cp=torch.load(a.parent_checkpoint,map_location='cpu',weights_only=False)
    if cp.get('schema')!='RealSaS.IRISSinglePoseV2.PV5R256EightByTwoCheckpoint.v2' or cp.get('label')!='TAIL_4096' or cp.get('total_optimizer_steps')!=6144: raise RuntimeError('parent checkpoint identity drift')
    m=IRISSinglePoseV2PV5R256(); tr=configure_p_only(m); m.load_state_dict(cp['model_state'],strict=True)
    opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=LR,betas=BETAS,weight_decay=WD); opt.load_state_dict(cp['optimizer_state'])
    if len(opt.state)!=len(tr): raise RuntimeError(f'optimizer state coverage drift {len(opt.state)} != {len(tr)}')
    for g in opt.param_groups:
        if abs(float(g['lr'])-LR)>1e-12 or tuple(g['betas'])!=BETAS or abs(float(g['weight_decay'])-WD)>1e-12: raise RuntimeError('optimizer hyperparameter drift')
    rng=cp.get('rng_state',{}); required={'python_random_state','numpy_random_state','torch_cpu_rng_state','torch_cuda_rng_state_all'}
    if set(rng)!=required: raise RuntimeError(f'RNG state drift {set(rng)}')
    out={'schema':'RealSaS.PV5R256EightByTwoV2ExactContinuationCPUPreflight.v1','status':'PASS','parent_checkpoint_sha256':sha(a.parent_checkpoint),'parent_label':cp['label'],'parent_total_optimizer_steps':cp['total_optimizer_steps'],'trainable_parameter_tensors':len(tr),'optimizer_state_entries':len(opt.state),'scaler_state_present':bool(cp.get('scaler_state')),'rng_state_present':True,'fresh_optimizer_moments':False,'scientific_optimizer_steps':0}
    Path(a.out).write_text(json.dumps(out,indent=2,sort_keys=True)+'\n',encoding='utf-8'); print(json.dumps(out,indent=2,sort_keys=True))
if __name__=='__main__': main()
