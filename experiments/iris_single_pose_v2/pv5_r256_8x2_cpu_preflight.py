from __future__ import annotations
import argparse,json,copy
from pathlib import Path
import torch
from model_pv5_r256 import IRISSinglePoseV2PV5R256
from train_pv5_r256_8x2_v1 import configure_p_only
from pv5_depth_objective import p_only_objective

def make_batch(B=4,H=32,S=64):
    images=torch.rand(B,8,4,H,H)
    yaw=(torch.arange(8,dtype=torch.float32)*45)[None].repeat(B,1)
    h=torch.linspace(.53,.62,B)
    xy=torch.rand(B,8,S,2)*1.8-.9
    truth=torch.randn(B,8,S,3)*.1
    mask=torch.ones(B,8,S,dtype=torch.bool)
    return {'images':images,'yaw_deg':yaw,'sheet_half_extent':h,'geom_xy':xy,'geom_p':truth,'geom_mask':mask}

def loss(model,b):
    o=model(b['images'],b['yaw_deg'],b['sheet_half_extent'])
    return p_only_objective(o,{'geom_xy':b['geom_xy'],'geom_p':b['geom_p'],'geom_mask':b['geom_mask'],'yaw_deg':b['yaw_deg']})['total']

def sl(b,s,e): return {k:v[s:e] for k,v in b.items()}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True); a=ap.parse_args(); torch.manual_seed(20260825)
    base=IRISSinglePoseV2PV5R256(); configure_p_only(base); base.train(); batch=make_batch()
    o=base(batch['images'][:2],batch['yaw_deg'][:2],batch['sheet_half_extent'][:2])
    if tuple(o['P'].shape)!=(2,8,3,32,32): raise RuntimeError('full-R CPU shape drift')
    bad_bn=[n for n,x in base.named_modules() if isinstance(x,torch.nn.modules.batchnorm._BatchNorm)]
    drops=[(n,float(x.p)) for n,x in base.named_modules() if isinstance(x,torch.nn.Dropout) and float(x.p)!=0.0]
    if bad_bn or drops: raise RuntimeError(f'batch-stat/stochastic module found: BN={bad_bn} dropout={drops}')
    full=copy.deepcopy(base); acc=copy.deepcopy(base); configure_p_only(full); configure_p_only(acc)
    full.zero_grad(set_to_none=True); loss(full,batch).backward()
    acc.zero_grad(set_to_none=True)
    (loss(acc,sl(batch,0,2))/2).backward(); (loss(acc,sl(batch,2,4))/2).backward()
    probes=['p_depth_head.weight','p_s1_stem.0.conv.weight','encoder.s2.0.conv.weight']
    max_abs=0.0; max_rel=0.0; rows=[]
    fd=dict(full.named_parameters()); ad=dict(acc.named_parameters())
    for n in probes:
        g1=fd[n].grad; g2=ad[n].grad
        if g1 is None or g2 is None: raise RuntimeError(f'missing gradient {n}')
        d=(g1-g2).abs(); ma=float(d.max()); den=float(g1.abs().max())+1e-12; rel=ma/den; max_abs=max(max_abs,ma); max_rel=max(max_rel,rel); rows.append({'param':n,'max_abs_diff':ma,'relative_to_full_max':rel})
    if max_abs>2e-5 and max_rel>2e-4: raise RuntimeError(f'accumulation equivalence failed: abs={max_abs} rel={max_rel}')
    rep={'schema':'RealSaS.IRISSinglePoseV2.PV5R256EightByTwoCPUPreflight.v1','status':'PASS','checks':{'full_res_output':True,'no_batchnorm':True,'nonzero_dropout_absent':True,'gradient_accumulation_equivalence':'PASS','equivalence_probe_rows':rows,'equivalence_max_abs_diff':max_abs,'equivalence_max_relative':max_rel,'production_microbatch_cells':8,'production_accumulation_steps':2,'scientific_optimizer_steps':0},'trainable_count':len([n for n,p in base.named_parameters() if p.requires_grad])}
    Path(a.out).write_text(json.dumps(rep,indent=2,sort_keys=True)+'\n'); print(json.dumps(rep,indent=2))
if __name__=='__main__': main()
