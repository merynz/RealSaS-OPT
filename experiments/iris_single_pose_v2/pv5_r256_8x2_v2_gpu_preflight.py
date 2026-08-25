from __future__ import annotations
import argparse, json, torch
from pathlib import Path
from model_pv5_r256 import IRISSinglePoseV2PV5R256
from train_pv5_r256_8x2_v2 import configure_p_only, checkpoint_payload
from pv5_depth_objective import p_only_objective
MIN_GIB=35.0

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True); a=ap.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError('high-memory CUDA GPU required')
    prop=torch.cuda.get_device_properties(0); gib=prop.total_memory/(1024**3)
    if gib<MIN_GIB: raise RuntimeError(f'8x2 V2 requires >= {MIN_GIB} GiB GPU; got {prop.name} {gib:.2f} GiB')
    torch.manual_seed(20260825); torch.cuda.manual_seed_all(20260825); device=torch.device('cuda'); torch.cuda.reset_peak_memory_stats()
    m=IRISSinglePoseV2PV5R256().to(device); configure_p_only(m); m.train()
    opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=3e-5,betas=(.9,.95),weight_decay=0.)
    scaler=torch.amp.GradScaler('cuda',enabled=True)
    for p in m.parameters():
        if p.requires_grad: p.grad=None
    B=8; S=256
    for _ in range(2):
        images=torch.rand(B,8,4,256,256,device=device); yaw=(torch.arange(8,device=device,dtype=torch.float32)*45)[None].repeat(B,1); h=torch.linspace(.52,.65,B,device=device)
        xy=torch.rand(B,8,S,2,device=device)*1.8-.9; truth=torch.zeros(B,8,S,3,device=device); mask=torch.ones(B,8,S,dtype=torch.bool,device=device)
        with torch.autocast(device_type='cuda',dtype=torch.float16): out=m(images,yaw,h)
        if tuple(out['P'].shape)!=(B,8,3,256,256): raise RuntimeError('microbatch R256 shape drift')
        parts=p_only_objective(out,{'geom_xy':xy,'geom_p':truth,'geom_mask':mask,'yaw_deg':yaw}); scaler.scale(parts['total']/2).backward()
        del images,yaw,h,xy,truth,mask,out,parts
    gh=m.p_depth_head.weight.grad; gs=m.p_s1_stem[0].conv.weight.grad
    if gh is None or gs is None or float(gh.abs().sum())<=0 or float(gs.abs().sum())<=0: raise RuntimeError('8-cell accumulated gradient missing')
    payload=checkpoint_payload(m,opt,scaler,'GPU_SCHEMA_TEST',0,'TAIL',0)
    if not {'optimizer_state','scaler_state','rng_state'}.issubset(payload): raise RuntimeError('checkpoint authority missing')
    rep={'schema':'RealSaS.IRISSinglePoseV2.PV5R256EightByTwoGPUPreflight.v2','status':'PASS','device':prop.name,'total_memory_gib':gib,'microbatch_cells':8,'views_per_microbatch':64,'accumulation_steps':2,'accumulation_microbatches_executed':2,'effective_cells_per_optimizer_step':16,'output_field_hw':256,'cuda_peak_allocated_bytes':int(torch.cuda.max_memory_allocated()),'p_depth_gradient_nonzero':True,'fullres_image_stem_gradient_nonzero':True,'resumable_checkpoint_schema':'PASS','scientific_optimizer_steps':0}
    Path(a.out).write_text(json.dumps(rep,indent=2,sort_keys=True)+'\n'); print(json.dumps(rep,indent=2))
if __name__=='__main__': main()
