from __future__ import annotations
import argparse,json
from pathlib import Path
import torch
from model_pv5 import IRISSinglePoseV2PV5
from pv5_depth_objective import p_only_objective
from train_pv5_depth_overfit_v1 import configure_p_only

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True); a=ap.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError('CUDA GPU REQUIRED for production-width P-V5 training preflight')
    torch.manual_seed(20260824); torch.cuda.manual_seed_all(20260824); device=torch.device('cuda'); model=IRISSinglePoseV2PV5().to(device); configure_p_only(model); model.train(); yaw=(torch.arange(8,device=device,dtype=torch.float32)*45.0)[None]; h=torch.tensor([0.58],device=device); images=torch.rand(1,8,4,256,256,device=device); S=512; xy=torch.rand(1,8,S,2,device=device)*1.8-.9; truth=torch.zeros(1,8,S,3,device=device); mask=torch.ones(1,8,S,dtype=torch.bool,device=device)
    with torch.autocast(device_type='cuda',dtype=torch.float16): outputs=model(images,yaw,h)
    parts=p_only_objective(outputs,{'geom_xy':xy,'geom_p':truth,'geom_mask':mask,'yaw_deg':yaw}); parts['total'].backward(); g=model.p_depth_head.weight.grad
    if not torch.isfinite(parts['total']) or g is None or not torch.isfinite(g).all() or float(g.abs().sum())<=0: raise RuntimeError('CUDA preflight nonfinite/zero gradient')
    report={'schema':'RealSaS.IRISSinglePoseV2.PV5DepthOverfitGPUPreflight.v1','status':'PASS','device':torch.cuda.get_device_name(0),'input_resolution':256,'amp_forward':True,'fp32_objective':True,'p_depth_gradient_nonzero':True,'scientific_optimizer_steps':0}; Path(a.out).write_text(json.dumps(report,indent=2,sort_keys=True)+'\n',encoding='utf-8'); print(json.dumps(report,indent=2,sort_keys=True))
if __name__=='__main__': main()
