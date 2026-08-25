from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from model_pv5_r256 import IRISSinglePoseV2PV5R256
from pv5_depth_objective import p_only_objective
from train_pv5_r256_36fb_twostyle_v1 import configure_p_only

def _capture(store):
    def hook(_m,_i,o):
        store['shape']=tuple(o.shape)
        return None
    return hook

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True); a=ap.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError('CUDA GPU REQUIRED for R256 two-style preflight')
    torch.manual_seed(20260825); torch.cuda.manual_seed_all(20260825)
    device=torch.device('cuda'); torch.cuda.reset_peak_memory_stats()
    model=IRISSinglePoseV2PV5R256().to(device); configure_p_only(model); model.train()
    images=torch.rand(2,8,4,256,256,device=device)
    yaw=(torch.arange(8,device=device,dtype=torch.float32)*45.0)[None].repeat(2,1)
    h=torch.tensor([0.58,0.61],device=device)
    S=256; xy=torch.rand(2,8,S,2,device=device)*1.8-.9
    truth=torch.zeros(2,8,S,3,device=device); mask=torch.ones(2,8,S,dtype=torch.bool,device=device)
    captured={}; hook=model.p_s1_fuse.register_forward_hook(_capture(captured))
    with torch.autocast(device_type='cuda',dtype=torch.float16):
        out=model(images,yaw,h)
    hook.remove()
    if tuple(out['P'].shape)!=(2,8,3,256,256):
        raise RuntimeError(f'not B2 R256 P: {tuple(out["P"].shape)}')
    if captured.get('shape') is None or tuple(captured['shape'])!=(16,48,256,256):
        raise RuntimeError(f'fullres B2 fuse drift: {captured}')
    parts=p_only_objective(out,{'geom_xy':xy,'geom_p':truth,'geom_mask':mask,'yaw_deg':yaw})
    parts['total'].backward()
    g_head=model.p_depth_head.weight.grad
    g_stem=next(p.grad for n,p in model.named_parameters() if n=='p_s1_stem.0.conv.weight')
    if g_head is None or g_stem is None or not torch.isfinite(g_head).all() or not torch.isfinite(g_stem).all():
        raise RuntimeError('B2 R256 gradient missing/nonfinite')
    report={'schema':'RealSaS.IRISSinglePoseV2.PV5R256HardAssetTwoStyleGPUPreflight.v1',
            'status':'PASS','device':torch.cuda.get_device_name(0),
            'batch_style_cells':2,'views_per_cell':8,'input_resolution':256,'output_field_hw':256,
            'fullres_fuse_shape':list(captured['shape']),
            'p_depth_gradient_nonzero':bool(float(g_head.abs().sum())>0),
            'fullres_image_stem_gradient_nonzero':bool(float(g_stem.abs().sum())>0),
            'cuda_peak_allocated_bytes':int(torch.cuda.max_memory_allocated()),
            'scientific_optimizer_steps':0}
    Path(a.out).write_text(json.dumps(report,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2,sort_keys=True))

if __name__=='__main__': main()
