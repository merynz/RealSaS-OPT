from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from model_pv5_r256 import IRISSinglePoseV2PV5R256
from pv5_depth_objective import p_only_objective
from train_pv5_r256_onecell_v1 import configure_p_only


def _capture_shape(store):
    def hook(_module, _inputs, output):
        store['shape'] = tuple(output.shape)
        return None
    return hook


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True); a=ap.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError('CUDA GPU REQUIRED for production R256 preflight')
    torch.manual_seed(20260824); torch.cuda.manual_seed_all(20260824)
    device=torch.device('cuda'); torch.cuda.reset_peak_memory_stats()
    model=IRISSinglePoseV2PV5R256().to(device); configure_p_only(model); model.train()
    yaw=(torch.arange(8,device=device,dtype=torch.float32)*45.0)[None]
    h=torch.tensor([0.58],device=device)
    images=torch.rand(1,8,4,256,256,device=device)
    S=512; xy=torch.rand(1,8,S,2,device=device)*1.8-.9
    truth=torch.zeros(1,8,S,3,device=device); mask=torch.ones(1,8,S,dtype=torch.bool,device=device)
    captured={}
    hook=model.p_s1_fuse.register_forward_hook(_capture_shape(captured))
    with torch.autocast(device_type='cuda',dtype=torch.float16):
        outputs=model(images,yaw,h)
    hook.remove()
    if tuple(outputs['P'].shape)!=(1,8,3,256,256):
        raise RuntimeError(f'not R256 P: {tuple(outputs["P"].shape)}')
    if captured.get('shape') is None or captured['shape'][-2:]!=(256,256):
        raise RuntimeError(f'fullres fuse not R256: {captured}')
    parts=p_only_objective(outputs,{'geom_xy':xy,'geom_p':truth,'geom_mask':mask,'yaw_deg':yaw})
    parts['total'].backward()
    g_head=model.p_depth_head.weight.grad
    g_stem=next(p.grad for n,p in model.named_parameters() if n=='p_s1_stem.0.conv.weight')
    if g_head is None or g_stem is None or not torch.isfinite(g_head).all() or not torch.isfinite(g_stem).all():
        raise RuntimeError('GPU R256 gradient missing/nonfinite')
    if float(g_head.abs().sum())<=0 or float(g_stem.abs().sum())<=0:
        raise RuntimeError('GPU R256 branch zero gradient')
    report={'schema':'RealSaS.IRISSinglePoseV2.PV5R256OneCellGPUPreflight.v1','status':'PASS',
            'device':torch.cuda.get_device_name(0),'input_resolution':256,'output_field_hw':256,
            'fullres_fuse_shape':list(captured['shape']),'amp_forward':True,'fp32_objective':True,
            'p_depth_gradient_nonzero':True,'fullres_image_stem_gradient_nonzero':True,
            'cuda_peak_allocated_bytes':int(torch.cuda.max_memory_allocated()),
            'scientific_optimizer_steps':0}
    Path(a.out).write_text(json.dumps(report,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2,sort_keys=True))

if __name__=='__main__': main()
