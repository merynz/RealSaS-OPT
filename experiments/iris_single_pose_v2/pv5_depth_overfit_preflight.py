from __future__ import annotations
import argparse,inspect,json,py_compile,hashlib
from pathlib import Path
import torch
from model_pv5 import IRISSinglePoseV2PV5,estimate_native_sheet_half_extent,validate_observable_sheet_half_extent
from pv5_depth_objective import p_only_objective
from train_pv5_depth_overfit_v1 import configure_p_only,ALLOWED_TRAINABLE_PREFIXES

EXPECTED_UPSTREAM_BLOBS={'coords.py':'8619a740da0b1d8b8f5e15266691b4c2bf82c1ca','model.py':'bdcac3990f981eb1857cdbbb3975c54e29d62c01','model_pv4.py':'56a052af89690ccdcfcfec4b3913c466536fec14','model_pv5.py':'d133caaacc14b1d64d4c55a07e028b2a9da2665d','geometry.py':'4b05c5289cfc0fbb73fcede21ff373e572af421e'}
def git_blob_sha(path):
    data=Path(path).read_bytes(); return hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
EXPECTED_IDS=['asset_551ea351b43a1787d0f55536','asset_36fb02305846592b1ecdf3d4','asset_0679fdef64f19a4832a6d521','asset_76313e4bd82b82fcd1659c70','asset_6f086a5b1a66378ffe04d7e4','asset_425122d500ecf5767404f9c0','asset_5a19f8c5254be7bf30c504f5','asset_f8a40d6c5d815fe79c8b5e42']

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',default=''); a=ap.parse_args(); here=Path(__file__).resolve().parent; checks={}
    for p in sorted(here.glob('*.py')): py_compile.compile(str(p),doraise=True)
    checks['py_compile']='PASS'
    for name,expected in EXPECTED_UPSTREAM_BLOBS.items():
        got=git_blob_sha(here/name)
        if got!=expected: raise RuntimeError(f'upstream source byte drift {name}: {got} != {expected}')
    checks['ci283_upstream_source_byte_identity']='PASS'
    mem=json.load(open(here/'P_V5_DEPTH_OVERFIT_MEMBERSHIP_V1.json',encoding='utf-8')); ids=[r['asset_id'] for r in mem['records']]
    if ids!=EXPECTED_IDS or mem.get('asset_count')!=8 or mem.get('tune_assets')!=0 or mem.get('sealed_assets')!=0 or any(r.get('split')!='FIT' for r in mem['records']): raise RuntimeError('membership drift')
    checks['membership_exact_frozen_fit_train']='PASS'
    for name in ('stage_pv5_depth_overfit_v1.py','prepare_pv5_depth_cache_v1.py','dataset_pv5_depth_overfit.py','train_pv5_depth_overfit_v1.py','run_pv5_depth_overfit_v1.py'):
        text=(here/name).read_text(encoding='utf-8')
        if 'camera.json' in text: raise RuntimeError(f'forbidden camera.json dependency in {name}')
        if '--tune' in text.lower() or '--dev' in text.lower() or '--cal' in text.lower() or '--external' in text.lower(): raise RuntimeError(f'forbidden sealed/TUNE CLI in {name}')
    checks['metadata_firewall']='PASS'
    sig=str(inspect.signature(IRISSinglePoseV2PV5.forward))
    if sig!='(self, images, yaw_deg, sheet_half_extent)': raise RuntimeError(f'PV5 forward signature drift: {sig}')
    checks['pv5_forward_signature']='PASS'; yaw=torch.arange(8,dtype=torch.float32)*45.0
    try:
        estimate_native_sheet_half_extent(torch.zeros(1,8,4,512,512),yaw); raise RuntimeError('native helper accepted 512')
    except ValueError: pass
    checks['native_512_rejected']='PASS'
    native=torch.zeros((1,8,4,1024,1024),dtype=torch.float16); native[:,:,3,192:832,256:768]=1; native[:,2,3,192:832,128:896]=1; h=estimate_native_sheet_half_extent(native,yaw); expected=2.0/3.0
    if abs(float(h.item())-expected)>1e-6 or h.requires_grad: raise RuntimeError(f'native h regression failed {h.item()} vs {expected}')
    h2=validate_observable_sheet_half_extent(h,1,'cpu')
    if h2.requires_grad or not torch.equal(h,h2): raise RuntimeError('h transport/detach drift')
    checks['native_scale_regression']='PASS'; del native
    torch.manual_seed(20260824); model=IRISSinglePoseV2PV5(); trainable=configure_p_only(model)
    if not trainable or any(not any(n.startswith(p) for p in ALLOWED_TRAINABLE_PREFIXES) for n in trainable): raise RuntimeError('trainable policy drift')
    images=torch.rand(1,8,4,32,32); hh=torch.tensor([0.58]); out=model(images,yaw,hh)
    if out['P'].shape!=(1,8,3,16,16) or out['Z_coarse'].shape[-2:]!=(4,4): raise RuntimeError('field shape drift')
    S=64; xy=torch.rand(1,8,S,2)*1.8-.9; truth=torch.zeros(1,8,S,3); mask=torch.ones(1,8,S,dtype=torch.bool); parts=p_only_objective(out,{'geom_xy':xy,'geom_p':truth,'geom_mask':mask,'yaw_deg':yaw[None]}); parts['total'].backward(); g=model.p_depth_head.weight.grad
    if not torch.isfinite(parts['total']) or g is None or not torch.isfinite(g).all() or float(g.abs().sum())<=0: raise RuntimeError('P-only backward/gradient failed')
    if any(p.grad is not None for n,p in model.named_parameters() if n.startswith(('n_head.','u_geo_head.','coarse_head.','fine_head.'))): raise RuntimeError('frozen head received gradient')
    checks['p_only_forward_backward']='PASS'; checks['scientific_optimizer_steps']=0
    report={'schema':'RealSaS.IRISSinglePoseV2.PV5DepthOverfitCPUPreflight.v1','status':'PASS','checks':checks,'runtime':'CPU sufficient for this preflight only; training requires CUDA GPU'}
    if a.out: Path(a.out).write_text(json.dumps(report,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2,sort_keys=True))
if __name__=='__main__': main()
