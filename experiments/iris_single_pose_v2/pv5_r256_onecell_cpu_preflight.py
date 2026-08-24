from __future__ import annotations
import argparse, hashlib, inspect, json, py_compile
from pathlib import Path
import torch
from model_pv5 import estimate_native_sheet_half_extent, validate_observable_sheet_half_extent
from model_pv5_r256 import IRISSinglePoseV2PV5R256
from pv5_depth_objective import p_only_objective
from train_pv5_r256_onecell_v1 import configure_p_only, ALLOWED_TRAINABLE_PREFIXES

EXPECTED_UPSTREAM_BLOBS={'coords.py':'8619a740da0b1d8b8f5e15266691b4c2bf82c1ca','model.py':'bdcac3990f981eb1857cdbbb3975c54e29d62c01','model_pv4.py':'56a052af89690ccdcfcfec4b3913c466536fec14','model_pv5.py':'d133caaacc14b1d64d4c55a07e028b2a9da2665d','geometry.py':'4b05c5289cfc0fbb73fcede21ff373e572af421e'}
AID='asset_76313e4bd82b82fcd1659c70'; STYLE='cel_clean'

def git_blob_sha(path):
    data=Path(path).read_bytes(); return hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',default=''); a=ap.parse_args(); here=Path(__file__).resolve().parent; checks={}
    for p in sorted(here.glob('*.py')): py_compile.compile(str(p),doraise=True)
    checks['py_compile']='PASS'
    for name,expected in EXPECTED_UPSTREAM_BLOBS.items():
        got=git_blob_sha(here/name)
        if got!=expected: raise RuntimeError(f'upstream byte drift {name}: {got} != {expected}')
    checks['upstream_pv5_source_byte_identity']='PASS'
    mem=json.load(open(here/'P_V5_R256_ONE_CELL_MEMBERSHIP_V1.json',encoding='utf-8')); rows=mem.get('records',[])
    if mem.get('asset_count')!=1 or mem.get('asset_style_cells')!=1 or len(rows)!=1: raise RuntimeError('membership count drift')
    r=rows[0]
    if r.get('asset_id')!=AID or r.get('style')!=STYLE or r.get('split')!='FIT' or mem.get('post_result_selection') is not False: raise RuntimeError('frozen membership drift')
    checks['membership_exact_one_cell']='PASS'
    for name in ('stage_pv5_r256_onecell_v1.py','prepare_pv5_r256_onecell_cache_v1.py','dataset_pv5_r256_onecell.py','train_pv5_r256_onecell_v1.py','run_pv5_r256_onecell_v1.py'):
        text=(here/name).read_text(encoding='utf-8')
        if 'camera.json' in text: raise RuntimeError(f'forbidden camera.json dependency in {name}')
        for x in ('--tune','--dev','--cal','--external'):
            if x in text.lower(): raise RuntimeError(f'forbidden split CLI {x} in {name}')
    checks['metadata_firewall']='PASS'
    sig=str(inspect.signature(IRISSinglePoseV2PV5R256.forward))
    if sig!='(self, images, yaw_deg, sheet_half_extent)': raise RuntimeError(f'R256 forward signature drift: {sig}')
    checks['r256_forward_signature']='PASS'
    yaw=torch.arange(8,dtype=torch.float32)*45.0
    try:
        estimate_native_sheet_half_extent(torch.zeros(1,8,4,512,512),yaw); raise RuntimeError('native helper accepted 512')
    except ValueError: pass
    checks['native_512_rejected']='PASS'
    native=torch.zeros((1,8,4,1024,1024),dtype=torch.float16); native[:,:,3,192:832,256:768]=1; native[:,2,3,192:832,128:896]=1
    h=estimate_native_sheet_half_extent(native,yaw)
    if abs(float(h.item())-(2/3))>1e-6 or h.requires_grad: raise RuntimeError('native h regression failed')
    if not torch.equal(h,validate_observable_sheet_half_extent(h,1,'cpu')): raise RuntimeError('h transport drift')
    checks['native_scale_regression']='PASS'; del native
    torch.manual_seed(20260824); model=IRISSinglePoseV2PV5R256(); trainable=configure_p_only(model)
    if not trainable or any(not any(n.startswith(pfx) for pfx in ALLOWED_TRAINABLE_PREFIXES) for n in trainable): raise RuntimeError('trainable policy drift')
    images=torch.rand(1,8,4,32,32); hh=torch.tensor([0.58]); out=model(images,yaw,hh)
    if out['P'].shape!=(1,8,3,32,32): raise RuntimeError(f'P is not full-resolution: {out["P"].shape}')
    if out['_P_FULLRES_FEATURE'].shape[-2:]!=(32,32): raise RuntimeError('full-resolution feature path missing')
    S=64; xy=torch.rand(1,8,S,2)*1.8-.9; truth=torch.zeros(1,8,S,3); mask=torch.ones(1,8,S,dtype=torch.bool)
    parts=p_only_objective(out,{'geom_xy':xy,'geom_p':truth,'geom_mask':mask,'yaw_deg':yaw[None]}); parts['total'].backward()
    g_head=model.p_depth_head.weight.grad; g_stem=next(p.grad for n,p in model.named_parameters() if n=='p_s1_stem.0.conv.weight')
    if g_head is None or g_stem is None or not torch.isfinite(g_head).all() or not torch.isfinite(g_stem).all(): raise RuntimeError('R256 branch gradient missing/nonfinite')
    if float(g_head.abs().sum())<=0 or float(g_stem.abs().sum())<=0: raise RuntimeError('R256 branch zero gradient')
    if any(p.grad is not None for n,p in model.named_parameters() if n.startswith(('n_head.','u_geo_head.','coarse_head.','fine_head.'))): raise RuntimeError('frozen head received gradient')
    checks['true_fullres_output_and_direct_image_branch']='PASS'; checks['p_only_forward_backward']='PASS'; checks['scientific_optimizer_steps']=0
    report={'schema':'RealSaS.IRISSinglePoseV2.PV5R256OneCellCPUPreflight.v1','status':'PASS','checks':checks,'runtime':'CPU sufficient for contract preflight only; scientific training requires CUDA GPU'}
    if a.out: Path(a.out).write_text(json.dumps(report,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2,sort_keys=True))

if __name__=='__main__': main()
