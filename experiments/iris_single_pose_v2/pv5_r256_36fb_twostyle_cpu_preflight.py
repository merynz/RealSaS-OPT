from __future__ import annotations
import argparse, inspect, json, py_compile
from pathlib import Path
import torch
from model_pv5_r256 import IRISSinglePoseV2PV5R256
from pv5_depth_objective import p_only_objective
from train_pv5_r256_36fb_twostyle_v1 import configure_p_only

AID='asset_36fb02305846592b1ecdf3d4'
STYLES=('cel_clean','ink_cel')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True); a=ap.parse_args()
    here=Path(__file__).resolve().parent; checks={}
    for p in sorted(here.glob('*.py')): py_compile.compile(str(p),doraise=True)
    checks['py_compile']='PASS'
    mem=json.load(open(here/'P_V5_R256_36FB_TWO_STYLE_MEMBERSHIP_V1.json',encoding='utf-8')); rows=mem['records']
    if mem.get('asset_count')!=1 or mem.get('asset_style_cells')!=2 or len(rows)!=2: raise RuntimeError('membership count drift')
    if {r.get('asset_id') for r in rows}!={AID} or {r.get('style') for r in rows}!=set(STYLES): raise RuntimeError('membership identity drift')
    if any(r.get('split')!='FIT' for r in rows): raise RuntimeError('non-FIT membership')
    checks['membership_exact_two_cells']='PASS'
    prereg=(here/'P_V5_R256_36FB_TWO_STYLE_PREREG_20260825.md').read_text(encoding='utf-8')
    for token in ('lr `3e-4`','2048 optimizer steps','lr `3e-5`','TAIL `64,128,256,512,1024,1536,2048`','both style cells'):
        if token not in prereg: raise RuntimeError(f'prereg token missing: {token}')
    checks['adequate_budget_prereg_frozen']='PASS'
    sig=str(inspect.signature(IRISSinglePoseV2PV5R256.forward))
    if sig!='(self, images, yaw_deg, sheet_half_extent)': raise RuntimeError(f'forward signature drift: {sig}')
    torch.manual_seed(20260825); model=IRISSinglePoseV2PV5R256(); configure_p_only(model)
    images=torch.rand(2,8,4,32,32); yaw=(torch.arange(8,dtype=torch.float32)*45.0)[None].repeat(2,1); h=torch.tensor([.58,.61])
    out=model(images,yaw,h); S=64; xy=torch.rand(2,8,S,2)*1.8-.9; truth=torch.zeros(2,8,S,3); mask=torch.ones(2,8,S,dtype=torch.bool)
    parts=p_only_objective(out,{'geom_xy':xy,'geom_p':truth,'geom_mask':mask,'yaw_deg':yaw}); parts['total'].backward()
    if float(model.p_depth_head.weight.grad.abs().sum())<=0: raise RuntimeError('depth gradient missing')
    if any(p.grad is not None for n,p in model.named_parameters() if n.startswith(('n_head.','u_geo_head.','coarse_head.','fine_head.'))): raise RuntimeError('frozen head gradient')
    checks['b2_fullres_p_only_forward_backward']='PASS'; checks['scientific_optimizer_steps']=0
    report={'schema':'RealSaS.IRISSinglePoseV2.PV5R256HardAssetTwoStyleCPUPreflight.v1','status':'PASS','checks':checks}
    Path(a.out).write_text(json.dumps(report,indent=2,sort_keys=True)+'\n'); print(json.dumps(report,indent=2))

if __name__=='__main__': main()
