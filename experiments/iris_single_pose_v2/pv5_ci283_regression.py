from __future__ import annotations
import inspect,json
import torch
from model import IRISV2Config
from model_pv4 import reconstruct_p_from_observable_scale
from model_pv5 import IRISSinglePoseV2PV5,estimate_native_sheet_half_extent

def tiny_cfg(): return IRISV2Config(widths=(8,16,24,32),coarse_dim=16,fine_dim=8,context_hw=8,within_heads=4,cross_heads=4,cross_layers=1)
def make_native(target):
    R=1024; x=torch.zeros(1,8,4,R,R,dtype=torch.float16); span=max(2,min(R,int(round(R/(2*target))))); sy=max(2,int(round(.72*span))); s2=max(2,int(round(.61*span))); cx=cy=R//2
    for v in range(8):
        w=s2 if v in (2,6) else span; x0=cx-w//2; y0=cy-sy//2; x[:,v,:3,y0:y0+sy,x0:x0+w]=1; x[:,v,3,y0:y0+sy,x0:x0+w]=1
    return x
def main():
    yaw=torch.arange(8,dtype=torch.float32)[None]*45.; sig=list(inspect.signature(IRISSinglePoseV2PV5.forward).parameters)
    if sig!=['self','images','yaw_deg','sheet_half_extent']: raise RuntimeError(sig)
    src=inspect.getsource(IRISSinglePoseV2PV5.forward).lower()
    if 'estimate_native_sheet_half_extent' in src or 'estimate_sheet_half_extent_from_alpha' in src or 'camera.json' in src: raise RuntimeError('PV5 forward boundary drift')
    report=[]
    for target in (0.54,0.5570941257476807,0.6172158837318421):
        native=make_native(target); h=estimate_native_sheet_half_extent(native,yaw); used=[]
        for res in (1024,512,256):
            depth=torch.zeros(1,8,1,res//2,res//2); P=reconstruct_p_from_observable_scale(depth,yaw,h); used.append(float(h[0]));
            if P.shape!=(1,8,3,res//2,res//2) or not torch.isfinite(P).all(): raise RuntimeError((target,res,P.shape))
        if not (used[0]==used[1]==used[2]): raise RuntimeError('scale transport drift')
        report.append({'target_fixture':target,'native_h':float(h[0]),'transported_exactly':True}); del native
    m=IRISSinglePoseV2PV5(tiny_cfg()).eval(); images=torch.rand(1,8,4,256,256); h=torch.tensor([.58])
    with torch.no_grad(): out=m(images,yaw,h)
    expected={'P':(1,8,3,128,128),'N':(1,8,3,128,128),'U_geo':(1,8,1,128,128),'Z_coarse':(1,8,16,32,32),'Z_fine':(1,8,8,128,128)}
    for k,s in expected.items():
        if tuple(out[k].shape)!=s: raise RuntimeError((k,out[k].shape,s))
    result={'schema':'RealSaS.IRISSinglePoseV2.PV5CI283RegressionSubset.v1','status':'PASS','camera_half_extent_input':False,'scale_reestimated_after_resize':False,'optimizer_steps':0,'scale_cases':report,'R256_field_shapes':'PASS'}; print(json.dumps(result,indent=2,sort_keys=True))
if __name__=='__main__': main()
