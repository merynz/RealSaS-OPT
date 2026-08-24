from __future__ import annotations
import math, numpy as np, torch
from torch.utils.data import DataLoader
from dataset_pv5_r256_onecell import PV5R256OneCellDataset
from pv5_depth_objective import sample_p_field

def q(a,x): return float(np.quantile(np.asarray(a,np.float64),x)) if len(a) else float('nan')

def evaluate(model,cache_manifest,device):
    ds=PV5R256OneCellDataset(cache_manifest); dl=DataLoader(ds,batch_size=1,shuffle=False,num_workers=0)
    model.eval()
    with torch.no_grad():
        batch=next(iter(dl))
        images=batch['images'].to(device); yaw=batch['yaw_deg'].to(device)
        h=batch['sheet_half_extent'].to(device); xy=batch['geom_xy'].to(device)
        truth=batch['geom_p'].to(device).float(); mask=batch['geom_mask'].to(device).bool()
        with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=(device.type=='cuda')):
            out=model(images,yaw,h)
        if tuple(out['P'].shape)!=(1,8,3,256,256):
            raise RuntimeError(f'R256 evaluator shape drift: {tuple(out["P"].shape)}')
        pred=sample_p_field(out['P'],xy)
        err=torch.linalg.norm(pred-truth,dim=-1)[mask].detach().cpu().numpy().astype(np.float64)
    p95=q(err,.95)
    return {'schema':'RealSaS.IRISSinglePoseV2.PV5R256OneCellEval.v1',
            'asset_id':batch['asset_id'][0],'style':batch['style'][0],
            'samples':int(len(err)),'P_p50':q(err,.5),'P_p90':q(err,.9),'P_p95':p95,
            'P_mean':float(np.mean(err)),'P_max':float(np.max(err)),
            'pass_0p005':bool(math.isfinite(p95) and p95<=.005),
            'output_field_hw':256}
