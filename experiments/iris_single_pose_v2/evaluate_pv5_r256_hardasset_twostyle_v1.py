from __future__ import annotations
import math, numpy as np, torch
from torch.utils.data import DataLoader
from dataset_pv5_r256_hardasset_twostyle import PV5R256HardAssetTwoStyleDataset
from pv5_depth_objective import sample_p_field

def q(a,x):
    a=np.asarray(a,np.float64)
    return float(np.quantile(a,x)) if a.size else float('nan')

def _metrics(err):
    return {'samples':int(len(err)),
            'P_p50':q(err,.5),'P_p90':q(err,.9),'P_p95':q(err,.95),
            'P_mean':float(np.mean(err)),'P_max':float(np.max(err))}

def evaluate(model,cache_manifest,device):
    ds=PV5R256HardAssetTwoStyleDataset(cache_manifest)
    dl=DataLoader(ds,batch_size=2,shuffle=False,num_workers=0)
    model.eval()
    with torch.no_grad():
        batch=next(iter(dl))
        images=batch['images'].to(device); yaw=batch['yaw_deg'].to(device)
        h=batch['sheet_half_extent'].to(device); xy=batch['geom_xy'].to(device)
        truth=batch['geom_p'].to(device).float(); mask=batch['geom_mask'].to(device).bool()
        with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=(device.type=='cuda')):
            out=model(images,yaw,h)
        if tuple(out['P'].shape)!=(2,8,3,256,256):
            raise RuntimeError(f'R256 two-style evaluator shape drift: {tuple(out["P"].shape)}')
        pred=sample_p_field(out['P'],xy)
        cell_rows=[]; all_err=[]
        for i in range(2):
            err=torch.linalg.norm(pred[i].float()-truth[i],dim=-1)[mask[i]].detach().cpu().numpy().astype(np.float64)
            all_err.append(err)
            m=_metrics(err)
            m.update(asset_id=batch['asset_id'][i],style=batch['style'][i],
                     pass_0p005=bool(math.isfinite(m['P_p95']) and m['P_p95']<=.005))
            cell_rows.append(m)
        agg=_metrics(np.concatenate(all_err))
        worst=max(float(x['P_p95']) for x in cell_rows)
        passed=all(bool(x['pass_0p005']) for x in cell_rows)
    return {'schema':'RealSaS.IRISSinglePoseV2.PV5R256HardAssetTwoStyleEval.v1',
            'per_cell':cell_rows,'aggregate':agg,'worst_cell_P_p95':worst,
            'all_cells_pass_0p005':bool(passed),'output_field_hw':256}
