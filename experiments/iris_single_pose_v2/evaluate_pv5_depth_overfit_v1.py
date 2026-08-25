from __future__ import annotations
import math
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset_pv5_depth_overfit import PV5DepthOverfitDataset
from pv5_depth_objective import sample_p_field

def q(a,x): return float(np.quantile(np.asarray(a,np.float64),x)) if len(a) else float('nan')

def evaluate(model,cache_manifest,device):
    ds=PV5DepthOverfitDataset(cache_manifest); dl=DataLoader(ds,batch_size=1,shuffle=False,num_workers=0); model.eval(); cells=[]; allerr=[]
    with torch.no_grad():
        for batch in dl:
            images=batch['images'].to(device); yaw=batch['yaw_deg'].to(device); h=batch['sheet_half_extent'].to(device); xy=batch['geom_xy'].to(device); truth=batch['geom_p'].to(device).float(); mask=batch['geom_mask'].to(device).bool()
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=(device.type=='cuda')): out=model(images,yaw,h)
            pred=sample_p_field(out['P'],xy); err=torch.linalg.norm(pred-truth,dim=-1)[mask].detach().cpu().numpy().astype(np.float64); allerr.extend(err.tolist())
            cell={'asset_id':batch['asset_id'][0],'style':batch['style'][0],'samples':int(len(err)),'P_p50':q(err,.5),'P_p90':q(err,.9),'P_p95':q(err,.95),'P_max':float(np.max(err))}; cells.append(cell)
    worst=max(cells,key=lambda c:c['P_p95']); best=min(cells,key=lambda c:c['P_p95'])
    return {'schema':'RealSaS.IRISSinglePoseV2.PV5DepthOverfitEval.v1','cells':cells,'cell_count':len(cells),'global':{'P_p50':q(allerr,.5),'P_p90':q(allerr,.9),'P_p95':q(allerr,.95),'P_max':float(np.max(allerr))},'worst_cell_P_p95':float(worst['P_p95']),'worst_cell':{'asset_id':worst['asset_id'],'style':worst['style']},'best_cell':{'asset_id':best['asset_id'],'style':best['style']},'all_cells_pass_0p005':all(math.isfinite(c['P_p95']) and c['P_p95']<=.005 for c in cells)}
