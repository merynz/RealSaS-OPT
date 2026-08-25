from __future__ import annotations
import math, numpy as np, torch
from torch.utils.data import DataLoader
from dataset_pv5_r256_8x2 import PV5R256EightByTwoDataset
from pv5_depth_objective import sample_p_field

def q(a,x):
    a=np.asarray(a,np.float64); return float(np.quantile(a,x)) if a.size else float('nan')
def metrics(err): return {'samples':int(len(err)),'P_p50':q(err,.5),'P_p90':q(err,.9),'P_p95':q(err,.95),'P_mean':float(np.mean(err)),'P_max':float(np.max(err))}
def evaluate(model,cache_manifest,device,eval_batch_cells=4):
    ds=PV5R256EightByTwoDataset(cache_manifest); dl=DataLoader(ds,batch_size=eval_batch_cells,shuffle=False,num_workers=0,pin_memory=True)
    model.eval(); rows=[]; all_err=[]
    with torch.no_grad():
      for batch in dl:
        images=batch['images'].to(device,non_blocking=True); yaw=batch['yaw_deg'].to(device,non_blocking=True); h=batch['sheet_half_extent'].to(device,non_blocking=True); xy=batch['geom_xy'].to(device,non_blocking=True); truth=batch['geom_p'].to(device,non_blocking=True).float(); mask=batch['geom_mask'].to(device,non_blocking=True).bool()
        with torch.autocast(device_type='cuda',dtype=torch.float16): out=model(images,yaw,h)
        if out['P'].shape[-2:]!=(256,256): raise RuntimeError('R256 evaluator field drift')
        pred=sample_p_field(out['P'],xy)
        for i in range(len(batch['asset_id'])):
            err=torch.linalg.norm(pred[i].float()-truth[i],dim=-1)[mask[i]].detach().cpu().numpy().astype(np.float64); all_err.append(err); m=metrics(err); m.update(asset_id=batch['asset_id'][i],style=batch['style'][i],pass_0p005=bool(math.isfinite(m['P_p95']) and m['P_p95']<=.005)); rows.append(m)
    agg=metrics(np.concatenate(all_err)); worst=max(float(x['P_p95']) for x in rows); passed=all(x['pass_0p005'] for x in rows)
    return {'schema':'RealSaS.IRISSinglePoseV2.PV5R256EightByTwoEval.v1','per_cell':rows,'aggregate':agg,'worst_cell_P_p95':worst,'all_cells_pass_0p005':bool(passed),'pass_count':sum(int(x['pass_0p005']) for x in rows),'cell_count':16,'output_field_hw':256}
