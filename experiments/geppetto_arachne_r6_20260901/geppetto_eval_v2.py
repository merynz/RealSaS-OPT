from __future__ import annotations
import numpy as np
import torch
try:
    from scipy.optimize import linear_sum_assignment
except ImportError as exc:
    raise ImportError("geppetto_eval_v2 requires scipy") from exc


def geppetto_metrics_v2(predicted_positions:torch.Tensor,target_positions:np.ndarray)->dict[str,float]:
    p=predicted_positions.detach().float().cpu(); t=torch.as_tensor(target_positions,dtype=p.dtype); J=len(t)
    if J==0: return {"target_count":0,"predicted_count":int(len(p)),"count_abs_error":float(len(p)),"matched_p95":float("nan")}
    if len(p)==0: return {"target_count":J,"predicted_count":0,"count_abs_error":float(J),"matched_p95":float("inf")}
    cost=torch.cdist(p,t); qi,ti=linear_sum_assignment(cost.numpy()); e=cost[qi,ti]; return {"target_count":J,"predicted_count":int(len(p)),"count_abs_error":float(abs(len(p)-J)),"matched_mae":float(e.mean()),"matched_p95":float(torch.quantile(e,0.95))}
