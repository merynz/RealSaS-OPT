from __future__ import annotations
import torch
import torch.nn.functional as F
from model import camera_basis_from_yaw

def sample_p_field(P,xy):
    b,v,c,h,w=P.shape; s=xy.shape[2]
    field=P.reshape(b*v,c,h,w).float(); grid=xy.reshape(b*v,1,s,2).float()
    samp=F.grid_sample(field,grid,mode='bilinear',padding_mode='border',align_corners=False)
    return samp.squeeze(2).permute(0,2,1).reshape(b,v,s,3)

def p_only_objective(outputs,batch,beta=0.01):
    pred=sample_p_field(outputs['P'],batch['geom_xy']); truth=batch['geom_p'].float(); mask=batch['geom_mask'].bool(); b,v,s,_=pred.shape
    _,_,forward=camera_basis_from_yaw(batch['yaw_deg'],batch=b,views=v,dtype=torch.float32); f=forward[:,:,None,:]
    d_pred=(pred.float()*f).sum(-1); d_truth=(truth*f).sum(-1)
    if not mask.any(): raise RuntimeError('empty P supervision mask')
    depth_loss=F.smooth_l1_loss(d_pred[mask],d_truth[mask],beta=float(beta),reduction='mean')
    p_err=torch.linalg.norm(pred.float()-truth,dim=-1)[mask]
    return {'total':depth_loss,'depth_smooth_l1':depth_loss,'p_mean':p_err.mean(),'p_p95':torch.quantile(p_err,0.95)}
