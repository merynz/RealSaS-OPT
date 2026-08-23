from __future__ import annotations
import torch
import torch.nn.functional as F


def sample_field(field, grid):
    """field [B,V,C,H,W], grid [B,V,K,2] -> [B,V,K,C]."""
    B,V,C,H,W=field.shape; K=grid.shape[2]
    f=field.reshape(B*V,C,H,W); g=grid.reshape(B*V,K,1,2)
    out=F.grid_sample(f,g,mode='bilinear',padding_mode='zeros',align_corners=False)
    return out[...,0].permute(0,2,1).reshape(B,V,K,C)


def sample_pair_field(field, view_index, grid):
    """field [B,V,C,H,W], view_index [B,M], grid [B,M,2] -> [B,M,C]."""
    B,V,C,H,W=field.shape
    out=[]
    for b in range(B):
        f=field[b,view_index[b]]
        g=grid[b,:,None,None,:]
        s=F.grid_sample(f,g,mode='bilinear',padding_mode='zeros',align_corners=False)
        out.append(s[:,:,0,0])
    return torch.stack(out,0)


def smooth_l1_vec(pred,target,beta=.01):
    return F.smooth_l1_loss(pred,target,reduction='none',beta=beta).mean(-1)


def geometry_loss(outputs,batch):
    grid=batch['geom_xy']; gt_p=batch['geom_p']; gt_n=batch['geom_n']; mask=batch['geom_mask'].float()
    pr_p=sample_field(outputs['P'],grid); pr_n=sample_field(outputs['N'],grid)
    p_elem=smooth_l1_vec(pr_p,gt_p,.01)
    n_elem=1.0-(pr_n*gt_n).sum(-1).clamp(-1,1)
    den=mask.sum().clamp_min(1.0)
    lp=(p_elem*mask).sum()/den; ln=(n_elem*mask).sum()/den
    s=sample_field(outputs['log_sigma'],grid)[...,0]
    euclid=torch.linalg.norm(pr_p.detach()-gt_p,dim=-1)
    lu=((torch.exp(-s)*euclid+s)*mask).sum()/den
    return {'P':lp,'N':ln,'U':lu,'p_euclid':(euclid*mask).sum().detach()/den,
            'n_cos':(n_elem*mask).sum().detach()/den}


def masked_multi_positive_nce(zs,zt,p,mask,temperature=.07,same_radius=.003,far_radius=.02):
    losses=[]
    B,M,D=zs.shape
    for b in range(B):
        m=mask[b].bool()
        if m.sum()<2: continue
        a=F.normalize(zs[b,m],dim=-1); c=F.normalize(zt[b,m],dim=-1); pp=p[b,m].float()
        logits=(a@c.T)/temperature
        dist=torch.cdist(pp,pp)
        pos=dist<=same_radius
        valid=(dist>=far_radius)|pos
        logits=logits.masked_fill(~valid,-1e4)
        losses.append((torch.logsumexp(logits,1)-torch.logsumexp(logits.masked_fill(~pos,-1e4),1)).mean())
    return torch.stack(losses).mean() if losses else zs.sum()*0.0


def correspondence_loss(outputs,batch):
    sv=batch['corr_src_view'].long(); tv=batch['corr_tgt_view'].long()
    sx=batch['corr_src_xy']; tx=batch['corr_tgt_xy']; p=batch['corr_p']; mask=batch['corr_mask'].bool()
    zfs=sample_pair_field(outputs['Z_fine'],sv,sx); zft=sample_pair_field(outputs['Z_fine'],tv,tx)
    zcs=sample_pair_field(outputs['Z_coarse'],sv,sx); zct=sample_pair_field(outputs['Z_coarse'],tv,tx)
    lzc=masked_multi_positive_nce(zcs,zct,p,mask,.08,.003,.025)
    lzf=masked_multi_positive_nce(zfs,zft,p,mask,.05,.0025,.015)
    ps=sample_pair_field(outputs['P'],sv,sx); pt=sample_pair_field(outputs['P'],tv,tx)
    if mask.any():
        lcons=F.smooth_l1_loss(ps[mask],pt[mask],beta=.01)
    else: lcons=ps.sum()*0.0
    return {'Z_coarse':lzc,'Z_fine':lzf,'P_consistency':lcons}


def total_loss(outputs,batch,epoch,warmup_epochs=4):
    g=geometry_loss(outputs,batch)
    total=g['P']+.25*g['N']+.05*g['U']; parts={**g}
    if epoch>=warmup_epochs and batch['corr_mask'].any():
        c=correspondence_loss(outputs,batch)
        total=total+.10*c['Z_coarse']+.05*c['Z_fine']+.20*c['P_consistency']; parts.update(c)
    parts['total']=total
    return parts
