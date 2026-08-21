from __future__ import annotations

from typing import Dict, Tuple
import torch
import torch.nn.functional as F


def _mean_or_zero(xs: list[torch.Tensor], ref: torch.Tensor) -> torch.Tensor:
    return torch.stack(xs).mean() if xs else ref.sum() * 0.0


def same_view_dual_softmax_loss(zA, zB, vA, vB, temperature: float = 0.08):
    losses=[]
    B,V,N,D=zA.shape
    for b in range(B):
        for v in range(V):
            valid=vA[b,v].bool() & vB[b,v].bool()
            if int(valid.sum()) < 2: continue
            a=F.normalize(zA[b,v,valid],dim=-1,eps=1e-6)
            c=F.normalize(zB[b,v,valid],dim=-1,eps=1e-6)
            logits=(a@c.T)/temperature
            lab=torch.arange(logits.shape[0],device=logits.device)
            losses.append(.5*(F.cross_entropy(logits,lab)+F.cross_entropy(logits.T,lab)))
    return _mean_or_zero(losses,zA)


def observation_multi_positive_loss(zA,zB,vA,vB,temperature: float = 0.08):
    losses=[]
    B,V,N,D=zA.shape
    for b in range(B):
        feats=[]; ids=[]
        for z,vis in ((zA[b],vA[b]),(zB[b],vB[b])):
            idx=vis.bool().nonzero(as_tuple=False)
            if idx.numel()==0: continue
            feats.append(z[idx[:,0],idx[:,1]])
            ids.append(idx[:,1])
        if not feats: continue
        f=F.normalize(torch.cat(feats,0),dim=-1,eps=1e-6)
        cid=torch.cat(ids,0)
        if f.shape[0] < 3: continue
        sim=(f@f.T)/temperature
        eye=torch.eye(f.shape[0],device=f.device,dtype=torch.bool)
        pos=cid[:,None].eq(cid[None,:]) & ~eye
        valid_anchor=pos.any(1)
        if not valid_anchor.any(): continue
        sim=sim-sim.max(1,keepdim=True).values.detach()
        e=torch.exp(sim)*(~eye)
        denom=e.sum(1).clamp_min(1e-12)
        numer=(e*pos).sum(1).clamp_min(1e-12)
        losses.append((-torch.log(numer/denom))[valid_anchor].mean())
    return _mean_or_zero(losses,zA)


def hard_negative_margin_loss(zA,zB,vA,vB,margin: float = 0.10):
    losses=[]
    B,V,N,D=zA.shape
    for b in range(B):
        for v in range(V):
            valid=vA[b,v].bool() & vB[b,v].bool()
            if int(valid.sum()) < 2: continue
            a=F.normalize(zA[b,v,valid],dim=-1,eps=1e-6)
            c=F.normalize(zB[b,v,valid],dim=-1,eps=1e-6)
            sim=a@c.T
            pos=sim.diag()
            eye=torch.eye(sim.shape[0],device=sim.device,dtype=torch.bool)
            neg=sim.masked_fill(eye,-1e9).max(1).values
            losses.append(F.relu(margin+neg-pos).mean())
    return _mean_or_zero(losses,zA)


def reciprocal_soft_cycle_loss(zA,zB,vA,vB,temperature: float = 0.08):
    """A->B soft assignment followed by B->A must return to the same carrier."""
    losses=[]
    B,V,N,D=zA.shape
    for b in range(B):
        for v in range(V):
            valid=vA[b,v].bool() & vB[b,v].bool()
            if int(valid.sum()) < 2: continue
            a=F.normalize(zA[b,v,valid],dim=-1,eps=1e-6)
            c=F.normalize(zB[b,v,valid],dim=-1,eps=1e-6)
            sim=(a@c.T)/temperature
            pab=F.softmax(sim,dim=1)
            pba=F.softmax(sim.T,dim=1)
            cyc=pab@pba
            losses.append(-torch.log(cyc.diag().clamp_min(1e-12)).mean())
    return _mean_or_zero(losses,zA)


def d1_descriptor_objective(sampled: Dict[str,torch.Tensor], target: Dict[str,torch.Tensor]) -> Tuple[torch.Tensor,Dict[str,torch.Tensor]]:
    zA=sampled['descriptor_srcA']; zB=sampled['descriptor_srcB']
    vA=target['V_A'].bool(); vB=target['V_B'].bool()
    parts={
        'D1_same_view_dual': same_view_dual_softmax_loss(zA,zB,vA,vB),
        'D1_multi_positive': observation_multi_positive_loss(zA,zB,vA,vB),
        'D1_hard_negative': hard_negative_margin_loss(zA,zB,vA,vB),
        'D1_reciprocal_cycle': reciprocal_soft_cycle_loss(zA,zB,vA,vB),
    }
    # Preregistered no-fit weighting: simple equal mean. No truth-driven coefficient search.
    total=sum(parts.values())/4.0
    parts['D1_Z_match']=total
    return total,parts


def observation_loss_d1(outputs, target, image_size: int = 256):
    """Replace only legacy descriptor Z_match; all other canonical N1D loss terms remain unchanged."""
    from realsas_iris_sees.losses import observation_loss as legacy_observation_loss
    from realsas_iris_sees.sampling import sample_predictions
    legacy_total, legacy_parts = legacy_observation_loss(outputs,target,image_size)
    sampled=sample_predictions(outputs,target['XY_A'],target['XY_B'],image_size)
    d1_z,d1_parts=d1_descriptor_objective(sampled,target)
    legacy_z=legacy_parts['Z_match']
    total=legacy_total - 0.5*legacy_z + 0.5*d1_z
    parts=dict(legacy_parts)
    parts['Z_match_legacy_diagnostic']=legacy_z
    parts.update(d1_parts)
    parts['Z_match']=d1_z
    parts['total']=total
    return total,parts


@torch.no_grad()
def descriptor_diagnostics(outputs,target,image_size: int = 256) -> Dict[str,float]:
    from realsas_iris_sees.sampling import sample_predictions
    s=sample_predictions(outputs,target['XY_A'],target['XY_B'],image_size)
    zA=F.normalize(s['descriptor_srcA'].float(),dim=-1,eps=1e-6)
    zB=F.normalize(s['descriptor_srcB'].float(),dim=-1,eps=1e-6)
    vA=target['V_A'].bool(); vB=target['V_B'].bool(); persistent=target['persistent_obs'].bool()
    pooled=[]; same=[]; recip=[]; margins=[]; pos_sims=[]
    B,V,N,D=zA.shape
    for b in range(B):
        ids=persistent[b].nonzero(as_tuple=False).flatten()
        if len(ids)>=2:
            ma=vA[b,:,ids].float().unsqueeze(-1); mb=vB[b,:,ids].float().unsqueeze(-1)
            a=F.normalize((zA[b,:,ids]*ma).sum(0)/ma.sum(0).clamp_min(1),dim=-1,eps=1e-6)
            c=F.normalize((zB[b,:,ids]*mb).sum(0)/mb.sum(0).clamp_min(1),dim=-1,eps=1e-6)
            sim=a@c.T
            pooled.append(float((sim.argmax(1)==torch.arange(len(ids),device=sim.device)).float().mean()))
        for v in range(V):
            valid=vA[b,v] & vB[b,v]
            if int(valid.sum())<2: continue
            a=zA[b,v,valid]; c=zB[b,v,valid]; sim=a@c.T
            lab=torch.arange(sim.shape[0],device=sim.device)
            ab=sim.argmax(1); ba=sim.argmax(0)
            same.append(float((ab==lab).float().mean()))
            recip.append(float(((ab==lab)&(ba==lab)).float().mean()))
            pos=sim.diag(); eye=torch.eye(sim.shape[0],device=sim.device,dtype=torch.bool)
            neg=sim.masked_fill(eye,-1e9).max(1).values
            margins.extend((pos-neg).detach().cpu().tolist())
            pos_sims.extend(pos.detach().cpu().tolist())
    import numpy as np
    return {
        'legacy_pooled_crosspose_top1': float(np.mean(pooled)) if pooled else float('nan'),
        'same_view_top1': float(np.mean(same)) if same else float('nan'),
        'reciprocal_same_view_top1': float(np.mean(recip)) if recip else float('nan'),
        'positive_minus_hardest_negative_margin_mean': float(np.mean(margins)) if margins else float('nan'),
        'positive_similarity_mean': float(np.mean(pos_sims)) if pos_sims else float('nan'),
        'same_view_rows': int(len(same)),
    }
