from __future__ import annotations

"""Behavior + hard-tail + articulated-consequence objective for Arachne A1 v4."""
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import torch
import torch.nn.functional as F


def _hash(x): return sha256(json.dumps(x, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class ArachneA1LossConfigV4:
    scalar_bce_weight: float = 1.0
    scalar_mse_weight: float = 0.1
    scalar_dice_weight: float = 1.0
    normalized_row_l1_weight: float = 1.0
    hard_tail_cvar10_weight: float = 0.5
    blend_boundary_weight: float = 0.5
    articulated_deformation_weight: float = 0.25
    dice_epsilon: float = 1e-4
    architecture_id: str = "RealSaS.Arachne.A1.BehaviorHardTailArticulatedLoss.v4"

    def validate(self):
        vals=(self.scalar_bce_weight,self.scalar_mse_weight,self.scalar_dice_weight,self.normalized_row_l1_weight,self.hard_tail_cvar10_weight,self.blend_boundary_weight,self.articulated_deformation_weight)
        if min(vals)<0: raise ValueError("loss weights must be nonnegative")
        if sum(vals[:3])<=0 or self.normalized_row_l1_weight<=0: raise ValueError("reconstruction and coupled row loss required")
        if self.dice_epsilon<=0: raise ValueError("invalid dice epsilon")

    @property
    def config_hash(self): self.validate(); return _hash(asdict(self))


def normalize_joint_fields(raw_prob, joint_mask=None):
    if raw_prob.ndim!=3: raise ValueError("raw_prob must be [B,N,J]")
    p=raw_prob
    if joint_mask is not None:
        if joint_mask.shape!=(p.shape[0],p.shape[2]): raise ValueError("joint_mask shape drift")
        p=p*joint_mask[:,None,:].to(p.dtype)
    return p/p.sum(-1,keepdim=True).clamp_min(1e-8)


def lbs(rest_world, weights, transforms):
    ones=torch.ones((*rest_world.shape[:2],1),device=rest_world.device,dtype=rest_world.dtype)
    hom=torch.cat([rest_world,ones],-1)
    moved=torch.einsum('bpjac,bnc->bpjna',transforms,hom)[...,:3]
    return torch.einsum('bnj,bpjna->bpna',weights,moved)


def articulated_deformation_ratio_loss(pred, truth, rest_world, transforms, row_mask):
    td=lbs(rest_world,truth,transforms); pd=lbs(rest_world,pred,transforms)
    rp=rest_world[:,None].expand_as(td); m=row_mask[:,None,:,None].to(td.dtype)
    denom=(m.sum()*td.shape[1]*td.shape[-1]).clamp_min(1.0)
    motion=torch.sqrt((((td-rp)*m).square().sum()/denom).clamp_min(1e-12))
    err=torch.sqrt((((pd-td)*m).square().sum()/denom).clamp_min(1e-12))
    return err/motion.clamp_min(1e-6), motion, err


def arachne_a1_behavior_loss_v4(logits, truth, row_mask, *, joint_mask=None, rest_world=None, articulated_transforms=None, config=ArachneA1LossConfigV4()):
    config.validate()
    if logits.shape!=truth.shape or logits.ndim!=3: raise ValueError("logits/truth must match [B,N,J]")
    if row_mask.shape!=logits.shape[:2] or not bool(row_mask.any()): raise ValueError("row mask drift/empty")
    with torch.autocast(device_type=logits.device.type, enabled=False):
        z=logits.float(); t=truth.float(); rm=row_mask.bool(); em=rm[...,None].expand_as(z)
        if joint_mask is not None: em=em & joint_mask[:,None,:].bool()
        zv=z[em]; tv=t[em]; raw=torch.sigmoid(z)
        bce=F.binary_cross_entropy_with_logits(zv,tv); mse=F.mse_loss(raw[em],tv)
        m=rm[...,None].to(raw.dtype)
        if joint_mask is not None: m=m*joint_mask[:,None,:].to(raw.dtype)
        pr=raw*m; tr=t*m
        numer=2*(pr*tr).sum(1)+config.dice_epsilon; denom=pr.square().sum(1)+tr.square().sum(1)+config.dice_epsilon
        dpj=1-numer/denom
        if joint_mask is None: dice=dpj.mean()
        else:
            jm=joint_mask.to(dpj.dtype); dice=(dpj*jm).sum()/jm.sum().clamp_min(1)
        pred=normalize_joint_fields(raw,joint_mask)
        row_l1=(pred-t).abs().sum(-1); rv=row_l1[rm]
        coupled=rv.mean()
        k=max(1,int((rv.numel()+9)//10)); hard_tail=torch.topk(rv,k=k,largest=True).values.mean()
        blend=(1.0-t.max(-1).values).clamp_min(0.0)
        bw=blend[rm]; boundary=(rv*bw).sum()/bw.sum().clamp_min(1e-8) if bool((bw>0).any()) else rv.new_zeros(())
        deform=rv.new_zeros(()); teacher_motion=rv.new_zeros(()); deform_rms=rv.new_zeros(())
        if config.articulated_deformation_weight>0:
            if rest_world is None or articulated_transforms is None: raise ValueError("articulated consequence tensors required")
            deform,teacher_motion,deform_rms=articulated_deformation_ratio_loss(pred,t,rest_world.float(),articulated_transforms.float(),rm)
        scalar=config.scalar_bce_weight*bce+config.scalar_mse_weight*mse+config.scalar_dice_weight*dice
        total=(scalar + config.normalized_row_l1_weight*coupled + config.hard_tail_cvar10_weight*hard_tail + config.blend_boundary_weight*boundary + config.articulated_deformation_weight*deform)
    return {"total":total,"scalar_total":scalar,"bce":bce,"mse":mse,"dice":dice,"normalized_row_l1":coupled,"hard_tail_cvar10":hard_tail,"blend_boundary_l1":boundary,"articulated_deformation_ratio":deform,"articulated_teacher_motion_rms":teacher_motion,"articulated_deformation_error_rms":deform_rms}


__all__=["ArachneA1LossConfigV4","normalize_joint_fields","lbs","articulated_deformation_ratio_loss","arachne_a1_behavior_loss_v4"]
