from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping
import torch
import torch.nn.functional as F
from .model_v1 import IRISVNextV1

@dataclass(frozen=True)
class IRISVNextOptimizationPolicyV1:
    scene_learning_rate:float=3e-5; evidence_learning_rate:float=1e-4; structure_learning_rate:float=5e-5; weight_decay:float=1e-2; betas:tuple[float,float]=(0.9,0.95); expected_cohort_size:int=8
    def validate(self):
        if min(self.scene_learning_rate,self.evidence_learning_rate,self.structure_learning_rate)<=0 or self.weight_decay<0 or self.expected_cohort_size<=0: raise ValueError("IRIS_VNEXT_OPTIMIZER_POLICY_INVALID")
@dataclass(frozen=True)
class IRISVNextLossWeightsV1:
    field:float=1.; mechanical:float=1.; appearance:float=0.; relations:float=0.; topology:float=0.; connectivity:float=0.
    def validate(self):
        v=(self.field,self.mechanical,self.appearance,self.relations,self.topology,self.connectivity)
        if min(v)<0 or sum(v)<=0: raise ValueError("IRIS_VNEXT_LOSS_WEIGHTS_INVALID")

def build_vnext_optimizer_v1(model:IRISVNextV1,policy=IRISVNextOptimizationPolicyV1()):
    """One optimizer lineage with controlled parameter-group LRs."""
    policy.validate(); groups=model.parameter_groups(); kwargs={"betas":policy.betas,"weight_decay":float(policy.weight_decay)}
    if next(model.parameters()).device.type=="cuda": kwargs["fused"]=True
    return torch.optim.AdamW([{"params":groups["scene_core"],"lr":policy.scene_learning_rate,"name":"scene_core"},{"params":groups["evidence_heads"],"lr":policy.evidence_learning_rate,"name":"evidence_heads"},{"params":groups["structure"],"lr":policy.structure_learning_rate,"name":"structure"}],**kwargs)

def mechanical_demand_loss_v1(predicted:Mapping[str,torch.Tensor],truth:Mapping[str,torch.Tensor],*,valid_mask=None):
    req=("stretch_major","stretch_minor","tangent_direction_2theta","bend_demand","rigidity_probability")
    if any(k not in predicted or k not in truth for k in req): raise ValueError("IRIS_VNEXT_MECHANICAL_TRUTH_INCOMPLETE")
    def reduce(x):
        if valid_mask is None:return x.mean()
        m=valid_mask.to(dtype=x.dtype)
        while m.ndim<x.ndim:m=m.unsqueeze(-1)
        denom=m.sum().clamp_min(1.)*(x.shape[-1] if x.ndim>valid_mask.ndim else 1); return (x*m).sum()/denom
    stretch=F.smooth_l1_loss(predicted["stretch_major"],truth["stretch_major"],reduction="none")+F.smooth_l1_loss(predicted["stretch_minor"],truth["stretch_minor"],reduction="none"); p=F.normalize(predicted["tangent_direction_2theta"],dim=-1,eps=1e-6); t=F.normalize(truth["tangent_direction_2theta"],dim=-1,eps=1e-6); tangent=1.-(p*t).sum(-1); bend=F.smooth_l1_loss(predicted["bend_demand"],truth["bend_demand"],reduction="none"); rigid=F.binary_cross_entropy(predicted["rigidity_probability"].clamp(1e-6,1-1e-6),truth["rigidity_probability"],reduction="none"); return reduce(stretch)+reduce(tangent)+reduce(bend)+reduce(rigid)

def graphics_relation_loss_v1(predicted,truth):
    keys=("same_drawable_affinity","cut_probability","attachment_affinity","alpha_boundary_probability","appearance_boundary_probability","front_probability","occlusion_probability","visibility_probability_a","visibility_probability_b"); terms=[]
    for k in keys:
        if k in truth:
            if k not in predicted: raise ValueError("IRIS_VNEXT_RELATION_PREDICTION_MISSING:"+k)
            terms.append(F.binary_cross_entropy(predicted[k].clamp(1e-6,1-1e-6),truth[k]))
    if not terms: raise ValueError("IRIS_VNEXT_RELATION_TRUTH_EMPTY")
    return sum(terms)/len(terms)

def assert_batched_fit8_v1(batch,expected=8):
    for k in ("image_tokens","token_xy","camera_features"):
        if k not in batch or int(batch[k].shape[0])!=int(expected): raise ValueError("IRIS_VNEXT_REQUIRES_BATCHED_COHORT:"+k)

class IRISVNextTrainStepV1:
    """One call = one vectorized whole-cohort optimizer update; no subject loop."""
    def __init__(self,model,optimizer,*,expected_cohort_size=8): self.model=model; self.optimizer=optimizer; self.expected_cohort_size=int(expected_cohort_size)
    def geometry_mechanical_step(self,batch,*,loss_weights=IRISVNextLossWeightsV1()):
        loss_weights.validate(); assert_batched_fit8_v1(batch,self.expected_cohort_size)
        for k in ("query_points","field_truth","mechanical_stretch_major","mechanical_stretch_minor","mechanical_tangent_direction_2theta","mechanical_bend_demand","mechanical_rigidity_probability"):
            if k not in batch: raise ValueError("IRIS_VNEXT_BATCH_MISSING:"+k)
        device=next(self.model.parameters()).device.type; self.model.train(); self.optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device,dtype=torch.bfloat16,enabled=device=="cuda"):
            planes=self.model.encode_scene(batch["image_tokens"],batch["token_xy"],batch["camera_features"]); evidence=self.model.query_evidence(planes,batch["query_points"]); field=F.smooth_l1_loss(evidence["field"].float(),batch["field_truth"].float()); truth={"stretch_major":batch["mechanical_stretch_major"],"stretch_minor":batch["mechanical_stretch_minor"],"tangent_direction_2theta":batch["mechanical_tangent_direction_2theta"],"bend_demand":batch["mechanical_bend_demand"],"rigidity_probability":batch["mechanical_rigidity_probability"]}; mech=mechanical_demand_loss_v1(evidence["mechanical_demand"],truth,valid_mask=batch.get("mechanical_valid_mask")); total=loss_weights.field*field+loss_weights.mechanical*mech
        if not bool(torch.isfinite(total)): raise RuntimeError("IRIS_VNEXT_TRAINING_NONFINITE_LOSS")
        total.backward(); self.optimizer.step(); return {"loss":float(total.detach().cpu()),"field_loss":float(field.detach().cpu()),"mechanical_loss":float(mech.detach().cpu()),"batch_subject_count":float(self.expected_cohort_size)}
