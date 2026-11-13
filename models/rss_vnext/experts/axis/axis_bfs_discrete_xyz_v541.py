from __future__ import annotations

"""AXIS V5.4.1: hard causal tree + deterministic ordered discrete XYZ.

This is the repo-native form of the sealed FIT1 research architecture. Historical
Geppetto diffusion implementations remain immutable for checkpoint replay.
"""

import math
from typing import Optional
import numpy as np
import torch
from torch import nn

from compiler.realsas_compiler_core.types import SkeletonProposalEdge,SkeletonProposalIR,SkeletonProposalJoint
from models.geppetto.reference_strength_v1.geppetto_reference_strength_no_learned_slot_v1 import GeppettoReferenceStrengthNoLearnedSlotV1
from models.geppetto.reference_strength_v1.geppetto_reference_strength_candidate_v1 import GeppettoReferenceStrengthRawOutputV1

AXIS_V541_ARCHITECTURE_ID="RealSaS.AXIS41.BFSHardParent.OrderedDiscreteXYZ256.FrozenCausal.v5_4_1"
AXIS_COORD_BINS=256
AXIS_COORD_ANCHOR_SIGMA=0.05


class ForbiddenRuntimeDiffusionV541(nn.Module):
    def __init__(self): super().__init__(); self.invocation_count=0
    def loss(self,*args,**kwargs): self.invocation_count+=1; raise RuntimeError("AXIS_V541_DIFFUSION_LOSS_FORBIDDEN")
    def sample(self,*args,**kwargs): self.invocation_count+=1; raise RuntimeError("AXIS_V541_DIFFUSION_SAMPLE_FORBIDDEN")


class AXISBFSDiscreteXYZV541(GeppettoReferenceStrengthNoLearnedSlotV1):
    def __init__(self):
        super().__init__()
        d=self.config.model_dim
        self.coord_token_delta=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,d),nn.GELU(),nn.Linear(d,3*AXIS_COORD_BINS))
        nn.init.zeros_(self.coord_token_delta[-1].weight); nn.init.zeros_(self.coord_token_delta[-1].bias)
        self.register_buffer("coord_token_centers",torch.linspace(-self.config.position_clip,self.config.position_clip,AXIS_COORD_BINS),persistent=False)
        self._runtime_diffusion_retired=False
        self._v541_training_authority_configured=False
        self._last_coord_logits=None; self._last_coord_indices=None; self._last_anchor_positions=None

    def retire_legacy_diffusion_runtime(self):
        """Call only after loading historical base/V5.3 checkpoint weights."""
        self.diffusion=ForbiddenRuntimeDiffusionV541()
        self._runtime_diffusion_retired=True
        return self

    def _require_runtime_ready(self):
        if not self._runtime_diffusion_retired or not isinstance(self.diffusion,ForbiddenRuntimeDiffusionV541):
            raise RuntimeError("AXIS_V541_LEGACY_DIFFUSION_NOT_RETIRED")

    def configure_v541_training_authority(self):
        """Freeze the solved V5.3 causal/tree system; train only discrete XYZ delta.

        Call after loading the sealed V5.3 causal checkpoint and after retiring
        legacy diffusion. This mirrors the V5.4.1 scientific run and prevents the
        coordinate migration from silently re-opening hard-parent authority.
        """
        self._require_runtime_ready()
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for parameter in self.coord_token_delta.parameters():
            parameter.requires_grad_(True)
        self._v541_training_authority_configured=True
        return self

    def v541_trainable_parameter_names(self):
        return tuple(name for name,parameter in self.named_parameters() if parameter.requires_grad)

    def coordinate_token_targets_v541(self, teacher_positions_normalized):
        target=torch.as_tensor(teacher_positions_normalized,device=self.coord_token_centers.device,dtype=torch.float32)
        if target.ndim==2:
            target=target.unsqueeze(0)
        if target.ndim!=3 or target.shape[-1]!=3:
            raise ValueError("AXIS_V541_COORD_TARGET_MUST_BE_BxKx3")
        centers=self.coord_token_centers.to(device=target.device,dtype=target.dtype)
        return torch.argmin(torch.abs(target[...,None]-centers),dim=-1)

    def coordinate_token_loss_v541(self, teacher_positions_normalized):
        if not self._v541_training_authority_configured:
            raise RuntimeError("AXIS_V541_TRAINING_AUTHORITY_NOT_CONFIGURED")
        if self._last_coord_logits is None:
            raise RuntimeError("AXIS_V541_COORD_LOGITS_UNAVAILABLE")
        target=self.coordinate_token_targets_v541(teacher_positions_normalized)
        logits=self._last_coord_logits
        if target.shape!=logits.shape[:-1]:
            raise ValueError("AXIS_V541_COORD_TARGET_SHAPE_DRIFT")
        return torch.nn.functional.cross_entropy(
            logits.reshape(-1,AXIS_COORD_BINS),target.reshape(-1)
        )

    def _previous_parent_distribution(self,current_state,current_position,previous_states,previous_positions):
        b,d=current_state.shape; k=len(previous_states)
        if k==0:
            return current_state.new_empty((b,0)),torch.zeros_like(current_position),torch.zeros_like(current_state)
        ps=torch.stack(previous_states,dim=1); pp=torch.stack(previous_positions,dim=1)
        cur=current_state[:,None,:].expand(b,k,d); cp=current_position[:,None,:].expand(b,k,3)
        delta=pp-cp; dist=torch.linalg.norm(delta,dim=-1,keepdim=True)
        logits=self.internal_parent(torch.cat([cur,ps,delta,dist],dim=-1)).squeeze(-1)
        hard=torch.argmax(logits,dim=-1)
        parent_pos=torch.gather(pp,1,hard[:,None,None].expand(-1,1,3)).squeeze(1)
        parent_state=torch.gather(ps,1,hard[:,None,None].expand(-1,1,d)).squeeze(1)
        return logits,parent_pos,parent_state

    def _discrete_coordinate(self,state):
        b=state.shape[0]
        anchor=torch.tanh(self.coarse_position(state))*self.config.position_clip
        centers=self.coord_token_centers.to(device=state.device,dtype=state.dtype)
        base=-((centers[None,None,:]-anchor[:,:,None])/AXIS_COORD_ANCHOR_SIGMA).square()
        logits=base+self.coord_token_delta(state).view(b,3,AXIS_COORD_BINS)
        index=torch.argmax(logits,dim=-1)
        return centers[index],logits,index,anchor

    def forward_surface(self,surface,*,decode_steps,teacher_positions_normalized=None,generator=None,diffusion_sample_steps=None):
        self._require_runtime_ready()
        if diffusion_sample_steps is not None: raise RuntimeError("AXIS_V541_DIFFUSION_SAMPLE_STEPS_FORBIDDEN")
        if decode_steps<1 or decode_steps>surface.node_count: raise ValueError("decode_steps exceeds surface-cardinality resource guard")
        device=next(self.parameters()).device; st=self._surface_tensors(surface,device); memory,pooled=self.encoder(**st); d=self.config.model_dim
        state_layers=[self.state_init(pooled) for _ in range(self.config.causal_layers)]
        prev_token=self.start[None].to(device=device).expand(1,-1); previous_states=[]; previous_positions=[]
        position_rows=[]; state_rows=[]; stop_rows=[]; exist_rows=[]; root_rows=[]; salience_rows=[]; support_presence_rows=[]; support_rows=[]; parent_rows=[]; sigma_rows=[]; attn_rows=[]; coord_logits_rows=[]; coord_index_rows=[]; anchor_rows=[]
        teacher=None
        if teacher_positions_normalized is not None:
            teacher=torch.as_tensor(teacher_positions_normalized,device=device,dtype=torch.float32)
            if teacher.shape!=(decode_steps,3): raise ValueError("teacher_positions_normalized must be [decode_steps,3]")
        for step in range(decode_steps):
            seq=self._sequence_embedding(step,d,device=device,dtype=pooled.dtype); query0=self.step_proj(seq+state_layers[-1])
            ctx,attn=self.cross_attn(query0[:,None,:],memory[None],memory[None],need_weights=True,average_attn_weights=True)
            surface_ctx=self.cross_fuse(torch.cat([query0,ctx[:,0]],dim=-1))
            if previous_positions:
                _,parent_pos,parent_state=self._previous_parent_distribution(state_layers[-1],previous_positions[-1],previous_states,previous_positions)
            else:
                parent_pos=torch.zeros((1,3),device=device,dtype=pooled.dtype); parent_state=torch.zeros_like(pooled)
            prev_pos=previous_positions[-1] if previous_positions else torch.zeros((1,3),device=device,dtype=pooled.dtype)
            fb=self.feedback(torch.cat([surface_ctx,prev_token,parent_state,prev_pos,parent_pos],dim=-1)); x=torch.cat([surface_ctx,prev_token,fb,seq],dim=-1)
            for li,cell in enumerate(self.cells):
                state_layers[li]=cell(x,state_layers[li]); x=torch.cat([surface_ctx,state_layers[li],fb,seq],dim=-1)
            state=state_layers[-1]
            position,coord_logits,coord_idx,anchor=self._discrete_coordinate(state)
            internal_logits,parent_pos2,parent_state2=self._previous_parent_distribution(state,position,previous_states,previous_positions); parent_rows.append(internal_logits)
            token=self.feedback(torch.cat([state,surface_ctx,parent_state2,position,parent_pos2],dim=-1)); previous_states.append(token); previous_positions.append(position); prev_token=token
            support_logits=torch.einsum("bd,nd->bn",self.support_q(state),self.support_k(memory))/math.sqrt(d)
            position_rows.append(position); state_rows.append(state); stop_rows.append(self.stop(state).squeeze(-1)); exist_rows.append(self.existence(state).squeeze(-1)); root_rows.append(self.root(state).squeeze(-1)); salience_rows.append(self.salience(state).squeeze(-1)); support_presence_rows.append(self.support_presence(state).squeeze(-1)); support_rows.append(support_logits); sigma_rows.append(self.position_log_sigma(state.detach()).clamp(-8.0,4.0)); attn_rows.append(attn[:,0]); coord_logits_rows.append(coord_logits); coord_index_rows.append(coord_idx); anchor_rows.append(anchor)
        states=torch.stack(state_rows,dim=1); positions=torch.stack(position_rows,dim=1); all_pair=self._all_pair_parent_logits(states,positions); k=decode_steps
        internal=states.new_full((1,k,k),-1e4)
        for child,logits in enumerate(parent_rows):
            if child: internal[:,child,:child]=logits
        self._last_coord_logits=torch.stack(coord_logits_rows,dim=1); self._last_coord_indices=torch.stack(coord_index_rows,dim=1); self._last_anchor_positions=torch.stack(anchor_rows,dim=1)
        zero=positions.new_zeros(()) if teacher is not None else None
        return GeppettoReferenceStrengthRawOutputV1(
            coarse_positions_normalized=positions,positions_normalized=positions,control_states=states,
            stop_logits=torch.stack(stop_rows,dim=1),existence_logits=torch.stack(exist_rows,dim=1),root_logits=torch.stack(root_rows,dim=1),salience_logits=torch.stack(salience_rows,dim=1),
            support_presence_logits=torch.stack(support_presence_rows,dim=1),support_logits=torch.stack(support_rows,dim=1),internal_parent_logits=internal,all_pair_parent_logits=all_pair,
            position_log_sigma=torch.stack(sigma_rows,dim=1),surface_attention=torch.stack(attn_rows,dim=1),diffusion_loss=zero,teacher_target_used=teacher is not None,teacher_feedback_used=False,
        )

    @torch.no_grad()
    def propose(self,surface,*,resource_step_limit:Optional[int]=None,generator:Optional[torch.Generator]=None,diffusion_sample_steps:Optional[int]=None)->SkeletonProposalIR:
        self.eval(); self._require_runtime_ready()
        out,count=self.generate_surface(surface,resource_step_limit=resource_step_limit,generator=generator,diffusion_sample_steps=diffusion_sample_steps)
        ids=tuple(f"P:GRS:{i:04d}" for i in range(count)); sp=torch.sigmoid(out.support_presence_logits[0,:count]); ex=torch.sigmoid(out.existence_logits[0,:count]); salience=torch.sigmoid(out.salience_logits[0,:count])
        joints=[]
        for i in range(count):
            pn=out.positions_normalized[0,i].detach().cpu().numpy(); world=pn.astype(np.float64)*float(surface.normalization_scale)+np.asarray(surface.normalization_center,np.float64); sigma=torch.exp(out.position_log_sigma[0,i]).mean(); support_ids=()
            if float(sp[i])>=self.config.support_presence_probability:
                top=min(self.config.support_topk,surface.node_count); idx=torch.argsort(out.support_logits[0,i],descending=True,stable=True)[:top].tolist(); support_ids=tuple(surface.surface_ids[j] for j in idx)
            confidence=float((ex[i]*torch.exp(-sigma)).clamp(0.0,1.0))
            joints.append(SkeletonProposalJoint(ids[i],tuple(map(float,world)),1.0 if i==0 else 0.0,confidence,support_ids,metadata={"generation_index_internal_only":i,"sequence_index":i,"tree_serialization":"BFS_PARENT_BEFORE_CHILD","hard_causal_parent_authority":True,"discrete_xyz_authority":True,"coord_bins_per_axis":AXIS_COORD_BINS,"mechanical_salience_probability":float(salience[i]),"teacher_feedback_used":False}))
        edges=[]
        for child in range(1,count):
            logits=out.internal_parent_logits[0,child,:child].float(); parent=int(torch.argmax(logits).item()); prob=torch.softmax(logits,dim=-1); score=float(prob[parent].item())
            edges.append(SkeletonProposalEdge(edge_id=f"E:{ids[parent]}->{ids[child]}",parent_proposal_id=ids[parent],child_proposal_id=ids[child],score=score,confidence=score,hard_required=True,hard_forbidden=False,reason="AXIS_V541_HARD_CAUSAL_PARENT_AUTHORITY",metadata={"compiler_may_reselect_parent":False,"all_pair_final_parent_authority":False}))
        return SkeletonProposalIR(tuple(joints),tuple(edges),surface.source_surface_hash,AXIS_V541_ARCHITECTURE_ID,metadata={
            "candidate_architecture":AXIS_V541_ARCHITECTURE_ID,"tree_serialization":"BFS_PARENT_BEFORE_CHILD","hard_causal_parent_authority":True,
            "compiler_role":"VALIDATE_AND_MATERIALIZE","compiler_parent_reselection_allowed":False,"geometry_authority":"ORDERED_256_BIN_XYZ_ARGMAX",
            "conditional_residual_diffusion":False,"diffusion_runtime_authority":False,"generation_index_is_not_identity":True,"generalization_claim":False,
        })

__all__=["AXISBFSDiscreteXYZV541","ForbiddenRuntimeDiffusionV541","AXIS_V541_ARCHITECTURE_ID","AXIS_COORD_BINS","AXIS_COORD_ANCHOR_SIGMA"]
