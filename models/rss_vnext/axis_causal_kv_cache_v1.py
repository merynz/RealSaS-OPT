from __future__ import annotations

"""Exact-weight causal KV replay of the existing trained AXIS V6 program.

A full prefix recomputed for every autoregressive JOINT/CHAIN/SKIN_CODE
causes O(N^3) total attention work. This decoder caches each Transformer
layer's pre-attention causal K/V and computes exactly one next-token query
per event. It uses the SAME module weights; no second checkpoint, no teacher
data, no change to AXIS V6 event grammar, and no invented controls.

Only batch-1 unpadded eval/no-grad generation is supported. Any unsupported
Transformer implementation must fail closed, not silently run a different
architecture. Numerics are verified against full-prefix forward in CI.
"""
import torch
from torch import Tensor
from torch.nn import functional as F

from models.axis.rig_program_model_v6 import (
    AXISRigProgramModelV6, AXISSequenceV6, RigProgramEventV6,
)


class AXISV6CausalKVCacheV1:
    def __init__(self, model:AXISRigProgramModelV6, scene_memory:Tensor):
        if (
            not isinstance(model,AXISRigProgramModelV6)
            or model.training
            or torch.is_grad_enabled()
            or scene_memory.ndim!=3 or scene_memory.shape[0]!=1
            or scene_memory.shape[-1]!=model.config.scene_dim
            or not model.causal.layers
        ):
            raise ValueError("AXIS_V6_KV_CACHE_EVAL_BATCH_ONE_REQUIRED")
        self.model=model
        self.count=0
        self.scene=model.scene_projection(scene_memory)
        self.layers=tuple(model.causal.layers)
        for layer in self.layers:
            a=layer.self_attn
            if (
                not layer.norm_first or layer.training
                or not a.batch_first
                or a.in_proj_weight is None
                or a.bias_k is not None or a.bias_v is not None
                or a.add_zero_attn
                or a.embed_dim!=model.config.hidden_dim
                or a.num_heads!=model.config.heads
            ):
                raise ValueError("AXIS_V6_KV_CACHE_UNSUPPORTED_LAYER_ABI")
        self.keys:list[Tensor|None]=[None]*len(self.layers)
        self.values:list[Tensor|None]=[None]*len(self.layers)

    def consume(self,event:RigProgramEventV6)->dict[str,Tensor]:
        if (
            type(event) is not RigProgramEventV6
            or self.count >= self.model.config.max_sequence_length
        ):
            raise ValueError("AXIS_V6_KV_CACHE_EVENT_RESOURCE_GUARD")
        seq=AXISSequenceV6.from_events((event,),device=self.scene.device)
        # A 1-token source sequence always gets position zero. Replace ONLY
        # that learned position embedding with the true event offset.
        x=self.model._embed(seq)
        pos=self.model.position_embedding.weight
        x=x-pos[0].view(1,1,-1)+pos[self.count].view(1,1,-1)
        for li,layer in enumerate(self.layers):
            attn=layer.self_attn
            h=attn.num_heads
            d=attn.head_dim
            u=layer.norm1(x)
            q,k,v=F.linear(
                u,attn.in_proj_weight,attn.in_proj_bias,
            ).chunk(3,dim=-1)
            def split(t:Tensor)->Tensor:
                return t.reshape(1,1,h,d).transpose(1,2)
            q,k,v=split(q),split(k),split(v)
            old_k,old_v=self.keys[li],self.values[li]
            k_all=k if old_k is None else torch.cat((old_k,k),dim=2)
            v_all=v if old_v is None else torch.cat((old_v,v),dim=2)
            self.keys[li],self.values[li]=k_all,v_all
            # The query is the newest token, so every cached key is in its
            # causal past. No triangular mask is needed for this one row.
            attention=F.scaled_dot_product_attention(
                q,k_all,v_all,dropout_p=0.,is_causal=False,
            )
            attention=attention.transpose(1,2).reshape(1,1,h*d)
            out=F.linear(
                attention,attn.out_proj.weight,attn.out_proj.bias,
            )
            x=x+layer.dropout1(out)
            x=x+layer._ff_block(layer.norm2(x))
        if self.model.causal.norm is not None:
            x=self.model.causal.norm(x)
        cross,_=self.model.scene_cross_attention(
            x,self.scene,self.scene,need_weights=False,
        )
        hidden=self.model.cross_norm(x+cross)
        self.count+=1
        return self.model._prediction_heads(hidden)


__all__=["AXISV6CausalKVCacheV1"]
