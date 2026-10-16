from __future__ import annotations
import math
import torch
from torch import nn
from models.iris.topology.head_v1 import TESSADecoderOnlyV1
from .contracts_v1 import IRISVNextConfigV1


def _triplane_position_features(h,w,*,device,dtype):
    y=torch.linspace(-1.,1.,h,device=device,dtype=dtype); x=torch.linspace(-1.,1.,w,device=device,dtype=dtype); yy,xx=torch.meshgrid(y,x,indexing="ij"); xy=torch.stack((xx,yy),dim=-1); pid=torch.eye(3,device=device,dtype=dtype)[:,None,None,:].expand(3,h,w,3); return torch.cat((xy[None].expand(3,-1,-1,-1),pid),dim=-1)


class IRISVNextStructureMemoryV1(nn.Module):
    """Bounded structure memory over the shared scene and SurfaceWitness."""
    architecture_id="RealSaS.IRIS.VNext.StructureMemory.v1"; authority="PROPOSAL_EVIDENCE_ONLY"
    def __init__(self,cfg:IRISVNextConfigV1):
        super().__init__(); self.cfg=cfg; self.scene_projection=nn.Sequential(nn.Linear(cfg.plane_channels+5,cfg.topology_source_width),nn.LayerNorm(cfg.topology_source_width),nn.GELU()); witness_dim=3+3+cfg.local_feature_dim+1+7; self.witness_projection=nn.Sequential(nn.Linear(witness_dim,cfg.topology_source_width),nn.LayerNorm(cfg.topology_source_width),nn.GELU()); self.memory_queries=nn.Parameter(torch.randn(cfg.topology_memory_tokens,cfg.topology_width)/math.sqrt(cfg.topology_width)); self.query_norm=nn.LayerNorm(cfg.topology_width); self.source_norm=nn.LayerNorm(cfg.topology_source_width); self.cross=nn.MultiheadAttention(embed_dim=cfg.topology_width,num_heads=cfg.topology_heads,dropout=cfg.dropout,kdim=cfg.topology_source_width,vdim=cfg.topology_source_width,batch_first=True)
        if cfg.topology_refinement_layers:
            layer=nn.TransformerEncoderLayer(d_model=cfg.topology_width,nhead=cfg.topology_heads,dim_feedforward=cfg.topology_width*cfg.feedforward_multiplier,dropout=cfg.dropout,activation="gelu",batch_first=True,norm_first=True); self.refinement=nn.TransformerEncoder(layer,num_layers=cfg.topology_refinement_layers)
        else: self.refinement=nn.Identity()
        self.out_norm=nn.LayerNorm(cfg.topology_width)
    def _scene_tokens(self,planes):
        if planes.ndim!=5 or planes.shape[1]!=3 or planes.shape[2]!=self.cfg.plane_channels: raise ValueError("IRIS_VNEXT_STRUCTURE_TRIPLANE_SHAPE_INVALID")
        b,p,c,h,w=planes.shape; values=planes.permute(0,1,3,4,2).contiguous(); pos=_triplane_position_features(h,w,device=planes.device,dtype=planes.dtype).unsqueeze(0).expand(b,-1,-1,-1,-1); return self.scene_projection(torch.cat((values,pos),dim=-1).reshape(b,p*h*w,c+5))
    @staticmethod
    def mechanical_vector(m):
        return torch.cat((m["stretch_major"][...,None],m["stretch_minor"][...,None],m["tangent_direction_2theta"],m["bend_demand"][...,None],m["rigidity_probability"][...,None],m["uncertainty"][...,None]),dim=-1)
    def forward(self,*,scene_planes,witness_xyz,witness_normal,witness_h,witness_log_uncertainty,witness_mechanical,witness_valid_mask=None):
        b=scene_planes.shape[0]
        if witness_xyz.ndim!=3 or witness_xyz.shape[-1]!=3 or witness_normal.shape!=witness_xyz.shape or witness_h.shape[:2]!=witness_xyz.shape[:2] or witness_h.shape[-1]!=self.cfg.local_feature_dim or witness_log_uncertainty.shape!=witness_xyz.shape[:2] or witness_xyz.shape[0]!=b: raise ValueError("IRIS_VNEXT_WITNESS_SHAPE_INVALID")
        mech=self.mechanical_vector(witness_mechanical)
        if mech.shape[:2]!=witness_xyz.shape[:2] or mech.shape[-1]!=7: raise ValueError("IRIS_VNEXT_WITNESS_MECHANICAL_SHAPE_INVALID")
        witness=self.witness_projection(torch.cat((witness_xyz,witness_normal,witness_h,witness_log_uncertainty[...,None],mech),dim=-1)); scene=self._scene_tokens(scene_planes); source=torch.cat((scene,witness),dim=1); mask=None
        if witness_valid_mask is not None:
            if witness_valid_mask.shape!=witness_xyz.shape[:2]: raise ValueError("IRIS_VNEXT_WITNESS_MASK_SHAPE_INVALID")
            valid=torch.cat((torch.ones((b,scene.shape[1]),device=scene.device,dtype=torch.bool),witness_valid_mask.bool()),dim=1); mask=~valid
        q=self.memory_queries.unsqueeze(0).expand(b,-1,-1); update,_=self.cross(query=self.query_norm(q),key=self.source_norm(source),value=self.source_norm(source),key_padding_mask=mask,need_weights=False); return self.out_norm(self.refinement(q+update))


class IRISVNextStructureHeadV1(nn.Module):
    """Animation-aware TESSA grammar decoder; emits proposal evidence only."""
    architecture_id="RealSaS.IRIS.VNext.StructureHead.v1"; output_authority="MESH_CANDIDATE_SET_PROPOSAL_ONLY"
    def __init__(self,cfg): super().__init__(); self.cfg=cfg; self.memory=IRISVNextStructureMemoryV1(cfg); self.decoder=TESSADecoderOnlyV1(cfg.tessa_decoder_config())
    def forward(self,*,scene_planes,witness_xyz,witness_normal,witness_h,witness_log_uncertainty,witness_mechanical,input_ids,labels=None,witness_valid_mask=None,sequence_position_offset=0):
        memory=self.memory(scene_planes=scene_planes,witness_xyz=witness_xyz,witness_normal=witness_normal,witness_h=witness_h,witness_log_uncertainty=witness_log_uncertainty,witness_mechanical=witness_mechanical,witness_valid_mask=witness_valid_mask); return self.decoder(memory=memory,input_ids=input_ids,labels=labels,sequence_position_offset=sequence_position_offset)
