from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .contracts_v1 import IRISVNextConfigV1


class MechanicalDemandHeadV1(nn.Module):
    """Predict deformation demand, not rig or skin authority."""
    output_semantics = "TRAINED_MECHANICAL_DEMAND_EVIDENCE_ONLY"
    def __init__(self, cfg: IRISVNextConfigV1):
        super().__init__(); self.net = nn.Sequential(nn.LayerNorm(cfg.local_feature_dim), nn.Linear(cfg.local_feature_dim, cfg.local_feature_dim), nn.GELU(), nn.Linear(cfg.local_feature_dim, 7))
    def forward(self, h):
        raw = self.net(h); minor = F.softplus(raw[..., 0]); major = minor + F.softplus(raw[..., 1]); tangent = F.normalize(raw[..., 2:4], dim=-1, eps=1e-6)
        return {"stretch_major": major, "stretch_minor": minor, "tangent_direction_2theta": tangent, "bend_demand": F.softplus(raw[..., 4]), "rigidity_probability": torch.sigmoid(raw[..., 5]), "uncertainty": F.softplus(raw[..., 6]) + 1e-6}


class AppearanceEvidenceHeadV1(nn.Module):
    """Non-authoritative view-conditioned Graphics evidence; observed source wins downstream."""
    output_semantics = "GRAPHICS_APPEARANCE_EVIDENCE_ONLY"
    def __init__(self, cfg):
        super().__init__(); self.camera = nn.Sequential(nn.Linear(cfg.camera_feature_dim, cfg.camera_embed_dim), nn.GELU(), nn.Linear(cfg.camera_embed_dim, cfg.camera_embed_dim)); hidden = cfg.local_feature_dim
        self.body = nn.Sequential(nn.LayerNorm(cfg.local_feature_dim + cfg.camera_embed_dim), nn.Linear(cfg.local_feature_dim + cfg.camera_embed_dim, hidden), nn.GELU()); self.rgba = nn.Linear(hidden,4); self.confidence=nn.Linear(hidden,1); self.descriptor=nn.Linear(hidden,cfg.appearance_descriptor_dim); self.boundary=nn.Linear(hidden,2)
    def forward(self,h,camera_features):
        if h.ndim != 3 or camera_features.ndim != 2 or camera_features.shape[0] != h.shape[0]: raise ValueError("IRIS_VNEXT_APPEARANCE_INPUT_SHAPE_INVALID")
        c=self.camera(camera_features)[:,None,:].expand(-1,h.shape[1],-1); z=self.body(torch.cat((h,c),dim=-1)); b=torch.sigmoid(self.boundary(z))
        return {"rgba_candidate":torch.sigmoid(self.rgba(z)),"confidence":torch.sigmoid(self.confidence(z)).squeeze(-1),"appearance_descriptor":F.normalize(self.descriptor(z),dim=-1,eps=1e-6),"alpha_boundary_probability":b[...,0],"appearance_boundary_probability":b[...,1]}


class GraphicsRelationHeadV1(nn.Module):
    """Role-free ownership/order/visibility/attachment evidence; no body-part taxonomy."""
    output_semantics = "ROLE_FREE_GRAPHICS_RELATION_EVIDENCE_ONLY"; categorical_semantics_used=False
    def __init__(self,cfg):
        super().__init__(); self.cfg=cfg; self.embedding=nn.Sequential(nn.LayerNorm(cfg.local_feature_dim),nn.Linear(cfg.local_feature_dim,cfg.relation_embedding_dim),nn.GELU(),nn.Linear(cfg.relation_embedding_dim,cfg.relation_embedding_dim)); pair=cfg.relation_embedding_dim*4
        self.static_pair=nn.Sequential(nn.LayerNorm(pair),nn.Linear(pair,cfg.relation_embedding_dim),nn.GELU(),nn.Linear(cfg.relation_embedding_dim,6)); self.view=nn.Sequential(nn.Linear(cfg.camera_feature_dim,cfg.camera_embed_dim),nn.GELU(),nn.Linear(cfg.camera_embed_dim,cfg.relation_embedding_dim)); self.view_pair=nn.Sequential(nn.LayerNorm(pair+cfg.relation_embedding_dim),nn.Linear(pair+cfg.relation_embedding_dim,cfg.relation_embedding_dim),nn.GELU(),nn.Linear(cfg.relation_embedding_dim,5))
    @staticmethod
    def _pair(a,b): return torch.cat((a,b,a-b,a*b),dim=-1)
    def face_embedding(self,h): return F.normalize(self.embedding(h),dim=-1,eps=1e-6)
    def static_relations(self,feature_a,feature_b):
        a,b=self.face_embedding(feature_a),self.face_embedding(feature_b); p=torch.sigmoid(self.static_pair(self._pair(a,b)))
        return {"same_drawable_affinity":p[...,0],"cut_probability":p[...,1],"attachment_affinity":p[...,2],"alpha_boundary_probability":p[...,3],"appearance_boundary_probability":p[...,4],"confidence":p[...,5],"fragment_embedding_a":a,"fragment_embedding_b":b}
    def view_relations(self,feature_a,feature_b,camera_features):
        a,b=self.face_embedding(feature_a),self.face_embedding(feature_b)
        if camera_features.shape[:-1] != a.shape[:-1]: raise ValueError("IRIS_VNEXT_RELATION_CAMERA_PREFIX_MISMATCH")
        p=torch.sigmoid(self.view_pair(torch.cat((self._pair(a,b),self.view(camera_features)),dim=-1)))
        return {"front_probability":p[...,0],"occlusion_probability":p[...,1],"visibility_probability_a":p[...,2],"visibility_probability_b":p[...,3],"confidence":p[...,4]}
