from __future__ import annotations
import torch
from torch import nn
from .contracts_v1 import IRISVNextConfigV1
from .heads_v1 import AppearanceEvidenceHeadV1, GraphicsRelationHeadV1, MechanicalDemandHeadV1
from .scene_core_v1 import IRISVNextSceneCoreV1
from .structure_v2 import IRISVNextStructureHeadV2


class IRISVNextV1(nn.Module):
    """One scene model. Outputs evidence only; Compiler mints product state."""
    architecture_id="RealSaS.IRIS.VNext.v1.1"; dino_policy="FROZEN_DINOV2_S_FRONTEND"; semantic_taxonomy_used=False; independent_visual_mesh_allowed=False
    def __init__(self,config:IRISVNextConfigV1=IRISVNextConfigV1()):
        super().__init__(); config.validate(); self.config=config; self.scene_core=IRISVNextSceneCoreV1(config); self.mechanical_demand=MechanicalDemandHeadV1(config); self.appearance=AppearanceEvidenceHeadV1(config); self.relations=GraphicsRelationHeadV1(config); self.structure=IRISVNextStructureHeadV2(config)
    def encode_scene(self,image_tokens,token_xy,camera_features): return self.scene_core.encode_scene(image_tokens,token_xy,camera_features)
    def query_evidence(self,scene_planes,points_normalized):
        local=self.scene_core.query(scene_planes,points_normalized); return {**local,"mechanical_demand":self.mechanical_demand(local["H"])}
    def query_appearance(self,h,camera_features): return self.appearance(h,camera_features)
    def decode_topology(self,**kwargs): return self.structure(**kwargs)
    def parameter_groups(self):
        scene=tuple(self.scene_core.parameters()); ids={id(p) for p in scene}; evidence=tuple(p for module in (self.mechanical_demand,self.appearance,self.relations) for p in module.parameters() if id(p) not in ids); used=ids|{id(p) for p in evidence}; structure=tuple(p for p in self.structure.parameters() if id(p) not in used); return {"scene_core":scene,"evidence_heads":evidence,"structure":structure}
    def trainable_parameter_count(self): return sum(p.numel() for p in self.parameters() if p.requires_grad)
