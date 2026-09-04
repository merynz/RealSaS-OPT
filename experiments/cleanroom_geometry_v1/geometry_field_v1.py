from __future__ import annotations
from dataclasses import dataclass
import math
import torch
from torch import nn
import torch.nn.functional as F

@dataclass(frozen=True)
class GeometryFieldConfigV1:
    image_token_dim: int
    scene_dim: int = 256
    camera_embed_dim: int = 96
    num_heads: int = 8
    num_layers: int = 4
    plane_size: int = 24
    plane_channels: int = 64
    field_hidden_dim: int = 192
    def __post_init__(self):
        vals=(self.image_token_dim,self.scene_dim,self.camera_embed_dim,self.num_heads,self.num_layers,self.plane_size,self.plane_channels,self.field_hidden_dim)
        if min(map(int,vals))<=0: raise ValueError("all geometry-field dimensions must be positive")
        if self.scene_dim % self.num_heads: raise ValueError("scene_dim must be divisible by num_heads")
        if self.plane_size < 4: raise ValueError("plane_size must be >= 4")

@dataclass
class ContinuousSurfaceFieldOutputV1:
    unsigned_distance: torch.Tensor
    log_uncertainty: torch.Tensor
    latent: torch.Tensor

@dataclass
class CleanroomGeometryFieldOutputV1:
    triplanes: torch.Tensor
    field: ContinuousSurfaceFieldOutputV1 | None

def pack_orthographic_camera_features_v1(origin,right,up,forward,half_extent):
    """Pack exact camera geometry with no categorical view identity.

    Output [B,8,13] = origin/right/up/forward/log(half_extent).
    """
    vectors=(origin,right,up,forward)
    if any(x.ndim!=3 or x.shape[-1]!=3 for x in vectors): raise ValueError("camera vectors must be [B,V,3]")
    if any(x.shape[:2]!=origin.shape[:2] for x in vectors): raise ValueError("camera vector batch/view mismatch")
    if origin.shape[1]!=8: raise ValueError("cleanroom V1 requires exact eight-view camera set")
    if half_extent.ndim==2: half_extent=half_extent[...,None]
    if half_extent.shape != (*origin.shape[:2],1): raise ValueError("half_extent must be [B,V] or [B,V,1]")
    if not all(torch.isfinite(x).all() for x in (*vectors,half_extent)): raise ValueError("camera features must be finite")
    if (half_extent<=0).any(): raise ValueError("half_extent must be positive")
    return torch.cat([origin,right,up,forward,half_extent.log()],dim=-1)

class ExactCameraTokenEmbeddingV1(nn.Module):
    """Learn exact geometric camera conditioning; no learned 8-slot table."""
    CAMERA_FEATURE_DIM=13
    def __init__(self,scene_dim,camera_embed_dim):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(13,camera_embed_dim),nn.GELU(),nn.Linear(camera_embed_dim,scene_dim))
    def forward(self,x):
        if x.ndim!=3 or x.shape[1:]!=(8,13): raise ValueError("camera_features must be [B,8,13]")
        return self.net(x)

class CameraAwareViewTokenBankV1(nn.Module):
    """Late-condition image tokens with exact camera geometry and raster position.

    The upstream foundation encoder remains untouched in V1. Encoder-internal
    camera modulation is reserved for a later challenger only if this arm shows a
    demonstrated ceiling.
    """
    def __init__(self,image_token_dim,scene_dim,camera_embed_dim):
        super().__init__()
        self.image_projection=nn.Linear(image_token_dim,scene_dim)
        self.camera_embedding=ExactCameraTokenEmbeddingV1(scene_dim,camera_embed_dim)
        self.xy_embedding=nn.Sequential(nn.Linear(2,camera_embed_dim),nn.GELU(),nn.Linear(camera_embed_dim,scene_dim))
        self.norm=nn.LayerNorm(scene_dim)
    def forward(self,image_tokens,token_xy,camera_features,token_valid=None):
        if image_tokens.ndim!=4 or image_tokens.shape[1]!=8: raise ValueError("image_tokens must be [B,8,T,C]")
        B,V,T,_=image_tokens.shape
        if token_xy.shape!=(B,V,T,2): raise ValueError("token_xy must be [B,8,T,2]")
        if camera_features.shape!=(B,V,13): raise ValueError("camera feature shape mismatch")
        if token_valid is not None and token_valid.shape!=(B,V,T): raise ValueError("token_valid must be [B,8,T]")
        if not torch.isfinite(image_tokens).all() or not torch.isfinite(token_xy).all(): raise ValueError("image tokens/xy must be finite")
        if (token_xy.abs()>1.00001).any(): raise ValueError("token_xy must be normalized to [-1,1]")
        memory=self.norm(self.image_projection(image_tokens)+self.camera_embedding(camera_features)[:,:,None,:]+self.xy_embedding(token_xy))
        memory=memory.reshape(B,V*T,-1)
        mask=None if token_valid is None else ~token_valid.reshape(B,V*T).bool()
        if mask is not None and mask.all(dim=1).any(): raise ValueError("every batch item needs at least one valid image token")
        return memory,mask

def _canonical_plane_coordinates(p):
    axis=torch.linspace(-1,1,p)
    yy,xx=torch.meshgrid(axis,axis,indexing="ij")
    base=torch.stack([xx,yy],-1)
    ids=torch.eye(3)[:,None,None,:].expand(3,p,p,3)
    return torch.cat([base[None].expand(3,-1,-1,-1),ids],-1)

class SharedTriplaneSceneFusionV1(nn.Module):
    """Global canonical scene queries cross-attend to all camera-aware view tokens."""
    def __init__(self,cfg):
        super().__init__()
        self.cfg=cfg
        self.scene_content=nn.Parameter(torch.randn(3,cfg.plane_size,cfg.plane_size,cfg.scene_dim)/math.sqrt(cfg.scene_dim))
        self.register_buffer("plane_coordinates",_canonical_plane_coordinates(cfg.plane_size),persistent=False)
        self.position_embedding=nn.Sequential(nn.Linear(5,cfg.scene_dim),nn.GELU(),nn.Linear(cfg.scene_dim,cfg.scene_dim))
        layer=nn.TransformerDecoderLayer(cfg.scene_dim,cfg.num_heads,cfg.scene_dim*3,dropout=0,batch_first=True,norm_first=True,activation="gelu")
        self.decoder=nn.TransformerDecoder(layer,cfg.num_layers,norm=nn.LayerNorm(cfg.scene_dim))
        self.upsample=nn.ConvTranspose2d(cfg.scene_dim,cfg.plane_channels,2,2)
    def forward(self,memory,memory_padding_mask=None):
        if memory.ndim!=3 or memory.shape[-1]!=self.cfg.scene_dim: raise ValueError("memory must be [B,M,scene_dim]")
        B=memory.shape[0]
        scene=(self.scene_content+self.position_embedding(self.plane_coordinates.to(device=memory.device,dtype=memory.dtype))).reshape(1,-1,self.cfg.scene_dim).expand(B,-1,-1)
        fused=self.decoder(scene,memory,memory_key_padding_mask=memory_padding_mask)
        p=self.cfg.plane_size
        planes=fused.reshape(B,3,p,p,self.cfg.scene_dim).permute(0,1,4,2,3).contiguous()
        up=self.upsample(planes.reshape(B*3,self.cfg.scene_dim,p,p))
        return up.reshape(B,3,self.cfg.plane_channels,up.shape[-2],up.shape[-1])

class ContinuousSurfaceFieldV1(nn.Module):
    """Query the shared scene representation at arbitrary canonical 3D points."""
    def __init__(self,cfg):
        super().__init__()
        cin=cfg.plane_channels*3
        self.backbone=nn.Sequential(nn.LayerNorm(cin),nn.Linear(cin,cfg.field_hidden_dim),nn.SiLU(),nn.Linear(cfg.field_hidden_dim,cfg.field_hidden_dim),nn.SiLU())
        self.distance_head=nn.Linear(cfg.field_hidden_dim,1)
        self.uncertainty_head=nn.Linear(cfg.field_hidden_dim,1)
    def sample_latent(self,triplanes,points):
        if triplanes.ndim!=5 or triplanes.shape[1]!=3: raise ValueError("triplanes must be [B,3,C,H,W]")
        if points.ndim!=3 or points.shape[0]!=triplanes.shape[0] or points.shape[-1]!=3: raise ValueError("points must be [B,N,3]")
        if not torch.isfinite(points).all() or (points.abs()>1.00001).any(): raise ValueError("points must be finite canonical [-1,1]^3")
        B,_,C,H,W=triplanes.shape
        x,y,z=points.unbind(-1)
        coords=torch.stack([torch.stack([x,y],-1),torch.stack([x,z],-1),torch.stack([y,z],-1)],1)
        samp=F.grid_sample(triplanes.reshape(B*3,C,H,W),coords.reshape(B*3,points.shape[1],1,2),mode="bilinear",align_corners=False,padding_mode="border")
        samp=samp.squeeze(-1).reshape(B,3,C,points.shape[1]).permute(0,3,1,2)
        return samp.reshape(B,points.shape[1],3*C)
    def forward(self,triplanes,points):
        latent=self.sample_latent(triplanes,points)
        h=self.backbone(latent)
        # UDF is deliberate: RiggingSurfaceIR needs a surface locus, not a watertight solid.
        return ContinuousSurfaceFieldOutputV1(F.softplus(self.distance_head(h)),self.uncertainty_head(h).clamp(-8,4),h)

class CleanroomGeometryFieldV1(nn.Module):
    """EXPERIMENTAL_ONLY: eight-view camera-aware fusion into a continuous surface field."""
    def __init__(self,cfg):
        super().__init__()
        self.view_tokens=CameraAwareViewTokenBankV1(cfg.image_token_dim,cfg.scene_dim,cfg.camera_embed_dim)
        self.scene=SharedTriplaneSceneFusionV1(cfg)
        self.field=ContinuousSurfaceFieldV1(cfg)
    def forward(self,image_tokens,token_xy,camera_features,query_points=None,token_valid=None):
        memory,mask=self.view_tokens(image_tokens,token_xy,camera_features,token_valid)
        triplanes=self.scene(memory,mask)
        field=None if query_points is None else self.field(triplanes,query_points)
        return CleanroomGeometryFieldOutputV1(triplanes,field)
