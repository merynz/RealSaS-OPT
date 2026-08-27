from __future__ import annotations
from typing import Dict, Any, Iterable, List
import torch
from torch import nn
import torch.nn.functional as F

from .config import OrthoConfig
from .camera import point_from_depth, normals_from_point_field, orthographic_basis
from .preprocess import prepare_global_views, masked_standardize


def _gn(c: int) -> nn.GroupNorm:
    g = min(8, c)
    while c % g: g -= 1
    return nn.GroupNorm(g, c)


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False); self.n1 = _gn(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False); self.n2 = _gn(cout)
        self.skip = nn.Identity() if cin == cout and stride == 1 else nn.Conv2d(cin, cout, 1, stride=stride, bias=False)
    def forward(self, x):
        y = F.gelu(self.n1(self.c1(x))); y = self.n2(self.c2(y)); return F.gelu(y + self.skip(x))


class NativeDetailRefiner(nn.Module):
    def __init__(self, cfg: OrthoConfig):
        super().__init__(); b,m,d,f=cfg.detail_base_dim,cfg.detail_mid_dim,cfg.detail_deep_dim,cfg.detail_fused_dim
        self.stem=nn.Sequential(nn.Conv2d(4,b,5,stride=2,padding=2,bias=False),_gn(b),nn.GELU(),ConvBlock(b,b))
        self.s4=ConvBlock(b,m,2); self.s8=ConvBlock(m,d,2)
        self.coarse=nn.Sequential(nn.Conv2d(8,d,3,padding=1,bias=False),_gn(d),nn.GELU(),ConvBlock(d,d))
        self.fuse128=ConvBlock(d*2,f); self.up256=ConvBlock(f+m,f); self.up512=ConvBlock(f+b,f)
        self.final=nn.Sequential(ConvBlock(f,f),nn.Dropout2d(cfg.dropout) if cfg.dropout>0 else nn.Identity())
        self.depth_head=nn.Conv2d(f,1,1); self.risk_head=nn.Conv2d(f,1,1)
        nn.init.zeros_(self.depth_head.weight); nn.init.zeros_(self.depth_head.bias)
    def forward(self,native_rgba,coarse8,depth_limit):
        B,V=native_rgba.shape[:2]; x=native_rgba.reshape(B*V,4,*native_rgba.shape[-2:])
        f512=self.stem(x); f256=self.s4(f512); f128=self.s8(f256)
        c=coarse8.reshape(B*V,8,*coarse8.shape[-2:]); c=F.interpolate(c,size=f128.shape[-2:],mode="bilinear",align_corners=False); c=self.coarse(c)
        y=self.fuse128(torch.cat([f128,c],1)); y=F.interpolate(y,size=f256.shape[-2:],mode="bilinear",align_corners=False); y=self.up256(torch.cat([y,f256],1))
        y=F.interpolate(y,size=f512.shape[-2:],mode="bilinear",align_corners=False); y=self.up512(torch.cat([y,f512],1)); y=self.final(y)
        depth=float(depth_limit)*torch.tanh(self.depth_head(y)); log_sigma=self.risk_head(y).clamp(-6.0,2.0); H,W=depth.shape[-2:]
        return depth.view(B,V,1,H,W),log_sigma.view(B,V,1,H,W)


class MapAnythingOrthoIRIS(nn.Module):
    """Pretrained Apache-labelled MapAnything geometry -> exact RealSaS orthographic P."""
    def __init__(self,cfg:OrthoConfig=OrthoConfig(),base_model:nn.Module|None=None):
        super().__init__(); self.cfg=cfg
        if base_model is None:
            try: from mapanything.models import MapAnything
            except Exception as exc: raise RuntimeError("Pinned MapAnything must be installed before constructing the model") from exc
            base_model=MapAnything.from_pretrained(cfg.upstream_model_id,revision=cfg.upstream_model_revision)
        self.base=base_model; self.refiner=NativeDetailRefiner(cfg); self._force_known_camera_only_base()
        if cfg.enable_gradient_checkpointing: self._enable_checkpointing_best_effort()
        self.set_train_stage("adapter")

    def _force_known_camera_only_base(self):
        g=getattr(self.base,"geometric_input_config",None)
        if not isinstance(g,dict): raise RuntimeError("unexpected MapAnything API: geometric_input_config missing")
        g.update({"overall_prob":1.0,"dropout_prob":0.0,"ray_dirs_prob":0.0,"depth_prob":0.0,"cam_prob":1.0,"sparse_depth_prob":0.0,"depth_scale_norm_all_prob":1.0,"pose_scale_norm_all_prob":1.0})

    def _enable_checkpointing_best_effort(self):
        touched=0
        for module in self.base.modules():
            for attr in ("gradient_checkpointing","checkpoint_gradient"):
                if hasattr(module,attr):
                    try: setattr(module,attr,True); touched+=1
                    except Exception: pass
        self._checkpointing_flags_touched=touched

    @staticmethod
    def _set_module_trainable(module,value):
        if module is not None: module.requires_grad_(value)

    def set_train_stage(self,stage:str):
        stage=stage.lower()
        if stage not in {"adapter","geometry","full"}: raise ValueError(stage)
        self.base.requires_grad_(False); self.refiner.requires_grad_(True)
        if stage in {"geometry","full"}:
            for n in ("info_sharing","dense_head","dpt_feature_head","dpt_regressor_head","scale_head","fusion_norm_layer","cam_rot_encoder"):
                self._set_module_trainable(getattr(self.base,n,None),True)
            if hasattr(self.base,"scale_token"): self.base.scale_token.requires_grad_(True)
        if stage=="full": self._set_module_trainable(getattr(self.base,"encoder",None),True)
        self.train_stage=stage; return self

    def parameter_groups(self)->List[Dict[str,Any]]:
        seen=set(); groups=[]
        def add(name,params,lr):
            ps=[p for p in params if p.requires_grad and id(p) not in seen]
            for p in ps: seen.add(id(p))
            if ps: groups.append({"name":name,"params":ps,"lr":lr})
        add("refiner",self.refiner.parameters(),self.cfg.lr_refiner)
        dense=[getattr(self.base,n,None) for n in ("dense_head","dpt_feature_head","dpt_regressor_head","scale_head")]
        add("base_dense",(p for m in dense if m is not None for p in m.parameters()),self.cfg.lr_dense)
        info=[getattr(self.base,n,None) for n in ("info_sharing","fusion_norm_layer","cam_rot_encoder")]; ip=[p for m in info if m is not None for p in m.parameters()]
        if hasattr(self.base,"scale_token") and self.base.scale_token.requires_grad: ip.append(self.base.scale_token)
        add("base_info",ip,self.cfg.lr_info)
        enc=getattr(self.base,"encoder",None)
        if enc is not None: add("base_encoder",enc.parameters(),self.cfg.lr_encoder)
        leftovers=[p for p in self.parameters() if p.requires_grad and id(p) not in seen]
        if leftovers: raise RuntimeError(f"unassigned trainable parameter tensors: {len(leftovers)}")
        return groups

    def _coarse_features(self,preds,global_alpha):
        if len(preds)!=8: raise RuntimeError(f"MapAnything returned {len(preds)} views")
        for v,p in enumerate(preds):
            miss=[k for k in ("pts3d_cam","conf") if k not in p]
            if miss: raise RuntimeError(f"upstream output V{v} missing {miss}; expected Apache raydirs+depth+pose+confidence model")
        z=torch.stack([p["pts3d_cam"][...,2] for p in preds],1).unsqueeze(2); conf=torch.stack([p["conf"] for p in preds],1).unsqueeze(2)
        amb=[]
        for p in preds:
            if "non_ambiguous_mask_logits" in p: amb.append(p["non_ambiguous_mask_logits"])
            elif "non_ambiguous_mask" in p: amb.append(p["non_ambiguous_mask"].to(z.dtype)*8.0-4.0)
            else: amb.append(torch.zeros_like(p["conf"]))
        amb=torch.stack(amb,1).unsqueeze(2)
        alpha=F.interpolate(global_alpha.reshape(-1,1,*global_alpha.shape[-2:]),size=z.shape[-2:],mode="bilinear",align_corners=False).view_as(z)
        z_rel,_,_=masked_standardize(z.float(),alpha.float()); logc_rel,_,_=masked_standardize(torch.log(conf.float().clamp_min(1e-6)),alpha.float())
        B,V,_,H,W=z.shape; yy=(torch.arange(H,device=z.device,dtype=torch.float32)+.5)/H*2-1; xx=(torch.arange(W,device=z.device,dtype=torch.float32)+.5)/W*2-1; y,x=torch.meshgrid(yy,xx,indexing="ij")
        x=x.view(1,1,1,H,W).expand(B,V,-1,-1,-1); y=y.view(1,1,1,H,W).expand(B,V,-1,-1,-1)
        _,f,_=orthographic_basis(z.device,torch.float32); sy=f[:,0].view(1,V,1,1,1).expand(B,-1,-1,H,W); cy=f[:,1].view(1,V,1,1,1).expand(B,-1,-1,H,W)
        return torch.cat([z_rel,logc_rel,amb.float(),alpha.float(),x,y,sy,cy],2)

    def forward(self,native_rgba):
        if native_rgba.ndim!=5 or native_rgba.shape[1:3]!=(8,4): raise ValueError(f"expected [B,8,4,H,W], got {tuple(native_rgba.shape)}")
        if native_rgba.shape[-2:]!=(self.cfg.native_size,self.cfg.native_size): raise ValueError(f"native authority mismatch: expected {self.cfg.native_size}, got {native_rgba.shape[-2:]}")
        views,global_alpha=prepare_global_views(native_rgba,self.cfg.global_size)
        preds=self.base(views,memory_efficient_inference=False)
        coarse=self._coarse_features(preds,global_alpha).to(native_rgba.dtype)
        depth,log_sigma=self.refiner(native_rgba,coarse,self.cfg.depth_limit)
        if depth.shape[-2:]!=(self.cfg.output_size,self.cfg.output_size): raise RuntimeError((depth.shape,self.cfg.output_size))
        P=point_from_depth(depth); N=normals_from_point_field(P)
        support=F.interpolate(native_rgba[:,:,3:4].reshape(-1,1,self.cfg.native_size,self.cfg.native_size),size=depth.shape[-2:],mode="bilinear",align_corners=False).view(depth.shape[0],8,1,*depth.shape[-2:])
        return {"depth":depth,"point":P,"normal":N,"log_sigma":log_sigma,"support":support,"upstream_coarse":coarse}
