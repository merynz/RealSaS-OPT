from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
import math
import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class SEESConfig:
    image_size: int = 128
    base_dim: int = 64
    token_dim: int = 192
    descriptor_dim: int = 64
    transformer_depth: int = 4
    transformer_heads: int = 6
    mlp_ratio: float = 3.0
    dropout: float = 0.0
    camera_residual_enabled: bool = False
    input_channels: int = 4
    max_delta: float = 0.35
    # N1D correspondence treatment. These are fixed operator hyperparameters,
    # not learned authority and add no state-dict entries.
    correspondence_radius_f4: int = 5
    correspondence_temperature: float = 0.03
    correspondence_position_lambda: float = 0.02
    correspondence_gain_threshold: float = 0.10
    correspondence_gain_width: float = 0.10


def _gn(c: int, affine: bool = True) -> nn.GroupNorm:
    groups = min(8, c)
    while c % groups:
        groups -= 1
    return nn.GroupNorm(groups, c, affine=affine)


class ResidualConv(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False)
        self.norm1 = _gn(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.norm2 = _gn(cout)
        self.skip = nn.Identity() if (cin == cout and stride == 1) else nn.Conv2d(cin, cout, 1, stride=stride, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.gelu(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        return F.gelu(y + self.skip(x))


class SharedImageEncoder(nn.Module):
    def __init__(self, base: int, token_dim: int, input_channels: int = 4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, base, 5, stride=2, padding=2, bias=False), _gn(base), nn.GELU(),
            ResidualConv(base, base, 1),
        )
        self.s4 = ResidualConv(base, base * 2, 2)
        self.s8 = ResidualConv(base * 2, base * 3, 2)
        self.s16 = ResidualConv(base * 3, token_dim, 2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem(x); f4 = self.s4(x); f8 = self.s8(f4); f16 = self.s16(f8)
        return f4, f8, f16


def canonical_orthographic_camera_features(height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    yy = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    xx = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    y, x = torch.meshgrid(yy, xx, indexing='ij'); rows=[]
    up = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype)
    for deg in range(0,360,45):
        t=math.radians(float(deg)); radial=torch.tensor([math.sin(t),0.0,math.cos(t)],device=device,dtype=dtype)
        direction=-radial; right=torch.cross(direction,up,dim=0); right=right/right.norm().clamp_min(1e-8)
        plane=x[...,None]*right+y[...,None]*up; d=direction.view(1,1,3).expand(height,width,3)
        rows.append(torch.cat([plane,d],dim=-1).permute(2,0,1))
    return torch.stack(rows,dim=0)


class GlobalMultiViewFusion(nn.Module):
    def __init__(self, dim:int, depth:int, heads:int, mlp_ratio:float, dropout:float):
        super().__init__()
        layer=nn.TransformerEncoderLayer(d_model=dim,nhead=heads,dim_feedforward=int(dim*mlp_ratio),dropout=dropout,activation='gelu',batch_first=True,norm_first=True)
        self.encoder=nn.TransformerEncoder(layer,num_layers=depth,enable_nested_tensor=False)
        self.view_embed=nn.Embedding(8,dim); self.pose_embed=nn.Embedding(2,dim)
        self.camera_embed=nn.Sequential(nn.Linear(6,dim),nn.GELU(),nn.Linear(dim,dim)); self.norm=nn.LayerNorm(dim)
    def forward(self,f16:torch.Tensor,cam6:torch.Tensor)->torch.Tensor:
        B,P,V,C,h,w=f16.shape; tokens=f16.permute(0,1,2,4,5,3).reshape(B,P*V*h*w,C)
        vi=torch.arange(V,device=f16.device).view(1,1,V,1,1).expand(B,P,V,h,w).reshape(B,-1)
        pi=torch.arange(P,device=f16.device).view(1,P,1,1,1).expand(B,P,V,h,w).reshape(B,-1)
        ce=self.camera_embed(cam6).view(B,P,V,1,1,C).expand(B,P,V,h,w,C).reshape(B,-1,C)
        tokens=tokens+self.view_embed(vi)+self.pose_embed(pi)+ce
        tokens=self.norm(self.encoder(tokens))
        return tokens.view(B,P,V,h,w,C).permute(0,1,2,5,3,4).contiguous()


class DenseFusionDecoder(nn.Module):
    def __init__(self,base:int,token_dim:int,out_dim:int=128):
        super().__init__(); c4,c8=base*2,base*3
        self.g16=nn.Conv2d(token_dim,out_dim,1); self.p8=nn.Conv2d(c8,out_dim,1); self.p4=nn.Conv2d(c4,out_dim,1)
        self.f8=ResidualConv(out_dim*2,out_dim); self.f4=ResidualConv(out_dim*2,out_dim)
        self.final=nn.Sequential(ResidualConv(out_dim,out_dim),nn.Conv2d(out_dim,out_dim,3,padding=1),nn.GELU())
    def forward(self,f4,f8,g16):
        x=self.g16(g16); x=F.interpolate(x,size=f8.shape[-2:],mode='bilinear',align_corners=False)
        x=self.f8(torch.cat([x,self.p8(f8)],dim=1)); x=F.interpolate(x,size=f4.shape[-2:],mode='bilinear',align_corners=False)
        x=self.f4(torch.cat([x,self.p4(f4)],dim=1)); return self.final(x)


class AntisymmetricDifferentialDecoder(nn.Module):
    """Odd response decoder. phi(d)-phi(-d) keeps sign reversal exact for a supplied differential field."""
    def __init__(self,base:int,token_dim:int,out_dim:int=128,max_delta:float=.35):
        super().__init__(); self.core=DenseFusionDecoder(base,token_dim,out_dim); self.point=nn.Conv2d(out_dim,3,1,bias=False); self.max_delta=float(max_delta)
    def _phi(self,d4,d8,d16): return self.core(d4,d8,d16)
    def forward(self,d4,d8,d16):
        odd=0.5*(self._phi(d4,d8,d16)-self._phi(-d4,-d8,-d16))
        return self.max_delta*torch.tanh(self.point(odd))


class TimeConditionedGeometryHead(nn.Module):
    def __init__(self,feat_dim:int):
        super().__init__(); self.time_embed=nn.Embedding(2,feat_dim)
        self.mod=nn.Sequential(nn.Linear(feat_dim,feat_dim*2),nn.GELU(),nn.Linear(feat_dim*2,feat_dim*2))
        self.point=nn.Conv2d(feat_dim,3,1); self.normal=nn.Conv2d(feat_dim,3,1); self.log_sigma=nn.Conv2d(feat_dim,1,1); self.visibility=nn.Conv2d(feat_dim,1,1)
    def forward(self,x,target_time:int)->Dict[str,torch.Tensor]:
        B,C,H,W=x.shape; e=self.time_embed(torch.full((B,),target_time,device=x.device,dtype=torch.long)); scale,bias=self.mod(e).chunk(2,-1)
        y=x*(1.0+0.1*torch.tanh(scale)[:,:,None,None])+bias[:,:,None,None]
        p=.55*torch.tanh(self.point(y)); n=F.normalize(self.normal(y),dim=1,eps=1e-6); ls=self.log_sigma(y).clamp(-6,2)
        return {'point':p,'normal':n,'log_sigma':ls,'visibility_logit':self.visibility(y)}


class DescriptorHead(nn.Module):
    def __init__(self,feat_dim:int,descriptor_dim:int): super().__init__(); self.desc=nn.Conv2d(feat_dim,descriptor_dim,1); self.conf=nn.Conv2d(feat_dim,1,1)
    def forward(self,x): return {'descriptor':F.normalize(self.desc(x),dim=1,eps=1e-6),'descriptor_log_sigma':self.conf(x).clamp(-6,2)}


class CameraResidualHead(nn.Module):
    def __init__(self,token_dim:int): super().__init__(); self.net=nn.Sequential(nn.LayerNorm(token_dim),nn.Linear(token_dim,64),nn.GELU(),nn.Linear(64,4))
    def forward(self,global_map): return self.net(global_map.mean(dim=(-1,-2)))


def _local_soft_offset(query: torch.Tensor, key: torch.Tensor, radius: int, temperature: float, position_lambda: float = 0.02, gain_threshold: float = 0.10, gain_width: float = 0.10):
    """Local soft correspondence. Returns expected offset (dx,in feature cells), entropy, peak, edge mass and gate."""
    B,C,H,W=query.shape; r=int(radius)
    offsets=[]; slices=[]
    for dy in range(-r,r+1):
        for dx in range(-r,r+1):
            xt=torch.roll(torch.roll(key,dy,dim=2),dx,dim=3)
            offsets.append((dx,dy)); slices.append(xt)
    keys}=torch.stack(slices,dim=1); off=torch.tensor(offsets,device=query.device,dtype=query.dtype)
    with torch.autocast(device_type=query.device.type,enabled=False):
        q=F.normalize(query.float(),dim=1,eps=1e-6); k=F.normalize(keys.movedim(2,4).float(),dim=-1,eps=1e-6)
        cosine=(q[.,None,.,.].movedim(1,4)*k).sum(-1)
        off2off*; dist2=(off[:,0]**d); logits=(cosine-float(position_lambda)*dist2)/max(float(temperature),1e-4)
        logits=logits.masked_fill(~valid,-1e4)
        w=F.softmax(logits,dim=1)
        expected=torch.einsum('bkn,kd->bdn',w,off).view(B,2,H,W)
        # Evidence gate: do not invent a warp when the local search cannot beat the zero-offset match by a material margin.
        center=cosine[:,K//2].view(B,1,H,W)
        best=cosine.masked_fill(~valid,-2.0).max(1).values.view(B,1,H,W)
        gain_gate=((best-center-float(gain_threshold))/max(float(gain_width),1e-4)).clamp(0.0,1.0)
        expected=expected*gain_gate
        entropy=(-(w.clamp_min(1e-12)*w.clamp_min(1e-12).log()).sum(1)/math.log(max(K,2))).view(B,1,H,W)
        peak=w.max(1).values.view(B,1,H,W)
        edge=((off[:,0].abs()==r)|(off[:,1].abs()==r)).float().view(1,K,1)
        edge_mass=(w*edge).sum(1).view(B,1,H,W)
    return expected,entropy,peak,edge_mass,gain_gate


def _resize_offset(offset: torch.Tensor, hw: tuple[int,int]) -> torch.Tensor:
    H0,W0=offset.shape[-2:]; H,W=hw
    if (H,W)==(H0,W0): return offset
    y=F.interpolate(offset,size=(H,W),mode='bilinear',align_corners=True)
    sx=(W-1)/max(W0-1,1); sy=(H-1)/max(H0-1,1)
    return torch.stack([y[:,0]*sx,y[:,1]*sy],dim=1)


def _warp_by_offset(value: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
    """Sample value at query-grid + offset. offset is in value-grid cell units [B,2,H,W]."""
    B,C,H,W=value.shape
    if offset.shape != (B,2,H,W): raise ValueError((value.shape,offset.shape))
    with torch.autocast(device_type=value.device.type, enabled=False):
        yy=torch.linspace(-1,1,H,device=value.device,dtype=torch.float32); xx=torch.linspace(-1,1,W,device=value.device,dtype=torch.float32)
        y,x=torch.meshgrid(yy,xx,indexing='ij'); grid=torch.stack([x,y],dim=-1).view(1,H,W,2).expand(B,-1,-1,-1).clone()
        grid[...,0]+=offset[:,0].float()*(2.0/max(W-1,1)); grid[...,1]+=offset[:,1].float()*(2.0/max(H-1,1))
        out=F.grid_sample(value.float(),grid,mode='bilinear',padding_mode='border',align_corners=True)
    return out


class IRISSEESN1(nn.Module):
    """N1D implementation under stable public class name: N1C weights, correspondence-aware differential forward."""
    def __init__(self,cfg:SEESConfig=SEESConfig()):
        super().__init__(); self.cfg=cfg
        self.encoder=SharedImageEncoder(cfg.base_dim,cfg.token_dim,cfg.input_channels)
        self.fusion=GlobalMultiViewFusion(cfg.token_dim,cfg.transformer_depth,cfg.transformer_heads,cfg.mlp_ratio,cfg.dropout)
        self.camera_token=nn.Sequential(nn.Linear(6,cfg.token_dim),nn.GELU(),nn.Linear(cfg.token_dim,cfg.token_dim))
        self.decoder=DenseFusionDecoder(cfg.base_dim,cfg.token_dim,out_dim=128); self.geometry=TimeConditionedGeometryHead(128); self.descriptor=DescriptorHead(128,cfg.descriptor_dim)
        self.differential=AntisymmetricDifferentialDecoder(cfg.base_dim,cfg.token_dim,out_dim=128,max_delta=cfg.max_delta)
        self.camera_residual=CameraResidualHead(cfg.token_dim)

    def forward(self,images:torch.Tensor)->Dict[str,torch.Tensor]:
        if images.ndim!=6 or images.shape[1:3]!=(2,8) or images.shape[3]!=self.cfg.input_channels: raise ValueError(f'expected [B,2,8,{self.cfg.input_channels},H,W], got {tuple(images.shape)}')
        B,P,V,C,H,W=images.shape; x=images.reshape(B*P*V,C,H,W); f4,f8,f16=self.encoder(x)
        h4,w4=f4.shape[-2:]; h8,w8=f8.shape[-2:]; h16,w16=f16.shape[-2:]
        f4=f4.view(B,P,V,-1,h4,w4); f8=f8.view(B,P,V,-1,h8,w8); f16=f16.view(B,P,V,-1,h16,w16)
        cam=canonical_orthographic_camera_features(1,1,images.device,images.dtype)[:,:,0,0]; cam6=cam.view(1,1,8,6).expand(B,2,8,6)
        global16=self.fusion(f16,cam6); residual=self.camera_residual(global16)
        if self.cfg.camera_residual_enabled: raise RuntimeError('N1D forbids camera residual authority')
        dense_views=[]
        for p in range(2):
            row=[]
            for v in range(8):
                g=global16[:,p,v]+self.camera_token(cam6[:,p,v])[:,:,None,None]
                row.append(self.decoder(f4[:,p,v],f8[:,p,v],g))
            dense_views.append(torch.stack(row,dim=1))
        dense=torch.stack(dense_views,dim=1); _,_,_,Fd,Hd,Wd=dense.shape; flat=dense.reshape(B*P*V,Fd,Hd,Wd)
        desc=self.descriptor(flat)
        out={'dense_feature':dense,'descriptor':desc['descriptor'].view(B,P,V,-1,Hd,Wd),'descriptor_log_sigma':desc['descriptor_log_sigma'].view(B,P,V,1,Hd,Wd),'camera_residual_diagnostic':residual}
        raw={}
        for t,name in [(0,'A'),(1,'B')]:
            g=self.geometry(flat,t)
            for k,val in g.items(): raw[f'{k}_{name}']=val.view(B,P,V,val.shape[1],Hd,Wd)
        out.update(raw)

        # N1D: pose-neutral local soft correspondence at f4, propagated to coarser scales.
        # Directed residuals are paired: swapping A/B maps dA -> -dB and dB -> -dA.
        deltaA=[]; deltaB=[]; entA=[]; entB=[]; peakA=[]; peakB=[]; edgeA=[]; edgeB=[]; gateA=[]; gateB=[]; offA=[]; offB=[]
        r=self.cfg.correspondence_radius_f4; temp=self.cfg.correspondence_temperature; pl=self.cfg.correspondence_position_lambda; gt=self.cfg.correspondence_gain_threshold; gw=self.cfg.correspondence_gain_width
        for v in range(8):
            oa,ea,pa,eda,ga=_local_soft_offset(f4[:,0,v],f4[:,1,v],r,temp,pl,gt,gw) # A-grid -> B-grid
            ob,eb,pb,edb,gb=_local_soft_offset(f4[:,1,v],f4[:,0,v],r,temp,pl,gt,gw) # B-grid -> A-grid
            b4a=_warp_by_offset(f4[:,1,v],oa); a4b=_warp_by_offset(f4[:,0,v],ob)
            oa8=_resize_offset(oa,(h8,w8)); ob8=_resize_offset(ob,(h8,w8)); oa16=_resize_offset(oa,(h16,w16)); ob16=_resize_offset(ob,(h16,w16))
            b8a=_warp_by_offset(f8[:,1,v],oa8); a8b=_warp_by_offset(f8[:,0,v],ob8)
            b16a=_warp_by_offset(f16[:,1,v],oa16); a16b=_warp_by_offset(f16[:,0,v],ob16)
            dA4=b4a-f4[:,0,v].float(); dA8=b8a-f8[:,0,v].float(); dA16=b16a-f16[:,0,v].float()
            dB4=f4[:,1,v].float()-a4b; dB8=f8[:,1,v].float()-a8b; dB16=f16[:,1,v].float()-a16b
            deltaA.append(self.differential(dA4,dA8,dA16)); deltaB.append(self.differential(dB4,dB8,dB16))
            entA.append(ea); entB.append(eb); peakA.append(pa); peakB.append(pb); edgeA.append(eda); edgeB.append(edb); gateA.append(ga); gateB.append(gb); offA.append(oa); offB.append(ob)
        deltaA=torch.stack(deltaA,dim=1); deltaB=torch.stack(deltaB,dim=1)
        out['delta_point_map_srcA']=deltaA; out['delta_point_map_srcB']=deltaB
        out['delta_point_map']=0.5*(deltaA+deltaB) # pair-authority diagnostic on the common image lattice
        out['transport_offset_srcA']=torch.stack(offA,dim=1); out['transport_offset_srcB']=torch.stack(offB,dim=1)
        out['transport_entropy_srcA']=torch.stack(entA,dim=1); out['transport_entropy_srcB']=torch.stack(entB,dim=1)
        out['transport_peak_srcA']=torch.stack(peakA,dim=1); out['transport_peak_srcB']=torch.stack(peakB,dim=1)
        out['transport_edge_mass_srcA']=torch.stack(edgeA,dim=1); out['transport_edge_mass_srcB']=torch.stack(edgeB,dim=1)
        out['transport_gate_srcA']=torch.stack(gateA,dim=1); out['transport_gate_srcB']=torch.stack(gateB,dim=1)

        # Cross-time positions remain algebraically coupled to the source-specific transported response.
        pA=raw['point_A'].clone(); pB=raw['point_B'].clone(); baseA=raw['point_A'][:,0]; baseB=raw['point_B'][:,1]
        pA[:,0]=baseA; pB[:,0]=baseA+deltaA
        pB[:,1]=baseB; pA[:,1]=baseB-deltaB
        out['point_A']=pA; out['point_B']=pB
        return out
