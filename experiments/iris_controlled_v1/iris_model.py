from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNormAct(nn.Module):
    def __init__(self,cin,cout,k=3,s=1,groups=1):
        super().__init__(); self.conv=nn.Conv2d(cin,cout,k,s,k//2,groups=groups,bias=False); self.norm=nn.GroupNorm(min(16,cout),cout); self.act=nn.GELU()
    def forward(self,x): return self.act(self.norm(self.conv(x)))
class ResidualBlock(nn.Module):
    def __init__(self,c):
        super().__init__(); self.dw=ConvNormAct(c,c,5,1,groups=c); self.pw1=nn.Conv2d(c,4*c,1); self.pw2=nn.Conv2d(4*c,c,1); self.norm=nn.GroupNorm(min(16,c),c)
    def forward(self,x):
        y=self.dw(x); y=self.pw2(F.gelu(self.pw1(y))); return x+self.norm(y)
class SharedImageEncoder(nn.Module):
    def __init__(self,in_ch=4,widths=(48,96,160,256)):
        super().__init__(); c2,c4,c8,c16=widths; self.s2=nn.Sequential(ConvNormAct(in_ch,c2,5,2),ResidualBlock(c2),ResidualBlock(c2)); self.s4=nn.Sequential(ConvNormAct(c2,c4,3,2),ResidualBlock(c4),ResidualBlock(c4)); self.s8=nn.Sequential(ConvNormAct(c4,c8,3,2),ResidualBlock(c8),ResidualBlock(c8)); self.s16=nn.Sequential(ConvNormAct(c8,c16,3,2),ResidualBlock(c16),ResidualBlock(c16))
    def forward(self,x):
        f2=self.s2(x); f4=self.s4(f2); f8=self.s8(f4); f16=self.s16(f8); return f2,f4,f8,f16
class AxialWithinViewReasoning(nn.Module):
    def __init__(self,c=256,heads=8,depth=1):
        super().__init__(); lr=nn.TransformerEncoderLayer(c,heads,2*c,dropout=0,batch_first=True,norm_first=True,activation='gelu'); lc=nn.TransformerEncoderLayer(c,heads,2*c,dropout=0,batch_first=True,norm_first=True,activation='gelu'); self.row=nn.TransformerEncoder(lr,num_layers=depth); self.col=nn.TransformerEncoder(lc,num_layers=depth); self.norm=nn.LayerNorm(c)
    def forward(self,x):
        N,C,H,W=x.shape; z=x.permute(0,2,3,1).contiguous(); r=self.row(z.reshape(N*H,W,C)).reshape(N,H,W,C); c=self.col(r.permute(0,2,1,3).reshape(N*W,H,C)); c=c.reshape(N,W,H,C).permute(0,2,1,3); return self.norm(c).permute(0,3,1,2).contiguous()
class RowWiseMultiViewFusion(nn.Module):
    def __init__(self,c=256,views=8,max_w=32,layers=2,heads=8):
        super().__init__(); self.views=views; self.c=c; self.max_w=max_w; self.x_pos=nn.Parameter(torch.zeros(1,1,1,max_w,c)); nn.init.trunc_normal_(self.x_pos,std=.02); self.view_mlp=nn.Sequential(nn.Linear(2,c),nn.GELU(),nn.Linear(c,c)); layer=nn.TransformerEncoderLayer(c,heads,2*c,dropout=0,batch_first=True,norm_first=True,activation='gelu'); self.tr=nn.TransformerEncoder(layer,num_layers=layers); self.out_norm=nn.LayerNorm(c)
    def forward(self,f,yaw_deg):
        B,V,C,H,W=f.shape
        if V!=self.views or W>self.max_w: raise ValueError(f'expected V={self.views}, W<={self.max_w}; got {V},{W}')
        if yaw_deg.ndim==1: yaw_deg=yaw_deg[None].expand(B,-1)
        t=torch.deg2rad(yaw_deg.float()); ve=self.view_mlp(torch.stack([torch.sin(t),torch.cos(t)],-1)); z=f.permute(0,3,1,4,2); z=z+ve[:,None,:,None,:]+self.x_pos[:,:,:,:W,:]; z=z.reshape(B*H,V*W,C); z=self.out_norm(self.tr(z)); return z.reshape(B,H,V,W,C).permute(0,2,4,1,3).contiguous()
class FuseBlock(nn.Module):
    def __init__(self,cin,skip,cout): super().__init__(); self.proj=ConvNormAct(cin+skip,cout,3,1); self.res=ResidualBlock(cout)
    def forward(self,x,skip): x=F.interpolate(x,size=skip.shape[-2:],mode='bilinear',align_corners=False); return self.res(self.proj(torch.cat([x,skip],1)))
class IRISControlledV1(nn.Module):
    def __init__(self,views=8,coarse_dim=64,fine_dim=32):
        super().__init__(); self.views=views; self.encoder=SharedImageEncoder(); self.within=AxialWithinViewReasoning(256,8,1); self.cross=RowWiseMultiViewFusion(256,views,32,2,8); self.d8=FuseBlock(256,160,160); self.d4=FuseBlock(160,96,96); self.d2=FuseBlock(96,48,64); self.p_head=nn.Conv2d(64,3,1); self.n_head=nn.Conv2d(64,3,1); self.u_head=nn.Conv2d(64,1,1); self.fine_head=nn.Conv2d(64,fine_dim,1); self.coarse_head=nn.Conv2d(160,coarse_dim,1)
    def forward(self,images,yaw_deg):
        B,V,C,H,W=images.shape
        if V!=self.views: raise ValueError(f'expected {self.views} views, got {V}')
        x=images.reshape(B*V,C,H,W); f2,f4,f8,f16=self.encoder(x); f16=self.within(f16).reshape(B,V,256,f16.shape[-2],f16.shape[-1]); fused=self.cross(f16,yaw_deg).reshape(B*V,256,f16.shape[-2],f16.shape[-1]); y8=self.d8(fused,f8); zc=F.normalize(self.coarse_head(y8),dim=1,eps=1e-8); y4=self.d4(y8,f4); y2=self.d2(y4,f2); p=.60*torch.tanh(self.p_head(y2)); n=F.normalize(self.n_head(y2),dim=1,eps=1e-8); log_sigma=torch.clamp(self.u_head(y2),-6,3); zf=F.normalize(self.fine_head(y2),dim=1,eps=1e-8)
        def rv(t): return t.reshape(B,V,*t.shape[1:])
        return {'P':rv(p),'N':rv(n),'log_sigma':rv(log_sigma),'Z_fine':rv(zf),'Z_coarse':rv(zc)}
def count_parameters(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)
