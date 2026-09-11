from __future__ import annotations
from dataclasses import replace
import torch

from models.arachne.v4.arachne_candidate_v4 import ArachneA1ConfigV4, ArachneA1V4
from models.arachne.v4.articulated_probe_v1 import build_articulated_probe_transforms


def _tiny_model():
    c=replace(ArachneA1ConfigV4(),model_dim=64,attention_heads=4,surface_graph_layers=1,surface_transformer_layers=1,skeleton_graph_layers=1,token_self_layers=1,bidirectional_fusion_rounds=1,ffn_ratio=2)
    return ArachneA1V4(c).eval()


def _batch():
    torch.manual_seed(3); B,N,J,E=1,18,5,24
    parent=torch.tensor([[-1,0,1,1,0]])
    return dict(surface_positions_normalized=torch.randn(B,N,3),surface_normals=torch.randn(B,N,3),surface_normal_valid=torch.ones(B,N,dtype=torch.bool),surface_support_views=torch.rand(B,N,8)>.4,surface_raster_xy=torch.randn(B,N,8,2),surface_raster_valid=torch.rand(B,N,8)>.4,surface_observed=torch.ones(B,N,dtype=torch.bool),surface_completed=torch.zeros(B,N,dtype=torch.bool),surface_mask=torch.ones(B,N,dtype=torch.bool),edge_index=torch.randint(0,N,(B,E,2)),edge_features=torch.randn(B,E,4),edge_mask=torch.ones(B,E,dtype=torch.bool),joint_positions_normalized=torch.randn(B,J,3),joint_mask=torch.ones(B,J,dtype=torch.bool),parent_indices=parent,root_mask=torch.tensor([[1,0,0,0,0]],dtype=torch.bool),deform_root_mask=torch.tensor([[1,0,0,0,0]],dtype=torch.bool),joint_depth_normalized=torch.tensor([[0.,.3,.6,.6,.3]]),support_anchor_matrix=torch.rand(B,J,N)>.75,pair_geometry=torch.randn(B,N,J,10),pair_mask=torch.ones(B,N,J,dtype=torch.bool),view_yaw_fourier=torch.randn(B,8,4))


def test_surface_joint_view_permutation_contracts():
    m=_tiny_model(); x=_batch()
    with torch.no_grad(): base=m(**x).field_tokens
    N=x['surface_mask'].shape[1]; p=torch.arange(N-1,-1,-1); inv=torch.empty_like(p); inv[p]=torch.arange(N)
    s=dict(x)
    for n in ('surface_positions_normalized','surface_normals','surface_normal_valid','surface_support_views','surface_raster_xy','surface_raster_valid','surface_observed','surface_completed','surface_mask'): s[n]=x[n][:,p]
    s['edge_index']=inv[x['edge_index']]; s['support_anchor_matrix']=x['support_anchor_matrix'][:,:,p]; s['pair_geometry']=x['pair_geometry'][:,p]; s['pair_mask']=x['pair_mask'][:,p]
    with torch.no_grad(): assert torch.allclose(base,m(**s).field_tokens,atol=2e-5,rtol=0)
    J=x['joint_mask'].shape[1]; q=torch.arange(J-1,-1,-1); qi=torch.empty_like(q); qi[q]=torch.arange(J); j=dict(x)
    for n in ('joint_positions_normalized','joint_mask','root_mask','deform_root_mask','joint_depth_normalized'): j[n]=x[n][:,q]
    op=x['parent_indices'][:,q]; j['parent_indices']=torch.where(op>=0,qi[op.clamp_min(0)],op); j['support_anchor_matrix']=x['support_anchor_matrix'][:,q]; j['pair_geometry']=x['pair_geometry'][:,:,q]; j['pair_mask']=x['pair_mask'][:,:,q]
    with torch.no_grad(): assert torch.allclose(base[:,q],m(**j).field_tokens,atol=2e-5,rtol=0)
    vp=torch.tensor([3,0,7,1,6,2,5,4]); v=dict(x)
    for n in ('surface_support_views','surface_raster_xy','surface_raster_valid'): v[n]=x[n][:,:,vp]
    v['view_yaw_fourier']=x['view_yaw_fourier'][:,vp]
    with torch.no_grad(): assert torch.allclose(base,m(**v).field_tokens,atol=2e-5,rtol=0)


def test_articulated_probe_joint_permutation_equivariance():
    p=torch.tensor([[[0.,0.,0.],[1.,0.,0.],[2.,0.,0.],[1.,1.,0.]]]); par=torch.tensor([[-1,0,1,0]]); mask=torch.ones(1,4,dtype=torch.bool)
    a=build_articulated_probe_transforms(p,par,mask); q=torch.tensor([2,0,3,1]); qi=torch.empty_like(q); qi[q]=torch.arange(4); op=par[:,q]; par2=torch.where(op>=0,qi[op.clamp_min(0)],op)
    b=build_articulated_probe_transforms(p[:,q],par2,mask[:,q]); assert torch.allclose(a[:,:,q],b,atol=1e-7,rtol=0)
