import torch
from realsas_mapanything_ortho.camera import orthographic_basis, dense_ray_field, point_from_depth, normals_from_point_field, forward_depth_from_points


def test_basis_orthonormal():
    r,f,u=orthographic_basis(dtype=torch.float64)
    assert torch.allclose((r*f).sum(-1),torch.zeros(8,dtype=torch.float64),atol=1e-12)
    assert torch.allclose((r*u).sum(-1),torch.zeros(8,dtype=torch.float64),atol=1e-12)
    assert torch.allclose((f*u).sum(-1),torch.zeros(8,dtype=torch.float64),atol=1e-12)
    assert torch.allclose(r.norm(dim=-1),torch.ones(8,dtype=torch.float64),atol=1e-12)


def test_depth_roundtrip():
    d=torch.linspace(-.4,.4,8*16*16).view(1,8,1,16,16);P=point_from_depth(d);O,F=dense_ray_field(1,16,16,P.device,P.dtype);got=((P-O)*F).sum(2,keepdim=True);assert torch.allclose(got,d,atol=1e-6)


def test_front_plane_normal_points_toward_camera():
    d=torch.zeros(1,8,1,16,16);P=point_from_depth(d);N=normals_from_point_field(P);_,f,_=orthographic_basis(P.device,P.dtype);dots=(N*f[None,:,:,None,None]).sum(2);assert float(dots.mean())<-.99


def test_sparse_forward_depth():
    xy=torch.tensor([[[[.5,.5]],[[.5,.5]],[[.5,.5]],[[.5,.5]],[[.5,.5]],[[.5,.5]],[[.5,.5]],[[.5,.5]]]],dtype=torch.float32);P=torch.tensor([[[.1,.2,0.]]]);d=forward_depth_from_points(P,xy);_,f,_=orthographic_basis(P.device,P.dtype);expect=(P[:,None]*f[None,:,None]).sum(-1);assert torch.allclose(d,expect,atol=1e-6)
