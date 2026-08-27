import torch
from realsas_mapanything_ortho.config import OrthoConfig
from realsas_mapanything_ortho.camera import point_from_depth, normals_from_point_field
from realsas_mapanything_ortho.losses import geometry_loss


def test_exact_origin_geometry_point_and_depth_zero():
    cfg=OrthoConfig(native_size=32,output_size=16,w_normal=0.0,w_risk=0.0);B,H,W=1,16,16;depth=torch.zeros(B,8,1,H,W);P=point_from_depth(depth);N=normals_from_point_field(P);Ptrue=torch.zeros(B,1,3);Ntrue=torch.tensor([[[0.,0.,1.]]]);xyv=torch.full((B,8,1,2),.5);V=torch.ones(B,8,1,dtype=torch.bool);out={'point':P,'normal':N,'depth':depth,'log_sigma':torch.zeros_like(depth)};target={'P':Ptrue,'N':Ntrue,'XY01':xyv,'V':V};loss,parts=geometry_loss(out,target,cfg);assert torch.isfinite(loss);assert float(parts['P'])<1e-5;assert float(parts['depth'])<1e-5;assert float(parts['consistency'])<1e-5
