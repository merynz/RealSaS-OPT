from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "vf11_certified_adaptive"))

from range_engine_v1 import (  # noqa: E402
    Interval,
    center_direction,
    certify_cell,
    layernorm_bounds,
    silu_bounds,
    triplane_bounds,
)
from profile_v1 import profile_depths  # noqa: E402


def _sample(planes: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    b,_,c,h,w = planes.shape
    x,y,z = points.unbind(-1)
    grids = torch.stack((
        torch.stack((x,y),-1),
        torch.stack((x,z),-1),
        torch.stack((y,z),-1),
    ),1)
    out = F.grid_sample(
        planes.reshape(b*3,c,h,w),
        grids.reshape(b*3,points.shape[1],1,2),
        mode="bilinear", padding_mode="border", align_corners=False,
    )
    return out.squeeze(-1).reshape(b,3,c,points.shape[1]).permute(0,3,1,2).reshape(b,points.shape[1],3*c)


class TinyField(nn.Module):
    def __init__(self, width: int, hidden: int = 8):
        super().__init__()
        self.body = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width,hidden), nn.SiLU(),
            nn.Linear(hidden,hidden), nn.SiLU(),
        )
        self.sdf_head = nn.Linear(hidden,1)
        self.uncertainty_head = nn.Linear(hidden,1)

    def forward(self, planes, points):
        h = self.body(_sample(planes,points))
        return {"sdf":self.sdf_head(h).squeeze(-1),
                "log_uncertainty":self.uncertainty_head(h).squeeze(-1)}


def test_silu_bounds_contain_value_and_directional():
    lo = torch.tensor([-4.0,-1.5,-0.3,0.4],dtype=torch.float64)
    hi = torch.tensor([-2.0,0.7,2.8,4.0],dtype=torch.float64)
    dx = torch.tensor([0.7,-1.2,0.2,2.0],dtype=torch.float64)
    yr,dyr = silu_bounds(Interval(lo,hi),Interval.exact(dx))
    for t in torch.linspace(0,1,101,dtype=torch.float64):
        x = (lo*(1-t)+hi*t).requires_grad_(True)
        y = x*torch.sigmoid(x)
        dy = torch.autograd.grad(y,x,grad_outputs=dx)[0]
        assert torch.all(y.detach() >= yr.lo-1e-12)
        assert torch.all(y.detach() <= yr.hi+1e-12)
        assert torch.all(dy.detach() >= dyr.lo-1e-12)
        assert torch.all(dy.detach() <= dyr.hi+1e-12)


def test_layernorm_bounds_contain_random_jvp():
    torch.manual_seed(5)
    n=8
    layer=nn.LayerNorm(n).double()
    center=torch.linspace(-1.2,1.4,n,dtype=torch.float64)
    radius=torch.full((n,),0.08,dtype=torch.float64)
    dx=torch.linspace(-0.4,0.5,n,dtype=torch.float64)
    yr,dyr=layernorm_bounds(Interval(center-radius,center+radius),Interval.exact(dx),layer)
    g=torch.Generator().manual_seed(123)
    for _ in range(40):
        x=(center-radius+2*radius*torch.rand(n,generator=g,dtype=torch.float64)).requires_grad_(True)
        y=layer(x)
        _,dy=torch.autograd.functional.jvp(layer,(x,),(dx,),create_graph=False)
        assert torch.all(y.detach() >= yr.lo-1e-10)
        assert torch.all(y.detach() <= yr.hi+1e-10)
        assert torch.all(dy.detach() >= dyr.lo-1e-10)
        assert torch.all(dy.detach() <= dyr.hi+1e-10)


def test_triplane_bounds_contain_random_samples_and_directionals():
    torch.manual_seed(7)
    planes=torch.randn(1,3,2,6,6,dtype=torch.float64)
    lo=torch.tensor([-0.32,-0.18,-0.28],dtype=torch.float64)
    hi=torch.tensor([0.22,0.27,0.19],dtype=torch.float64)
    d=torch.tensor([0.4,-0.2,0.7],dtype=torch.float64)
    d=d/torch.linalg.vector_norm(d)
    fr,dfr=triplane_bounds(planes,lo,hi,d)
    g=torch.Generator().manual_seed(77)
    for _ in range(30):
        p=(lo+(hi-lo)*torch.rand(3,generator=g,dtype=torch.float64)).reshape(1,1,3).requires_grad_(True)
        feat=_sample(planes,p).reshape(-1)
        ders=[]
        for i in range(feat.numel()):
            grad=torch.autograd.grad(feat[i],p,retain_graph=i+1<feat.numel())[0].reshape(3)
            ders.append(torch.dot(grad,d))
        der=torch.stack(ders)
        assert torch.all(feat.detach() >= fr.lo-1e-10)
        assert torch.all(feat.detach() <= fr.hi+1e-10)
        assert torch.all(der.detach() >= dfr.lo-1e-9)
        assert torch.all(der.detach() <= dfr.hi+1e-9)


def test_no_sampled_crossing_is_claimed_empty():
    torch.manual_seed(11)
    c=3
    planes=torch.randn(1,3,c,8,8)
    field=TinyField(3*c,9).eval()
    lo=torch.tensor([-0.25,-0.20,-0.22])
    hi=torch.tensor([0.18,0.21,0.17])
    cert=certify_cell(field,planes,lo,hi)
    g=torch.Generator().manual_seed(111)
    vals=[]
    for _ in range(150):
        p=lo+(hi-lo)*torch.rand(3,generator=g)
        vals.append(float(field(planes,p.reshape(1,1,3))["sdf"].item()))
    assert all(v >= cert.field_lo-2e-6 for v in vals)
    assert all(v <= cert.field_hi+2e-6 for v in vals)
    if cert.state == "PROVEN_EMPTY":
        assert min(vals)>0 or max(vals)<0


def test_emitted_regular_direction_is_samplewise_one_signed():
    torch.manual_seed(19)
    c=2
    planes=torch.randn(1,3,c,7,7)
    field=TinyField(3*c,8).eval()
    lo=torch.tensor([-0.13,-0.11,-0.12])
    hi=torch.tensor([0.07,0.08,0.09])
    cert=certify_cell(field,planes,lo,hi)
    if cert.state != "PROVEN_REGULAR":
        return
    d,_=center_direction(field,planes,lo,hi)
    g=torch.Generator().manual_seed(191)
    rows=[]
    for _ in range(30):
        p=(lo+(hi-lo)*torch.rand(3,generator=g)).reshape(1,1,3).requires_grad_(True)
        sdf=field(planes,p)["sdf"].reshape(-1)[0]
        grad=torch.autograd.grad(sdf,p)[0].reshape(3)
        rows.append(float(torch.dot(grad,d)))
    assert min(rows)>-2e-5 or max(rows)<2e-5


def test_profiler_is_deterministic_and_research_only():
    torch.manual_seed(23)
    c=2
    planes=torch.randn(1,3,c,6,6)
    field=TinyField(3*c,7).eval()
    kwargs=dict(
        depths=[1,2],
        max_cells_per_depth=24,
        domain_lo=-0.75,
        domain_hi=0.75,
        seed=12345,
        logger=None,
    )
    a=profile_depths(field,planes,**kwargs)
    b=profile_depths(field,planes,**kwargs)
    assert a["schema"]=="RealSaS.VF11CertifiabilityProfile.v1"
    assert a["status"]=="RESEARCH_MEASUREMENT_ONLY__NO_PRODUCT_AUTHORITY"
    assert a["numerically_rigorous"] is False
    assert [r["states"] for r in a["depths"]]==[r["states"] for r in b["depths"]]
    assert [r["reason_histogram"] for r in a["depths"]]==[r["reason_histogram"] for r in b["depths"]]
