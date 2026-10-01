from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"experiments"/"vf11_certified_adaptive"))

import range_engine_c0_v3 as scalar  # noqa:E402
import batched_c0_x5 as batch  # noqa:E402


class TinyField(nn.Module):
    def __init__(self,channels=3,hidden=12):
        super().__init__()
        width=3*channels
        self.body=nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width,hidden),
            nn.SiLU(),
            nn.Linear(hidden,hidden),
            nn.SiLU(),
        )
        self.sdf_head=nn.Linear(hidden,1)

    @staticmethod
    def sample(planes,points):
        b,_,c,h,w=planes.shape
        x,y,z=points.unbind(-1)
        grids=torch.stack([
            torch.stack([x,y],-1),
            torch.stack([x,z],-1),
            torch.stack([y,z],-1),
        ],dim=1)
        sampled=F.grid_sample(
            planes.reshape(b*3,c,h,w),
            grids.reshape(b*3,points.shape[1],1,2),
            mode="bilinear",padding_mode="border",align_corners=False,
        )
        return sampled.squeeze(-1).reshape(
            b,3,c,points.shape[1]
        ).permute(0,3,1,2).reshape(b,points.shape[1],3*c)

    def forward(self,planes,points):
        return self.sdf_head(self.body(self.sample(planes,points))).squeeze(-1)


def _boxes(size=8,count=10):
    knots=scalar.interpolation_knots(size)
    rows=[]
    for i in range(count):
        ix=1+(i*2)%(size-2)
        iy=1+(i*3)%(size-2)
        iz=1+(i*5)%(size-2)
        fracs=(0.07+0.01*i,0.84-0.01*i)
        lo=np.array([
            knots[ix]+fracs[0]*(knots[ix+1]-knots[ix]),
            knots[iy]+(fracs[0]+0.03)*(knots[iy+1]-knots[iy]),
            knots[iz]+(fracs[0]+0.05)*(knots[iz+1]-knots[iz]),
        ],dtype=np.float64)
        hi=np.array([
            knots[ix]+fracs[1]*(knots[ix+1]-knots[ix]),
            knots[iy]+(fracs[1]-0.02)*(knots[iy+1]-knots[iy]),
            knots[iz]+(fracs[1]-0.04)*(knots[iz+1]-knots[iz]),
        ],dtype=np.float64)
        rows.append((lo,hi))
    return np.stack([x[0] for x in rows]),np.stack([x[1] for x in rows])


def test_batched_single_regime_bounds_match_scalar():
    torch.manual_seed(1201)
    field=TinyField().double().eval()
    planes=torch.randn(1,3,3,8,8,dtype=torch.float64)
    p=scalar.prepare_field(field)
    pp=scalar.prepare_planes(planes)
    los,his=_boxes(count=10)

    blo,bhi=batch.bound_batch_single_regime_prepared(p,pp,los,his)

    for i in range(len(los)):
        slo,shi=scalar.bound_single_regime_prepared(p,pp,los[i],his[i])
        assert abs(float(blo[i])-slo)<=1e-10
        assert abs(float(bhi[i])-shi)<=1e-10
        # Batched result may not accidentally be materially narrower.
        assert float(blo[i])<=slo+1e-10
        assert float(bhi[i])>=shi-1e-10


def test_batched_ladder_matches_scalar_states_bounds_counts_and_evals():
    torch.manual_seed(1202)
    field=TinyField(channels=2,hidden=10).double().eval()
    planes=torch.randn(1,3,2,8,8,dtype=torch.float64)
    p=scalar.prepare_field(field)
    pp=scalar.prepare_planes(planes)
    los,his=_boxes(count=8)

    got=batch.certify_batch_single_regime_ladders_prepared(
        p,pp,los,his,max_micro_depth=2
    )

    for i in range(len(los)):
        ref=scalar.certify_cell_c0_ladder_prepared(
            p,pp,los[i],his[i],max_micro_depth=2
        )
        assert got[i].regime_box_count==1
        assert got[i].actual_evaluated_box_count==ref.actual_evaluated_box_count
        for m in (0,1,2):
            a=got[i].by_micro_depth[m]
            b=ref.by_micro_depth[m]
            assert a.state==b.state
            assert a.sign==b.sign
            assert a.unresolved_leaf_count==b.unresolved_leaf_count
            assert a.positive_leaf_count==b.positive_leaf_count
            assert a.negative_leaf_count==b.negative_leaf_count
            assert abs(a.lower-b.lower)<=1e-10
            assert abs(a.upper-b.upper)<=1e-10


def test_batch_order_does_not_change_per_box_result():
    torch.manual_seed(1203)
    field=TinyField(channels=2,hidden=9).double().eval()
    planes=torch.randn(1,3,2,8,8,dtype=torch.float64)
    p=scalar.prepare_field(field)
    pp=scalar.prepare_planes(planes)
    los,his=_boxes(count=7)
    perm=np.array([6,2,0,5,1,4,3],dtype=np.int64)

    a=batch.bound_batch_single_regime_prepared(p,pp,los,his)
    b=batch.bound_batch_single_regime_prepared(p,pp,los[perm],his[perm])
    inv=np.argsort(perm)
    np.testing.assert_allclose(a[0],b[0][inv],rtol=0.0,atol=1e-12)
    np.testing.assert_allclose(a[1],b[1][inv],rtol=0.0,atol=1e-12)


def test_all_decisive_batch_copies_ladder_state_forward_without_extra_evals():
    torch.manual_seed(1204)
    field=TinyField(channels=2,hidden=8).double().eval()
    with torch.no_grad():
        field.sdf_head.weight.zero_()
        field.sdf_head.bias.fill_(2.0)
    planes=torch.randn(1,3,2,8,8,dtype=torch.float64)
    p=scalar.prepare_field(field)
    pp=scalar.prepare_planes(planes)
    los,his=_boxes(count=5)

    got=batch.certify_batch_single_regime_ladders_prepared(
        p,pp,los,his,max_micro_depth=2
    )
    for cert in got:
        assert cert.actual_evaluated_box_count==1
        assert [cert.by_micro_depth[m].state for m in (0,1,2)]==[
            "PROVEN_EMPTY_POSITIVE",
            "PROVEN_EMPTY_POSITIVE",
            "PROVEN_EMPTY_POSITIVE",
        ]
