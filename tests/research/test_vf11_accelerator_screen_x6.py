from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"experiments"/"vf11_certified_adaptive"))

import range_engine_c0_v3 as scalar  # noqa:E402
import batched_c0_x5 as batch  # noqa:E402
import accelerator_screen_x6 as x6  # noqa:E402


class TinyField(nn.Module):
    def __init__(self,channels=2,hidden=8):
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


def _boxes(size=8,count=6):
    knots=scalar.interpolation_knots(size)
    los=[];his=[]
    for i in range(count):
        ix=1+(i*2)%(size-2);iy=1+(i*3)%(size-2);iz=1+(i*5)%(size-2)
        lo=np.array([
            knots[ix]+0.11*(knots[ix+1]-knots[ix]),
            knots[iy]+0.13*(knots[iy+1]-knots[iy]),
            knots[iz]+0.17*(knots[iz+1]-knots[iz]),
        ])
        hi=np.array([
            knots[ix]+0.79*(knots[ix+1]-knots[ix]),
            knots[iy]+0.77*(knots[iy+1]-knots[iy]),
            knots[iz]+0.75*(knots[iz+1]-knots[iz]),
        ])
        los.append(lo);his.append(hi)
    return np.asarray(los),np.asarray(his)


def test_device_helpers_preserve_cpu_bounds():
    torch.manual_seed(2201)
    field=TinyField().double().eval()
    planes=torch.randn(1,3,2,8,8,dtype=torch.float64)
    p=scalar.prepare_field(field)
    pp=scalar.prepare_planes(planes)
    p2=batch.prepared_field_to_device(p,"cpu")
    pp2=batch.planes_to_device(pp,"cpu")
    los,his=_boxes(count=4)
    a=batch.bound_batch_single_regime_prepared(p,pp,los,his)
    b=batch.bound_batch_single_regime_prepared(p2,pp2,los,his)
    np.testing.assert_allclose(a[0],b[0],rtol=0.0,atol=0.0)
    np.testing.assert_allclose(a[1],b[1],rtol=0.0,atol=0.0)


def test_confirm_candidate_signs_rejects_false_decisive_proposals():
    s_lo=np.array([0.5,-2.0,0.2,-0.3])
    s_hi=np.array([1.0,-0.1,0.7,0.4])
    c_lo=np.array([-0.2,-1.5,0.1,-0.2])
    c_hi=np.array([0.8,-0.2,0.9,0.5])
    ss,cs,conf=x6.confirm_candidate_signs(s_lo,s_hi,c_lo,c_hi)
    assert ss.tolist()==[1,-1,1,0]
    assert cs.tolist()==[0,-1,1,0]
    assert conf.tolist()==[0,-1,1,0]


def test_cpu_emulated_accelerator_screen_is_subset_equal_to_cpu_x5():
    torch.manual_seed(2202)
    field=TinyField(channels=2,hidden=10).double().eval()
    planes=torch.randn(1,3,2,8,8,dtype=torch.float64)
    p=scalar.prepare_field(field)
    pp=scalar.prepare_planes(planes)
    los,his=_boxes(count=6)

    ref=batch.certify_batch_single_regime_ladders_prepared(
        p,pp,los,his,max_micro_depth=2,node_batch_size=8
    )
    got=x6.screen_verify_single_regime_roots(
        p,pp,p,pp,los,his,
        max_micro_depth=2,
        accelerator_node_batch_size=16,
        cpu_verify_batch_size=8,
    )

    assert got.cpu_rejected_count==0
    assert got.root_states==tuple(c.by_micro_depth[2].state for c in ref)
    for i,c in enumerate(ref):
        s=c.by_micro_depth[2]
        assert got.root_positive_terminal_counts[i]==s.positive_leaf_count
        assert got.root_negative_terminal_counts[i]==s.negative_leaf_count
        assert got.root_unresolved_leaf_counts[i]==s.unresolved_leaf_count


def test_constant_positive_field_confirms_without_refinement():
    torch.manual_seed(2203)
    field=TinyField().double().eval()
    with torch.no_grad():
        field.sdf_head.weight.zero_()
        field.sdf_head.bias.fill_(3.0)
    planes=torch.randn(1,3,2,8,8,dtype=torch.float64)
    p=scalar.prepare_field(field)
    pp=scalar.prepare_planes(planes)
    los,his=_boxes(count=5)

    got=x6.screen_verify_single_regime_roots(
        p,pp,p,pp,los,his,max_micro_depth=2
    )
    assert got.root_states==tuple(["PROVEN_EMPTY_POSITIVE"]*5)
    assert got.terminal_count==5
    assert got.final_unresolved_leaf_count==0
    assert got.accelerator_box_eval_count==5
    assert got.cpu_verify_box_eval_count==5
