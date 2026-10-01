from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"experiments"/"vf11_certified_adaptive"))

import range_engine_c0_v3 as scalar  # noqa:E402
import accelerator_screen_x6 as x6  # noqa:E402
import parallel_cpu_verify_x6b as x6b  # noqa:E402


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
        ix=1+(i*2)%(size-2)
        iy=1+(i*3)%(size-2)
        iz=1+(i*5)%(size-2)
        lo=np.array([
            knots[ix]+0.11*(knots[ix+1]-knots[ix]),
            knots[iy]+0.13*(knots[iy+1]-knots[iy]),
            knots[iz]+0.17*(knots[iz+1]-knots[iz]),
        ],dtype=np.float64)
        hi=np.array([
            knots[ix]+0.79*(knots[ix+1]-knots[ix]),
            knots[iy]+0.77*(knots[iy+1]-knots[iy]),
            knots[iz]+0.75*(knots[iz+1]-knots[iz]),
        ],dtype=np.float64)
        los.append(lo);his.append(hi)
    return np.asarray(los),np.asarray(his)


def test_parallel_cpu_bound_matches_serial_x5():
    torch.manual_seed(3201)
    field=TinyField(channels=2,hidden=10).double().eval()
    planes=torch.randn(1,3,2,8,8,dtype=torch.float64)
    p=scalar.prepare_field(field)
    pp=scalar.prepare_planes(planes)
    los,his=_boxes(count=8)

    with x6b.ParallelCPUVerifier(
        p,pp,workers=2,node_batch_size=8,
        torch_threads_per_worker=1,start_method="spawn"
    ) as verifier:
        plo,phi=verifier.bound(los,his)

    from batched_c0_x5 import _bound_batch_chunked_prepared
    slo,shi=_bound_batch_chunked_prepared(
        p,pp,los,his,node_batch_size=8
    )
    np.testing.assert_allclose(plo,slo,rtol=0.0,atol=1e-10)
    np.testing.assert_allclose(phi,shi,rtol=0.0,atol=1e-10)


def test_parallel_x6b_matches_serial_x6_exactly_on_cpu_emulation():
    torch.manual_seed(3202)
    field=TinyField(channels=2,hidden=10).double().eval()
    planes=torch.randn(1,3,2,8,8,dtype=torch.float64)
    p=scalar.prepare_field(field)
    pp=scalar.prepare_planes(planes)
    los,his=_boxes(count=6)

    serial=x6.screen_verify_single_regime_roots(
        p,pp,p,pp,los,his,
        max_micro_depth=2,
        accelerator_node_batch_size=16,
        cpu_verify_batch_size=8,
        store_terminals=False,
    )
    parallel=x6b.screen_verify_single_regime_roots_parallel_cpu(
        p,pp,p,pp,los,his,
        max_micro_depth=2,
        accelerator_node_batch_size=16,
        cpu_verify_batch_size=8,
        cpu_workers=2,
        cpu_torch_threads_per_worker=1,
        cpu_start_method="spawn",
        store_terminals=False,
    )

    assert parallel.root_states==serial.root_states
    assert parallel.root_positive_terminal_counts==serial.root_positive_terminal_counts
    assert parallel.root_negative_terminal_counts==serial.root_negative_terminal_counts
    assert parallel.root_unresolved_leaf_counts==serial.root_unresolved_leaf_counts
    assert parallel.terminal_count==serial.terminal_count
    assert parallel.final_unresolved_leaf_count==serial.final_unresolved_leaf_count
    assert parallel.screen_decisive_proposal_count==serial.screen_decisive_proposal_count
    assert parallel.cpu_confirmed_count==serial.cpu_confirmed_count
    assert parallel.cpu_rejected_count==serial.cpu_rejected_count
    assert parallel.cpu_verify_box_eval_count==serial.cpu_verify_box_eval_count


def test_parallel_path_preserves_false_decisive_rejection_rule():
    # Direct sign merge rule is still X6's rule; parallelism may not alter it.
    s_lo=np.array([0.5,-2.0,0.2])
    s_hi=np.array([1.0,-0.1,0.7])
    c_lo=np.array([-0.2,-1.5,0.1])
    c_hi=np.array([0.8,-0.2,0.9])
    ss,cs,conf=x6.confirm_candidate_signs(s_lo,s_hi,c_lo,c_hi)
    assert ss.tolist()==[1,-1,1]
    assert cs.tolist()==[0,-1,1]
    assert conf.tolist()==[0,-1,1]


def test_preforked_verifier_can_be_reused_by_x6b_traversal():
    torch.manual_seed(3203)
    field=TinyField(channels=2,hidden=9).double().eval()
    planes=torch.randn(1,3,2,8,8,dtype=torch.float64)
    p=scalar.prepare_field(field)
    pp=scalar.prepare_planes(planes)
    los,his=_boxes(count=5)

    serial=x6.screen_verify_single_regime_roots(
        p,pp,p,pp,los,his,
        max_micro_depth=2,
        accelerator_node_batch_size=16,
        cpu_verify_batch_size=8,
        store_terminals=False,
    )

    verifier=x6b.ParallelCPUVerifier(
        p,pp,workers=2,node_batch_size=8,
        torch_threads_per_worker=1,start_method="fork"
    )
    verifier.__enter__()
    try:
        parallel=x6b.screen_verify_single_regime_roots_parallel_cpu(
            p,pp,p,pp,los,his,
            max_micro_depth=2,
            accelerator_node_batch_size=16,
            cpu_verify_batch_size=8,
            cpu_workers=2,
            cpu_torch_threads_per_worker=1,
            cpu_start_method="fork",
            store_terminals=False,
            cpu_verifier=verifier,
        )
    finally:
        verifier.__exit__(None,None,None)

    assert parallel.root_states==serial.root_states
    assert parallel.root_positive_terminal_counts==serial.root_positive_terminal_counts
    assert parallel.root_negative_terminal_counts==serial.root_negative_terminal_counts
    assert parallel.root_unresolved_leaf_counts==serial.root_unresolved_leaf_counts
    assert parallel.cpu_rejected_count==0
