from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"experiments"/"vf11_certified_adaptive"))

import resumable_full_domain_x6c as x6c  # noqa:E402


def test_chunk_ranges_cover_exactly_once():
    assert x6c.chunk_ranges(10,4)==[(0,4),(4,8),(8,10)]


def test_atomic_chunk_roundtrip_and_merge(tmp_path):
    expected=x6c.chunk_ranges(10,4)
    paths=[]
    for start,stop in expected:
        n=stop-start
        states=[
            "UNKNOWN" if i%2==0 else "PROVEN_EMPTY_POSITIVE"
            for i in range(start,stop)
        ]
        p,sha=x6c.write_chunk_atomic(
            tmp_path,
            start=start,
            stop=stop,
            root_states=states,
            positive_terminal_counts=np.arange(n,dtype=np.int32),
            negative_terminal_counts=np.zeros(n,dtype=np.int32),
            unresolved_leaf_counts=np.ones(n,dtype=np.int32),
            metadata={
                "terminal_count":n,
                "final_unresolved_leaf_count":n,
                "screen_decisive_proposal_count":n+1,
                "cpu_confirmed_count":n,
                "cpu_rejected_count":0,
                "accelerator_box_eval_count":2*n,
                "cpu_verify_box_eval_count":n,
                "wall_seconds":0.5,
                "accelerator_seconds":0.1,
                "parallel_cpu_verify_seconds":0.3,
                "refinement_overhead_seconds":0.01,
            },
        )
        assert len(sha)==64
        paths.append(p)

    merged=x6c.merge_complete_chunks(
        paths,
        total_roots=10,
        expected_ranges=expected,
    )
    assert len(merged["state_codes"])==10
    assert merged["chunk_count"]==3
    assert merged["aggregate"]["cpu_rejected_count"]==0
    assert merged["aggregate"]["terminal_count"]==10
    assert merged["aggregate"]["final_unresolved_leaf_count"]==10
    assert merged["aggregate"]["wall_seconds"]==pytest.approx(1.5)


def test_merge_rejects_gap_or_overlap(tmp_path):
    p1,_=x6c.write_chunk_atomic(
        tmp_path,start=0,stop=2,
        root_states=["UNKNOWN","UNKNOWN"],
        positive_terminal_counts=[0,0],
        negative_terminal_counts=[0,0],
        unresolved_leaf_counts=[1,1],
        metadata={"cpu_rejected_count":0},
    )
    p2,_=x6c.write_chunk_atomic(
        tmp_path,start=3,stop=5,
        root_states=["UNKNOWN","UNKNOWN"],
        positive_terminal_counts=[0,0],
        negative_terminal_counts=[0,0],
        unresolved_leaf_counts=[1,1],
        metadata={"cpu_rejected_count":0},
    )
    with pytest.raises(ValueError,match="CHUNK_GAP_OR_OVERLAP"):
        x6c.merge_complete_chunks([p1,p2],total_roots=5)


def test_merge_rejects_cpu_rejection(tmp_path):
    p,_=x6c.write_chunk_atomic(
        tmp_path,start=0,stop=2,
        root_states=["UNKNOWN","UNKNOWN"],
        positive_terminal_counts=[0,0],
        negative_terminal_counts=[0,0],
        unresolved_leaf_counts=[1,1],
        metadata={"cpu_rejected_count":1},
    )
    with pytest.raises(ValueError,match="CPU_REJECTION_PRESENT"):
        x6c.merge_complete_chunks([p],total_roots=2)
