from __future__ import annotations

import sys
from pathlib import Path
from collections import Counter

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"experiments"/"vf11_certified_adaptive"))

import proof_aware_planner_x3 as x3  # noqa:E402
import block_mc_x2 as x2  # noqa:E402

pytest.importorskip("skimage")


def _one_root_profiles():
    rows=[]
    for i in range(8):
        if i==0:
            state="EMPTY_POSITIVE"; c0="PROVEN_EMPTY_POSITIVE"; axis=None
        elif i==1:
            state="UNRESOLVED"; c0="PROVEN_ZERO_EXISTS"; axis=None
        elif i==2:
            state="GRAPH"; c0="PROVEN_ZERO_EXISTS"; axis="X"
        else:
            state="UNRESOLVED"; c0="UNKNOWN"; axis=None
        rows.append({
            "root_anchor_id":7,
            "path":str(i),
            "relative_depth":1,
            "state":state,
            "c0_state":c0,
            "selected_axis":axis,
        })
    return {"records":rows},{"new_records":[]}


def test_block_path_d11_bits():
    assert x3.block_path_d11((0,0,0))=="000"
    assert x3.block_path_d11((7,7,7))=="777"
    assert x3.block_path_d11((4,0,0))=="400"
    assert x3.block_path_d11((0,4,0))=="200"
    assert x3.block_path_d11((0,0,4))=="100"


def test_proof_path_bounds_d13():
    lo,hi=x3.proof_path_cell_bounds_d13("")
    assert lo.tolist()==[0,0,0] and hi.tolist()==[32,32,32]
    lo,hi=x3.proof_path_cell_bounds_d13("4")
    assert lo.tolist()==[16,0,0] and hi.tolist()==[32,16,16]
    lo,hi=x3.proof_path_cell_bounds_d13("47")
    assert lo.tolist()==[24,8,8] and hi.tolist()==[32,16,16]


def test_planner_maps_every_d13_block_once_and_preserves_leaf_class():
    v3,v4=_one_root_profiles()
    plan=x3.plan_d13_blocks(v3,v4)
    assert len(plan)==512
    counts=Counter(b.planner_class for b in plan)
    assert counts["CERTIFIED_EMPTY"]==64
    assert counts["ACTIVE_ZERO_EXISTS"]==64
    assert counts["ACTIVE_GRAPH"]==64
    assert counts["ACTIVE_UNKNOWN"]==320
    assert all(len(b.octree_path)==3 for b in plan)


def test_graph_leaf_line_audit_captures_every_transverse_line():
    # Root-wide X graph f=x-0.031 has one crossing on every YZ line.
    axis=np.linspace(-1.0,1.0,33,dtype=np.float64)
    z,y,x=np.meshgrid(axis,axis,axis,indexing="ij")
    f=x-0.031
    audit=x3.audit_graph_leaf_lines(f,"","X")
    assert audit["line_miss_count"]==0
    assert audit["transverse_line_count"]==33*33
    assert audit["line_capture_fraction"]==1.0


def test_zero_exists_leaf_can_be_missed_by_d13_corners_and_is_not_empty():
    # Tiny negative bubble inside one cube: all sampled corners positive.
    f=np.ones((33,33,33),dtype=np.float64)
    assert x3.leaf_mixed_cube_count(f,"")==0


def test_query_corner_count_is_less_when_certified_empty_blocks_pruned():
    v3,v4=_one_root_profiles()
    plan=x3.plan_d13_blocks(v3,v4)
    active=x3.unique_query_corner_count(plan)
    dense=33**3
    assert 0 < active < dense


def test_selected_bracketed_blocks_equal_dense_d13_lewiner_mc():
    axis=np.linspace(-0.31,0.31,33,dtype=np.float64)
    z,y,x=np.meshgrid(axis,axis,axis,indexing="ij")
    f=(x*x+y*y+z*z-0.223**2+1.73e-6).astype(np.float32)
    assert not np.any(f==0.0)
    spacing=(float(axis[1]-axis[0]),)*3
    origin=(float(axis[0]),)*3

    selected=[]
    for bx in range(8):
        for by in range(8):
            for bz in range(8):
                if x3.block_corner_brackets(f,(bx,by,bz)):
                    selected.append((bx,by,bz))

    sparse=x3.extract_selected_d13_blocks(
        f,selected,
        root_origin_xyz=origin,
        root_spacing_xyz=spacing,
    )
    dense=x2.extract_dense_lewiner_mc(
        f,origin_xyz=origin,spacing_xyz=spacing
    )
    assert x2.canonical_block_mc_signature(sparse)==x2.canonical_block_mc_signature(dense)


def test_final_partition_rejects_incomplete_root_volume():
    v3={"records":[{
        "root_anchor_id":3,"path":"0","relative_depth":1,
        "state":"EMPTY_POSITIVE","c0_state":"PROVEN_EMPTY_POSITIVE"
    }]}
    with pytest.raises(RuntimeError,match="does not cover root"):
        x3.final_partition_records(v3,{"new_records":[]})
