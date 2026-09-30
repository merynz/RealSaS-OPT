import numpy as np
import torch

from models.iris.v4.source_hull_lattice_v4_3 import (
    SourceHullLatticePolicyV43,
    balanced_exterior_lower_bound_loss_v43,
    deterministic_lattice_tile_indices_v43,
    lattice_points_from_linear_indices_v43,
    select_worst_hull_deficit_indices_v43,
)


def test_exact_lattice_index_mapping_matches_dense_grid_order():
    r=8
    idx=torch.tensor([0,1,r-1,r,r*r,r**3-1])
    p=lattice_points_from_linear_indices_v43(idx,resolution=r)
    assert torch.allclose(p[0],torch.tensor([-1.0,-1.0,-1.0]))
    assert torch.allclose(p[2],torch.tensor([1.0,-1.0,-1.0]))
    assert torch.allclose(p[-1],torch.tensor([1.0,1.0,1.0]))


def test_affine_tile_schedule_has_no_overlap_before_wrap():
    policy=SourceHullLatticePolicyV43(resolution=32,tile_points_per_step=1024)
    a=deterministic_lattice_tile_indices_v43(training_step=1,fit_seed=7,policy=policy)
    b=deterministic_lattice_tile_indices_v43(training_step=2,fit_seed=7,policy=policy)
    assert len(torch.unique(a))==len(a)
    assert len(torch.unique(torch.cat([a,b])))==len(a)+len(b)


def test_uncertified_inside_points_are_strictly_silent():
    policy=SourceHullLatticePolicyV43()
    sdf=torch.tensor([-0.7,-0.2,0.1,0.3])
    margin=torch.tensor([0.0,0.0,0.2,0.2])
    distance=torch.tensor([0.0,0.0,10.0,40.0])
    certified=torch.tensor([False,False,True,True])
    out=balanced_exterior_lower_bound_loss_v43(sdf,margin,distance,certified,policy=policy)
    ref=balanced_exterior_lower_bound_loss_v43(torch.tensor([99.0,99.0,0.1,0.3]),margin,distance,certified,policy=policy)
    assert torch.allclose(out["total"],ref["total"])


def test_uncapped_far_target_is_preserved():
    policy=SourceHullLatticePolicyV43()
    out=balanced_exterior_lower_bound_loss_v43(
        torch.tensor([0.04]),torch.tensor([0.40]),torch.tensor([200.0]),torch.tensor([True]),policy=policy
    )
    assert float(out["maximum_metric_deficit"]) > 0.35


def test_replay_selects_largest_metric_deficits_deterministically():
    idx=np.asarray([9,2,7,4],dtype=np.int64)
    sdf=np.asarray([0.0,0.1,-0.2,0.0])
    margin=np.asarray([0.1,0.3,0.1,0.4])
    certified=np.asarray([True,True,True,True])
    got=select_worst_hull_deficit_indices_v43(idx,sdf,margin,certified,bank_size=2)
    assert got.tolist()==[4,7]
