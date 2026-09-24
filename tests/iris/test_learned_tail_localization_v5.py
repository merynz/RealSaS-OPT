from __future__ import annotations

import numpy as np

from models.iris.v5.indexed_sparse_tetra_decoder_v5 import (
    SparseRegularTetraPolicyV5,
    boundary_parent_cells_v5,
    count_sparse_mt_v5,
    fine_positions_from_gids_v5,
    fine_vertex_gids_v5,
)
from models.iris.v5.learned_tail_localization_v5 import (
    LearnedTailLocalizationPolicyV5,
    deterministic_top_residual_indices_v5,
    heldout_tail_report_v5,
    sign_disagreement_indices_v5,
    spatial_concentration_v5,
    target_abs_band_counts_v5,
    uniform_spatial_bin_ids_v5,
    boundary_crossing_tetra_localization_v5,
)


def test_top_residual_selection_is_deterministic_with_index_tiebreak():
    p=np.asarray([0.0,2.0,-2.0,0.0],dtype=np.float32)
    t=np.zeros(4,dtype=np.float32)
    idx=deterministic_top_residual_indices_v5(p,t,fraction=0.5)
    assert idx.tolist()==[1,2]


def test_sign_convention_matches_v5_direct_metrics():
    p=np.asarray([-1.0,0.0,-0.1,0.1],dtype=np.float32)
    t=np.asarray([-1.0,-0.0,0.1,-0.1],dtype=np.float32)
    assert sign_disagreement_indices_v5(p,t).tolist()==[2,3]


def test_spatial_bin_boundary_clamps_positive_one():
    q=np.asarray([[-1,-1,-1],[1,1,1],[0,0,0]],dtype=np.float32)
    ids=uniform_spatial_bin_ids_v5(q,resolution=2)
    assert ids.tolist()==[0,7,7]


def test_spatial_concentration_reports_localized_cloud():
    q=np.zeros((10,3),dtype=np.float32)
    r=spatial_concentration_v5(q,resolutions=(32,))
    assert r["32"]["occupied_bin_count"]==1
    assert r["32"]["maximum_bin_share"]==1.0


def test_target_abs_bands_partition_all_points():
    x=np.asarray([0.0,0.001,0.003,0.01,0.1],dtype=np.float32)
    rows=target_abs_band_counts_v5(x,edges=(0.002,0.005,0.02))
    assert sum(row["count"] for row in rows)==len(x)


def test_heldout_report_returns_expected_tail_arrays():
    q=np.asarray([
        [-.8,-.8,-.8],[-.4,-.4,-.4],[0,0,0],[.4,.4,.4],[.8,.8,.8]
    ],dtype=np.float32)
    t=np.asarray([-0.1,-0.01,0.01,0.1,0.2],dtype=np.float32)
    p=np.asarray([-0.1,0.02,-0.02,0.08,0.5],dtype=np.float32)
    policy=LearnedTailLocalizationPolicyV5(
        top_residual_fraction=0.2,
        spatial_resolutions=(32,),
        target_abs_band_edges=(0.002,0.01,0.03),
    )
    report,arrays=heldout_tail_report_v5(q,t,p,policy=policy)
    assert report["count"]==5
    assert report["sign_disagreement_count"]==2
    assert arrays["sign_error_indices"].tolist()==[1,2]
    assert arrays["top_residual_indices"].tolist()==[4]
    assert report["spatial_top_residual"]["32"]["point_count"]==1


def test_policy_validation_rejects_nonincreasing_edges():
    try:
        LearnedTailLocalizationPolicyV5(target_abs_band_edges=(0.1,0.05)).validate()
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def test_decoder_policy_still_frozen_512_1024():
    p=SparseRegularTetraPolicyV5()
    assert p.base_cells==512
    assert p.fine_cells==1024


def test_boundary_crossing_localization_matches_independent_mt_count():
    p=SparseRegularTetraPolicyV5(
        base_cells=2,
        fine_cells=4,
        query_chunk=128,
        cell_chunk=2,
        fine_gid_cell_chunk=2,
    )
    refined=np.asarray(
        [(i,j,k) for i in range(2) for j in range(2) for k in range(2)],
        dtype=np.int32,
    )
    boundary=boundary_parent_cells_v5(refined,policy=p)
    gids=fine_vertex_gids_v5(refined,policy=p)
    pos=fine_positions_from_gids_v5(gids,fine_cells=p.fine_cells)
    scalar=pos[:,0].astype(np.float32)
    count=count_sparse_mt_v5(refined,gids,scalar,boundary,policy=p)
    loc=boundary_crossing_tetra_localization_v5(
        refined,gids,scalar,boundary,policy=p
    )
    assert len(loc["parent_index"])==count["boundary_crossing_tets"]
    assert len(loc["tet_center_normalized"])==count["boundary_crossing_tets"]
    assert loc["tet_vertex_scalar"].shape==(count["boundary_crossing_tets"],4)
