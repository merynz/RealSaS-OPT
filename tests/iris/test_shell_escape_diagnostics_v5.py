from __future__ import annotations

import numpy as np

from models.iris.v5.indexed_sparse_tetra_decoder_v5 import SparseRegularTetraPolicyV5
from models.iris.v5.shell_escape_diagnostics_v5 import (
    face_zero_crossing_from_scalar_v5,
    missing_neighbor_faces_v5,
    parent_face_fine_ijk_v5,
    refine_rule_for_corner_values_v5,
)


def test_missing_neighbor_faces_excludes_domain_boundary():
    # Two adjacent refined cells in a 4^3 domain.
    c=np.asarray([[1,1,1],[2,1,1]],dtype=np.int32)
    m=missing_neighbor_faces_v5(c,base_cells=4)
    assert m.shape==(2,6)
    # Shared X face is present.
    assert not m[0,1]
    assert not m[1,0]
    # Other in-domain neighbors are missing.
    assert int(m[0].sum())==5
    assert int(m[1].sum())==5


def test_domain_boundary_is_not_missing_neighbor():
    c=np.asarray([[0,0,0]],dtype=np.int32)
    m=missing_neighbor_faces_v5(c,base_cells=4)
    assert not m[0,0] and not m[0,2] and not m[0,4]
    assert m[0,1] and m[0,3] and m[0,5]


def test_parent_face_has_nine_unique_fine_vertices():
    for fi in range(6):
        q=parent_face_fine_ijk_v5(np.asarray([3,4,5]),fi)
        assert q.shape==(9,3)
        assert len(np.unique(q,axis=0))==9
        if fi==0: assert np.all(q[:,0]==6)
        if fi==1: assert np.all(q[:,0]==8)


def test_face_zero_crossing_requires_actual_sign_mix_or_zero():
    pos=np.ones((3,3),dtype=np.float32)
    neg=-np.ones((3,3),dtype=np.float32)
    mix=pos.copy(); mix[1,1]=-1.0
    zero=pos.copy(); zero[0,0]=0.0
    assert not face_zero_crossing_from_scalar_v5(pos)
    assert not face_zero_crossing_from_scalar_v5(neg)
    assert face_zero_crossing_from_scalar_v5(mix)
    assert face_zero_crossing_from_scalar_v5(zero)


def test_refine_rule_matches_sign_or_band_contract():
    v=np.asarray([
        [1,1,1,1,1,1,1,1],
        [-1,1,1,1,1,1,1,1],
        [.2,.2,.2,.2,.2,.2,.2,.01],
    ],dtype=np.float32)
    got=refine_rule_for_corner_values_v5(v,refine_band=.02)
    assert got.tolist()==[False,True,True]


def test_policy_geometry_is_still_512_to_1024():
    p=SparseRegularTetraPolicyV5()
    assert p.base_cells==512
    assert p.fine_cells==1024
