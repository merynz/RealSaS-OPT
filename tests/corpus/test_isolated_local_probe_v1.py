import numpy as np
import pytest
from models.iris.truth.local_oriented_probe_v1 import certify


def test_equal_queries_with_opposite_labels_are_both_masked():
    v=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.], [0.,0.,.02],[1.,0.,.02],[0.,1.,.02]])
    f=np.array([[0,1,2],[3,4,5]])
    r=certify(v,f,np.array([0,1]),np.array([[.5,.25,.25]]*2),np.array([.01,-.01]))
    assert not r['valid_mask'].any() and not len(r['signed_local_distance'])
    assert set(r['reason_codes'])=={'COMPETING_OR_AMBIGUOUS_SURFACE'}
    assert r['global_sign_authorized'] is False and r['occupancy_authorized'] is False


def test_isolated_oriented_offsets_are_valid_without_solid_authority():
    v=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
    r=certify(v,np.array([[0,1,2]]),np.array([0,0]),np.array([[.5,.25,.25]]*2),np.array([.01,-.01]))
    np.testing.assert_array_equal(r['probe_index'],[0,1])
    np.testing.assert_allclose(r['signed_local_distance'],[.01,-.01])
    assert r['scope']=='LOCAL_ORIENTED_DISTANCE_ONLY'


def test_unqualified_competing_surface_cannot_be_ignored():
    v=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.], [0.,0.,.02],[1.,0.,.02],[0.,1.,.02]])
    r=certify(v,np.array([[0,1,2],[3,4,5]]),np.array([0]),np.array([[.5,.25,.25]]),np.array([.01]),face_valid=np.array([True,False]))
    assert not r['valid_mask'].any()


@pytest.mark.parametrize('offset',[0.,1e-12])
def test_unresolved_offset_is_not_a_fabricated_zero_target(offset):
    r=certify(np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]]),np.array([[0,1,2]]),np.array([0]),np.array([[.5,.25,.25]]),np.array([offset]))
    assert not len(r['signed_local_distance'])


def test_anchor_order_does_not_choose_one_side_of_an_ambiguous_tie():
    v=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.], [0.,0.,.02],[1.,0.,.02],[0.,1.,.02]])
    f=np.array([[3,4,5],[0,1,2]])
    r=certify(v,f,np.array([1,0]),np.array([[.5,.25,.25]]*2),np.array([.01,-.01]))
    assert not r['valid_mask'].any()
