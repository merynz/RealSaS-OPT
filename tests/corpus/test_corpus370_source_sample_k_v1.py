"""Variable-K original source sample depth composition redteam and V18n K2 parity."""
import numpy as np
import pytest
from tools.corpus.compose_corpus370_source_sample_k_v1 import (
    compose_isolated_original_source_k as compose,
)
from tools.corpus.court_corpus370_v18n_original_material_sample_replay import (
    predict_material_from_source_isolates,
)


def sample(colors, depth):
    """Convert N samples each containing K (straight_rgba, view_z) layers."""
    rows=len(colors)
    k=len(colors[0])
    c=np.zeros((rows,k,2,2,4),dtype=np.float32)
    d=np.zeros_like(c)
    for i in range(rows):
        for j in range(k):
            value=colors[i][j]
            if value is not None:
                c[i,j,:,:]=value
                d[i,j,:,:,0]=depth[i][j]
                d[i,j,:,:,3]=1
    return c,d


def test_k1_opaque_and_empty_and_unseen_targets():
    c,d=sample([[(.2,.3,.4,1)], [None]],[[.3],[.8]])
    result=compose(c,d)
    np.testing.assert_allclose(result["predicted_source_sample_premultiplied_rgba"][0,0,0],
                               [.2,.3,.4,1],atol=1e-7)
    np.testing.assert_allclose(result["predicted_source_sample_premultiplied_rgba"][1],0)
    assert result["qualified_sample_mask"].all()
    assert result["frontmost_layer_index"][0,0,0]==0
    assert result["frontmost_layer_index"][1,0,0]==-1
    assert result["training_authorized"] is False


def test_k3_depth_order_is_pixel_local_and_rgb_correct():
    colors=[[ (.9,0,0,1), (0,.8,0,1),(0,0,.7,1)],
            [ (.9,0,0,1), (0,.8,0,1),(0,0,.7,1)]]
    c,d=sample(colors,[[.7,.2,.5],[.1,.4,.6]])
    r=compose(c,d)
    assert r["physical_source_group_hit_count"].min()==3
    np.testing.assert_allclose(r["predicted_source_sample_premultiplied_rgba"][0,0,0],
                               [0,.8,0,1],atol=1e-6)
    np.testing.assert_allclose(r["predicted_source_sample_premultiplied_rgba"][1,0,0],
                               [.9,0,0,1],atol=1e-6)
    assert r["frontmost_layer_index"][0,0,0]==1
    assert r["frontmost_layer_index"][1,0,0]==0


def test_k3_partial_alpha_is_source_material_not_geometric_hit():
    c,d=sample([[(1,0,0,.5),(0,1,0,.25),(0,0,1,1)]],[[.1,.3,.5]])
    r=compose(c,d)
    np.testing.assert_allclose(r["predicted_source_sample_premultiplied_rgba"][0,0,0],
                               [.5,.125,.375,1.],atol=1e-7)


def test_v18n_k2_original_regression_same_sample_compatible():
    a=np.zeros((8,2,192,192,4),np.float32)
    d=np.zeros_like(a)
    for i in range(8):
        a[i,0,45:48,30:33]=(.91,.10,.05,.5)
        a[i,1,45:48,30:33]=(.01,.21,.94,1.)
        d[i,0,45:48,30:33]=(.15 if i%2 else .65,0,0,1)
        d[i,1,45:48,30:33]=(.65 if i%2 else .15,0,0,1)
    current=predict_material_from_source_isolates(a,d)
    actual=compose(a,d)
    np.testing.assert_allclose(actual["predicted_source_sample_premultiplied_rgba"],
                               current["predicted_sample_premultiplied_rgba"],atol=1e-7)
    np.testing.assert_array_equal(actual["qualified_sample_mask"],
                                  current["source_valid_sample_mask"])


def test_depth_tie_abstains_without_order_guess():
    c,d=sample([[(1,0,0,1),(0,0,1,1),(0,1,0,1)]],[[.24,.2401,.8]])
    r=compose(c,d)
    assert not r["qualified_sample_mask"].any()
    assert r["unknown_depth_tie_mask"].all()


def test_true_hidden_geometry_with_zero_material_alpha():
    c,d=sample([[(1,0,0,0),(0,1,0,1),(0,0,1,1)]],[[.1,.4,.7]])
    r=compose(c,d)
    np.testing.assert_allclose(r["predicted_source_sample_premultiplied_rgba"][0,0,0],
                               [0,1,0,1],atol=1e-7)
    assert r["physical_source_group_hit_count"].min()==3


@pytest.mark.parametrize("issue",["nan","nonbinary_depth","color_without_geometry","bad_shape"])
def test_source_court_rejects_unauthorized_or_invalid(issue):
    c,d=sample([[(1,0,0,1),(0,1,0,1)]],[[.2,.4]])
    if issue=="nan":c[0,0,0,0,0]=np.nan
    if issue=="nonbinary_depth":d[0,0,0,0,3]=.5
    if issue=="color_without_geometry":d[0,0,:,:,3]=0
    if issue=="bad_shape":d=d[...,0]
    with pytest.raises(ValueError,match="SAMPLE_K_SOURCE"):
        compose(c,d)


def test_sample_prediction_cannot_read_full_target():
    c,d=sample([[(.3,.3,.3,1),(.7,.7,.7,1)]],[[.1,.3]])
    r=compose(c,d)
    assert r["sample_decision_used_full_target"] is False
    assert "source_full_rgba" not in r
    assert "same_drawable" not in r
    assert r["source_artist_drawable_owner"]=="UNKNOWN"
