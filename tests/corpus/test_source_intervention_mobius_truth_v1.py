"""Original-source factorial intervention algebra synthetic mathematical tests.

All numerical toy arrays below are synthetic and DO NOT assert a real
character source physical edit. Real Cat is independently SHA-courted in CI.
"""
import numpy as np
import pytest

from tools.corpus.source_intervention_mobius_truth_v1 import (
    mobius_transform, inverse_mobius, source_pm, measure_original_source_mobius,
    MASK_TO_ORIGINAL_ARM_INDEX,
)


def test_three_bit_intervention_mobius_is_uniquely_invertible():
    rng = np.random.default_rng(7)
    y = rng.normal(size=(8, 2, 3, 4))
    m = mobius_transform(y, control_count=3)
    restored = inverse_mobius(m, control_count=3)
    assert np.allclose(restored, y, atol=1e-12)


def test_interaction_terms_are_about_operations_not_original_artist_part():
    baseline = 0.17
    effects = (0.07, -0.03, 0.14)
    xy_interaction = 0.006
    y = np.asarray([
        baseline + sum(effects[i] for i in range(3) if mask & (1 << i))
        + xy_interaction * bool(mask & 1 and mask & 2)
        for mask in range(8)
    ],dtype=np.float64)[:,None]
    m = mobius_transform(y, control_count=3)[:,0]
    assert np.isclose(m[0], baseline)
    assert np.allclose(m[[1,2,4]], effects)
    assert np.isclose(m[3], xy_interaction)
    assert np.allclose(m[[5,6,7]], 0, atol=1e-12)
    assert np.allclose(inverse_mobius(m[:,None],control_count=3)[:,0],
                       y[:,0],atol=1e-12)


def test_linear_premultiplied_source_rgba_is_the_declared_value_domain():
    a=np.asarray([[[0.2,0.3,0.4,0.5], [0.9,0.1,0.6,1.0]]])
    got=source_pm(a)
    assert np.allclose(got[0,0],(0.1,0.15,0.2,0.5))
    assert np.allclose(got[0,1],a[0,1])


def test_incomplete_subset_table_fails_closed():
    with pytest.raises(ValueError,match="INCOMPLETE_OR_INVALID_FULL_FACTORIAL"):
        mobius_transform(np.zeros((7,2,4)),control_count=3)
    with pytest.raises(ValueError,match="INVALID_INTERVENTION_COEFFICIENTS"):
        inverse_mobius(np.zeros((7,4)),control_count=3)


def test_source_alpha_nonfinite_is_rejected():
    src=np.zeros((2,4))
    src[1,3]=float("nan")
    with pytest.raises(ValueError,match="ORIGINAL_SOURCE_LINEAR_RGBA_INVALID"):
        source_pm(src)


def test_exact_original_cat_arm_map_differs_from_boolean_mask_order():
    assert MASK_TO_ORIGINAL_ARM_INDEX == (0,1,2,4,3,5,6,7)
    assert MASK_TO_ORIGINAL_ARM_INDEX[3]==4 # physically rendered PAIR_01
    assert MASK_TO_ORIGINAL_ARM_INDEX[4]==3 # physically rendered SINGLE_2


def test_runtime_does_not_claim_real_cat_results_from_toy_shapes():
    with pytest.raises(ValueError,match="SOURCE_PHYSICAL_SAMPLE_SCOPE_DRIFT"):
        measure_original_source_mobius(
            np.zeros((4,8,8,2,2,4),dtype=np.float32))


def test_nonbinary_physical_alpha_does_not_convert_to_boolean_false():
    data=np.asarray([[0.2,0.3,0.4,0.4],[0.1,0.2,0.3,0.0]])
    pm=source_pm(data)
    assert pm[0,3]==0.4
    assert np.all(pm[1,:3]==0)
