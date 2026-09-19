from compiler.realsas_compiler_core.product_appearance_v1 import _select_donor_view

def test_nearest_observed_eight_direction_donor_wraps_circularly():
    assert _select_donor_view(0,(7,3))==7
    assert _select_donor_view(7,(0,4))==0
    assert _select_donor_view(2,(2,6))==2
    assert _select_donor_view(4,(2,6))==2
    assert _select_donor_view(0,()) is None
