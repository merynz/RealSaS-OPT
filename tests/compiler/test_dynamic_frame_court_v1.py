import numpy as np

from compiler.realsas_compiler_core.mesh.dynamic_frame_court_v1 import measure_dynamic_frame_geometry_v1
from compiler.realsas_compiler_core.product_authority_v1 import build_mesh_qualification_policy,CarrierCoverageThresholdIR


def measure(posed):
    policy = build_mesh_qualification_policy(
        g1_max_normal_refinement_ratio=.125, g1_max_tangential_to_normal_ratio=1.,
        g3_min_angle_deg=7.5, g3_max_aspect_longest_over_min_altitude=16.,
        coverage_thresholds=tuple(CarrierCoverageThresholdIR(k,.99,.99,.001,.001) for k in ("MESH","PLANAR")),
    )
    return measure_dynamic_frame_geometry_v1(rest=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]]),
        posed=np.array(posed,dtype=float),faces=np.array([[0,1,2]]),policy=policy)


def test_detects_flattening_without_edge_extension_or_exact_degeneracy():
    result = measure([[0,0,0],[1,0,0],[0,.001,0]])
    assert result["maximum_edge_ratio"] == 1.
    assert result["minimum_area_ratio"] == .001
    assert result["maximum_condition_number"] == 1000.
    assert result["failure_counts"]["DYNAMIC_AREA_RATIO_BELOW_MIN"] == 1
    assert result["failure_counts"]["DYNAMIC_CONDITION_NUMBER_ABOVE_MAX"] == 1
    assert not result["passed"]


def test_rigid_three_dimensional_turn_and_translation_preserve_mechanics():
    result = measure([[2,3,4],[2,3,5],[2,4,4]])
    assert result["passed"]
    assert result["maximum_condition_number"] == 1.


def test_uniform_expansion_trips_area_gate_with_good_condition():
    result = measure([[0,0,0],[5,0,0],[0,5,0]])
    assert result["maximum_condition_number"] == 1.
    assert result["failure_counts"]["DYNAMIC_AREA_RATIO_ABOVE_MAX"] == 1
    assert not result["passed"]
