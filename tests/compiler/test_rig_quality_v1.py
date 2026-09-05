import numpy as np

from compiler.realsas_compiler_core.rig_quality_v1 import (
    control_unique_fraction_v1,
    finite_deformation_scores_v1,
    mechanical_coverage_scores_v1,
    skeleton_reference_scores_v1,
)


def test_skeleton_reference_exact_identity_is_perfect():
    p=np.array([[0,0,0],[1,0,0],[2,0,0]],np.float64)
    parent=np.array([-1,0,1],np.int64)
    s=skeleton_reference_scores_v1(p,parent,p,parent,tolerance=1e-4)
    assert s.iou==1.0
    assert s.precision==1.0
    assert s.recall==1.0
    assert s.cd_j2j==0.0
    assert s.cd_j2b==0.0
    assert s.cd_b2b==0.0


def test_mechanical_coverage_is_joint_count_independent_when_span_matches():
    # Reference has a redundant third mode; candidate has only the two basis modes.
    ref=np.array([[1,0,0,0],[0,1,0,0],[1,1,0,0]],np.float64)
    cand=np.array([[1,0,0,0],[0,1,0,0]],np.float64)
    s=mechanical_coverage_scores_v1(ref,cand)
    assert s.reference_to_candidate_residual < 1e-10
    assert s.reference_to_candidate_coverage > 0.999999


def test_finite_deformation_scores_zero_for_equivalent_motion():
    x=np.zeros((4,10,2),np.float64)
    s=finite_deformation_scores_v1(x,x.copy())
    assert s.rms==0.0 and s.p95==0.0 and s.max_error==0.0


def test_unique_fraction_flags_exact_redundancy_without_threshold_policy():
    modes=np.array([[1,0,0],[0,1,0],[1,0,0]],np.float64)
    u=control_unique_fraction_v1(modes)
    assert u.shape==(3,)
    assert u[0] < 1e-10
    assert u[2] < 1e-10
    assert u[1] > 0.99
