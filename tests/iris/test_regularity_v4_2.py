import torch

from models.iris.v4.audit_scheduler_v4_2 import (
    AuditComputeBudgetV42,
    exact_stage13_compute_admissible_v42,
    r512_promotion_audit_required_v42,
)
from models.iris.v4.regularity_v4_2 import (
    GlobalEikonalPolicyV42,
    RayLipschitzPolicyV42,
    deterministic_global_points_v42,
    ray_directional_lipschitz_v42,
    sign_transition_diagnostic_v42,
)


def test_global_points_are_deterministic_and_interior():
    p=GlobalEikonalPolicyV42(maximum_points_per_step=32)
    a=deterministic_global_points_v42(fit_seed=7,training_step=11,policy=p,device='cpu')
    b=deterministic_global_points_v42(fit_seed=7,training_step=11,policy=p,device='cpu')
    assert torch.equal(a,b)
    assert a.shape==(1,32,3)
    assert torch.all(a>-1.0) and torch.all(a<1.0)


def test_ray_lipschitz_zero_for_unit_slope_and_positive_for_oscillation():
    points=torch.zeros(1,1,5,3)
    points[0,0,:,0]=torch.linspace(0.0,0.4,5)
    good=torch.tensor([[[0.0,0.1,0.2,0.3,0.4]]])
    row=ray_directional_lipschitz_v42(good,points,policy=RayLipschitzPolicyV42())
    assert float(row['total']) < 1e-12
    bad=torch.tensor([[[0.0,0.3,0.0,0.3,0.0]]])
    row=ray_directional_lipschitz_v42(bad,points,policy=RayLipschitzPolicyV42())
    assert float(row['total']) > 0.0
    assert float(row['violating_fraction']) > 0.0


def test_sign_transitions_are_diagnostic_only_shape():
    sdf=torch.tensor([[[1.0,-1.0,1.0,-1.0,1.0]]])
    row=sign_transition_diagnostic_v42(sdf)
    assert int(row['transition_max'])==4
    assert float(row['rays_gt2_fraction'])==1.0


def test_compute_guard_is_apparatus_only_and_promotion_is_fail_fast():
    budget=AuditComputeBudgetV42()
    ok,reason=exact_stage13_compute_admissible_v42(resolution=128,face_count=1000,budget=budget)
    assert ok and reason=='COMPUTE_ADMISSIBLE'
    ok,reason=exact_stage13_compute_admissible_v42(
        resolution=128,face_count=budget.r128_exact_stage13_face_cap+1,budget=budget
    )
    assert not ok and reason.startswith('COMPUTE_GUARD_FACE_CAP')
    assert r512_promotion_audit_required_v42(r256_stage13_pass=False) is False
    assert r512_promotion_audit_required_v42(r256_stage13_pass=True) is True
