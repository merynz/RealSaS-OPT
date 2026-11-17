from __future__ import annotations

"""Numerical all-face dynamic preflight, not a trained product acceptance."""

import pytest
import torch

from models.rss_vnext.seeded_dynamic_full_face_court_v1 import (
    run_seeded_dynamic_full_face_court_v1,
    SeededDynamicLimitsV1,
)


def _fixture():
    xyz=torch.tensor([[
        [0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[1.,1.,0.],
    ]])
    faces=torch.tensor([[0,1,2],[1,3,2]],dtype=torch.long)
    skin=torch.tensor([[
        [1.,0.],[1.,0.],[1.,0.],[1.,0.],
    ]])
    mats=torch.eye(4).repeat(1,3,2,1,1)
    mats[0,1,:,0,3]=.4
    mats[0,2,:,1,3]=.2
    mask=torch.tensor([[1,1,1]],dtype=torch.bool)
    return xyz,faces,skin,mats,mask


def _run(xyz,faces,skin,mats,mask):
    return run_seeded_dynamic_full_face_court_v1(
        rest_xyz=xyz,faces=faces,skin_weights=skin,
        joint_matrices=mats,qualified_pose_mask=mask,
    )


def test_same_indexed_mesh_safe_rigid_skin_and_pose_court_is_deterministic():
    x,f,s,m,mask=_fixture()
    a=_run(x,f,s,m,mask)
    b=_run(x,f,s,m,mask)
    assert a.report_sha256==b.report_sha256
    assert a.face_count==2 and a.observed_pose_count==3
    assert a.passed_numerical_preflight
    assert a.failures==()
    assert not a.compiler_g3_g3b_or_stage35_pass_claimed
    assert not a.animation_ready_qualified
    assert a.risky_face_count==0
    assert a.maximum_condition_number<1.0001
    a.require_numerical_preflight()


def test_one_collapsed_triangle_fails_even_if_other_faces_remain_good():
    x,f,s,m,mask=_fixture()
    s[0,3]=torch.tensor([0.,1.])
    m[0,1,1,0,3]=-1.
    outcome=_run(x,f,s,m,mask)
    assert not outcome.passed_numerical_preflight
    assert "G3_COLLAPSED_FACE_AREA" in outcome.failures
    assert "G3B_SKIN_TOPOLOGY_RISK_FACE" in outcome.failures
    assert outcome.risky_face_count==1
    with pytest.raises(ValueError,match="NUMERICAL_REJECT"):
        outcome.require_numerical_preflight()


def test_only_qualified_pose_is_stressed_and_completely_unqualified_mask_fails():
    x,f,s,m,mask=_fixture()
    s[0,3]=torch.tensor([0.,1.])
    m[0,1,1,0,3]=-1.
    mask[:]=torch.tensor([[1,0,1]],dtype=torch.bool)
    a=_run(x,f,s,m,mask)
    assert a.passed_numerical_preflight
    assert a.observed_pose_count==2
    mask[:]=False
    with pytest.raises(ValueError,match="EXACT_M_POSE_DOMAIN"):
        _run(x,f,s,m,mask)


def test_skin_index_domain_and_duplicate_triangles_fail_closed():
    x,f,s,m,mask=_fixture()
    with pytest.raises(ValueError,match="EXACT_M_POSE_DOMAIN"):
        _run(x,f,s[:,:-1],m,mask)
    with pytest.raises(ValueError,match="DUPLICATE_TRIANGLES"):
        _run(x,torch.cat((f,f[[0]])),s,m,mask)
    with pytest.raises(ValueError,match="DUPLICATE_FACE_CORNER"):
        _run(x,torch.tensor([[0,0,1]]),s,m,mask)


def test_no_average_loss_may_promote_hard_admission():
    x,f,s,m,mask=_fixture()
    assert not _run(x,f,s,m,mask).animation_ready_qualified
    with pytest.raises(ValueError,match="LIMIT_POLICY"):
        SeededDynamicLimitsV1(min_area_ratio=-.1).validate()
    with pytest.raises(ValueError,match="LIMIT_POLICY"):
        SeededDynamicLimitsV1(g3b_skin_l1_risk_threshold=2.3).validate()
