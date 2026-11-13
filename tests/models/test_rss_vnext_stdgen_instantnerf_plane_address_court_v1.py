from __future__ import annotations

import pytest
import torch

from models.iris.vnext.scene_core_v1 import LocalEvidenceFieldV1
from models.rss_vnext.stdgen_instantnerf_native_v1 import (
    sample_stdgen_instantnerf_triplanes_v1,
)
from models.rss_vnext.stdgen_instantnerf_plane_address_court_v1 import (
    reorient_instantnerf_zy_to_iris_yz_v1,
    check_instantnerf_iris_address_equivalence_v1,
)


@pytest.fixture(autouse=True)
def deterministic():
    torch.manual_seed(3611)
    torch.set_num_threads(1)


def test_transpose_proves_exact_interior_addresses_and_gradient_ownership():
    planes=torch.randn(1,3,5,8,8,requires_grad=True)
    points=torch.tensor([
        [[-.6,.12,.25],[.12,-.3,.51],[0.,0.,0.],[.2,.4,-.45]]
    ],requires_grad=True)
    converted=reorient_instantnerf_zy_to_iris_yz_v1(planes)
    native=sample_stdgen_instantnerf_triplanes_v1(planes,points)
    rss=LocalEvidenceFieldV1.sample_triplanes(converted,points)
    assert native.shape==rss.shape==(1,4,15)
    assert torch.allclose(native,rss,atol=1.e-6,rtol=1.e-6)
    court=check_instantnerf_iris_address_equivalence_v1(
        native_planes=planes,xyz=points,rss_target_channels=96,
    )
    assert court.max_absolute_interpolation_difference < 1.e-5
    assert court.source_channels==5 and court.rss_target_channels==96
    assert not court.width_feature_transfer_qualified
    assert not court.full_volume_interpolation_equivalence_qualified
    assert not court.pretrained_checkpoint_loaded
    # Differentiable orientation is a pure permutation, not a new detached
    # neural memory: source-plane and XYZ gradients must agree.
    wl=torch.linspace(.1,.9,native.shape[-1]).reshape(1,1,-1)
    lhs=(native*wl).square().mean()
    rhs=(rss*wl).square().mean()
    a,b=torch.autograd.grad(lhs,(planes,points),retain_graph=True)
    c,d=torch.autograd.grad(rhs,(planes,points))
    assert a.abs().sum()>0 and b.abs().sum()>0
    assert torch.allclose(a,c,atol=1.e-6,rtol=1.e-6)
    assert torch.allclose(b,d,atol=1.e-6,rtol=1.e-6)


def test_untransposed_third_plane_is_not_fake_parity():
    planes=torch.zeros(1,3,3,8,8)
    x=torch.arange(8,dtype=torch.float32)
    planes[0,2,:, :, :]=x[None,None,:]
    pts=torch.tensor([[[0.,-.5,.5]]])
    upstream=sample_stdgen_instantnerf_triplanes_v1(planes,pts)
    wrong=LocalEvidenceFieldV1.sample_triplanes(planes,pts)
    right=LocalEvidenceFieldV1.sample_triplanes(
        reorient_instantnerf_zy_to_iris_yz_v1(planes),pts
    )
    assert not torch.allclose(upstream,wrong)
    assert torch.allclose(upstream,right)


def test_boundary_padding_cannot_be_promoted_to_full_volume_parity():
    planes=torch.ones(1,3,5,8,8)
    points=torch.tensor([[[1.,0.,0.]]])
    with pytest.raises(ValueError,match="PADDING_POLICY_NOT_EQUIVALENT"):
        check_instantnerf_iris_address_equivalence_v1(
            native_planes=planes,xyz=points,rss_target_channels=96,
        )
    converted=reorient_instantnerf_zy_to_iris_yz_v1(planes)
    upstream=sample_stdgen_instantnerf_triplanes_v1(planes,points)
    rss=LocalEvidenceFieldV1.sample_triplanes(converted,points)
    # This difference is not a rounding artefact; zero/border are different
    # functions at the canonical boundary.
    assert not torch.allclose(upstream,rss)


@pytest.mark.parametrize("invalid",["channels","nonsquare","nan"])
def test_invalid_native_triplane_is_rejected(invalid):
    p=torch.ones(1,3,5,8,8)
    if invalid=="channels":p=p[:,:2]
    if invalid=="nonsquare":p=p[:,:,:,:7,:]
    if invalid=="nan":p[0,0,0,0,0]=float("nan")
    with pytest.raises(ValueError,match="REORIENTATION_DOMAIN"):
        reorient_instantnerf_zy_to_iris_yz_v1(p)
