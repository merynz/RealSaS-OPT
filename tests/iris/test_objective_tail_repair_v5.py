from __future__ import annotations

import math
import torch

from models.iris.v5.objective_tail_repair_v5 import (
    ObjectiveTailRepairPolicyV5,
    near_zero_wrong_sign_loss_v5,
    objective_tail_repair_loss_v5,
    orthographic_source_rays_normalized_v5,
    soft_ray_minimum_v5,
    source_silhouette_wrong_sign_loss_v5,
    stratified_ray_samples_v5,
)


def test_near_zero_sign_loss_is_one_sided_and_symmetric():
    target=torch.tensor([0.002,-0.002,0.02,-0.02,0.1],dtype=torch.float32)
    pred=torch.tensor([-0.001,0.001,0.03,-0.03,-0.1],dtype=torch.float32)
    r=near_zero_wrong_sign_loss_v5(pred,target,band=0.03)
    assert r["eligible_count"]==4
    assert r["wrong_sign_count"]==2
    assert float(r["total"])==torch.tensor([0.001,0.001,0.0,0.0]).mean().item()


def test_exact_target_zero_is_excluded_from_sign_auxiliary():
    r=near_zero_wrong_sign_loss_v5(
        torch.tensor([-1.0,1.0]),torch.tensor([0.0,0.0]),band=0.03
    )
    assert r["eligible_count"]==0
    assert float(r["total"])==0.0


def _identity_cameras():
    origins=torch.tensor([[0.0,0.0,-3.0]]*8)
    right=torch.tensor([[1.0,0.0,0.0]]*8)
    up=torch.tensor([[0.0,1.0,0.0]]*8)
    forward=torch.tensor([[0.0,0.0,1.0]]*8)
    half=torch.tensor([1.0]*8)
    return origins,right,up,forward,half


def test_center_pixel_ray_intersects_normalized_cube_exactly():
    origins,right,up,forward,half=_identity_cameras()
    # For an even 4x4 raster, pixel center x=y=1.5 corresponds to gx=gy=0.
    base,d,te,tx=orthographic_source_rays_normalized_v5(
        torch.tensor([[1.5,1.5]],dtype=torch.float32),
        torch.tensor([0]),
        resolution=4,
        center_xyz=torch.zeros(3),
        normalization_half_extent=1.0,
        camera_origins=origins,camera_right=right,camera_screen_up=up,
        camera_forward=forward,camera_half_extent=half,
    )
    torch.testing.assert_close(base,torch.tensor([[0.0,0.0,-3.0]]),atol=1e-6,rtol=0)
    torch.testing.assert_close(d,torch.tensor([[0.0,0.0,1.0]]),atol=1e-6,rtol=0)
    torch.testing.assert_close(te,torch.tensor([2.0]),atol=1e-6,rtol=0)
    torch.testing.assert_close(tx,torch.tensor([4.0]),atol=1e-6,rtol=0)


def test_ray_samples_stay_inside_domain():
    origins,right,up,forward,half=_identity_cameras()
    base,d,te,tx=orthographic_source_rays_normalized_v5(
        torch.tensor([[1.5,1.5]],dtype=torch.float32),torch.tensor([0]),resolution=4,
        center_xyz=torch.zeros(3),normalization_half_extent=1.0,
        camera_origins=origins,camera_right=right,camera_screen_up=up,
        camera_forward=forward,camera_half_extent=half,
    )
    q=stratified_ray_samples_v5(base,d,te,tx,torch.tensor([0.5]),depth_samples=8)
    assert q.shape==(1,8,3)
    assert bool(torch.all(q>=-1.0)) and bool(torch.all(q<=1.0))


def test_soft_ray_minimum_is_constant_preserving():
    f=torch.full((3,128),0.0123)
    sm=soft_ray_minimum_v5(f,tau=1/512)
    torch.testing.assert_close(sm,torch.full((3,),0.0123),atol=1e-6,rtol=0)


def test_silhouette_wrong_sign_loss_penalizes_only_wrong_side():
    # Ray0 foreground has negative min => correct. Ray1 background has negative min => wrong.
    f=torch.tensor([
        [0.02,-0.01,0.03],
        [0.02,-0.01,0.03],
    ])
    r=source_silhouette_wrong_sign_loss_v5(
        f,torch.tensor([True,False]),tau=1e-4
    )
    assert r["ray_count"]==2
    assert r["wrong_ray_count"]==1
    assert float(r["total"]) > 0.0


def test_objective_arms_change_only_declared_auxiliaries():
    pred=torch.tensor([-0.001,0.001,0.02,-0.02],requires_grad=True)
    target=torch.tensor([0.002,-0.002,0.02,-0.02])
    sf=torch.tensor([[-0.01,0.01],[0.01,0.02]],requires_grad=True)
    sy=torch.tensor([True,False])
    p=ObjectiveTailRepairPolicyV5(silhouette_softmin_tau=1e-4)

    a=objective_tail_repair_loss_v5(pred,target,arm="A0_CONTINUE_DIRECT_FSTAR_ONLY",policy=p)
    b=objective_tail_repair_loss_v5(pred,target,arm="B_DIRECT_FSTAR_PLUS_NEAR_ZERO_SIGN",policy=p)
    c=objective_tail_repair_loss_v5(
        pred,target,arm="C_DIRECT_FSTAR_PLUS_SOURCE_SILHOUETTE",
        silhouette_field_samples=sf,silhouette_foreground=sy,policy=p
    )
    d=objective_tail_repair_loss_v5(
        pred,target,arm="D_DIRECT_FSTAR_PLUS_SIGN_AND_SILHOUETTE",
        silhouette_field_samples=sf,silhouette_foreground=sy,policy=p
    )
    assert float(a["sign_total"])==0.0 and float(a["silhouette_total"])==0.0
    assert float(b["sign_total"])>0.0 and float(b["silhouette_total"])==0.0
    assert float(c["sign_total"])==0.0
    assert float(d["sign_total"])>0.0
    torch.testing.assert_close(
        d["total"],a["total"]+d["sign_total"]+d["silhouette_total"],
        atol=1e-7,rtol=0
    )
