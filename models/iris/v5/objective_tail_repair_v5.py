from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
import torch.nn.functional as F

from .direct_fstar_fit_v5 import DirectFStarFitPolicyV5, direct_fstar_loss_v5


OBJECTIVE_ARMS_V5 = (
    "A0_CONTINUE_DIRECT_FSTAR_ONLY",
    "B_DIRECT_FSTAR_PLUS_NEAR_ZERO_SIGN",
    "C_DIRECT_FSTAR_PLUS_SOURCE_SILHOUETTE",
    "D_DIRECT_FSTAR_PLUS_SIGN_AND_SILHOUETTE",
)


@dataclass(frozen=True)
class ObjectiveTailRepairPolicyV5:
    """Frozen causal objective-ablation policy.

    Auxiliary penalties are one-sided wrong-sign penalties in the same normalized
    field units as F*.  They introduce no teacher geometry, no metric-SDF target,
    no Eikonal equality, and no R512 audit labels.
    """

    near_zero_sign_band: float = 0.03
    sign_weight: float = 1.0
    silhouette_weight: float = 1.0
    silhouette_softmin_tau: float = 1.0 / 512.0
    silhouette_depth_samples: int = 128
    silhouette_rays_per_step: int = 256
    source_boundary_band_px: float = 8.0

    def validate(self) -> None:
        for name in (
            "near_zero_sign_band",
            "sign_weight",
            "silhouette_weight",
            "silhouette_softmin_tau",
            "source_boundary_band_px",
        ):
            value=float(getattr(self,name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if int(self.silhouette_depth_samples) < 2:
            raise ValueError("silhouette_depth_samples must be >= 2")
        if int(self.silhouette_rays_per_step) <= 0:
            raise ValueError("silhouette_rays_per_step must be positive")


def near_zero_wrong_sign_loss_v5(
    predicted_field: torch.Tensor,
    target_fstar: torch.Tensor,
    *,
    band: float,
) -> dict[str, torch.Tensor | int]:
    """One-sided symmetric sign repair around the canonical F* zero set.

    Correct-sign samples have exactly zero auxiliary loss.  There is no artificial
    positive/negative margin and therefore no competing metric magnitude target.
    Exact target zeros are excluded from this auxiliary term.
    """

    pred=predicted_field.float().reshape(-1)
    target=target_fstar.float().reshape(-1)
    if pred.shape != target.shape or pred.numel()==0:
        raise ValueError("predicted_field and target_fstar must match and be non-empty")
    if not bool(torch.isfinite(pred).all()) or not bool(torch.isfinite(target).all()):
        raise ValueError("sign loss inputs must be finite")
    b=float(band)
    if not math.isfinite(b) or b <= 0.0:
        raise ValueError("band must be finite and positive")

    eligible=(torch.abs(target) <= b) & (target != 0.0)
    eligible_count=int(eligible.sum().detach().cpu())
    if eligible_count == 0:
        zero=pred.sum()*0.0
        return {
            "total":zero,
            "eligible_count":0,
            "wrong_sign_count":0,
            "wrong_sign_fraction":zero.detach(),
            "mean_wrong_side_depth":zero.detach(),
        }

    sign=torch.where(target[eligible] > 0.0, torch.ones_like(target[eligible]), -torch.ones_like(target[eligible]))
    signed_prediction=sign*pred[eligible]
    wrong_depth=F.relu(-signed_prediction)
    wrong=wrong_depth > 0.0
    total=wrong_depth.mean()
    wrong_count=int(wrong.sum().detach().cpu())
    mean_wrong=(
        wrong_depth[wrong].mean().detach()
        if bool(wrong.any())
        else torch.zeros((),device=pred.device,dtype=pred.dtype)
    )
    return {
        "total":total,
        "eligible_count":eligible_count,
        "wrong_sign_count":wrong_count,
        "wrong_sign_fraction":wrong.float().mean().detach(),
        "mean_wrong_side_depth":mean_wrong,
    }


def _camera_basis_contract_v5(
    right: torch.Tensor,
    up: torch.Tensor,
    forward: torch.Tensor,
    *,
    atol: float=2e-4,
) -> None:
    for name,x in (("right",right),("up",up),("forward",forward)):
        norm=torch.linalg.vector_norm(x,dim=-1)
        if not bool(torch.all(torch.abs(norm-1.0) <= float(atol))):
            raise ValueError(f"qualified camera {name} must be unit length; no silent normalization")
    if not bool(torch.all(torch.abs((right*up).sum(-1)) <= float(atol))):
        raise ValueError("qualified camera right/up are not orthogonal")
    if not bool(torch.all(torch.abs((right*forward).sum(-1)) <= float(atol))):
        raise ValueError("qualified camera right/forward are not orthogonal")
    if not bool(torch.all(torch.abs((up*forward).sum(-1)) <= float(atol))):
        raise ValueError("qualified camera up/forward are not orthogonal")


def orthographic_source_rays_normalized_v5(
    pixel_xy: torch.Tensor,
    view_index: torch.Tensor,
    *,
    resolution: int,
    center_xyz: torch.Tensor,
    normalization_half_extent: float | torch.Tensor,
    camera_origins: torch.Tensor,
    camera_right: torch.Tensor,
    camera_screen_up: torch.Tensor,
    camera_forward: torch.Tensor,
    camera_half_extent: torch.Tensor,
) -> tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
    """Exact pixel-center orthographic rays intersected with normalized cube [-1,1]^3.

    Returns base points, unit directions, entry t, exit t.  No camera basis vector is
    silently normalized; the qualified camera contract is validated first.
    """

    px=torch.as_tensor(pixel_xy,dtype=torch.float32)
    vi=torch.as_tensor(view_index,device=px.device,dtype=torch.long).reshape(-1)
    if px.ndim != 2 or px.shape[1] != 2 or len(px)==0 or len(vi)!=len(px):
        raise ValueError("pixel_xy must be non-empty [N,2] and view_index [N]")
    if int(resolution) <= 0:
        raise ValueError("resolution must be positive")
    if bool(torch.any(vi < 0)) or bool(torch.any(vi >= 8)):
        raise ValueError("view_index must lie in [0,7]")

    device=px.device
    center=torch.as_tensor(center_xyz,device=device,dtype=torch.float32).reshape(3)
    half=torch.as_tensor(normalization_half_extent,device=device,dtype=torch.float32).reshape(())
    origins=torch.as_tensor(camera_origins,device=device,dtype=torch.float32)
    right=torch.as_tensor(camera_right,device=device,dtype=torch.float32)
    up=torch.as_tensor(camera_screen_up,device=device,dtype=torch.float32)
    forward=torch.as_tensor(camera_forward,device=device,dtype=torch.float32)
    cam_half=torch.as_tensor(camera_half_extent,device=device,dtype=torch.float32).reshape(-1)
    if origins.shape!=(8,3) or right.shape!=(8,3) or up.shape!=(8,3) or forward.shape!=(8,3) or cam_half.shape!=(8,):
        raise ValueError("camera arrays must be 8-view qualified arrays")
    for x in (px,center,half,origins,right,up,forward,cam_half):
        if not bool(torch.isfinite(x).all()):
            raise ValueError("ray inputs must be finite")
    if float(half.detach().cpu()) <= 0.0 or bool(torch.any(cam_half <= 0.0)):
        raise ValueError("half extents must be positive")
    _camera_basis_contract_v5(right,up,forward)

    r=right[vi]; u=up[vi]; f=forward[vi]
    o=(origins[vi]-center[None,:])/half
    h=cam_half[vi]/half
    gx=((px[:,0]+0.5)/float(resolution))*2.0-1.0
    gy=((px[:,1]+0.5)/float(resolution))*2.0-1.0
    base=o + r*(gx*h)[:,None] - u*(gy*h)[:,None]

    eps=1e-12
    t_lo=torch.full((len(px),),-float("inf"),device=device,dtype=torch.float32)
    t_hi=torch.full((len(px),), float("inf"),device=device,dtype=torch.float32)
    valid=torch.ones((len(px),),device=device,dtype=torch.bool)
    for axis in range(3):
        b=base[:,axis]; d=f[:,axis]
        parallel=torch.abs(d)<=eps
        valid &= ~(parallel & ((b < -1.0) | (b > 1.0)))
        safe_d=torch.where(parallel,torch.ones_like(d),d)
        t1=(-1.0-b)/safe_d
        t2=( 1.0-b)/safe_d
        lo=torch.minimum(t1,t2)
        hi=torch.maximum(t1,t2)
        lo=torch.where(parallel,torch.full_like(lo,-float("inf")),lo)
        hi=torch.where(parallel,torch.full_like(hi, float("inf")),hi)
        t_lo=torch.maximum(t_lo,lo)
        t_hi=torch.minimum(t_hi,hi)
    valid &= t_hi >= t_lo
    if not bool(valid.all()):
        bad=int((~valid).sum().detach().cpu())
        raise ValueError(f"{bad} source pixel rays do not intersect normalized domain")
    return base,f,t_lo,t_hi


def stratified_ray_samples_v5(
    base: torch.Tensor,
    direction: torch.Tensor,
    t_enter: torch.Tensor,
    t_exit: torch.Tensor,
    phase: torch.Tensor,
    *,
    depth_samples: int,
) -> torch.Tensor:
    """One deterministic stratified phase per ray; samples remain strictly inside slabs."""

    b=torch.as_tensor(base,dtype=torch.float32)
    d=torch.as_tensor(direction,device=b.device,dtype=torch.float32)
    te=torch.as_tensor(t_enter,device=b.device,dtype=torch.float32).reshape(-1)
    tx=torch.as_tensor(t_exit,device=b.device,dtype=torch.float32).reshape(-1)
    ph=torch.as_tensor(phase,device=b.device,dtype=torch.float32).reshape(-1)
    if b.ndim!=2 or b.shape[1]!=3 or d.shape!=b.shape or len(te)!=len(b) or len(tx)!=len(b) or len(ph)!=len(b):
        raise ValueError("ray sample arrays have incompatible shapes")
    n=int(depth_samples)
    if n < 2:
        raise ValueError("depth_samples must be >=2")
    if bool(torch.any(ph < 0.0)) or bool(torch.any(ph >= 1.0)):
        raise ValueError("phase must lie in [0,1)")
    k=torch.arange(n,device=b.device,dtype=torch.float32)[None,:]
    frac=(k+ph[:,None])/float(n)
    t=te[:,None] + frac*(tx-te)[:,None]
    q=b[:,None,:] + t[:,:,None]*d[:,None,:]
    if bool(torch.any(q < -1.00001)) or bool(torch.any(q > 1.00001)):
        raise ValueError("stratified ray samples escaped normalized cube")
    return q


def soft_ray_minimum_v5(field_samples: torch.Tensor, *, tau: float) -> torch.Tensor:
    """Differentiable minimum as a softmax-weighted field average.

    Unlike -tau*logsumexp(), this form is constant-preserving and has no N-dependent
    additive bias.  It converges to hard min as tau -> 0.
    """

    f=torch.as_tensor(field_samples).float()
    if f.ndim!=2 or f.shape[1] < 2 or f.numel()==0:
        raise ValueError("field_samples must be [R,S] with S>=2")
    if not bool(torch.isfinite(f).all()):
        raise ValueError("field_samples must be finite")
    t=float(tau)
    if not math.isfinite(t) or t <= 0.0:
        raise ValueError("tau must be finite and positive")
    w=torch.softmax(-f/t,dim=1)
    return torch.sum(w*f,dim=1)


def source_silhouette_wrong_sign_loss_v5(
    field_samples: torch.Tensor,
    foreground: torch.Tensor,
    *,
    tau: float,
) -> dict[str,torch.Tensor | int]:
    """Source-authoritative ray sign constraint.

    Foreground rays require a negative ray minimum; background rays require a positive
    ray minimum.  Correct-side rays have exactly zero auxiliary penalty.
    """

    f=torch.as_tensor(field_samples).float()
    y=torch.as_tensor(foreground,device=f.device,dtype=torch.bool).reshape(-1)
    if f.ndim!=2 or len(y)!=f.shape[0] or f.numel()==0:
        raise ValueError("field_samples [R,S] and foreground [R] must align")
    smin=soft_ray_minimum_v5(f,tau=tau)
    violation=torch.where(y,F.relu(smin),F.relu(-smin))
    wrong=violation > 0.0
    total=violation.mean()
    return {
        "total":total,
        "ray_count":int(len(y)),
        "foreground_count":int(y.sum().detach().cpu()),
        "background_count":int((~y).sum().detach().cpu()),
        "wrong_ray_count":int(wrong.sum().detach().cpu()),
        "wrong_ray_fraction":wrong.float().mean().detach(),
        "mean_wrong_side_depth":(
            violation[wrong].mean().detach()
            if bool(wrong.any())
            else torch.zeros((),device=f.device,dtype=f.dtype)
        ),
        "soft_ray_minimum":smin,
    }


def objective_tail_repair_loss_v5(
    predicted_field: torch.Tensor,
    target_fstar: torch.Tensor,
    *,
    arm: str,
    silhouette_field_samples: torch.Tensor | None=None,
    silhouette_foreground: torch.Tensor | None=None,
    direct_policy: DirectFStarFitPolicyV5=DirectFStarFitPolicyV5(),
    policy: ObjectiveTailRepairPolicyV5=ObjectiveTailRepairPolicyV5(),
) -> dict[str,torch.Tensor | int | str]:
    policy.validate()
    if arm not in OBJECTIVE_ARMS_V5:
        raise ValueError(f"unknown objective arm: {arm}")
    direct=direct_fstar_loss_v5(predicted_field,target_fstar,policy=direct_policy)
    zero=direct["total"]*0.0

    sign_report={
        "total":zero,
        "eligible_count":0,
        "wrong_sign_count":0,
        "wrong_sign_fraction":zero.detach(),
        "mean_wrong_side_depth":zero.detach(),
    }
    if arm in (
        "B_DIRECT_FSTAR_PLUS_NEAR_ZERO_SIGN",
        "D_DIRECT_FSTAR_PLUS_SIGN_AND_SILHOUETTE",
    ):
        sign_report=near_zero_wrong_sign_loss_v5(
            predicted_field,target_fstar,band=float(policy.near_zero_sign_band)
        )

    silhouette_report={
        "total":zero,
        "ray_count":0,
        "foreground_count":0,
        "background_count":0,
        "wrong_ray_count":0,
        "wrong_ray_fraction":zero.detach(),
        "mean_wrong_side_depth":zero.detach(),
    }
    if arm in (
        "C_DIRECT_FSTAR_PLUS_SOURCE_SILHOUETTE",
        "D_DIRECT_FSTAR_PLUS_SIGN_AND_SILHOUETTE",
    ):
        if silhouette_field_samples is None or silhouette_foreground is None:
            raise ValueError("silhouette arm requires silhouette field samples and labels")
        sr=source_silhouette_wrong_sign_loss_v5(
            silhouette_field_samples,silhouette_foreground,
            tau=float(policy.silhouette_softmin_tau),
        )
        silhouette_report={k:v for k,v in sr.items() if k!="soft_ray_minimum"}

    total=direct["total"]
    if arm in (
        "B_DIRECT_FSTAR_PLUS_NEAR_ZERO_SIGN",
        "D_DIRECT_FSTAR_PLUS_SIGN_AND_SILHOUETTE",
    ):
        total=total+float(policy.sign_weight)*sign_report["total"]
    if arm in (
        "C_DIRECT_FSTAR_PLUS_SOURCE_SILHOUETTE",
        "D_DIRECT_FSTAR_PLUS_SIGN_AND_SILHOUETTE",
    ):
        total=total+float(policy.silhouette_weight)*silhouette_report["total"]

    return {
        "arm":arm,
        "total":total,
        "direct_huber":direct["direct_huber"],
        "direct_mae":direct["mae"],
        "direct_rmse":direct["rmse"],
        "direct_maximum_abs":direct["maximum_abs"],
        "sign_total":sign_report["total"],
        "sign_eligible_count":sign_report["eligible_count"],
        "sign_wrong_count":sign_report["wrong_sign_count"],
        "sign_wrong_fraction":sign_report["wrong_sign_fraction"],
        "silhouette_total":silhouette_report["total"],
        "silhouette_ray_count":silhouette_report["ray_count"],
        "silhouette_wrong_count":silhouette_report["wrong_ray_count"],
        "silhouette_wrong_fraction":silhouette_report["wrong_ray_fraction"],
    }
