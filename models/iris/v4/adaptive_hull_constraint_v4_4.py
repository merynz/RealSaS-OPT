from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from models.iris.v4.source_hull_lattice_v4_3 import SourceHullLatticePolicyV43


@dataclass(frozen=True)
class AdaptiveHullConstraintPolicyV44:
    """Single-state adaptive controller over the frozen V4.3 hull constraint.

    No Knight-tuned threshold is introduced. The signed controller statistic uses
    the already-frozen V4.3 Huber beta as its only scale and equal-weights the same
    four source-distance bands as the primal hull loss.
    """

    full_scan_interval_steps: int = 100
    initial_lambda: float = 1.0

    def validate(self) -> None:
        if int(self.full_scan_interval_steps) <= 0:
            raise ValueError("full_scan_interval_steps must be positive")
        if not math.isfinite(float(self.initial_lambda)) or float(self.initial_lambda) < 0.0:
            raise ValueError("initial_lambda must be finite and non-negative")


@dataclass(frozen=True)
class AdaptiveHullConstraintStateV44:
    lambda_value: float
    full_anchor_control: float
    replay_anchor_control: float
    anchor_step: int


def _band_masks(
    distance_px: torch.Tensor,
    certified: torch.Tensor,
    *,
    hull_policy: SourceHullLatticePolicyV43,
) -> tuple[torch.Tensor, ...]:
    d=torch.as_tensor(distance_px).float().reshape(-1)
    c=torch.as_tensor(certified,device=d.device).bool().reshape(-1)
    if d.shape!=c.shape:
        raise ValueError("distance/certified shape mismatch")
    e=tuple(float(x) for x in hull_policy.band_edges_px)
    return (
        c&(d<=e[0]),
        c&(d>e[0])&(d<=e[1]),
        c&(d>e[1])&(d<=e[2]),
        c&(d>e[2]),
    )


def signed_equal_band_control_v44(
    sdf: torch.Tensor,
    margin_normalized: torch.Tensor,
    distance_px: torch.Tensor,
    certified: torch.Tensor,
    *,
    hull_policy: SourceHullLatticePolicyV43=SourceHullLatticePolicyV43(),
    require_all_bands: bool=True,
) -> dict[str, object]:
    """Bounded signed hull deficit statistic for the dual controller.

    Each certified point contributes clip((h-f)/beta,-1,1). Negative means
    comfortable satisfaction, positive means deficit. Empty bands are forbidden
    for controller anchors so the four-band meaning cannot silently change.
    """
    hull_policy.validate()
    s=torch.as_tensor(sdf).float().reshape(-1)
    m=torch.as_tensor(margin_normalized,device=s.device).float().reshape(-1)
    d=torch.as_tensor(distance_px,device=s.device).float().reshape(-1)
    c=torch.as_tensor(certified,device=s.device).bool().reshape(-1)
    if not (s.shape==m.shape==d.shape==c.shape) or s.numel()==0:
        raise ValueError("V4.4 controller tensors must match and be non-empty")
    if not torch.isfinite(s).all() or not torch.isfinite(m).all() or not torch.isfinite(d).all():
        raise ValueError("V4.4 controller inputs must be finite")
    beta=float(hull_policy.huber_beta_normalized)
    deficit=m-s
    clipped=torch.clamp(deficit/beta,min=-1.0,max=1.0)
    masks=_band_masks(d,c,hull_policy=hull_policy)
    names=("0_2","2_8","8_32","gt_32")
    rows=[]
    controls=[]
    for name,mask in zip(names,masks):
        count=int(mask.sum().detach().cpu())
        if count<=0:
            if require_all_bands:
                raise ValueError(f"V44_CONTROLLER_EMPTY_BAND:{name}")
            continue
        value=clipped[mask].mean()
        controls.append(value)
        rows.append({
            "band":name,
            "certified_count":count,
            "signed_clipped_deficit_mean":value,
            "metric_deficit_violation_fraction":(deficit[mask]>0).float().mean(),
            "nonpositive_sign_fraction":(s[mask]<=0).float().mean(),
            "maximum_metric_deficit":torch.amax(deficit[mask]),
        })
    if not controls:
        raise ValueError("V44_CONTROLLER_NO_ACTIVE_BANDS")
    control=torch.stack(controls).mean()
    certified_count=c.sum()
    return {
        "control":control,
        "bands":tuple(rows),
        "active_band_names":tuple(row["band"] for row in rows),
        "certified_count":certified_count,
        "metric_deficit_violation_count":((deficit>0)&c).sum(),
        "metric_deficit_violation_fraction":((deficit>0)&c).float().sum()/certified_count.clamp_min(1),
        "nonpositive_sign_count":((s<=0)&c).sum(),
        "nonpositive_sign_fraction":((s<=0)&c).float().sum()/certified_count.clamp_min(1),
        "maximum_metric_deficit":torch.amax(deficit[c]),
    }


def initialize_controller_v44(
    *,
    full_anchor_control: float,
    replay_anchor_control: float,
    anchor_step: int,
    policy: AdaptiveHullConstraintPolicyV44=AdaptiveHullConstraintPolicyV44(),
) -> AdaptiveHullConstraintStateV44:
    policy.validate()
    values=(float(full_anchor_control),float(replay_anchor_control))
    if not all(math.isfinite(x) and -1.0<=x<=1.0 for x in values):
        raise ValueError("V4.4 anchor controls must be finite in [-1,1]")
    if int(anchor_step)<0:
        raise ValueError("anchor_step must be non-negative")
    return AdaptiveHullConstraintStateV44(
        lambda_value=float(policy.initial_lambda),
        full_anchor_control=values[0],
        replay_anchor_control=values[1],
        anchor_step=int(anchor_step),
    )


def reanchor_controller_v44(
    state: AdaptiveHullConstraintStateV44,
    *,
    full_anchor_control: float,
    replay_anchor_control: float,
    anchor_step: int,
) -> AdaptiveHullConstraintStateV44:
    """Replace observation anchors without resetting lambda; avoids scan-time jump."""
    values=(float(full_anchor_control),float(replay_anchor_control))
    if not all(math.isfinite(x) and -1.0<=x<=1.0 for x in values):
        raise ValueError("V4.4 anchor controls must be finite in [-1,1]")
    if int(anchor_step)<int(state.anchor_step):
        raise ValueError("V4.4 anchor step cannot move backwards")
    return AdaptiveHullConstraintStateV44(
        lambda_value=float(state.lambda_value),
        full_anchor_control=values[0],
        replay_anchor_control=values[1],
        anchor_step=int(anchor_step),
    )


def update_lambda_from_replay_v44(
    state: AdaptiveHullConstraintStateV44,
    *,
    replay_control_now: float,
    policy: AdaptiveHullConstraintPolicyV44=AdaptiveHullConstraintPolicyV44(),
) -> tuple[AdaptiveHullConstraintStateV44, dict[str,float]]:
    """Short feedback loop using only replay change relative to the full-scan anchor.

    The enriched replay bank is never treated as a population estimate. Only its
    within-bank change is transported onto the full-scan population anchor.
    """
    policy.validate()
    current=float(replay_control_now)
    if not math.isfinite(current) or not -1.0<=current<=1.0:
        raise ValueError("replay_control_now must be finite in [-1,1]")
    estimated=float(state.full_anchor_control)+(current-float(state.replay_anchor_control))
    estimated=max(-1.0,min(1.0,estimated))
    eta=1.0/float(policy.full_scan_interval_steps)
    new_lambda=max(0.0,float(state.lambda_value)+eta*estimated)
    new_state=AdaptiveHullConstraintStateV44(
        lambda_value=new_lambda,
        full_anchor_control=float(state.full_anchor_control),
        replay_anchor_control=float(state.replay_anchor_control),
        anchor_step=int(state.anchor_step),
    )
    return new_state,{
        "estimated_population_control":estimated,
        "controller_eta_per_optimizer_step":eta,
        "lambda_before":float(state.lambda_value),
        "lambda_after":new_lambda,
    }


def calibrate_frozen_rho_v44(*, base_gradient_norm: float, hull_gradient_norm: float) -> float:
    """Pre-optimizer one-shot rho calibration; rho is frozen after this call."""
    base=float(base_gradient_norm); hull=float(hull_gradient_norm)
    if not math.isfinite(base) or not math.isfinite(hull) or base<=0.0 or hull<=0.0:
        raise ValueError("V4.4 rho calibration requires positive finite gradient norms")
    return base/hull


def augmented_hull_objective_v44(
    *,
    base_total: torch.Tensor,
    hull_total: torch.Tensor,
    lambda_value: float,
    rho: float,
) -> torch.Tensor:
    """Primal objective: L_base + lambda*phi + 0.5*rho*phi^2."""
    lam=float(lambda_value); damping=float(rho)
    if not math.isfinite(lam) or lam<0.0:
        raise ValueError("lambda must be finite and non-negative")
    if not math.isfinite(damping) or damping<=0.0:
        raise ValueError("rho must be finite and positive")
    if base_total.ndim!=0 or hull_total.ndim!=0:
        raise ValueError("base_total/hull_total must be scalar tensors")
    if not torch.isfinite(base_total) or not torch.isfinite(hull_total):
        raise ValueError("base_total/hull_total must be finite")
    return base_total + lam*hull_total + 0.5*damping*hull_total.square()
