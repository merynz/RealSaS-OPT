from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from models.iris.v4.source_hull_lattice_v4_3 import SourceHullLatticePolicyV43


@dataclass(frozen=True)
class AdaptiveHullConstraintPolicyV441:
    """Controller repair for V4.4.

    Single adaptive state remains lambda. Full-population control is zero iff the
    exact sign-violation set is empty. Satisfied certified points never contribute
    negative mass, so a sparse violating tail cannot be cancelled by the majority.
    """

    full_scan_interval_steps: int = 100
    initial_lambda: float = 1.0

    def validate(self) -> None:
        if int(self.full_scan_interval_steps) <= 0:
            raise ValueError("full_scan_interval_steps must be positive")
        if not math.isfinite(float(self.initial_lambda)) or float(self.initial_lambda) < 0.0:
            raise ValueError("initial_lambda must be finite and non-negative")


@dataclass(frozen=True)
class AdaptiveHullConstraintStateV441:
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


def violation_conditioned_sign_control_v441(
    sdf: torch.Tensor,
    margin_normalized: torch.Tensor,
    distance_px: torch.Tensor,
    certified: torch.Tensor,
    *,
    hull_policy: SourceHullLatticePolicyV43=SourceHullLatticePolicyV43(),
    require_all_bands: bool=True,
) -> dict[str, object]:
    """Tail-visible control for the exact certified-exterior sign constraint.

    For each frozen source-distance band:
      - sign violation means certified and sdf <= 0;
      - satisfied points contribute exactly zero;
      - violating points contribute clip((margin-sdf)/beta, 0, 1);
      - the band control is the mean severity *conditioned on violation*;
      - a band with no sign violations contributes zero.

    The final control is the equal mean over the present frozen bands. Therefore
    control == 0 iff no present certified band contains a sign violation, while a
    sparse tail remains visible instead of being cancelled by satisfied points.
    """
    hull_policy.validate()
    s=torch.as_tensor(sdf).float().reshape(-1)
    m=torch.as_tensor(margin_normalized,device=s.device).float().reshape(-1)
    d=torch.as_tensor(distance_px,device=s.device).float().reshape(-1)
    c=torch.as_tensor(certified,device=s.device).bool().reshape(-1)
    if not (s.shape==m.shape==d.shape==c.shape) or s.numel()==0:
        raise ValueError("V4.4.1 controller tensors must match and be non-empty")
    if not torch.isfinite(s).all() or not torch.isfinite(m).all() or not torch.isfinite(d).all():
        raise ValueError("V4.4.1 controller inputs must be finite")

    beta=float(hull_policy.huber_beta_normalized)
    deficit=m-s
    severity=torch.clamp(deficit/beta,min=0.0,max=1.0)
    masks=_band_masks(d,c,hull_policy=hull_policy)
    names=("0_2","2_8","8_32","gt_32")
    rows=[]
    controls=[]
    for name,mask in zip(names,masks):
        certified_count=int(mask.sum().detach().cpu())
        if certified_count<=0:
            if require_all_bands:
                raise ValueError(f"V441_CONTROLLER_EMPTY_BAND:{name}")
            continue
        sign_violation=mask&(s<=0)
        sign_count=int(sign_violation.sum().detach().cpu())
        if sign_count:
            band_control=severity[sign_violation].mean()
        else:
            band_control=torch.zeros((),device=s.device,dtype=s.dtype)
        controls.append(band_control)
        rows.append({
            "band":name,
            "certified_count":certified_count,
            "nonpositive_sign_count":sign_count,
            "nonpositive_sign_fraction":sign_violation.float().sum()/max(certified_count,1),
            "violation_conditioned_sign_severity_mean":band_control,
            "metric_deficit_violation_count":((deficit>0)&mask).sum(),
            "maximum_metric_deficit":torch.amax(deficit[mask]),
        })

    if not controls:
        raise ValueError("V441_CONTROLLER_NO_ACTIVE_BANDS")
    control=torch.stack(controls).mean()
    certified_count=c.sum()
    return {
        "control":control,
        "bands":tuple(rows),
        "active_band_names":tuple(row["band"] for row in rows),
        "certified_count":certified_count,
        "nonpositive_sign_count":((s<=0)&c).sum(),
        "nonpositive_sign_fraction":((s<=0)&c).float().sum()/certified_count.clamp_min(1),
        "metric_deficit_violation_count":((deficit>0)&c).sum(),
        "metric_deficit_violation_fraction":((deficit>0)&c).float().sum()/certified_count.clamp_min(1),
        "maximum_metric_deficit":torch.amax(deficit[c]),
    }


def initialize_controller_v441(
    *,
    full_anchor_control: float,
    replay_anchor_control: float,
    anchor_step: int,
    policy: AdaptiveHullConstraintPolicyV441=AdaptiveHullConstraintPolicyV441(),
) -> AdaptiveHullConstraintStateV441:
    policy.validate()
    values=(float(full_anchor_control),float(replay_anchor_control))
    if not all(math.isfinite(x) and 0.0<=x<=1.0 for x in values):
        raise ValueError("V4.4.1 anchor controls must be finite in [0,1]")
    if int(anchor_step)<0:
        raise ValueError("anchor_step must be non-negative")
    return AdaptiveHullConstraintStateV441(
        lambda_value=float(policy.initial_lambda),
        full_anchor_control=values[0],
        replay_anchor_control=values[1],
        anchor_step=int(anchor_step),
    )


def reanchor_controller_v441(
    state: AdaptiveHullConstraintStateV441,
    *,
    full_anchor_control: float,
    replay_anchor_control: float,
    anchor_step: int,
) -> AdaptiveHullConstraintStateV441:
    values=(float(full_anchor_control),float(replay_anchor_control))
    if not all(math.isfinite(x) and 0.0<=x<=1.0 for x in values):
        raise ValueError("V4.4.1 anchor controls must be finite in [0,1]")
    if int(anchor_step)<int(state.anchor_step):
        raise ValueError("V4.4.1 anchor step cannot move backwards")
    return AdaptiveHullConstraintStateV441(
        lambda_value=float(state.lambda_value),
        full_anchor_control=values[0],
        replay_anchor_control=values[1],
        anchor_step=int(anchor_step),
    )


def update_lambda_from_replay_v441(
    state: AdaptiveHullConstraintStateV441,
    *,
    replay_control_now: float,
    policy: AdaptiveHullConstraintPolicyV441=AdaptiveHullConstraintPolicyV441(),
) -> tuple[AdaptiveHullConstraintStateV441, dict[str,float]]:
    """Relative replay feedback plus zero-only leak.

    Replay is enriched and is not a population estimator. Only its change from
    the current full-scan anchor is transported to the full-population control.

    While estimated violations remain positive, lambda can only accumulate.
    Lambda leaks by the same pre-registered 1/scan_interval time scale only when
    the estimated control is exactly zero.
    """
    policy.validate()
    current=float(replay_control_now)
    if not math.isfinite(current) or not 0.0<=current<=1.0:
        raise ValueError("replay_control_now must be finite in [0,1]")
    estimated=float(state.full_anchor_control)+(current-float(state.replay_anchor_control))
    estimated=max(0.0,min(1.0,estimated))
    eta=1.0/float(policy.full_scan_interval_steps)
    before=float(state.lambda_value)
    if estimated>0.0:
        after=before+eta*estimated
        mode="ACCUMULATE"
    else:
        after=max(0.0,(1.0-eta)*before)
        mode="ZERO_ONLY_LEAK"
    new_state=AdaptiveHullConstraintStateV441(
        lambda_value=after,
        full_anchor_control=float(state.full_anchor_control),
        replay_anchor_control=float(state.replay_anchor_control),
        anchor_step=int(state.anchor_step),
    )
    return new_state,{
        "estimated_population_control":estimated,
        "controller_eta_per_optimizer_step":eta,
        "lambda_before":before,
        "lambda_after":after,
        "controller_mode":mode,
    }


def calibrate_frozen_rho_v441(*, base_gradient_norm: float, hull_gradient_norm: float) -> float:
    base=float(base_gradient_norm); hull=float(hull_gradient_norm)
    if not math.isfinite(base) or not math.isfinite(hull) or base<=0.0 or hull<=0.0:
        raise ValueError("V4.4.1 rho calibration requires positive finite gradient norms")
    return base/hull


def augmented_hull_objective_v441(
    *,
    base_total: torch.Tensor,
    hull_total: torch.Tensor,
    lambda_value: float,
    rho: float,
) -> torch.Tensor:
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
