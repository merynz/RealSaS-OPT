from __future__ import annotations

import torch

from models.iris.v4.adaptive_hull_constraint_v4_4_1 import (
    augmented_hull_objective_v441,
    violation_conditioned_sign_control_v441,
)
from models.iris.v4.source_hull_lattice_v4_3 import (
    SourceHullLatticePolicyV43,
    balanced_exterior_lower_bound_loss_v43,
)
from models.iris.v4.train_fit_v4_2 import compute_v42_tp64_geometry_objective


def compute_v441_tp64_geometry_terms(
    *,
    hull_lattice_points_normalized: torch.Tensor,
    hull_lattice_margin_normalized: torch.Tensor,
    hull_lattice_distance_px: torch.Tensor,
    hull_lattice_certified: torch.Tensor,
    hull_lattice_policy: SourceHullLatticePolicyV43=SourceHullLatticePolicyV43(),
    **v42_kwargs,
) -> dict[str, object]:
    hull_lattice_policy.validate()
    base=compute_v42_tp64_geometry_objective(**v42_kwargs)
    model=v42_kwargs["model"]; scene_planes=v42_kwargs["scene_planes"]
    q=hull_lattice_points_normalized
    if q.ndim==2:
        q=q.unsqueeze(0)
    if q.ndim!=3 or q.shape[0]!=1 or q.shape[-1]!=3:
        raise ValueError("hull lattice points must be [1,N,3] or [N,3]")
    hull_sdf=model.query(scene_planes,q)["sdf"].reshape(-1)
    hull=balanced_exterior_lower_bound_loss_v43(
        hull_sdf,
        hull_lattice_margin_normalized,
        hull_lattice_distance_px,
        hull_lattice_certified,
        policy=hull_lattice_policy,
    )
    sampled_control=violation_conditioned_sign_control_v441(
        hull_sdf,
        hull_lattice_margin_normalized,
        hull_lattice_distance_px,
        hull_lattice_certified,
        hull_policy=hull_lattice_policy,
        require_all_bands=False,
    )
    out=dict(base)
    out.update({
        "v42_total_before_v441_hull":base["total"],
        "hull_lattice_total":hull["total"],
        "hull_lattice_certified_count":hull["certified_count"],
        "hull_lattice_nonpositive_exterior_fraction":hull["nonpositive_exterior_fraction"],
        "hull_lattice_metric_deficit_violating_fraction":hull["metric_deficit_violating_fraction"],
        "hull_lattice_maximum_metric_deficit":hull["maximum_metric_deficit"],
        "hull_lattice_bands":hull["bands"],
        "sampled_violation_conditioned_controller_control":sampled_control["control"],
        "sampled_violation_conditioned_controller_bands":sampled_control["bands"],
    })
    return out


def assemble_v441_tp64_geometry_objective(
    terms: dict[str,object],
    *,
    lambda_value: float,
    rho: float,
) -> dict[str,object]:
    total=augmented_hull_objective_v441(
        base_total=terms["v42_total_before_v441_hull"],
        hull_total=terms["hull_lattice_total"],
        lambda_value=lambda_value,
        rho=rho,
    )
    out=dict(terms)
    out.update({
        "total":total,
        "adaptive_hull_lambda":float(lambda_value),
        "frozen_hull_rho":float(rho),
        "hull_linear_contribution":float(lambda_value)*terms["hull_lattice_total"],
        "hull_quadratic_contribution":0.5*float(rho)*terms["hull_lattice_total"].square(),
    })
    return out
