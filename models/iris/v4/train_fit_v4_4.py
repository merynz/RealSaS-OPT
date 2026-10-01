from __future__ import annotations

import torch

from models.iris.v4.adaptive_hull_constraint_v4_4 import (
    augmented_hull_objective_v44,
    signed_equal_band_control_v44,
)
from models.iris.v4.source_hull_lattice_v4_3 import (
    SourceHullLatticePolicyV43,
    balanced_exterior_lower_bound_loss_v43,
)
from models.iris.v4.train_fit_v4_2 import compute_v42_tp64_geometry_objective


def compute_v44_tp64_geometry_terms(
    *,
    hull_lattice_points_normalized: torch.Tensor,
    hull_lattice_margin_normalized: torch.Tensor,
    hull_lattice_distance_px: torch.Tensor,
    hull_lattice_certified: torch.Tensor,
    hull_lattice_policy: SourceHullLatticePolicyV43=SourceHullLatticePolicyV43(),
    **v42_kwargs,
) -> dict[str, object]:
    """Frozen V4.2 base plus the unchanged V4.3 sampled hull term.

    This function does not apply lambda/rho. V4.4 keeps those controller values
    external so rho can be calibrated once before optimizer step 1 and lambda can
    evolve as the sole adaptive controller state without contaminating the base terms.
    """
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
    sampled_control=signed_equal_band_control_v44(
        hull_sdf,
        hull_lattice_margin_normalized,
        hull_lattice_distance_px,
        hull_lattice_certified,
        hull_policy=hull_lattice_policy,
        require_all_bands=False,
    )
    out=dict(base)
    out.update({
        "v42_total_before_v44_hull":base["total"],
        "hull_lattice_total":hull["total"],
        "hull_lattice_certified_count":hull["certified_count"],
        "hull_lattice_nonpositive_exterior_fraction":hull["nonpositive_exterior_fraction"],
        "hull_lattice_metric_deficit_violating_fraction":hull["metric_deficit_violating_fraction"],
        "hull_lattice_maximum_metric_deficit":hull["maximum_metric_deficit"],
        "hull_lattice_bands":hull["bands"],
        "sampled_signed_controller_control":sampled_control["control"],
        "sampled_signed_controller_bands":sampled_control["bands"],
    })
    return out


def assemble_v44_tp64_geometry_objective(
    terms: dict[str,object],
    *,
    lambda_value: float,
    rho: float,
) -> dict[str,object]:
    total=augmented_hull_objective_v44(
        base_total=terms["v42_total_before_v44_hull"],
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
