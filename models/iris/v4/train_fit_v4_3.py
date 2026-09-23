from __future__ import annotations

import torch

from models.iris.v4.source_hull_lattice_v4_3 import (
    SourceHullLatticePolicyV43,
    balanced_exterior_lower_bound_loss_v43,
)
from models.iris.v4.train_fit_v4_2 import compute_v42_tp64_geometry_objective


def compute_v43_tp64_geometry_objective(
    *,
    hull_lattice_points_normalized: torch.Tensor,
    hull_lattice_margin_normalized: torch.Tensor,
    hull_lattice_distance_px: torch.Tensor,
    hull_lattice_certified: torch.Tensor,
    hull_lattice_policy: SourceHullLatticePolicyV43 = SourceHullLatticePolicyV43(),
    **v42_kwargs,
) -> dict[str, object]:
    """Frozen V4.2 objective plus source-only exact-MC-lattice exterior lower bound."""
    hull_lattice_policy.validate()
    base=compute_v42_tp64_geometry_objective(**v42_kwargs)
    model=v42_kwargs["model"]; scene_planes=v42_kwargs["scene_planes"]
    q=hull_lattice_points_normalized
    if q.ndim==2: q=q.unsqueeze(0)
    if q.ndim!=3 or q.shape[0]!=1 or q.shape[-1]!=3:
        raise ValueError("hull lattice points must be [1,N,3] or [N,3]")
    hull_sdf=model.query(scene_planes,q)["sdf"].reshape(-1)
    hull=balanced_exterior_lower_bound_loss_v43(
        hull_sdf,hull_lattice_margin_normalized,hull_lattice_distance_px,hull_lattice_certified,
        policy=hull_lattice_policy,
    )
    total=base["total"]+float(hull_lattice_policy.top_level_weight)*hull["total"]
    out=dict(base)
    out.update({
        "total":total,
        "v42_total_before_v43_hull":base["total"],
        "hull_lattice_total":hull["total"],
        "hull_lattice_weight":float(hull_lattice_policy.top_level_weight),
        "hull_lattice_certified_count":hull["certified_count"],
        "hull_lattice_nonpositive_exterior_fraction":hull["nonpositive_exterior_fraction"],
        "hull_lattice_metric_deficit_violating_fraction":hull["metric_deficit_violating_fraction"],
        "hull_lattice_maximum_metric_deficit":hull["maximum_metric_deficit"],
        "hull_lattice_bands":hull["bands"],
    })
    return out
