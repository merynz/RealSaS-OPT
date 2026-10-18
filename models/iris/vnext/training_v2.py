from __future__ import annotations

"""Post-V8 specialist losses with explicit target validity and conditioning.

This module is intentionally separate from `training_v1.py`: V1 remains part of
historical V8 provenance. V2 refuses implicit fallback rules and expects
projected boundary outputs to come from a view-conditioned prediction path.
"""

from typing import Mapping

import torch
import torch.nn.functional as F

from .graphics_truth_v2 import ALL_RELATION_KEYS_V2, assert_required_validity_keys_v2


RELATION_KEYS_V2 = ALL_RELATION_KEYS_V2


def graphics_relation_loss_v2(
    predicted: Mapping[str, torch.Tensor],
    truth: Mapping[str, torch.Tensor],
    validity: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """Average BCE only over explicitly valid targets.

    The caller may abstain independently per family and per sample. A family
    with no valid examples contributes no loss rather than becoming an invented
    all-negative target. At least one family must contain valid supervision.

    Naming is part of the contract: projected boundary targets use the
    `projected_*` keys so a caller cannot silently feed historical camera-free
    static boundary outputs under the same name.
    """

    assert_required_validity_keys_v2(truth, validity, required=RELATION_KEYS_V2)
    missing_prediction = [name for name in RELATION_KEYS_V2 if name not in predicted]
    if missing_prediction:
        raise ValueError(
            "IRIS_VNEXT_RELATION_V2_PREDICTION_MISSING:" + ",".join(missing_prediction)
        )

    forbidden_historical_boundary_keys = {
        "alpha_boundary_probability",
        "appearance_boundary_probability",
    }
    if forbidden_historical_boundary_keys.intersection(predicted):
        raise ValueError("IRIS_VNEXT_RELATION_V2_LEGACY_STATIC_BOUNDARY_KEY_FORBIDDEN")

    terms: list[torch.Tensor] = []
    for name in RELATION_KEYS_V2:
        p = predicted[name]
        t = truth[name].to(device=p.device, dtype=p.dtype)
        mask = validity[name].to(device=p.device, dtype=torch.bool)
        if p.shape != t.shape or p.shape != mask.shape:
            raise ValueError(f"IRIS_VNEXT_RELATION_V2_SHAPE_MISMATCH:{name}")
        if not bool(mask.any()):
            continue
        raw = F.binary_cross_entropy(
            p.clamp(1.0e-6, 1.0 - 1.0e-6),
            t,
            reduction="none",
        )
        terms.append(raw[mask].mean())

    if not terms:
        raise ValueError("IRIS_VNEXT_RELATION_V2_ALL_TARGETS_ABSTAINED")
    return torch.stack(terms).mean()
