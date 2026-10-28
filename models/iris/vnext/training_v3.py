from __future__ import annotations

from typing import Mapping

import torch
import torch.nn.functional as F


MECHANICAL_KEYS_V3 = (
    "stretch_major",
    "stretch_minor",
    "tangent_direction_2theta",
    "bend_demand",
    "rigidity_probability",
    "contact_persistence_probability",
)

RELATION_STATIC_KEYS_V3 = (
    "same_drawable_affinity",
    "cut_probability",
    "attachment_affinity",
)

RELATION_VIEW_KEYS_V3 = (
    "front_probability",
    "occlusion_probability",
    "projected_alpha_boundary_probability",
    "projected_appearance_boundary_probability",
)


def _masked_mean(raw: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(device=raw.device, dtype=torch.bool)
    if raw.shape[: mask.ndim] != mask.shape:
        raise ValueError("IRIS_V3_MASK_PREFIX_SHAPE_MISMATCH")
    if not bool(mask.any()):
        return raw.new_zeros(())
    while mask.ndim < raw.ndim:
        mask = mask.unsqueeze(-1)
    expanded = mask.expand_as(raw)
    return raw[expanded].mean()


def mechanical_demand_loss_v3(
    predicted: Mapping[str, torch.Tensor],
    truth: Mapping[str, torch.Tensor],
    *,
    mechanical_valid_mask: torch.Tensor | None = None,
    contact_valid_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Objective-complete mechanical loss including authored contact persistence."""

    missing_pred = [name for name in MECHANICAL_KEYS_V3 if name not in predicted]
    missing_truth = [name for name in MECHANICAL_KEYS_V3 if name not in truth]
    if missing_pred:
        raise ValueError("IRIS_MECHANICAL_V3_PREDICTION_MISSING:" + ",".join(missing_pred))
    if missing_truth:
        raise ValueError("IRIS_MECHANICAL_V3_TRUTH_MISSING:" + ",".join(missing_truth))

    reference = predicted["stretch_major"]
    base_mask = (
        torch.ones_like(reference, dtype=torch.bool)
        if mechanical_valid_mask is None
        else mechanical_valid_mask.to(device=reference.device, dtype=torch.bool)
    )
    if base_mask.shape != reference.shape:
        raise ValueError("IRIS_MECHANICAL_V3_VALID_MASK_SHAPE_INVALID")

    stretch = (
        _masked_mean(
            F.smooth_l1_loss(
                predicted["stretch_major"],
                truth["stretch_major"].to(predicted["stretch_major"]),
                reduction="none",
            ),
            base_mask,
        )
        + _masked_mean(
            F.smooth_l1_loss(
                predicted["stretch_minor"],
                truth["stretch_minor"].to(predicted["stretch_minor"]),
                reduction="none",
            ),
            base_mask,
        )
    )
    p_tangent = F.normalize(predicted["tangent_direction_2theta"], dim=-1, eps=1.0e-6)
    t_tangent = F.normalize(
        truth["tangent_direction_2theta"].to(p_tangent), dim=-1, eps=1.0e-6
    )
    tangent = _masked_mean(1.0 - (p_tangent * t_tangent).sum(dim=-1), base_mask)
    bend = _masked_mean(
        F.smooth_l1_loss(
            predicted["bend_demand"],
            truth["bend_demand"].to(predicted["bend_demand"]),
            reduction="none",
        ),
        base_mask,
    )
    rigidity = _masked_mean(
        F.binary_cross_entropy(
            predicted["rigidity_probability"].clamp(1.0e-6, 1.0 - 1.0e-6),
            truth["rigidity_probability"].to(predicted["rigidity_probability"]),
            reduction="none",
        ),
        base_mask,
    )

    if contact_valid_mask is None:
        contact_mask = torch.zeros_like(reference, dtype=torch.bool)
    else:
        contact_mask = contact_valid_mask.to(device=reference.device, dtype=torch.bool)
        if contact_mask.shape != reference.shape:
            raise ValueError("IRIS_MECHANICAL_V3_CONTACT_VALID_MASK_SHAPE_INVALID")
    contact_raw = F.binary_cross_entropy(
        predicted["contact_persistence_probability"].clamp(1.0e-6, 1.0 - 1.0e-6),
        truth["contact_persistence_probability"].to(
            predicted["contact_persistence_probability"]
        ),
        reduction="none",
    )
    contact = _masked_mean(contact_raw, contact_mask)

    terms = [stretch, tangent, bend, rigidity]
    if bool(contact_mask.any()):
        terms.append(contact)
    total = torch.stack(terms).mean()
    return {
        "total": total,
        "stretch": stretch,
        "tangent": tangent,
        "bend": bend,
        "rigidity": rigidity,
        "contact_persistence": contact,
        "contact_supervision_present": total.new_tensor(float(bool(contact_mask.any()))),
    }


def appearance_loss_v3(
    predicted: Mapping[str, torch.Tensor],
    truth: Mapping[str, torch.Tensor],
    *,
    valid_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    required = (
        "rgba_candidate",
        "alpha_boundary_probability",
        "appearance_boundary_probability",
    )
    if any(name not in predicted for name in required):
        raise ValueError("IRIS_APPEARANCE_V3_PREDICTION_INCOMPLETE")
    truth_required = ("rgba", "alpha_boundary", "appearance_boundary")
    if any(name not in truth for name in truth_required):
        raise ValueError("IRIS_APPEARANCE_V3_TRUTH_INCOMPLETE")

    base = predicted["alpha_boundary_probability"]
    mask = (
        torch.ones_like(base, dtype=torch.bool)
        if valid_mask is None
        else valid_mask.to(device=base.device, dtype=torch.bool)
    )
    if mask.shape != base.shape:
        raise ValueError("IRIS_APPEARANCE_V3_VALID_MASK_SHAPE_INVALID")

    rgba = _masked_mean(
        F.smooth_l1_loss(
            predicted["rgba_candidate"],
            truth["rgba"].to(predicted["rgba_candidate"]),
            reduction="none",
        ),
        mask,
    )
    alpha = _masked_mean(
        F.binary_cross_entropy(
            predicted["alpha_boundary_probability"].clamp(1.0e-6, 1.0 - 1.0e-6),
            truth["alpha_boundary"].to(predicted["alpha_boundary_probability"]),
            reduction="none",
        ),
        mask,
    )
    appearance = _masked_mean(
        F.binary_cross_entropy(
            predicted["appearance_boundary_probability"].clamp(1.0e-6, 1.0 - 1.0e-6),
            truth["appearance_boundary"].to(
                predicted["appearance_boundary_probability"]
            ),
            reduction="none",
        ),
        mask,
    )
    total = rgba + 0.25 * alpha + 0.25 * appearance
    return {
        "total": total,
        "rgba": rgba,
        "alpha_boundary": alpha,
        "appearance_boundary": appearance,
    }


def graphics_relation_loss_v3(
    predicted: Mapping[str, torch.Tensor],
    truth: Mapping[str, torch.Tensor],
    validity: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Validity-aware pair loss with no pairwise visibility or neural confidence."""

    keys = RELATION_STATIC_KEYS_V3 + RELATION_VIEW_KEYS_V3
    missing_pred = [name for name in keys if name not in predicted]
    missing_truth = [name for name in keys if name not in truth]
    missing_valid = [name for name in keys if name not in validity]
    if missing_pred:
        raise ValueError("IRIS_RELATION_V3_PREDICTION_MISSING:" + ",".join(missing_pred))
    if missing_truth:
        raise ValueError("IRIS_RELATION_V3_TRUTH_MISSING:" + ",".join(missing_truth))
    if missing_valid:
        raise ValueError("IRIS_RELATION_V3_VALIDITY_MISSING:" + ",".join(missing_valid))
    forbidden = {
        "confidence",
        "visibility_probability_a",
        "visibility_probability_b",
    }
    if forbidden.intersection(predicted):
        raise ValueError("IRIS_RELATION_V3_FORBIDDEN_PAIR_OUTPUT_PRESENT")

    family: dict[str, torch.Tensor] = {}
    trained: list[torch.Tensor] = []
    for name in keys:
        p = predicted[name]
        t = truth[name].to(device=p.device, dtype=p.dtype)
        mask = validity[name].to(device=p.device, dtype=torch.bool)
        if p.shape != t.shape or p.shape != mask.shape:
            raise ValueError("IRIS_RELATION_V3_SHAPE_MISMATCH:" + name)
        if not bool(mask.any()):
            family[name] = p.new_zeros(())
            continue
        raw = F.binary_cross_entropy(
            p.clamp(1.0e-6, 1.0 - 1.0e-6), t, reduction="none"
        )
        value = raw[mask].mean()
        family[name] = value
        trained.append(value)

    if not trained:
        raise ValueError("IRIS_RELATION_V3_ALL_TARGETS_ABSTAINED")
    total = torch.stack(trained).mean()
    return {"total": total, **family}


def ray_visibility_loss_v3(
    probability: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Global ray-stack visibility objective, separate from pair relations."""

    if probability.shape != target.shape or probability.shape != valid_mask.shape:
        raise ValueError("IRIS_RAY_VIS_V3_LOSS_SHAPE_MISMATCH")
    mask = valid_mask.to(device=probability.device, dtype=torch.bool)
    if not bool(mask.any()):
        raise ValueError("IRIS_RAY_VIS_V3_LOSS_EMPTY")
    return F.binary_cross_entropy(
        probability[mask].clamp(1.0e-6, 1.0 - 1.0e-6),
        target.to(probability)[mask],
    )
